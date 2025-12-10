### LOGIC:
# This module turns your trained model + SIP features into a real advisor brain.
#
# REQUIREMENTS (NOT OPTIONAL):
# - monthly_amount      (₹ per month)
# - sip_years           (years of SIP)
# - hold_years          (years after SIP with no new investment; can be 0)
# - risk_level          ("low", "medium", "high")
# - category            ("small_cap", "mid_cap", "large_cap", "other")
# - mode                ("best" → top performers, "worst" → bottom performers)
#
# PIPELINE:
# 1. Load engineered features (sip_features.csv)
# 2. Load model predictions (sip_predictions.csv)
# 3. Merge → one row per scheme with:
#       - scheme_code
#       - scheme_name
#       - features (incl. daily_return_std, max_drawdown, etc.)
#       - predicted_cagr_percent
# 4. Infer:
#       - category (small_cap/mid_cap/large_cap/other) from scheme_name text
#       - risk_label ("low"/"medium"/"high") from volatility & drawdown
# 5. Filter schemes by:
#       - category (MANDATORY)
#       - risk_level (MANDATORY, with light relaxation if too strict)
# 6. For each remaining scheme:
#       - simulate SIP + hold:
#             monthly_amount, sip_years, hold_years, predicted CAGR
#       - compute total_invested, final_value, profit, return%
# 7. Normalize scheme_name to base_fund_name to avoid duplicates:
#       - group variants like "Direct/Growth/Quarterly" together
#       - keep best variant per base_fund_name
# 8. Rank:
#       mode = "best"  → highest projected_final_value
#       mode = "worst" → lowest projected_final_value
# 9. Return top_k recommendations as list of dicts with:
#       - scheme_code
#       - scheme_name
#       - base_fund_name
#       - category
#       - risk_label
#       - predicted_cagr_percent
#       - total_invested
#       - projected_final_value
#       - projected_profit
#       - projected_return_percent
#       - projection_curve  (yearly values for graphs)
#
# NOTE:
# - This file is NOT responsible for asking the user follow-up questions.
#   The agent/NLP layer must ensure that all required inputs are provided
#   before calling get_recommendations().
# - This file WILL raise errors if required inputs are missing/invalid.


import pandas as pd
import numpy as np
from pathlib import Path

FEATURE_PATH = Path("data/sip_features_model_ready.csv")
PREDICTIONS_PATH = Path("data/sip_predictions.csv")
# Optional backup metadata (if you created it): data/scheme_metadata.csv
METADATA_PATH = Path("data/scheme_metadata.csv")


# ---------- CATEGORY & NAME HELPERS ----------

def normalize_scheme_name(s: str) -> str:
    """
    Normalize scheme_name to a base fund name by stripping plan/option noise.

    Example:
        "Axis Bluechip Fund - Direct Plan - Growth"
        "Axis Bluechip Fund - Regular Plan - IDCW"

    Both become something like:
        "AXIS BLUECHIP FUND"
    """
    if not isinstance(s, str):
        return ""

    x = s.upper()

    junk_words = [
        "DIRECT PLAN", "REGULAR PLAN",
        "GROWTH OPTION", "GROWTH",
        "IDCW", "DIVIDEND", "REINVESTMENT",
        "BONUS",
        "MONTHLY", "QUARTERLY", "ANNUAL",
        "PAYOUT", "PAY OUT", "PAY-OUT",
        "OPTION", "PLAN",
        "-", "  ",
    ]

    for w in junk_words:
        x = x.replace(w, " ")

    # Collapse multiple spaces and strip
    x = " ".join(x.split())
    return x


def infer_category_from_name(name: str) -> str:
    """
    Rough category inference from scheme_name text.

    Returns one of:
        "small_cap", "mid_cap", "large_cap", "other"
    """
    if not isinstance(name, str):
        return "other"

    n = name.upper()

    if "SMALL CAP" in n:
        return "small_cap"
    if "MIDCAP" in n or "MID CAP" in n:
        return "mid_cap"
    if "LARGE CAP" in n:
        return "large_cap"

    return "other"


# ---------- RISK LABEL FROM VOLATILITY / DRAWDOWN ----------

def assign_risk_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign risk_label ("low", "medium", "high") using volatility and drawdown.

    - Uses:
        daily_return_std  → volatility
        max_drawdown      → worst drop from peak

    - If these columns are missing, defaults to "medium".
    """
    df = df.copy()

    if "daily_return_std" not in df.columns or "max_drawdown" not in df.columns:
        df["risk_label"] = "medium"
        return df

    vol = df["daily_return_std"]
    dd = df["max_drawdown"]  # negative (e.g., -0.45)

    vol_low = vol.quantile(0.33)
    vol_high = vol.quantile(0.66)
    dd_low = dd.quantile(0.33)
    dd_high = dd.quantile(0.66)

    def label_row(row):
        v = row["daily_return_std"]
        d = row["max_drawdown"]

        # Low risk: low volatility, mild drawdown
        if (v <= vol_low) and (d >= dd_low):
            return "low"

        # High risk: high volatility, deep drawdown
        if (v >= vol_high) and (d <= dd_high):
            return "high"

        # Everything else: medium
        return "medium"

    df["risk_label"] = df.apply(label_row, axis=1)
    return df


# ---------- LOADING & MERGING DATA ----------

def load_merged_data() -> pd.DataFrame:
    """
    Load sip_features + sip_predictions.
    Ensure:
        - scheme_code
        - scheme_name
        - predicted_cagr_percent
        - category
        - risk_label
    """
    features = pd.read_csv(FEATURE_PATH)
    preds = pd.read_csv(PREDICTIONS_PATH)

    features["scheme_code"] = features["scheme_code"].astype(str)
    preds["scheme_code"] = preds["scheme_code"].astype(str)

    # Merge predictions
    df = features.merge(
        preds[["scheme_code", "predicted_cagr_percent"]],
        on="scheme_code",
        how="inner",
    )

    # Ensure scheme_name present (try backup metadata if missing)
    if "scheme_name" not in df.columns or df["scheme_name"].isna().all():
        if METADATA_PATH.exists():
            meta = pd.read_csv(METADATA_PATH)
            meta["scheme_code"] = meta["scheme_code"].astype(str)
            df = df.merge(
                meta[["scheme_code", "scheme_name"]],
                on="scheme_code",
                how="left",
            )

    # Infer category from scheme_name
    df["category"] = df["scheme_name"].apply(infer_category_from_name)

    # Assign risk_label from volatility/drawdown
    df = assign_risk_label(df)

    # Clean bad prediction rows
    df = df.dropna(subset=["predicted_cagr_percent"])

    return df

def filter_by_recency(df: pd.DataFrame,
                      require_active: bool = True,
                      min_recent_years: int = 3,
                      require_positive_recent: bool = True) -> pd.DataFrame:
    """
    Filter schemes by:
      - is_active flag (optional)
      - having a valid recent_{years}y_cagr_percent
      - optionally positive recent performance

    This ensures we don't recommend dead schemes
    or funds that have been trash for the last few years.
    """
    df = df.copy()

    # Active schemes only
    if require_active and "is_active" in df.columns:
        df = df[df["is_active"] == True]

    col_name = f"recent_{min_recent_years}y_cagr_percent"
    if col_name in df.columns:
        df = df[~df[col_name].isna()]

        if require_positive_recent:
            df = df[df[col_name] > 0.0]

    if df.empty:
        raise RuntimeError(
            "No schemes left after recency filter. "
            "Upstream data or thresholds may be too strict."
        )

    return df


# ---------- FILTERS: CATEGORY & RISK ----------

def filter_by_category(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """
    Filter schemes by category.

    category must be one of:
        "small_cap", "mid_cap", "large_cap", "other"
    """
    df = df.copy()
    category = category.lower()

    valid = {"small_cap", "mid_cap", "large_cap", "other"}
    if category not in valid:
        raise ValueError(f"Invalid category '{category}'. Must be one of {valid}.")

    subset = df[df["category"] == category]

    if subset.empty:
        raise RuntimeError(
            f"No schemes found for category '{category}'. "
            "Agent should handle this and possibly ask user to choose another category."
        )

    return subset


def filter_by_risk(df: pd.DataFrame, risk_level: str) -> pd.DataFrame:
    """
    Filter schemes by desired risk_level.

    risk_level must be one of:
        "low", "medium", "high"

    If too strict, relax slightly:
        - low   → low or medium
        - high  → medium or high
        - medium → medium only
    """
    df = df.copy()
    risk_level = risk_level.lower()

    valid = {"low", "medium", "high"}
    if risk_level not in valid:
        raise ValueError(f"Invalid risk_level '{risk_level}'. Must be one of {valid}.")

    if "risk_label" not in df.columns:
        df["risk_label"] = "medium"

    if risk_level == "low":
        subset = df[df["risk_label"].isin(["low", "medium"])]
    elif risk_level == "high":
        subset = df[df["risk_label"].isin(["medium", "high"])]
    else:  # medium
        subset = df[df["risk_label"] == "medium"]

    if subset.empty:
        raise RuntimeError(
            f"No schemes available for risk_level '{risk_level}'. "
            "Agent should handle and ask user to relax risk preference."
        )

    return subset


# ---------- SIP + HOLD SIMULATION ----------

def simulate_sip_and_hold(
    annual_cagr_percent: float,
    monthly_amount: float,
    sip_years: int,
    hold_years: int,
):
    """
    Simulate:
      - SIP: invest monthly_amount for sip_years
      - HOLD: no new investment, grow for hold_years
    using a constant annual CAGR.

    Returns:
      total_invested, final_value, profit, return_percent, projection_curve
    """
    r = annual_cagr_percent / 100.0
    sip_months = sip_years * 12
    total_years = sip_years + hold_years

    if r <= -1.0:
        # catastrophic case, avoid math blow-up
        return 0.0, 0.0, 0.0, 0.0, []

    # Monthly growth rate
    if r == 0:
        rm = 0.0
    else:
        rm = (1.0 + r) ** (1.0 / 12.0) - 1.0

    # Future value at end of SIP
    if rm == 0:
        fv_sip = monthly_amount * sip_months
    else:
        fv_sip = monthly_amount * ((1.0 + rm) ** sip_months - 1.0) / rm

    # Grow during hold period
    fv_final = fv_sip * ((1.0 + r) ** hold_years)

    total_invested = monthly_amount * sip_months
    profit = fv_final - total_invested
    return_percent = (profit / total_invested * 100.0) if total_invested > 0 else 0.0

    # Build projection curve (yearly)
    curve = []
    for year in range(0, total_years + 1):
        if year == 0:
            curve.append({"year": 0, "value": 0.0})
            continue

        if year <= sip_years:
            months_so_far = year * 12
            if rm == 0:
                value = monthly_amount * months_so_far
            else:
                value = monthly_amount * ((1.0 + rm) ** months_so_far - 1.0) / rm
        else:
            extra_years = year - sip_years
            value = fv_sip * ((1.0 + r) ** extra_years)

        curve.append({"year": year, "value": float(value)})

    return float(total_invested), float(fv_final), float(profit), float(return_percent), curve


# ---------- CORE RECOMMENDER ----------

def get_recommendations(
    monthly_amount: float,
    sip_years: int,
    hold_years: int,
    risk_level: str,
    category: str,
    mode: str = "best",  # "best" or "worst"
    top_k: int = 3,
):
    """
    Main advisor logic.

    REQUIRED:
      - monthly_amount > 0
      - sip_years >= 1
      - hold_years >= 0
      - risk_level in {"low", "medium", "high"}
      - category   in {"small_cap", "mid_cap", "large_cap", "other"}
      - mode       in {"best", "worst"}

    Returns:
      List[dict] of length <= top_k with fields:
        scheme_code, scheme_name, base_fund_name,
        category, risk_label,
        predicted_cagr_percent,
        total_invested, projected_final_value,
        projected_profit, projected_return_percent,
        projection_curve
    """
    # Basic validation (agent should NOT call this with missing inputs)
    if monthly_amount <= 0:
        raise ValueError("monthly_amount must be > 0")

    if sip_years <= 0:
        raise ValueError("sip_years must be > 0")

    if hold_years < 0:
        raise ValueError("hold_years cannot be negative")

    mode = mode.lower()
    if mode not in {"best", "worst"}:
        raise ValueError("mode must be 'best' or 'worst'")

    # 1) Load data + predictions
    df = load_merged_data()

    # 2) Filter by category (MANDATORY)
    df = filter_by_category(df, category)

    # 3) Filter by risk_level (MANDATORY)
    df = filter_by_risk(df, risk_level)

    # 4) Filter by recency / current performance
    df = filter_by_recency(df, require_active=True, min_recent_years=3, require_positive_recent=True)

    # 5) Simulate SIP + hold for each scheme
    recs = []

    for _, row in df.iterrows():
        cagr_pred = float(row["predicted_cagr_percent"])

        total_invested, fv_final, profit, ret_pct, curve = simulate_sip_and_hold(
            annual_cagr_percent=cagr_pred,
            monthly_amount=monthly_amount,
            sip_years=sip_years,
            hold_years=hold_years,
        )

        rec = {
            "scheme_code": row.get("scheme_code"),
            "scheme_name": row.get("scheme_name"),
            "base_fund_name": normalize_scheme_name(row.get("scheme_name", "")),
            "category": row.get("category", "other"),
            "risk_label": row.get("risk_label", "medium"),
            "predicted_cagr_percent": cagr_pred,
            "total_invested": total_invested,
            "projected_final_value": fv_final,
            "projected_profit": profit,
            "projected_return_percent": ret_pct,
            "projection_curve": curve,
        }
        recs.append(rec)

    if not recs:
        raise RuntimeError("No schemes left after filtering. Something is off upstream.")

    # 6) Remove duplicate variants → keep best/worst variant per base fund
    recs_df = pd.DataFrame(recs)
    # Sort by final value first (direction depends on mode)
    asc = True if mode == "worst" else False
    recs_df = recs_df.sort_values("projected_final_value", ascending=asc)

    # group by base_fund_name, keep the first (best/worst variant)
    grouped = recs_df.groupby("base_fund_name", as_index=False).first()

    # Now sort again and take top_k distinct base funds
    grouped = grouped.sort_values("projected_final_value", ascending=asc)
    top_recs_df = grouped.head(top_k)

    top_recs = top_recs_df.to_dict(orient="records")
    return top_recs


def main():
    # Example test:
    monthly_amount = 5000
    sip_years = 6
    hold_years = 4
    risk_level = "medium"
    category = "large_cap"    # "small_cap", "mid_cap", "large_cap", "other"
    mode = "best"             # or "worst"

    recs = get_recommendations(
        monthly_amount=monthly_amount,
        sip_years=sip_years,
        hold_years=hold_years,
        risk_level=risk_level,
        category=category,
        mode=mode,
        top_k=3,
    )

    print(f"Top {len(recs)} {mode.upper()} recommendations for:")
    print(f"₹{monthly_amount}/month, SIP {sip_years} years, hold {hold_years} years, "
          f"risk={risk_level}, category={category}")
    print()

    for i, rec in enumerate(recs, start=1):
        print(f"{i}. {rec['scheme_name']} (Code: {rec['scheme_code']})")
        print(f"   Base Fund:    {rec['base_fund_name']}")
        print(f"   Category:     {rec['category']}")
        print(f"   Risk Label:   {rec['risk_label']}")
        print(f"   Pred CAGR:    {rec['predicted_cagr_percent']:.2f}%")
        print(f"   Invested:     ₹{rec['total_invested']:.2f}")
        print(f"   Final Value:  ₹{rec['projected_final_value']:.2f}")
        print(f"   Profit:       ₹{rec['projected_profit']:.2f} "
              f"({rec['projected_return_percent']:.2f}% total)")
        print()


if __name__ == "__main__":
    main()