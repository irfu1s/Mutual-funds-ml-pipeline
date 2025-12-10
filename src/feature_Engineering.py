### LOGIC:
# 1. Load nav_daily_clean.csv and sip_results_active_clean.csv
# 2. Create a permanent identity backup:
#       scheme_code → scheme_name
#    Saved as: data/scheme_metadata.csv
#
# 3. Merge scheme_name into SIP features (optional but recommended)
#
# 4. Continue feature engineering normally:
#       - risk stats
#       - ratios
#       - logs
#       - cagr features
#
# 5. Save sip_features.csv with scheme_name included
#
# This ensures:
#   - Recommender can work even if NAV is missing
#   - Identity mapping is ALWAYS recoverable
#   - ML pipeline remains stable long-term


import pandas as pd
import numpy as np
from pathlib import Path

NAV_DAILY_PATH = Path("data/nav_daily_clean_filtered.csv")
SIP_RESULTS_PATH = Path("data/sip_results_active_clean.csv")
OUT_FEATURES_PATH = Path("data/sip_features.csv")
OUT_FEATURES_CLEAN_PATH = Path("data/sip_features_model_ready.csv")
IDENTITY_BACKUP_PATH = Path("data/scheme_metadata.csv")


def load_data():
    nav = pd.read_csv(NAV_DAILY_PATH)
    sip = pd.read_csv(SIP_RESULTS_PATH)

    # Ensure dtypes
    nav["scheme_code"] = nav["scheme_code"].astype(str)
    sip["scheme_code"] = sip["scheme_code"].astype(str)

    return nav, sip


def create_identity_backup(nav):
    """
    Save scheme_code ↔ scheme_name mapping for restoring metadata if ML drops names.
    """
    metadata = nav[["scheme_code", "scheme_name"]].drop_duplicates()

    IDENTITY_BACKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(IDENTITY_BACKUP_PATH, index=False)

    print(f"🔒 Saved identity backup → {IDENTITY_BACKUP_PATH}")


def merge_scheme_name(nav, sip):
    """
    Attach scheme_name to SIP features.
    """
    metadata = nav[["scheme_code", "scheme_name"]].drop_duplicates()
    sip = sip.merge(metadata, on="scheme_code", how="left")

    return sip


def add_sip_features(sip: pd.DataFrame) -> pd.DataFrame:
    df = sip.copy()

    # Ensure numeric columns
    numeric_cols = ["total_invested", "final_value", "gain", "cagr_percent"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derived features
    df["cagr_decimal"] = df["cagr_percent"] / 100.0
    df["abs_return_pct"] = df["gain"] / df["total_invested"]
    df["final_value_ratio"] = df["final_value"] / df["total_invested"]

    # Log transforms
    df["log_total_invested"] = np.log1p(df["total_invested"])
    df["log_final_value"] = np.log1p(df["final_invested"]) if "final_invested" in df else np.log1p(df["final_value"])
    df["log_gain_plus1"] = np.log1p(df["gain"] + 1)

    return df


def compute_recent_stats(nav: pd.DataFrame, years: int = 3) -> pd.DataFrame:
    """
    Compute:
      - last_nav_date per scheme
      - is_active flag (has NAV in last 60 days)
      - recent_{years}y_cagr_percent based on NAV change over last N years
      - max_drawdown based on historical NAV path
    """
    df = nav.copy()

    # Ensure correct types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["scheme_code", "date", "nav"])

    df = df.sort_values(["scheme_code", "date"])

    # Global latest date in dataset
    global_last_date = df["date"].max()
    cutoff_date = global_last_date - pd.DateOffset(years=years)

    # last NAV date per scheme
    last_dates = (
        df.groupby("scheme_code")["date"]
          .max()
          .rename("last_nav_date")
    )

    # active if updated in last 60 days
    is_active = (global_last_date - last_dates).dt.days <= 60
    active_flag = is_active.rename("is_active")

    # filter to last N years for CAGR calc
    recent = df[df["date"] >= cutoff_date].copy()

    def recent_cagr_for_scheme(scheme_df: pd.DataFrame) -> float:
        s_df = scheme_df.sort_values("date")
        if len(s_df) < 2:
            return np.nan

        first_date = s_df["date"].iloc[0]
        last_date = s_df["date"].iloc[-1]
        first_nav = s_df["nav"].iloc[0]
        last_nav = s_df["nav"].iloc[-1]

        days = (last_date - first_date).days
        if days <= 0 or first_nav <= 0:
            return np.nan

        years_span = days / 365.25
        cagr = (last_nav / first_nav) ** (1 / years_span) - 1
        return cagr * 100.0

    recent_cagr = (
        recent.groupby("scheme_code")
              .apply(recent_cagr_for_scheme)
              .rename(f"recent_{years}y_cagr_percent")
    )

    # Combine last date, active flag, and recent CAGR into stats
    stats = pd.concat([last_dates, active_flag, recent_cagr], axis=1)

    # ---- Max Drawdown ----
    def max_drawdown(series: pd.Series) -> float:
        roll_max = series.cummax()
        dd = (series - roll_max) / roll_max
        return dd.min()

    mdd = (
        df.dropna(subset=["nav"])
          .groupby("scheme_code")["nav"]
          .apply(max_drawdown)
          .rename("max_drawdown")
    )

    # Join mdd into stats (index aligned on scheme_code)
    stats = stats.join(mdd, how="left")

    # Return with scheme_code as a column
    return stats.reset_index()


def build_features():
    nav, sip = load_data()

    # 1. Create permanent identity backup
    create_identity_backup(nav)

    # 2. Add scheme_name to SIP features
    sip = merge_scheme_name(nav, sip)

    # 3. Add engineered SIP features
    sip_feat = add_sip_features(sip)

    # 4. Recent stats (last_nav_date, is_active, recent_3y_cagr_percent, max_drawdown)
    recent_stats: pd.DataFrame = compute_recent_stats(nav, years=3)

    # 5. Merge SIP + recent stats
    features = sip_feat.merge(recent_stats, on="scheme_code", how="left")

    # 6. Remove critical missing values (WITHOUT daily_return_std)
    features = features.dropna(
        subset=[
            "total_invested",
            "final_value",
            "gain",
            "cagr_percent",
        ]
    )

    # 7. Save sip_features.csv WITH scheme_name
    OUT_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUT_FEATURES_PATH, index=False)

    print(f"✅ Saved engineered features → {OUT_FEATURES_PATH}")
    print("Sample:")
    print(features.head())

    # 8. Build model-ready clean features: numeric-only, finite-only
    features_numeric = features.select_dtypes(include=[np.number])

    # Replace infinities with NaN
    features_numeric = features_numeric.replace([np.inf, -np.inf], np.nan)

    # Keep only rows where all numeric features are finite (no NaN after replacement)
    mask_finite = ~features_numeric.isna().any(axis=1)
    features_clean = features.loc[mask_finite].copy()

    # Save model-ready dataset
    OUT_FEATURES_CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_clean.to_csv(OUT_FEATURES_CLEAN_PATH, index=False)

    print(f"\n✅ Saved model-ready features → {OUT_FEATURES_CLEAN_PATH}")
    print(f"   Raw schemes   : {features.shape[0]}")
    print(f"   Clean schemes : {features_clean.shape[0]}")




def main():
    build_features()


if __name__ == "__main__":
    main()
