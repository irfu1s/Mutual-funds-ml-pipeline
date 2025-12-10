### LOGIC:
# 1. Take clean daily NAV data for a single scheme.
# 2. Resample it to one NAV per month (start or end of month).
# 3. Simulate investing a fixed amount every month (SIP).
# 4. Track units, invested amount, and portfolio value.
# 5. Return the per-month history + summary stats (gain, CAGR, etc.).

from __future__ import annotations
import pandas as pd
import numpy as np


def _monthly_nav_from_daily(
    df_scheme: pd.DataFrame,
    month_day: str = "start",
) -> pd.DataFrame:
    """
    Reduce daily NAV for ONE scheme into monthly NAV.

    month_day = "start" → first trading day NAV
    month_day = "end"   → last trading day NAV
    """
    df = df_scheme.copy()
    df = df.sort_values("date")

    df["year_month"] = df["date"].dt.to_period("M")

    if month_day == "start":
        monthly = (
            df.groupby("year_month")
              .first()[["date", "nav"]]
              .reset_index()
        )
    else:
        monthly = (
            df.groupby("year_month")
              .last()[["date", "nav"]]
              .reset_index()
        )

    return monthly


### LOGIC:
# Calculate SIP for a single scheme.
# - Aggregate daily NAV → monthly NAV
# - If there aren't enough months, return (None, None) instead of crashing.
# - Otherwise, return (monthly_df, summary_dict).

def calculate_sip_for_scheme(
    nav_df,
    scheme_code,
    monthly_amount=1000,
    min_months=24,
):
    fund = nav_df[nav_df["scheme_code"] == scheme_code].copy()
    fund = fund.sort_values("date")

    # convert to monthly using last NAV of month (you may be using first, that's fine too)
    fund["year_month"] = fund["date"].dt.to_period("M")
    monthly = (
        fund.groupby("year_month")
            .last()  # or .first()
            .reset_index()
    )

    # 👇 THIS IS THE IMPORTANT CHANGE
    if len(monthly) < min_months:
        # not enough history → tell caller to skip
        return None, None

    # ---- your existing SIP logic below ----
    monthly["investment"] = monthly_amount
    monthly["units"] = monthly["investment"] / monthly["nav"]
    monthly["cum_units"] = monthly["units"].cumsum()
    monthly["cum_invested"] = monthly["investment"].cumsum()
    monthly["portfolio_value"] = monthly["cum_units"] * monthly["nav"]

    total_invested = float(monthly["cum_invested"].iloc[-1])
    final_value = float(monthly["portfolio_value"].iloc[-1])
    gain = final_value - total_invested

    months = len(monthly)
    years = months / 12.0
    if years <= 0 or total_invested <= 0:
        cagr = 0.0
    else:
        cagr = ((final_value / total_invested) ** (1 / years) - 1) * 100

    summary = {
        "scheme_code": scheme_code,
        "months": months,
        "years": years,
        "total_invested": total_invested,
        "final_value": final_value,
        "gain": gain,
        "cagr_percent": cagr,
    }

    return monthly, summary



### LOGIC:
# Loop over all schemes:
# - Try to compute SIP
# - If a scheme has too few months, we get (None, None) and skip it
# - Collect summaries for all valid schemes and save once at the end.

if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    nav_df = pd.read_csv(r"C:\Users\irfui\mutual\data\nav_daily_clean_filtered.csv", parse_dates=["date"])
    unique_codes = sorted(nav_df["scheme_code"].unique())

    results = []
    skipped_short = []

    for i, sc in enumerate(unique_codes, start=1):
        monthly_df, summary = calculate_sip_for_scheme(
            nav_df,
            scheme_code=sc,
            monthly_amount=1000,
            min_months=24,  # tune this if you want
        )

        if summary is None:
            skipped_short.append(sc)
            continue

        results.append(summary)

        if i % 500 == 0:
            print(f"Processed {i}/{len(unique_codes)} schemes")

    # Save the combined summary
    out_path = Path("data") / "sip_results_active_clean.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)

    print(f"\n✅ Saved SIP summaries for {len(results)} schemes → {out_path}")
    print(f"⚠ Skipped {len(skipped_short)} schemes with too few months of data.")