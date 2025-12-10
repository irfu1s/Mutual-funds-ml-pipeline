### LOGIC:
# 1. Load cleaned NAV file: data/nav_daily_clean.csv
# 2. For each scheme_code:
#       - first_date, last_date
#       - number of distinct years
#       - number of distinct months with data
#       - span_months = months between first_date and last_date
#       - coverage_ratio = n_months / span_months
# 3. Print:
#       - basic stats
#       - how many schemes have "good" coverage
#       - a few examples of bad ones
# 4. OPTIONAL: save a filtered version (good schemes only) as
#       data/nav_daily_clean_filtered.csv
#    This is just a side-product; running this script ALONE changes nothing
#    in SIP, features, training or inference unless you intentionally use it.

import pandas as pd
from pathlib import Path

NAV_IN_PATH = Path("data/nav_daily_clean.csv")
NAV_OUT_PATH = Path("data/nav_daily_clean_filtered.csv")

# You can tweak these thresholds
MIN_YEARS = 8          # require at least 8 distinct years of data
MIN_COVERAGE = 0.8     # require at least 80% of months present in the span

# Set this to False if you ONLY want diagnostics and NO filtered file
SAVE_FILTERED = True


def main():
    print(f"📂 Loading NAV data from: {NAV_IN_PATH}")
    df = pd.read_csv(NAV_IN_PATH, parse_dates=["date"])
    df["scheme_code"] = df["scheme_code"].astype(str)

    # Derive helper columns
    df["year"] = df["date"].dt.year
    df["year_month"] = df["date"].dt.to_period("M")

    # Group per scheme
    grouped = df.groupby("scheme_code")

    stats = grouped.agg(
        first_date=("date", "min"),
        last_date=("date", "max"),
        n_years=("year", "nunique"),
        n_months=("year_month", "nunique"),
    )

    # How many months should exist between first and last date?
    span_months = (
        stats["last_date"].dt.to_period("M") - stats["first_date"].dt.to_period("M")
    ).apply(lambda p: p.n + 1)

    stats["span_months"] = span_months
    stats["coverage_ratio"] = stats["n_months"] / stats["span_months"]

    print("\n🧾 Sample stats for first 5 schemes:")
    print(stats.head())

    print("\n📊 Year span & coverage summary:")
    print(stats[["n_years", "n_months", "span_months", "coverage_ratio"]].describe())

    # Decide which schemes are "good"
    good_mask = (stats["n_years"] >= MIN_YEARS) & (stats["coverage_ratio"] >= MIN_COVERAGE)
    good_schemes = stats[good_mask].index.tolist()

    print(f"\nTotal schemes           : {len(stats)}")
    print(f"Schemes with >= {MIN_YEARS} years data "
          f"and coverage >= {MIN_COVERAGE*100:.0f}% : {len(good_schemes)}")

    # Show some bad examples
    bad = stats[~good_mask].sort_values("coverage_ratio").head(10)
    print("\n❌ Examples of schemes with poor coverage (worst 10):")
    print(bad)

    if SAVE_FILTERED:
        # Create filtered version (optional)
        df_filtered = df[df["scheme_code"].isin(good_schemes)].copy()
        # Drop helper cols before saving
        df_filtered = df_filtered.drop(columns=["year", "year_month"], errors="ignore")
        NAV_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_filtered.to_csv(NAV_OUT_PATH, index=False)
        print(f"\n✅ Saved filtered NAV (good schemes only) → {NAV_OUT_PATH}")
        print(f"Filtered rows: {len(df_filtered):,}")
    else:
        print("\nℹ SAVE_FILTERED is False → no filtered file created. "
              "This run was diagnostics-only.")


if __name__ == "__main__":
    main()