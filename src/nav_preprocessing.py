### LOGIC:
# 1. Load multi-year NAV history from AMFI (semicolon-separated).
# 2. Sanitize corrupted rows, strip spaces, drop duplicates.
# 3. Rename columns to clean names.
# 4. Convert NAV to float and Date to datetime.
# 5. Sort by scheme_code then date (fixes AMFI disorder).
# 6. Output clean chronological dataset for SIP engine.


import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/nav_history_raw.txt")
OUT_PATH = Path("data/nav_daily_clean.csv")


def load_raw_history(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=";",
        header=None,       # AMFI file has no header
        dtype=str,
        engine="python",
        on_bad_lines="skip"
    )
    print(f"Loaded NAV history: {len(df):,} rows")
    print("RAW COLUMNS (index):", df.columns.tolist())
    return df


def clean_and_standardize(df: pd.DataFrame) -> pd.DataFrame:
    # 1) Assign AMFI standard 8 columns
    df.columns = [
        "scheme_code",        # 0
        "scheme_name",        # 1
        "isin_growth",        # 2
        "isin_div",           # 3
        "nav",                # 4
        "repurchase_price",   # 5
        "sale_price",         # 6
        "date",               # 7
    ]

    # 2) Keep only what we actually need for SIP engine
    df = df[["scheme_code", "scheme_name", "nav", "date"]]

    # 3) Drop rows missing essentials
    df = df.dropna(subset=["scheme_code", "scheme_name", "nav", "date"])

    # 4) Strip whitespace
    df["scheme_code"] = df["scheme_code"].str.strip()
    df["scheme_name"] = df["scheme_name"].str.strip()
    df["nav"] = df["nav"].str.strip()
    df["date"] = df["date"].str.strip()

    # 5) Convert types
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df["date"] = pd.to_datetime(
        df["date"],
        format="%d-%b-%Y",    # Example: 03-Jan-2017
        errors="coerce"
    )

    # 6) Drop rows where conversion failed
    df = df.dropna(subset=["nav", "date"])

    # 7) Sort & deduplicate → one row per scheme per date, in order
    df = (
        df.sort_values(["scheme_code", "date"])
          .drop_duplicates(subset=["scheme_code", "date"])
          .reset_index(drop=True)
    )

    return df


def main():
    df_raw = load_raw_history()
    df_clean = clean_and_standardize(df_raw)

    # 🔍 Dtype verification
    print("\nDtypes after cleaning:")
    print(df_clean.dtypes)

    # Explicit proof that date is datetime
    print("\nExample date value & type:")
    print(df_clean["date"].iloc[0], type(df_clean["date"].iloc[0]))

    # Quick sanity sample
    print("\nSample cleaned rows:")
    print(df_clean.head())

    # Distribution of available unique dates per scheme
    print("\nDate coverage per scheme (unique date counts):")
    print(df_clean.groupby("scheme_code")["date"].nunique().describe())

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUT_PATH, index=False)
    print(f"\n✅ Saved clean chronological NAV → {OUT_PATH}")


if __name__ == "__main__":
    main()