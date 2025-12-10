### LOGIC:
# 1. Walk from start_date → today in 90-day windows.
# 2. For each window:
#       - Make AMFI request (no JS)
#       - Retry on failure (exponential backoff)
#       - Append to nav_history_raw.txt
# 3. Never silently skip missing data.
# 4. Later: nav_preprocessing.py cleans + sorts.


import os
import requests
from datetime import datetime, timedelta
import time

DATA_DIR = "data"
RAW_PATH = os.path.join(DATA_DIR, "nav_history_raw.txt")

BASE_URL = "https://portal.amfiindia.com/DownloadNAVHistoryReport_Po.aspx"


def download_window(start_date, end_date, write_mode):
    frmdt = start_date.strftime("%d-%b-%Y")
    todt = end_date.strftime("%d-%b-%Y")

    params = {
        "tp": "1",
        "frmdt": frmdt,
        "todt": todt,
    }

    print(f"\n📥 Downloading: {frmdt} → {todt}")

    max_retries = 4
    delay = 5
    text = None
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=120)
            resp.raise_for_status()

            if len(resp.text) < 200:
                raise RuntimeError("Response too small. Probably empty.")

            text = resp.text
            print(f"✅ Download OK (attempt {attempt}, len={len(text)})")
            break

        except Exception as e:
            last_err = e
            print(f"⚠ Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                print(f"⏳ Retrying in {delay} seconds…")
                time.sleep(delay)
                delay *= 2

    # Hard-stop on failure (DO NOT silently lose data)
    if text is None:
        raise RuntimeError(
            f"❌ Failed to fetch window {frmdt} → {todt}. Last error: {last_err}"
        )

    os.makedirs(DATA_DIR, exist_ok=True)

    # append mode skips header to avoid duplication
    if write_mode == "a":
        lines = text.splitlines()
        text_to_write = "\n".join(lines[1:]) + "\n"
    else:
        text_to_write = text if text.endswith("\n") else (text + "\n")

    with open(RAW_PATH, write_mode, encoding="utf-8") as f:
        f.write(text_to_write)

    print(f"💾 Written: {frmdt} → {todt} (mode={write_mode})")


def main():
    # YOU CAN CHANGE THIS START YEAR IF YOU WANT
    start_year = 2015

    start_date = datetime(start_year, 1, 1)
    end_date = datetime.now()

    # wipe old file
    if os.path.exists(RAW_PATH):
        os.remove(RAW_PATH)

    window_days = 90
    mode = "w"

    cur = start_date

    while cur <= end_date:
        nxt = cur + timedelta(days=window_days - 1)
        if nxt > end_date:
            nxt = end_date

        download_window(cur, nxt, write_mode=mode)
        mode = "a"  # all next windows append

        cur = nxt + timedelta(days=1)

    size = os.path.getsize(RAW_PATH) / 1024 / 1024
    print(f"\n🎉 DONE. Saved raw history → {RAW_PATH} ({size:.2f} MB)")


if __name__ == "__main__":
    main()