"""
Cross-Market Generalisation — Phase 1: S&P 500 Data Collection
Downloads S&P 500 index + top-50 constituent stocks OHLC data (2000-2025)

This script is part of the IEEE Access revision generalisation experiment.
Saves data to data/raw_sp500/ — does NOT touch the original NIFTY-50 data.
"""

import yfinance as yf
import pandas as pd
import os
import time

# ── Paths (separate from NIFTY-50 data) ─────────────────────────────────────
BASE_DIR  = r"C:\Users\EV-Car\Main_Project_2_Review1"
RAW_DIR   = os.path.join(BASE_DIR, "data", "raw_sp500")
os.makedirs(RAW_DIR, exist_ok=True)

# ── S&P 500 Top-50 by Market Cap (stable large-caps with long history) ───────
# Note: Using tickers that existed pre-2007 where possible.
# BRK-B included as proxy for Berkshire; avoids survivorship bias in mega-caps.
SP500_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "BRK-B", "LLY",
    "AVGO", "TSLA", "JPM", "WMT", "UNH", "XOM", "V", "MA", "ORCL",
    "COST", "HD", "PG", "JNJ", "ABBV", "BAC", "NFLX", "KO",
    "CRM", "CVX", "MRK", "AMD", "PEP", "TMO", "LIN", "ACN", "MCD",
    "CSCO", "ABT", "GE", "IBM", "DHR", "ADBE", "TXN", "NEE", "CAT",
    "NOW", "RTX", "AMGN", "INTC", "QCOM", "VZ", "INTU", "NVDA"
]

# S&P 500 Index for regime detection
SP500_INDEX_TICKER = "^GSPC"

# Match the NIFTY-50 paper date range exactly
START_DATE = "2000-01-01"
END_DATE   = "2025-12-31"


def download_index():
    """Download S&P 500 index data for HMM regime detection."""
    print("=" * 65)
    print("Downloading S&P 500 Index (^GSPC) for regime detection...")
    print("=" * 65)
    try:
        data = yf.download(SP500_INDEX_TICKER, start=START_DATE, end=END_DATE,
                           progress=False, auto_adjust=True)
        if data.empty:
            print("  WARNING: No data received for ^GSPC")
            return False
        filepath = os.path.join(RAW_DIR, "SP500_INDEX.csv")
        data.to_csv(filepath)
        print(f"  OK Saved : {filepath}")
        print(f"  Rows    : {len(data)}")
        print(f"  Range   : {data.index[0].date()} -> {data.index[-1].date()}")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def download_stocks():
    """Download all 50 S&P 500 constituent stocks."""
    print(f"\n{'=' * 65}")
    print(f"Downloading {len(SP500_TICKERS)} S&P 500 Constituent Stocks...")
    print(f"{'=' * 65}")

    success, failed = [], []

    for i, ticker in enumerate(SP500_TICKERS, 1):
        print(f"  [{i:2d}/{len(SP500_TICKERS)}] {ticker:12s}...", end=" ", flush=True)
        try:
            data = yf.download(ticker, start=START_DATE, end=END_DATE,
                               progress=False, auto_adjust=True)
            if data.empty:
                print("NO DATA")
                failed.append(ticker)
                continue

            # Flatten MultiIndex columns if present (yfinance quirk)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            filepath = os.path.join(RAW_DIR, f"{ticker.replace('-', '_')}.csv")
            data.to_csv(filepath)
            print(f"OK  ({len(data)} rows, "
                  f"{data.index[0].date()} -> {data.index[-1].date()})")
            success.append(ticker)
        except Exception as e:
            print(f"FAILED  ({e})")
            failed.append(ticker)

        time.sleep(0.4)   # polite rate-limiting

    return success, failed


def generate_summary(success):
    """Generate a data coverage summary CSV."""
    print(f"\n{'=' * 65}")
    print("Data Coverage Summary")
    print(f"{'=' * 65}")

    rows_list = []
    all_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.csv')]
    for fname in sorted(all_files):
        try:
            df = pd.read_csv(os.path.join(RAW_DIR, fname), index_col=0,
                             parse_dates=True)
            rows_list.append({
                "Stock"      : fname.replace(".csv", ""),
                "Rows"       : len(df),
                "Start"      : str(df.index[0].date()),
                "End"        : str(df.index[-1].date()),
                "Missing_Pct": round(
                    df.isnull().sum().sum() / max(len(df) * len(df.columns), 1) * 100, 2
                ),
            })
        except Exception:
            pass

    summary = pd.DataFrame(rows_list)
    out_path = os.path.join(BASE_DIR, "data", "data_summary_sp500.csv")
    summary.to_csv(out_path, index=False)
    print(summary.to_string(index=False))
    print(f"\n  Summary saved -> {out_path}")


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  HMM-GP Alpha Cross-Market Generalisation - S&P 500 Data Download")
    print("=" * 65 + "\n")

    ok_idx = download_index()
    success, failed = download_stocks()

    print(f"\n  Successfully downloaded : {len(success)}/{len(SP500_TICKERS)} stocks")
    if failed:
        print(f"  Failed                  : {failed}")
    if ok_idx:
        print(f"  Index (^GSPC)           : OK")

    generate_summary(success)

    print("\n" + "=" * 65)
    print("  Phase 1 — S&P 500 Data Download Complete!")
    print(f"  Data saved to: {RAW_DIR}")
    print("=" * 65)
