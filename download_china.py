"""
Cross-Market Generalisation -- Chinese Market (CSI 300) Data Collection
Downloads SSE Composite index + top-50 Chinese A-share stocks (2000-2025)

Uses yfinance tickers:
  Shanghai: XXXXXX.SS
  Shenzhen: XXXXXX.SZ
"""

import yfinance as yf
import pandas as pd
import os
import time

# -- Paths (separate from NIFTY-50 and S&P 500 data) -------------------------
BASE_DIR = r"C:\Users\EV-Car\Main_Project_2_Review1"
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw_china")
os.makedirs(RAW_DIR, exist_ok=True)

# -- Top ~50 CSI 300 large-cap stocks with long data history ------------------
# Mix of Shanghai (.SS) and Shenzhen (.SZ) stocks
# Selected for: large market cap, high liquidity, and data availability pre-2010
CHINA_TICKERS = [
    # Financials
    "601398.SS",  # ICBC
    "601939.SS",  # China Construction Bank
    "601288.SS",  # Agricultural Bank of China
    "601988.SS",  # Bank of China
    "600036.SS",  # China Merchants Bank
    "601166.SS",  # Industrial Bank
    "600016.SS",  # Minsheng Bank
    "601328.SS",  # Bank of Communications
    "601318.SS",  # Ping An Insurance
    "601601.SS",  # PICC Property
    "600030.SS",  # CITIC Securities
    # Energy / Materials
    "601857.SS",  # PetroChina
    "600028.SS",  # Sinopec
    "601088.SS",  # China Shenhua Energy
    "600019.SS",  # Baoshan Iron & Steel
    "601600.SS",  # Aluminum Corp of China
    "600585.SS",  # Conch Cement
    # Consumer / Healthcare
    "600519.SS",  # Kweichow Moutai
    "000858.SZ",  # Wuliangye Yibin
    "000333.SZ",  # Midea Group
    "000651.SZ",  # Gree Electric
    "600276.SS",  # Hengrui Medicine
    "000568.SZ",  # Luzhou Laojiao
    "002304.SZ",  # Yanghe Brewery
    "600887.SS",  # Inner Mongolia Yili
    "000661.SZ",  # ChangChun High-Tech
    # Technology / Telecom
    "600050.SS",  # China Unicom
    "601728.SS",  # China Telecom
    "000063.SZ",  # ZTE
    "002415.SZ",  # Hikvision
    "600588.SS",  # Yonyou Network
    "002230.SZ",  # iFlytek
    "300059.SZ",  # East Money Info
    # Industrials / Infrastructure
    "601668.SS",  # China State Construction
    "601800.SS",  # China Communications Construction
    "601766.SS",  # China South Locomotive (CRRC)
    "600104.SS",  # SAIC Motor
    "601006.SS",  # Daqin Railway
    "600009.SS",  # Shanghai International Airport
    "600900.SS",  # China Yangtze Power
    "600690.SS",  # Haier Smart Home
    # Real Estate / Utilities
    "000002.SZ",  # China Vanke
    "600048.SS",  # Poly Developments
    "600886.SS",  # Huaneng Power
    # Additional large caps
    "601138.SS",  # Industrial Securities
    "601169.SS",  # Bank of Beijing
    "600000.SS",  # Shanghai Pudong Development Bank
    "600015.SS",  # Hua Xia Bank
    "601186.SS",  # China Railway Construction
]

# Index for regime detection
CHINA_INDEX_TICKER = "000001.SS"  # SSE Composite Index

START_DATE = "2000-01-01"
END_DATE   = "2025-12-31"


def download_index():
    """Download SSE Composite index for HMM regime detection."""
    print("=" * 65)
    print("Downloading SSE Composite Index (000001.SS) for regime detection...")
    print("=" * 65)
    try:
        data = yf.download(CHINA_INDEX_TICKER, start=START_DATE, end=END_DATE,
                           progress=False, auto_adjust=True)
        if data.empty:
            print("  WARNING: No data received for SSE Composite")
            return False
        # Flatten MultiIndex if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        filepath = os.path.join(RAW_DIR, "SSE_INDEX.csv")
        data.to_csv(filepath)
        print(f"  OK Saved : {filepath}")
        print(f"  Rows     : {len(data)}")
        print(f"  Range    : {data.index[0].date()} -> {data.index[-1].date()}")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def download_stocks():
    """Download all Chinese A-share constituent stocks."""
    print(f"\n{'=' * 65}")
    print(f"Downloading {len(CHINA_TICKERS)} Chinese A-share Stocks...")
    print(f"{'=' * 65}")

    success, failed = [], []

    for i, ticker in enumerate(CHINA_TICKERS, 1):
        print(f"  [{i:2d}/{len(CHINA_TICKERS)}] {ticker:12s}...", end=" ", flush=True)
        try:
            data = yf.download(ticker, start=START_DATE, end=END_DATE,
                               progress=False, auto_adjust=True)
            if data.empty:
                print("NO DATA")
                failed.append(ticker)
                continue

            # Flatten MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # Safe filename: replace dots
            safe_name = ticker.replace(".", "_")
            filepath = os.path.join(RAW_DIR, f"{safe_name}.csv")
            data.to_csv(filepath)
            print(f"OK  ({len(data)} rows, "
                  f"{data.index[0].date()} -> {data.index[-1].date()})")
            success.append(ticker)
        except Exception as e:
            print(f"FAILED  ({e})")
            failed.append(ticker)

        time.sleep(0.5)  # polite rate-limiting

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
    out_path = os.path.join(BASE_DIR, "data", "data_summary_china.csv")
    summary.to_csv(out_path, index=False)
    print(summary.to_string(index=False))
    print(f"\n  Summary saved -> {out_path}")


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  HMM-GP Alpha Cross-Market -- Chinese A-Share Data Download")
    print("=" * 65 + "\n")

    ok_idx = download_index()
    success, failed = download_stocks()

    print(f"\n  Successfully downloaded : {len(success)}/{len(CHINA_TICKERS)} stocks")
    if failed:
        print(f"  Failed                  : {failed}")
    if ok_idx:
        print(f"  Index (SSE Composite)   : OK")

    generate_summary(success)

    print("\n" + "=" * 65)
    print("  Chinese A-Share Data Download Complete!")
    print(f"  Data saved to: {RAW_DIR}")
    print("=" * 65)
