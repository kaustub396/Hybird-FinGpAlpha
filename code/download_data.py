"""
Main_Project_2 - Phase 1: Data Collection
Downloads NIFTY-50 index + 50 constituent stocks OHLC data (2000-2025)
"""

import yfinance as yf
import pandas as pd
import os
import time

DATA_DIR = r"C:\Users\EV-Car\Main_Project_2\data\raw"

# NIFTY-50 constituent stocks (current composition)
# Yahoo Finance tickers for NSE stocks use .NS suffix
NIFTY50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "TITAN.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "NTPC.NS", "TATAMOTORS.NS",
    "WIPRO.NS", "ONGC.NS", "NESTLEIND.NS", "POWERGRID.NS", "M&M.NS",
    "JSWSTEEL.NS", "TATASTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS", "HCLTECH.NS",
    "COALINDIA.NS", "BAJAJFINSV.NS", "GRASIM.NS", "TECHM.NS", "INDUSINDBK.NS",
    "HINDALCO.NS", "CIPLA.NS", "DRREDDY.NS", "EICHERMOT.NS", "DIVISLAB.NS",
    "BPCL.NS", "BRITANNIA.NS", "APOLLOHOSP.NS", "TATACONSUM.NS", "HEROMOTOCO.NS",
    "SBILIFE.NS", "BAJAJ-AUTO.NS", "HDFCLIFE.NS", "SHRIRAMFIN.NS", "BEL.NS"
]

# NIFTY-50 Index
NIFTY_INDEX_TICKER = "^NSEI"

START_DATE = "2000-01-01"
END_DATE = "2025-12-31"


def download_index():
    """Download NIFTY-50 index data"""
    print("=" * 60)
    print("Downloading NIFTY-50 Index...")
    print("=" * 60)
    
    try:
        data = yf.download(NIFTY_INDEX_TICKER, start=START_DATE, end=END_DATE, progress=False)
        if data.empty:
            print("WARNING: No data received for NIFTY-50 index")
            return False
        
        filepath = os.path.join(DATA_DIR, "NIFTY50_INDEX.csv")
        data.to_csv(filepath)
        print(f"  Saved: {filepath}")
        print(f"  Rows: {len(data)} | Date range: {data.index[0].date()} to {data.index[-1].date()}")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def download_stocks():
    """Download all 50 constituent stocks"""
    print("\n" + "=" * 60)
    print("Downloading 50 NIFTY Constituent Stocks...")
    print("=" * 60)
    
    success = []
    failed = []
    
    for i, ticker in enumerate(NIFTY50_TICKERS, 1):
        name = ticker.replace(".NS", "")
        print(f"  [{i:2d}/50] {name}...", end=" ")
        
        try:
            data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if data.empty:
                print("NO DATA")
                failed.append(ticker)
                continue
            
            filepath = os.path.join(DATA_DIR, f"{name}.csv")
            data.to_csv(filepath)
            print(f"OK ({len(data)} rows, {data.index[0].date()} to {data.index[-1].date()})")
            success.append(ticker)
            
        except Exception as e:
            print(f"FAILED ({e})")
            failed.append(ticker)
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    return success, failed


def generate_summary():
    """Generate summary of downloaded data"""
    print("\n" + "=" * 60)
    print("Data Summary")
    print("=" * 60)
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    
    summary_rows = []
    for f in sorted(files):
        filepath = os.path.join(DATA_DIR, f)
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        summary_rows.append({
            'File': f,
            'Rows': len(df),
            'Start': str(df.index[0].date()),
            'End': str(df.index[-1].date()),
            'Missing_Pct': round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2)
        })
    
    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(DATA_DIR, "..", "data_summary.csv")
    summary.to_csv(summary_path, index=False)
    
    print(f"\n  Total files: {len(files)}")
    print(f"  Summary saved to: {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # NIFTY-50 index already exists at data/raw/NIFTY50_INDEX.csv
    print("NIFTY-50 index data already present. Skipping index download.")
    
    # Download constituent stocks only
    success, failed = download_stocks()
    
    print(f"\n  Successfully downloaded: {len(success)}/50")
    if failed:
        print(f"  Failed: {[t.replace('.NS','') for t in failed]}")
    
    # Summary
    generate_summary()
    
    print("\n" + "=" * 60)
    print("Phase 1 Data Collection Complete!")
    print("=" * 60)
