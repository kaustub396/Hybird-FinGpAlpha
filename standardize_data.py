"""
Phase 1: Standardize all OHLC data to uniform format.
Output format: Date, Open, High, Low, Close, Volume
All saved to data/processed/
"""

import os
import pandas as pd

RAW_DIR = r"C:\Users\EV-Car\Main_Project_2\data\raw"
PROC_DIR = r"C:\Users\EV-Car\Main_Project_2\data\processed"
os.makedirs(PROC_DIR, exist_ok=True)

files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('.csv')])
print(f"Standardizing {len(files)} files...\n")

stats = []

for f in files:
    name = f.replace('.csv', '')
    fp = os.path.join(RAW_DIR, f)
    
    if name == 'NIFTY50_INDEX':
        # Your original file: clean Date, Open, High, Low, Close (no Volume)
        df = pd.read_csv(fp)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        df['Volume'] = 0  # No volume data for index
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    elif name == 'TATAMOTORS':
        # jugaad_data format: Date has timestamp, reverse chronological
        df = pd.read_csv(fp)
        df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
        df = df.sort_values('Date').reset_index(drop=True)
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    else:
        # Yahoo Finance multi-level header format
        # Row 0: "Ticker", ticker names
        # Row 1: "Date", NaN...
        # Row 2+: actual data
        df = pd.read_csv(fp, skiprows=[0, 1])  # Skip ticker and empty rows
        df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        
        # Convert OHLCV to numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.sort_values('Date').reset_index(drop=True)
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Drop rows with NaN in OHLC
    before = len(df)
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    dropped = before - len(df)
    
    # Save
    out_path = os.path.join(PROC_DIR, f)
    df.to_csv(out_path, index=False)
    
    stats.append({
        'Stock': name,
        'Rows': len(df),
        'Start': str(df['Date'].iloc[0].date()),
        'End': str(df['Date'].iloc[-1].date()),
        'Dropped': dropped
    })
    
    print(f"  {name:20s} | {len(df):5d} rows | {df['Date'].iloc[0].date()} to {df['Date'].iloc[-1].date()} | dropped: {dropped}")

summary = pd.DataFrame(stats)
summary.to_csv(os.path.join(PROC_DIR, '..', 'data_summary.csv'), index=False)

print(f"\n{'='*60}")
print(f"All {len(files)} files standardized to data/processed/")
print(f"Format: Date, Open, High, Low, Close, Volume")
print(f"Total rows dropped (NaN OHLC): {summary['Dropped'].sum()}")
print(f"{'='*60}")
