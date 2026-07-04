"""
Phase 1: Data Quality Check
Verifies all downloaded OHLC data for consistency and completeness.
"""

import os
import pandas as pd

DATA_DIR = r"C:\Users\EV-Car\Main_Project_2\data\raw"

files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
print(f"Total files: {len(files)}\n")

results = []
for f in files:
    fp = os.path.join(DATA_DIR, f)
    df = pd.read_csv(fp)
    
    name = f.replace('.csv', '')
    cols = [c.upper() for c in df.columns]
    has_ohlc = all(c in cols for c in ['OPEN', 'HIGH', 'LOW', 'CLOSE'])
    has_date = 'DATE' in cols
    
    # Find the date column
    date_col = None
    for c in df.columns:
        if c.upper() == 'DATE':
            date_col = c
            break
    
    if date_col:
        dates = pd.to_datetime(df[date_col], errors='coerce')
        start = dates.min()
        end = dates.max()
    else:
        start = end = None
    
    # Count nulls in OHLC columns
    ohlc_cols = [c for c in df.columns if c.upper() in ['OPEN','HIGH','LOW','CLOSE']]
    null_count = df[ohlc_cols].isnull().sum().sum() if ohlc_cols else -1
    
    results.append({
        'File': name,
        'Rows': len(df),
        'Has_OHLC': has_ohlc,
        'Has_Date': has_date,
        'Start': start,
        'End': end,
        'Null_OHLC': null_count,
        'Columns': len(df.columns)
    })
    
summary = pd.DataFrame(results)
print(summary.to_string(index=False))

print("\n--- Issues ---")
no_ohlc = summary[~summary['Has_OHLC']]
if len(no_ohlc):
    print(f"Missing OHLC columns: {list(no_ohlc['File'])}")
else:
    print("All files have OHLC columns: OK")

nulls = summary[summary['Null_OHLC'] > 0]
if len(nulls):
    print(f"Files with null OHLC values: {list(nulls['File'])} (counts: {list(nulls['Null_OHLC'])})")
else:
    print("No null OHLC values: OK")

# Check column count consistency (Yahoo vs jugaad_data format)
col_counts = summary['Columns'].unique()
print(f"Distinct column counts: {col_counts}")
if len(col_counts) > 1:
    print("WARNING: Mixed data formats detected - will need standardization")

summary.to_csv(r"C:\Users\EV-Car\Main_Project_2\data\data_summary.csv", index=False)
print("\nSummary saved to data/data_summary.csv")
