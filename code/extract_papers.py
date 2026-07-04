"""Extract datasets and metrics from downloaded research papers."""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import fitz
import re
import os

papers_dir = r"C:\Users\EV-Car\Downloads\Research 2.0-20260605T171613Z-3-001\Research 2.0"

# All 15 papers
paper_names = {
    1: "QuantFactor REINFORCE",
    2: "AlphaQCM",
    3: "STGP-SATA",
    4: "UMI Irrationality",
    5: "AlphaCFG",
    6: "QuantEvolver",
    7: "Alpha Jungle (MCTS)",
    8: "DRL Portfolio",
    9: "PLTA-FinBERT",
    10: "Moirai-MoE",
    11: "Kronos",
    12: "FinCast",
    13: "LFTD",
    14: "Toto 2.0",
    15: "NRSM-MIA",
}

dataset_keywords = [
    "csi 300", "csi 500", "csi300", "csi500", "csi 800", "csi800",
    "s&p 500", "s&p500", "sp500", "sp 500",
    "nasdaq", "nyse", "a-share", "a share",
    "nifty", "ftse", "hang seng", "dax", "nikkei",
    "qlib", "crsp", "compustat",
]

metric_keywords = [
    "ic ", "icir", "rank ic", "rankic", "sharpe",
    "annual return", "annualized return", "ann. return",
    "max drawdown", "maximum drawdown", "mdd",
    "win rate", "information ratio",
]

sep = "=" * 65

for num in range(1, 16):
    path = os.path.join(papers_dir, f"#{num}.pdf")
    if not os.path.exists(path):
        continue
    
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    
    name = paper_names.get(num, f"Paper #{num}")
    print(f"\n{sep}")
    print(f"  PAPER #{num}: {name}")
    print(sep)
    
    text_lower = text.lower()
    
    # Check which datasets are mentioned
    found_datasets = set()
    for kw in dataset_keywords:
        if kw in text_lower:
            found_datasets.add(kw)
    
    print(f"  Datasets found: {sorted(found_datasets) if found_datasets else 'NONE'}")
    
    # Extract lines with numeric metrics near IC/Sharpe/Return
    lines = text.split("\n")
    metric_lines = []
    for i, line in enumerate(lines):
        ll = line.lower().strip()
        if len(ll) < 5:
            continue
        # Look for lines with numbers + metric keywords
        has_metric = any(mk in ll for mk in metric_keywords)
        has_number = bool(re.search(r"\d+\.\d+", line))
        if has_metric and has_number and len(line.strip()) > 10:
            metric_lines.append(line.strip()[:180])
    
    if metric_lines:
        print(f"  Key metric lines ({len(metric_lines)} found, showing top 15):")
        for ml in metric_lines[:15]:
            print(f"    > {ml}")
    else:
        print("  No numeric metric lines found in text extraction")
    
    # Check if GP is mentioned
    gp_mentions = text_lower.count("genetic programming")
    print(f"  'genetic programming' mentions: {gp_mentions}")
