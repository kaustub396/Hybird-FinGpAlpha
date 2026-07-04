import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

FULL_YEARS = pd.DataFrame({"Year": list(range(2005, 2026))})

CANONICAL_COLS = {
    "roe": "ROE", "roe_%": "ROE", "return_on_equity": "ROE",
    "roce": "ROCE", "roce_%": "ROCE", "return_on_capital_employed": "ROCE",
    "current_ratio": "Current_Ratio",
    "debt_equity": "Debt_Equity", "debt_to_equity": "Debt_Equity",
    "pe": "PE", "p_e": "PE"
}

company_files = [
    {"company": "ITC", "sector": "FMCG", "path": r"C:\Users\EV-Car\Downloads\FMCG\ITC RATIOS.csv"},
    {"company": "HUL", "sector": "FMCG", "path": r"C:\Users\EV-Car\Downloads\FMCG\HULratios.csv"},
    {"company": "Nestle", "sector": "FMCG", "path": r"C:\Users\EV-Car\Downloads\FMCG\NESTLERATIOS.xlsx"},
    {"company": "Britannia", "sector": "FMCG", "path": r"C:\Users\EV-Car\Downloads\FMCG\BritaniaRatios.csv"},
    {"company": "Cipla", "sector": "Pharma", "path": r"C:\Users\EV-Car\Downloads\PHARMA\CiplaRatios.csv"},
    {"company": "SunPharma", "sector": "Pharma", "path": r"C:\Users\EV-Car\Downloads\PHARMA\SunpharmaRatios.csv"},
    {"company": "DrReddy", "sector": "Pharma", "path": r"C:\Users\EV-Car\Downloads\PHARMA\DrReddyRatios.csv"},
    {"company": "Apollo", "sector": "Pharma", "path": r"C:\Users\EV-Car\Downloads\PHARMA\ApolloRatios.csv"},
]

all_companies = []
for meta in company_files:
    path = meta["path"]
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path, encoding="latin1")
    else:
        df = pd.read_excel(path)

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
    df = df.rename(columns={k: v for k, v in CANONICAL_COLS.items() if k in df.columns})

    if "year" not in df.columns:
        raise ValueError(f"'Year' missing in {path}")

    df["Year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    keep_cols = ["Year", "ROE", "ROCE", "Current_Ratio", "Debt_Equity", "PE"]
    df = df[[c for c in keep_cols if c in df.columns]]
    df["Company"] = meta["company"]
    df["Sector"] = meta["sector"]
    df = FULL_YEARS.merge(df, on="Year", how="left")
    df["Company"] = meta["company"]
    df["Sector"] = meta["sector"]
    all_companies.append(df)

master_ratios = pd.concat(all_companies, ignore_index=True)

num_cols = ["ROE", "ROCE", "Current_Ratio", "Debt_Equity", "PE"]
master_ratios[num_cols] = master_ratios[num_cols].apply(pd.to_numeric, errors="coerce")

sector_ratios = (
    master_ratios
    .groupby(["Year", "Sector"])
    .agg(
        ROE_mean=("ROE", "mean"),
        ROCE_mean=("ROCE", "mean"),
        CurrentRatio_mean=("Current_Ratio", "mean"),
        DebtEquity_mean=("Debt_Equity", "mean"),
        PE_mean=("PE", "mean"),
        company_count=("Company", "nunique")
    )
    .reset_index()
)

ratio_cols = ["ROE_mean", "ROCE_mean", "CurrentRatio_mean", "DebtEquity_mean", "PE_mean"]
sector_ratios[ratio_cols] = sector_ratios[ratio_cols].apply(pd.to_numeric, errors="coerce")

sector_ratios_norm = sector_ratios.copy()
for sector in sector_ratios_norm["Sector"].unique():
    mask = sector_ratios_norm["Sector"] == sector
    scaler = StandardScaler()
    sector_ratios_norm.loc[mask, ratio_cols] = scaler.fit_transform(
        sector_ratios_norm.loc[mask, ratio_cols]
    )

sector_ratios_norm["Fundamental_Score"] = (
    sector_ratios_norm["ROE_mean"]
    + sector_ratios_norm["ROCE_mean"]
    + sector_ratios_norm["CurrentRatio_mean"]
    - sector_ratios_norm["DebtEquity_mean"]
    - sector_ratios_norm["PE_mean"]
)

phase1_fundamentals = sector_ratios_norm[
    ["Year", "Sector", "Fundamental_Score", "company_count"]
].sort_values(["Sector", "Year"]).reset_index(drop=True)

df_sent = pd.read_csv(r"C:\Users\EV-Car\Downloads\filtered_fmcg_pharma_news_with_sentiments.csv")
df_sent['date'] = pd.to_datetime(df_sent['published_date'], format='mixed', dayfirst=True, errors='coerce')
df_sent = df_sent.dropna(subset=['date'])

SECTOR_WEIGHTS = {'FMCG': 0.0644, 'Pharma': 0.0415}
df_sent['weighted_article_sentiment'] = (
    df_sent['sentiment_scores'] * df_sent['Sector'].map(SECTOR_WEIGHTS)
)

index_sentiment = (
    df_sent
    .groupby('date')
    .agg(
        index_sentiment=('weighted_article_sentiment', 'sum'),
        total_news=('weighted_article_sentiment', 'count')
    )
    .reset_index()
)

index_sentiment["date"] = pd.to_datetime(index_sentiment["date"], errors="coerce")
index_sentiment = index_sentiment.dropna(subset=["date"])
index_sentiment["FY"] = np.where(
    index_sentiment["date"].dt.month >= 4,
    index_sentiment["date"].dt.year,
    index_sentiment["date"].dt.year - 1
)

annual_sentiment = (
    index_sentiment
    .groupby("FY")
    .agg(
        Annual_Sentiment=("index_sentiment", "mean"),
        Avg_Daily_News=("total_news", "mean")
    )
    .reset_index()
    .rename(columns={"FY": "Year"})
)

scaler = StandardScaler()
annual_sentiment["Annual_Sentiment_Norm"] = scaler.fit_transform(
    annual_sentiment[["Annual_Sentiment"]]
)

phase2_sentiment = annual_sentiment[
    ["Year", "Annual_Sentiment_Norm", "Avg_Daily_News"]
].sort_values("Year").reset_index(drop=True)

fusion_df = phase1_fundamentals.merge(phase2_sentiment, on="Year", how="inner")
fusion_df = fusion_df.sort_values("Year")
fusion_df["Fundamental_Score_FF"] = fusion_df.groupby("Sector")["Fundamental_Score"].ffill()
fusion_df["Fusion_Score"] = fusion_df["Fundamental_Score_FF"] * fusion_df["Annual_Sentiment_Norm"]

scaler = StandardScaler()
fusion_df["Fusion_Score_Norm"] = scaler.fit_transform(fusion_df[["Fusion_Score"]])

phase3_fusion = fusion_df[
    ["Year", "Sector", "Fundamental_Score_FF", "Annual_Sentiment_Norm", "Fusion_Score_Norm"]
].sort_values(["Sector", "Year"]).reset_index(drop=True)

phase3_fusion.to_csv(r"C:\Users\EV-Car\Main-Project_3\phase3_fused_signal.csv", index=False)

df_nifty = pd.read_csv(r"C:\Users\EV-Car\Downloads\data.csv")
df_nifty["Date"] = pd.to_datetime(df_nifty["Date"])
df_nifty["Year"] = df_nifty["Date"].dt.year

annual_nifty = (
    df_nifty
    .groupby("Year")
    .agg(Close_Year_End=("Close", "last"), Close_Year_Start=("Close", "first"))
    .reset_index()
)
annual_nifty["annual_return"] = np.log(annual_nifty["Close_Year_End"]) - np.log(annual_nifty["Close_Year_Start"])

year_agg = (
    phase3_fusion
    .groupby("Year")
    .agg(
        Fundamental_Score_FF=("Fundamental_Score_FF", "mean"),
        Annual_Sentiment_Norm=("Annual_Sentiment_Norm", "mean"),
        Fusion_Score_Norm=("Fusion_Score_Norm", "mean")
    )
    .reset_index()
)

final_df = year_agg.merge(annual_nifty[["Year", "annual_return"]], on="Year", how="inner")
final_df = final_df.sort_values("Year").reset_index(drop=True)
final_df["market_up_next"] = (final_df["annual_return"].shift(-1) > 0).astype(int)
final_df = final_df.dropna(subset=["market_up_next"])
final_df["market_up_next"] = final_df["market_up_next"].astype(int)

final_df.to_csv(r"C:\Users\EV-Car\Main-Project_3\final_df.csv", index=False)

print(f"Shape: {final_df.shape}")
print(f"Class dist: {final_df['market_up_next'].value_counts().to_dict()}")
print(final_df[["Year", "Annual_Sentiment_Norm", "Fundamental_Score_FF", "market_up_next"]].to_string())
