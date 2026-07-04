import pandas as pd
import numpy as np
import pickle
import os


def create_integrated_panel():
    final_df = pd.read_csv(r"C:\Users\EV-Car\Main-Project_3\final_df.csv")
    phase3 = pd.read_csv(r"C:\Users\EV-Car\Main-Project_3\phase3_fused_signal.csv")

    gp_panel_path = r"C:\Users\EV-Car\Main_Project_2\data\processed\panel.pkl"
    with open(gp_panel_path, 'rb') as f:
        panel = pickle.load(f)

    dates = list(panel.values())[0].index
    stocks = list(panel.values())[0].columns
    year_map = dates.year

    afm_sentiment_df = pd.DataFrame(np.nan, index=dates, columns=stocks)
    afm_fundamental_df = pd.DataFrame(np.nan, index=dates, columns=stocks)
    afm_regime_df = pd.DataFrame(np.nan, index=dates, columns=stocks)

    mean_fusion = final_df["Fusion_Score_Norm"].mean()

    for _, row in final_df.iterrows():
        yr = int(row['Year'])
        mask = (year_map == yr)
        afm_sentiment_df.loc[mask, :] = row["Annual_Sentiment_Norm"]
        afm_fundamental_df.loc[mask, :] = row["Fundamental_Score_FF"]
        afm_regime_df.loc[mask, :] = 1 if row["Fusion_Score_Norm"] > mean_fusion else 0

    afm_sentiment_df = afm_sentiment_df.ffill().bfill().astype(float)
    afm_fundamental_df = afm_fundamental_df.ffill().bfill().astype(float)
    afm_regime_df = afm_regime_df.ffill().bfill().astype(float)

    panel['afm_sentiment'] = afm_sentiment_df
    panel['afm_fundamental'] = afm_fundamental_df
    panel['afm_regime'] = afm_regime_df

    out_path = os.path.join(r"C:\Users\EV-Car\Main-Project_3\gp", "integrated_panel.pkl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(panel, f)

    print(f"Panel features: {list(panel.keys())}")
    print(f"Dates: {dates[0].date()} to {dates[-1].date()} ({len(dates)} days)")
    print(f"Stocks: {len(stocks)}")
    print(f"AFM sentiment sample (2020): {afm_sentiment_df.loc[afm_sentiment_df.index.year == 2020].iloc[0, 0]:.4f}")
    print(f"AFM regime distribution: Bull={int(afm_regime_df.iloc[:, 0].sum())}, Bear={int(len(dates) - afm_regime_df.iloc[:, 0].sum())}")
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    create_integrated_panel()
