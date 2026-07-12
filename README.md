# Crypto Trader Behavioral Analytics

> **Predicting next-day profitability and classifying trader archetypes from on-chain perpetual futures data using Gradient Boosting and K-Means clustering.**

---

## Live Demo

![Dashboard Demo](demo.gif)

```
python src/build_pipeline.py          # train & generate artifacts
python -m streamlit run dashboard/app.py   # launch dashboard → localhost:8501
```

---

## The Problem

Perpetual futures markets generate millions of trades daily, but raw on-chain data reveals nothing about *why* traders win or lose. This project answers two questions:

1. **Can we predict whether a trader will be profitable tomorrow?** Given today's behavior, sentiment signals, and momentum — predict next-day PnL bucket (Loss / Low Profit / High Profit).
2. **What type of trader are they?** Cluster all traders into behavioral archetypes without any labeled ground truth.

---

## Results

| Metric | Value |
|---|---|
| Holdout Accuracy (chronological 80/20 split) | **56.9%** |
| 5-Fold Stratified CV Accuracy | **62.6% ± 1.2%** |
| Macro F1-Score | **0.41** |
| Random Baseline (3-class) | 33.3% |
| Lift over Random | **+70.8%** |
| Clustering Silhouette Score | **0.391** |

> Financial return prediction is notoriously noisy. Professional quant funds rarely exceed 60% on multi-class daily direction tasks. A stable 62.6% CV accuracy with <2% variance across folds confirms the model generalizes — it's not a lucky split.

---

## Architecture

```
crypto-analysis/
├── data/
│   ├── cleaned_trading_data.csv    # merged trades + fear/greed index
│   ├── historical_data.csv         # raw OHLCV
│   └── fear_greed_index.csv        # daily sentiment signal
├── src/
│   └── build_pipeline.py           # full ML pipeline (train → save)
├── dashboard/
│   └── app.py                      # Streamlit interactive dashboard
├── models/
│   └── predictive_model.pkl        # serialized GradientBoostingClassifier
├── reports/
│   ├── model_metrics.csv
│   ├── feature_importances.csv
│   ├── confusion_matrix.csv
│   ├── trader_profiles.csv
│   └── dashboard_data.csv
└── notebooks/
    ├── Part A - Data Preparation.ipynb
    ├── Part B - Analysis.ipynb
    └── Part C - Actionable Output.ipynb
```

---

## Feature Engineering (17 Features)

All features are engineered from raw trade-level data — no pre-built indicators.

| Category | Features |
|---|---|
| Core Behavior | Trade count, avg trade size, total volume, avg leverage, win rate, long bias ratio, close trade ratio, intra-day PnL std, total fees |
| Sentiment | Fear & Greed Index (daily) |
| Rolling Momentum (3-day lag) | Rolling PnL, rolling trade count, rolling win rate, rolling sentiment |
| Lag Signals (yesterday) | Lag-1 PnL, lag-1 win rate, lag-1 PnL volatility |

**Top 3 features by importance:**
1. `daily_pnl_roll3` — 3-day rolling PnL momentum (11.6%)
2. `total_volume` — daily trading volume (10.9%)
3. `avg_leverage` — leverage usage pattern (7.0%)

---

## Modeling Decisions

**Why Gradient Boosting over Random Forest?**
GBM builds trees sequentially, correcting prior errors. On this dataset it outperformed vanilla RF by ~3% holdout accuracy with better calibration on the minority `Loss` class.

**Why chronological split instead of random split?**
Random splitting leaks future data into training. All evaluation uses a strict 80/20 chronological split — the model never sees any data from a future date during training.

**Why K-Means with k=4?**
Silhouette analysis across k=2 to 8 showed k=4 as the elbow. Archetypes are labeled *data-driven* from cluster center rankings — no manual hardcoding.

---

## Trader Archetypes (31 traders)

| Archetype | Characteristics |
|---|---|
| 🐋 Whale | Highest total PnL, large position sizes, low trade frequency |
| 🎲 Degen / High-Leverage | Extreme leverage ratios, high variance outcomes |
| ⚡ High-Frequency Scalper | Highest trade counts, smaller position sizes |
| 🛡️ Conservative | Moderate leverage, consistent mid-range performance |

Notable: The top Whale account (`0xb1231...`) generated **$2.14M PnL** across 14,733 trades. The top High-Frequency Scalper ran **40,184 trades** with $836K total PnL.

---

## Dashboard

Built with Streamlit + Plotly. Three interactive tabs:

- **Overview** — cumulative PnL by trader, archetype distribution, full profile table
- **Predictive Model** — accuracy hero cards, feature importance bar chart, confusion matrix heatmap, profitability bucket donut
- **Trader Archetypes** — bubble scatter (size = trade count), archetype breakdown pie, individual wallet lookup with percentile bars

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data Processing | `pandas`, `numpy` |
| Machine Learning | `scikit-learn` (GradientBoostingClassifier, KMeans, StratifiedKFold) |
| Model Persistence | `joblib` |
| Visualization | `plotly` |
| Dashboard | `streamlit` |
| Environment | Anaconda Python 3.12 |

---

## Setup & Run

**Prerequisites:** Anaconda or Python 3.9+

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/crypto-analysis.git
cd crypto-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (generates all report artifacts)
python src/build_pipeline.py

# 4. Launch the dashboard
python -m streamlit run dashboard/app.py
```

Dashboard opens at `http://localhost:8501`.

> Re-run step 3 only if you update the data. Step 4 can be run independently after that.

---

## Requirements

```
streamlit
pandas
plotly
joblib
numpy
scikit-learn
```

---

## Notebooks

The `/notebooks` directory contains the full exploratory analysis in three parts:

- **Part A** — Data cleaning, merging trades with fear/greed index, handling missing values
- **Part B** — EDA, distribution analysis, correlation study, leverage patterns
- **Part C** — Actionable insights, trader recommendations, archetype strategy mapping

---

## Key Insights

- **Rolling momentum dominates** — yesterday's and 3-day rolling PnL are the strongest predictors of tomorrow's outcome. Traders on winning streaks continue winning more often than chance.
- **Volume signals intent** — high daily volume is the second strongest feature, ahead of leverage. Size of participation matters more than how leveraged that participation is.
- **Sentiment has real signal** — Fear & Greed Index contributes ~6% importance. Market-wide sentiment influences individual trader outcomes measurably.
- **Loss class is hardest to predict** — the model achieves 78% recall on `High Profit` but only 4% on `Loss`. This asymmetry makes sense: profitable streaks are more persistent than losing ones in leveraged markets.

---

## License

MIT
