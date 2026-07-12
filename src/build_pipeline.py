import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, accuracy_score,
                             confusion_matrix, silhouette_score, f1_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PROJECT PATHS  (all relative to repo root)
# ─────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA    = os.path.join(ROOT, "data")
MODELS  = os.path.join(ROOT, "models")
REPORTS = os.path.join(ROOT, "reports")

# Make sure output directories exist
os.makedirs(MODELS,  exist_ok=True)
os.makedirs(REPORTS, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD & ENRICH RAW DATA
# ─────────────────────────────────────────────
def load_and_enrich(path=None):
    if path is None:
        path = os.path.join(DATA, "cleaned_trading_data.csv")
    print("Loading data...")
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])

    # Trade-level binary flags for richer aggregation
    df['is_profitable_trade'] = (df['Closed PnL'] > 0).astype(int)
    df['is_long']   = df['Direction'].str.contains('Long|Buy', case=False, na=False).astype(int)
    df['is_closing'] = df['Direction'].str.contains('Close|Settlement|Liquidat', case=False, na=False).astype(int)
    df['net_pnl_after_fee'] = df['Closed PnL'] - df['Fee'].fillna(0)
    return df


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer_features(df):
    print("Engineering features...")

    # ── Daily aggregation per trader ──────────────────────────────────────────
    daily = df.groupby(['Account', 'date']).agg(
        daily_pnl          = ('Closed PnL', 'sum'),
        net_pnl_after_fee  = ('net_pnl_after_fee', 'sum'),
        trade_count        = ('Order ID', 'count'),
        avg_trade_size     = ('Size USD', 'mean'),
        total_volume       = ('Size USD', 'sum'),
        avg_leverage       = ('Proxy_Leverage', 'mean'),
        pnl_std            = ('Closed PnL', 'std'),        # intra-day PnL volatility
        win_rate           = ('is_profitable_trade', 'mean'),
        long_ratio         = ('is_long', 'mean'),          # directional bias
        close_ratio        = ('is_closing', 'mean'),       # % of closing trades
        total_fees         = ('Fee', 'sum'),
        fear_greed_value   = ('value', 'first'),
    ).reset_index()

    daily['pnl_std'] = daily['pnl_std'].fillna(0)

    # ── Sort for rolling / lag features ──────────────────────────────────────
    daily = daily.sort_values(['Account', 'date']).reset_index(drop=True)

    # ── Rolling behavioural momentum (3-day) ─────────────────────────────────
    for feat in ['daily_pnl', 'trade_count', 'win_rate', 'fear_greed_value']:
        daily[f'{feat}_roll3'] = (
            daily.groupby('Account')[feat]
                 .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        )

    # ── Lag-1 features (yesterday's values) ──────────────────────────────────
    for feat in ['daily_pnl', 'win_rate', 'pnl_std']:
        daily[f'{feat}_lag1'] = daily.groupby('Account')[feat].shift(1)

    # ── Target: next-day PnL bucket ──────────────────────────────────────────
    daily['next_day_pnl'] = daily.groupby('Account')['daily_pnl'].shift(-1)

    # Use quantile-based thresholds (more robust than hard $50 cutoff)
    pos_mask = daily['next_day_pnl'] > 0
    q50_pos  = daily.loc[pos_mask, 'next_day_pnl'].quantile(0.50)

    def categorize_pnl(pnl):
        if pnl < 0:
            return 'Loss'
        elif pnl < q50_pos:
            return 'Low Profit'
        else:
            return 'High Profit'

    model_df = daily.dropna(subset=['next_day_pnl']).copy()
    model_df['profitability_bucket'] = model_df['next_day_pnl'].apply(categorize_pnl)
    return model_df, q50_pos


# ─────────────────────────────────────────────
# 3. PREDICTIVE MODELLING
# ─────────────────────────────────────────────
def train_model(model_df):
    print("Training Predictive Model...")

    FEATURES = [
        # Core behaviour
        'trade_count', 'avg_trade_size', 'total_volume',
        'avg_leverage', 'win_rate', 'long_ratio', 'close_ratio',
        'pnl_std', 'total_fees',
        # Sentiment
        'fear_greed_value',
        # Rolling momentum
        'daily_pnl_roll3', 'trade_count_roll3', 'win_rate_roll3', 'fear_greed_value_roll3',
        # Yesterday's signal
        'daily_pnl_lag1', 'win_rate_lag1', 'pnl_std_lag1',
    ]

    X = model_df[FEATURES].fillna(0)
    y = model_df['profitability_bucket']

    # ── Chronological split (no look-ahead leakage) ───────────────────────────
    # Sort by date then split 80/20
    model_df_sorted = model_df.sort_values('date').reset_index(drop=True)
    X_sorted = model_df_sorted[FEATURES].fillna(0)
    y_sorted = model_df_sorted['profitability_bucket']
    split    = int(len(X_sorted) * 0.80)
    X_train, X_test = X_sorted.iloc[:split], X_sorted.iloc[split:]
    y_train, y_test = y_sorted.iloc[:split], y_sorted.iloc[split:]

    # ── Gradient Boosting (better generalisation than vanilla RF) ─────────────
    model = GradientBoostingClassifier(
        n_estimators   = 300,
        learning_rate  = 0.05,
        max_depth      = 4,
        subsample      = 0.8,
        min_samples_leaf = 10,
        random_state   = 42
    )
    model.fit(X_train, y_train)

    preds    = model.predict(X_test)
    acc      = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average='macro')

    # ── 5-fold stratified CV for honest estimate ──────────────────────────────
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

    print(f"\n{'='*55}")
    print(f"  Hold-out Accuracy  : {acc:.4f}  ({acc:.1%})")
    print(f"  Macro F1-Score     : {f1_macro:.4f}")
    print(f"  5-Fold CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"{'='*55}")
    print(classification_report(y_test, preds))

    return model, preds, y_test, FEATURES, acc, f1_macro, cv_scores


# ─────────────────────────────────────────────
# 4. CLUSTERING — TRADER ARCHETYPES
# ─────────────────────────────────────────────
def cluster_traders(df):
    print("Clustering Traders...")
    trader_profile = df.groupby('Account').agg(
        total_pnl      = ('Closed PnL', 'sum'),
        total_trades   = ('Order ID', 'count'),
        avg_trade_size = ('Size USD', 'mean'),
        avg_leverage   = ('Proxy_Leverage', 'mean'),
        avg_margin     = ('Proxy_Margin', 'mean'),
        overall_win_rate = ('is_profitable_trade', 'mean'),
        long_bias      = ('is_long', 'mean'),
    ).reset_index().fillna(0)

    cluster_features = ['total_pnl', 'total_trades', 'avg_trade_size', 'avg_leverage']
    scaler   = StandardScaler()
    X_clust  = scaler.fit_transform(trader_profile[cluster_features])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_clust)
    trader_profile['cluster'] = labels

    sil_score = silhouette_score(X_clust, labels)
    print(f"Silhouette Score: {sil_score:.3f}")

    centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=cluster_features
    )
    centers['cluster_id'] = range(len(centers))

    # Data-driven labelling
    archetype_map, assigned = {}, set()
    whale_c = centers.loc[centers['total_pnl'].idxmax(),    'cluster_id']
    archetype_map[whale_c] = 'Whale';             assigned.add(whale_c)
    rem = centers[~centers['cluster_id'].isin(assigned)]
    degen_c = rem.loc[rem['avg_leverage'].idxmax(),          'cluster_id']
    archetype_map[degen_c] = 'Degen/High-Leverage';  assigned.add(degen_c)
    rem = centers[~centers['cluster_id'].isin(assigned)]
    hf_c = rem.loc[rem['total_trades'].idxmax(),             'cluster_id']
    archetype_map[hf_c] = 'High-Frequency Scalper'; assigned.add(hf_c)
    for cid in centers['cluster_id']:
        if cid not in assigned:
            archetype_map[cid] = 'Conservative'

    trader_profile['Archetype'] = trader_profile['cluster'].map(archetype_map)
    return trader_profile, sil_score


# ─────────────────────────────────────────────
# 5. SAVE ARTEFACTS
# ─────────────────────────────────────────────
def save_artifacts(model, model_df, trader_profile,
                   features, preds, y_test,
                   acc, f1_macro, cv_scores, sil_score):

    joblib.dump(model, os.path.join(MODELS,  "predictive_model.pkl"))
    trader_profile.to_csv(os.path.join(REPORTS, "trader_profiles.csv"),  index=False)
    model_df.sample(min(10000, len(model_df)), random_state=42).to_csv(
        os.path.join(REPORTS, "dashboard_data.csv"), index=False)

    # Feature importances
    importances = pd.DataFrame({
        'feature'   : features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    importances.to_csv(os.path.join(REPORTS, "feature_importances.csv"), index=False)

    # Confusion matrix
    cm = confusion_matrix(y_test, preds, labels=model.classes_)
    pd.DataFrame(cm, index=model.classes_, columns=model.classes_).to_csv(
        os.path.join(REPORTS, "confusion_matrix.csv"))

    # Model metrics for dashboard
    metrics = pd.DataFrame({
        'metric': ['Holdout Accuracy', 'Macro F1-Score',
                   'CV Accuracy (mean)', 'CV Accuracy (std)', 'Silhouette Score'],
        'value' : [round(acc, 4), round(f1_macro, 4),
                   round(cv_scores.mean(), 4), round(cv_scores.std(), 4),
                   round(sil_score, 4)]
    })
    metrics.to_csv(os.path.join(REPORTS, "model_metrics.csv"), index=False)
    print(f"\nAll artifacts saved successfully.")
    print(f"  Model   → {os.path.join(MODELS,  'predictive_model.pkl')}")
    print(f"  Reports → {REPORTS}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    df = load_and_enrich()
    model_df, q50_pos = engineer_features(df)

    print(f"\nTarget distribution:\n{model_df['profitability_bucket'].value_counts()}\n")
    print(f"Profit split threshold (median positive PnL): ${q50_pos:.2f}\n")

    model, preds, y_test, features, acc, f1_macro, cv_scores = train_model(model_df)
    trader_profile, sil_score = cluster_traders(df)
    save_artifacts(model, model_df, trader_profile,
                   features, preds, y_test,
                   acc, f1_macro, cv_scores, sil_score)

if __name__ == "__main__":
    main()
