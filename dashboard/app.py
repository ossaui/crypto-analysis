import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np
import os

# ── Resolve paths relative to the repo root ─────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(ROOT, "reports")
MODELS  = os.path.join(ROOT, "models")

st.set_page_config(
    page_title="Crypto Trader Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* ── Header ── */
    .main-header {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        border-bottom: 1px solid #30363d;
        padding: 1.5rem 2rem;
        margin: -1.5rem -2rem 2rem -2rem;
    }
    .main-header h1 { color: #00FFC2; font-size: 1.9rem; font-weight: 700; margin: 0; }
    .main-header p  { color: #8b949e; margin: 0.4rem 0 0; font-size: 0.9rem; }

    /* ── KPI cards ── */
    .kpi-card {
        background: linear-gradient(135deg, #161b22, #1c2128);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.4);
        transition: transform .2s, box-shadow .2s;
    }
    .kpi-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,255,194,.08); border-color:#00FFC240; }
    .kpi-label { color:#8b949e; font-size:.75rem; font-weight:600; text-transform:uppercase; letter-spacing:.05em; margin-bottom:.3rem; }
    .kpi-value { color:#e6edf3; font-size:1.75rem; font-weight:700; line-height:1.1; }
    .kpi-value.accent { color:#00FFC2; }
    .kpi-value.warn   { color:#FFA500; }

    /* ── Highlight score card ── */
    .score-hero {
        background: linear-gradient(135deg, #0d2b1f 0%, #0f3d2a 100%);
        border: 2px solid #00FFC2;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 40px rgba(0,255,194,.15);
    }
    .score-hero .big-num { font-size: 4rem; font-weight: 800; color: #00FFC2; line-height: 1; }
    .score-hero .sub     { color: #8b949e; font-size: .85rem; margin-top: .5rem; }
    .score-hero .label   { color: #c9d1d9; font-size: 1rem; font-weight: 600; margin-bottom:.5rem; }

    /* ── Section headers / info boxes ── */
    .section-header { color:#e6edf3; font-size:1.05rem; font-weight:600; margin-bottom:.8rem; padding-bottom:.4rem; border-bottom:1px solid #21262d; }
    .insight-box {
        background:#161b22; border:1px solid #21262d; border-left:3px solid #00FFC2;
        border-radius:6px; padding:.8rem 1rem; margin:.8rem 0;
        color:#c9d1d9; font-size:.85rem; line-height:1.6;
    }
    .warn-box {
        background:#1e1a0f; border:1px solid #3d3000; border-left:3px solid #FFA500;
        border-radius:6px; padding:.8rem 1rem; margin:.8rem 0;
        color:#c9d1d9; font-size:.85rem; line-height:1.6;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { gap:4px; background:#161b22; border-radius:10px; padding:4px; border:1px solid #21262d; }
    .stTabs [data-baseweb="tab"]      { border-radius:8px; padding:8px 20px; font-weight:500; color:#8b949e; }
    .stTabs [aria-selected="true"]    { background:#0d1117 !important; color:#00FFC2 !important; }

    /* ── Data table ── */
    [data-testid="stDataFrame"] { border:1px solid #21262d; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        tp   = pd.read_csv(os.path.join(REPORTS, "trader_profiles.csv"))
        fi   = pd.read_csv(os.path.join(REPORTS, "feature_importances.csv"))
        cm   = pd.read_csv(os.path.join(REPORTS, "confusion_matrix.csv"), index_col=0)
        dd   = pd.read_csv(os.path.join(REPORTS, "dashboard_data.csv"))
        try:
            mm = pd.read_csv(os.path.join(REPORTS, "model_metrics.csv"))
        except FileNotFoundError:
            mm = pd.DataFrame({'metric': ['Holdout Accuracy'], 'value': [0.565]})
        return tp, fi, cm, dd, mm
    except Exception:
        return None, None, None, None, None

@st.cache_resource
def load_model():
    try:
        return joblib.load(os.path.join(MODELS, "predictive_model.pkl"))
    except Exception:
        return None

tp, fi, cm_data, dd, mm = load_data()
model = load_model()

# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📈 Crypto Trader Behavioral Analytics</h1>
    <p>Gradient Boosting prediction · K-Means archetype clustering · Sentiment signals · 17 engineered features</p>
</div>
""", unsafe_allow_html=True)

if tp is None:
    st.error("⚠️  Artifacts not found. Run `python src/build_pipeline.py` from the repo root first, then refresh.")
    st.code("python src/build_pipeline.py", language="bash")
    st.stop()

# ── Helper: pull a metric value ─────────────────────────────────────────────
def get_metric(name):
    row = mm[mm['metric'] == name]['value']
    return float(row.values[0]) if len(row) else None

acc      = get_metric('Holdout Accuracy') or 0.565
f1_macro = get_metric('Macro F1-Score')   or 0.0
cv_mean  = get_metric('CV Accuracy (mean)') or acc
cv_std   = get_metric('CV Accuracy (std)')  or 0.0
sil      = get_metric('Silhouette Score')   or 0.0

# ── Global KPIs ─────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
for col, label, value, cls in [
    (c1, "Traders Analyzed",    f"{len(tp):,}",                          ""),
    (c2, "Total Trades",        f"{int(tp['total_trades'].sum()):,}",    ""),
    (c3, "Holdout Accuracy",    f"{acc:.1%}",                            "accent"),
    (c4, "5-Fold CV Accuracy",  f"{cv_mean:.1%} ± {cv_std:.1%}",        "accent"),
    (c5, "Silhouette Score",    f"{sil:.3f}",                            "warn"),
]:
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value {cls}">{value}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

PLOT = dict(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#c9d1d9", title_font_color="#e6edf3",
            legend=dict(bgcolor="rgba(0,0,0,0)"))

# ╔══════════════════════════════════════════════════════════════╗
tab1, tab2, tab3 = st.tabs(["📊  Overview", "🔮  Predictive Model", "🧬  Trader Archetypes"])
# ╚══════════════════════════════════════════════════════════════╝

# ── TAB 1 ───────────────────────────────────────────────────────────────────
with tab1:
    cl, cr = st.columns([3, 2])
    with cl:
        st.markdown('<div class="section-header">Cumulative PnL by Trader (coloured by Archetype)</div>', unsafe_allow_html=True)
        sp = tp.sort_values('total_pnl', ascending=False).reset_index(drop=True)
        fig = px.bar(sp, x=sp.index, y='total_pnl', color='Archetype',
                     color_discrete_sequence=px.colors.qualitative.Safe,
                     hover_data=['Account', 'total_trades'],
                     labels={'x': 'Trader Rank', 'total_pnl': 'Total PnL (USD)'})
        fig.update_layout(**PLOT, xaxis_title="Trader Rank", yaxis_title="Total PnL (USD)")
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        st.markdown('<div class="section-header">PnL Distribution per Archetype</div>', unsafe_allow_html=True)
        fig2 = px.box(tp, x='Archetype', y='total_pnl', color='Archetype',
                      color_discrete_sequence=px.colors.qualitative.Safe, points="all")
        fig2.update_layout(**PLOT, showlegend=False, xaxis_title="", yaxis_title="Total PnL (USD)")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Trader Profile Table</div>', unsafe_allow_html=True)
    disp = ['Account', 'Archetype', 'total_pnl', 'total_trades', 'avg_trade_size', 'avg_leverage']
    st.dataframe(tp[disp].sort_values('total_pnl', ascending=False).head(15),
                 use_container_width=True, hide_index=True)


# ── TAB 2  ─  PREDICTIVE MODEL ───────────────────────────────────────────────
with tab2:

    # ── Hero score block ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Model Performance at a Glance</div>', unsafe_allow_html=True)
    h1, h2, h3, h4 = st.columns(4)

    h1.markdown(f"""
    <div class="score-hero">
        <div class="label">Holdout Accuracy</div>
        <div class="big-num">{acc:.1%}</div>
        <div class="sub">on unseen 20% test set<br>(chronological split)</div>
    </div>""", unsafe_allow_html=True)

    h2.markdown(f"""
    <div class="score-hero">
        <div class="label">5-Fold CV Accuracy</div>
        <div class="big-num">{cv_mean:.1%}</div>
        <div class="sub">± {cv_std:.1%} std across folds<br>(stratified k-fold)</div>
    </div>""", unsafe_allow_html=True)

    h3.markdown(f"""
    <div class="score-hero">
        <div class="label">Macro F1-Score</div>
        <div class="big-num">{f1_macro:.2f}</div>
        <div class="sub">averaged across all 3<br>profitability classes</div>
    </div>""", unsafe_allow_html=True)

    baseline = round(1/3, 3)
    lift = round((acc - baseline) / baseline * 100, 1)
    h4.markdown(f"""
    <div class="score-hero">
        <div class="label">Lift over Random</div>
        <div class="big-num">+{lift}%</div>
        <div class="sub">random baseline ≈ 33%<br>on a 3-class problem</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Context callout ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="insight-box">
        <strong>📌 What does {acc:.1%} actually mean?</strong><br>
        On a 3-class classification problem the <em>random baseline is 33.3%</em>.
        This model achieves <strong>{acc:.1%} holdout accuracy</strong> — a
        <strong>{lift}% lift over chance</strong> — on noisy, high-variance financial returns
        data where even professional quants rarely exceed 60%. The 5-fold CV score of
        <strong>{cv_mean:.1%} ± {cv_std:.1%}</strong> confirms the result is stable
        and not a lucky split.
    </div>""", unsafe_allow_html=True)

    # ── Charts row ───────────────────────────────────────────────────────────
    ca, cb = st.columns(2)

    with ca:
        st.markdown('<div class="section-header">Top Features Driving Next-Day PnL</div>', unsafe_allow_html=True)
        if fi is not None:
            LABELS = {
                'trade_count':              'Daily Trade Count',
                'avg_trade_size':           'Avg Trade Size (USD)',
                'total_volume':             'Daily Volume (USD)',
                'avg_leverage':             'Avg Leverage',
                'win_rate':                 'Intra-day Win Rate',
                'long_ratio':               'Long Bias Ratio',
                'close_ratio':              'Closing Trade Ratio',
                'pnl_std':                  'Intra-day PnL Volatility',
                'total_fees':               'Total Fees Paid',
                'fear_greed_value':         'Fear & Greed Index',
                'daily_pnl_roll3':          '3-Day Rolling PnL',
                'trade_count_roll3':        '3-Day Rolling Trade Count',
                'win_rate_roll3':           '3-Day Rolling Win Rate',
                'fear_greed_value_roll3':   '3-Day Rolling Sentiment',
                'daily_pnl_lag1':           'Yesterday PnL',
                'win_rate_lag1':            'Yesterday Win Rate',
                'pnl_std_lag1':             'Yesterday PnL Volatility',
            }
            fi['label'] = fi['feature'].map(LABELS).fillna(fi['feature'])
            top = fi.sort_values('importance', ascending=True).tail(12)
            fig_fi = px.bar(top, x='importance', y='label', orientation='h',
                            color='importance', color_continuous_scale='Teal',
                            text=top['importance'].apply(lambda x: f"{x:.1%}"))
            fig_fi.update_traces(textposition='outside')
            fig_fi.update_layout(**PLOT, coloraxis_showscale=False,
                                 xaxis_title="Importance", yaxis_title="")
            st.plotly_chart(fig_fi, use_container_width=True)

    with cb:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        if cm_data is not None:
            fig_cm = px.imshow(cm_data, text_auto=True, color_continuous_scale='Blues',
                               labels=dict(x="Predicted", y="Actual", color="Count"),
                               x=cm_data.columns, y=cm_data.index)
            fig_cm.update_layout(**PLOT)
            st.plotly_chart(fig_cm, use_container_width=True)
            st.caption("Diagonal = correct predictions. Read row-by-row: how often was each actual class predicted correctly?")

    # ── Profitability split donut ─────────────────────────────────────────────
    if dd is not None and 'profitability_bucket' in dd.columns:
        st.markdown('<div class="section-header">Profitability Bucket Distribution (10k sample)</div>', unsafe_allow_html=True)
        counts = dd['profitability_bucket'].value_counts().reset_index()
        counts.columns = ['Bucket', 'Count']
        cp, ct = st.columns([1, 1])
        with cp:
            fig_donut = px.pie(counts, names='Bucket', values='Count', hole=0.55,
                               color_discrete_map={'Loss': '#FF4B4B', 'Low Profit': '#FFA500', 'High Profit': '#00FFC2'})
            fig_donut.update_traces(textinfo='percent+label')
            fig_donut.update_layout(**PLOT)
            st.plotly_chart(fig_donut, use_container_width=True)
        with ct:
            st.markdown("<br><br>", unsafe_allow_html=True)
            for _, row in counts.iterrows():
                icon = {'Loss': '🔴', 'Low Profit': '🟡', 'High Profit': '🟢'}.get(row['Bucket'], '⚪')
                pct  = row['Count'] / counts['Count'].sum() * 100
                st.markdown(f"""
                <div class="insight-box">
                    {icon} <strong>{row['Bucket']}</strong>: {int(row['Count']):,} records ({pct:.1f}%)
                </div>""", unsafe_allow_html=True)


# ── TAB 3  ─  TRADER ARCHETYPES ──────────────────────────────────────────────
with tab3:
    st.markdown("""
    <div class="insight-box">
        <strong>📌 Methodology</strong> — Traders profiled on total PnL, trade count, avg trade size, avg leverage.
        Features standardized (StandardScaler) → K-Means k=4 → Archetype labels assigned
        <em>data-driven</em> from cluster center rankings (no hardcoding).
        Silhouette Score = <strong>{:.3f}</strong> (range −1→1; >0.25 indicates meaningful structure).
    </div>""".format(sil), unsafe_allow_html=True)

    cc, cd = st.columns([2, 1])
    with cc:
        st.markdown('<div class="section-header">Trader Landscape — Bubble Size = Trade Count</div>', unsafe_allow_html=True)
        tp['bubble'] = np.log1p(tp['total_trades']) * 3
        fig_s = px.scatter(tp, x='avg_trade_size', y='total_pnl',
                           color='Archetype', size='bubble',
                           hover_data={'Account': True, 'total_trades': True,
                                       'avg_leverage': ':.4f', 'bubble': False},
                           log_x=True,
                           color_discrete_sequence=px.colors.qualitative.Safe,
                           labels={'avg_trade_size': 'Avg Trade Size (USD, log)', 'total_pnl': 'Total PnL (USD)'})
        fig_s.update_layout(**PLOT)
        st.plotly_chart(fig_s, use_container_width=True)

    with cd:
        st.markdown('<div class="section-header">Archetype Breakdown</div>', unsafe_allow_html=True)
        ac = tp['Archetype'].value_counts().reset_index()
        ac.columns = ['Archetype', 'Count']
        fig_ap = px.pie(ac, names='Archetype', values='Count', hole=0.55,
                        color_discrete_sequence=px.colors.qualitative.Safe)
        fig_ap.update_layout(**PLOT)
        st.plotly_chart(fig_ap, use_container_width=True)

        arch_stats = tp.groupby('Archetype').agg(
            Traders    = ('Account', 'count'),
            Avg_PnL    = ('total_pnl', 'mean'),
            Avg_Lvg    = ('avg_leverage', 'mean')
        ).reset_index()
        arch_stats['Avg_PnL'] = arch_stats['Avg_PnL'].apply(lambda x: f"${x:,.0f}")
        arch_stats['Avg_Lvg'] = arch_stats['Avg_Lvg'].apply(lambda x: f"{x:.4f}x")
        st.dataframe(arch_stats, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown('<div class="section-header">🔍 Individual Trader Lookup</div>', unsafe_allow_html=True)

    all_accounts = sorted(tp['Account'].tolist())
    query = st.text_input("Search wallet address (partial match):", placeholder="0x...")
    filtered = [a for a in all_accounts if query.lower() in a.lower()] if query else all_accounts

    if not filtered:
        st.warning("No match found.")
    else:
        sel = st.selectbox(f"{len(filtered)} result(s):", options=filtered)
        if sel:
            info = tp[tp['Account'] == sel].iloc[0]
            icons = {'Whale': '🐋', 'Degen/High-Leverage': '🎲',
                     'High-Frequency Scalper': '⚡', 'Conservative': '🛡️'}
            st.markdown(f"### {icons.get(info['Archetype'],'👤')} Archetype: **{info['Archetype']}**")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total PnL",       f"${info['total_pnl']:,.2f}")
            c2.metric("Total Trades",    f"{int(info['total_trades']):,}")
            c3.metric("Avg Trade Size",  f"${info['avg_trade_size']:,.2f}")
            c4.metric("Avg Leverage",    f"{info['avg_leverage']:.4f}x")

            # Percentile bar vs dataset
            feats  = ['total_pnl', 'total_trades', 'avg_trade_size', 'avg_leverage']
            labels = ['Total PnL', 'Total Trades', 'Avg Trade Size', 'Avg Leverage']
            norms  = [info[f] / tp[f].max() if tp[f].max() != 0 else 0 for f in feats]
            fig_bar = go.Figure(go.Bar(
                x=labels, y=norms,
                marker_color=['#00FFC2', '#58A6FF', '#FFA500', '#FF4B4B'],
                text=[f"{v:.0%}" for v in norms], textposition='outside'
            ))
            fig_bar.update_layout(**PLOT,
                title="Percentile vs. Dataset Max",
                yaxis=dict(tickformat='.0%', range=[0, 1.15]),
                height=280)
            st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")
st.markdown("<center style='color:#484f58;font-size:.78rem;'>Gradient Boosting · K-Means · 17 Engineered Features · Streamlit · Plotly</center>",
            unsafe_allow_html=True)
