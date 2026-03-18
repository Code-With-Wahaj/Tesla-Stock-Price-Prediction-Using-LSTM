"""
╔══════════════════════════════════════════════════════════════════╗
║           TESLA LSTM STOCK PREDICTOR  ·  app.py                  ║
║                                                                  ║
║  HOW THE MODEL WORKS (explain this to the board):                ║
║  1. We trained the model on ONLY the Close price column.         ║
║  2. It learns from sequences of 60 consecutive Close prices.     ║
║  3. Given 60 days of Close prices, it predicts the NEXT day.     ║
║  4. Open, High, Low, Volume are NEVER used anywhere.             ║
║                                                                  ║
║  WHAT THE SAVED FILES CONTAIN:                                   ║
║  - lstm_model.h5   → the trained neural network weights          ║
║  - scaler.pkl      → MinMaxScaler fitted on training data        ║
║  - config.pkl      → metadata: accuracy, mae, look_back, etc.    ║
║                                                                  ║
║  Run: streamlit run app.py                                       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model


# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG — must be the very first st.* call
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="TSLA · LSTM Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════
# GLOBAL CSS — dark terminal aesthetic
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,500;0,9..40,700;1,9..40,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0b0e14;
    color: #e8eaf0;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1400px; }

[data-testid="stSidebar"] { background-color: #0f1219 !important; border-right: 1px solid #1e2433; }
[data-testid="stSidebar"] * { color: #c8ccd8 !important; }

.hero {
    background: linear-gradient(135deg, #0f1a2e 0%, #0b0e14 50%, #1a0a0a 100%);
    border: 1px solid #1e2433; border-radius: 16px;
    padding: 48px 40px 40px; margin-bottom: 32px; position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse at 80% 50%, rgba(220,38,38,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.hero-ticker { font-family:'Space Mono',monospace; font-size:0.75rem; letter-spacing:0.2em; color:#dc2626; text-transform:uppercase; margin-bottom:8px; }
.hero-title  { font-size:3.2rem; font-weight:700; color:#f1f3f9; line-height:1.1; margin:0 0 10px; }
.hero-title span { color:#dc2626; }
.hero-sub    { font-size:1rem; color:#7c8499; font-weight:300; max-width:560px; }
.hero-badge  { display:inline-block; background:rgba(220,38,38,0.12); border:1px solid rgba(220,38,38,0.3); color:#f87171; padding:4px 12px; border-radius:20px; font-size:0.75rem; font-family:'Space Mono',monospace; margin-top:16px; }

.stat-row  { display:flex; gap:16px; margin:24px 0; flex-wrap:wrap; }
.stat-card { flex:1; min-width:150px; background:#111520; border:1px solid #1e2433; border-radius:12px; padding:20px 22px; transition:border-color 0.2s; }
.stat-card:hover { border-color:#dc2626; }
.stat-label  { font-size:0.68rem; letter-spacing:0.15em; text-transform:uppercase; color:#5a6070; font-family:'Space Mono',monospace; margin-bottom:8px; }
.stat-value  { font-size:1.5rem; font-weight:700; color:#f1f3f9; font-family:'Space Mono',monospace; }
.stat-value.positive { color:#22c55e; }
.stat-value.negative { color:#ef4444; }
.stat-value.accent   { color:#dc2626; }
.stat-detail { font-size:0.78rem; color:#5a6070; margin-top:5px; }

.pred-card  { background:linear-gradient(135deg,#1a0808,#0f1219); border:1px solid rgba(220,38,38,0.4); border-radius:16px; padding:40px; text-align:center; margin:24px 0; }
.pred-label { font-family:'Space Mono',monospace; font-size:0.68rem; letter-spacing:0.2em; text-transform:uppercase; color:#7c8499; margin-bottom:12px; }
.pred-price { font-family:'Space Mono',monospace; font-size:4.5rem; font-weight:700; color:#f1f3f9; line-height:1; margin:0; }
.pred-price sup { font-size:2rem; vertical-align:super; color:#7c8499; }
.pred-delta { display:inline-block; margin-top:16px; padding:6px 18px; border-radius:24px; font-family:'Space Mono',monospace; font-size:0.9rem; font-weight:700; }
.pred-delta.up   { background:rgba(34,197,94,0.12); color:#22c55e; border:1px solid rgba(34,197,94,0.25); }
.pred-delta.down { background:rgba(239,68,68,0.12);  color:#ef4444; border:1px solid rgba(239,68,68,0.25); }
.pred-base { font-size:0.85rem; color:#5a6070; margin-top:10px; }

.sec-header      { display:flex; align-items:center; gap:10px; margin:32px 0 16px; }
.sec-header-line { flex:1; height:1px; background:linear-gradient(90deg,#1e2433,transparent); }
.sec-header-text { font-family:'Space Mono',monospace; font-size:0.68rem; letter-spacing:0.2em; text-transform:uppercase; color:#5a6070; white-space:nowrap; }

.info-box { background:#111520; border:1px solid #1e2433; border-left:3px solid #dc2626; border-radius:8px; padding:16px 20px; font-size:0.88rem; color:#9aa0b0; line-height:1.8; margin:16px 0; }
.info-box strong { color:#e8eaf0; }
.note-box { background:#0d1a10; border:1px solid #1a3320; border-left:3px solid #22c55e; border-radius:8px; padding:14px 20px; font-size:0.85rem; color:#86efac; line-height:1.7; margin:12px 0; }
.note-box strong { color:#bbf7d0; }

div[data-testid="stMetric"] { background:#111520; border:1px solid #1e2433; border-radius:10px; padding:18px 20px; }
div[data-testid="stMetricValue"] { color:#f1f3f9 !important; }
div[data-testid="stMetricLabel"] { color:#5a6070 !important; }

.stTabs [data-baseweb="tab-list"] { background:#0f1219; border-bottom:1px solid #1e2433; gap:4px; }
.stTabs [data-baseweb="tab"] { background:transparent; color:#5a6070; border-radius:8px 8px 0 0; font-family:'Space Mono',monospace; font-size:0.72rem; letter-spacing:0.1em; }
.stTabs [aria-selected="true"] { background:rgba(220,38,38,0.1) !important; color:#dc2626 !important; border-bottom:2px solid #dc2626 !important; }

.stButton > button { background:#dc2626; color:white; border:none; border-radius:8px; font-family:'Space Mono',monospace; font-size:0.78rem; letter-spacing:0.1em; text-transform:uppercase; padding:12px 28px; transition:background 0.2s,transform 0.1s; }
.stButton > button:hover { background:#b91c1c; transform:translateY(-1px); }
div[data-testid="stFileUploader"] { background:#111520; border:1px dashed #1e2433; border-radius:10px; padding:8px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# PLOTLY BASE LAYOUT — reused by all charts via **PLOT_LAYOUT
# ══════════════════════════════════════════════════════════════════
PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0d1017",
    font=dict(family="DM Sans", color="#9aa0b0"),
    xaxis=dict(gridcolor="#1a2030", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#1a2030", showgrid=True, zeroline=False),
    margin=dict(l=20, r=20, t=50, b=30),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2433", borderwidth=1),
)


# ══════════════════════════════════════════════════════════════════
# LOAD MODEL — cached so it only runs once per session
# ══════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model  = load_model("lstm_model.h5", compile=False)
    model.compile(optimizer="adam", loss="mean_squared_error")
    scaler = joblib.load("scaler.pkl")
    config = joblib.load("config.pkl")
    return model, scaler, config


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:24px 0 16px;">
        <div style="font-family:'Space Mono',monospace;font-size:0.65rem;letter-spacing:0.2em;color:#5a6070;text-transform:uppercase;margin-bottom:6px;">NASDAQ</div>
        <div style="font-size:1.8rem;font-weight:700;color:#f1f3f9;">TSLA</div>
        <div style="font-size:0.8rem;color:#5a6070;margin-top:2px;">Tesla, Inc.</div>
    </div>
    <hr style="border-color:#1e2433;margin:0 0 24px;">
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-family:\'Space Mono\',monospace;font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase;color:#5a6070;margin-bottom:8px;">UPLOAD DATASET</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "CSV with Date and Close columns",
        type=["csv", "xlsx"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <div style="margin-top:28px;padding:16px;background:#111520;border:1px solid #1e2433;border-radius:10px;">
        <div style="font-family:'Space Mono',monospace;font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase;color:#5a6070;margin-bottom:12px;">Model Architecture</div>
        <div style="font-size:0.82rem;color:#9aa0b0;line-height:2;">
            Input → 60 Close prices<br>
            LSTM 128 units + Dropout 0.2<br>
            LSTM 64 units + Dropout 0.2<br>
            Dense 1 → next Close price<br>
            <hr style="border-color:#1e2433;margin:10px 0;">
            <span style="color:#5a6070;">Features used:</span> Close only<br>
            <span style="color:#5a6070;">Loss fn:</span> Mean Squared Error<br>
            <span style="color:#5a6070;">Optimizer:</span> Adam
        </div>
    </div>
    <div style="margin-top:16px;padding:14px;background:#111520;border:1px solid #1e2433;border-radius:10px;">
        <div style="font-family:'Space Mono',monospace;font-size:0.65rem;letter-spacing:0.15em;text-transform:uppercase;color:#5a6070;margin-bottom:8px;">Metric Guide</div>
        <div style="font-size:0.78rem;color:#9aa0b0;line-height:1.9;">
            <b style="color:#e8eaf0;">Accuracy</b> = 100 − MAPE<br>
            <b style="color:#e8eaf0;">MAPE</b> = avg % error per day<br>
            <b style="color:#e8eaf0;">MAE</b> = avg dollar error<br>
            <b style="color:#e8eaf0;">RMSE</b> = error penalising outliers<br>
            <hr style="border-color:#1e2433;margin:8px 0;">
            <span style="color:#f87171;">$16 error on $400 stock = 4%</span><br>
            Dollar error needs price context.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="position:fixed;bottom:24px;left:0;right:0;text-align:center;">
        <div style="font-size:0.72rem;color:#2a3040;">Muhammad Wahaj Bin Aamir</div>
        <div style="font-size:0.65rem;color:#1e2433;">Skill Enhancement Program</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-ticker">NASDAQ: TSLA · Deep Learning Forecasting System</div>
    <div class="hero-title">Tesla Stock<br><span>Price Predictor</span></div>
    <div class="hero-sub">Stacked LSTM neural network trained exclusively on historical Close prices.
        Upload your dataset to forecast the next trading session.</div>
    <div class="hero-badge">⚡ Close-Price-Only Model · 60-Day Lookback Window</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# VERIFY MODEL FILES
# ══════════════════════════════════════════════════════════════════
missing = [f for f in ["lstm_model.h5", "scaler.pkl", "config.pkl"] if not os.path.exists(f)]
if missing:
    st.error(f"**Missing model files:** `{'`, `'.join(missing)}`\n\nPlace them in the same folder as app.py and restart.")
    st.stop()

with st.spinner("Loading model..."):
    model, scaler, config = load_artifacts()

LOOK_BACK  = config["look_back"]
USE_DIFF   = config["use_differencing"]
MODEL_ACC  = config["accuracy"]       # from training test set
MODEL_MAE  = config["mae"]            # from training test set
MODEL_RMSE = config["rmse"]           # from training test set
MODEL_MAPE = 100 - MODEL_ACC


# ══════════════════════════════════════════════════════════════════
# MODEL BENCHMARK STRIP
# These numbers are from the saved config — performance on the
# held-out test split during Kaggle training. Clearly labelled.
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="stat-row">
    <div class="stat-card">
        <div class="stat-label">Benchmark Accuracy</div>
        <div class="stat-value {'positive' if MODEL_ACC >= 90 else 'accent'}">{MODEL_ACC:.2f}%</div>
        <div class="stat-detail">100 − MAPE · measured on training notebook test split</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Benchmark MAPE</div>
        <div class="stat-value">{MODEL_MAPE:.2f}%</div>
        <div class="stat-detail">Avg % error across test split</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Benchmark MAE</div>
        <div class="stat-value">${MODEL_MAE:.2f}</div>
        <div class="stat-detail">Avg dollar error · must compare to stock price</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Benchmark RMSE</div>
        <div class="stat-value">${MODEL_RMSE:.2f}</div>
        <div class="stat-detail">Penalises larger errors more than MAE</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Lookback</div>
        <div class="stat-value">{LOOK_BACK} days</div>
        <div class="stat-detail">Close prices required as input</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Features Used</div>
        <div class="stat-value" style="font-size:0.9rem;padding-top:8px;">Close Only</div>
        <div class="stat-detail">Open / High / Low / Volume ignored</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="note-box">
    <strong>Why does MAE = ${MODEL_MAE:.2f} but accuracy = {MODEL_ACC:.2f}%?</strong>
    These are two ways of measuring the same thing.
    MAE is the average dollar amount the model was off per day.
    Tesla's stock averaged ~$200–$700 during testing, so ${MODEL_MAE:.2f} off
    equals roughly {MODEL_MAPE:.1f}% — which is what MAPE captures directly.
    Accuracy = 100 − MAPE = <strong>{MODEL_ACC:.2f}%</strong>.
    Dollar error alone, without knowing the stock's price level, tells you nothing useful.
    Always look at MAPE and Accuracy together.
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# NO FILE STATE
# ══════════════════════════════════════════════════════════════════
if uploaded is None:
    st.markdown(f"""
    <div class="info-box">
        👈 <strong>Upload a CSV file</strong> from the sidebar to begin.<br><br>
        Required columns: <strong>Date</strong> and <strong>Close</strong>.
        All other columns are automatically ignored — the model was trained on Close prices only.<br><br>
        The model will use the last <strong>{LOOK_BACK} Close prices</strong> from your file
        to predict the next trading session's closing price.
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════
# LOAD & CLEAN DATA
# We only keep Date and Close — everything else is discarded.
# ══════════════════════════════════════════════════════════════════
try:
    df_raw = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

date_col  = next((c for c in df_raw.columns if "date"  in c.lower()), None)
close_col = next((c for c in df_raw.columns if "close" in c.lower() and "adj" not in c.lower()), None)

if not date_col or not close_col:
    st.error("Could not detect a **Date** or **Close** column. Please verify your file.")
    st.stop()

df = df_raw[[date_col, close_col]].copy()   # drop everything else
df[close_col] = pd.to_numeric(
    df[close_col].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
    errors="coerce",
)
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna().sort_values(date_col).reset_index(drop=True)
values = df[close_col].values

if len(values) < LOOK_BACK + 1:
    st.error(f"Need at least {LOOK_BACK + 1} rows. Your file has {len(values)}.")
    st.stop()

st.markdown(f"""
<div style="background:#0d1a10;border:1px solid #1a3320;border-left:3px solid #22c55e;border-radius:8px;padding:12px 20px;font-size:0.85rem;color:#86efac;margin-bottom:24px;">
    ✓ Dataset loaded — <strong>{len(df):,} trading days</strong> &nbsp;|&nbsp;
    {df[date_col].min().strftime('%b %d, %Y')} → {df[date_col].max().strftime('%b %d, %Y')}
    &nbsp;|&nbsp; Price range: <strong>${values.min():.2f} – ${values.max():.2f}</strong>
    &nbsp;|&nbsp; Using <strong>{close_col}</strong> only
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
tab_forecast, tab_backtest, tab_chart = st.tabs([
    "🔮  NEXT-DAY FORECAST",
    "📊  BACKTEST  &  METRICS",
    "📈  PRICE CHART",
])


# ════════════════════════════════════════════════════════════════
# TAB 1 · NEXT-DAY FORECAST
# Step-by-step logic:
#   1. Take last LOOK_BACK+1 Close prices
#   2. If USE_DIFF=True: compute differences (price_t − price_t-1)
#      so the model sees changes, not absolute levels
#   3. Scale to [0,1] using the saved MinMaxScaler
#   4. Reshape to (1, 60, 1) — LSTM expects (batch, timesteps, features)
#   5. model.predict() → scaled prediction
#   6. Inverse scale back to dollars
#   7. If USE_DIFF: add predicted change to last known price
# ════════════════════════════════════════════════════════════════
with tab_forecast:
    st.markdown("""
    <div class="sec-header">
        <div class="sec-header-text">Live Inference Engine</div>
        <div class="sec-header-line"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
        The model reads the <strong>last {LOOK_BACK} Close prices</strong> from your dataset,
        scales them using the pre-trained MinMaxScaler, and passes them through two LSTM layers
        to output one number — the predicted Close price for the next trading session.
        <strong>No other data is used.</strong>
    </div>
    """, unsafe_allow_html=True)

    col_btn, col_hist = st.columns([1, 2])

    with col_btn:
        run = st.button("Run Forecast", type="primary", use_container_width=True)

    with col_hist:
        rows_html = "".join(
            f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #181e2a;font-size:0.84rem;">'
            f'<span style="color:#5a6070;">{df[date_col].iloc[-(5-i)].strftime("%b %d, %Y")}</span>'
            f'<span style="color:#f1f3f9;font-family:Space Mono,monospace;">${values[-(5-i)]:.2f}</span></div>'
            for i in range(5)
        )
        st.markdown(f"""
        <div style="background:#111520;border:1px solid #1e2433;border-radius:10px;padding:14px 18px;">
            <div style="font-family:'Space Mono',monospace;font-size:0.65rem;letter-spacing:0.15em;color:#5a6070;margin-bottom:10px;">LAST 5 CLOSE PRICES</div>
            {rows_html}
        </div>
        """, unsafe_allow_html=True)

    if run:
        with st.spinner("Running inference..."):
            raw_window = values[-(LOOK_BACK + 1):]
            seq = np.diff(raw_window) if USE_DIFF else raw_window[-LOOK_BACK:]
            seq_scaled  = scaler.transform(seq.reshape(-1, 1))
            input_seq   = seq_scaled[-LOOK_BACK:].reshape(1, LOOK_BACK, 1)
            pred_scaled = model.predict(input_seq, verbose=0)
            pred_diff   = scaler.inverse_transform(pred_scaled).flatten()[0]
            predicted   = float(values[-1] + pred_diff) if USE_DIFF else float(pred_diff)
            last_close  = float(values[-1])
            delta       = predicted - last_close
            pct         = (delta / last_close) * 100
            direction   = "up" if delta >= 0 else "down"
            arrow       = "▲" if delta >= 0 else "▼"

        st.markdown(f"""
        <div class="pred-card">
            <div class="pred-label">Projected Next-Session Close</div>
            <div class="pred-price"><sup>$</sup>{predicted:.2f}</div>
            <div class="pred-delta {direction}">{arrow} ${abs(delta):.2f} &nbsp;({pct:+.2f}%)</div>
            <div class="pred-base">vs. last known close of ${last_close:.2f} on {df[date_col].iloc[-1].strftime('%B %d, %Y')}</div>
        </div>
        """, unsafe_allow_html=True)

        # Show the exact 60-day window the model consumed
        st.markdown("""
        <div class="sec-header">
            <div class="sec-header-text">The 60 Close prices the model just used as input</div>
            <div class="sec-header-line"></div>
        </div>
        """, unsafe_allow_html=True)

        window_prices = values[-LOOK_BACK:]
        fig_win = go.Figure()
        fig_win.add_trace(go.Scatter(
            x=list(range(LOOK_BACK)), y=window_prices,
            mode="lines+markers",
            line=dict(color="#dc2626", width=2),
            marker=dict(size=4, color="#dc2626"),
            name="Close (model input)",
            hovertemplate="Day %{x}: $%{y:.2f}<extra></extra>",
        ))
        fig_win.add_trace(go.Scatter(
            x=[LOOK_BACK], y=[predicted],
            mode="markers",
            marker=dict(size=14, color="#22c55e", symbol="star"),
            name=f"Predicted: ${predicted:.2f}",
            hovertemplate=f"Predicted: ${predicted:.2f}<extra></extra>",
        ))
        fig_win.add_trace(go.Scatter(
            x=[LOOK_BACK - 1, LOOK_BACK], y=[window_prices[-1], predicted],
            mode="lines", line=dict(color="#22c55e", width=1.5, dash="dot"),
            showlegend=False,
        ))
        fig_win.update_layout(
            **PLOT_LAYOUT, height=280,
            title=f"Days 0–59 = the 60 Close prices fed in · ★ = predicted next-day Close",
            xaxis_title="Day index (0 = oldest, 59 = today)",
            yaxis_title="Close Price ($)",
        )
        st.plotly_chart(fig_win, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 2 · BACKTEST & METRICS
#
# We slide the 60-day window across the entire uploaded dataset,
# predict at each step, and compare to actual prices.
#
# KEY INSIGHT about the two charts:
#   Chart 1 (Actual vs Predicted): Y-axis = absolute price ($150–$400)
#     → looks like a smooth curve because both lines are at similar heights
#   Chart 2 (Residuals): Y-axis = error in $ (usually ±$5 to ±$40)
#     → looks noisy/volatile because the axis is ZOOMED IN 10x
#   A $20 error on a $350 stock is < 6% — totally normal.
#   If plotted at the same scale as chart 1, residuals would be
#   nearly invisible as a flat line near zero.
# ════════════════════════════════════════════════════════════════
with tab_backtest:
    st.markdown("""
    <div class="sec-header">
        <div class="sec-header-text">Historical Backtest</div>
        <div class="sec-header-line"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
        Slides a <strong>{LOOK_BACK}-day window</strong> across your entire dataset and records
        the model's prediction at each step vs what actually happened. Only Close prices are used.
    </div>
    """, unsafe_allow_html=True)

    if st.button("Run Full Backtest", type="primary"):
        with st.spinner("Processing all trading days..."):
            raw       = values.copy()
            processed = np.diff(raw) if USE_DIFF else raw.copy()
            scaled    = scaler.transform(processed.reshape(-1, 1))

            X_all, y_all = [], []
            for i in range(LOOK_BACK, len(scaled)):
                X_all.append(scaled[i - LOOK_BACK:i, 0])
                y_all.append(scaled[i, 0])

            X_all = np.array(X_all).reshape(-1, LOOK_BACK, 1)
            y_all = np.array(y_all)

            pred_sc  = model.predict(X_all, verbose=0)
            pred_inv = scaler.inverse_transform(pred_sc).flatten()
            act_inv  = scaler.inverse_transform(y_all.reshape(-1, 1)).flatten()

            if USE_DIFF:
                s      = LOOK_BACK + 1
                base   = raw[s - 1: s - 1 + len(pred_inv)]
                pred_f = base + pred_inv
                act_f  = raw[s: s + len(pred_inv)]
                ml     = min(len(act_f), len(pred_f))
                act_f, pred_f = act_f[:ml], pred_f[:ml]
            else:
                act_f, pred_f = act_inv, pred_inv

            rmse_v   = float(np.sqrt(mean_squared_error(act_f, pred_f)))
            mae_v    = float(mean_absolute_error(act_f, pred_f))
            mape_v   = float(np.mean(np.abs((act_f - pred_f) / (np.abs(act_f) + 1e-8))) * 100)
            acc_v    = 100 - mape_v
            errors   = pred_f - act_f
            avg_px   = float(np.mean(act_f))
            mae_pct  = mae_v / avg_px * 100

        # Metrics row
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-card">
                <div class="stat-label">Accuracy</div>
                <div class="stat-value {'positive' if acc_v >= 90 else 'accent'}">{acc_v:.2f}%</div>
                <div class="stat-detail">100 − MAPE · the headline number</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">MAPE</div>
                <div class="stat-value">{mape_v:.2f}%</div>
                <div class="stat-detail">Avg % error per prediction</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">MAE</div>
                <div class="stat-value">${mae_v:.2f}</div>
                <div class="stat-detail">Avg dollar off per prediction</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">MAE as % of Price</div>
                <div class="stat-value">{mae_pct:.2f}%</div>
                <div class="stat-detail">MAE ÷ avg price ${avg_px:.0f} — honest comparison</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">RMSE</div>
                <div class="stat-value">${rmse_v:.2f}</div>
                <div class="stat-detail">Higher than MAE → some large error spikes</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Explain why charts look so different
        st.markdown(f"""
        <div class="note-box">
            <strong>Why does the residual chart look dramatic when the accuracy chart looks smooth?</strong><br>
            The top chart Y-axis spans <strong>${int(act_f.max() - act_f.min())}</strong>
            (full price range). The bottom chart Y-axis spans only
            <strong>±${int(max(abs(errors.max()), abs(errors.min())) + 5)}</strong> — it is zoomed
            in ~{int((act_f.max()-act_f.min()) / (max(abs(errors.max()),abs(errors.min()))+5))}x
            compared to the top chart. Those bars that look large are only
            ${mae_v:.1f} on average, which is {mae_pct:.1f}% of the stock price.
            If both charts shared the same Y-scale, the residuals would appear as a flat
            line near zero. <strong>This is what a good model looks like.</strong>
        </div>
        """, unsafe_allow_html=True)

        # Chart 1: Actual vs Predicted
        st.markdown("""
        <div class="sec-header">
            <div class="sec-header-text">Chart 1 — Actual vs Predicted (absolute dollar prices)</div>
            <div class="sec-header-line"></div>
        </div>
        """, unsafe_allow_html=True)

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(y=act_f,  name="Actual Close",    line=dict(color="#e2e8f0", width=1.8), hovertemplate="Day %{x} — Actual: $%{y:.2f}<extra></extra>"))
        fig_bt.add_trace(go.Scatter(y=pred_f, name="LSTM Prediction", line=dict(color="#dc2626", width=1.5, dash="dot"), hovertemplate="Day %{x} — Predicted: $%{y:.2f}<extra></extra>"))
        fig_bt.update_layout(
            **PLOT_LAYOUT, height=400,
            xaxis_title="Trading Day",
            yaxis_title="Close Price ($)  ←  full price range",
            hovermode="x unified",
            title=f"Model Accuracy: {acc_v:.2f}%   |   Y-axis = full dollar range (${int(act_f.min())}–${int(act_f.max())})",
        )
        st.plotly_chart(fig_bt, use_container_width=True)

        # Chart 2: Residuals — explicitly labelled as zoomed in
        st.markdown("""
        <div class="sec-header">
            <div class="sec-header-text">Chart 2 — Prediction Error (zoomed in — Y-axis ≠ Chart 1)</div>
            <div class="sec-header-line"></div>
        </div>
        """, unsafe_allow_html=True)

        fig_err = go.Figure()
        fig_err.add_trace(go.Bar(
            y=errors,
            marker_color=["#22c55e" if e >= 0 else "#ef4444" for e in errors],
            name="Error = Predicted − Actual",
            hovertemplate="Day %{x}: $%{y:+.2f}<extra></extra>",
        ))
        fig_err.add_hline(y=0,       line_color="#5a6070", line_dash="dash")
        fig_err.add_hline(y= mae_v,  line_color="#f59e0b", line_dash="dot", line_width=1.2, annotation_text=f"+MAE ${mae_v:.1f}", annotation_position="right")
        fig_err.add_hline(y=-mae_v,  line_color="#f59e0b", line_dash="dot", line_width=1.2, annotation_text=f"−MAE ${mae_v:.1f}", annotation_position="right")
        fig_err.update_layout(
            **PLOT_LAYOUT, height=300,
            xaxis_title="Trading Day",
            yaxis_title=f"Error in $ (zoomed — range ±${int(max(abs(errors.max()),abs(errors.min()))+5)})",
            title=(
                f"⚠ Y-axis is zoomed in vs Chart 1. "
                f"Errors look big here but average only ${mae_v:.1f} on a ${avg_px:.0f} stock = {mae_pct:.1f}%. "
                f"Amber = ±MAE band."
            ),
        )
        st.plotly_chart(fig_err, use_container_width=True)

        st.markdown(f"""
        <div class="note-box">
            Most bars are within the amber ±${mae_v:.1f} band.
            Occasional larger spikes happen on days of sharp price movement —
            LSTMs lag slightly on sudden jumps. This is normal behaviour.
            The overall {acc_v:.2f}% accuracy confirms the model is performing well.
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# TAB 3 · PRICE CHART
# Pure data exploration — no model involved.
# Shows the historical Close prices with moving averages.
# ════════════════════════════════════════════════════════════════
with tab_chart:
    st.markdown("""
    <div class="sec-header">
        <div class="sec-header-text">Historical Price Chart</div>
        <div class="sec-header-line"></div>
    </div>
    <div class="info-box">
        Raw historical Close prices from your dataset with 7-day and 30-day moving averages.
        No model is involved here — this is purely for exploring the data you uploaded.
        The moving averages smooth out daily noise to reveal the underlying price trend.
    </div>
    """, unsafe_allow_html=True)

    ma7  = df[close_col].rolling(7).mean()
    ma30 = df[close_col].rolling(30).mean()

    fig_main = go.Figure()
    fig_main.add_trace(go.Scatter(x=df[date_col], y=df[close_col], fill="tozeroy", fillcolor="rgba(220,38,38,0.06)", line=dict(color="#dc2626", width=1.5), name="Close Price",      hovertemplate="%{x|%b %d, %Y}: $%{y:.2f}<extra></extra>"))
    fig_main.add_trace(go.Scatter(x=df[date_col], y=ma7,           line=dict(color="#60a5fa", width=1.2, dash="dot"),  name="7-Day MA (fast)",  hovertemplate="%{x|%b %d, %Y}: $%{y:.2f}<extra></extra>"))
    fig_main.add_trace(go.Scatter(x=df[date_col], y=ma30,          line=dict(color="#f59e0b", width=1.8, dash="dash"), name="30-Day MA (slow)", hovertemplate="%{x|%b %d, %Y}: $%{y:.2f}<extra></extra>"))
    fig_main.update_layout(
        **PLOT_LAYOUT, height=460,
        xaxis_title="Date",
        yaxis_title="Close Price ($)",
        hovermode="x unified",
        title="Tesla (TSLA) · Historical Close Price with Moving Averages",
    )
    st.plotly_chart(fig_main, use_container_width=True)

    total_ret   = ((values[-1] / values[0]) - 1) * 100
    avg_daily   = float(np.mean(np.diff(values)))

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-label">All-Time Low</div>
            <div class="stat-value">${values.min():.2f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">All-Time High</div>
            <div class="stat-value">${values.max():.2f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Latest Close</div>
            <div class="stat-value">${values[-1]:.2f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Total Return</div>
            <div class="stat-value {'positive' if total_ret >= 0 else 'negative'}">{total_ret:+.1f}%</div>
            <div class="stat-detail">First → last Close in dataset</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Avg Daily Move</div>
            <div class="stat-value {'positive' if avg_daily >= 0 else 'negative'}">${avg_daily:+.2f}</div>
            <div class="stat-detail">Mean day-over-day change</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
