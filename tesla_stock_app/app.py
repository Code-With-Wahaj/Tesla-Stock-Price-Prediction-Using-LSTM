import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from tensorflow.keras.models import load_model
import joblib
import os

# ── Page Config ──
st.set_page_config(page_title="Tesla Stock Predictor", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem; font-weight: 700;
        text-align: center; color: #E31937; margin-bottom: 0;
    }
    .sub-title {
        font-size: 1.15rem; text-align: center;
        color: #888; margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem; font-weight: 600; color: #1a1a2e;
        border-left: 4px solid #E31937;
        padding-left: 12px; margin: 1.5rem 0 1rem 0;
    }
    div[data-testid="stMetric"] {
        background-color: #f8f9fa; border: 1px solid #e9ecef;
        border-radius: 10px; padding: 15px;
    }
    .prediction-box {
        background: linear-gradient(135deg, #E31937, #ff6b6b);
        color: white; padding: 30px; border-radius: 15px;
        text-align: center; font-size: 1.2rem; margin: 1rem 0;
    }
    .prediction-box h1 {
        font-size: 3rem; margin: 10px 0; color: white;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚡ Tesla Stock Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">LSTM Deep Learning — Predict Next Day Close Price</div>', unsafe_allow_html=True)
st.markdown("---")


# ── Load Model ──
@st.cache_resource
def load_artifacts():
    model = load_model('lstm_model.h5', compile=False)
    model.compile(optimizer='adam', loss='mean_squared_error')
    scaler = joblib.load('scaler.pkl')
    config = joblib.load('config.pkl')
    return model, scaler, config

missing = [f for f in ['lstm_model.h5','scaler.pkl','config.pkl'] if not os.path.exists(f)]
if missing:
    st.error(f"❌ Missing: {missing}")
    st.stop()

model, scaler, config = load_artifacts()
LOOK_BACK = config['look_back']
USE_DIFF = config['use_differencing']


# ── Sidebar ──
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/Tesla_Motors.svg/279px-Tesla_Motors.svg.png", width=180)
    st.markdown("---")
    st.markdown("### 🧠 Model Info")
    st.write(f"**Trained On:** {config['trained_on']}")
    st.write(f"**Training Accuracy:** {config['accuracy']:.2f}%")
    st.write(f"**Window Size:** {LOOK_BACK} days")
    st.markdown("---")

    st.markdown("### 📂 Upload Data")
    uploaded = st.file_uploader("Upload stock CSV", type=['csv','xlsx'])

    st.markdown("---")
    st.markdown("### How It Works")
    st.markdown(
        "1. Model reads last **60 days** of close prices\n"
        "2. Predicts the **next day's** close price\n"
        "3. Compares with actual prices to show accuracy"
    )


# ── No File ──
if uploaded is None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", "Stacked LSTM")
    c2.metric("Accuracy", f"{config['accuracy']:.2f}%")
    c3.metric("Input", f"Last {LOOK_BACK} Days")

    st.info("👈 Upload a stock CSV to predict prices")

    st.markdown('<div class="section-header">How This Model Works</div>', unsafe_allow_html=True)
    st.markdown(f"""
    ```
    INPUT:  Last {LOOK_BACK} closing prices
                ↓
    MODEL:  Stacked LSTM (128 → 64 neurons)
                ↓
    OUTPUT: Predicted closing price for NEXT day
    ```
    
    **For real predictions:** Upload Tesla data from dates 
    AFTER June 2021 (data the model has never seen).
    
    Download fresh Tesla data from 
    [Yahoo Finance](https://finance.yahoo.com/quote/TSLA/history/)
    """)
    st.stop()



# ── FILE UPLOADED ──

try:
    df = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

# ── Column Selection ──
st.markdown('<div class="section-header">📥 Data Setup</div>', unsafe_allow_html=True)
cols = df.columns.tolist()
c1, c2 = st.columns(2)

with c1:
    date_col = st.selectbox("Date Column", cols, index=0)
with c2:
    for c in cols:
        if c != date_col:
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace('$','',regex=False).str.replace(',','',regex=False),
                errors='coerce')
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        st.error("No numeric columns found!")
        st.stop()
    default = 0
    for i, c in enumerate(num_cols):
        if 'close' in c.lower():
            default = i
            break
    value_col = st.selectbox("Price Column", num_cols, index=default)

# ── Clean ──
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df = df.dropna(subset=[date_col, value_col])
df = df.sort_values(date_col).reset_index(drop=True)

st.success(f"✅ **{len(df)}** records  •  "
           f"{df[date_col].min().date()} to {df[date_col].max().date()}")

with st.expander("Preview Data"):
    st.dataframe(df[[date_col, value_col]].head(10), use_container_width=True, hide_index=True)


# ── EDA ──
st.markdown('<div class="section-header">📊 Data Analysis</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📈 Chart", "📐 ADF Test"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[date_col], y=df[value_col], mode='lines',
                              line=dict(color='#E31937', width=1.5)))
    fig.update_layout(title="Stock Price", xaxis_title="Date", yaxis_title="Price ($)",
                      template="plotly_white", height=400, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    values = df[value_col].values
    adf = adfuller(values, autolag='AIC')
    a1, a2, a3 = st.columns(3)
    a1.metric("ADF Statistic", f"{adf[0]:.4f}")
    a2.metric("P-Value", f"{adf[1]:.6f}")
    a3.metric("Stationary?", "✅ Yes" if adf[1] < 0.05 else "❌ No")


# ── Choose Mode ──
st.markdown("---")
st.markdown('<div class="section-header">🎯 What Do You Want To Do?</div>', unsafe_allow_html=True)

mode = st.radio(
    "Select mode:",
    ["🔮 Predict Next Day Price", "📊 Test Model on Full Dataset"],
    horizontal=True
)

values = df[value_col].values


# ════════════════════════════════════
# MODE 1: PREDICT NEXT DAY
# ════════════════════════════════════

if mode == "🔮 Predict Next Day Price":

    st.markdown(f"""
    The model will use the **last {LOOK_BACK} closing prices** from your 
    uploaded data to predict what the **next trading day's** close price will be.
    """)

    if len(values) < LOOK_BACK:
        st.error(f"Need at least {LOOK_BACK} records. You have {len(values)}.")
        st.stop()

    if st.button("🔮 Predict Next Day", type="primary", use_container_width=True):

        with st.spinner("Predicting..."):

            # Get last 60 prices
            last_prices = values[-LOOK_BACK - 1:]  # +1 extra for differencing

            if USE_DIFF:
                last_diff = np.diff(last_prices)
            else:
                last_diff = last_prices[-LOOK_BACK:]

            # Scale
            last_scaled = scaler.transform(last_diff.reshape(-1, 1))

            # Take last LOOK_BACK values and reshape for LSTM
            input_seq = last_scaled[-LOOK_BACK:].reshape(1, LOOK_BACK, 1)

            # Predict
            pred_scaled = model.predict(input_seq, verbose=0)
            pred_value = scaler.inverse_transform(pred_scaled).flatten()[0]

            # If differencing, add to last known price
            if USE_DIFF:
                predicted_price = values[-1] + pred_value
            else:
                predicted_price = pred_value

            last_date = df[date_col].iloc[-1]
            last_price = values[-1]

        # Show Result
        st.markdown(f"""
        <div class="prediction-box">
            <p>📅 Based on data up to <b>{last_date.strftime('%B %d, %Y')}</b></p>
            <p>Last known close: <b>${last_price:.2f}</b></p>
            <hr style="border-color: rgba(255,255,255,0.3);">
            <p>Predicted next trading day close:</p>
            <h1>${predicted_price:.2f}</h1>
            <p>Change: <b>${predicted_price - last_price:+.2f}</b> 
            ({((predicted_price - last_price)/last_price)*100:+.2f}%)</p>
        </div>
        """, unsafe_allow_html=True)

        # Show the 60 days used
        with st.expander(f"📋 Last {LOOK_BACK} prices used as input"):
            input_df = df[[date_col, value_col]].tail(LOOK_BACK)
            st.dataframe(input_df, use_container_width=True, hide_index=True)


# ════════════════════════════════════
# MODE 2: TEST ON FULL DATASET
# ════════════════════════════════════

elif mode == "📊 Test Model on Full Dataset":

    st.markdown("""
    The model goes through your data day by day — uses past 60 days 
    to predict each next day, then compares with the actual price.
    """)

    if st.button("📊 Run Full Test", type="primary", use_container_width=True):

        with st.spinner("Processing..."):

            raw = values.copy()
            processed = np.diff(values) if USE_DIFF else values.copy()

            if len(processed) <= LOOK_BACK:
                st.error(f"Need at least {LOOK_BACK+1} records.")
                st.stop()

            scaled = scaler.transform(processed.reshape(-1, 1))

            X, y_sc = [], []
            for i in range(LOOK_BACK, len(scaled)):
                X.append(scaled[i-LOOK_BACK:i, 0])
                y_sc.append(scaled[i, 0])

            X = np.array(X).reshape(-1, LOOK_BACK, 1)
            y_sc = np.array(y_sc)

            pred_sc = model.predict(X, verbose=0)
            pred_inv = scaler.inverse_transform(pred_sc).flatten()
            actual_inv = scaler.inverse_transform(y_sc.reshape(-1, 1)).flatten()

            if USE_DIFF:
                start = LOOK_BACK + 1
                base = raw[start-1:start-1+len(pred_inv)]
                pred_f = base + pred_inv
                actual_f = raw[start:start+len(pred_inv)]
                ml = min(len(actual_f), len(pred_f))
                actual_f, pred_f = actual_f[:ml], pred_f[:ml]
            else:
                actual_f, pred_f = actual_inv, pred_inv

            rmse = np.sqrt(mean_squared_error(actual_f, pred_f))
            mae = mean_absolute_error(actual_f, pred_f)
            mape = np.mean(np.abs((actual_f-pred_f)/(np.abs(actual_f)+1e-8)))*100
            accuracy = 100 - mape
            errors = actual_f - pred_f

        # Metrics
        st.markdown('<div class="section-header">📊 Results</div>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RMSE", f"${rmse:.2f}")
        m2.metric("MAE", f"${mae:.2f}")
        m3.metric("MAPE", f"{mape:.2f}%")
        m4.metric("Accuracy", f"{accuracy:.2f}%",
                  delta="✅ Target met" if accuracy >= 85 else "⚠️ Below target")

        if accuracy >= 90:
            st.success(f"🎉 {accuracy:.2f}% accuracy — Excellent!")
        elif accuracy >= 85:
            st.success(f"✅ {accuracy:.2f}% accuracy — Above target!")
        else:
            st.warning(f"{accuracy:.2f}% accuracy")

        # Charts
        st.markdown('<div class="section-header">📈 Actual vs Predicted</div>', unsafe_allow_html=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=actual_f, name='Actual',
                                   line=dict(color='#1E88E5', width=2)))
        fig2.add_trace(go.Scatter(y=pred_f, name='Predicted',
                                   line=dict(color='#E31937', width=2)))
        fig2.update_layout(
            title=f"Accuracy: {accuracy:.2f}%",
            xaxis_title="Trading Day", yaxis_title="Price ($)",
            template="plotly_white", height=480, hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig2, use_container_width=True)

        # Zoomed
        zoom = min(100, len(actual_f))
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(y=actual_f[-zoom:], name='Actual',
                                   line=dict(color='#1E88E5', width=2.5)))
        fig3.add_trace(go.Scatter(y=pred_f[-zoom:], name='Predicted',
                                   line=dict(color='#E31937', width=2.5)))
        fig3.update_layout(title=f"Last {zoom} Days", template="plotly_white",
                           height=380, hovermode='x unified')
        st.plotly_chart(fig3, use_container_width=True)

        # Errors
        st.markdown('<div class="section-header">📊 Error Analysis</div>', unsafe_allow_html=True)

        e1, e2 = st.columns(2)
        with e1:
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(y=errors, line=dict(color='#E53935', width=1)))
            fig4.add_hline(y=0, line_dash="dash", line_color="gray")
            fig4.update_layout(title="Error Over Time", template="plotly_white", height=320)
            st.plotly_chart(fig4, use_container_width=True)
        with e2:
            fig5 = go.Figure()
            fig5.add_trace(go.Histogram(x=errors, nbinsx=40,
                                         marker_color='#E53935', opacity=0.7))
            fig5.update_layout(title="Error Distribution", template="plotly_white", height=320)
            st.plotly_chart(fig5, use_container_width=True)

        # Summary
        st.markdown('<div class="section-header">📝 Summary</div>', unsafe_allow_html=True)
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | **Accuracy** | **{accuracy:.2f}%** |
        | RMSE | ${rmse:.2f} |
        | MAE | ${mae:.2f} |
        | MAPE | {mape:.2f}% |
        | Records | {len(actual_f)} |
        | Window | {LOOK_BACK} days |
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#aaa;padding:1rem;'>"
    "Tesla Stock Predictor • LSTM Deep Learning • Wahaj • Aptech</div>",
    unsafe_allow_html=True)