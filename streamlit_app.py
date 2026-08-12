"""
╔══════════════════════════════════════════════════════════════════╗
║   📈 STOCK PRICE PREDICTION — STREAMLIT WEB APP                 ║
║   Top 10 Indian Companies | ML + Deep Learning                   ║
║   Run: streamlit run streamlit_app.py                            ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings('ignore')
import os, math, datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "📈 Indian Stock Predictor",
    page_icon   = "📈",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 800;
        background: linear-gradient(135deg, #1a237e, #0288d1);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; padding: 10px 0;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 12px; padding: 16px 20px;
        border-left: 5px solid #0288d1; margin: 8px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .forecast-up   { color: #2e7d32; font-weight: 700; font-size: 1.1rem; }
    .forecast-down { color: #c62828; font-weight: 700; font-size: 1.1rem; }
    .section-title {
        font-size: 1.3rem; font-weight: 700; color: #1a237e;
        border-bottom: 2px solid #0288d1; padding-bottom: 6px; margin: 20px 0 12px 0;
    }
    .stProgress > div > div > div { background-color: #0288d1 !important; }
    .disclaimer {
        background: #fff3e0; border: 1px solid #ff9800; border-radius: 8px;
        padding: 12px; font-size: 0.85rem; color: #e65100; margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
STOCKS = {
    'RELIANCE.NS'  : 'Reliance Industries',
    'TCS.NS'       : 'Tata Consultancy Services',
    'INFY.NS'      : 'Infosys',
    'HDFCBANK.NS'  : 'HDFC Bank',
    'ICICIBANK.NS' : 'ICICI Bank',
    'ITC.NS'       : 'ITC Limited',
    'SBIN.NS'      : 'State Bank of India',
    'LT.NS'        : 'Larsen & Toubro',
    'HINDUNILVR.NS': 'Hindustan Unilever',
    'WIPRO.NS'     : 'Wipro'
}

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/stock-market.png", width=80)
    st.markdown("## ⚙️ Settings")

    selected_ticker = st.selectbox(
        "🏢 Select Company",
        options = list(STOCKS.keys()),
        format_func = lambda x: f"{STOCKS[x]} ({x})",
        index = 0
    )

    years_data = st.slider("📅 Years of Historical Data", 1, 7, 5)

    model_choice = st.radio(
        "🤖 Prediction Model",
        ["Linear Regression", "LSTM (Deep Learning)", "Both (Compare)"],
        index = 2
    )

    forecast_days = st.slider("🔮 Forecast Days", 7, 60, 30)

    lstm_epochs = st.slider("🔁 LSTM Epochs (Training Iterations)", 10, 100, 30) \
                  if model_choice in ["LSTM (Deep Learning)", "Both (Compare)"] else 30

    st.markdown("---")
    st.markdown("**📌 About**")
    st.info(
        "This app uses Machine Learning (Linear Regression) and Deep Learning (LSTM) "
        "to forecast Indian stock prices. Data is fetched live from Yahoo Finance."
    )
    st.markdown("""
    <div class='disclaimer'>
    ⚠️ <b>Disclaimer:</b> For educational purposes only. Not financial advice.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(ticker, years):
    end   = datetime.date.today()
    start = end - datetime.timedelta(days=years * 365)
    df    = yf.download(ticker, start=start, end=end,
                        progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    return df


def add_features(df):
    df = df.copy()
    df['MA_7']         = df['Close'].rolling(7).mean()
    df['MA_21']        = df['Close'].rolling(21).mean()
    df['MA_50']        = df['Close'].rolling(50).mean()
    df['EMA_12']       = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26']       = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']         = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal']  = df['MACD'].ewm(span=9, adjust=False).mean()
    rolling_std        = df['Close'].rolling(20).std()
    df['BB_Mid']       = df['Close'].rolling(20).mean()
    df['BB_Upper']     = df['BB_Mid'] + 2 * rolling_std
    df['BB_Lower']     = df['BB_Mid'] - 2 * rolling_std
    df['BB_Width']     = (df['BB_Upper'] - df['BB_Lower']) / (df['BB_Mid'] + 1e-10)
    delta              = df['Close'].diff()
    gain               = delta.clip(lower=0).rolling(14).mean()
    loss               = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI']          = 100 - 100 / (1 + gain / (loss + 1e-10))
    df['Daily_Return'] = df['Close'].pct_change() * 100
    df['High_Low_Range'] = df['High'] - df['Low']
    df['Volume_MA_10'] = df['Volume'].rolling(10).mean()
    df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA_10'] + 1e-10)
    for lag in [1, 2, 3, 5]:
        df[f'Lag_{lag}'] = df['Close'].shift(lag)
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    return df


def train_lr(df):
    features = ['Close','Open','High','Low','Volume','MA_7','MA_21','MA_50',
                'EMA_12','EMA_26','MACD','RSI','BB_Width','Daily_Return',
                'High_Low_Range','Volume_Ratio','Lag_1','Lag_2','Lag_3','Lag_5']
    available = [f for f in features if f in df.columns]
    X, y      = df[available].values, df['Target'].values
    split     = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    dates_te   = df.index[split:]
    scaler     = MinMaxScaler()
    X_tr       = scaler.fit_transform(X_tr)
    X_te       = scaler.transform(X_te)
    model      = LinearRegression().fit(X_tr, y_tr)
    y_pred     = model.predict(X_te)
    rmse = math.sqrt(mean_squared_error(y_te, y_pred))
    mae  = mean_absolute_error(y_te, y_pred)
    r2   = r2_score(y_te, y_pred)
    mape = np.mean(np.abs((y_te - y_pred) / (y_te + 1e-10))) * 100
    # Future prediction: predict next N steps (simple: keep updating Lag_1)
    last_row   = df[available].iloc[-1].copy()
    future_preds = []
    for _ in range(60):   # max 60 days buffer
        row_scaled = scaler.transform(last_row.values.reshape(1, -1))
        p          = model.predict(row_scaled)[0]
        future_preds.append(p)
        # Slide lag features
        for lag in [5, 3, 2, 1]:
            if f'Lag_{lag}' in last_row.index and f'Lag_{lag-1}' in last_row.index:
                last_row[f'Lag_{lag}'] = last_row[f'Lag_{lag-1}']
        if 'Lag_1' in last_row.index:
            last_row['Lag_1'] = last_row['Close']
        if 'Close' in last_row.index:
            last_row['Close'] = p
    return dict(y_te=y_te, y_pred=y_pred, dates_te=dates_te[:len(y_pred)],
                rmse=rmse, mae=mae, r2=r2, mape=mape,
                future_preds=future_preds)


def train_lstm_model(df, epochs=30):
    LOOKBACK   = 60
    feats      = ['Close','Volume','MA_21','RSI','MACD','BB_Width','Lag_1']
    avail      = [f for f in feats if f in df.columns]
    data       = df[avail].values
    scaler     = MinMaxScaler()
    data_sc    = scaler.fit_transform(data)
    n_f        = data_sc.shape[1]

    def make_seq(d, lb):
        Xs, ys = [], []
        for i in range(lb, len(d)):
            Xs.append(d[i-lb:i]); ys.append(d[i, 0])
        return np.array(Xs), np.array(ys)

    X, y     = make_seq(data_sc, LOOKBACK)
    split    = int(len(X) * 0.8)
    Xtr,Xte  = X[:split], X[split:]
    ytr,yte  = y[:split], y[split:]
    dates_te = df.index[LOOKBACK + split:]

    model = Sequential([
        Input(shape=(LOOKBACK, n_f)),
        LSTM(64, return_sequences=True), Dropout(0.2),
        LSTM(32, return_sequences=False), Dropout(0.2),
        Dense(16, activation='relu'), Dense(1)
    ])
    model.compile(optimizer='adam', loss='huber', metrics=['mae'])
    model.fit(Xtr, ytr, epochs=epochs, batch_size=32,
              validation_split=0.1,
              callbacks=[EarlyStopping(patience=6, restore_best_weights=True)],
              verbose=0)

    yp_sc = model.predict(Xte, verbose=0)

    def inv(vals, n):
        tmp        = np.zeros((len(vals), n))
        tmp[:, 0]  = vals.flatten()
        return scaler.inverse_transform(tmp)[:, 0]

    y_pred = inv(yp_sc, n_f)
    y_test = inv(yte.reshape(-1, 1), n_f)

    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100

    # Future forecasting
    seq       = data_sc[-LOOKBACK:].copy()
    future_pr = []
    for _ in range(60):
        inp   = seq.reshape(1, LOOKBACK, n_f)
        p_sc  = model.predict(inp, verbose=0)[0][0]
        nr    = seq[-1].copy(); nr[0] = p_sc
        seq   = np.vstack([seq[1:], nr])
        tmp   = np.zeros((1, n_f)); tmp[0, 0] = p_sc
        price = scaler.inverse_transform(tmp)[0, 0]
        future_pr.append(price)

    return dict(y_te=y_test, y_pred=y_pred, dates_te=dates_te[:len(y_pred)],
                rmse=rmse, mae=mae, r2=r2, mape=mape, future_preds=future_pr)


# ─────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📈 Indian Stock Price Predictor</div>',
            unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#555;font-size:1rem;'>"
    "Machine Learning & Deep Learning | Top 10 NSE-Listed Companies | Live Data from Yahoo Finance"
    "</p>", unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────────────────────
# FETCH DATA
# ─────────────────────────────────────────────────────────────
with st.spinner(f"⏳ Fetching live data for {STOCKS[selected_ticker]}..."):
    raw_df = fetch_data(selected_ticker, years_data)

if raw_df.empty:
    st.error("❌ Could not fetch data. Please check your internet connection or try again.")
    st.stop()

# ── KPI Cards ─────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
current_price = float(raw_df['Close'].iloc[-1])
prev_price    = float(raw_df['Close'].iloc[-2])
day_change    = current_price - prev_price
day_pct       = (day_change / prev_price) * 100
year_high     = float(raw_df['High'].max())
year_low      = float(raw_df['Low'].min())
avg_vol       = raw_df['Volume'].mean()

col1.metric("💰 Current Price",  f"₹{current_price:,.2f}", f"{day_pct:+.2f}%")
col2.metric("📈 52-Week High",   f"₹{year_high:,.2f}")
col3.metric("📉 52-Week Low",    f"₹{year_low:,.2f}")
col4.metric("📊 Avg Volume",     f"{avg_vol/1e6:.1f}M")
col5.metric("📅 Data Points",    f"{len(raw_df):,} days")

# ─────────────────────────────────────────────────────────────
# TAB LAYOUT
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Historical Data",
    "🔬 Technical Analysis",
    "🤖 ML Prediction",
    "🔮 30-Day Forecast",
    "📋 Model Report"
])

# ────────────────────────────────
# TAB 1: Historical Data
# ────────────────────────────────
with tab1:
    st.markdown(f'<div class="section-title">📊 {STOCKS[selected_ticker]} — Price History</div>',
                unsafe_allow_html=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # Price chart
    axes[0].plot(raw_df.index, raw_df['Close'], color='#1565C0', lw=1.5, label='Close Price')
    axes[0].fill_between(raw_df.index, raw_df['Close'].min(), raw_df['Close'],
                         alpha=0.08, color='#1565C0')
    axes[0].set_title(f'{STOCKS[selected_ticker]} — Historical Closing Price',
                      fontweight='bold', fontsize=13)
    axes[0].set_ylabel('Price (₹)', fontsize=11)
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Volume
    colors_vol = ['#4CAF50' if c >= o else '#F44336'
                  for c, o in zip(raw_df['Close'], raw_df['Open'])]
    axes[1].bar(raw_df.index, raw_df['Volume'], color=colors_vol, alpha=0.7)
    axes[1].set_title('Trading Volume (Green = Up Day, Red = Down Day)',
                      fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Volume', fontsize=11)
    axes[1].set_xlabel('Date', fontsize=11)
    axes[1].grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # Raw data table
    with st.expander("📋 View Raw Data (Last 30 rows)"):
        st.dataframe(
            raw_df[['Open','High','Low','Close','Volume']].tail(30).style.format({
                'Open': '₹{:.2f}', 'High': '₹{:.2f}',
                'Low': '₹{:.2f}', 'Close': '₹{:.2f}',
                'Volume': '{:,.0f}'
            }), use_container_width=True)

# ────────────────────────────────
# TAB 2: Technical Analysis
# ────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">🔬 Technical Indicators Dashboard</div>',
                unsafe_allow_html=True)

    with st.spinner("Computing technical indicators..."):
        feat_df = add_features(raw_df)
        recent  = feat_df.last('365D')

    fig, axes = plt.subplots(4, 1, figsize=(14, 18), sharex=True)

    # 1. Price + Bollinger Bands + MAs
    axes[0].plot(recent.index, recent['Close'],  lw=1.8, color='#1565C0', label='Close')
    axes[0].plot(recent.index, recent['MA_21'],  lw=1.2, color='#FF9800', label='MA 21', linestyle='--')
    axes[0].plot(recent.index, recent['MA_50'],  lw=1.2, color='#9C27B0', label='MA 50', linestyle='--')
    axes[0].fill_between(recent.index, recent['BB_Lower'], recent['BB_Upper'],
                         alpha=0.12, color='#78909C', label='Bollinger Bands')
    axes[0].set_title('Price + Moving Averages + Bollinger Bands', fontweight='bold', fontsize=11)
    axes[0].set_ylabel('Price (₹)'); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    # 2. Volume
    axes[1].bar(recent.index, recent['Volume'], color='#43A047', alpha=0.7, label='Volume')
    axes[1].plot(recent.index, recent['Volume_MA_10'], color='red', lw=1.5, label='Vol MA10')
    axes[1].set_title('Trading Volume', fontweight='bold', fontsize=11)
    axes[1].set_ylabel('Volume'); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    # 3. MACD
    axes[2].plot(recent.index, recent['MACD'],        color='#1565C0', lw=1.3, label='MACD')
    axes[2].plot(recent.index, recent['MACD_Signal'], color='#FF5722', lw=1.3, label='Signal')
    axes[2].bar(recent.index, recent['MACD'] - recent['MACD_Signal'],
                color=np.where(recent['MACD'] > recent['MACD_Signal'], '#4CAF50', '#F44336'),
                alpha=0.5, label='Histogram')
    axes[2].axhline(0, color='black', lw=0.8, linestyle='--')
    axes[2].set_title('MACD Indicator', fontweight='bold', fontsize=11)
    axes[2].set_ylabel('MACD'); axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

    # 4. RSI
    axes[3].plot(recent.index, recent['RSI'], color='#7B1FA2', lw=1.5, label='RSI(14)')
    axes[3].axhline(70, color='red',   lw=1.2, linestyle='--', label='Overbought (70)')
    axes[3].axhline(30, color='green', lw=1.2, linestyle='--', label='Oversold (30)')
    axes[3].fill_between(recent.index, 30, 70, alpha=0.05, color='gray')
    axes[3].set_ylim(0, 100)
    axes[3].set_title('RSI — Relative Strength Index', fontweight='bold', fontsize=11)
    axes[3].set_ylabel('RSI'); axes[3].legend(fontsize=8); axes[3].grid(True, alpha=0.3)
    axes[3].set_xlabel('Date')

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.tick_params(axis='x', rotation=45, labelsize=8)

    plt.suptitle(f'📉 Technical Analysis — {STOCKS[selected_ticker]}',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # RSI Signal
    rsi_now = float(recent['RSI'].iloc[-1])
    if rsi_now > 70:
        st.warning(f"⚠️ RSI = {rsi_now:.1f} → Stock is **OVERBOUGHT** — potential pullback")
    elif rsi_now < 30:
        st.success(f"✅ RSI = {rsi_now:.1f} → Stock is **OVERSOLD** — potential buying opportunity")
    else:
        st.info(f"ℹ️ RSI = {rsi_now:.1f} → Stock is in **NEUTRAL** zone")

# ────────────────────────────────
# TAB 3: ML Prediction
# ────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">🤖 Machine Learning Predictions vs Actual</div>',
                unsafe_allow_html=True)

    with st.spinner("⏳ Training model(s)... Please wait."):
        feat_df = add_features(raw_df)
        lr_res, lstm_res = None, None

        if model_choice in ["Linear Regression", "Both (Compare)"]:
            lr_res = train_lr(feat_df)

        if model_choice in ["LSTM (Deep Learning)", "Both (Compare)"]:
            lstm_res = train_lstm_model(feat_df, epochs=lstm_epochs)

    # ── Metrics ─────────────────────────────────────────────
    if model_choice == "Both (Compare)":
        st.markdown("#### 📊 Model Metrics Comparison")
        mcol1, mcol2 = st.columns(2)

        with mcol1:
            st.markdown("**📐 Linear Regression**")
            m1c1, m1c2 = st.columns(2)
            m1c1.metric("RMSE", f"₹{lr_res['rmse']:.2f}")
            m1c2.metric("MAE",  f"₹{lr_res['mae']:.2f}")
            m1c1.metric("R² Score", f"{lr_res['r2']:.4f}")
            m1c2.metric("MAPE", f"{lr_res['mape']:.2f}%")

        with mcol2:
            st.markdown("**🧠 LSTM Deep Learning**")
            m2c1, m2c2 = st.columns(2)
            m2c1.metric("RMSE", f"₹{lstm_res['rmse']:.2f}",
                        delta=f"{lstm_res['rmse'] - lr_res['rmse']:+.2f} vs LR",
                        delta_color="inverse")
            m2c2.metric("MAE",  f"₹{lstm_res['mae']:.2f}")
            m2c1.metric("R² Score", f"{lstm_res['r2']:.4f}")
            m2c2.metric("MAPE", f"{lstm_res['mape']:.2f}%")

    elif lr_res:
        cols = st.columns(4)
        cols[0].metric("RMSE",      f"₹{lr_res['rmse']:.2f}")
        cols[1].metric("MAE",       f"₹{lr_res['mae']:.2f}")
        cols[2].metric("R² Score",  f"{lr_res['r2']:.4f}")
        cols[3].metric("MAPE",      f"{lr_res['mape']:.2f}%")

    elif lstm_res:
        cols = st.columns(4)
        cols[0].metric("RMSE",      f"₹{lstm_res['rmse']:.2f}")
        cols[1].metric("MAE",       f"₹{lstm_res['mae']:.2f}")
        cols[2].metric("R² Score",  f"{lstm_res['r2']:.4f}")
        cols[3].metric("MAPE",      f"{lstm_res['mape']:.2f}%")

    # ── Prediction Plot ──────────────────────────────────────
    n_plots = sum([lr_res is not None, lstm_res is not None])
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 6 * n_plots))
    if n_plots == 1: axes = [axes]
    ax_idx = 0

    if lr_res:
        axes[ax_idx].plot(lr_res['dates_te'], lr_res['y_te'],   label='Actual',    color='#1565C0', lw=1.8)
        axes[ax_idx].plot(lr_res['dates_te'], lr_res['y_pred'], label='LR Predict',color='#E53935', lw=1.6, linestyle='--')
        axes[ax_idx].set_title(f'📐 Linear Regression — Actual vs Predicted | {STOCKS[selected_ticker]}',
                               fontweight='bold', fontsize=12)
        axes[ax_idx].set_ylabel('Price (₹)'); axes[ax_idx].legend(); axes[ax_idx].grid(True, alpha=0.3)
        axes[ax_idx].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        axes[ax_idx].tick_params(axis='x', rotation=45)
        ax_idx += 1

    if lstm_res:
        axes[ax_idx].plot(lstm_res['dates_te'], lstm_res['y_te'],   label='Actual',      color='#1565C0', lw=1.8)
        axes[ax_idx].plot(lstm_res['dates_te'], lstm_res['y_pred'], label='LSTM Predict', color='#E91E63', lw=1.6, linestyle='--')
        axes[ax_idx].set_title(f'🧠 LSTM — Actual vs Predicted | {STOCKS[selected_ticker]}',
                               fontweight='bold', fontsize=12)
        axes[ax_idx].set_ylabel('Price (₹)'); axes[ax_idx].legend(); axes[ax_idx].grid(True, alpha=0.3)
        axes[ax_idx].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        axes[ax_idx].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ────────────────────────────────
# TAB 4: 30-Day Forecast
# ────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">🔮 Future Price Forecast</div>',
                unsafe_allow_html=True)

    if lr_res is None and lstm_res is None:
        st.info("👆 Please train a model in the 'ML Prediction' tab first.")
    else:
        last_date    = raw_df.index[-1]
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1),
                                      periods=forecast_days)

        fig, ax = plt.subplots(figsize=(14, 6))
        hist_recent = raw_df.last('180D')
        ax.plot(hist_recent.index, hist_recent['Close'],
                color='#1565C0', lw=2, label='Historical Price')
        ax.axvline(x=last_date, color='gray', linestyle=':', lw=2, alpha=0.8)
        ax.text(last_date, ax.get_ylim()[0], ' Today', fontsize=9, color='gray', va='bottom')

        if lr_res:
            fp = [raw_df['Close'].iloc[-1]] + lr_res['future_preds'][:forecast_days]
            fd = [last_date] + list(future_dates[:forecast_days])
            ax.plot(fd, fp, color='#E53935', lw=2, linestyle='--',
                    label=f'LR Forecast ({forecast_days}d)')
            ax.fill_between(fd, np.array(fp)*0.97, np.array(fp)*1.03,
                            alpha=0.1, color='#E53935')

        if lstm_res:
            fp2 = [raw_df['Close'].iloc[-1]] + lstm_res['future_preds'][:forecast_days]
            fd2 = [last_date] + list(future_dates[:forecast_days])
            ax.plot(fd2, fp2, color='#E91E63', lw=2, linestyle='-.',
                    label=f'LSTM Forecast ({forecast_days}d)')
            ax.fill_between(fd2, np.array(fp2)*0.97, np.array(fp2)*1.03,
                            alpha=0.1, color='#E91E63')

        ax.set_title(f'🔮 {STOCKS[selected_ticker]} — {forecast_days}-Day Price Forecast',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Date'); ax.set_ylabel('Price (₹)')
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        # Forecast table
        st.markdown("#### 📅 Day-by-Day Forecast")
        rows = []
        for i, date in enumerate(future_dates[:forecast_days]):
            row = {'Date': date.strftime('%d %b %Y')}
            if lr_res:
                row['LR Forecast (₹)'] = f"₹{lr_res['future_preds'][i]:.2f}"
            if lstm_res:
                row['LSTM Forecast (₹)'] = f"₹{lstm_res['future_preds'][i]:.2f}"
            rows.append(row)

        forecast_table = pd.DataFrame(rows)
        st.dataframe(forecast_table, use_container_width=True, height=300)

        # Summary
        curr_p = float(raw_df['Close'].iloc[-1])
        st.markdown("#### 🎯 Forecast Summary")
        sc1, sc2 = st.columns(2)
        if lr_res:
            fp_end = lr_res['future_preds'][forecast_days - 1]
            chg = ((fp_end - curr_p) / curr_p) * 100
            direction = "📈 BULLISH" if chg > 0 else "📉 BEARISH"
            sc1.markdown(f"""
            <div class='metric-card'>
                <b>📐 Linear Regression</b><br>
                Now: ₹{curr_p:.2f}<br>
                {forecast_days}-Day Target: <b>₹{fp_end:.2f}</b><br>
                <span class='{"forecast-up" if chg > 0 else "forecast-down"}'>
                    {direction} | {chg:+.2f}%
                </span>
            </div>
            """, unsafe_allow_html=True)

        if lstm_res:
            fp2_end = lstm_res['future_preds'][forecast_days - 1]
            chg2 = ((fp2_end - curr_p) / curr_p) * 100
            direction2 = "📈 BULLISH" if chg2 > 0 else "📉 BEARISH"
            sc2.markdown(f"""
            <div class='metric-card'>
                <b>🧠 LSTM Deep Learning</b><br>
                Now: ₹{curr_p:.2f}<br>
                {forecast_days}-Day Target: <b>₹{fp2_end:.2f}</b><br>
                <span class='{"forecast-up" if chg2 > 0 else "forecast-down"}'>
                    {direction2} | {chg2:+.2f}%
                </span>
            </div>
            """, unsafe_allow_html=True)

# ────────────────────────────────
# TAB 5: Model Report
# ────────────────────────────────
with tab5:
    st.markdown('<div class="section-title">📋 Model Report & Explanation</div>',
                unsafe_allow_html=True)

    st.markdown("""
    ### 📝 What is this project?
    This app predicts stock prices of **Top 10 Indian NSE-listed companies** using two machine
    learning approaches: **Linear Regression** (classical ML) and **LSTM** (Deep Learning).

    ---

    ### 🔍 How does it work?

    | Step | Description |
    |------|-------------|
    | **1. Data Fetch** | Live OHLCV data fetched from Yahoo Finance via `yfinance` |
    | **2. Feature Engineering** | Technical indicators: MA, EMA, MACD, RSI, Bollinger Bands, Lags |
    | **3. Preprocessing** | Min-Max normalization, train-test split (80/20, no shuffle) |
    | **4. Model Training** | Linear Regression or LSTM trained on historical data |
    | **5. Prediction** | Test-set predictions + recursive future forecasting |
    | **6. Evaluation** | RMSE, MAE, R², MAPE metrics |

    ---

    ### 📐 Linear Regression
    - Simple, fast, interpretable model
    - Assumes **linear relationship** between features and target
    - Works well for **short-term** and **trend-following** stocks
    - Features: 22 technical indicators as input variables

    ### 🧠 LSTM (Long Short-Term Memory)
    - Deep learning model designed for **time-series data**
    - Has **memory cells** to capture long-range dependencies
    - Architecture: 2 LSTM layers + 2 Dense layers + Dropout
    - Lookback window: 60 days
    - Loss function: Huber (robust to outliers)

    ---

    ### 📊 Evaluation Metrics

    | Metric | Formula | Meaning |
    |--------|---------|---------|
    | **RMSE** | √(Σ(actual-pred)²/n) | Average error in ₹ |
    | **MAE** | Σ\|actual-pred\|/n | Mean absolute error in ₹ |
    | **R²** | 1 - SS_res/SS_tot | % variance explained (1 = perfect) |
    | **MAPE** | Σ\|actual-pred\|/actual × 100 | % error |

    ---

    ### ⚠️ Limitations & Disclaimer
    - Stock markets are influenced by **news, events, and sentiment** — not captured in price data alone
    - Predictions are based on **historical patterns** and may not reflect future reality
    - This project is for **educational and research purposes ONLY**
    - **NOT financial advice** — do not use for actual investment decisions

    ---
    """)

    st.markdown("""
    <div class='disclaimer'>
    ⚠️ <b>IMPORTANT DISCLAIMER:</b> Stock market predictions carry significant uncertainty.
    This tool is built for academic learning only. Always consult a SEBI-registered financial
    advisor before making investment decisions. Past performance does not guarantee future results.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888; font-size:0.85rem;'>"
    "📈 Stock Prediction Dashboard | Python • yfinance • scikit-learn • TensorFlow • Streamlit<br>"
    "Data Source: Yahoo Finance | For Educational Purposes Only"
    "</p>",
    unsafe_allow_html=True
)
