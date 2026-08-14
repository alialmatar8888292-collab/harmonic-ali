import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(page_title="داش بورد الهارمونيك", layout="wide", page_icon="📈")

st.title("📊 داش بورد نماذج الهارمونيك والشارت التفاعلي")

# شريط التحكم الجانبي
symbol = st.sidebar.text_input("رمز السهم/الأصل (Ticker):", value="SPY")
timeframe = st.sidebar.selectbox("الإطار الزمني:", ["1d", "1h", "15m", "5m"], index=0)
period = st.sidebar.selectbox("الفترة:", ["1mo", "3mo", "6mo", "1y"], index=1)

@st.cache_data(ttl=60)
def get_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

try:
    df = get_data(symbol, period, timeframe)
    
    st.subheader("🔔 تنبيهات نماذج الهارمونيك")
    st.info("ℹ️ يتم معالجة البيانات واكتشاف النماذج سحابياً...")
    
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=symbol
    )])
    
    fig.update_layout(
        template="plotly_dark",
        height=600,
        title=f"الشارت المباشر - {symbol.upper()}",
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"يرجى التأكد من الرمز المدخل: {e}")
