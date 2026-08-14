import sys
import subprocess
try:
    import yfinance as yf
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                            "yfinance", "plotly", "pandas", "numpy", "scipy"])
    import yfinance as yf

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

# ---------------------------------------------------------------------------
# إعداد الصفحة + دعم اتجاه RTL
# ---------------------------------------------------------------------------
st.set_page_config(page_title="داش بورد الهارمونيك", layout="wide", page_icon="📈")
st.markdown("""
<style>
html, body, [class*="css"] { direction: rtl; text-align: right; }
.stSidebar { direction: rtl; text-align: right; }
</style>
""", unsafe_allow_html=True)

st.title("📊 داش بورد نماذج الهارمونيك والشارت التفاعلي")

# ---------------------------------------------------------------------------
# تعريف نسب فيبوناتشي لكل نموذج هارمونيك (XABCD)
# كل نموذج معرف بمجالات مقبولة لثلاث نسب: XAB, ABC (retracement من AB), BCD, XAD
# ---------------------------------------------------------------------------
TOL = 0.06  # هامش تسامح حول النسبة (6%)

def rng(target, tol=TOL):
    return (target * (1 - tol), target * (1 + tol))

HARMONIC_PATTERNS = {
    "Gartley (جارتلي)": {
        "XAB": [rng(0.618)],
        "ABC": [(0.382, 0.886)],
        "BCD": [rng(1.272), rng(1.618)],
        "XAD": [rng(0.786)],
    },
    "Bat (الخفاش)": {
        "XAB": [(0.382, 0.50)],
        "ABC": [(0.382, 0.886)],
        "BCD": [rng(1.618), rng(2.618)],
        "XAD": [rng(0.886)],
    },
    "Butterfly (الفراشة)": {
        "XAB": [rng(0.786)],
        "ABC": [(0.382, 0.886)],
        "BCD": [rng(1.618), rng(2.24)],
        "XAD": [(1.27, 1.618)],
    },
    "Crab (السرطان)": {
        "XAB": [(0.382, 0.618)],
        "ABC": [(0.382, 0.886)],
        "BCD": [rng(2.24), rng(3.618)],
        "XAD": [rng(1.618)],
    },
    "Deep Crab (السرطان العميق)": {
        "XAB": [rng(0.886)],
        "ABC": [(0.382, 0.886)],
        "BCD": [rng(2.0), rng(3.618)],
        "XAD": [rng(1.618)],
    },
    "Shark (القرش)": {
        "XAB": [(0.446, 0.618)],
        "ABC": [(1.13, 1.618)],
        "BCD": [rng(1.618), rng(2.24)],
        "XAD": [(0.886, 1.13)],
    },
    "Cypher (السايفر)": {
        "XAB": [(0.382, 0.618)],
        "ABC": [(1.13, 1.414)],
        "BCD": [rng(0.786)],  # هنا تُقاس على XC وليس AB
        "XAD": [rng(0.786)],
    },
    "AB=CD": {
        "XAB": [(0.0, 5.0)],       # لا يشترط نسبة XAB محددة
        "ABC": [(0.382, 0.886)],
        "BCD": [rng(1.272), rng(1.618)],
        "XAD": [(0.0, 5.0)],
    },
}

# ---------------------------------------------------------------------------
# اكتشاف نقاط الارتكاز (Pivots / ZigZag)
# ---------------------------------------------------------------------------
def get_pivots(df, order=5):
    highs = df['High'].values
    lows = df['Low'].values
    hi_idx = argrelextrema(highs, np.greater_equal, order=order)[0]
    lo_idx = argrelextrema(lows, np.less_equal, order=order)[0]

    pivots = []
    for i in hi_idx:
        pivots.append((i, highs[i], 'H'))
    for i in lo_idx:
        pivots.append((i, lows[i], 'L'))
    pivots.sort(key=lambda x: x[0])

    # تنظيف: إزالة القمم/القيعان المتتالية من نفس النوع (نحتفظ بالأقصى/الأدنى)
    cleaned = []
    for p in pivots:
        if cleaned and cleaned[-1][2] == p[2]:
            if p[2] == 'H' and p[1] > cleaned[-1][1]:
                cleaned[-1] = p
            elif p[2] == 'L' and p[1] < cleaned[-1][1]:
                cleaned[-1] = p
        else:
            cleaned.append(p)
    return cleaned


def ratio(a, b):
    return abs(b) / abs(a) if a != 0 else np.nan


def in_any_range(value, ranges):
    return any(lo <= value <= hi for lo, hi in ranges)


# ---------------------------------------------------------------------------
# فحص كل نافذة من 5 نقاط ارتكاز متتالية (X-A-B-C-D) مقابل كل نموذج هارمونيك
# ---------------------------------------------------------------------------
def detect_harmonics(pivots):
    found = []
    for i in range(len(pivots) - 4):
        X, A, B, C, D = pivots[i:i+5]
        # يجب أن تتناوب بين قمة وقاع (X,B,D من نفس النوع و A,C من النوع الآخر)
        types = [p[2] for p in (X, A, B, C, D)]
        if not (types[0] != types[1] and types[1] != types[2] and
                types[2] != types[3] and types[3] != types[4]):
            continue

        XA = A[1] - X[1]
        AB = B[1] - A[1]
        BC = C[1] - B[1]
        CD = D[1] - C[1]
        XD = D[1] - X[1]
        XC = C[1] - X[1]

        xab = ratio(XA, AB)
        abc = ratio(AB, BC)
        bcd = ratio(BC, CD)
        xad = ratio(XA, XD)
        xac = ratio(XA, XC)  # يستخدم لنموذج Cypher بدل XAD القياسي

        for name, rules in HARMONIC_PATTERNS.items():
            xab_ok = in_any_range(xab, rules["XAB"])
            abc_ok = in_any_range(abc, rules["ABC"])
            bcd_ok = in_any_range(bcd, rules["BCD"])
            if name == "Cypher (السايفر)":
                xad_ok = in_any_range(xac, rules["XAD"])
            else:
                xad_ok = in_any_range(xad, rules["XAD"])

            if xab_ok and abc_ok and bcd_ok and xad_ok:
                direction = "صاعد (Bullish)" if D[2] == 'L' else "هابط (Bearish)"
                found.append({
                    "pattern": name,
                    "direction": direction,
                    "X": X, "A": A, "B": B, "C": C, "D": D,
                    "D_index": D[0],
                    "completion_price": D[1],
                })
    return found


# ---------------------------------------------------------------------------
# إرسال تنبيه بالإيميل
# ---------------------------------------------------------------------------
def send_email_alert(smtp_user, smtp_pass, to_addr, subject, body):
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = to_addr
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, [to_addr], msg.as_string())
        return True, None
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# الشريط الجانبي
# ---------------------------------------------------------------------------
symbol = st.sidebar.text_input("رمز السهم/الأصل (Ticker):", value="SPY")
timeframe = st.sidebar.selectbox("الإطار الزمني:", ["1d", "1h", "15m", "5m"], index=0)
period = st.sidebar.selectbox("الفترة:", ["1mo", "3mo", "6mo", "1y"], index=1)
pivot_order = st.sidebar.slider("حساسية اكتشاف نقاط الارتكاز (Pivot Order)", 2, 15, 5,
                                 help="رقم أصغر = نقاط ارتكاز أكثر وحساسية أعلى")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ إعدادات نماذج الهارمونيك")
selected_patterns = st.sidebar.multiselect(
    "النماذج المفعّلة:",
    list(HARMONIC_PATTERNS.keys()),
    default=list(HARMONIC_PATTERNS.keys())
)

st.sidebar.markdown("---")
st.sidebar.subheader("📧 تنبيهات الإيميل")
enable_email = st.sidebar.checkbox("تفعيل إرسال تنبيه إيميل عند اكتشاف نموذج جديد")
smtp_user = st.sidebar.text_input("بريد المرسل (Gmail):", type="default", disabled=not enable_email)
smtp_pass = st.sidebar.text_input("App Password (وليس كلمة المرور العادية):", type="password", disabled=not enable_email)
to_addr = st.sidebar.text_input("بريد الاستقبال:", disabled=not enable_email)

if enable_email:
    st.sidebar.caption(
        "⚠️ استخدم App Password من إعدادات جوجل (Google Account → Security → App Passwords)، "
        "وليس كلمة مرور حسابك العادية."
    )

st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("تحديث تلقائي كل 60 ثانية", value=False)
if auto_refresh:
    st.markdown("<meta http-equiv='refresh' content='60'>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# جلب البيانات
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60)
def get_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

# تتبّع النماذج التي تم إرسال تنبيه لها بالفعل (لتفادي التكرار خلال نفس الجلسة)
if "alerted_keys" not in st.session_state:
    st.session_state.alerted_keys = set()

try:
    df = get_data(symbol, period, timeframe)
    if df.empty:
        st.error("لا توجد بيانات لهذا الرمز/الفترة. تأكد من الرمز المدخل.")
        st.stop()

    pivots = get_pivots(df, order=pivot_order)
    all_matches = detect_harmonics(pivots)
    matches = [m for m in all_matches if m["pattern"] in selected_patterns]

    st.subheader("🔔 تنبيهات نماذج الهارمونيك")

    if matches:
        for m in matches:
            key = f"{symbol}-{m['pattern']}-{m['D_index']}-{m['completion_price']:.4f}"
            d_time = df.index[m["D_index"]]
            st.success(
                f"✅ **{m['pattern']}** | الاتجاه: {m['direction']} | "
                f"نقطة D عند {m['completion_price']:.2f} بتاريخ {d_time}"
            )

            if enable_email and key not in st.session_state.alerted_keys:
                if smtp_user and smtp_pass and to_addr:
                    subject = f"تنبيه هارمونيك: {m['pattern']} على {symbol}"
                    body = (
                        f"تم اكتشاف نموذج {m['pattern']} ({m['direction']}) على {symbol}\n"
                        f"الإطار الزمني: {timeframe}\n"
                        f"نقطة اكتمال النموذج (D): {m['completion_price']:.2f}\n"
                        f"الوقت: {d_time}\n"
                        f"وقت الإرسال: {datetime.now()}"
                    )
                    ok, err = send_email_alert(smtp_user, smtp_pass, to_addr, subject, body)
                    if ok:
                        st.session_state.alerted_keys.add(key)
                        st.info(f"📧 تم إرسال تنبيه بالإيميل عن نموذج {m['pattern']}")
                    else:
                        st.warning(f"⚠️ فشل إرسال الإيميل: {err}")
                else:
                    st.warning("⚠️ فعّل الإيميل لكن الحقول ناقصة (المرسل / App Password / المستقبل)")
    else:
        st.info("ℹ️ لا توجد نماذج هارمونيك مكتملة حاليًا ضمن النطاق المعروض.")

    # -----------------------------------------------------------------------
    # الشارت
    # -----------------------------------------------------------------------
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name=symbol
    )])

    colors = ["#00E5FF", "#FFD700", "#FF6EC7", "#7CFC00", "#FF4500", "#00FA9A", "#BA55D3", "#FFA500"]
    for idx, m in enumerate(matches):
        pts = [m["X"], m["A"], m["B"], m["C"], m["D"]]
        xs = [df.index[p[0]] for p in pts]
        ys = [p[1] for p in pts]
        labels = ["X", "A", "B", "C", "D"]
        color = colors[idx % len(colors)]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers+text",
            text=labels, textposition="top center",
            line=dict(color=color, width=2, dash="dot"),
            marker=dict(size=8, color=color),
            name=f"{m['pattern']} ({m['direction']})"
        ))

    fig.update_layout(
        template="plotly_dark",
        height=650,
        title=f"الشارت المباشر - {symbol.upper()}",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 كل النماذج المكتشفة (بما فيها غير المفعّلة حاليًا)"):
        if all_matches:
            table = pd.DataFrame([{
                "النموذج": m["pattern"],
                "الاتجاه": m["direction"],
                "سعر D": round(m["completion_price"], 2),
                "تاريخ D": df.index[m["D_index"]],
            } for m in all_matches])
            st.dataframe(table, use_container_width=True)
        else:
            st.write("لا يوجد.")

except Exception as e:
    st.error(f"يرجى التأكد من الرمز المدخل: {e}")
