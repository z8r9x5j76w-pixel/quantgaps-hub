"""
QuantGaps Research — Home
Entry point: password gate + navigation to the two scanner apps.
"""

import streamlit as st

st.set_page_config(
    page_title="QuantGaps Research",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Password gate ──────────────────────────────────────────────
def check_password() -> bool:
    if st.session_state.get("_authenticated"):
        return True
    st.markdown("""
    <style>
    .stApp { background-color: #080d18; color: #dde6f0; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("## 🔒 QuantGaps Research Hub")
    pw = st.text_input("Password", type="password", key="pw_home")
    if st.button("Enter"):
        try:
            correct = st.secrets["DASHBOARD_PASSWORD"]
        except Exception:
            correct = "quantgaps2024"
        if pw == correct:
            st.session_state["_authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False

if not check_password():
    st.stop()

# ── Authenticated landing ──────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #080d18; color: #dde6f0; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="padding: 20px 0 10px">
  <div style="font-size:10px;letter-spacing:5px;color:#00d4ff;text-transform:uppercase">QuantGaps Research</div>
  <div style="font-size:36px;font-weight:700;color:#ffffff;letter-spacing:-1px">Trading Dashboard Hub</div>
  <div style="font-size:14px;color:#556070;margin-top:6px">Select a scanner from the sidebar →</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style="background:#0f1825;border:1px solid #1a2d42;border-radius:12px;padding:24px;">
      <div style="color:#00d4ff;font-size:13px;letter-spacing:2px;text-transform:uppercase">Page 1</div>
      <div style="font-size:22px;font-weight:700;color:#fff;margin:8px 0">📡 Pattern Scanner</div>
      <div style="color:#556070;font-size:13px;line-height:1.6">
        3-candle pattern strategies (DUD, DDD, DD, DU)<br>
        IWM · CRM · PYPL · UPS · COF · NTR · ALB<br>
        ATR-based exits · Backtest metrics · Rankings tab
      </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background:#0f1825;border:1px solid #1a2d42;border-radius:12px;padding:24px;">
      <div style="color:#00ff9d;font-size:13px;letter-spacing:2px;text-transform:uppercase">Page 2</div>
      <div style="font-size:22px;font-weight:700;color:#fff;margin:8px 0">📈 Breakout Scanners</div>
      <div style="color:#556070;font-size:13px;line-height:1.6">
        Double Bottom v4.4 (105 tickers)<br>
        Mixed-100 Trend Breakout (55-day breakout)<br>
        TBB15 Triple-Bottom Basket (15 mega-caps)
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")
st.info("👈 Use the sidebar to navigate between scanners. All data is from Yahoo Finance (EOD). Not financial advice.")
