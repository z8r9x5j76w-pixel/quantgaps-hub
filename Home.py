"""
QuantGaps Research — Home
Entry point: password gate + navigation to all scanner pages.
"""

import streamlit as st

st.set_page_config(
    page_title="QuantGaps Research",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

def check_password() -> bool:
    if st.session_state.get("_authenticated"):
        return True
    st.markdown("<style>.stApp { background-color: #080d18; color: #dde6f0; }</style>", unsafe_allow_html=True)
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

st.markdown("<style>.stApp { background-color: #080d18; color: #dde6f0; }</style>", unsafe_allow_html=True)

st.markdown("""
<div style="padding: 20px 0 10px">
  <div style="font-size:10px;letter-spacing:5px;color:#00d4ff;text-transform:uppercase">QuantGaps Research</div>
  <div style="font-size:36px;font-weight:700;color:#ffffff;letter-spacing:-1px">Trading Dashboard Hub</div>
  <div style="font-size:14px;color:#556070;margin-top:6px">Select a scanner from the sidebar →</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

cards = [
    ("📡", "Pattern Scanner",        "#00d4ff", "IWM · CRM · PYPL · UPS · COF · NTR · ALB\nDUD/DDD/DD/DU candle patterns · ATR exits · Rankings"),
    ("📈", "Breakout Scanners",       "#00ff9d", "Double Bottom v4.4 (105 tickers)\nMixed-100 Trend Breakout · TBB15 Triple-Bottom"),
    ("🧪", "Momentum Sequence Lab",   "#ffd166", "Multi-ticker momentum sequence strategies\nForward-test dashboard · Validation metrics"),
    ("〽️", "Skewed-W Scanner",        "#ff9f43", "Skewed-W V5 pattern · OOS Top-60\nOpen position tracker · Persistent state"),
    ("🔬", "Signal Scanner",          "#ee5a24", "22 tech tickers: FTNT, PANW, CRWD, MU, TSM\nAMD, DDOG, COIN, PLTR, UPST and more"),
]

cols = st.columns(len(cards))
for col, (icon, title, color, desc) in zip(cols, cards):
    with col:
        st.markdown(f"""
        <div style="background:#0f1825;border:1px solid #1a2d42;border-radius:12px;padding:20px;height:160px">
          <div style="color:{color};font-size:22px">{icon}</div>
          <div style="font-size:16px;font-weight:700;color:#fff;margin:6px 0">{title}</div>
          <div style="color:#556070;font-size:12px;line-height:1.6">{desc.replace(chr(10),'<br>')}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")
st.info("👈 Use the sidebar to navigate. All data from Yahoo Finance (EOD). Not financial advice.")
