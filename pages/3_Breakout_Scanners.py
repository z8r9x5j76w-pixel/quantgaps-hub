"""
QuantGaps Research — Multi-Strategy Hub
Three scanners in one Streamlit app:
  Tab 1 — Double Bottom v4.4
  Tab 2 — Mixed-100 Trend Breakout
  Tab 3 — TBB15 Triple-Bottom Basket
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import json, math
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# PASSWORD GATE
# ─────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #080d18; color: #dde6f0; }
.metric-card {
    background: #0f1825; border: 1px solid #1a2d42;
    border-radius: 10px; padding: 16px 20px; text-align: center;
}
.metric-val { font-size: 28px; font-weight: 700; font-family: monospace; }
.metric-lbl { font-size: 11px; color: #556070; letter-spacing: 2px;
              text-transform: uppercase; margin-top: 4px; }
.signal-tag {
    display: inline-block; padding: 2px 10px; border-radius: 4px;
    font-size: 12px; font-weight: 600; border: 1px solid;
}
div[data-testid="stTabs"] button { font-size: 15px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# STATE FILES  (use /tmp — survives restarts within session)
# ─────────────────────────────────────────────────────────────
TMP = Path("/tmp/quantgaps")
TMP.mkdir(exist_ok=True)

def load_state(name: str) -> dict:
    p = TMP / f"{name}_state.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"open": {}, "closed": []}

def save_state(name: str, state: dict):
    p = TMP / f"{name}_state.json"
    p.write_text(json.dumps(state, indent=2, default=str))

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def metric_card(label: str, value: str, color: str = "#00d4ff"):
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-val" style="color:{color}">{value}</div>'
        f'<div class="metric-lbl">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def color_ret(val):
    try:
        v = float(str(val).replace("%",""))
        return "color: #00ff9d" if v >= 0 else "color: #ff4d6d"
    except Exception:
        return ""

def styled_df(df: pd.DataFrame):
    """Return a styled dataframe with green/red on return columns."""
    ret_cols = [c for c in df.columns if any(k in c.lower() for k in ["ret","pct","return","pnl","cagr"])]
    s = df.style
    for c in ret_cols:
        try:
            s = s.applymap(color_ret, subset=[c])
        except Exception:
            pass
    return s.format(precision=2)


# ═════════════════════════════════════════════════════════════
#  STRATEGY 1 — DOUBLE BOTTOM v4.4
# ═════════════════════════════════════════════════════════════

DB_DEFAULT_TICKERS = [
    "SPY","QQQ","DIA","IWM","VTI","VOO","SMH","XLF","XLE","XLK",
    "XLV","XLI","XLP","XLY","XLU","AAPL","MSFT","GOOGL","AMZN","NVDA",
    "META","TSLA","BRK-B","AVGO","LLY","JPM","V","MA","UNH","HD",
    "COST","PG","KO","PEP","MRK","ABBV","ADBE","CRM","NFLX","AMD",
    "INTC","ORCL","CSCO","WMT","DIS","MCD","NKE","TMO","LIN","NEE",
    "RTX","HON","QCOM","TXN","AMAT","NOW","ISRG","XOM","CVX","CAT",
    "DE","LMT","BA","SLB","COP","GS","MS","BAC","WFC","BLK",
    "C","SHOP","SNOW","PLTR","CRWD","PANW","ZS","DDOG","MDB","COIN",
    "RIVN","MU","TSM","ASML","KLAC","LRCX","NXPI","ON","MDLZ","PM",
    "BTI","CL","GIS","KMB","PFE","JNJ","BMY","REGN","VRTX","MRNA",
    "SBUX","LOW","TGT","BKNG","ABNB",
]

LOOKBACK   = 180
MIN_SEP    = 7
MAX_SEP    = 120
TOL        = 0.06
DB_SL      = 0.03
DB_TP      = 0.08
DB_MAXHOLD = 20
DB_NOTIONAL= 2000.0
DB_MAX_POS = 10

def pivot_lows(series, left=2, right=2):
    x = series.values.astype(float)
    n = len(x)
    piv = np.zeros(n, dtype=bool)
    for i in range(left, n - right):
        w = x[i-left:i+right+1]
        if np.isfinite(x[i]) and x[i] == np.nanmin(w):
            piv[i] = True
    return piv

def find_best_double_bottom(window: pd.DataFrame):
    if len(window) < 60:
        return None
    lows  = window["Low"]
    highs = window["High"]
    piv   = pivot_lows(lows, 2, 2)
    idxs  = np.where(piv)[0]
    if len(idxs) < 2:
        return None
    best = None
    for a in range(len(idxs) - 1):
        i = idxs[a]
        for b in range(a + 1, len(idxs)):
            j = idxs[b]
            sep = j - i
            if sep < MIN_SEP: continue
            if sep > MAX_SEP: break
            low1 = float(lows.iloc[i])
            low2 = float(lows.iloc[j])
            if low1 <= 0 or low2 <= 0: continue
            avg = (low1 + low2) / 2.0
            if abs(low1 - low2) / avg > TOL: continue
            neckline = float(highs.iloc[i:j+1].max())
            if neckline <= 0: continue
            score = j - (abs(low1 - low2) / avg) * 1000.0
            if best is None or score > best[0]:
                best = (score, neckline)
    return {"neckline": best[1]} if best else None

def db_compute_signal(df: pd.DataFrame, end_pos: int):
    if end_pos <= 0:
        return None
    start  = max(0, end_pos - LOOKBACK + 1)
    window = df.iloc[start:end_pos + 1]
    patt   = find_best_double_bottom(window)
    if patt is None:
        return None
    neckline     = float(patt["neckline"])
    close_today  = float(df["Close"].iloc[end_pos])
    close_yday   = float(df["Close"].iloc[end_pos - 1])
    if not (close_yday <= neckline and close_today > neckline):
        return None
    strength = (close_today - neckline) / neckline
    return {"neckline": neckline, "strength": strength}

@st.cache_data(ttl=3600, show_spinner=False)
def db_run_scan(tickers_tuple):
    tickers = list(tickers_tuple)
    today   = datetime.utcnow().strftime("%Y-%m-%d")

    raw = yf.download(
        tickers, period="5y", interval="1d",
        group_by="ticker", progress=False, auto_adjust=False,
    )

    data_by_ticker = {}
    date_sets = []
    for t in tickers:
        try:
            if hasattr(raw.columns, "levels") and len(raw.columns.levels) > 1:
                if t not in raw.columns.levels[0]: continue
                df = raw[t].copy()
            else:
                df = raw.copy()
            df = df.dropna()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            needed = {"Open","High","Low","Close"}
            if not needed.issubset(set(df.columns)): continue
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            if len(df) < 200: continue
            data_by_ticker[t] = df[["Open","High","Low","Close"]]
            date_sets.append(set(df.index))
        except Exception:
            continue

    if not data_by_ticker or not date_sets:
        return [], [], {}, today

    common_dates = sorted(list(set.intersection(*date_sets)))
    dates = pd.DatetimeIndex(common_dates)

    # Backtest (simplified — just trades for stats)
    open_positions = {}
    trades = []
    pending_entries = {}

    for di in range(1, len(dates)):
        date      = dates[di]
        prev_date = dates[di - 1]

        if date in pending_entries:
            cands = pending_entries.pop(date)
            cands.sort(key=lambda x: x[1], reverse=True)
            for ticker, strength, neckline in cands:
                if ticker in open_positions: continue
                if len(open_positions) >= DB_MAX_POS: break
                df = data_by_ticker.get(ticker)
                if df is None or date not in df.index: continue
                o = float(df.loc[date, "Open"])
                if not np.isfinite(o) or o <= 0: continue
                open_positions[ticker] = {
                    "ticker": ticker, "entry_date": date,
                    "entry_price": o, "shares": DB_NOTIONAL / o,
                    "sl_price": o * (1 - DB_SL), "tp_price": o * (1 + DB_TP),
                    "days_held": 0,
                }

        to_close = []
        for ticker, pos in open_positions.items():
            df = data_by_ticker.get(ticker)
            if df is None or date not in df.index: continue
            bar   = df.loc[date]
            lo    = float(bar["Low"])
            hi    = float(bar["High"])
            close = float(bar["Close"])
            pos["days_held"] += 1
            exit_reason = exit_price = None
            if np.isfinite(lo) and lo <= pos["sl_price"]:
                exit_reason, exit_price = "SL", pos["sl_price"]
            elif np.isfinite(hi) and hi >= pos["tp_price"]:
                exit_reason, exit_price = "TP", pos["tp_price"]
            elif pos["days_held"] >= DB_MAXHOLD:
                exit_reason, exit_price = "MaxHold", close
            if exit_reason:
                pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                ret = (exit_price / pos["entry_price"] - 1.0) * 100.0
                trades.append({
                    "ticker": ticker,
                    "entry_date": str(pos["entry_date"].date()),
                    "exit_date":  str(date.date()),
                    "pnl":        round(float(pnl), 2),
                    "ret_pct":    round(float(ret), 2),
                    "reason":     exit_reason,
                })
                to_close.append(ticker)
        for t in to_close:
            open_positions.pop(t, None)

        if di < len(dates) - 1:
            next_date  = dates[di + 1]
            candidates = []
            for ticker, df in data_by_ticker.items():
                if ticker in open_positions: continue
                if date not in df.index or prev_date not in df.index: continue
                end_pos = df.index.get_loc(date)
                sig = db_compute_signal(df, end_pos)
                if sig:
                    candidates.append((ticker, sig["strength"], sig["neckline"]))
            if candidates:
                pending_entries.setdefault(next_date, []).extend(candidates)

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values(["exit_date","ticker"], ascending=[False,True])

    # Live signals from latest close
    latest = dates[-1]
    prev   = dates[-2] if len(dates) >= 2 else dates[-1]
    signals = []
    slots   = DB_MAX_POS - len(open_positions)
    if slots > 0:
        for ticker, df in data_by_ticker.items():
            if ticker in open_positions: continue
            if latest not in df.index or prev not in df.index: continue
            end_pos = df.index.get_loc(latest)
            sig = db_compute_signal(df, end_pos)
            if sig:
                close = float(df.loc[latest, "Close"])
                signals.append({
                    "ticker":        ticker,
                    "signal_date":   str(latest.date()),
                    "close":         round(close, 2),
                    "stop_loss":     round(close * (1 - DB_SL), 2),
                    "take_profit":   round(close * (1 + DB_TP), 2),
                    "max_hold":      DB_MAXHOLD,
                    "strength_pct":  round(sig["strength"] * 100.0, 2),
                })
        signals.sort(key=lambda x: x["strength_pct"], reverse=True)

    # Stats
    stats = {}
    if not trades_df.empty:
        rets = trades_df["ret_pct"].values / 100
        pnls = trades_df["pnl"].values
        cum  = np.cumsum(pnls)
        peak = np.maximum.accumulate(cum)
        dd   = (cum - peak) / (peak + DB_NOTIONAL * DB_MAX_POS + 1e-9)
        start_cap = DB_NOTIONAL * DB_MAX_POS
        total_ret = pnls.sum() / start_cap
        try:
            d0 = datetime.strptime(trades_df["entry_date"].min(), "%Y-%m-%d")
            d1 = datetime.strptime(trades_df["exit_date"].max(),  "%Y-%m-%d")
            years = max((d1 - d0).days / 365.25, 0.01)
        except Exception:
            years = 5
        stats = {
            "cagr":     round(((1 + total_ret) ** (1/years) - 1) * 100, 2),
            "max_dd":   round(float(dd.min()) * 100, 2),
            "n_trades": len(trades_df),
            "win_rate": round(float((trades_df["ret_pct"] > 0).mean() * 100), 1),
            "avg_ret":  round(float(trades_df["ret_pct"].mean()), 2),
        }

    return signals, trades_df.head(10).to_dict("records") if not trades_df.empty else [], stats, str(latest.date())


# ═════════════════════════════════════════════════════════════
#  STRATEGY 2 — MIXED-100 TREND BREAKOUT
# ═════════════════════════════════════════════════════════════

M100_ENTRY_LB   = 55
M100_EXIT_LB    = 20
M100_SL         = 0.20
M100_MAXHOLD    = 120
M100_ATR_PERIOD = 20
M100_MIN_ATR    = 0.015
M100_TOTAL_CAP  = 10000.0
M100_MAX_POS    = 10
M100_START_DATE = "2005-01-01"

M100_UNIVERSE = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","SPY","QQQ",
    "XOM","JPM","V","MA","HD","UNH","PG","BAC","COST","PEP","KO",
    "ABBV","LLY","MRK","CVX","WMT","CRM","ACN","MCD","NEE","TMO",
]

@dataclass
class M100Trade:
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    exit_reason: str
    position_usd: float = 0.0

    @property
    def ret_pct(self):
        return (self.exit_price - self.entry_price) / self.entry_price

def m100_add_atr(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(M100_ATR_PERIOD).mean()

def m100_download(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, start=M100_START_DATE, auto_adjust=True, progress=False)
    if df.empty: raise RuntimeError(f"No data for {ticker}")
    df = df.sort_index().dropna(how="any")
    # flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "open":   rename[c] = "Open"
        elif cl == "high":  rename[c] = "High"
        elif cl == "low":   rename[c] = "Low"
        elif cl == "close": rename[c] = "Close"
    df = df.rename(columns=rename)
    df = df[["Open","High","Low","Close"]].dropna()
    return df.astype(float)

def m100_prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["entry_high"] = df["Close"].shift(1).rolling(M100_ENTRY_LB).max()
    df["exit_low"]   = df["Close"].shift(1).rolling(M100_EXIT_LB).min()
    df["atr"]        = m100_add_atr(df)
    df["atr_pct"]    = df["atr"] / df["Close"]
    return df

def m100_trade_candidates(ticker: str) -> List[M100Trade]:
    try:
        df = m100_download(ticker)
    except Exception:
        return []
    dfb = m100_prepare(df)
    n = len(dfb)
    if n < max(M100_ENTRY_LB, M100_EXIT_LB) + M100_ATR_PERIOD + 5:
        return []
    dates      = dfb.index.to_list()
    open_      = dfb["Open"].to_list()
    close_     = dfb["Close"].to_list()
    entry_high = dfb["entry_high"].to_list()
    exit_low   = dfb["exit_low"].to_list()
    atr_pct    = dfb["atr_pct"].to_list()
    trades = []
    in_pos = False
    entry_idx = entry_date = entry_price = None
    i = 1
    while i < n:
        if not in_pos:
            si = i - 1
            eh = entry_high[si]; vp = atr_pct[si]
            if eh and not (isinstance(eh, float) and math.isnan(eh)):
                pv = M100_MIN_ATR <= 0.0 or (vp and not math.isnan(vp) and vp >= M100_MIN_ATR)
                if pv and close_[si] > float(eh):
                    entry_idx, entry_date, entry_price = i, pd.Timestamp(dates[i]), float(open_[i])
                    in_pos = True
        else:
            si = i - 1
            cur_date  = pd.Timestamp(dates[si])
            cur_close = close_[si]
            stop_hit  = cur_close <= entry_price * (1 - M100_SL)
            el = exit_low[si]
            brk_exit  = (el is not None and not math.isnan(el) and cur_close < float(el))
            days_held = (cur_date - entry_date).days
            mh_hit    = days_held >= M100_MAXHOLD
            reason    = "STOP_LOSS" if stop_hit else ("EXIT_BREAKOUT_LOW" if brk_exit else ("MAX_HOLD" if mh_hit else None))
            if reason:
                trades.append(M100Trade(
                    ticker=ticker, entry_date=entry_date,
                    exit_date=pd.Timestamp(dates[i]), entry_price=entry_price,
                    exit_price=float(open_[i]), exit_reason=reason,
                ))
                in_pos = False; entry_idx = entry_date = entry_price = None
        i += 1
    return trades

def m100_fresh_signals(tickers: List[str]) -> List[dict]:
    signals = []
    for ticker in tickers:
        try:
            df  = m100_download(ticker)
            if len(df) < M100_ENTRY_LB + M100_ATR_PERIOD + 5: continue
            dfb = m100_prepare(df)
            last = dfb.iloc[-1]; prev = dfb.iloc[-2]
            eh   = prev["entry_high"]; atrp = prev["atr_pct"]
            ct   = float(last["Close"])
            if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in [eh, atrp]): continue
            if M100_MIN_ATR > 0 and atrp < M100_MIN_ATR: continue
            if ct > float(eh):
                st_pct = (ct - float(eh)) / float(eh) * 100
                signals.append({
                    "ticker":        ticker,
                    "signal_date":   str(dfb.index[-1].date()),
                    "close":         round(ct, 2),
                    "stop_loss":     round(ct * (1 - M100_SL), 2),
                    "take_profit":   round(ct * (1 + M100_SL * 2.5), 2),
                    "max_hold_days": M100_MAXHOLD,
                    "strength_pct":  round(st_pct, 2),
                })
        except Exception:
            continue
    signals.sort(key=lambda x: x["strength_pct"], reverse=True)
    return signals

@st.cache_data(ttl=3600, show_spinner=False)
def m100_run_scan():
    tickers = M100_UNIVERSE
    signals = m100_fresh_signals(tickers)

    # Backtest
    all_candidates: List[M100Trade] = []
    for t in tickers:
        all_candidates.extend(m100_trade_candidates(t))

    if not all_candidates:
        return signals, [], {}, str(datetime.utcnow().date())

    all_candidates.sort(key=lambda x: (x.entry_date, x.exit_date, x.ticker))
    open_pos: List[M100Trade] = []
    completed = []
    cum_pnl = 0.0

    def close_up_to(date):
        nonlocal cum_pnl, open_pos
        still_open, to_close = [], []
        for p in open_pos:
            (to_close if p.exit_date <= date else still_open).append(p)
        for p in sorted(to_close, key=lambda x: x.exit_date):
            pnl = p.position_usd * p.ret_pct
            cum_pnl += pnl
            eq = M100_TOTAL_CAP + cum_pnl
            completed.append({
                "ticker": p.ticker,
                "entry_date": str(p.entry_date.date()),
                "exit_date":  str(p.exit_date.date()),
                "entry_price": round(p.entry_price, 2),
                "exit_price":  round(p.exit_price, 2),
                "exit_reason": p.exit_reason,
                "ret_pct":     round(p.ret_pct * 100, 2),
                "pnl_usd":     round(pnl, 2),
                "equity_after": round(eq, 2),
            })
        open_pos = still_open

    for t in all_candidates:
        close_up_to(t.entry_date)
        if len(open_pos) < M100_MAX_POS:
            eq = M100_TOTAL_CAP + cum_pnl
            t.position_usd = eq / M100_MAX_POS
            open_pos.append(t)
    close_up_to(pd.Timestamp.max)

    trades_df = pd.DataFrame(completed) if completed else pd.DataFrame()

    stats = {}
    if not trades_df.empty:
        eq = trades_df["equity_after"]
        final_eq = float(eq.iloc[-1])
        try:
            d0 = datetime.strptime(trades_df["entry_date"].min(), "%Y-%m-%d")
            d1 = datetime.strptime(trades_df["exit_date"].max(),  "%Y-%m-%d")
            years = max((d1 - d0).days / 365.25, 1e-9)
        except Exception:
            years = 15
        total_ret = final_eq / M100_TOTAL_CAP - 1.0
        cagr = ((1 + total_ret) ** (1 / years) - 1) * 100
        roll_max = eq.cummax()
        max_dd   = float((eq / roll_max - 1).min()) * 100
        stats = {
            "cagr":       round(cagr, 2),
            "final_eq":   round(final_eq, 2),
            "max_dd":     round(max_dd, 2),
            "n_trades":   len(trades_df),
            "win_rate":   round(float((trades_df["ret_pct"] > 0).mean() * 100), 1),
        }

    last5 = trades_df.tail(10).to_dict("records") if not trades_df.empty else []
    return signals, last5, stats, str(datetime.utcnow().date())


# ═════════════════════════════════════════════════════════════
#  STRATEGY 3 — TBB15 TRIPLE-BOTTOM BASKET
# ═════════════════════════════════════════════════════════════

TBB_UNIVERSE = [
    "NVDA","AMD","META","SPY","MSFT",
    "AAPL","GOOG","AMZN","XOM","CAT",
    "JPM","V","MA","COST","TSM",
]
TBB_ATR_PERIOD  = 14
TBB_SL_MULT     = 2.0
TBB_TP_MULT     = 4.0
TBB_MAXHOLD     = 60
TBB_MAX_POS     = 4
TBB_PRICE_TOL   = 0.03
TBB_MIN_SPACING = 5
TBB_BREAKOUT_BUF= 0.001
TBB_MAX_BARS    = 60
TBB_SWING_WIN   = 2
TBB_NOTIONAL    = 2000
TBB_SIG_YEARS   = 3
TBB_BT_YEARS    = 10

@dataclass
class TBBSignal:
    ticker: str; signal_date: str; entry: float; sl: float; tp: float
    sl_pct: float; atr: float; neckline: float; strength: float

def tbb_bulk_download(years: int) -> dict:
    end   = datetime.utcnow().date()
    start = end - timedelta(days=int(years * 365.25))
    raw = yf.download(
        tickers=TBB_UNIVERSE, start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        interval="1d", auto_adjust=False, progress=False,
    )
    result = {}
    for ticker in TBB_UNIVERSE:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                df = raw.xs(ticker, axis=1, level=1).copy()
            else:
                df = raw.copy()
            df = df.dropna(how="all").sort_index()
            df.index = pd.to_datetime(df.index)
            df.columns = [c.title() if isinstance(c,str) and c.islower() else c for c in df.columns]
            rename = {}
            for c in df.columns:
                if str(c).lower() in ["open","high","low","close","volume","adj close"]:
                    rename[c] = str(c).title()
            df = df.rename(columns=rename)
            needed = {"Open","High","Low","Close"}
            if not needed.issubset(df.columns): continue
            result[ticker] = df[list(needed)].astype(float)
        except Exception:
            continue
    return result

def tbb_add_atr(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(TBB_ATR_PERIOD).mean()
    return df

def tbb_swing_lows(df: pd.DataFrame) -> np.ndarray:
    lows = df["Low"].values
    n    = len(lows)
    sw   = np.zeros(n, dtype=bool)
    w    = TBB_SWING_WIN
    for i in range(w, n - w):
        if lows[i] == np.min(lows[i-w:i+w+1]) and np.isfinite(lows[i]):
            sw[i] = True
    return sw

def tbb_find_patterns(df: pd.DataFrame) -> List[int]:
    lows  = df["Low"].values
    highs = df["High"].values
    sw    = tbb_swing_lows(df)
    sl_idxs = np.where(sw)[0]
    if len(sl_idxs) < 3:
        return []
    breakouts = []
    for x in range(len(sl_idxs) - 2):
        i, j, k = sl_idxs[x], sl_idxs[x+1], sl_idxs[x+2]
        if (j - i) < TBB_MIN_SPACING or (k - j) < TBB_MIN_SPACING:
            continue
        li, lj, lk = lows[i], lows[j], lows[k]
        avg = (li + lj + lk) / 3
        if avg <= 0: continue
        if max(abs(li-lj), abs(lj-lk), abs(li-lk)) / avg > TBB_PRICE_TOL:
            continue
        neck = float(np.nanmax(highs[i:k+1]))
        for b in range(k+1, min(k+1+TBB_MAX_BARS, len(df))):
            if highs[b] > neck * (1 + TBB_BREAKOUT_BUF):
                breakouts.append(b)
                break
    return breakouts

def tbb_latest_signal(ticker: str, df: pd.DataFrame) -> Optional[TBBSignal]:
    df = tbb_add_atr(df)
    if len(df) < TBB_ATR_PERIOD + 20:
        return None
    last_idx  = len(df) - 1
    breakouts = tbb_find_patterns(df)
    if not breakouts or breakouts[-1] != last_idx:
        return None
    b_idx = breakouts[-1]
    atr = float(df["ATR"].iloc[b_idx])
    if math.isnan(atr) or atr <= 0:
        return None
    entry = float(df["Close"].iloc[b_idx])
    neck  = float(df["High"].iloc[:b_idx+1].max())
    sl    = round(entry - TBB_SL_MULT * atr, 2)
    tp    = round(entry + TBB_TP_MULT * atr, 2)
    sl_pct = round((entry - sl) / entry * 100, 2)
    strength = round((entry - neck) / neck * 100, 4) if neck > 0 else 0.0
    sig_date = str(df.index[b_idx])[:10]
    return TBBSignal(ticker, sig_date, round(entry,2), sl, tp, sl_pct,
                     round(atr,4), round(neck,2), strength)

def tbb_backtest_summary(data: dict) -> dict:
    all_Rs, all_pnls = [], []
    for ticker, df in data.items():
        df = tbb_add_atr(df)
        breakouts = tbb_find_patterns(df)
        if not breakouts: continue
        closes = df["Close"].values.astype(float)
        highs  = df["High"].values.astype(float)
        lows   = df["Low"].values.astype(float)
        atrs   = df["ATR"].values.astype(float)
        for b_idx in breakouts:
            if b_idx >= len(df): continue
            atr = atrs[b_idx]
            if np.isnan(atr) or atr <= 0: continue
            entry = closes[b_idx]
            sl    = entry - TBB_SL_MULT * atr
            tp    = entry + TBB_TP_MULT * atr
            risk  = entry - sl
            if risk <= 0: continue
            qty   = TBB_NOTIONAL / entry
            exit_R = exit_px = None
            last = min(len(df)-1, b_idx + TBB_MAXHOLD)
            for i in range(b_idx+1, last+1):
                if lows[i] <= sl:
                    exit_R, exit_px = (sl - entry) / risk, sl; break
                if highs[i] >= tp:
                    exit_R, exit_px = (tp - entry) / risk, tp; break
            if exit_R is None:
                exit_px = closes[last]
                exit_R  = (exit_px - entry) / risk
            all_Rs.append(float(exit_R))
            all_pnls.append((float(exit_px) - entry) * qty)
    if not all_Rs:
        return {"cagr": 0, "sharpe": 0, "max_dd": 0, "n_trades": 0, "win_rate": 0, "avg_R": 0}
    Rs  = np.array(all_Rs)
    pnl = np.array(all_pnls)
    cum  = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    dd   = (cum - peak) / (peak + TBB_NOTIONAL * TBB_MAX_POS + 1e-9)
    start_cap = TBB_NOTIONAL * TBB_MAX_POS
    total_ret = pnl.sum() / start_cap
    cagr = ((1 + total_ret) ** (1 / TBB_BT_YEARS) - 1) * 100
    sharpe = (Rs.mean() / Rs.std() * math.sqrt(252/15)) if Rs.std() > 0 else 0
    return {
        "cagr":     round(cagr, 2),
        "sharpe":   round(float(sharpe), 2),
        "max_dd":   round(float(dd.min()) * 100, 2),
        "n_trades": len(all_Rs),
        "win_rate": round(float((Rs > 0).mean() * 100), 1),
        "avg_R":    round(float(Rs.mean()), 2),
    }

@st.cache_data(ttl=3600, show_spinner=False)
def tbb_run_scan():
    today       = datetime.utcnow().strftime("%Y-%m-%d")
    signal_data = tbb_bulk_download(TBB_SIG_YEARS)
    bt_data     = tbb_bulk_download(TBB_BT_YEARS)
    state       = load_state("tbb15")

    # Update open positions
    exits_today = []
    still_open  = {}
    for ticker, pos in state["open"].items():
        if ticker not in signal_data:
            still_open[ticker] = pos; continue
        df  = signal_data[ticker]
        lo  = float(df["Low"].iloc[-1])
        hi  = float(df["High"].iloc[-1])
        cl  = float(df["Close"].iloc[-1])
        sl  = float(pos["sl"]); tp = float(pos["tp"])
        bars = int(pos.get("bars_held", 0)) + 1
        reason = None
        if lo <= sl:    exit_px, reason = sl, "SL"
        elif hi >= tp:  exit_px, reason = tp, "TP"
        elif bars >= TBB_MAXHOLD: exit_px, reason = cl, "MaxHold"
        if reason:
            entry_px = float(pos["entry"]); qty = float(pos["qty"])
            rec = {
                "ticker": ticker, "entry_date": pos["entry_date"],
                "exit_date": today, "pnl": round((exit_px-entry_px)*qty,2),
                "ret_pct": round((exit_px/entry_px-1)*100,2), "reason": reason,
            }
            exits_today.append(rec)
            state["closed"].append({**rec, "entry": pos["entry"], "exit": round(exit_px,2)})
        else:
            pos["bars_held"] = bars; still_open[ticker] = pos
    state["open"] = still_open
    save_state("tbb15", state)

    # New signals
    open_tickers = set(state["open"].keys())
    new_signals  = []
    for ticker in TBB_UNIVERSE:
        if ticker not in signal_data or ticker in open_tickers: continue
        if len(state["open"]) + len(new_signals) >= TBB_MAX_POS: break
        sig = tbb_latest_signal(ticker, signal_data[ticker])
        if sig:
            new_signals.append(sig)

    bt = tbb_backtest_summary(bt_data)
    return new_signals, exits_today, state, bt, today


# ═════════════════════════════════════════════════════════════
#  MAIN UI
# ═════════════════════════════════════════════════════════════
st.markdown("""
<div style="padding:20px 0 10px">
<div style="font-size:10px;letter-spacing:5px;color:#00d4ff;text-transform:uppercase">QuantGaps Research</div>
<div style="font-size:32px;font-weight:700;color:#ffffff;letter-spacing:-1px">Multi-Strategy Hub</div>
<div style="font-size:13px;color:#556070;margin-top:4px">Double Bottom · Mixed-100 Trend · TBB15 Triple-Bottom</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "📐 Double Bottom v4.4",
    "📈 Mixed-100 Trend Breakout",
    "🔱 TBB15 Triple-Bottom",
])

# ─────────────────────────────────────────────────────────────
# TAB 1 — DOUBLE BOTTOM
# ─────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Double Bottom Scanner v4.4")
    st.caption("Pivot-low pattern · SL 3% · TP 8% · MaxHold 20d")

    with st.expander("⚙️ Ticker Universe (edit and click Scan)", expanded=False):
        ticker_input = st.text_area(
            "One ticker per line",
            value="\n".join(DB_DEFAULT_TICKERS),
            height=200,
            key="db_tickers",
        )

    tickers_clean = tuple(sorted(set(
        t.strip().upper() for t in ticker_input.splitlines()
        if t.strip() and not t.strip().startswith("#")
    )))

    if st.button("▶ Run Double Bottom Scan", key="db_run", type="primary"):
        st.cache_data.clear()
        st.session_state["db_loaded"] = True

    if not st.session_state.get("db_loaded"):
        st.info("Click **▶ Run Double Bottom Scan** to start.")
        st.stop()

    with st.spinner("Downloading data and scanning…"):
        db_signals, db_last_trades, db_stats, db_asof = db_run_scan(tickers_clean)

    st.caption(f"As-of close: **{db_asof}** · Universe: {len(tickers_clean)} tickers")

    if db_stats:
        c1,c2,c3,c4 = st.columns(4)
        with c1: metric_card("CAGR (5y bt)", f"{db_stats['cagr']:.1f}%", "#00d4ff")
        with c2: metric_card("Win Rate",     f"{db_stats['win_rate']:.1f}%", "#00ff9d")
        with c3: metric_card("Max DD",       f"{db_stats['max_dd']:.1f}%", "#ff4d6d")
        with c4: metric_card("Trades (bt)",  str(db_stats['n_trades']), "#ffd166")
    st.markdown("")

    st.markdown("#### 🟢 Signals — Enter at Next Open")
    if db_signals:
        df_sig = pd.DataFrame(db_signals)
        df_sig.columns = [c.replace("_"," ").title() for c in df_sig.columns]
        st.dataframe(df_sig, use_container_width=True, hide_index=True)
    else:
        st.info("No double-bottom breakout signals on latest close.")

    st.markdown("#### 📋 Last 10 Closed Trades (backtest)")
    if db_last_trades:
        df_tr = pd.DataFrame(db_last_trades)
        df_tr.columns = [c.replace("_"," ").title() for c in df_tr.columns]
        st.dataframe(df_tr, use_container_width=True, hide_index=True)
    else:
        st.info("No closed trades yet.")


# ─────────────────────────────────────────────────────────────
# TAB 2 — MIXED-100
# ─────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Mixed-100 Trend Breakout")
    st.caption("55-day breakout · SL 20% hard · MaxHold 120 calendar days · 30-ticker universe")

    if st.button("▶ Run Mixed-100 Scan", key="m100_run", type="primary"):
        st.cache_data.clear()
        st.session_state["m100_loaded"] = True

    if not st.session_state.get("m100_loaded"):
        st.info("Click **▶ Run Mixed-100 Scan** to start. First run takes ~60 seconds.")
        st.stop()

    with st.spinner("Downloading history and running backtest… (this takes ~60s on first run)"):
        m100_signals, m100_trades, m100_stats, m100_asof = m100_run_scan()

    st.caption(f"As-of: **{m100_asof}** · Universe: {len(M100_UNIVERSE)} tickers")

    if m100_stats:
        c1,c2,c3,c4 = st.columns(4)
        with c1: metric_card("CAGR",       f"{m100_stats['cagr']:.1f}%", "#00d4ff")
        with c2: metric_card("Win Rate",   f"{m100_stats['win_rate']:.1f}%", "#00ff9d")
        with c3: metric_card("Max DD",     f"{m100_stats['max_dd']:.1f}%", "#ff4d6d")
        with c4: metric_card("Final Eq",   f"${m100_stats['final_eq']:,.0f}", "#ffd166")
    st.markdown("")

    st.markdown("#### 🟢 Signals — Enter at Next Open")
    if m100_signals:
        df_sig = pd.DataFrame(m100_signals)
        df_sig.columns = [c.replace("_"," ").title() for c in df_sig.columns]
        st.dataframe(df_sig, use_container_width=True, hide_index=True)
    else:
        st.info("No 55-day breakout signals on latest close.")

    st.markdown("#### 📋 Last 10 Closed Trades (backtest)")
    if m100_trades:
        df_tr = pd.DataFrame(m100_trades)
        df_tr.columns = [c.replace("_"," ").title() for c in df_tr.columns]
        st.dataframe(df_tr, use_container_width=True, hide_index=True)
    else:
        st.info("No closed trades yet.")


# ─────────────────────────────────────────────────────────────
# TAB 3 — TBB15
# ─────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### TBB15 ATR Triple-Bottom Basket")
    st.caption("Triple-bottom breakout · SL 2×ATR · TP 4×ATR · MaxHold 60d · 15 mega-caps")

    if st.button("▶ Run TBB15 Scan", key="tbb_run", type="primary"):
        st.cache_data.clear()
        st.session_state["tbb_loaded"] = True

    if not st.session_state.get("tbb_loaded"):
        st.info("Click **▶ Run TBB15 Scan** to start.")
        st.stop()

    with st.spinner("Scanning for triple-bottom breakouts…"):
        tbb_sigs, tbb_exits, tbb_state, tbb_bt, tbb_today = tbb_run_scan()

    st.caption(f"As-of: **{tbb_today}** · Universe: {', '.join(TBB_UNIVERSE)}")

    if tbb_bt:
        c1,c2,c3,c4 = st.columns(4)
        with c1: metric_card("CAGR (10y bt)", f"{tbb_bt['cagr']:.1f}%", "#00d4ff")
        with c2: metric_card("Win Rate",      f"{tbb_bt['win_rate']:.1f}%", "#00ff9d")
        with c3: metric_card("Max DD",        f"{tbb_bt['max_dd']:.1f}%", "#ff4d6d")
        with c4: metric_card("Avg R",         f"{tbb_bt['avg_R']:.2f}", "#ffd166")
    st.markdown("")

    st.markdown("#### 🟢 New Signals — Enter at Next Open")
    if tbb_sigs:
        rows = [{"Ticker": s.ticker, "Signal Date": s.signal_date,
                 "Entry": s.entry, "SL": s.sl, "TP": s.tp,
                 "SL %": f"{s.sl_pct:.2f}%", "ATR": s.atr,
                 "Neckline": s.neckline, "Strength %": s.strength}
                for s in tbb_sigs]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No fresh triple-bottom breakout signals today.")

    if tbb_exits:
        st.markdown("#### 🔴 Exits Today")
        st.dataframe(pd.DataFrame(tbb_exits), use_container_width=True, hide_index=True)

    col_open, col_closed = st.columns(2)
    with col_open:
        st.markdown("#### 📂 Open Positions")
        if tbb_state["open"]:
            rows = [{"Ticker": tk, "Entry Date": pos["entry_date"],
                     "Entry": float(pos["entry"]), "SL": float(pos["sl"]),
                     "TP": float(pos["tp"]), "Bars": int(pos.get("bars_held",0))}
                    for tk, pos in tbb_state["open"].items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No open positions.")

    with col_closed:
        st.markdown("#### 📋 Last 5 Closed")
        closed = tbb_state.get("closed", [])
        if closed:
            st.dataframe(pd.DataFrame(closed[-5:][::-1]), use_container_width=True, hide_index=True)
        else:
            st.info("No closed trades yet.")

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#556070;font-size:11px">'
    'QuantGaps Research · All signals hypothetical · Based on EOD data from Yahoo Finance · '
    'Not financial advice'
    '</div>',
    unsafe_allow_html=True,
)
