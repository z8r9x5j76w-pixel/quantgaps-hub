#!/usr/bin/env python3
"""
skewed_w_dashboard.py
======================
Skewed-W V5 — Daily Signal Dashboard
Strategy: SKEWED_W_V5_OOS_TOP60_MAX3_FORWARD_TEST

⚠️ RESEARCH / FORWARD-TEST ONLY. NOT FINANCIAL ADVICE.

Run locally:
    streamlit run skewed_w_dashboard.py

Deploy: push to a private GitHub repo, connect to Streamlit Community Cloud.

Architecture
------------
All scanner logic is embedded directly — no subprocess, no local files.
Data is fetched live from yfinance on each scan.
Open positions are stored in Streamlit persistent storage (survives sessions).
"""

from __future__ import annotations

import io
import json
import math
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import yfinance as yf
except ImportError:
    yf = None


# ══════════════════════════════════════════════════════════════════════════════
# PASSWORD
# ══════════════════════════════════════════════════════════════════════════════

def _check_password() -> bool:
    return True


# ══════════════════════════════════════════════════════════════════════════════
# DEFAULT UNIVERSE
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_TOP60 = [
    "GLW","MNST","AAPL","KMX","MAR","HSIC","JBHT","CAH","RVTY","MCHP",
    "SBAC","AMD","PODD","MELI","LKQ","ISRG","NTES","SMCI","MSFT","SKY",
    "TEL","HAL","WM","ACN","RMBS","IT","YUM","HST","TSM","FFIV",
    "SRPT","WAT","DRI","BG","DECK","NFLX","MRVL","TAP","AMAT","EIX",
    "AMGN","NVDA","BKR","AAON","PPG","HOLX","NDSN","DVN","KKR","PEG",
    "FTI","AMZN","BBY","TYL","NSC","EXPE","SBUX","HAS","ULTA","BIDU",
]

CLOSED_STATUSES = {"CLOSED","EXITED","STOPPED","TARGET","EXPIRED","IGNORE"}

# ══════════════════════════════════════════════════════════════════════════════
# SCANNER — embedded from skewed_w_six_pivot_v5_forward_safe_scanner.py
# Strict preset parameters (matching daily tool)
# ══════════════════════════════════════════════════════════════════════════════

# Strict preset values (same as --preset strict + --invalidate-on-close)
PARAMS = dict(
    pivot_left              = 6,
    pivot_right             = 6,
    min_pivot_gap           = 8,
    min_total_bars          = 40,
    max_total_bars          = 420,
    max_leg_bars            = 160,
    min_leg_pct             = 3.0,
    min_drop_pct            = 12.0,
    min_high_decline_pct    = 1.5,
    min_low_decline_pct     = 1.5,
    min_higher_low_pct      = 1.5,
    min_pullback_pct        = 2.0,
    max_pullback_pct        = 40.0,
    max_h3_above_h2_pct     = 6.0,
    max_h3_below_h2_pct     = 20.0,
    max_subsequence_span    = 10,
    top_patterns_per_l3     = 1,
    entry_buffer_pct        = 0.10,
    require_breakout_close  = False,
    max_breakout_wait       = 126,
    max_entry_gap_pct       = 3.0,
    invalidate_below_l3     = True,
    invalidate_on_close     = True,
    pre_break_invalid_pct   = 0.25,
    pre_break_invalid_atr   = 0.05,
    sl_atr_buffer           = 0.05,
    sl_pct_buffer           = 0.25,
    rr                      = 2.0,
    max_hold                = 20,
    atr_n                   = 14,
    near_breakout_pct       = 5.0,
    lookback_days           = 126,
)


@dataclass
class Pivot:
    idx:   int
    date:  pd.Timestamp
    kind:  str
    price: float


@dataclass
class Pattern:
    ticker: str
    h1_idx: int;  l1_idx: int;  h2_idx: int
    l2_idx: int;  h3_idx: int;  l3_idx: int
    h1_date: pd.Timestamp; l1_date: pd.Timestamp; h2_date: pd.Timestamp
    l2_date: pd.Timestamp; h3_date: pd.Timestamp; l3_date: pd.Timestamp
    h1_price: float; l1_price: float; h2_price: float
    l2_price: float; h3_price: float; l3_price: float
    quality_score: float
    drop_h1_l2_pct: float; h2_below_h1_pct: float; l2_below_l1_pct: float
    h3_vs_h2_pct: float;   l3_above_l2_pct: float; pullback_h3_l3_pct: float
    bars_h1_l3: int


def _clean_download(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        if len(df.columns.levels[0]) == 1:
            df.columns = df.columns.droplevel(0)
        elif len(df.columns.levels[-1]) == 1:
            df.columns = df.columns.droplevel(-1)
    df = df.rename(columns={c: str(c).strip().lower().replace(" ","_") for c in df.columns})
    required = ["open","high","low","close"]
    if any(c not in df.columns for c in required):
        return pd.DataFrame()
    out = df[required + (["volume"] if "volume" in df.columns else [])].dropna(subset=required).copy()
    out.index = pd.to_datetime(out.index)
    out = out[~out.index.duplicated(keep="last")]
    return out


@st.cache_data(ttl=300, show_spinner=False)
def _download(ticker: str, start: str) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    try:
        df = yf.download(ticker, start=start, auto_adjust=False, progress=False, threads=False)
        return _clean_download(df)
    except Exception:
        return pd.DataFrame()


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = PARAMS["atr_n"]
    prev = df["close"].shift(1)
    tr = pd.concat([df["high"]-df["low"],
                    (df["high"]-prev).abs(),
                    (df["low"]-prev).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(n, min_periods=max(2, n//2)).mean()
    return df


def _find_pivots(df: pd.DataFrame) -> List[Pivot]:
    left  = PARAMS["pivot_left"]
    right = PARAMS["pivot_right"]
    highs = df["high"].to_numpy(float)
    lows  = df["low"].to_numpy(float)
    dates = df.index
    pivots: List[Pivot] = []
    n = len(df)
    for i in range(left, n - right):
        h = highs[i]; l = lows[i]
        wh = highs[i-left:i+right+1]; wl = lows[i-left:i+right+1]
        if np.isfinite(h) and h >= np.nanmax(wh) and h > np.nanmax(np.r_[highs[i-left:i], highs[i+1:i+right+1]]):
            pivots.append(Pivot(i, dates[i], "H", float(h)))
        if np.isfinite(l) and l <= np.nanmin(wl) and l < np.nanmin(np.r_[lows[i-left:i], lows[i+1:i+right+1]]):
            pivots.append(Pivot(i, dates[i], "L", float(l)))
    pivots.sort(key=lambda p: (p.idx, 0 if p.kind=="H" else 1))
    # compress
    out: List[Pivot] = []
    for p in pivots:
        if not out: out.append(p); continue
        last = out[-1]
        if p.kind == last.kind:
            if p.kind=="H" and p.price > last.price: out[-1] = p
            elif p.kind=="L" and p.price < last.price: out[-1] = p
        else:
            out.append(p)
    return out


def _leg_pct(p1: Pivot, p2: Pivot) -> float:
    return abs(p2.price/p1.price - 1.0)*100.0 if p1.price > 0 else np.nan

def _pct_down(h: float, l: float) -> float:
    return (h-l)/h*100.0 if h > 0 else np.nan

def _pct_up(h: float, l: float) -> float:
    return (h-l)/l*100.0 if l > 0 else np.nan

def _pct_chg(a: float, b: float) -> float:
    return (b/a-1.0)*100.0 if a > 0 else np.nan


def _pattern_quality(h1,l1,h2,l2,h3,l3) -> Tuple[float, dict]:
    dh1l2   = _pct_down(h1.price, l2.price)
    h2bh1   = _pct_down(h1.price, h2.price)
    l2bl1   = _pct_down(l1.price, l2.price)
    h3vh2   = _pct_chg(h2.price, h3.price)
    l3al2   = _pct_up(l3.price, l2.price)
    pb      = _pct_down(h3.price, l3.price)
    drop_s  = min(dh1l2/25.0, 2.0)
    lh_s    = min(max(h2bh1,0)/8.0, 1.5)
    ll_s    = min(max(l2bl1,0)/8.0, 1.5)
    hl_s    = min(max(l3al2,0)/12.0, 2.0)
    pb_s    = min(max(pb,0)/10.0, 1.5)
    h3_s    = 1.0 if h3vh2 >= -8.0 else 0.5
    dur_s   = min((l3.idx-h1.idx)/160.0, 2.0)
    pen     = 1.0
    if pb < 2.0:        pen *= 0.65
    if pb > 40.0:       pen *= 0.75
    if l3al2 < 1.5:     pen *= 0.70
    if (l3.idx-h1.idx) < 40: pen *= 0.75
    q = (drop_s+lh_s+ll_s+hl_s+pb_s+h3_s+dur_s)*pen
    return float(q), dict(drop_h1_l2_pct=dh1l2,h2_below_h1_pct=h2bh1,
                          l2_below_l1_pct=l2bl1,h3_vs_h2_pct=h3vh2,
                          l3_above_l2_pct=l3al2,pullback_h3_l3_pct=pb)


def _detect_patterns(ticker: str, df: pd.DataFrame) -> List[Pattern]:
    P = PARAMS
    pivots = _find_pivots(df)
    n = len(pivots)
    candidates: List[Pattern] = []
    span = P["max_subsequence_span"]
    for i in range(n):
        if pivots[i].kind != "H": continue
        end = min(n, i+span)
        for j1 in range(i+1,end):
            if pivots[j1].kind!="L": continue
            for j2 in range(j1+1,end):
                if pivots[j2].kind!="H": continue
                for j3 in range(j2+1,end):
                    if pivots[j3].kind!="L": continue
                    for j4 in range(j3+1,end):
                        if pivots[j4].kind!="H": continue
                        for j5 in range(j4+1,end):
                            if pivots[j5].kind!="L": continue
                            h1,l1,h2,l2,h3,l3 = pivots[i],pivots[j1],pivots[j2],pivots[j3],pivots[j4],pivots[j5]
                            idxs = [h1.idx,l1.idx,h2.idx,l2.idx,h3.idx,l3.idx]
                            gaps = [idxs[k+1]-idxs[k] for k in range(5)]
                            if any(g < P["min_pivot_gap"] for g in gaps): continue
                            if any(g > P["max_leg_bars"] for g in gaps): continue
                            total = l3.idx-h1.idx
                            if not (P["min_total_bars"] <= total <= P["max_total_bars"]): continue
                            if any(_leg_pct(a,b) < P["min_leg_pct"] for a,b in [(h1,l1),(l1,h2),(h2,l2),(l2,h3),(h3,l3)]): continue
                            if _pct_down(h1.price,h2.price) < P["min_high_decline_pct"]: continue
                            if _pct_down(l1.price,l2.price) < P["min_low_decline_pct"]: continue
                            if _pct_down(h1.price,l2.price) < P["min_drop_pct"]: continue
                            if _pct_up(l3.price,l2.price) < P["min_higher_low_pct"]: continue
                            pb = _pct_down(h3.price,l3.price)
                            if not (P["min_pullback_pct"] <= pb <= P["max_pullback_pct"]): continue
                            if l3.price >= h3.price: continue
                            if h3.price >= h1.price: continue
                            h3v = _pct_chg(h2.price,h3.price)
                            if h3v > P["max_h3_above_h2_pct"]: continue
                            if h3v < -P["max_h3_below_h2_pct"]: continue
                            q,comps = _pattern_quality(h1,l1,h2,l2,h3,l3)
                            candidates.append(Pattern(
                                ticker=ticker,
                                h1_idx=h1.idx,l1_idx=l1.idx,h2_idx=h2.idx,
                                l2_idx=l2.idx,h3_idx=h3.idx,l3_idx=l3.idx,
                                h1_date=h1.date,l1_date=l1.date,h2_date=h2.date,
                                l2_date=l2.date,h3_date=h3.date,l3_date=l3.date,
                                h1_price=h1.price,l1_price=l1.price,h2_price=h2.price,
                                l2_price=l2.price,h3_price=h3.price,l3_price=l3.price,
                                quality_score=q,**comps,bars_h1_l3=total,
                            ))
    by_h3l3: Dict[Tuple[int,int],Pattern] = {}
    for p in candidates:
        key = (p.h3_idx,p.l3_idx)
        if key not in by_h3l3 or p.quality_score > by_h3l3[key].quality_score:
            by_h3l3[key] = p
    by_l3: Dict[int,List[Pattern]] = {}
    for p in by_h3l3.values():
        by_l3.setdefault(p.l3_idx,[]).append(p)
    final: List[Pattern] = []
    for _,ps in by_l3.items():
        ps = sorted(ps,key=lambda x:x.quality_score,reverse=True)[:1]
        final.extend(ps)
    return sorted(final, key=lambda p:(p.l3_idx,p.h3_idx))


def _invalid_level(df: pd.DataFrame, p: Pattern) -> float:
    P = PARAMS
    atr_l3 = float(df["atr"].iloc[p.l3_idx]) if "atr" in df.columns else np.nan
    if not np.isfinite(atr_l3) or atr_l3 <= 0:
        atr_l3 = p.l3_price * 0.02
    buf = max(p.l3_price * P["pre_break_invalid_pct"]/100.0,
              atr_l3 * P["pre_break_invalid_atr"])
    return p.l3_price - buf


def _estimated_levels(df: pd.DataFrame, p: Pattern, trigger: float, atr_idx: int) -> Tuple[float,float,float,float]:
    P = PARAMS
    atr = float(df["atr"].iloc[atr_idx]) if "atr" in df.columns and atr_idx < len(df) else np.nan
    if not np.isfinite(atr) or atr <= 0:
        atr = p.l3_price * 0.02
    stop = p.l3_price - max(atr*P["sl_atr_buffer"], p.l3_price*P["sl_pct_buffer"]/100.0)
    risk = trigger - stop
    target = trigger + risk * P["rr"] if risk > 0 else trigger * 1.10
    risk_pct = (risk/trigger*100.0) if trigger > 0 else np.nan
    return float(stop), float(target), float(risk), float(risk_pct)


def _classify_pattern(ticker: str, df: pd.DataFrame, p: Pattern) -> dict:
    P = PARAMS
    latest_idx   = len(df) - 1
    latest_close = float(df["close"].iloc[latest_idx])
    trigger      = float(p.h3_price * (1.0 + P["entry_buffer_pct"]/100.0))
    dist         = float((trigger/latest_close - 1.0)*100.0) if latest_close > 0 else np.nan
    inv_level    = _invalid_level(df, p)

    wait_end = min(latest_idx, p.l3_idx + P["max_breakout_wait"])
    first_trigger_idx: Optional[int] = None
    trigger_gap_large = False
    invalidated_idx:  Optional[int] = None

    for i in range(p.l3_idx+1, wait_end+1):
        row = df.iloc[i]
        if P["invalidate_below_l3"]:
            val = float(row["close"] if P["invalidate_on_close"] else row["low"])
            if val < inv_level:
                invalidated_idx = i; break
        triggered = bool(row["close"] >= trigger) if P["require_breakout_close"] else bool(row["high"] >= trigger)
        if not triggered: continue
        open_p = float(row["open"])
        entry  = max(open_p, trigger)
        gap    = (entry/trigger - 1.0)*100.0
        trigger_gap_large = gap > P["max_entry_gap_pct"]
        first_trigger_idx = i
        break

    level_atr_idx = first_trigger_idx if first_trigger_idx is not None else latest_idx
    stop_e, target_e, risk_e, risk_pct_e = _estimated_levels(df, p, trigger, level_atr_idx)

    status = ""; entry_date = None; entry_price = np.nan; max_hold_until = None; actionable = False

    if invalidated_idx is not None:
        status = "invalidated_before_breakout"
    elif first_trigger_idx is not None:
        row = df.iloc[first_trigger_idx]
        entry_price = max(float(row["open"]), trigger)
        entry_date  = df.index[first_trigger_idx].date()
        mh_idx      = min(latest_idx, first_trigger_idx + P["max_hold"])
        max_hold_until = df.index[mh_idx].date()
        if trigger_gap_large:
            status = "triggered_but_gap_too_large"
        elif first_trigger_idx == latest_idx:
            status = "fresh_signal_today"; actionable = True
        elif latest_idx <= first_trigger_idx + P["max_hold"]:
            status = "already_triggered_in_hold_window"
        else:
            status = "expired_or_already_traded"
    elif latest_idx > wait_end:
        status = "expired_no_breakout"
    elif np.isfinite(dist) and 0.0 <= dist <= P["near_breakout_pct"]:
        status = "near_breakout"; actionable = True
    elif np.isfinite(dist) and dist > P["near_breakout_pct"]:
        status = "forming_below_breakout"; actionable = True
    else:
        status = "inactive_unclassified_above_breakout"

    return dict(
        ticker=ticker, status=status, actionable=actionable,
        latest_date=df.index[latest_idx].date(), latest_close=latest_close,
        breakout_level=round(trigger,3), distance_to_breakout_pct=round(float(dist),2) if np.isfinite(dist) else None,
        entry_date=str(entry_date) if entry_date else None,
        entry_price_est=round(float(entry_price),3) if np.isfinite(entry_price) else round(trigger,3),
        stop_price_est=round(stop_e,3), target_price_est=round(target_e,3),
        risk_pct_est=round(risk_pct_e,2) if np.isfinite(risk_pct_e) else None,
        max_hold_until=str(max_hold_until) if max_hold_until else None,
        h3_price=round(p.h3_price,3), l3_price=round(p.l3_price,3),
        h3_date=str(p.h3_date.date()), l3_date=str(p.l3_date.date()),
        quality_score=round(p.quality_score,3),
        bars_since_l3=int(latest_idx - p.l3_idx),
    )


def _scan_ticker(ticker: str) -> Tuple[Optional[dict], Optional[dict]]:
    """Return (active_row, inactive_row) for a single ticker."""
    df = _download(ticker, "2000-01-01")
    if df.empty or len(df) < 300:
        return None, None
    df = _add_indicators(df)
    patterns = _detect_patterns(ticker, df)
    if not patterns:
        return None, None

    latest_date = df.index[-1]
    recent = [p for p in patterns if (latest_date - p.l3_date).days <= PARAMS["lookback_days"]]
    if not recent:
        return None, None

    rows = [_classify_pattern(ticker, df, p) for p in recent]

    active_order = {"fresh_signal_today":0,"near_breakout":1,"forming_below_breakout":2}
    active_rows   = [r for r in rows if r["status"] in active_order]
    inactive_rows = [r for r in rows if r["status"] not in active_order]

    active_best = None
    if active_rows:
        active_rows.sort(key=lambda r: (
            active_order[r["status"]],
            abs(r["distance_to_breakout_pct"]) if r["distance_to_breakout_pct"] is not None else 999,
            -r["quality_score"],
        ))
        active_best = active_rows[0]

    inactive_best = None
    if inactive_rows:
        inactive_rows.sort(key=lambda r: (r["l3_date"], r["quality_score"]), reverse=True)
        inactive_best = inactive_rows[0]

    return active_best, inactive_best


# ══════════════════════════════════════════════════════════════════════════════
# OPEN POSITIONS — persistent storage
# ══════════════════════════════════════════════════════════════════════════════

POSITIONS_KEY = "sw_open_positions"

OP_COLS = ["ticker","entry_date","entry_price","stop_price","target_price","max_exit_date","status","notes"]


def _load_positions() -> pd.DataFrame:
    try:
        result = st.context.get(POSITIONS_KEY)
        if result and result.get("value"):
            data = json.loads(result["value"])
            return pd.DataFrame(data) if data else pd.DataFrame(columns=OP_COLS)
    except Exception:
        pass
    return pd.DataFrame(columns=OP_COLS)


def _save_positions(df: pd.DataFrame) -> None:
    try:
        st.context.set(POSITIONS_KEY, json.dumps(df.to_dict("records")))
    except Exception:
        pass


def _active_positions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "status" not in df.columns:
        df["status"] = "OPEN"
    df["status"] = df["status"].astype(str).str.upper().str.strip()
    active = df[~df["status"].isin(CLOSED_STATUSES)].copy()
    if "max_exit_date" in active.columns:
        active["max_exit_date_dt"] = pd.to_datetime(active["max_exit_date"], errors="coerce")
        today = pd.Timestamp.today().normalize()
        expired = active["max_exit_date_dt"].notna() & (active["max_exit_date_dt"] < today)
        active.loc[expired, "status"] = "⚠️ NEEDS EXIT (EXPIRED)"
    return active.drop(columns=["max_exit_date_dt"], errors="ignore")


# ══════════════════════════════════════════════════════════════════════════════
# SCAN — full universe
# ══════════════════════════════════════════════════════════════════════════════

def run_scan(tickers: List[str], max_open: int, active_count: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scan all tickers, classify setups, return (signals_df, inactive_df).
    Signals: ENTRY_TODAY / WATCH / BLOCKED / IGNORE
    """
    active_rows, inactive_rows = [], []
    slots_remaining = max(0, max_open - active_count)

    progress = st.progress(0, text="Starting scan…")
    n = len(tickers)

    for i, ticker in enumerate(tickers):
        progress.progress((i+1)/n, text=f"Scanning {ticker} ({i+1}/{n})…")
        try:
            active, inactive = _scan_ticker(ticker)
        except Exception:
            active, inactive = None, None

        if inactive:
            inactive_rows.append(inactive)

        if active is None:
            continue

        st_val = active["status"]
        if st_val == "fresh_signal_today":
            if slots_remaining > 0:
                active["signal_bucket"] = "ENTRY_TODAY"
                active["entry_allowed"] = True
                active["decision_reason"] = f"Fresh breakout; {slots_remaining} slot(s) available"
                slots_remaining -= 1
            else:
                active["signal_bucket"] = "BLOCKED"
                active["entry_allowed"] = False
                active["decision_reason"] = f"Fresh breakout but max_open={max_open} reached"
        elif st_val in {"near_breakout","forming_below_breakout"}:
            active["signal_bucket"] = "WATCH"
            active["entry_allowed"] = False
            active["decision_reason"] = "Watch — no fresh breakout yet"
        else:
            active["signal_bucket"] = "IGNORE"
            active["entry_allowed"] = False
            active["decision_reason"] = f"Non-actionable: {st_val}"

        active_rows.append(active)

    progress.empty()

    signals_df = pd.DataFrame(active_rows) if active_rows else pd.DataFrame()
    inactive_df = pd.DataFrame(inactive_rows) if inactive_rows else pd.DataFrame()

    if not signals_df.empty:
        order = {"ENTRY_TODAY":0,"WATCH":1,"BLOCKED":2,"IGNORE":3}
        signals_df["_s"] = signals_df["signal_bucket"].map(order).fillna(9)
        if "distance_to_breakout_pct" in signals_df.columns:
            signals_df["_d"] = pd.to_numeric(signals_df["distance_to_breakout_pct"], errors="coerce").abs().fillna(999)
        else:
            signals_df["_d"] = 0
        signals_df = signals_df.sort_values(["_s","_d"]).drop(columns=["_s","_d"]).reset_index(drop=True)

    return signals_df, inactive_df


# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

BG    = "#0E1117"
GREEN = "#26C281"
RED   = "#FF4B4B"
AMBER = "#FFB300"
CYAN  = "#00C4FF"
GRID  = "#2D2D2D"

BUCKET_COLORS = {
    "ENTRY_TODAY": GREEN,
    "WATCH":       CYAN,
    "BLOCKED":     RED,
    "IGNORE":      "#555",
}

STATUS_ICONS = {
    "fresh_signal_today":           "🟢",
    "near_breakout":                "🔵",
    "forming_below_breakout":       "⚪",
    "invalidated_before_breakout":  "🔴",
    "already_triggered_in_hold_window": "🟡",
    "expired_or_already_traded":    "⚫",
    "expired_no_breakout":          "⚫",
}


def _badge(text: str, color: str) -> str:
    return (f"<span style='background:{color};color:#000;font-weight:700;"
            f"padding:3px 10px;border-radius:6px;font-size:0.85em'>{text}</span>")


def render_signal_card(row: dict, context: str = "tab"):
    bucket = row.get("signal_bucket","IGNORE")
    color  = BUCKET_COLORS.get(bucket, "#555")
    icon   = STATUS_ICONS.get(row.get("status",""), "⚪")
    ticker = row.get("ticker","?")
    dist   = row.get("distance_to_breakout_pct")
    dist_s = f"{dist:+.1f}%" if dist is not None else "n/a"

    st.markdown(f"""
    <div style='background:#161b22;border-radius:8px;padding:12px 16px;
                border-left:5px solid {color};margin-bottom:8px'>
      <b style='color:{color};font-size:1.15em'>{ticker}</b>
      &nbsp; {icon} &nbsp;
      <span style='color:#aaa'>{row.get("status","")}</span>
      &nbsp;&nbsp;
      {_badge(bucket, color)}
      <span style='float:right;color:#888;font-size:0.9em'>
        dist: {dist_s} &nbsp;|&nbsp; Q: {row.get("quality_score","n/a")}
      </span>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Close",        f"{row.get('latest_close','n/a')}")
    c2.metric("Breakout lvl", f"{row.get('breakout_level','n/a')}")
    c3.metric("Est. stop",    f"{row.get('stop_price_est','n/a')}")
    c4.metric("Est. target",  f"{row.get('target_price_est','n/a')}")
    c5.metric("Risk %",       f"{row.get('risk_pct_est','n/a')}")

    if bucket == "ENTRY_TODAY":
        st.success(
            f"**⚡ ENTRY TODAY — {ticker}**  "
            f"Entry on next session open above **{row.get('breakout_level')}**. "
            f"Max hold until {row.get('max_hold_until','n/a')}. "
            f"Est. stop {row.get('stop_price_est')} | Est. target {row.get('target_price_est')}. "
            f"⚠️ Recalculate stop/target from actual entry open price."
        )

    with st.expander(f"Pattern details — {ticker}", expanded=False):
        st.markdown(f"""
        | Field | Value |
        |---|---|
        | H3 neckline | {row.get('h3_price')} ({row.get('h3_date')}) |
        | L3 support  | {row.get('l3_price')} ({row.get('l3_date')}) |
        | Bars since L3 | {row.get('bars_since_l3')} |
        | Entry date  | {row.get('entry_date') or 'not yet triggered'} |
        | Max hold until | {row.get('max_hold_until') or 'n/a'} |
        | Decision    | {row.get('decision_reason','')} |
        """)


def render_open_positions(all_pos: pd.DataFrame, max_open: int):
    active = _active_positions(all_pos.copy()) if not all_pos.empty else pd.DataFrame(columns=OP_COLS)
    count  = len(active)

    color = GREEN if count < max_open else RED
    st.markdown(f"""
    <div style='background:#12122A;border-radius:8px;padding:10px 16px;
                border-left:4px solid {color};margin-bottom:12px'>
      <b style='color:{color}'>Open positions: {count} / {max_open}</b>
      &nbsp;&nbsp;
      <span style='color:#666;font-size:0.85em'>Slots available: {max(0, max_open-count)}</span>
    </div>""", unsafe_allow_html=True)

    if not active.empty:
        display_cols = [c for c in OP_COLS if c in active.columns]
        st.dataframe(active[display_cols], use_container_width=True, hide_index=True)
    else:
        st.caption("No active open positions.")

    st.markdown("##### Add / update position")
    with st.form("add_position_form"):
        fc1,fc2,fc3 = st.columns(3)
        new_ticker    = fc1.text_input("Ticker").upper().strip()
        new_entry_d   = fc2.text_input("Entry date (YYYY-MM-DD)")
        new_entry_p   = fc3.number_input("Entry price", min_value=0.0, format="%.4f")
        fc4,fc5,fc6   = st.columns(3)
        new_stop      = fc4.number_input("Stop price", min_value=0.0, format="%.4f")
        new_target    = fc5.number_input("Target price", min_value=0.0, format="%.4f")
        new_max_exit  = fc6.text_input("Max exit date (YYYY-MM-DD)")
        fn1,fn2       = st.columns(2)
        new_status    = fn1.selectbox("Status", ["OPEN","CLOSED","STOPPED","TARGET","EXPIRED"])
        new_notes     = fn2.text_input("Notes")
        submitted = st.form_submit_button("Save position")
        if submitted and new_ticker:
            new_row = dict(ticker=new_ticker, entry_date=new_entry_d,
                           entry_price=new_entry_p, stop_price=new_stop,
                           target_price=new_target, max_exit_date=new_max_exit,
                           status=new_status, notes=new_notes)
            if all_pos.empty:
                all_pos = pd.DataFrame([new_row])
            else:
                # Update if ticker already exists as open, otherwise append
                mask = (all_pos["ticker"].astype(str).str.upper() == new_ticker) & \
                       (~all_pos.get("status","OPEN").astype(str).str.upper().isin(CLOSED_STATUSES))
                if mask.any():
                    for k,v in new_row.items():
                        all_pos.loc[mask, k] = v
                else:
                    all_pos = pd.concat([all_pos, pd.DataFrame([new_row])], ignore_index=True)
            _save_positions(all_pos)
            st.success(f"Saved: {new_ticker}")
            st.rerun()

    return active


# ══════════════════════════════════════════════════════════════════════════════
# CSV EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def _to_csv(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():


    st.markdown("""
    <style>
      .stApp{background:#0E1117}
      .block-container{padding-top:1rem}
    </style>""", unsafe_allow_html=True)

    st.title("〽️ Skewed-W V5 — Daily Signal Dashboard")
    st.caption("⚠️ **RESEARCH / FORWARD-TEST ONLY. NOT FINANCIAL ADVICE. NOT PRODUCTION-READY.**  "
               "Strategy: SKEWED_W_V5_OOS_TOP60_MAX3_FORWARD_TEST")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        max_open = st.number_input("Max open positions", min_value=1, max_value=10, value=3)
        st.markdown("---")
        st.markdown("### 📋 Universe")
        universe_text = st.text_area(
            "Tickers (one per line or comma-separated)",
            value="\n".join(DEFAULT_TOP60),
            height=200,
        )
        st.markdown("---")
        run_scan_btn = st.button("▶ Run Scan", use_container_width=True, type="primary")
        if st.button("🗑 Clear cache & rescan", use_container_width=True):
            st.cache_data.clear()
            st.session_state.pop("scan_results", None)
            st.session_state.pop("scan_inactive", None)
            st.session_state.pop("scan_time", None)
            st.rerun()
        st.caption("First scan: ~2-4 min for 60 tickers. Cached 5 min after that.")

    # Parse universe
    raw = universe_text.replace(",","\n")
    tickers = [t.strip().upper() for t in raw.splitlines() if t.strip() and not t.startswith("#")]
    tickers = list(dict.fromkeys(tickers))  # deduplicate preserving order

    # Load persistent open positions
    all_positions = _load_positions()
    active_pos    = _active_positions(all_positions.copy()) if not all_positions.empty else pd.DataFrame(columns=OP_COLS)
    active_count  = len(active_pos)

    # Run scan
    if run_scan_btn or "scan_results" not in st.session_state:
        with st.spinner(f"Scanning {len(tickers)} tickers…"):
            signals, inactive = run_scan(tickers, max_open, active_count)
        st.session_state["scan_results"] = signals
        st.session_state["scan_inactive"] = inactive
        st.session_state["scan_time"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    signals:  pd.DataFrame = st.session_state.get("scan_results", pd.DataFrame())
    inactive: pd.DataFrame = st.session_state.get("scan_inactive", pd.DataFrame())
    scan_time = st.session_state.get("scan_time","—")

    # ── Summary banner ────────────────────────────────────────────────────────
    n_entry   = 0 if signals.empty else int((signals["signal_bucket"]=="ENTRY_TODAY").sum())
    n_watch   = 0 if signals.empty else int((signals["signal_bucket"]=="WATCH").sum())
    n_blocked = 0 if signals.empty else int((signals["signal_bucket"]=="BLOCKED").sum())

    st.markdown(f"**Last scan:** {scan_time} &nbsp;|&nbsp; "
                f"**Tickers:** {len(tickers)} &nbsp;|&nbsp; "
                f"**Open positions:** {active_count}/{max_open}")

    if n_entry:
        entry_tickers = signals[signals["signal_bucket"]=="ENTRY_TODAY"]["ticker"].tolist()
        st.success(f"🚨 **{n_entry} ENTRY TODAY**: {', '.join(entry_tickers)}")
    if n_watch:
        st.info(f"👁 {n_watch} watchlist setup(s)")
    if not n_entry and not n_watch:
        st.info("⚪ No actionable setups today.")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tabs = st.tabs(["🚨 Entry Today", "👁 Watchlist", "🔴 Blocked",
                    "📋 Open Positions", "📊 All Signals", "⬇️ Export"])

    # Tab 0 — Entry today
    with tabs[0]:
        st.subheader("🚨 Entry Today")
        st.warning("⚠️ Entry is next session OPEN above the breakout level. "
                   "Recalculate stop/target from the actual open price before placing the trade.")
        entries = signals[signals["signal_bucket"]=="ENTRY_TODAY"] if not signals.empty else pd.DataFrame()
        if entries.empty:
            st.info("No fresh breakouts today.")
        else:
            for _, row in entries.iterrows():
                render_signal_card(row.to_dict(), context="entry")

    # Tab 1 — Watchlist
    with tabs[1]:
        st.subheader("👁 Watchlist — Near / Forming")
        watches = signals[signals["signal_bucket"]=="WATCH"] if not signals.empty else pd.DataFrame()
        if watches.empty:
            st.info("No watchlist setups.")
        else:
            for _, row in watches.iterrows():
                render_signal_card(row.to_dict(), context="watch")

    # Tab 2 — Blocked
    with tabs[2]:
        st.subheader("🔴 Blocked — Max positions reached")
        blocked = signals[signals["signal_bucket"]=="BLOCKED"] if not signals.empty else pd.DataFrame()
        if blocked.empty:
            st.info("No blocked setups.")
        else:
            st.warning(f"These {len(blocked)} setup(s) had fresh signals today but "
                       f"are blocked because max_open={max_open} is already reached.")
            for _, row in blocked.iterrows():
                render_signal_card(row.to_dict(), context="blocked")

    # Tab 3 — Open positions
    with tabs[3]:
        st.subheader("📋 Open Positions")
        render_open_positions(all_positions, int(max_open))

    # Tab 4 — All signals table
    with tabs[4]:
        st.subheader("📊 All Signals — Summary Table")
        if signals.empty:
            st.info("Run scan first.")
        else:
            display_cols = [c for c in [
                "signal_bucket","ticker","status","latest_close","breakout_level",
                "distance_to_breakout_pct","stop_price_est","target_price_est",
                "risk_pct_est","quality_score","bars_since_l3","max_hold_until",
            ] if c in signals.columns]
            st.dataframe(signals[display_cols], use_container_width=True, hide_index=True)

    # Tab 5 — Export
    with tabs[5]:
        st.subheader("⬇️ Export CSVs")
        c1,c2,c3,c4 = st.columns(4)
        today_s = datetime.utcnow().strftime("%Y%m%d")

        c1.download_button("ENTRY_TODAY.csv",
            _to_csv(signals[signals["signal_bucket"]=="ENTRY_TODAY"] if not signals.empty else pd.DataFrame()),
            f"ENTRY_TODAY_{today_s}.csv", "text/csv")
        c2.download_button("WATCHLIST.csv",
            _to_csv(signals[signals["signal_bucket"]=="WATCH"] if not signals.empty else pd.DataFrame()),
            f"WATCHLIST_{today_s}.csv", "text/csv")
        c3.download_button("ALL_SIGNALS.csv",
            _to_csv(signals if not signals.empty else pd.DataFrame()),
            f"ALL_SIGNALS_{today_s}.csv", "text/csv")
        c4.download_button("OPEN_POSITIONS.csv",
            _to_csv(all_positions),
            f"OPEN_POSITIONS_{today_s}.csv", "text/csv")


if __name__ == "__main__":
    main()
