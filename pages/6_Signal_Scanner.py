#!/usr/bin/env python3
"""
signal_scanner_dashboard_20260516.py
======================================
Streamlit web conversion of signal_dashboardV16.py
22 ticker scanners: FTNT, SWKS, ON, PANW, CRWD, MU, TSM, AMD, MRVL,
                    DDOG, NET, OKTA, BILL, U, ROKU, TWLO, DOCU, COIN,
                    PLTR, UPST, PATH, RBLX

Run locally:
    streamlit run dashboard.py

⚠️ RESEARCH ONLY. NOT FINANCIAL ADVICE.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import requests
except ImportError:
    requests = None


# ── PASSWORD GATE ─────────────────────────────────────────────────────────────
# Change the string below to set your password.

def _check_password() -> bool:
    return True


# ── DATA ──────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_ohlc(ticker: str, bars: int = 120) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance not installed")
    raw = yf.download(ticker, period=f"{bars}d", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    df = raw[["Open", "High", "Low", "Close"]].dropna().reset_index()
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    return df.sort_values("Date").reset_index(drop=True)


# ── SIGNAL RESULT ─────────────────────────────────────────────────────────────

@dataclass
class SignalResult:
    ticker:     str
    strategy:   str
    d0_date:    str
    d0_ohlc:    str
    prev_close: float
    gap_pct:    float
    ref_date:   str
    ref_label:  str
    ref_value:  float
    ref_field:  str
    cond1:      bool
    cond2:      bool
    signal:     bool
    direction:  str
    action:     str
    sl_pct:     float
    tp_pct:     float
    max_hold:   int
    extra:      dict = field(default_factory=dict)


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _prev_weekday(df, current_idx, weekday):
    for i in range(current_idx - 1, -1, -1):
        if df.iloc[i]["Date"].dayofweek == weekday:
            return i, df.iloc[i]
    return None, None


def _down_streak(df) -> int:
    closes = df["Close"].values
    streak = 0
    for i in range(len(closes) - 1, 0, -1):
        if closes[i] < closes[i - 1]:
            streak += 1
        else:
            break
    return streak


def _up_streak(df) -> int:
    closes = df["Close"].values
    streak = 0
    for i in range(len(closes) - 1, 0, -1):
        if closes[i] > closes[i - 1]:
            streak += 1
        else:
            break
    return streak


# ── SCANNERS ──────────────────────────────────────────────────────────────────

def scan_ftnt_r2(ticker="FTNT") -> SignalResult:
    df = fetch_ohlc(ticker)
    d0 = df.iloc[-1]; pc = df.iloc[-2]["Close"]; gap = d0["Open"] / pc - 1
    _, pw = _prev_weekday(df, len(df)-1, 0)
    cond1 = bool(pw is not None and pw["Low"] > d0["High"])
    cond2 = bool(gap >= 0.01)
    sig = cond1 and cond2
    return SignalResult(ticker=ticker, strategy="R2 — Gap-Up Breakout Long",
        d0_date=str(d0["Date"].date()), d0_ohlc=f"O={d0['Open']:.2f} H={d0['High']:.2f} L={d0['Low']:.2f} C={d0['Close']:.2f}",
        prev_close=round(float(pc),2), gap_pct=round(float(gap)*100,3),
        ref_date=str(pw["Date"].date()) if pw is not None else "n/a",
        ref_label="Prev-Mon Low", ref_value=round(float(pw["Low"]),2) if pw is not None else 0.0,
        ref_field="Low", cond1=cond1, cond2=cond2, signal=sig, direction="LONG",
        action="ENTER_LONG_NEXT_SESSION_OPEN" if sig else "NO_SIGNAL",
        sl_pct=8.0, tp_pct=15.0, max_hold=5)


def scan_swks_r3(ticker="SWKS") -> SignalResult:
    df = fetch_ohlc(ticker)
    d0 = df.iloc[-1]; pc = df.iloc[-2]["Close"]; gap = d0["Open"] / pc - 1
    _, pw = _prev_weekday(df, len(df)-1, 2)
    cond1 = bool(pw is not None and pw["Open"] < d0["Low"])
    cond2 = bool(gap <= -0.01)
    sig = cond1 and cond2
    return SignalResult(ticker=ticker, strategy="R3 — Gap-Down Rebound Long",
        d0_date=str(d0["Date"].date()), d0_ohlc=f"O={d0['Open']:.2f} H={d0['High']:.2f} L={d0['Low']:.2f} C={d0['Close']:.2f}",
        prev_close=round(float(pc),2), gap_pct=round(float(gap)*100,3),
        ref_date=str(pw["Date"].date()) if pw is not None else "n/a",
        ref_label="Prev-Wed Open", ref_value=round(float(pw["Open"]),2) if pw is not None else 0.0,
        ref_field="Open", cond1=cond1, cond2=cond2, signal=sig, direction="LONG",
        action="ENTER_LONG_NEXT_SESSION_OPEN" if sig else "NO_SIGNAL",
        sl_pct=10.0, tp_pct=8.0, max_hold=5)


def scan_on_r2(ticker="ON") -> SignalResult:
    df = fetch_ohlc(ticker)
    d0 = df.iloc[-1]; pc = df.iloc[-2]["Close"]; gap = d0["Open"] / pc - 1
    _, pt = _prev_weekday(df, len(df)-1, 1)
    _, pw = _prev_weekday(df, len(df)-1, 2)
    cond1 = bool(pt is not None and pt["Low"] > d0["Open"])
    cond2 = bool(pw is not None and pw["Open"] < d0["Open"])
    sig = cond1 and cond2
    return SignalResult(ticker=ticker, strategy="R2 — Rare Continuation Long",
        d0_date=str(d0["Date"].date()), d0_ohlc=f"O={d0['Open']:.2f} H={d0['High']:.2f} L={d0['Low']:.2f} C={d0['Close']:.2f}",
        prev_close=round(float(pc),2), gap_pct=round(float(gap)*100,3),
        ref_date=str(pt["Date"].date()) if pt is not None else "n/a",
        ref_label="Prev-Tue Low", ref_value=round(float(pt["Low"]),2) if pt is not None else 0.0,
        ref_field="Low", cond1=cond1, cond2=cond2, signal=sig, direction="LONG",
        action="ENTER_LONG_NEXT_SESSION_OPEN" if sig else "NO_SIGNAL",
        sl_pct=8.0, tp_pct=10.0, max_hold=5,
        extra={"prev_wed_date": str(pw["Date"].date()) if pw is not None else "n/a",
               "prev_wed_open": round(float(pw["Open"]),2) if pw is not None else 0.0})


def scan_panw_r10(ticker="PANW") -> SignalResult:
    df = fetch_ohlc(ticker)
    d0 = df.iloc[-1]; pc = df.iloc[-2]["Close"]; gap = d0["Open"] / pc - 1
    _, pt = _prev_weekday(df, len(df)-1, 3)
    cond1 = bool(pt is not None and pt["Low"] > d0["Open"])
    cond2 = bool(gap >= 0.01)
    sig = cond1 and cond2
    return SignalResult(ticker=ticker, strategy="R10 — Rare Gap-Up Continuation Long",
        d0_date=str(d0["Date"].date()), d0_ohlc=f"O={d0['Open']:.2f} H={d0['High']:.2f} L={d0['Low']:.2f} C={d0['Close']:.2f}",
        prev_close=round(float(pc),2), gap_pct=round(float(gap)*100,3),
        ref_date=str(pt["Date"].date()) if pt is not None else "n/a",
        ref_label="Prev-Thu Low", ref_value=round(float(pt["Low"]),2) if pt is not None else 0.0,
        ref_field="Low", cond1=cond1, cond2=cond2, signal=sig, direction="LONG",
        action="ENTER_LONG_NEXT_SESSION_OPEN" if sig else "NO_SIGNAL",
        sl_pct=8.0, tp_pct=12.0, max_hold=5)


def scan_crwd_r4(ticker="CRWD") -> SignalResult:
    df = fetch_ohlc(ticker)
    d0 = df.iloc[-1]; pc = df.iloc[-2]["Close"]; gap = d0["Open"] / pc - 1
    _, pf = _prev_weekday(df, len(df)-1, 4)
    cond1 = bool(pf is not None and pf["High"] < d0["Low"])
    cond2 = bool(gap <= -0.01)
    sig = cond1 and cond2
    return SignalResult(ticker=ticker, strategy="R4 — Rare Gap-Down Rebound Long",
        d0_date=str(d0["Date"].date()), d0_ohlc=f"O={d0['Open']:.2f} H={d0['High']:.2f} L={d0['Low']:.2f} C={d0['Close']:.2f}",
        prev_close=round(float(pc),2), gap_pct=round(float(gap)*100,3),
        ref_date=str(pf["Date"].date()) if pf is not None else "n/a",
        ref_label="Prev-Fri High", ref_value=round(float(pf["High"]),2) if pf is not None else 0.0,
        ref_field="High", cond1=cond1, cond2=cond2, signal=sig, direction="LONG",
        action="ENTER_LONG_NEXT_SESSION_OPEN" if sig else "NO_SIGNAL",
        sl_pct=6.0, tp_pct=12.0, max_hold=3)


def scan_mu_r1(ticker="MU") -> SignalResult:
    df = fetch_ohlc(ticker)
    d0 = df.iloc[-1]; pc = df.iloc[-2]["Close"]; gap = d0["Open"] / pc - 1
    _, pt = _prev_weekday(df, len(df)-1, 1)
    _, pf = _prev_weekday(df, len(df)-1, 4)
    cond1 = bool(pt is not None and pt["Low"] > d0["High"])
    cond2 = bool(pf is not None and pf["High"] < d0["Close"])
    sig = cond1 and cond2
    return SignalResult(ticker=ticker, strategy="R1 — Rare Continuation Long",
        d0_date=str(d0["Date"].date()), d0_ohlc=f"O={d0['Open']:.2f} H={d0['High']:.2f} L={d0['Low']:.2f} C={d0['Close']:.2f}",
        prev_close=round(float(pc),2), gap_pct=round(float(gap)*100,3),
        ref_date=str(pt["Date"].date()) if pt is not None else "n/a",
        ref_label="Prev-Tue Low", ref_value=round(float(pt["Low"]),2) if pt is not None else 0.0,
        ref_field="Low", cond1=cond1, cond2=cond2, signal=sig, direction="LONG",
        action="ENTER_LONG_NEXT_SESSION_OPEN" if sig else "NO_SIGNAL",
        sl_pct=4.0, tp_pct=12.0, max_hold=5,
        extra={"prev_fri_date": str(pf["Date"].date()) if pf is not None else "n/a",
               "prev_fri_high": round(float(pf["High"]),2) if pf is not None else 0.0})


def scan_tsm_r3(ticker="TSM") -> SignalResult:
    df = fetch_ohlc(ticker)
    d0 = df.iloc[-1]; pc = df.iloc[-2]["Close"]; gap = d0["Open"] / pc - 1
    _, pw = _prev_weekday(df, len(df)-1, 2)
    streak = _down_streak(df)
    cond1 = bool(pw is not None and pw["Close"] < d0["Low"])
    cond2 = bool(streak >= 2)
    sig = cond1 and cond2
    return SignalResult(ticker=ticker, strategy="R3 — Rare Rebound Long",
        d0_date=str(d0["Date"].date()), d0_ohlc=f"O={d0['Open']:.2f} H={d0['High']:.2f} L={d0['Low']:.2f} C={d0['Close']:.2f}",
        prev_close=round(float(pc),2), gap_pct=round(float(gap)*100,3),
        ref_date=str(pw["Date"].date()) if pw is not None else "n/a",
        ref_label="Prev-Wed Close", ref_value=round(float(pw["Close"]),2) if pw is not None else 0.0,
        ref_field="Close", cond1=cond1, cond2=cond2, signal=sig, direction="LONG",
        action="ENTER_LONG_NEXT_SESSION_OPEN" if sig else "NO_SIGNAL",
        sl_pct=3.0, tp_pct=10.0, max_hold=5, extra={"down_streak": streak})


def scan_amd_r8(ticker="AMD") -> SignalResult:
    df = fetch_ohlc(ticker)
    d0 = df.iloc[-1]; pc = df.iloc[-2]["Close"]; gap = d0["Open"] / pc - 1
    _, pw  = _prev_weekday(df, len(df)-1, 2)
    _, pth = _prev_weekday(df, len(df)-1, 3)
    cond1 = bool(pw  is not None and pw["Close"] < d0["Low"])
    cond2 = bool(pth is not None and pth["High"] > d0["High"])
    sig = cond1 and cond2
    return SignalResult(ticker=ticker, strategy="R8 — Rare Rebound Long",
        d0_date=str(d0["Date"].date()), d0_ohlc=f"O={d0['Open']:.2f} H={d0['High']:.2f} L={d0['Low']:.2f} C={d0['Close']:.2f}",
        prev_close=round(float(pc),2), gap_pct=round(float(gap)*100,3),
        ref_date=str(pw["Date"].date()) if pw is not None else "n/a",
        ref_label="Prev-Wed Close", ref_value=round(float(pw["Close"]),2) if pw is not None else 0.0,
        ref_field="Close", cond1=cond1, cond2=cond2, signal=sig, direction="LONG",
        action="ENTER_LONG_NEXT_SESSION_OPEN" if sig else "NO_SIGNAL",
        sl_pct=10.0, tp_pct=10.0, max_hold=5,
        extra={"prev_thu_date": str(pth["Date"].date()) if pth is not None else "n/a",
               "prev_thu_high": round(float(pth["High"]),2) if pth is not None else 0.0})


def scan_mrvl_r7(ticker="MRVL") -> SignalResult:
    df = fetch_ohlc(ticker)
    d0 = df.iloc[-1]; pc = df.iloc[-2]["Close"]; gap = d0["Open"] / pc - 1
    _, pw = _prev_weekday(df, len(df)-1, 2)
    streak = _down_streak(df)
    cond1 = bool(pw is not None and pw["Close"] < d0["Low"])
    cond2 = bool(streak >= 2)
    sig = cond1 and cond2
    return SignalResult(ticker=ticker, strategy="R7 — Rare Rebound Long",
        d0_date=str(d0["Date"].date()), d0_ohlc=f"O={d0['Open']:.2f} H={d0['High']:.2f} L={d0['Low']:.2f} C={d0['Close']:.2f}",
        prev_close=round(float(pc),2), gap_pct=round(float(gap)*100,3),
        ref_date=str(pw["Date"].date()) if pw is not None else "n/a",
        ref_label="Prev-Wed Close", ref_value=round(float(pw["Close"]),2) if pw is not None else 0.0,
        ref_field="Close", cond1=cond1, cond2=cond2, signal=sig, direction="LONG",
        action="ENTER_LONG_NEXT_SESSION_OPEN" if sig else "NO_SIGNAL",
        sl_pct=5.0, tp_pct=12.0, max_hold=3, extra={"down_streak": streak})


def scan_ddog_r9(ticker="DDOG") -> SignalResult:
    df = fetch_ohlc(ticker)
    d0 = df.iloc[-1]; pc = df.iloc[-2]["Close"]; gap = d0["Open"] / pc - 1
    _, pm = _prev_weekday(df, len(df)-1, 0)
    _, pt = _prev_weekday(df, len(df)-1, 3)
    cond1 = bool(pm is not None and pm["Open"] < d0["Close"])
    cond2 = bool(pt  is not None and pt["Low"]  > d0["High"])
    sig = cond1 and cond2
    return SignalResult(ticker=ticker, strategy="R9 — Rare Momentum Continuation Long",
        d0_date=str(d0["Date"].date()), d0_ohlc=f"O={d0['Open']:.2f} H={d0['High']:.2f} L={d0['Low']:.2f} C={d0['Close']:.2f}",
        prev_close=round(float(pc),2), gap_pct=round(float(gap)*100,3),
        ref_date=str(pm["Date"].date()) if pm is not None else "n/a",
        ref_label="Prev-Mon Open", ref_value=round(float(pm["Open"]),2) if pm is not None else 0.0,
        ref_field="Open", cond1=cond1, cond2=cond2, signal=sig, direction="LONG",
        action="ENTER_LONG_NEXT_SESSION_OPEN" if sig else "NO_SIGNAL",
        sl_pct=8.0, tp_pct=10.0, max_hold=5,
        extra={"prev_thu_date": str(pt["Date"].date()) if pt is not None else "n/a",
               "prev_thu_low": round(float(pt["Low"]),2) if pt is not None else 0.0})


def scan_net_r9(ticker="NET") -> SignalResult:
    df = fetch_ohlc(ticker)
    d0 = df.iloc[-1]; pc = df.iloc[-2]["Close"]; gap = d0["Open"] / pc - 1
    _, pw = _prev_weekday(df, len(df)-1, 2)
    _, pf = _prev_weekday(df, len(df)-1, 4)
    cond1 = bool(pw is not None and pw["Low"] < d0["Open"])
    cond2 = bool(pf is not None and pf["Low"] > d0["High"])
    sig = cond1 and cond2
    return SignalResult(ticker=ticker, strategy="R9 — Rare Momentum Continuation Long",
        d0_date=str(d0["Date"].date()), d0_ohlc=f"O={d0['Open']:.2f} H={d0['High']:.2f} L={d0['Low']:.2f} C={d0['Close']:.2f}",
        prev_close=round(float(pc),2), gap_pct=round(float(gap)*100,3),
        ref_date=str(pw["Date"].date()) if pw is not None else "n/a",
        ref_label="Prev-Wed Low", ref_value=round(float(pw["Low"]),2) if pw is not None else 0.0,
        ref_field="Low", cond1=cond1, cond2=cond2, signal=sig, direction="LONG",
        action="ENTER_LONG_NEXT_SESSION_OPEN" if sig else "NO_SIGNAL",
        sl_pct=8.0, tp_pct=12.0, max_hold=3,
        extra={"prev_fri_date": str(pf["Date"].date()) if pf is not None else "n/a",
               "prev_fri_low": round(float(pf["Low"]),2) if pf is not None else 0.0})


def scan_okta_r9(ticker="OKTA") -> SignalResult:
    df = fetch_ohlc(ticker)
    d0 = df.iloc[-1]; pc = df.iloc[-2]["Close"]; gap = d0["Open"] / pc - 1
    _, pm = _prev_weekday(df, len(df)-1, 0)
    _, pw = _prev_weekday(df, len(df)-1, 2)
    cond1 = bool(pm is not None and pm["Low"] < d0["Low"])
    cond2 = bool(pw is not None and pw["Low"] > d0["High"])
    sig = cond1 and cond2
    return SignalResult(ticker=ticker, strategy="R9 — Rare Rebound Long",
        d0_date=str(d0["Date"].date()), d0_ohlc=f"O={d0['Open']:.2f} H={d0['High']:.2f} L={d0['Low']:.2f} C={d0['Close']:.2f}",
        prev_close=round(float(pc),2), gap_pct=round(float(gap)*100,3),
        ref_date=str(pm["Date"].date()) if pm is not None else "n/a",
        ref_label="Prev-Mon Low", ref_value=round(float(pm["Low"]),2) if pm is not None else 0.0,
        ref_field="Low", cond1=cond1, cond2=cond2, signal=sig, direction="LONG",
        action="ENTER_LONG_NEXT_SESSION_OPEN" if sig else "NO_SIGNAL",
        sl_pct=10.0, tp_pct=3.0, max_hold=3,
        extra={"prev_wed_date": str(pw["Date"].date()) if pw is not None else "n/a",
               "prev_wed_low": round(float(pw["Low"]),2) if pw is not None else 0.0})


def scan_bill_r9(ticker="BILL") -> SignalResult:
    df = fetch_ohlc(ticker)
    d0 = df.iloc[-1]; pc = df.iloc[-2]["Close"]; gap = d0["Open"] / pc - 1
    _, pm = _prev_weekday(df, len(df)-1, 0)
    _, pw = _prev_weekday(df, len(df)-1, 2)
    cond1 = bool(pm is not None and pm["High"] < d0["Low"])
    cond2 = bool(pw is not None and pw["Close"] > d0["Close"])
    sig = cond1 and cond2
    return SignalResult(ticker=ticker, strategy="R9 — Rare Breakdown Short",
        d0_date=str(d0["Date"].date()), d0_ohlc=f"O={d0['Open']:.2f} H={d0['High']:.2f} L={d0['Low']:.2f} C={d0['Close']:.2f}",
        prev_close=round(float(pc),2), gap_pct=round(float(gap)*100,3),
        ref_date=str(pm["Date"].date()) if pm is not None else "n/a",
        ref_label="Prev-Mon High", ref_value=round(float(pm["High"]),2) if pm is not None else 0.0,
        ref_field="High", cond1=cond1, cond2=cond2, signal=sig, direction="SHORT",
        action="ENTER_SHORT_NEXT_SESSION_OPEN" if sig else "NO_SIGNAL",
        sl_pct=8.0, tp_pct=15.0, max_hold=3,
        extra={"prev_wed_date": str(pw["Date"].date()) if pw is not None else "n/a",
               "prev_wed_close": round(float(pw["Close"]),2) if pw is not None else 0.0})


def scan_u_r9(ticker="U") -> SignalResult:
    df = fetch_ohlc(ticker)
    d0 = df.iloc[-1]; pc = df.iloc[-2]["Close"]; gap = d0["Open"] / pc - 1
    _, pt = _prev_weekday(df, len(df)-1, 1)
    candle_range = d0["High"] - d0["Low"]
    close_pos = float((d0["Close"] - d0["Low"]) / candle_range) if candle_range > 0 else 0.5
    cond1 = bool(pt is not None and pt["High"] < d0["Close"])
    cond2 = bool(close_pos <= 0.25)
    sig = cond1 and cond2
    return SignalResult(ticker=ticker, strategy="R9 — Rare Breakdown Short",
        d0_date=str(d0["Date"].date()), d0_ohlc=f"O={d0['Open']:.2f} H={d0['High']:.2f} L={d0['Low']:.2f} C={d0['Close']:.2f}",
        prev_close=round(float(pc),2), gap_pct=round(float(gap)*100,3),
        ref_date=str(pt["Date"].date()) if pt is not None else "n/a",
        ref_label="Prev-Tue High", ref_value=round(float(pt["High"]),2) if pt is not None else 0.0,
        ref_field="High", cond1=cond1, cond2=cond2, signal=sig, direction="SHORT",
        action="ENTER_SHORT_NEXT_SESSION_OPEN" if sig else "NO_SIGNAL",
        sl_pct=8.0, tp_pct=12.0, max_hold=2, extra={"close_pos": round(close_pos, 3)})


def scan_roku_r1(ticker="ROKU") -> SignalResult:
    df = fetch_ohlc(ticker)
    d0 = df.iloc[-1]; pc = df.iloc[-2]["Close"]; gap = d0["Open"] / pc - 1
    _, pt  = _prev_weekday(df, len(df)-1, 1)
    _, pth = _prev_weekday(df, len(df)-1, 3)
    cond1 = bool(pt  is not None and pt["Low"]   > d0["Open"])
    cond2 = bool(pth is not None and pth["High"] < d0["Open"])
    sig = cond1 and cond2
    return SignalResult(ticker=ticker, strategy="R1 — Rare Breakdown Short",
        d0_date=str(d0["Date"].date()), d0_ohlc=f"O={d0['Open']:.2f} H={d0['High']:.2f} L={d0['Low']:.2f} C={d0['Close']:.2f}",
        prev_close=round(float(pc),2), gap_pct=round(float(gap)*100,3),
        ref_date=str(pt["Date"].date()) if pt is not None else "n/a",
        ref_label="Prev-Tue Low", ref_value=round(float(pt["Low"]),2) if pt is not None else 0.0,
        ref_field="Low", cond1=cond1, cond2=cond2, signal=sig, direction="SHORT",
        action="ENTER_SHORT_NEXT_SESSION_OPEN" if sig else "NO_SIGNAL",
        sl_pct=4.0, tp_pct=15.0, max_hold=5,
        extra={"prev_thu_date": str(pth["Date"].date()) if pth is not None else "n/a",
               "prev_thu_high": round(float(pth["High"]),2) if pth is not None else 0.0})


def scan_twlo_r5(ticker="TWLO") -> SignalResult:
    df = fetch_ohlc(ticker)
    d0 = df.iloc[-1]; pc = df.iloc[-2]["Close"]; gap = d0["Open"] / pc - 1
    _, pth = _prev_weekday(df, len(df)-1, 3)
    _, pf  = _prev_weekday(df, len(df)-1, 4)
    cond1 = bool(pth is not None and pth["High"] < d0["Open"])
    cond2 = bool(pf  is not None and pf["Low"]   > d0["Close"])
    sig = cond1 and cond2
    return SignalResult(ticker=ticker, strategy="R5 — Rare Momentum Continuation Long",
        d0_date=str(d0["Date"].date()), d0_ohlc=f"O={d0['Open']:.2f} H={d0['High']:.2f} L={d0['Low']:.2f} C={d0['Close']:.2f}",
        prev_close=round(float(pc),2), gap_pct=round(float(gap)*100,3),
        ref_date=str(pth["Date"].date()) if pth is not None else "n/a",
        ref_label="Prev-Thu High", ref_value=round(float(pth["High"]),2) if pth is not None else 0.0,
        ref_field="High", cond1=cond1, cond2=cond2, signal=sig, direction="LONG",
        action="ENTER_LONG_NEXT_SESSION_OPEN" if sig else "NO_SIGNAL",
        sl_pct=4.0, tp_pct=10.0, max_hold=3,
        extra={"prev_fri_date": str(pf["Date"].date()) if pf is not None else "n/a",
               "prev_fri_low": round(float(pf["Low"]),2) if pf is not None else 0.0})


def scan_docu_r5(ticker="DOCU") -> SignalResult:
    df = fetch_ohlc(ticker)
    d0 = df.iloc[-1]; pc = df.iloc[-2]["Close"]; gap = d0["Open"] / pc - 1
    _, pm = _prev_weekday(df, len(df)-1, 0)
    streak = _up_streak(df)
    cond1 = bool(pm is not None and pm["Close"] > d0["High"])
    cond2 = bool(streak >= 2)
    sig = cond1 and cond2
    return SignalResult(ticker=ticker, strategy="R5 — Rare Reversal Short",
        d0_date=str(d0["Date"].date()), d0_ohlc=f"O={d0['Open']:.2f} H={d0['High']:.2f} L={d0['Low']:.2f} C={d0['Close']:.2f}",
        prev_close=round(float(pc),2), gap_pct=round(float(gap)*100,3),
        ref_date=str(pm["Date"].date()) if pm is not None else "n/a",
        ref_label="Prev-Mon Close", ref_value=round(float(pm["Close"]),2) if pm is not None else 0.0,
        ref_field="Close", cond1=cond1, cond2=cond2, signal=sig, direction="SHORT",
        action="ENTER_SHORT_NEXT_SESSION_OPEN" if sig else "NO_SIGNAL",
        sl_pct=8.0, tp_pct=15.0, max_hold=5, extra={"up_streak": streak})


def scan_coin_r1(ticker="COIN") -> SignalResult:
    df = fetch_ohlc(ticker)
    d0 = df.iloc[-1]; pc = df.iloc[-2]["Close"]; gap = d0["Open"] / pc - 1
    _, pt = _prev_weekday(df, len(df)-1, 1)
    streak = _down_streak(df)
    cond1 = bool(pt is not None and pt["Close"] < d0["Open"])
    cond2 = bool(streak >= 3)
    sig = cond1 and cond2
    return SignalResult(ticker=ticker, strategy="R1 — Rare Breakdown Short",
        d0_date=str(d0["Date"].date()), d0_ohlc=f"O={d0['Open']:.2f} H={d0['High']:.2f} L={d0['Low']:.2f} C={d0['Close']:.2f}",
        prev_close=round(float(pc),2), gap_pct=round(float(gap)*100,3),
        ref_date=str(pt["Date"].date()) if pt is not None else "n/a",
        ref_label="Prev-Tue Close", ref_value=round(float(pt["Close"]),2) if pt is not None else 0.0,
        ref_field="Close", cond1=cond1, cond2=cond2, signal=sig, direction="SHORT",
        action="ENTER_SHORT_NEXT_SESSION_OPEN" if sig else "NO_SIGNAL",
        sl_pct=2.5, tp_pct=10.0, max_hold=2, extra={"down_streak": streak})


def scan_pltr_r2(ticker="PLTR") -> SignalResult:
    df = fetch_ohlc(ticker)
    d0 = df.iloc[-1]; pc = df.iloc[-2]["Close"]; gap = d0["Open"] / pc - 1
    _, pw  = _prev_weekday(df, len(df)-1, 2)
    _, pth = _prev_weekday(df, len(df)-1, 3)
    cond1 = bool(pw  is not None and pw["High"]  < d0["Open"])
    cond2 = bool(pth is not None and pth["Low"]  > d0["Close"])
    sig = cond1 and cond2
    return SignalResult(ticker=ticker, strategy="R2 — Rare Momentum Continuation Long",
        d0_date=str(d0["Date"].date()), d0_ohlc=f"O={d0['Open']:.2f} H={d0['High']:.2f} L={d0['Low']:.2f} C={d0['Close']:.2f}",
        prev_close=round(float(pc),2), gap_pct=round(float(gap)*100,3),
        ref_date=str(pw["Date"].date()) if pw is not None else "n/a",
        ref_label="Prev-Wed High", ref_value=round(float(pw["High"]),2) if pw is not None else 0.0,
        ref_field="High", cond1=cond1, cond2=cond2, signal=sig, direction="LONG",
        action="ENTER_LONG_NEXT_SESSION_OPEN" if sig else "NO_SIGNAL",
        sl_pct=5.0, tp_pct=12.0, max_hold=2,
        extra={"prev_thu_date": str(pth["Date"].date()) if pth is not None else "n/a",
               "prev_thu_low": round(float(pth["Low"]),2) if pth is not None else 0.0})


def scan_upst_r10(ticker="UPST") -> SignalResult:
    df = fetch_ohlc(ticker)
    d0 = df.iloc[-1]; pc = df.iloc[-2]["Close"]; gap = d0["Open"] / pc - 1
    _, pm = _prev_weekday(df, len(df)-1, 0)
    _, pw = _prev_weekday(df, len(df)-1, 2)
    cond1 = bool(pm is not None and pm["Low"]  > d0["High"])
    cond2 = bool(pw is not None and pw["Open"] < d0["Open"])
    sig = cond1 and cond2
    return SignalResult(ticker=ticker, strategy="R10 — Rare Rebound Long",
        d0_date=str(d0["Date"].date()), d0_ohlc=f"O={d0['Open']:.2f} H={d0['High']:.2f} L={d0['Low']:.2f} C={d0['Close']:.2f}",
        prev_close=round(float(pc),2), gap_pct=round(float(gap)*100,3),
        ref_date=str(pm["Date"].date()) if pm is not None else "n/a",
        ref_label="Prev-Mon Low", ref_value=round(float(pm["Low"]),2) if pm is not None else 0.0,
        ref_field="Low", cond1=cond1, cond2=cond2, signal=sig, direction="LONG",
        action="ENTER_LONG_NEXT_SESSION_OPEN" if sig else "NO_SIGNAL",
        sl_pct=2.0, tp_pct=6.0, max_hold=3,
        extra={"prev_wed_date": str(pw["Date"].date()) if pw is not None else "n/a",
               "prev_wed_open": round(float(pw["Open"]),2) if pw is not None else 0.0})


def scan_path_r5(ticker="PATH") -> SignalResult:
    df = fetch_ohlc(ticker)
    d0 = df.iloc[-1]; pc = df.iloc[-2]["Close"]; gap = d0["Open"] / pc - 1
    _, pm = _prev_weekday(df, len(df)-1, 0)
    _, pt = _prev_weekday(df, len(df)-1, 1)
    cond1 = bool(pm is not None and pm["Close"] > d0["Low"])
    cond2 = bool(pt  is not None and pt["High"]  < d0["Low"])
    sig = cond1 and cond2
    return SignalResult(ticker=ticker, strategy="R5 — Rare Breakdown Short",
        d0_date=str(d0["Date"].date()), d0_ohlc=f"O={d0['Open']:.2f} H={d0['High']:.2f} L={d0['Low']:.2f} C={d0['Close']:.2f}",
        prev_close=round(float(pc),2), gap_pct=round(float(gap)*100,3),
        ref_date=str(pm["Date"].date()) if pm is not None else "n/a",
        ref_label="Prev-Mon Close", ref_value=round(float(pm["Close"]),2) if pm is not None else 0.0,
        ref_field="Close", cond1=cond1, cond2=cond2, signal=sig, direction="SHORT",
        action="ENTER_SHORT_NEXT_SESSION_OPEN" if sig else "NO_SIGNAL",
        sl_pct=8.0, tp_pct=10.0, max_hold=2,
        extra={"prev_tue_date": str(pt["Date"].date()) if pt is not None else "n/a",
               "prev_tue_high": round(float(pt["High"]),2) if pt is not None else 0.0})


def scan_rblx_r1(ticker="RBLX") -> SignalResult:
    df = fetch_ohlc(ticker)
    d0 = df.iloc[-1]; pc = df.iloc[-2]["Close"]; gap = d0["Open"] / pc - 1
    _, pt = _prev_weekday(df, len(df)-1, 1)
    streak = _down_streak(df)
    cond1 = bool(pt is not None and pt["High"] < d0["Open"])
    cond2 = bool(streak >= 2)
    sig = cond1 and cond2
    return SignalResult(ticker=ticker, strategy="R1 — Rare Rebound Long",
        d0_date=str(d0["Date"].date()), d0_ohlc=f"O={d0['Open']:.2f} H={d0['High']:.2f} L={d0['Low']:.2f} C={d0['Close']:.2f}",
        prev_close=round(float(pc),2), gap_pct=round(float(gap)*100,3),
        ref_date=str(pt["Date"].date()) if pt is not None else "n/a",
        ref_label="Prev-Tue High", ref_value=round(float(pt["High"]),2) if pt is not None else 0.0,
        ref_field="High", cond1=cond1, cond2=cond2, signal=sig, direction="LONG",
        action="ENTER_LONG_NEXT_SESSION_OPEN" if sig else "NO_SIGNAL",
        sl_pct=5.0, tp_pct=12.0, max_hold=2, extra={"down_streak": streak})


# ── SCANNER REGISTRY ──────────────────────────────────────────────────────────
# To add a new ticker: define a scan_X() function above, then add it here.

SCANNERS: Dict[str, Callable] = {
    "FTNT": scan_ftnt_r2,
    "SWKS": scan_swks_r3,
    "ON":   scan_on_r2,
    "PANW": scan_panw_r10,
    "CRWD": scan_crwd_r4,
    "MU":   scan_mu_r1,
    "TSM":  scan_tsm_r3,
    "AMD":  scan_amd_r8,
    "MRVL": scan_mrvl_r7,
    "DDOG": scan_ddog_r9,
    "NET":  scan_net_r9,
    "OKTA": scan_okta_r9,
    "BILL": scan_bill_r9,
    "U":    scan_u_r9,
    "ROKU": scan_roku_r1,
    "TWLO": scan_twlo_r5,
    "DOCU": scan_docu_r5,
    "COIN": scan_coin_r1,
    "PLTR": scan_pltr_r2,
    "UPST": scan_upst_r10,
    "PATH": scan_path_r5,
    "RBLX": scan_rblx_r1,
}


# ── TELEGRAM ──────────────────────────────────────────────────────────────────

def send_telegram(token: str, chat_id: str, text: str):
    if requests is None:
        return False, "requests not installed"
    try:
        url  = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, json={"chat_id": chat_id, "text": text,
                                         "parse_mode": "HTML"}, timeout=10)
        return (True, "OK") if resp.status_code == 200 else (False, resp.text)
    except Exception as e:
        return False, str(e)


def format_telegram_message(r: SignalResult) -> str:
    icon = ("🔴" if r.direction == "SHORT" else "🟢") if r.signal else "⚪"
    lines = [
        f"{icon} <b>{r.ticker} — {r.strategy}</b>",
        f"D0: {r.d0_date}   {r.d0_ohlc}",
        f"Prev Close: {r.prev_close}   Gap: {r.gap_pct:+.2f}%",
        f"{r.ref_label}: {r.ref_value}  ({r.ref_date})",
        f"Cond1: {'✅' if r.cond1 else '❌'}   Cond2: {'✅' if r.cond2 else '❌'}",
        f"",
        f"<b>ACTION: {r.action}</b>",
    ]
    if r.signal:
        lines.append(f"SL: −{r.sl_pct:.0f}%  |  TP: +{r.tp_pct:.0f}%  |  MaxHold: {r.max_hold} sessions")
    lines.append(f"\nScanned: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    return "\n".join(lines)


# ── UI HELPERS ────────────────────────────────────────────────────────────────

def render_result_card(r: SignalResult, token: str, chat_id: str, context: str = "tab"):
    if r.signal:
        border = "#26C281" if r.direction == "LONG" else "#FF4B4B"
        icon   = "🟢 SIGNAL — LONG" if r.direction == "LONG" else "🔴 SIGNAL — SHORT"
    else:
        border = "#444"
        icon   = "⚪ NO SIGNAL"

    st.markdown(f"""
    <div style='background:#161b22;border-radius:8px;padding:14px 18px;
                border-left:5px solid {border};margin-bottom:10px'>
      <b style='color:{border};font-size:1.1em'>{r.ticker}</b>
      <span style='color:#888;margin-left:10px'>{r.strategy}</span>
      <span style='float:right;color:{border};font-weight:700'>{icon}</span>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Date", r.d0_date)
    c2.metric("Gap %", f"{r.gap_pct:+.2f}%")
    c3.metric("Cond1", "✅" if r.cond1 else "❌")
    c4.metric("Cond2", "✅" if r.cond2 else "❌")

    with st.expander(f"Details — {r.ticker}", expanded=r.signal):
        st.markdown(f"""
        | Field | Value |
        |---|---|
        | D0 OHLC | `{r.d0_ohlc}` |
        | Prev close | {r.prev_close} |
        | {r.ref_label} | {r.ref_value} ({r.ref_date}) |
        | Direction | {r.direction} |
        | Action | **{r.action}** |
        | SL / TP / MaxHold | -{r.sl_pct:.0f}% / +{r.tp_pct:.0f}% / {r.max_hold} sessions |
        """)
        if r.extra:
            for k, v in r.extra.items():
                st.caption(f"{k}: {v}")

        if r.signal and token and chat_id:
            if st.button(f"📡 Send Telegram alert — {r.ticker}", key=f"tg_{context}_{r.ticker}"):
                ok, msg = send_telegram(token, chat_id, format_telegram_message(r))
                st.success("✅ Sent!") if ok else st.error(f"❌ Failed: {msg}")


def render_summary_table(results: List[SignalResult]):
    rows = []
    for r in results:
        rows.append({
            "Ticker":    r.ticker,
            "Strategy":  r.strategy,
            "Direction": r.direction,
            "Signal":    "✅ YES" if r.signal else "—",
            "Action":    r.action,
            "Gap %":     f"{r.gap_pct:+.2f}%",
            "C1":        "✅" if r.cond1 else "❌",
            "C2":        "✅" if r.cond2 else "❌",
            "SL":        f"-{r.sl_pct:.0f}%",
            "TP":        f"+{r.tp_pct:.0f}%",
            "MaxHold":   r.max_hold,
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    return df


# ── MAIN APP ──────────────────────────────────────────────────────────────────

def main():

    st.markdown("""
    <style>
      .stApp { background-color: #0d1117; }
      .block-container { padding-top: 1rem; }
    </style>""", unsafe_allow_html=True)

    st.title("📡 Signal Scanner — 20260516")
    st.caption("⚠️ RESEARCH ONLY. NOT FINANCIAL ADVICE. 22 tickers · Rare pattern strategies.")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        token   = st.text_input("Telegram Bot Token", type="password",
                                 placeholder="123456:ABC-...", key="tg_token")
        chat_id = st.text_input("Telegram Chat ID",
                                 placeholder="-1001234567890", key="tg_chat")
        st.caption("Credentials session-only. Not stored.")
        st.markdown("---")
        st.markdown("### 🔍 Ticker filter")
        all_tickers = list(SCANNERS.keys())
        selected = st.multiselect("Show only:", all_tickers, default=all_tickers)
        st.markdown("---")
        st.markdown("### 🔄 Scan")
        run_scan = st.button("▶ Run Scan Now", use_container_width=True, type="primary")
        st.caption("Data cached 5 min. Click to force refresh.")
        if st.button("🗑 Clear cache & rescan", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ── Run scan ──────────────────────────────────────────────────────────────
    if run_scan or "scan_results" not in st.session_state:
        results = []
        progress = st.progress(0, text="Scanning...")
        tickers_to_scan = [t for t in all_tickers if t in selected]
        for i, ticker in enumerate(tickers_to_scan):
            progress.progress((i + 1) / len(tickers_to_scan),
                               text=f"Scanning {ticker}...")
            try:
                r = SCANNERS[ticker](ticker)
                results.append(r)
            except Exception as e:
                st.warning(f"{ticker}: error — {e}")
        progress.empty()
        st.session_state["scan_results"] = results
        st.session_state["scan_time"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    results: List[SignalResult] = st.session_state.get("scan_results", [])
    scan_time = st.session_state.get("scan_time", "—")

    if not results:
        st.info("Click ▶ Run Scan Now in the sidebar.")
        return

    # ── Summary ───────────────────────────────────────────────────────────────
    active = [r for r in results if r.signal]
    st.markdown(f"**Last scan:** {scan_time} &nbsp;|&nbsp; "
                f"**Tickers scanned:** {len(results)} &nbsp;|&nbsp; "
                f"**Active signals:** {len(active)}")

    if active:
        st.success(f"🚨 {len(active)} active signal(s): {', '.join(r.ticker for r in active)}")

    # ── Summary table ─────────────────────────────────────────────────────────
    st.subheader("📋 Summary Table")
    summary_df = render_summary_table(results)

    # CSV export
    buf = io.StringIO()
    summary_df.to_csv(buf, index=False)
    st.download_button("⬇️ Export summary CSV",
                        buf.getvalue().encode(),
                        f"signal_scan_{datetime.utcnow().strftime('%Y%m%d')}.csv",
                        "text/csv")

    # ── Active signals first ──────────────────────────────────────────────────
    if active:
        st.markdown("---")
        st.subheader("🚨 Active Signals")
        for r in active:
            render_result_card(r, token, chat_id, context="active")

    # ── All results ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 All Tickers")
    tab_names = [r.ticker for r in results]
    if tab_names:
        tabs = st.tabs(tab_names)
        for tab, r in zip(tabs, results):
            with tab:
                render_result_card(r, token, chat_id, context="tab")


if __name__ == "__main__":
    main()
