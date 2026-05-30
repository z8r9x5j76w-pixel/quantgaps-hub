#!/usr/bin/env python3
"""
momentum_sequence_lab_dashboard.py
====================================
Momentum Sequence Lab — Forward Test Dashboard

⚠️ RESEARCH / FORWARD-TEST ONLY. NOT FINANCIAL ADVICE. NOT PRODUCTION-READY.

Run:
    streamlit run momentum_sequence_lab_dashboard.py

═══════════════════════════════════════════════════════
HOW TO ADD A NEW STRATEGY
═══════════════════════════════════════════════════════
1. Define a signal function:
       def signal_my_strategy(df: pd.DataFrame, cfg: StrategyConfig) -> pd.Series:
           # Return a boolean Series aligned to df.index
           # Use only past data (no lookahead)
           return (df["Low"].shift(N) > df["Open"].shift(M)) & (df["ATR_PCT"] < df["ATR_PCT_MED"])

2. Register it in SIGNAL_REGISTRY:
       SIGNAL_REGISTRY["MY_RULE_TYPE"] = signal_my_strategy

3. Add a StrategyConfig entry to STRATEGIES:
       StrategyConfig(
           strategy_id="XYZ_MOMSEQ_...",
           ticker="XYZ",
           rule_type="MY_RULE_TYPE",
           ...
       )

4. The backtest, metrics, charts, rankings, and CSV exports
   are all reused automatically. No other changes needed.
═══════════════════════════════════════════════════════
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import requests
except ImportError:
    requests = None


# ══════════════════════════════════════════════════════════════════════════════
# PASSWORD GATE
# Change the string below to set your own password.
# ══════════════════════════════════════════════════════════════════════════════

def _check_password() -> bool:
    return True


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StrategyConfig:
    strategy_id:          str
    ticker:               str
    direction:            str            # "LONG" or "SHORT"
    status:               str            # "FORWARD_TEST" / "LIVE" / "WATCHLIST"
    rule_type:            str            # key into SIGNAL_REGISTRY
    full_history_start:   str            # e.g. "2015-01-01"
    validation_start:     str            # e.g. "2022-01-20"

    # Exit params
    tp_atr:               float = 4.0
    sl_atr:               float = 2.0
    max_hold:             int   = 10

    # Costs
    cost_bps_per_side:     float = 2.0
    slippage_bps_per_side: float = 2.0

    # Display / snapshot metadata (for drift detection only)
    notes:                str   = ""
    snapshot_trades:      int   = 0
    snapshot_win_rate:    float = 0.0
    snapshot_cagr:        float = 0.0
    snapshot_sharpe:      float = 0.0
    snapshot_pf:          float = 0.0
    extra_tags:           List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGIES LIST — add new StrategyConfig entries here
# ══════════════════════════════════════════════════════════════════════════════

STRATEGIES: List[StrategyConfig] = [

    StrategyConfig(
        strategy_id         = "APPF_MOMSEQ_LOW10_GT_OPEN9_ATRLOW_LONG_TP4_SL2_H10",
        ticker              = "APPF",
        direction           = "LONG",
        status              = "FORWARD_TEST",
        rule_type           = "APPF_LOW10_OPEN9",
        full_history_start  = "2015-01-01",
        validation_start    = "2022-01-20",
        tp_atr              = 4.0,
        sl_atr              = 2.0,
        max_hold            = 10,
        cost_bps_per_side   = 2.0,
        slippage_bps_per_side = 2.0,
        notes=(
            "Signal: Low[t-10] > Open[t-9]  AND  ATR% < 252-session median ATR%. "
            "Entry next session open. ATR-based exit (TP 4×, SL 2×, max 10 sessions). "
            "Validation (2022-2026): CAGR 29.35%, excess +21.70%, Sharpe 1.39. FORWARD_TEST only."
        ),
        snapshot_trades     = 33,
        snapshot_win_rate   = 72.73,
        snapshot_cagr       = 29.35,
        snapshot_sharpe     = 1.39,
        snapshot_pf         = 3.98,
        extra_tags          = ["momentum", "sequence", "ATR-low-regime"],
    ),

    # ── Strategy 2: WDAY SHORT ────────────────────────────────────────────────
    StrategyConfig(
        strategy_id         = "WDAY_MOMSEQ_CLOSE3_LT_LOW2_SMA20_SHORT_TP1p5_SL1_H3",
        ticker              = "WDAY",
        direction           = "SHORT",
        status              = "FORWARD_TEST",
        rule_type           = "WDAY_CLOSE3_LT_LOW2_SMA20",
        full_history_start  = "2015-01-01",
        validation_start    = "2022-01-20",
        tp_atr              = 1.5,
        sl_atr              = 1.0,
        max_hold            = 3,
        cost_bps_per_side   = 2.0,
        slippage_bps_per_side = 2.0,
        notes=(
            "Signal: Close[t-3] < Low[t-2]  AND  Close < SMA20. SHORT. "
            "Entry next session open. ATR-based exit (TP 1.5×, SL 1.0×, max 3 sessions). "
            "Validation (2022+): CAGR 16.43%, excess +30.70%, Sharpe 1.35. FORWARD_TEST only."
        ),
        snapshot_trades     = 55,
        snapshot_win_rate   = 0.0,
        snapshot_cagr       = 16.43,
        snapshot_sharpe     = 1.35,
        snapshot_pf         = 2.62,
        extra_tags          = ["SaaS", "momentum", "sequence", "short", "SMA20"],
    ),

    # ── Strategy 3: HAL LONG ─────────────────────────────────────────────────
    StrategyConfig(
        strategy_id         = "HAL_MOMSEQ_PREVFRILOW_GT_CURRTUECLOSE_ATRLOW_LONG_TP3_SL2_H5",
        ticker              = "HAL",
        direction           = "LONG",
        status              = "FORWARD_TEST",
        rule_type           = "HAL_PREVFRI_LOW_GT_TUE_CLOSE",
        full_history_start  = "2015-01-01",
        validation_start    = "2022-01-20",
        tp_atr              = 3.0,
        sl_atr              = 2.0,
        max_hold            = 5,
        cost_bps_per_side   = 2.0,
        slippage_bps_per_side = 2.0,
        notes=(
            "Signal: prev-week Friday Low > current-week Tuesday Close  AND  ATR% < 252-med. "
            "Entry next session open. ATR-based exit (TP 3×, SL 2×, max 5 sessions). "
            "Validation (2022+): CAGR 17.85%, excess +5.33%, Sharpe 1.36. FORWARD_TEST only."
        ),
        snapshot_trades     = 31,
        snapshot_win_rate   = 0.0,
        snapshot_cagr       = 17.85,
        snapshot_sharpe     = 1.36,
        snapshot_pf         = 4.11,
        extra_tags          = ["energy", "momentum", "sequence", "weekly-anchor", "ATR-low-regime"],
    ),

    # ── Strategy 4: CHRW LONG ────────────────────────────────────────────────
    StrategyConfig(
        strategy_id         = "CHRW_MOMSEQ_CLOSE8_LT_OPEN4_RSI40_LONG_TP3_SL2_H5",
        ticker              = "CHRW",
        direction           = "LONG",
        status              = "FORWARD_TEST",
        rule_type           = "CHRW_CLOSE8_LT_OPEN4_RSI40",
        full_history_start  = "2015-01-01",
        validation_start    = "2022-01-20",
        tp_atr              = 3.0,
        sl_atr              = 2.0,
        max_hold            = 5,
        cost_bps_per_side   = 2.0,
        slippage_bps_per_side = 2.0,
        notes=(
            "Signal: Close[t-8] < Open[t-4]  AND  RSI14 < 40. "
            "Entry next session open. ATR-based exit (TP 3×, SL 2×, max 5 sessions). "
            "Validation (2022+): CAGR 19.38%, excess +4.07%, Sharpe 1.29. FORWARD_TEST only."
        ),
        snapshot_trades     = 36,
        snapshot_win_rate   = 0.0,
        snapshot_cagr       = 19.38,
        snapshot_sharpe     = 1.29,
        snapshot_pf         = 3.54,
        extra_tags          = ["logistics", "momentum", "sequence", "RSI40"],
    ),

    # ── Add more strategies below ─────────────────────────────────────────────
    # StrategyConfig(
    #     strategy_id       = "XYZ_MOMSEQ_...",
    #     ticker            = "XYZ",
    #     direction         = "LONG",
    #     status            = "FORWARD_TEST",
    #     rule_type         = "XYZ_RULE",
    #     full_history_start= "2015-01-01",
    #     validation_start  = "2022-01-20",
    # ),
]


# ══════════════════════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker: str, start: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance not installed — run: pip install yfinance")
    end = datetime.today().strftime("%Y-%m-%d")
    raw = yf.download(ticker, start=start, end=end,
                      auto_adjust=True, progress=False, threads=True)
    if raw.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    # Handle MultiIndex columns from yfinance ≥0.2.x
    if isinstance(raw.columns, pd.MultiIndex):
        try:
            raw = raw.xs(ticker, axis=1, level=-1)
        except KeyError:
            raw.columns = raw.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close"]
    for c in required:
        if c not in raw.columns:
            raise RuntimeError(f"Missing column '{c}' for {ticker}")
    if "Volume" not in raw.columns:
        raw["Volume"] = np.nan
    raw = raw.dropna(subset=required).copy()
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    return raw.sort_index()


# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _atr14(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(14).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_l = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all indicator columns to df.
    Columns added:
      ATR14, ATR_PCT, ATR_PCT_MED (252-session rolling median, min 80)
      SMA20, SMA50, SMA200
      RSI14
      VOL_MA20 (20-day avg volume)
      DIR (U/D/F candle direction)
    """
    df = df.copy()

    # ATR family
    df["ATR14"]      = _atr14(df)
    df["ATR_PCT"]    = df["ATR14"] / df["Close"]
    df["ATR_PCT_MED"]= df["ATR_PCT"].rolling(252, min_periods=80).median()
    df["ATR_LOW_REG"] = (df["ATR_PCT"] < df["ATR_PCT_MED"]).fillna(False)
    df["ATR_HIGH_REG"]= (df["ATR_PCT"] > df["ATR_PCT_MED"]).fillna(False)

    # Moving averages (optional — available for future strategies)
    df["SMA20"]  = df["Close"].rolling(20).mean()
    df["SMA50"]  = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    # RSI (optional — available for future strategies)
    df["RSI14"]  = _rsi(df["Close"], 14)

    # Volume MA
    df["VOL_MA20"] = df["Volume"].rolling(20).mean()

    # Directional candle
    df["DIR"] = np.where(df["Close"] > df["Open"], "U",
                np.where(df["Close"] < df["Open"], "D", "F"))

    return df


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL ENGINE — one function per rule_type
# ══════════════════════════════════════════════════════════════════════════════

def signal_appf_low10_open9(df: pd.DataFrame, cfg: StrategyConfig) -> pd.Series:
    """
    APPF_LOW10_OPEN9:
      Signal = Low[t-10] > Open[t-9]  AND  ATR_PCT[t] < ATR_PCT_MED[t]
    Uses only past data. No lookahead.
    """
    rule_ohlc = df["Low"].shift(10) > df["Open"].shift(9)
    rule_atr  = df["ATR_LOW_REG"]
    return (rule_ohlc & rule_atr).fillna(False)


def signal_wday_close3_lt_low2_sma20(df: pd.DataFrame, cfg: StrategyConfig) -> pd.Series:
    """
    WDAY SHORT:
      Signal = Close[t-3] < Low[t-2]  AND  Close[t] < SMA20[t]
    No lookahead — all values are known at signal candle close.
    """
    rule_seq  = df["Close"].shift(3) < df["Low"].shift(2)
    rule_sma  = df["Close"] < df["SMA20"]
    return (rule_seq & rule_sma).fillna(False)


def signal_hal_prevfri_low_gt_tue_close_atrlow(df: pd.DataFrame, cfg: StrategyConfig) -> pd.Series:
    """
    HAL LONG:
      Signal = prev-week Friday Low > current-week Tuesday Close
               AND ATR% < rolling 252-session median ATR%

    Implementation (no lookahead):
    - For each row at date t, identify:
        * The most recent completed Tuesday close ON OR BEFORE t
        * The most recent completed Friday low that falls in the PREVIOUS calendar week
          relative to that Tuesday
    - Signal is only valid when BOTH anchors exist in the data.

    Specifically:
      tue_close[t]  = Close of the last Tuesday <= t
      prev_fri_low[t] = Low of the last Friday that is < (start of the week containing tue_close[t])
                       i.e. belongs to the week before the Tuesday's week.

    "Week" = Mon–Sun ISO week.
    """
    # Build lookup: date -> Close/Low for fast access
    closes = df["Close"]
    lows   = df["Low"]
    dates  = df.index

    tue_close_series  = pd.Series(np.nan, index=dates, dtype=float)
    prev_fri_low_series = pd.Series(np.nan, index=dates, dtype=float)

    # Precompute: for each date that is a Tuesday, record its close
    # For each date that is a Friday, record its low
    tue_dates = dates[dates.dayofweek == 1]   # Tuesday = 1
    fri_dates = dates[dates.dayofweek == 4]   # Friday  = 4

    tue_close_map  = {d: float(closes.loc[d]) for d in tue_dates}
    fri_low_map    = {d: float(lows.loc[d])   for d in fri_dates}

    # For each signal row t, find the last Tuesday <= t,
    # then find the last Friday that is in the PREVIOUS ISO week
    sorted_tue = sorted(tue_close_map.keys())
    sorted_fri = sorted(fri_low_map.keys())

    for t in dates:
        # Last Tuesday on or before t
        last_tue = None
        for d in reversed(sorted_tue):
            if d <= t:
                last_tue = d
                break
        if last_tue is None:
            continue

        tue_close_series.loc[t] = tue_close_map[last_tue]

        # ISO week of the Tuesday
        tue_year, tue_week, _ = last_tue.isocalendar()

        # Find last Friday strictly in the PREVIOUS ISO week
        last_fri = None
        for d in reversed(sorted_fri):
            fy, fw, _ = d.isocalendar()
            # Previous week = same year/week-1, or year-1/last-week if week=1
            if fy < tue_year or (fy == tue_year and fw < tue_week):
                last_fri = d
                break
        if last_fri is None:
            continue

        prev_fri_low_series.loc[t] = fri_low_map[last_fri]

    rule_seq = prev_fri_low_series > tue_close_series
    rule_atr = df["ATR_LOW_REG"]
    return (rule_seq & rule_atr).fillna(False)


def signal_chrw_close8_lt_open4_rsi40(df: pd.DataFrame, cfg: StrategyConfig) -> pd.Series:
    """
    CHRW LONG:
      Signal = Close[t-8] < Open[t-4]  AND  RSI14[t] < 40
    No lookahead — all values known at signal candle close.
    """
    rule_seq = df["Close"].shift(8) < df["Open"].shift(4)
    rule_rsi = df["RSI14"] < 40
    return (rule_seq & rule_rsi).fillna(False)


# ── Register all signal functions here ───────────────────────────────────────
# key = rule_type string in StrategyConfig
# value = callable(df, cfg) -> pd.Series[bool]

SIGNAL_REGISTRY: Dict[str, Callable] = {
    "APPF_LOW10_OPEN9":              signal_appf_low10_open9,
    "WDAY_CLOSE3_LT_LOW2_SMA20":     signal_wday_close3_lt_low2_sma20,
    "HAL_PREVFRI_LOW_GT_TUE_CLOSE":  signal_hal_prevfri_low_gt_tue_close_atrlow,
    "CHRW_CLOSE8_LT_OPEN4_RSI40":    signal_chrw_close8_lt_open4_rsi40,
}


def compute_signals(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Add SIGNAL column to df using the registered function for cfg.rule_type."""
    fn = SIGNAL_REGISTRY.get(cfg.rule_type)
    if fn is None:
        raise ValueError(f"No signal function registered for rule_type='{cfg.rule_type}'")
    df = df.copy()
    df["SIGNAL"] = fn(df, cfg)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    signal_date:  pd.Timestamp
    entry_date:   pd.Timestamp
    exit_date:    pd.Timestamp
    direction:    str
    entry:        float
    exit:         float
    target:       float
    stop:         float
    atr_at_signal:float
    bars_held:    int
    return_pct:   float
    exit_reason:  str   # "TP" / "SL" / "TIMEOUT"


def simulate_trades(df: pd.DataFrame, cfg: StrategyConfig) -> List[Trade]:
    """
    Non-overlapping trade simulator.
    - Entry at next session open after signal.
    - ATR-based TP/SL from signal candle.
    - Conservative: if TP and SL both touched same session, SL wins.
    - Max hold exit at close of last session.
    - Supports LONG and SHORT.
    """
    opens   = df["Open"].to_numpy(float)
    highs   = df["High"].to_numpy(float)
    lows    = df["Low"].to_numpy(float)
    closes  = df["Close"].to_numpy(float)
    atrs    = df["ATR14"].to_numpy(float)
    signals = df["SIGNAL"].to_numpy(bool)
    dates   = list(df.index)

    rt_cost = 2 * (cfg.cost_bps_per_side + cfg.slippage_bps_per_side) / 10_000.0
    is_long = cfg.direction.upper() == "LONG"
    trades: List[Trade] = []
    in_trade_until: Optional[int] = None

    for i in range(len(df) - 1):
        if in_trade_until is not None and i <= in_trade_until:
            continue
        if not signals[i]:
            continue
        ei = i + 1
        if ei >= len(df):
            continue
        atr   = atrs[i]
        entry = opens[ei]
        if not (np.isfinite(atr) and np.isfinite(entry) and atr > 0 and entry > 0):
            continue

        if is_long:
            target = entry + cfg.tp_atr * atr
            stop   = entry - cfg.sl_atr * atr
        else:
            target = entry - cfg.tp_atr * atr
            stop   = entry + cfg.sl_atr * atr

        last_i = min(ei + cfg.max_hold - 1, len(df) - 1)
        exit_p = closes[last_i]
        exit_i = last_i
        reason = "TIMEOUT"

        for j in range(ei, last_i + 1):
            if is_long:
                sl_hit = lows[j]  <= stop
                tp_hit = highs[j] >= target
            else:
                sl_hit = highs[j] >= stop
                tp_hit = lows[j]  <= target

            if sl_hit:                  # stop first (conservative)
                exit_p = stop; exit_i = j; reason = "SL"; break
            if tp_hit:
                exit_p = target; exit_i = j; reason = "TP"; break

        if is_long:
            raw_ret = exit_p / entry - 1
        else:
            raw_ret = 1 - exit_p / entry

        net = raw_ret - rt_cost
        trades.append(Trade(
            signal_date   = dates[i],
            entry_date    = dates[ei],
            exit_date     = dates[exit_i],
            direction     = cfg.direction,
            entry         = float(entry),
            exit          = float(exit_p),
            target        = float(target),
            stop          = float(stop),
            atr_at_signal = float(atr),
            bars_held     = int(exit_i - ei + 1),
            return_pct    = float(net * 100),
            exit_reason   = reason,
        ))
        in_trade_until = exit_i

    return trades


def filter_trades(trades: List[Trade],
                  start: pd.Timestamp,
                  end: pd.Timestamp) -> List[Trade]:
    return [t for t in trades if start <= t.signal_date <= end]


# ══════════════════════════════════════════════════════════════════════════════
# METRICS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _equity_curve(df: pd.DataFrame, trades: List[Trade]) -> pd.Series:
    eq  = pd.Series(1.0, index=df.index, dtype=float)
    lut = {t.exit_date: t.return_pct / 100.0 for t in trades}
    cur = 1.0
    for dt in df.index:
        if dt in lut:
            cur *= 1 + lut[dt]
        eq.loc[dt] = cur
    return eq.ffill()


def _max_dd(eq: pd.Series) -> float:
    if eq.empty:
        return float("nan")
    return float((eq / eq.cummax() - 1).min() * 100)


def _cagr(total_ret: float, s: pd.Timestamp, e: pd.Timestamp) -> float:
    years = max((e - s).days / 365.25, 1e-9)
    return ((1 + total_ret) ** (1 / years) - 1) * 100


def _pf(rets: np.ndarray) -> float:
    g = rets[rets > 0].sum()
    l = -rets[rets < 0].sum()
    if l == 0:
        return float("inf") if g > 0 else 0.0
    return float(g / l)


def compute_metrics(df: pd.DataFrame,
                    trades: List[Trade],
                    label: str) -> Tuple[dict, Optional[pd.Series]]:
    bh_ret  = df["Close"].iloc[-1] / df["Close"].iloc[0] - 1
    bh_eq   = df["Close"] / df["Close"].iloc[0]
    bh_cagr = _cagr(bh_ret, df.index[0], df.index[-1])
    bh_dd   = _max_dd(bh_eq)

    empty = dict(
        period=label, start=df.index[0].date(), end=df.index[-1].date(),
        trades=0, win_rate=np.nan, total_return=0.0, cagr=0.0,
        sharpe=np.nan, profit_factor=np.nan, max_drawdown=np.nan,
        time_in_market=0.0, avg_bars=np.nan,
        bh_cagr=bh_cagr, bh_dd=bh_dd, excess_cagr=-bh_cagr,
    )
    if not trades:
        return empty, None

    rets     = np.array([t.return_pct / 100 for t in trades])
    tot      = float(np.prod(1 + rets) - 1)
    cagr_val = _cagr(tot, df.index[0], df.index[-1])
    eq       = _equity_curve(df, trades)
    daily    = eq.pct_change().fillna(0)
    sharpe   = float(np.sqrt(252) * daily.mean() / daily.std()) \
               if daily.std() > 0 else np.nan
    bars     = sum(t.bars_held for t in trades)

    m = dict(
        period=label, start=df.index[0].date(), end=df.index[-1].date(),
        trades=len(trades),
        win_rate=float((rets > 0).mean() * 100),
        total_return=float(tot * 100),
        cagr=cagr_val,
        sharpe=sharpe,
        profit_factor=_pf(rets),
        max_drawdown=_max_dd(eq),
        time_in_market=float(bars / len(df) * 100),
        avg_bars=float(np.mean([t.bars_held for t in trades])),
        bh_cagr=bh_cagr,
        bh_dd=bh_dd,
        excess_cagr=cagr_val - bh_cagr,
    )
    return m, eq


# ══════════════════════════════════════════════════════════════════════════════
# OPEN POSITION TRACKER
# ══════════════════════════════════════════════════════════════════════════════

def detect_open_trade(df: pd.DataFrame,
                      cfg: StrategyConfig) -> dict:
    """
    Look back from today through max_hold sessions for a recent signal
    that has not yet hit TP, SL, or timeout.
    Returns a dict describing the open trade status.
    """
    today      = df.index[-1]
    cur_close  = float(df["Close"].iloc[-1])
    is_long    = cfg.direction.upper() == "LONG"
    signals    = df["SIGNAL"]
    opens_s    = df["Open"]
    highs_s    = df["High"]
    lows_s     = df["Low"]
    closes_s   = df["Close"]
    atrs_s     = df["ATR14"]

    # Scan backwards up to max_hold + 1 bars for a signal
    lookback = min(cfg.max_hold + 2, len(df) - 1)
    for i in range(len(df) - 2, len(df) - 2 - lookback, -1):
        if i < 0:
            break
        if not signals.iloc[i]:
            continue
        ei = i + 1
        if ei >= len(df):
            continue

        atr   = float(atrs_s.iloc[i])
        entry = float(opens_s.iloc[ei])
        if not (np.isfinite(atr) and np.isfinite(entry) and atr > 0):
            continue

        if is_long:
            target = entry + cfg.tp_atr * atr
            stop   = entry - cfg.sl_atr * atr
        else:
            target = entry - cfg.tp_atr * atr
            stop   = entry + cfg.sl_atr * atr

        signal_date = df.index[i]
        entry_date  = df.index[ei]
        last_i      = min(ei + cfg.max_hold - 1, len(df) - 1)
        age         = len(df) - 1 - ei + 1   # sessions since entry (inclusive)

        # Check what happened in the hold window up to today
        status = "OPEN_WITHIN_HOLD_WINDOW"
        for j in range(ei, min(len(df), last_i + 1)):
            h = float(highs_s.iloc[j])
            l = float(lows_s.iloc[j])
            if is_long:
                if l <= stop:   status = "WOULD_HAVE_HIT_STOP";   break
                if h >= target: status = "WOULD_HAVE_HIT_TARGET";  break
            else:
                if h >= stop:   status = "WOULD_HAVE_HIT_STOP";   break
                if l <= target: status = "WOULD_HAVE_HIT_TARGET";  break

        if status == "OPEN_WITHIN_HOLD_WINDOW" and age > cfg.max_hold:
            status = "WOULD_HAVE_HIT_TIMEOUT"

        if is_long:
            unreal = (cur_close / entry - 1) * 100
        else:
            unreal = (1 - cur_close / entry) * 100

        return dict(
            status       = status,
            signal_date  = signal_date.date(),
            entry_date   = entry_date.date(),
            entry        = round(entry, 4),
            target       = round(target, 4),
            stop         = round(stop, 4),
            atr          = round(atr, 4),
            cur_close    = round(cur_close, 4),
            age          = age,
            max_hold     = cfg.max_hold,
            unrealized   = round(unreal, 2),
        )

    return dict(status="NO_OPEN_TRADE_DETECTED")


# ══════════════════════════════════════════════════════════════════════════════
# CHARTING
# ══════════════════════════════════════════════════════════════════════════════

BG    = "#0E1117"
GRID  = "#2D2D2D"
TEXT  = "#E0E0E0"
CYAN  = "#00C4FF"
AMBER = "#FFB300"
GREEN = "#26C281"
RED   = "#FF4B4B"
PALE  = "#7FFFFF"
GOLD  = "#FFD54F"


def _base_layout(title: str = "") -> dict:
    return dict(
        title=dict(text=title, font=dict(color=TEXT, size=13)),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TEXT),
        xaxis=dict(gridcolor=GRID, showgrid=True),
        yaxis=dict(gridcolor=GRID, showgrid=True),
        margin=dict(l=50, r=20, t=40, b=40),
        hovermode="x unified",
        legend=dict(bgcolor="#1A1A2E", bordercolor=GRID),
    )


def equity_chart(df_full, eq_full, df_val, eq_val, cfg: StrategyConfig):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35], vertical_spacing=0.04,
                        subplot_titles=["Equity (normalised)", "Drawdown (%)"])

    bh_f = df_full["Close"] / df_full["Close"].iloc[0]
    bh_v = df_val["Close"]  / df_val["Close"].iloc[0]

    fig.add_trace(go.Scatter(x=eq_full.index, y=eq_full.values,
        name="Strategy (full)", line=dict(color=CYAN, width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=bh_f.index, y=bh_f.values,
        name="B&H (full)", line=dict(color=AMBER, width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=eq_val.index, y=eq_val.values,
        name="Strategy (val)", line=dict(color=PALE, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=bh_v.index, y=bh_v.values,
        name="B&H (val)", line=dict(color=GOLD, width=1.5, dash="dash")), row=1, col=1)

    dd_f = (eq_full / eq_full.cummax() - 1) * 100
    dd_v = (eq_val  / eq_val.cummax()  - 1) * 100
    fig.add_trace(go.Scatter(x=dd_f.index, y=dd_f.values, name="DD (full)",
        fill="tozeroy", line=dict(color=RED, width=1),
        fillcolor="rgba(255,75,75,0.15)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=dd_v.index, y=dd_v.values, name="DD (val)",
        line=dict(color="#FF8A80", width=1.5, dash="dash")), row=2, col=1)

    fig.update_layout(**_base_layout(f"{cfg.ticker} — {cfg.strategy_id}"))
    return fig


def trade_bar_chart(trades: List[Trade], label: str):
    if not trades:
        return go.Figure()
    df = pd.DataFrame([{"date": t.signal_date,
                         "ret":  t.return_pct,
                         "reason": t.exit_reason} for t in trades])
    colors = [GREEN if r >= 0 else RED for r in df["ret"]]
    fig = go.Figure(go.Bar(
        x=df["date"], y=df["ret"], marker_color=colors,
        text=df["reason"], textposition="outside",
        hovertemplate="%{x}<br>Return: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(**_base_layout(f"Trade returns — {label}"))
    fig.update_yaxes(title_text="Return (%)")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# CSV HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _to_csv(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


def trades_to_df(trades: List[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([t.__dict__ for t in trades])


# ══════════════════════════════════════════════════════════════════════════════
# LATEST SIGNAL PANEL
# ══════════════════════════════════════════════════════════════════════════════

def _get_rule_legs(df: pd.DataFrame, cfg: StrategyConfig) -> List[Tuple[str, str, bool]]:
    """
    Return a list of (label, value_str, passed) tuples describing
    each rule leg for the latest bar, for display in the signal panel.
    """
    latest = df.iloc[-1]

    def _safe(series, label):
        v = series.iloc[-1] if len(series) > 0 else None
        return float(v) if (v is not None and np.isfinite(float(v))) else None

    legs: List[Tuple[str, str, bool]] = []

    if cfg.rule_type == "APPF_LOW10_OPEN9":
        low10 = _safe(df["Low"].shift(10),  "low10")
        open9 = _safe(df["Open"].shift(9),  "open9")
        ohlc_ok = low10 is not None and open9 is not None and low10 > open9
        atr_ok  = bool(latest.get("ATR_LOW_REG", False))
        legs.append(("Low[t-10] > Open[t-9]",
                      f"{low10:.2f} > {open9:.2f}" if (low10 and open9) else "n/a",
                      ohlc_ok))
        legs.append(("ATR% < 252-med (low regime)",
                      f"{float(latest['ATR_PCT'])*100:.2f}% < {float(latest['ATR_PCT_MED'])*100:.2f}%"
                      if not pd.isna(latest.get("ATR_PCT_MED", float("nan"))) else "warming up",
                      atr_ok))

    elif cfg.rule_type == "WDAY_CLOSE3_LT_LOW2_SMA20":
        c3   = _safe(df["Close"].shift(3), "c3")
        l2   = _safe(df["Low"].shift(2),   "l2")
        sma  = _safe(df["SMA20"],          "sma20")
        seq_ok = c3 is not None and l2 is not None and c3 < l2
        sma_ok = sma is not None and float(latest["Close"]) < sma
        legs.append(("Close[t-3] < Low[t-2]",
                      f"{c3:.2f} < {l2:.2f}" if (c3 and l2) else "n/a",
                      seq_ok))
        legs.append(("Close < SMA20",
                      f"{float(latest['Close']):.2f} < {sma:.2f}" if sma else "n/a",
                      sma_ok))

    elif cfg.rule_type == "HAL_PREVFRI_LOW_GT_TUE_CLOSE":
        # Recompute the two anchors for the latest row for display only
        dates    = df.index
        lows_s   = df["Low"]
        closes_s = df["Close"]
        fri_dates = dates[dates.dayofweek == 4]
        tue_dates = dates[dates.dayofweek == 1]
        t = dates[-1]
        last_tue = next((d for d in reversed(tue_dates) if d <= t), None)
        prev_fri = None
        if last_tue is not None:
            tue_year, tue_week, _ = last_tue.isocalendar()
            prev_fri = next(
                (d for d in reversed(fri_dates)
                 if d.isocalendar()[0] < tue_year
                 or (d.isocalendar()[0] == tue_year and d.isocalendar()[1] < tue_week)),
                None)
        pfl = float(lows_s.loc[prev_fri]) if prev_fri is not None else None
        tc  = float(closes_s.loc[last_tue]) if last_tue is not None else None
        seq_ok = pfl is not None and tc is not None and pfl > tc
        atr_ok = bool(latest.get("ATR_LOW_REG", False))
        legs.append(("Prev-Fri Low > Curr-Tue Close",
                      f"{pfl:.2f} > {tc:.2f}" if (pfl and tc) else "n/a",
                      seq_ok))
        legs.append(("ATR% < 252-med (low regime)",
                      f"{float(latest['ATR_PCT'])*100:.2f}% < {float(latest['ATR_PCT_MED'])*100:.2f}%"
                      if not pd.isna(latest.get("ATR_PCT_MED", float("nan"))) else "warming up",
                      atr_ok))

    elif cfg.rule_type == "CHRW_CLOSE8_LT_OPEN4_RSI40":
        c8  = _safe(df["Close"].shift(8), "c8")
        o4  = _safe(df["Open"].shift(4),  "o4")
        rsi = float(latest.get("RSI14", float("nan")))
        seq_ok = c8 is not None and o4 is not None and c8 < o4
        rsi_ok = np.isfinite(rsi) and rsi < 40
        legs.append(("Close[t-8] < Open[t-4]",
                      f"{c8:.2f} < {o4:.2f}" if (c8 and o4) else "n/a",
                      seq_ok))
        legs.append(("RSI14 < 40",
                      f"{rsi:.1f}" if np.isfinite(rsi) else "n/a",
                      rsi_ok))

    return legs


def render_signal_panel(df: pd.DataFrame, cfg: StrategyConfig):
    latest   = df.iloc[-1]
    sig_date = df.index[-1].date()
    close    = float(latest["Close"])
    atr_raw  = latest["ATR14"]
    atr      = float(atr_raw) if (not pd.isna(atr_raw) and np.isfinite(float(atr_raw))) else None
    atr_pct  = float(latest["ATR_PCT"]) if np.isfinite(float(latest["ATR_PCT"])) else None
    atr_med_raw = latest.get("ATR_PCT_MED", float("nan"))
    atr_med  = float(atr_med_raw) \
               if (not pd.isna(atr_med_raw) and np.isfinite(float(atr_med_raw))) else None

    sig_active = bool(latest.get("SIGNAL", False))

    if sig_active:
        label = "🟢 ACTIVE_NEW_SIGNAL"; color = GREEN
    elif atr is None:
        label = "⚠️ DATA_ERROR";        color = AMBER
    else:
        label = "🔴 NO_NEW_SIGNAL";     color = RED

    st.markdown(f"""
    <div style='background:#1A1A2E;border-radius:10px;padding:14px 20px;
                border-left:5px solid {color};margin-bottom:14px'>
      <span style='color:{color};font-size:1.3em;font-weight:700'>{label}</span>
      <span style='color:#888;margin-left:16px'>as of {sig_date}</span>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Close",    f"{close:.2f}")
    c2.metric("ATR14",    f"{atr:.2f}"          if atr     else "n/a")
    c3.metric("ATR%",     f"{atr_pct*100:.2f}%" if atr_pct else "n/a")
    c4.metric("ATR% Med", f"{atr_med*100:.2f}%" if atr_med else "n/a")

    # Generic rule legs
    legs = _get_rule_legs(df, cfg)
    cols = st.columns(max(len(legs) * 2, 4))
    for i, (lbl, val, passed) in enumerate(legs):
        cols[i * 2].metric(lbl, val,
                            delta="✓" if passed else "✗",
                            delta_color="normal" if passed else "inverse")

    if sig_active and atr:
        is_long = cfg.direction.upper() == "LONG"
        est_target = close + cfg.tp_atr * atr if is_long else close - cfg.tp_atr * atr
        est_stop   = close - cfg.sl_atr * atr if is_long else close + cfg.sl_atr * atr
        dir_label  = "LONG" if is_long else "SHORT"
        st.success(
            f"**Signal ACTIVE — {dir_label}.** Entry is next session **OPEN** — not today's close.\n\n"
            f"⚠️ Estimates use today's close as proxy only. "
            f"**Recalculate TP/SL from actual next open before placing trade.**\n\n"
            f"Est. entry ≈ {close:.2f} | "
            f"Est. target ≈ {est_target:.2f} ({'+'if is_long else '-'}{cfg.tp_atr}×ATR) | "
            f"Est. stop ≈ {est_stop:.2f} ({'-'if is_long else '+'}{cfg.sl_atr}×ATR) | "
            f"ATR = {atr:.2f} | Max hold = {cfg.max_hold} sessions"
        )

    return sig_active, close, atr


# ══════════════════════════════════════════════════════════════════════════════
# OPEN TRADE PANEL
# ══════════════════════════════════════════════════════════════════════════════

def render_open_trade_panel(ot: dict):
    status = ot["status"]
    color_map = {
        "OPEN_WITHIN_HOLD_WINDOW":  GREEN,
        "NO_OPEN_TRADE_DETECTED":   "#555",
        "WOULD_HAVE_HIT_TARGET":    AMBER,
        "WOULD_HAVE_HIT_STOP":      RED,
        "WOULD_HAVE_HIT_TIMEOUT":   "#888",
    }
    color = color_map.get(status, "#555")

    st.markdown(f"""
    <div style='background:#12122A;border-radius:8px;padding:12px 18px;
                border-left:4px solid {color};margin-bottom:10px'>
      <b style='color:{color}'>{status}</b>
    </div>""", unsafe_allow_html=True)

    if status == "NO_OPEN_TRADE_DETECTED":
        st.caption("No signal within the current max-hold window.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Signal date",  str(ot.get("signal_date", "—")))
    c2.metric("Entry date",   str(ot.get("entry_date",  "—")))
    c3.metric("Entry",        f"{ot.get('entry', 0):.2f}")
    c4.metric("ATR at signal",f"{ot.get('atr', 0):.2f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Target",       f"{ot.get('target', 0):.2f}")
    c6.metric("Stop",         f"{ot.get('stop', 0):.2f}")
    c7.metric("Current close",f"{ot.get('cur_close', 0):.2f}")
    unreal = ot.get("unrealized", 0)
    c8.metric("Unrealized",   f"{unreal:+.2f}%",
              delta_color="normal" if unreal >= 0 else "inverse")

    age = ot.get("age", 0); mh = ot.get("max_hold", 10)
    st.progress(min(age / mh, 1.0), text=f"Session {age} / {mh}")


# ══════════════════════════════════════════════════════════════════════════════
# METRICS TABLE
# ══════════════════════════════════════════════════════════════════════════════

def render_metrics_table(m_full: dict, m_val: dict, cfg: StrategyConfig):
    def f(v, suffix="%", dec=2):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "n/a"
        if isinstance(v, float):
            return f"{v:.{dec}f}{suffix}"
        return str(v)

    rows = [
        ("Period",          m_full["period"],                        m_val["period"]),
        ("Start",           str(m_full["start"]),                    str(m_val["start"])),
        ("End",             str(m_full["end"]),                      str(m_val["end"])),
        ("Trades",          m_full["trades"],                        m_val["trades"]),
        ("Win rate",        f(m_full["win_rate"]),                   f(m_val["win_rate"])),
        ("Total return",    f(m_full["total_return"]),               f(m_val["total_return"])),
        ("CAGR",            f(m_full["cagr"]),                       f(m_val["cagr"])),
        ("Sharpe",          f(m_full["sharpe"], "", 3),              f(m_val["sharpe"], "", 3)),
        ("Profit factor",   f(m_full["profit_factor"], "", 3),       f(m_val["profit_factor"], "", 3)),
        ("Max drawdown",    f(m_full["max_drawdown"]),               f(m_val["max_drawdown"])),
        ("Time in market",  f(m_full["time_in_market"]),             f(m_val["time_in_market"])),
        ("Avg bars held",   f(m_full["avg_bars"], " bars", 1),       f(m_val["avg_bars"], " bars", 1)),
        ("B&H CAGR",        f(m_full["bh_cagr"]),                    f(m_val["bh_cagr"])),
        ("B&H max DD",      f(m_full["bh_dd"]),                      f(m_val["bh_dd"])),
        ("Excess CAGR",     f"{m_full['excess_cagr']:+.2f}%",        f"{m_val['excess_cagr']:+.2f}%"),
    ]
    tbl = pd.DataFrame(rows, columns=["Metric", "Full history", "Validation period"])
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    # Snapshot drift warning
    if cfg.snapshot_trades > 0 and m_val["trades"] >= cfg.snapshot_trades:
        drift = abs(m_val["cagr"] - cfg.snapshot_cagr)
        if drift > 3.0:
            st.warning(
                f"⚠️ Validation CAGR drifted {drift:.1f}pp from locked snapshot "
                f"({cfg.snapshot_cagr:.2f}% → {m_val['cagr']:.2f}%). Review assumptions."
            )


# ══════════════════════════════════════════════════════════════════════════════
# TELEGRAM HELPER (optional)
# ══════════════════════════════════════════════════════════════════════════════

def telegram_send(token: str, chat_id: str, text: str) -> Tuple[bool, str]:
    if requests is None:
        return False, "requests not installed"
    try:
        url  = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url,
                             json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
                             timeout=10)
        return (True, "OK") if resp.status_code == 200 else (False, resp.text)
    except Exception as e:
        return False, str(e)


def render_telegram_panel(sig_active: bool, status_label: str,
                           close: float, atr: Optional[float],
                           cfg: StrategyConfig):
    with st.expander("📡 Telegram Alerts", expanded=False):
        st.caption("Session-only. Credentials not stored.")
        col1, col2 = st.columns(2)
        token   = col1.text_input("Bot token", type="password",
                                   placeholder="123456:ABC-…",
                                   key=f"tg_tok_{cfg.strategy_id}")
        chat_id = col2.text_input("Chat ID",
                                   placeholder="-1001234567890",
                                   key=f"tg_cid_{cfg.strategy_id}")
        lines = [
            f"<b>📊 {cfg.ticker} — {cfg.strategy_id}</b>",
            f"Status: {status_label}",
            f"Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        ]
        if sig_active and atr:
            lines += [
                f"Est. entry ≈ {close:.2f}",
                f"Est. target ≈ {close + cfg.tp_atr*atr:.2f}  (+{cfg.tp_atr}×ATR)",
                f"Est. stop  ≈ {close - cfg.sl_atr*atr:.2f}  (−{cfg.sl_atr}×ATR)",
                f"ATR14 = {atr:.2f}",
                "",
                "⚠️ Recalculate from actual next-session open.",
                "Research only — not financial advice.",
            ]
        else:
            lines.append("No new signal today.")

        msg = st.text_area("Message (editable)", value="\n".join(lines),
                            height=180, key=f"tg_msg_{cfg.strategy_id}")
        if st.button("Send to Telegram", key=f"tg_btn_{cfg.strategy_id}"):
            if not token or not chat_id:
                st.error("Enter both bot token and chat ID.")
            else:
                ok, info = telegram_send(token, chat_id, msg)
                st.success("✅ Sent!") if ok else st.error(f"❌ Failed: {info}")


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY TAB
# ══════════════════════════════════════════════════════════════════════════════

def run_strategy_tab(cfg: StrategyConfig):
    # Header card
    st.markdown(f"""
    <div style='background:#12122A;border-radius:8px;padding:10px 18px;
                border:1px solid #2D2D5E;margin-bottom:12px'>
      <b style='color:{CYAN}'>{cfg.strategy_id}</b>
      &nbsp;|&nbsp;
      <span style='color:#888'>{cfg.ticker} · {cfg.direction} · </span>
      <span style='color:{AMBER}'>{cfg.status}</span>
      &nbsp;|&nbsp;
      <span style='color:#666;font-size:0.85em'>{cfg.notes}</span>
    </div>""", unsafe_allow_html=True)

    st.warning("⚠️ **RESEARCH / FORWARD-TEST ONLY. NOT FINANCIAL ADVICE. NOT A PRODUCTION TRADING SIGNAL.**")

    # Download data
    with st.spinner(f"Downloading {cfg.ticker}…"):
        try:
            df_raw = fetch_data(cfg.ticker, cfg.full_history_start)
        except Exception as e:
            st.error(f"Data error: {e}")
            return

    df = add_indicators(df_raw)
    df = compute_signals(df, cfg)

    # Drop rows with NaN in required columns (keep ATR_PCT_MED NaN rows for early warm-up)
    req = ["Open", "High", "Low", "Close", "ATR14", "ATR_PCT"]
    df  = df.dropna(subset=req).copy()

    if df.empty:
        st.error("Not enough data after indicator calculation.")
        return

    # ── Latest signal ─────────────────────────────────────────────────────────
    st.subheader("📌 Latest Signal")
    sig_active, close, atr = render_signal_panel(df, cfg)

    # ── Open position tracker ─────────────────────────────────────────────────
    st.subheader("🔍 Open Position Tracker")
    ot = detect_open_trade(df, cfg)
    render_open_trade_panel(ot)

    # ── Backtest ──────────────────────────────────────────────────────────────
    val_start  = pd.Timestamp(cfg.validation_start)
    df_full    = df.copy()
    df_val     = df[df.index >= val_start].copy()

    all_trades = simulate_trades(df, cfg)
    val_trades = filter_trades(all_trades, df_val.index[0], df_val.index[-1]) \
                 if not df_val.empty else []

    m_full, eq_full = compute_metrics(df_full, all_trades, "FULL_HISTORY")
    m_val,  eq_val  = compute_metrics(df_val,  val_trades, "VALIDATION_PERIOD")
    if eq_full is None: eq_full = pd.Series(1.0, index=df_full.index)
    if eq_val  is None: eq_val  = pd.Series(1.0, index=df_val.index)

    # ── Metrics table ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Backtest Metrics")
    render_metrics_table(m_full, m_val, cfg)

    # ── Charts ────────────────────────────────────────────────────────────────
    st.subheader("📈 Equity & Drawdown")
    st.plotly_chart(equity_chart(df_full, eq_full, df_val, eq_val, cfg),
                    use_container_width=True)

    tab_f, tab_v = st.tabs(["Full history trades", "Validation trades"])
    with tab_f:
        st.plotly_chart(trade_bar_chart(all_trades, "Full history"),
                        use_container_width=True)
    with tab_v:
        st.plotly_chart(trade_bar_chart(val_trades, "Validation period"),
                        use_container_width=True)

    # ── Recent trades table ───────────────────────────────────────────────────
    st.subheader("🗂️ Recent Trades (validation period)")
    vt_df = trades_to_df(val_trades)
    if vt_df.empty:
        st.info("No validation trades found.")
    else:
        st.dataframe(vt_df.tail(25).reset_index(drop=True), use_container_width=True)

    # ── CSV exports ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⬇️ Export CSV")
    c1, c2, c3, c4, c5 = st.columns(5)

    sig_row = df.iloc[-1]
    active_df = pd.DataFrame([{
        "ticker": cfg.ticker, "date": df.index[-1].date(),
        "close": round(close, 4),
        "atr14": round(atr, 4) if atr else None,
        "signal_active": sig_active,
        "status": cfg.status,
    }])
    ot_df = pd.DataFrame([ot])
    metrics_df = pd.DataFrame([m_full, m_val])

    c1.download_button("active_signals.csv",
                        _to_csv(active_df),
                        f"active_signals_{cfg.ticker}.csv", "text/csv")
    c2.download_button("open_trade.csv",
                        _to_csv(ot_df),
                        f"open_trade_{cfg.ticker}.csv", "text/csv")
    c3.download_button("metrics.csv",
                        _to_csv(metrics_df),
                        f"metrics_{cfg.ticker}.csv", "text/csv")
    c4.download_button("trades_full_history.csv",
                        _to_csv(trades_to_df(all_trades)),
                        f"trades_full_{cfg.ticker}.csv", "text/csv")
    c5.download_button("trades_validation.csv",
                        _to_csv(vt_df),
                        f"trades_val_{cfg.ticker}.csv", "text/csv")

    # ── Telegram ──────────────────────────────────────────────────────────────
    st.markdown("---")
    status_label = "🟢 ACTIVE" if sig_active else "🔴 NO SIGNAL"
    render_telegram_panel(sig_active, status_label, close, atr, cfg)

    return m_full, m_val


# ══════════════════════════════════════════════════════════════════════════════
# RANKINGS TAB
# ══════════════════════════════════════════════════════════════════════════════

def render_rankings(all_metrics: dict):
    st.subheader("🏆 Strategy Rankings — Validation Period")
    if not all_metrics:
        st.info("Run strategy tabs first to populate rankings.")
        return

    rows = []
    for sid, (mf, mv) in all_metrics.items():
        rows.append({
            "Strategy ID":      sid,
            "Val Trades":       mv["trades"],
            "Val Win%":         f"{mv['win_rate']:.1f}%" if not math.isnan(mv.get("win_rate", float("nan"))) else "n/a",
            "Val CAGR":         f"{mv['cagr']:.2f}%",
            "Val Excess CAGR":  f"{mv['excess_cagr']:+.2f}%",
            "Val Sharpe":       f"{mv['sharpe']:.3f}" if not math.isnan(mv.get("sharpe", float("nan"))) else "n/a",
            "Val MaxDD":        f"{mv['max_drawdown']:.2f}%",
            "Full CAGR":        f"{mf['cagr']:.2f}%",
            "Full MaxDD":       f"{mf['max_drawdown']:.2f}%",
        })

    df = pd.DataFrame(rows)
    # Sort by validation excess CAGR descending
    try:
        df["_sort"] = df["Val Excess CAGR"].str.replace("%","").str.replace("+","").astype(float)
        df = df.sort_values("_sort", ascending=False).drop(columns=["_sort"])
    except Exception:
        pass
    st.dataframe(df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():


    st.markdown("""
    <style>
      .stApp { background-color: #0E1117; }
      .block-container { padding-top: 1rem; }
      [data-testid="stMetricDeltaIcon-Up"]   { color: #26C281 !important; }
      [data-testid="stMetricDeltaIcon-Down"] { color: #FF4B4B !important; }
    </style>""", unsafe_allow_html=True)

    st.title("🧪 Momentum Sequence Lab — Forward Test Dashboard")
    st.caption(
        "⚠️ **RESEARCH / FORWARD-TEST ONLY. NOT FINANCIAL ADVICE. NOT PRODUCTION-READY.** "
        "Add new StrategyConfig entries + signal functions to extend this dashboard."
    )

    tab_labels = [f"{s.ticker} · {s.strategy_id}" for s in STRATEGIES] + ["🏆 Rankings"]
    tabs = st.tabs(tab_labels)

    all_metrics: dict = {}

    for i, cfg in enumerate(STRATEGIES):
        with tabs[i]:
            result = run_strategy_tab(cfg)
            if result:
                all_metrics[cfg.strategy_id] = result

    with tabs[-1]:
        render_rankings(all_metrics)


if __name__ == "__main__":
    main()
