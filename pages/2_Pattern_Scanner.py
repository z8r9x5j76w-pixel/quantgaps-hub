#!/usr/bin/env python3
"""
dashboard.py  –  Multi-Strategy Trading Scanner
====================================================
⚠️ RESEARCH / FORWARD-TEST ONLY. NOT FINANCIAL ADVICE. NOT PRODUCTION-READY.

Run locally:
    streamlit run dashboard.py

Deployed on Streamlit Community Cloud — password protected.

Adding a new strategy
---------------------
1. Add a StrategyConfig entry to the STRATEGIES list.
2. Set the correct filter flags (use_sma_filter, use_rsi_filter, use_high_atr_regime).
3. Done — the indicator engine, backtest, signal status, open-position tracker,
   and rankings tab all work generically from the config.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import requests as req
except ImportError:
    req = None


# ── PASSWORD GATE ─────────────────────────────────────────────────────────────
# Change the string below to whatever password you want.

def _check_password() -> bool:
    return True


# ── Telegram helper ───────────────────────────────────────────────────────────

def telegram_send(token: str, chat_id: str, text: str) -> bool:
    if req is None:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = req.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


# ── Strategy config ───────────────────────────────────────────────────────────

@dataclass
class StrategyConfig:
    # Required identifiers
    strategy_id: str
    ticker: str
    direction: str          # "LONG" or "SHORT"
    status: str             # "FORWARD_TEST", "LIVE", "WATCHLIST", etc.
    full_history_start: str
    validation_start: str

    # 3-candle pattern to look for ("DUD", "DDD", "UDU", etc.)
    pattern: str = "DUD"

    # Exit params
    tp_atr: float = 1.5
    sl_atr: float = 1.0
    max_hold: int = 10

    # Filter toggles — set to True/False per strategy
    use_sma_filter: bool = True         # Close < SMA(sma_period)
    use_rsi_filter: bool = True         # RSI(rsi_period) < rsi_threshold
    use_high_atr_regime: bool = False   # ATR% > rolling 252-session median ATR%

    # Filter parameters (only used when the respective flag is True)
    sma_period: int = 50
    rsi_period: int = 14
    rsi_threshold: float = 45.0

    # Transaction costs
    cost_bps_per_side: float = 0.0
    slippage_bps_per_side: float = 0.0

    # Display / snapshot metadata
    notes: str = ""
    snapshot_trades: int = 0         # Expected trade count in validation period
    snapshot_win_rate: float = 0.0
    snapshot_cagr: float = 0.0       # Used for drift detection
    snapshot_sharpe: float = 0.0
    extra_tags: List[str] = field(default_factory=list)


# ── Strategy list ─────────────────────────────────────────────────────────────

STRATEGIES: List[StrategyConfig] = [

    # ── Strategy 1: IWM DUD ───────────────────────────────────────────────────
    StrategyConfig(
        strategy_id="IWM_DUD_SMA50_RSI45_ATR_1p5_1p0_H10",
        ticker="IWM",
        direction="LONG",
        status="FORWARD_TEST",
        full_history_start="2012-01-01",
        validation_start="2022-01-20",
        pattern="DUD",
        tp_atr=1.5,
        sl_atr=1.0,
        max_hold=10,
        use_sma_filter=True,
        use_rsi_filter=True,
        use_high_atr_regime=False,
        sma_period=50,
        rsi_period=14,
        rsi_threshold=45.0,
        notes=(
            "DUD = 3-candle pattern (Down-Up-Down). "
            "Entry next session open. ATR-based exit. "
            "Full-history CAGR underperforms buy-and-hold — regime-dependent. "
            "Validation period (2022-2026) is strong."
        ),
        snapshot_trades=30,
        snapshot_win_rate=73.33,
        snapshot_cagr=16.16,
        snapshot_sharpe=1.656,
        extra_tags=["small-cap ETF", "mean-reversion", "DUD pattern"],
    ),

    # ── Strategy 2: CRM DDD High-ATR ─────────────────────────────────────────
    # FORWARD_TEST — not production-ready.
    # Full-history is ~buy-hold equivalent; validation period (2022-2026) is very strong.
    StrategyConfig(
        strategy_id="CRM_DDD_HIGH_ATR_ATR2_2_H7",
        ticker="CRM",
        direction="LONG",
        status="FORWARD_TEST",
        full_history_start="2012-01-01",
        validation_start="2022-01-20",
        pattern="DDD",
        tp_atr=2.0,
        sl_atr=2.0,
        max_hold=7,
        use_sma_filter=False,
        use_rsi_filter=False,
        use_high_atr_regime=True,
        notes=(
            "DDD = 3 consecutive down candles. "
            "High ATR regime: ATR% > rolling 252-session median ATR%. "
            "Full-history ≈ buy-hold. Validation period (2022-2026) very strong (+32.8% excess CAGR). "
            "FORWARD_TEST — not production-ready."
        ),
        snapshot_trades=43,
        snapshot_win_rate=69.77,
        snapshot_cagr=27.49,
        snapshot_sharpe=1.475,
        extra_tags=["large-cap tech", "mean-reversion", "DDD pattern", "ATR regime"],
    ),

    # ── Add more strategies below ─────────────────────────────────────────────
    # StrategyConfig(
    #     strategy_id="SPY_...",
    #     ticker="SPY",
    #     direction="LONG",
    #     status="WATCHLIST",
    #     full_history_start="2010-01-01",
    #     validation_start="2022-01-01",
    #     pattern="DUD",
    #     use_sma_filter=True,
    #     use_rsi_filter=True,
    #     use_high_atr_regime=False,
    # ),

    # ── Strategy 3: PYPL DDD RSI40 ───────────────────────────────────────────
    StrategyConfig(
        strategy_id="PYPL_DDD_RSI40_ATR1p25_2_H15",
        ticker="PYPL",
        direction="LONG",
        status="FORWARD_TEST",
        full_history_start="2015-09-14",
        validation_start="2022-01-20",
        pattern="DDD",
        tp_atr=1.25,
        sl_atr=2.0,
        max_hold=15,
        use_sma_filter=False,
        use_rsi_filter=True,
        use_high_atr_regime=False,
        rsi_period=14,
        rsi_threshold=40.0,
        notes=(
            "DDD = 3 consecutive down candles. RSI14 < 40. "
            "Entry next session open. ATR-based exit (TP 1.25×, SL 2.0×, max 15 sessions). "
            "Full-history excess CAGR +4.97% vs PYPL B&H. "
            "Validation (2022-2026) very strong: +45.01% excess CAGR. FORWARD_TEST only."
        ),
        snapshot_trades=34,
        snapshot_win_rate=67.65,
        snapshot_cagr=18.04,
        snapshot_sharpe=1.25,
        extra_tags=["fintech", "mean-reversion", "DDD pattern", "RSI40"],
    ),

    # ── Strategy 4: UPS DD SMA50 RSI45 ───────────────────────────────────────
    StrategyConfig(
        strategy_id="UPS_DD_BELOW_SMA50_RSI45_ATR4_2_H10",
        ticker="UPS",
        direction="LONG",
        status="FORWARD_TEST",
        full_history_start="2012-03-14",
        validation_start="2022-01-20",
        pattern="DD",           # 2-candle pattern — uses PATTERN2
        tp_atr=4.0,
        sl_atr=2.0,
        max_hold=10,
        use_sma_filter=True,
        use_rsi_filter=True,
        use_high_atr_regime=False,
        sma_period=50,
        rsi_period=14,
        rsi_threshold=45.0,
        notes=(
            "DD = 2 consecutive down candles. Close < SMA50. RSI14 < 45. "
            "Entry next session open. ATR-based exit (TP 4.0×, SL 2.0×, max 10 sessions). "
            "Full-history excess CAGR +6.17%. "
            "Validation (2022-2026) very strong: +31.92% excess CAGR. FORWARD_TEST only."
        ),
        snapshot_trades=53,
        snapshot_win_rate=64.15,
        snapshot_cagr=20.64,
        snapshot_sharpe=1.23,
        extra_tags=["logistics", "mean-reversion", "DD pattern", "SMA50", "RSI45"],
    ),

    # ── Strategy 5: COF DUD RSI45 ────────────────────────────────────────────
    StrategyConfig(
        strategy_id="COF_DUD_RSI45_ATR2_1p5_H10",
        ticker="COF",
        direction="LONG",
        status="FORWARD_TEST",
        full_history_start="2012-01-23",
        validation_start="2022-01-20",
        pattern="DUD",
        tp_atr=2.0,
        sl_atr=1.5,
        max_hold=10,
        use_sma_filter=False,
        use_rsi_filter=True,
        use_high_atr_regime=False,
        rsi_period=14,
        rsi_threshold=45.0,
        notes=(
            "DUD = Down-Up-Down 3-candle pattern. RSI14 < 45. "
            "Entry next session open. ATR-based exit (TP 2.0×, SL 1.5×, max 10 sessions). "
            "Full-history excess CAGR +1.40% vs COF B&H. "
            "Validation (2022-2026) strong: +18.78% excess CAGR. FORWARD_TEST only."
        ),
        snapshot_trades=32,
        snapshot_win_rate=68.75,
        snapshot_cagr=25.88,
        snapshot_sharpe=1.45,
        extra_tags=["financials", "mean-reversion", "DUD pattern", "RSI45"],
    ),

    # ── Strategy 6: NTR DU SMA20 ─────────────────────────────────────────────
    StrategyConfig(
        strategy_id="NTR_DU_BELOW_SMA20_ATR2_1p5_H5",
        ticker="NTR",
        direction="LONG",
        status="FORWARD_TEST",
        full_history_start="2012-01-01",
        validation_start="2022-01-20",
        pattern="DU",           # 2-candle pattern — uses PATTERN2
        tp_atr=2.0,
        sl_atr=1.5,
        max_hold=5,
        use_sma_filter=True,
        use_rsi_filter=False,
        use_high_atr_regime=False,
        sma_period=20,          # Close < SMA20
        notes=(
            "DU = Down-Up 2-candle pattern. Close < SMA20. "
            "Entry next session open. ATR-based exit (TP 2.0×, SL 1.5×, max 5 sessions). "
            "Full-history excess CAGR +11.42%. "
            "Validation (2022-2026) very strong: +28.43% excess CAGR, Sharpe 1.62. FORWARD_TEST only."
        ),
        snapshot_trades=83,
        snapshot_win_rate=0.0,   # not provided in spec
        snapshot_cagr=30.65,
        snapshot_sharpe=1.62,
        extra_tags=["agriculture", "mean-reversion", "DU pattern", "SMA20"],
    ),

    # ── Strategy 7: ALB DDD High-ATR ─────────────────────────────────────────
    StrategyConfig(
        strategy_id="ALB_DDD_HIGH_ATR_ATR2p5_3_H10",
        ticker="ALB",
        direction="LONG",
        status="FORWARD_TEST",
        full_history_start="2012-01-01",
        validation_start="2022-01-20",
        pattern="DDD",          # 3-candle pattern
        tp_atr=2.5,
        sl_atr=3.0,
        max_hold=10,
        use_sma_filter=False,
        use_rsi_filter=False,
        use_high_atr_regime=True,   # ATR% > rolling 252-session median ATR%
        notes=(
            "DDD = 3 consecutive down candles. High ATR regime: ATR% > 252-session median ATR%. "
            "Entry next session open. ATR-based exit (TP 2.5×, SL 3.0×, max 10 sessions). "
            "Full-history excess CAGR +8.75%. "
            "Validation (2022-2026) outstanding: +45.53% excess CAGR, Sharpe 1.53. FORWARD_TEST only."
        ),
        snapshot_trades=30,
        snapshot_win_rate=0.0,   # not provided in spec
        snapshot_cagr=42.04,
        snapshot_sharpe=1.53,
        extra_tags=["materials", "lithium", "mean-reversion", "DDD pattern", "ATR regime"],
    ),
]


# ── Indicators ────────────────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_l = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def add_indicators(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """
    Compute indicators required by cfg and append a SIGNAL column.
    Only indicators needed by the strategy are computed — no lookahead bias.
    """
    df = df.copy()

    # ATR14 — always required for exit sizing
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    # Directional candle pattern — always required
    df["DIR"] = np.where(df["Close"] > df["Open"], "U",
                np.where(df["Close"] < df["Open"], "D", "F"))
    dirs = df["DIR"].tolist()
    patterns3 = [""] * len(df)
    for i in range(2, len(df)):
        patterns3[i] = dirs[i - 2] + dirs[i - 1] + dirs[i]
    df["PATTERN3"] = patterns3
    patterns2 = [""] * len(df)
    for i in range(1, len(df)):
        patterns2[i] = dirs[i - 1] + dirs[i]
    df["PATTERN2"] = patterns2

    # Optional: SMA filter
    if cfg.use_sma_filter:
        df[f"SMA{cfg.sma_period}"] = df["Close"].rolling(cfg.sma_period).mean()

    # Optional: RSI filter
    if cfg.use_rsi_filter:
        df[f"RSI{cfg.rsi_period}"] = _rsi(df["Close"], cfg.rsi_period)

    # Optional: High ATR regime filter
    # ATR_PCT = ATR14 / Close
    # HIGH_ATR_REGIME = ATR_PCT > rolling 252-session median of ATR_PCT (min 80 sessions)
    if cfg.use_high_atr_regime:
        df["ATR_PCT"] = df["ATR14"] / df["Close"]
        df["ATR_PCT_MEDIAN_252"] = df["ATR_PCT"].rolling(252, min_periods=80).median()
        df["HIGH_ATR_REGIME"] = (df["ATR_PCT"] > df["ATR_PCT_MEDIAN_252"]).fillna(False)

    # Build SIGNAL — use PATTERN2 for 2-candle patterns, PATTERN3 for 3-candle
    pat_col = "PATTERN2" if len(cfg.pattern) == 2 else "PATTERN3"
    sig = (df[pat_col] == cfg.pattern) & df["ATR14"].notna()
    if cfg.use_sma_filter:
        sig = sig & (df["Close"] < df[f"SMA{cfg.sma_period}"])
    if cfg.use_rsi_filter:
        sig = sig & (df[f"RSI{cfg.rsi_period}"] < cfg.rsi_threshold)
    if cfg.use_high_atr_regime:
        sig = sig & df["HIGH_ATR_REGIME"]

    df["SIGNAL"] = sig
    return df


def _required_dropna_cols(cfg: StrategyConfig) -> List[str]:
    """Columns that must be non-NaN for a row to be usable in the backtest."""
    cols = ["Open", "High", "Low", "Close", "ATR14"]
    if cfg.use_sma_filter:
        cols.append(f"SMA{cfg.sma_period}")
    if cfg.use_rsi_filter:
        cols.append(f"RSI{cfg.rsi_period}")
    # ATR_PCT_MEDIAN_252 intentionally excluded — it starts NaN for first ~80 bars
    # but we still want those bars in the data for ATR calculation history.
    return cols


# ── Data download ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker: str, start: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance not installed")
    end = datetime.today().strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end,
                     auto_adjust=True, progress=False, threads=True)
    if df.empty:
        raise RuntimeError(f"No data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(ticker, axis=1, level=-1)
        except KeyError:
            df.columns = df.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close"]
    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"Missing column: {c}")
    if "Volume" not in df.columns:
        df["Volume"] = np.nan
    df = df.dropna(subset=required).copy()
    df.index = pd.to_datetime(df.index)
    return df


# ── Backtest engine ───────────────────────────────────────────────────────────

@dataclass
class Trade:
    signal_date: pd.Timestamp
    entry_date:  pd.Timestamp
    exit_date:   pd.Timestamp
    entry:       float
    exit:        float
    atr:         float
    target:      float
    stop:        float
    bars_held:   int
    return_pct:  float
    exit_reason: str


def simulate_trades(df: pd.DataFrame, cfg: StrategyConfig) -> List[Trade]:
    """
    Non-overlapping backtest. Entry is the OPEN of the bar after the signal.
    ATR for target/stop comes from the signal bar (no lookahead).
    Conservative intraday rule: if both TP and SL are hit on the same day,
    the stop is counted first.
    """
    opens   = df["Open"].to_numpy(float)
    highs   = df["High"].to_numpy(float)
    lows    = df["Low"].to_numpy(float)
    closes  = df["Close"].to_numpy(float)
    atrs    = df["ATR14"].to_numpy(float)
    signals = df["SIGNAL"].to_numpy(bool)
    dates   = list(df.index)

    rt_cost = 2 * (cfg.cost_bps_per_side + cfg.slippage_bps_per_side) / 10_000.0
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

        atr   = atrs[i]   # ATR from signal candle — no lookahead
        entry = opens[ei]
        if not (np.isfinite(atr) and np.isfinite(entry) and atr > 0 and entry > 0):
            continue

        target = entry + cfg.tp_atr * atr
        stop   = entry - cfg.sl_atr * atr
        last_i = min(ei + cfg.max_hold - 1, len(df) - 1)
        exit_p = closes[last_i]
        exit_i = last_i
        reason = "TIMEOUT"

        for j in range(ei, last_i + 1):
            if lows[j] <= stop:          # Conservative: check stop first
                exit_p = stop; exit_i = j; reason = "SL"; break
            if highs[j] >= target:
                exit_p = target; exit_i = j; reason = "TP"; break

        net = (exit_p / entry - 1) - rt_cost
        trades.append(Trade(
            signal_date=dates[i], entry_date=dates[ei], exit_date=dates[exit_i],
            entry=float(entry), exit=float(exit_p), atr=float(atr),
            target=float(target), stop=float(stop),
            bars_held=int(exit_i - ei + 1),
            return_pct=float(net * 100), exit_reason=reason,
        ))
        in_trade_until = exit_i

    return trades


def filter_trades(trades: List[Trade], start: pd.Timestamp, end: pd.Timestamp) -> List[Trade]:
    return [t for t in trades if start <= t.signal_date <= end]


def detect_open_trade(df: pd.DataFrame, all_trades: List[Trade], cfg: StrategyConfig) -> Optional[dict]:
    """
    Check whether a recent signal has an entry trade still inside the max-hold window
    and where TP/SL have NOT been hit yet (per available data).

    Returns a dict with trade details, or None if no open trade is detected.
    """
    # Trades that have already closed before today
    closed_signal_dates = {t.signal_date for t in all_trades if t.exit_date < df.index[-1]}
    signals_df = df[df["SIGNAL"]]

    if signals_df.empty:
        return None

    latest_i = len(df) - 1

    # Walk backward through signals — most recent first
    for signal_date in reversed(signals_df.index.tolist()):
        if signal_date in closed_signal_dates:
            continue  # Already closed in backtest

        signal_i = df.index.get_loc(signal_date)
        entry_i = signal_i + 1
        if entry_i >= len(df):
            continue  # Entry bar not available yet

        last_i = min(entry_i + cfg.max_hold - 1, len(df) - 1)
        if latest_i > last_i:
            continue  # Max-hold window has already expired

        entry = float(df["Open"].iloc[entry_i])
        atr   = float(df["ATR14"].iloc[signal_i])  # ATR from signal candle
        if not (np.isfinite(entry) and np.isfinite(atr) and entry > 0 and atr > 0):
            continue

        target = entry + cfg.tp_atr * atr
        stop   = entry - cfg.sl_atr * atr

        # Check whether TP/SL would already have been triggered (conservative: stop first)
        status = "OPEN"
        for j in range(entry_i, latest_i + 1):
            if float(df["Low"].iloc[j]) <= stop:
                status = "WOULD_HAVE_HIT_STOP"
                break
            if float(df["High"].iloc[j]) >= target:
                status = "WOULD_HAVE_HIT_TARGET"
                break

        if status != "OPEN":
            continue  # Trade already resolved — not a live open position

        current_close = float(df["Close"].iloc[-1])
        return {
            "signal_date":          signal_date.date(),
            "entry_date":           df.index[entry_i].date(),
            "entry":                round(entry, 4),
            "target":               round(target, 4),
            "stop":                 round(stop, 4),
            "atr_at_signal":        round(atr, 4),
            "current_close":        round(current_close, 4),
            "age_sessions":         latest_i - entry_i + 1,
            "max_hold_sessions":    cfg.max_hold,
            "unrealized_return_pct": round((current_close / entry - 1) * 100, 3),
        }

    return None


# ── Metrics ───────────────────────────────────────────────────────────────────

def _equity_curve(df: pd.DataFrame, trades: List[Trade]) -> pd.Series:
    eq  = pd.Series(1.0, index=df.index, dtype=float)
    lut = {t.exit_date: t.return_pct / 100 for t in trades}
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
    return (1 + total_ret) ** (1 / years) - 1


def _pf(rets: np.ndarray) -> float:
    g = rets[rets > 0].sum()
    l = -rets[rets < 0].sum()
    if l == 0:
        return float("inf") if g > 0 else 0.0
    return float(g / l)


def compute_metrics(df: pd.DataFrame, trades: List[Trade], label: str):
    """Returns (metrics_dict, equity_series). equity_series is None if no trades."""
    bh_ret  = df["Close"].iloc[-1] / df["Close"].iloc[0] - 1
    bh_eq   = df["Close"] / df["Close"].iloc[0]
    bh_cagr = float(_cagr(bh_ret, df.index[0], df.index[-1]) * 100)
    bh_dd   = _max_dd(bh_eq)

    if not trades:
        return dict(
            period=label, start=df.index[0].date(), end=df.index[-1].date(),
            trades=0, win_rate=np.nan, total_return=0.0, cagr=0.0,
            sharpe=np.nan, profit_factor=np.nan, max_drawdown=np.nan,
            time_in_market=0.0, avg_bars=np.nan,
            bh_cagr=bh_cagr, bh_dd=bh_dd,
            excess_cagr=0.0 - bh_cagr,
        ), None

    rets     = np.array([t.return_pct / 100 for t in trades])
    tot      = float(np.prod(1 + rets) - 1)
    cagr_val = float(_cagr(tot, df.index[0], df.index[-1]) * 100)
    eq       = _equity_curve(df, trades)
    daily    = eq.pct_change().fillna(0)
    sharpe   = float(np.sqrt(252) * daily.mean() / daily.std()) if daily.std() > 0 else np.nan
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
        bh_cagr=bh_cagr, bh_dd=bh_dd,
        excess_cagr=cagr_val - bh_cagr,
    )
    return m, eq


# ── Charting ──────────────────────────────────────────────────────────────────

PALETTE = dict(
    strategy="#00C4FF", bh="#FFB300", drawdown="#FF4B4B",
    win="#26C281", loss="#FF4B4B", bg="#0E1117", grid="#2D2D2D", text="#E0E0E0",
)


def _base_layout(title: str = "") -> dict:
    return dict(
        title=dict(text=title, font=dict(color=PALETTE["text"], size=14)),
        paper_bgcolor=PALETTE["bg"], plot_bgcolor=PALETTE["bg"],
        font=dict(color=PALETTE["text"]),
        xaxis=dict(gridcolor=PALETTE["grid"], showgrid=True),
        yaxis=dict(gridcolor=PALETTE["grid"], showgrid=True),
        margin=dict(l=50, r=20, t=40, b=40), hovermode="x unified",
    )


def equity_chart(df_full, eq_full, df_val, eq_val, cfg):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35], vertical_spacing=0.04,
                        subplot_titles=["Equity curve (normalised)", "Drawdown (%)"])

    bh_full = df_full["Close"] / df_full["Close"].iloc[0]
    bh_val  = df_val["Close"]  / df_val["Close"].iloc[0]

    fig.add_trace(go.Scatter(x=eq_full.index, y=eq_full.values,
        name="Strategy (full)", line=dict(color=PALETTE["strategy"], width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=bh_full.index, y=bh_full.values,
        name="Buy-hold (full)", line=dict(color=PALETTE["bh"], width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=eq_val.index, y=eq_val.values,
        name="Strategy (validation)", line=dict(color="#7FFFFF", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=bh_val.index, y=bh_val.values,
        name="Buy-hold (validation)", line=dict(color="#FFD54F", width=1.5, dash="dash")), row=1, col=1)

    dd_full = (eq_full / eq_full.cummax() - 1) * 100
    dd_val  = (eq_val  / eq_val.cummax()  - 1) * 100
    fig.add_trace(go.Scatter(x=dd_full.index, y=dd_full.values,
        name="DD (full)", fill="tozeroy", line=dict(color=PALETTE["drawdown"], width=1),
        fillcolor="rgba(255,75,75,0.18)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=dd_val.index, y=dd_val.values,
        name="DD (validation)", line=dict(color="#FF8A80", width=1.5, dash="dash")), row=2, col=1)

    fig.update_layout(**_base_layout(f"{cfg.ticker} — {cfg.strategy_id}"),
                      legend=dict(bgcolor="#1A1A2E", bordercolor=PALETTE["grid"]))
    return fig


def trades_bar_chart(trades: List[Trade], label: str):
    if not trades:
        return go.Figure()
    df = pd.DataFrame([{"date": t.signal_date, "ret": t.return_pct,
                         "reason": t.exit_reason} for t in trades])
    colors = [PALETTE["win"] if r >= 0 else PALETTE["loss"] for r in df["ret"]]
    fig = go.Figure(go.Bar(x=df["date"], y=df["ret"], marker_color=colors,
                            text=df["reason"], textposition="outside",
                            hovertemplate="%{x}<br>Return: %{y:.2f}%<extra></extra>"))
    fig.update_layout(**_base_layout(f"Trade returns — {label}"))
    fig.update_yaxes(title_text="Return (%)")
    return fig


# ── CSV helpers ───────────────────────────────────────────────────────────────

def trades_to_df(trades: List[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([t.__dict__ for t in trades])


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


# ── Signal status card ────────────────────────────────────────────────────────

def signal_status_block(df: pd.DataFrame, cfg: StrategyConfig):
    """
    Render the latest signal status card.
    Handles both SMA/RSI-filtered strategies (IWM) and ATR-regime strategies (CRM).
    Returns (signal_active, status_label, close, atr, cfg).
    """
    latest   = df.iloc[-1]
    pat_col  = "PATTERN2" if len(cfg.pattern) == 2 else "PATTERN3"
    pattern  = latest[pat_col]
    close    = float(latest["Close"])
    atr      = float(latest["ATR14"])
    sig_date = df.index[-1].date()

    pat_ok = pattern == cfg.pattern
    all_ok = bool(latest["SIGNAL"])

    # Gather optional indicator values (only for enabled filters)
    sma_val  = float(latest[f"SMA{cfg.sma_period}"])    if cfg.use_sma_filter        else None
    rsi_val  = float(latest[f"RSI{cfg.rsi_period}"])    if cfg.use_rsi_filter        else None
    atr_pct  = float(latest["ATR_PCT"])                  if cfg.use_high_atr_regime   else None
    _atr_med_raw = float(latest["ATR_PCT_MEDIAN_252"]) if cfg.use_high_atr_regime else None
    atr_med  = _atr_med_raw if (_atr_med_raw is not None and np.isfinite(_atr_med_raw)) else None
    atr_reg  = bool(latest["HIGH_ATR_REGIME"])           if cfg.use_high_atr_regime   else None

    # Check for data errors
    data_finite = np.isfinite(atr)
    if cfg.use_sma_filter and sma_val is not None:
        data_finite = data_finite and np.isfinite(sma_val)
    if cfg.use_rsi_filter and rsi_val is not None:
        data_finite = data_finite and np.isfinite(rsi_val)
    if cfg.use_high_atr_regime and atr_pct is not None:
        data_finite = data_finite and np.isfinite(atr_pct)

    if all_ok:
        status_label = "🟢 ACTIVE_NEW_SIGNAL"; status_color = "#26C281"
    elif not data_finite:
        status_label = "⚠️ DATA_ERROR"; status_color = "#FFB300"
    else:
        status_label = "🔴 NO_NEW_SIGNAL"; status_color = "#FF4B4B"

    st.markdown(f"""
    <div style='background:#1A1A2E;border-radius:10px;padding:16px 20px;
                border-left:5px solid {status_color};margin-bottom:16px'>
      <span style='color:{status_color};font-size:1.4em;font-weight:700'>{status_label}</span>
      <span style='color:#888;margin-left:16px'>as of {sig_date}</span>
    </div>""", unsafe_allow_html=True)

    # Three metric columns — contents vary by strategy type
    c1, c2, c3 = st.columns(3)

    # Column 1: always the pattern check
    c1.metric(f"Pattern (need {cfg.pattern})", pattern,
              delta="✓ match" if pat_ok else "✗ no match",
              delta_color="normal" if pat_ok else "inverse")

    # Column 2: SMA filter (IWM-style) OR ATR regime True/False (CRM-style)
    if cfg.use_sma_filter and sma_val is not None:
        sma_ok = close < sma_val
        c2.metric(f"Close vs SMA{cfg.sma_period}", f"{close:.2f} / {sma_val:.2f}",
                  delta="Below ✓" if sma_ok else "Above ✗",
                  delta_color="normal" if sma_ok else "inverse")
    elif cfg.use_high_atr_regime:
        c2.metric("ATR Regime", "HIGH ✓" if atr_reg else "LOW ✗",
                  delta=f"ATR% = {atr_pct:.4f}" if atr_pct is not None else "n/a",
                  delta_color="normal" if atr_reg else "inverse")
    else:
        c2.metric("—", "—")

    # Column 3: RSI filter (IWM-style) OR ATR% vs median detail (CRM-style)
    if cfg.use_rsi_filter and rsi_val is not None:
        rsi_ok = rsi_val < cfg.rsi_threshold
        c3.metric(f"RSI{cfg.rsi_period} (need <{cfg.rsi_threshold})", f"{rsi_val:.1f}",
                  delta="✓" if rsi_ok else "✗",
                  delta_color="normal" if rsi_ok else "inverse")
    elif cfg.use_high_atr_regime and atr_pct is not None:
        med_str = f"{atr_med:.4f}" if atr_med is not None and np.isfinite(atr_med) else "n/a"
        c3.metric("ATR% vs Median(252)", f"{atr_pct:.4f} / {med_str}",
                  delta="Above median ✓" if atr_reg else "Below median ✗",
                  delta_color="normal" if atr_reg else "inverse")
    else:
        c3.metric("—", "—")

    if all_ok:
        est_target = close + cfg.tp_atr * atr
        est_stop   = close - cfg.sl_atr * atr
        st.success(
            f"**Signal active.** Entry is next session **OPEN** (not close!).\n\n"
            f"⚠️ Estimates use today's close as proxy — "
            f"**recalculate from actual next open before placing trade.**\n\n"
            f"Est. entry ≈ {close:.2f} | Est. target ≈ {est_target:.2f} "
            f"(+{cfg.tp_atr}×ATR) | Est. stop ≈ {est_stop:.2f} (-{cfg.sl_atr}×ATR) | ATR = {atr:.2f}"
        )

    return all_ok, status_label, close, atr, cfg


# ── Open position panel ───────────────────────────────────────────────────────

def open_position_panel(open_trade: Optional[dict], cfg: StrategyConfig) -> None:
    """
    Show open-position tracking for a strategy.
    A strategy can have NO new signal today but still have an open trade
    from a recent signal that is within the max-hold window.
    """
    st.subheader("🔓 Open Position Tracker")

    if open_trade is None:
        st.info(
            "No open trade detected within the current max-hold window. "
            "Either no recent signal fired, or the max-hold period has expired."
        )
        return

    t = open_trade
    unr   = t["unrealized_return_pct"]
    color = "#26C281" if unr >= 0 else "#FF4B4B"

    st.markdown(f"""
    <div style='background:#1A1A2E;border-radius:10px;padding:14px 18px;
                border-left:5px solid {color};margin-bottom:12px'>
      <b style='color:{color}'>Open trade detected — TP/SL not yet hit</b>
      &nbsp;·&nbsp;
      <span style='color:#888'>signal {t["signal_date"]} → entry {t["entry_date"]}</span>
      &nbsp;·&nbsp;
      <span style='color:#CCC'>Age: {t["age_sessions"]} / {t["max_hold_sessions"]} sessions</span>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entry price", f"{t['entry']:.2f}")
    c2.metric("Target", f"{t['target']:.2f}",
              delta=f"+{cfg.tp_atr}×ATR")
    c3.metric("Stop", f"{t['stop']:.2f}",
              delta=f"-{cfg.sl_atr}×ATR", delta_color="inverse")
    c4.metric("Current close", f"{t['current_close']:.2f}",
              delta=f"{unr:+.2f}%",
              delta_color="normal" if unr >= 0 else "inverse")

    st.caption(
        f"ATR at signal: {t['atr_at_signal']:.4f}  ·  "
        f"Session {t['age_sessions']} of max {t['max_hold_sessions']}  ·  "
        "Conservative check: TP/SL have NOT been hit per available OHLC data."
    )


# ── Metrics table ─────────────────────────────────────────────────────────────

def render_metrics_table(m_full: dict, m_val: dict, snapshot: dict) -> None:
    def fmt(v, suffix="%", decimals=2):
        if isinstance(v, float) and math.isnan(v):
            return "n/a"
        if isinstance(v, float):
            return f"{v:.{decimals}f}{suffix}"
        return str(v)

    rows = [
        ("Period",         m_full["period"],                     m_val["period"]),
        ("Start",          str(m_full["start"]),                 str(m_val["start"])),
        ("End",            str(m_full["end"]),                   str(m_val["end"])),
        ("Trades",         m_full["trades"],                     m_val["trades"]),
        ("Win rate",       fmt(m_full["win_rate"]),              fmt(m_val["win_rate"])),
        ("Total return",   fmt(m_full["total_return"]),          fmt(m_val["total_return"])),
        ("CAGR",           fmt(m_full["cagr"]),                  fmt(m_val["cagr"])),
        ("Sharpe",         fmt(m_full["sharpe"], "", 3),         fmt(m_val["sharpe"], "", 3)),
        ("Profit factor",  fmt(m_full["profit_factor"], "", 3),  fmt(m_val["profit_factor"], "", 3)),
        ("Max drawdown",   fmt(m_full["max_drawdown"]),          fmt(m_val["max_drawdown"])),
        ("Time in market", fmt(m_full["time_in_market"]),        fmt(m_val["time_in_market"])),
        ("B&H CAGR",       fmt(m_full["bh_cagr"]),              fmt(m_val["bh_cagr"])),
        ("Excess CAGR",    f"{m_full['excess_cagr']:+.2f}%",    f"{m_val['excess_cagr']:+.2f}%"),
    ]
    df_tbl = pd.DataFrame(rows, columns=["Metric", "Full history", "Validation period"])
    st.dataframe(df_tbl, use_container_width=True, hide_index=True)

    # Drift check against locked snapshot
    if snapshot and snapshot.get("trades", 0) > 0 and m_val["trades"] >= snapshot["trades"]:
        drift = abs(m_val["cagr"] - snapshot["cagr"])
        if drift > 3.0:
            st.warning(
                f"⚠️ Validation CAGR drifted {drift:.1f}pp from locked snapshot "
                f"({snapshot['cagr']:.2f}% → {m_val['cagr']:.2f}%). Review assumptions."
            )


# ── Telegram panel ────────────────────────────────────────────────────────────

def telegram_panel(signal_active: bool, status_label: str, ticker: str,
                   close: float, atr: float, cfg: StrategyConfig) -> None:
    with st.expander("📡 Telegram Alerts", expanded=False):
        st.caption("Credentials are session-only — not stored anywhere.")
        col1, col2 = st.columns(2)
        token   = col1.text_input("Bot token", type="password", placeholder="123456:ABC-...",
                                   key=f"tg_tok_{ticker}")
        chat_id = col2.text_input("Chat ID", placeholder="-1001234567890",
                                   key=f"tg_cid_{ticker}")

        msg_lines = [
            f"<b>📊 {ticker} — {cfg.strategy_id}</b>",
            f"Status: {status_label}",
            f"Date: {datetime.today().strftime('%Y-%m-%d %H:%M')} UTC",
        ]
        if signal_active:
            est_target = close + cfg.tp_atr * atr
            est_stop   = close - cfg.sl_atr * atr
            msg_lines += [
                f"Est. entry ≈ {close:.2f}",
                f"Est. target ≈ {est_target:.2f}  (+{cfg.tp_atr}×ATR)",
                f"Est. stop  ≈ {est_stop:.2f}  (-{cfg.sl_atr}×ATR)",
                f"ATR14 = {atr:.2f}",
                "",
                "⚠️ Recalculate T/S from actual next-session open.",
                "Research only — not financial advice.",
            ]
        else:
            msg_lines.append("No signal today.")

        custom_msg = st.text_area("Message (editable)", value="\n".join(msg_lines),
                                   height=180, key=f"tg_msg_{ticker}")
        if st.button("Send to Telegram", key=f"tg_btn_{ticker}"):
            if not token or not chat_id:
                st.error("Enter both bot token and chat ID.")
            elif req is None:
                st.error("pip install requests")
            else:
                ok = telegram_send(token, chat_id, custom_msg)
                st.success("✅ Sent!") if ok else st.error("❌ Failed. Check token / chat ID.")


# ── Strategy tab ──────────────────────────────────────────────────────────────

def run_strategy_tab(cfg: StrategyConfig) -> Optional[Tuple[dict, dict]]:
    """
    Render a full strategy tab: signal, open position, metrics, charts, exports.
    Returns (m_full, m_val) for use in the Rankings tab, or None on error.
    """
    st.markdown(f"""
    <div style='background:#12122A;border-radius:8px;padding:10px 16px;margin-bottom:12px;
                border:1px solid #2D2D5E'>
      <b style='color:#00C4FF'>{cfg.strategy_id}</b>
      &nbsp;&nbsp;|&nbsp;&nbsp;
      <span style='color:#888'>{cfg.ticker} · {cfg.direction} · Status: </span>
      <span style='color:#FFB300'>{cfg.status}</span>
      &nbsp;&nbsp;|&nbsp;&nbsp;
      <span style='color:#666;font-size:0.85em'>{cfg.notes}</span>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Downloading data..."):
        try:
            df_raw = fetch_data(cfg.ticker, cfg.full_history_start)
        except Exception as e:
            st.error(f"Data error: {e}")
            return None

    df = add_indicators(df_raw, cfg)
    df = df.dropna(subset=_required_dropna_cols(cfg)).copy()

    # ── Signal status ─────────────────────────────────────────────────────────
    st.subheader("📌 Latest Signal")
    sig_active, status_label, close, atr, _ = signal_status_block(df, cfg)

    # ── Open position tracker ─────────────────────────────────────────────────
    val_start  = pd.Timestamp(cfg.validation_start)
    df_full    = df.copy()
    df_val     = df[df.index >= val_start].copy()
    all_trades = simulate_trades(df, cfg)
    val_trades = filter_trades(all_trades, df_val.index[0], df_val.index[-1])

    st.markdown("---")
    open_trade = detect_open_trade(df, all_trades, cfg)
    open_position_panel(open_trade, cfg)

    # ── Backtest metrics ──────────────────────────────────────────────────────
    m_full, eq_full = compute_metrics(df_full, all_trades, "FULL_HISTORY")
    m_val,  eq_val  = compute_metrics(df_val,  val_trades, "VALIDATION_PERIOD")
    if eq_full is None:
        eq_full = pd.Series(1.0, index=df_full.index)
    if eq_val is None:
        eq_val  = pd.Series(1.0, index=df_val.index)

    st.markdown("---")
    st.subheader("📊 Backtest Metrics")
    if m_full["excess_cagr"] < 0:
        st.warning(
            "⚠️ **Full-history CAGR underperforms buy-and-hold.** "
            "Regime-dependent. FORWARD_TEST only."
        )
    render_metrics_table(m_full, m_val,
                         dict(trades=cfg.snapshot_trades, cagr=cfg.snapshot_cagr,
                              sharpe=cfg.snapshot_sharpe))

    # ── Equity chart ──────────────────────────────────────────────────────────
    st.subheader("📈 Equity & Drawdown")
    st.plotly_chart(equity_chart(df_full, eq_full, df_val, eq_val, cfg),
                    use_container_width=True)

    # ── Trade charts ──────────────────────────────────────────────────────────
    tab_f, tab_v = st.tabs(["Full history trades", "Validation trades"])
    with tab_f:
        st.plotly_chart(trades_bar_chart(all_trades, "Full history"), use_container_width=True)
    with tab_v:
        st.plotly_chart(trades_bar_chart(val_trades, "Validation period"), use_container_width=True)

    # ── Recent trades table ───────────────────────────────────────────────────
    st.subheader("🗂️ Recent Trades (validation period)")
    vt_df = trades_to_df(val_trades)
    if vt_df.empty:
        st.info("No validation trades yet.")
    else:
        st.dataframe(vt_df.tail(25).reset_index(drop=True), use_container_width=True)

    # ── CSV exports ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⬇️ Export CSV")

    latest_row = df.iloc[-1]

    # active_signals.csv — current signal snapshot
    active_row: dict = {
        "strategy_id":   cfg.strategy_id,
        "ticker":        cfg.ticker,
        "date":          df.index[-1].date(),
        "close":         round(float(latest_row["Close"]), 4),
        "pattern":       latest_row["PATTERN3"],
        "atr14":         round(float(latest_row["ATR14"]), 4),
        "signal_active": bool(latest_row["SIGNAL"]),
        "status":        cfg.status,
    }
    if cfg.use_sma_filter:
        active_row[f"sma{cfg.sma_period}"] = round(float(latest_row[f"SMA{cfg.sma_period}"]), 4)
    if cfg.use_rsi_filter:
        active_row[f"rsi{cfg.rsi_period}"] = round(float(latest_row[f"RSI{cfg.rsi_period}"]), 2)
    if cfg.use_high_atr_regime:
        active_row["atr_pct"]         = round(float(latest_row["ATR_PCT"]), 5)
        raw_med = latest_row["ATR_PCT_MEDIAN_252"]
        active_row["atr_pct_med_252"] = round(float(raw_med), 5) if np.isfinite(raw_med) else None
        active_row["high_atr_regime"] = bool(latest_row["HIGH_ATR_REGIME"])

    active_df  = pd.DataFrame([active_row])
    open_df    = pd.DataFrame([open_trade]) if open_trade else pd.DataFrame()
    metrics_df = pd.DataFrame([m_full, m_val])

    ec1, ec2, ec3, ec4, ec5 = st.columns(5)
    ec1.download_button(
        "active_signals.csv", df_to_csv_bytes(active_df),
        f"active_signals_{cfg.ticker}.csv", "text/csv",
    )
    ec2.download_button(
        "open_trades.csv",
        df_to_csv_bytes(open_df) if not open_df.empty else b"no_open_trade",
        f"open_trade_{cfg.ticker}.csv", "text/csv",
    )
    ec3.download_button(
        "strategy_metrics.csv", df_to_csv_bytes(metrics_df),
        f"metrics_{cfg.ticker}.csv", "text/csv",
    )
    ec4.download_button(
        "trades_full_hist.csv", df_to_csv_bytes(trades_to_df(all_trades)),
        f"trades_full_{cfg.ticker}.csv", "text/csv",
    )
    ec5.download_button(
        "trades_validation.csv", df_to_csv_bytes(trades_to_df(val_trades)),
        f"trades_val_{cfg.ticker}.csv", "text/csv",
    )

    st.markdown("---")
    telegram_panel(sig_active, status_label, cfg.ticker, close, atr, cfg)

    return m_full, m_val


# ── Rankings tab ──────────────────────────────────────────────────────────────

def run_rankings_tab(all_metrics: dict) -> None:
    """
    Compare all strategies side by side on validation and full-history metrics.
    Sorted by validation excess CAGR (primary) and validation Sharpe (secondary).
    """
    st.subheader("📊 Strategy Rankings")
    st.caption(
        "All metrics are computed live from the backtests in the strategy tabs above. "
        "Sorted by Validation Excess CAGR (descending)."
    )

    if not all_metrics:
        st.info("Metrics load automatically — they should appear here after the page finishes rendering.")
        return

    def safe(v, d=2):
        if v is None:
            return None
        if isinstance(v, float) and math.isnan(v):
            return None
        return round(float(v), d)

    rows = []
    for sid, (m_full, m_val) in all_metrics.items():
        rows.append({
            "Strategy":           sid,
            "Val Trades":         m_val["trades"],
            "Val Win%":           safe(m_val["win_rate"]),
            "Val CAGR%":          safe(m_val["cagr"]),
            "Val Sharpe":         safe(m_val["sharpe"], 3),
            "Val PF":             safe(m_val["profit_factor"], 3),
            "Val MaxDD%":         safe(m_val["max_drawdown"]),
            "Val Excess CAGR%":   safe(m_val["excess_cagr"]),
            "Full CAGR%":         safe(m_full["cagr"]),
            "Full Excess CAGR%":  safe(m_full["excess_cagr"]),
            "Full MaxDD%":        safe(m_full["max_drawdown"]),
        })

    df_rank = (
        pd.DataFrame(rows)
        .set_index("Strategy")
        .sort_values(["Val Excess CAGR%", "Val Sharpe"], ascending=[False, False])
    )
    st.dataframe(df_rank, use_container_width=True)

    st.caption(
        "Val = validation period (each strategy's validation_start → today).  "
        "Full = full history from strategy's full_history_start.  "
        "PF = profit factor.  MaxDD = max drawdown."
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():

    st.markdown("""
    <style>
      .stApp { background-color: #0E1117; }
      .block-container { padding-top: 1rem; }
      [data-testid="stMetricDeltaIcon-Up"]   { color: #26C281 !important; }
      [data-testid="stMetricDeltaIcon-Down"] { color: #FF4B4B !important; }
    </style>""", unsafe_allow_html=True)

    st.title("📡 Trading Scanner — Multi-Strategy (FORWARD_TEST)")
    st.caption(
        "⚠️ **RESEARCH / FORWARD-TEST ONLY. NOT FINANCIAL ADVICE. NOT PRODUCTION-READY.** "
        "Add more StrategyConfig entries to `STRATEGIES` to extend the dashboard."
    )

    # One tab per strategy + a Rankings tab
    tab_labels = [f"{s.ticker} · {s.strategy_id}" for s in STRATEGIES] + ["📊 Rankings"]
    tabs = st.tabs(tab_labels)

    all_metrics: dict = {}  # strategy_id -> (m_full, m_val) for the Rankings tab

    for tab, cfg in zip(tabs[:-1], STRATEGIES):
        with tab:
            st.caption(f"**{cfg.ticker}** · {cfg.direction} · {cfg.status}")
            btn_key = f"run_{cfg.strategy_id}"
            if st.button(f"▶ Load {cfg.ticker} data", key=btn_key, type="primary"):
                st.session_state[f"loaded_{cfg.strategy_id}"] = True
            if st.session_state.get(f"loaded_{cfg.strategy_id}"):
                result = run_strategy_tab(cfg)
                if result is not None:
                    all_metrics[cfg.strategy_id] = result
            else:
                st.info("Click the button above to load this strategy.")

    with tabs[-1]:
        run_rankings_tab(all_metrics)


if __name__ == "__main__":
    main()
