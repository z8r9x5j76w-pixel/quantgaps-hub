#!/usr/bin/env python3
"""
QuantGaps Research Hub v2 — single file, fast-loading.
5 scanners in tabs. Signals load instantly. Backtests on demand.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import io, math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(page_title="QuantGaps Hub", page_icon="📊",
                   layout="wide", initial_sidebar_state="collapsed")

# ── PASSWORD ──────────────────────────────────────────────────
def check_password():
    if st.session_state.get("_auth"): return True
    st.markdown("<style>.stApp{background:#080d18;color:#dde6f0}</style>",
                unsafe_allow_html=True)
    st.markdown("## 🔒 QuantGaps Research Hub")
    pw = st.text_input("Password", type="password")
    if st.button("Enter"):
        try:    correct = st.secrets["DASHBOARD_PASSWORD"]
        except: correct = "quantgaps2024"
        if pw == correct:
            st.session_state["_auth"] = True; st.rerun()
        else: st.error("Incorrect password.")
    return False

if not check_password(): st.stop()

st.markdown("""<style>
.stApp{background:#080d18;color:#dde6f0}
.block-container{padding-top:1rem}
[data-testid="stMetricDeltaIcon-Up"]{color:#26C281!important}
[data-testid="stMetricDeltaIcon-Down"]{color:#FF4B4B!important}
</style>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# SHARED DATA & INDICATOR HELPERS
# ═════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_long(ticker: str, start: str) -> pd.DataFrame:
    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if raw.empty: raise RuntimeError(f"No data: {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        try: raw = raw.xs(ticker, axis=1, level=-1)
        except: raw.columns = raw.columns.get_level_values(0)
    raw = raw.dropna(subset=["Open","High","Low","Close"]).copy()
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    if "Volume" not in raw.columns: raw["Volume"] = np.nan
    return raw.sort_index()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_short(ticker: str, bars: int = 120) -> pd.DataFrame:
    raw = yf.download(ticker, period=f"{bars}d", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex): raw.columns = raw.columns.get_level_values(0)
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    df = raw[["Open","High","Low","Close"]].dropna().reset_index()
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    return df.sort_values("Date").reset_index(drop=True)

def _rsi(s, p=14):
    d=s.diff(); g=d.clip(lower=0); l=-d.clip(upper=0)
    ag=g.ewm(alpha=1/p,adjust=False,min_periods=p).mean()
    al=l.ewm(alpha=1/p,adjust=False,min_periods=p).mean()
    return (100-100/(1+ag/al.replace(0,np.nan))).fillna(50)

def _atr14(df):
    pc=df["Close"].shift(1)
    tr=pd.concat([df["High"]-df["Low"],(df["High"]-pc).abs(),(df["Low"]-pc).abs()],axis=1).max(axis=1)
    return tr.rolling(14).mean()

def enrich(df):
    df=df.copy()
    df["ATR14"]=_atr14(df)
    df["ATR_PCT"]=df["ATR14"]/df["Close"]
    df["ATR_PCT_MED"]=df["ATR_PCT"].rolling(252,min_periods=80).median()
    df["ATR_LOW"]=df["ATR_PCT"]<df["ATR_PCT_MED"]
    df["SMA20"]=df["Close"].rolling(20).mean()
    df["SMA50"]=df["Close"].rolling(50).mean()
    df["RSI14"]=_rsi(df["Close"])
    df["DIR"]=np.where(df["Close"]>df["Open"],"U",np.where(df["Close"]<df["Open"],"D","F"))
    return df

def metrics(rets, s, e):
    if not len(rets): return {}
    years=max((e-s).days/365.25,0.01); tot=float(np.prod(1+rets)-1)
    cagr=((1+tot)**(1/years)-1)*100
    cum=np.cumprod(1+rets); pk=np.maximum.accumulate(cum)
    mdd=float((cum/pk-1).min()*100)
    sr=float(rets.mean()/rets.std()*np.sqrt(252)) if rets.std()>0 else 0
    return dict(n=len(rets),cagr=round(cagr,2),mdd=round(mdd,2),
                sharpe=round(sr,2),win=round((rets>0).mean()*100,1),
                avg=round(float(rets.mean()*100),2))

def show_metrics(m):
    if not m: st.info("No trades."); return
    c=st.columns(5)
    c[0].metric("CAGR",f"{m['cagr']:.1f}%"); c[1].metric("Win%",f"{m['win']:.1f}%")
    c[2].metric("MaxDD",f"{m['mdd']:.1f}%"); c[3].metric("Sharpe",f"{m['sharpe']:.2f}")
    c[4].metric("Trades",str(m['n']))

def sig_banner(ticker, note, signal, direction):
    c="#26C281" if signal else "#444"
    icon=f"🟢 {direction}" if signal else "⚪ NO SIGNAL"
    st.markdown(f"<div style='background:#0f1825;border-left:5px solid {c};"
                f"padding:10px 16px;border-radius:6px;margin-bottom:10px'>"
                f"<b style='color:{c}'>{ticker}</b>"
                f"<span style='color:#667;margin-left:10px;font-size:0.85em'>{note}</span>"
                f"<span style='float:right;color:{c};font-weight:700'>{icon}</span></div>",
                unsafe_allow_html=True)

def run_backtest_atr(df, tp_mult, sl_mult, max_hold, direction="LONG"):
    o=df["Open"].values; h=df["High"].values; l=df["Low"].values
    cl=df["Close"].values; a=df["ATR14"].values; s=df["SIGNAL"].values
    trades=[]; lock=None; is_long=direction=="LONG"
    for i in range(len(df)-1):
        if lock and i<=lock: continue
        if not s[i]: continue
        ei=i+1; atr=a[i]; en=o[ei]
        if not(np.isfinite(atr) and np.isfinite(en) and atr>0 and en>0): continue
        tp=en+tp_mult*atr if is_long else en-tp_mult*atr
        sl=en-sl_mult*atr if is_long else en+sl_mult*atr
        last=min(ei+max_hold-1,len(df)-1)
        ep=cl[last]; er=last; rsn="TO"
        for j in range(ei,last+1):
            sh=l[j]<=sl if is_long else h[j]>=sl
            th=h[j]>=tp if is_long else l[j]<=tp
            if sh: ep=sl; er=j; rsn="SL"; break
            if th: ep=tp; er=j; rsn="TP"; break
        ret=(ep/en-1) if is_long else (1-ep/en)
        trades.append({"ret":ret,"reason":rsn,"ed":df.index[ei],"xd":df.index[er]})
        lock=er
    return trades


# ═════════════════════════════════════════════════════════════
# TAB 1 — PATTERN SCANNER
# ═════════════════════════════════════════════════════════════

@dataclass
class PCfg:
    id:str; ticker:str; direction:str; pattern:str
    start:str; val_start:str
    tp:float=1.5; sl:float=1.0; mh:int=10
    use_sma:bool=True; use_rsi:bool=True; use_hatr:bool=False
    sma_p:int=50; rsi_thr:float=45.0; notes:str=""
    s_cagr:float=0; s_sharpe:float=0; s_win:float=0; s_tr:int=0

PS = [
    PCfg("IWM_DUD","IWM","LONG","DUD","2012-01-01","2022-01-20",1.5,1.0,10,True,True,False,50,45.0,"DUD·SMA50·RSI<45",16.16,1.656,73.33,30),
    PCfg("CRM_DDD","CRM","LONG","DDD","2012-01-01","2022-01-20",2.0,2.0,7,False,False,True,50,45.0,"DDD·High-ATR",27.49,1.475,69.77,43),
    PCfg("PYPL_DDD","PYPL","LONG","DDD","2015-09-14","2022-01-20",1.25,2.0,15,False,True,False,50,40.0,"DDD·RSI<40",18.04,1.25,67.65,34),
    PCfg("UPS_DD","UPS","LONG","DD","2012-03-14","2022-01-20",4.0,2.0,10,True,True,False,50,45.0,"DD·SMA50·RSI<45",20.64,1.23,64.15,53),
    PCfg("COF_DUD","COF","LONG","DUD","2012-01-23","2022-01-20",2.0,1.5,10,False,True,False,50,45.0,"DUD·RSI<45",25.88,1.45,68.75,32),
    PCfg("NTR_DU","NTR","LONG","DU","2012-01-01","2022-01-20",2.0,1.5,5,True,False,False,20,45.0,"DU·SMA20",30.65,1.62,0,83),
    PCfg("ALB_DDD","ALB","LONG","DDD","2012-01-01","2022-01-20",2.5,3.0,10,False,False,True,50,45.0,"DDD·High-ATR",42.04,1.53,0,30),
]

def build_ps(df, cfg: PCfg):
    df=enrich(df)
    if cfg.use_sma: df[f"SMA{cfg.sma_p}"]=df["Close"].rolling(cfg.sma_p).mean()
    dirs=df["DIR"].tolist(); n=len(df)
    p3=[""]*n; p2=[""]*n
    for i in range(2,n): p3[i]=dirs[i-2]+dirs[i-1]+dirs[i]
    for i in range(1,n): p2[i]=dirs[i-1]+dirs[i]
    df["P3"]=p3; df["P2"]=p2
    pc="P2" if len(cfg.pattern)==2 else "P3"
    sig=(df[pc]==cfg.pattern)&df["ATR14"].notna()
    if cfg.use_sma:  sig&=df["Close"]<df[f"SMA{cfg.sma_p}"]
    if cfg.use_rsi:  sig&=df["RSI14"]<cfg.rsi_thr
    if cfg.use_hatr: sig&=(df["ATR_PCT"]>df["ATR_PCT_MED"]).fillna(False)
    df["SIGNAL"]=sig; return df

def render_ps(cfg: PCfg):
    with st.spinner(f"Loading {cfg.ticker}…"):
        try: raw=fetch_long(cfg.ticker,cfg.start)
        except Exception as e: st.error(str(e)); return
    df=build_ps(raw,cfg); df=df.dropna(subset=["ATR14"]).copy()
    sig=bool(df["SIGNAL"].iloc[-1]); close=float(df["Close"].iloc[-1]); atr=float(df["ATR14"].iloc[-1])
    sig_banner(cfg.ticker, cfg.notes, sig, cfg.direction)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Close",f"${close:.2f}"); c2.metric("ATR14",f"{atr:.2f}")
    c3.metric("TP ref",f"${close+cfg.tp*atr:.2f}"); c4.metric("SL ref",f"${close-cfg.sl*atr:.2f}")
    st.caption(f"Snapshot (val) — CAGR {cfg.s_cagr:.1f}% · Sharpe {cfg.s_sharpe:.2f} · Win {cfg.s_win:.0f}% · {cfg.s_tr} trades")
    with st.expander("📊 Backtest (click to run)"):
        if st.button(f"▶ {cfg.ticker}", key=f"psbt_{cfg.id}"):
            tds=run_backtest_atr(df,cfg.tp,cfg.sl,cfg.mh,cfg.direction)
            if tds:
                show_metrics(metrics(np.array([t["ret"] for t in tds]),df.index[0],df.index[-1]))
                tdf=pd.DataFrame(tds); tdf["ret%"]=(tdf["ret"]*100).round(2)
                st.dataframe(tdf[["ed","xd","ret%","reason"]].tail(15),use_container_width=True,hide_index=True)
    st.line_chart(df["Close"].iloc[-120:],height=180,use_container_width=True)


# ═════════════════════════════════════════════════════════════
# TAB 2 — MOMENTUM LAB
# ═════════════════════════════════════════════════════════════

@dataclass
class MCfg:
    id:str; ticker:str; direction:str; rule:str
    start:str; tp:float=4.0; sl:float=2.0; mh:int=10
    notes:str=""; s_cagr:float=0; s_sharpe:float=0

MS = [
    MCfg("APPF","APPF","LONG","APPF","2015-01-01",4.0,2.0,10,"Low[t-10]>Open[t-9]·ATR-low",29.35,1.39),
    MCfg("WDAY","WDAY","SHORT","WDAY","2015-01-01",1.5,1.0,3,"Close[t-3]<Low[t-2]·SMA20·SHORT",16.43,1.35),
    MCfg("HAL","HAL","LONG","HAL","2015-01-01",3.0,2.0,5,"PrevFri-Low>Tue-Close·ATR-low",17.85,1.36),
    MCfg("CHRW","CHRW","LONG","CHRW","2015-01-01",3.0,2.0,5,"Close[t-8]<Open[t-4]·RSI<40",19.38,1.29),
]

def mom_sig(df, rule):
    if rule=="APPF":   return ((df["Low"].shift(10)>df["Open"].shift(9))&df["ATR_LOW"]).fillna(False)
    if rule=="WDAY":   return ((df["Close"].shift(3)<df["Low"].shift(2))&(df["Close"]<df["SMA20"])).fillna(False)
    if rule=="CHRW":   return ((df["Close"].shift(8)<df["Open"].shift(4))&(df["RSI14"]<40)).fillna(False)
    if rule=="HAL":
        dates=df.index; lows=df["Low"]; closes=df["Close"]
        fm={d:float(lows.loc[d]) for d in dates[dates.dayofweek==4]}
        tm={d:float(closes.loc[d]) for d in dates[dates.dayofweek==1]}
        sf=sorted(fm); st2=sorted(tm)
        pfl=pd.Series(np.nan,index=dates); tc=pd.Series(np.nan,index=dates)
        for t in dates:
            lt=next((d for d in reversed(st2) if d<=t),None)
            if lt is None: continue
            tc.loc[t]=tm[lt]; ty,tw,_=lt.isocalendar()
            pf=next((d for d in reversed(sf) if d.isocalendar()[0]<ty or (d.isocalendar()[0]==ty and d.isocalendar()[1]<tw)),None)
            if pf: pfl.loc[t]=fm[pf]
        return ((pfl>tc)&df["ATR_LOW"]).fillna(False)
    return pd.Series(False,index=df.index)

def render_ms(cfg: MCfg):
    with st.spinner(f"Loading {cfg.ticker}…"):
        try: raw=fetch_long(cfg.ticker,cfg.start)
        except Exception as e: st.error(str(e)); return
    df=enrich(raw); df["SIGNAL"]=mom_sig(df,cfg.rule); df=df.dropna(subset=["ATR14"]).copy()
    sig=bool(df["SIGNAL"].iloc[-1]); close=float(df["Close"].iloc[-1]); atr=float(df["ATR14"].iloc[-1])
    sig_banner(cfg.ticker,cfg.notes,sig,cfg.direction)
    c1,c2,c3=st.columns(3)
    c1.metric("Close",f"${close:.2f}"); c2.metric("ATR14",f"{atr:.2f}"); c3.metric("Snapshot CAGR",f"{cfg.s_cagr:.1f}%")
    with st.expander("📊 Backtest (click to run)"):
        if st.button(f"▶ {cfg.ticker}",key=f"mbt_{cfg.id}"):
            tds=run_backtest_atr(df,cfg.tp,cfg.sl,cfg.mh,cfg.direction)
            if tds:
                show_metrics(metrics(np.array([t["ret"] for t in tds]),df.index[0],df.index[-1]))
                tdf=pd.DataFrame(tds); tdf["ret%"]=(tdf["ret"]*100).round(2)
                st.dataframe(tdf[["ed","xd","ret%","reason"]].tail(15),use_container_width=True,hide_index=True)
    st.line_chart(df["Close"].iloc[-120:],height=180,use_container_width=True)


# ═════════════════════════════════════════════════════════════
# TAB 3 — SIGNAL SCANNER (22 tickers)
# ═════════════════════════════════════════════════════════════

@dataclass
class SR:
    ticker:str; strategy:str; signal:bool; direction:str
    action:str; gap_pct:float; cond1:bool; cond2:bool
    sl_pct:float; tp_pct:float; max_hold:int; details:str=""

def _pw(df,idx,wd):
    for i in range(idx-1,-1,-1):
        if df.iloc[i]["Date"].dayofweek==wd: return df.iloc[i]
    return None

def _ds(df):
    c=df["Close"].values; s=0
    for i in range(len(c)-1,0,-1):
        if c[i]<c[i-1]: s+=1
        else: break
    return s

def _us(df):
    c=df["Close"].values; s=0
    for i in range(len(c)-1,0,-1):
        if c[i]>c[i-1]: s+=1
        else: break
    return s

def mks(t,st,sl,tp,mh,dr,c1,c2,df,gap,det=""):
    sig=c1 and c2; d0=df.iloc[-1]
    return SR(t,st,sig,dr,f"ENTER_{dr}_NEXT_OPEN" if sig else "NO_SIGNAL",
              gap*100,c1,c2,sl,tp,mh,det)

def run_signal_scan():
    res=[]; tickers=["FTNT","SWKS","ON","PANW","CRWD","MU","TSM","AMD","MRVL",
                     "DDOG","NET","OKTA","BILL","U","ROKU","TWLO","DOCU","COIN",
                     "PLTR","UPST","PATH","RBLX"]
    prog=st.progress(0,"Scanning…")
    for i,t in enumerate(tickers):
        prog.progress((i+1)/len(tickers),f"Scanning {t}…")
        try:
            df=fetch_short(t); d0=df.iloc[-1]; pc=df.iloc[-2]["Close"]; g=d0["Open"]/pc-1
            pm=_pw(df,len(df)-1,0); pt=_pw(df,len(df)-1,1); pw=_pw(df,len(df)-1,2)
            pth=_pw(df,len(df)-1,3); pf=_pw(df,len(df)-1,4)
            ds=_ds(df); us=_us(df)
            if   t=="FTNT": res.append(mks(t,"R2 Gap-Up",8,15,5,"LONG",pw is not None and pw["Low"]>d0["High"],g>=0.01,df,g))
            elif t=="SWKS": res.append(mks(t,"R3 Gap-Down",10,8,5,"LONG",pw is not None and pw["Open"]<d0["Low"],g<=-0.01,df,g))
            elif t=="ON":   res.append(mks(t,"R2 Cont",8,10,5,"LONG",pt is not None and pt["Low"]>d0["Open"],pw is not None and pw["Open"]<d0["Open"],df,g))
            elif t=="PANW": res.append(mks(t,"R10 Gap-Up",8,12,5,"LONG",pth is not None and pth["Low"]>d0["Open"],g>=0.01,df,g))
            elif t=="CRWD": res.append(mks(t,"R4 Gap-Down",6,12,3,"LONG",pf is not None and pf["High"]<d0["Low"],g<=-0.01,df,g))
            elif t=="MU":   res.append(mks(t,"R1 Cont",4,12,5,"LONG",pt is not None and pt["Low"]>d0["High"],pf is not None and pf["High"]<d0["Close"],df,g))
            elif t=="TSM":  res.append(mks(t,"R3 Rebound",3,10,5,"LONG",pw is not None and pw["Close"]<d0["Low"],ds>=2,df,g,f"streak={ds}"))
            elif t=="AMD":  res.append(mks(t,"R8 Rebound",10,10,5,"LONG",pw is not None and pw["Close"]<d0["Low"],pth is not None and pth["High"]>d0["High"],df,g))
            elif t=="MRVL": res.append(mks(t,"R7 Rebound",5,12,3,"LONG",pw is not None and pw["Close"]<d0["Low"],ds>=2,df,g,f"streak={ds}"))
            elif t=="DDOG": res.append(mks(t,"R9 Momentum",8,10,5,"LONG",pm is not None and pm["Open"]<d0["Close"],pth is not None and pth["Low"]>d0["High"],df,g))
            elif t=="NET":  res.append(mks(t,"R9 Momentum",8,12,3,"LONG",pw is not None and pw["Low"]<d0["Open"],pf is not None and pf["Low"]>d0["High"],df,g))
            elif t=="OKTA": res.append(mks(t,"R9 Rebound",10,3,3,"LONG",pm is not None and pm["Low"]<d0["Low"],pw is not None and pw["Low"]>d0["High"],df,g))
            elif t=="BILL": res.append(mks(t,"R9 Short",8,15,3,"SHORT",pm is not None and pm["High"]<d0["Low"],pw is not None and pw["Close"]>d0["Close"],df,g))
            elif t=="U":
                cr=d0["High"]-d0["Low"]; cp=float((d0["Close"]-d0["Low"])/cr) if cr>0 else 0.5
                res.append(mks(t,"R9 Short",8,12,2,"SHORT",pt is not None and pt["High"]<d0["Close"],cp<=0.25,df,g,f"close_pos={cp:.2f}"))
            elif t=="ROKU": res.append(mks(t,"R1 Short",4,15,5,"SHORT",pt is not None and pt["Low"]>d0["Open"],pth is not None and pth["High"]<d0["Open"],df,g))
            elif t=="TWLO": res.append(mks(t,"R5 Cont",4,10,3,"LONG",pth is not None and pth["High"]<d0["Open"],pf is not None and pf["Low"]>d0["Close"],df,g))
            elif t=="DOCU": res.append(mks(t,"R5 Short",8,15,5,"SHORT",pm is not None and pm["Close"]>d0["High"],us>=2,df,g,f"streak={us}"))
            elif t=="COIN": res.append(mks(t,"R1 Short",2.5,10,2,"SHORT",pt is not None and pt["Close"]<d0["Open"],ds>=3,df,g,f"streak={ds}"))
            elif t=="PLTR": res.append(mks(t,"R2 Cont",5,12,2,"LONG",pw is not None and pw["High"]<d0["Open"],pth is not None and pth["Low"]>d0["Close"],df,g))
            elif t=="UPST": res.append(mks(t,"R10 Rebound",2,6,3,"LONG",pm is not None and pm["Low"]>d0["High"],pw is not None and pw["Open"]<d0["Open"],df,g))
            elif t=="PATH": res.append(mks(t,"R5 Short",8,10,2,"SHORT",pm is not None and pm["Close"]>d0["Low"],pt is not None and pt["High"]<d0["Low"],df,g))
            elif t=="RBLX": res.append(mks(t,"R1 Rebound",5,12,2,"LONG",pt is not None and pt["High"]<d0["Open"],ds>=2,df,g,f"streak={ds}"))
        except Exception as e:
            res.append(SR(t,"error",False,"—","ERROR",0,False,False,0,0,0,str(e)))
    prog.empty()
    return res


# ═════════════════════════════════════════════════════════════
# TAB 4 — SKEWED-W
# ═════════════════════════════════════════════════════════════

SW_TICKERS=["NVDA","MSFT","AAPL","META","AMZN","GOOGL","TSLA","JPM","V","MA",
            "UNH","COST","XOM","HD","AVGO","LLY","MRK","PG","KO","PEP",
            "BAC","WMT","CRM","ADBE","NFLX","AMD","QCOM","TXN","AMAT","CAT",
            "GS","MS","BLK","ABBV","NOW","ISRG","CVX","NKE","DIS","SBUX"]
SW_SL=0.05; SW_TP=0.10; SW_STREAK=3; SW_VOL=1.2

def sw_detect(df):
    if len(df)<30: return None
    c=df["Close"].values; v=df.get("Volume",pd.Series(np.ones(len(df)))).values if "Volume" in df.columns else np.ones(len(df))
    n=len(c); streak=0
    for i in range(n-1,0,-1):
        if c[i]<c[i-1]: streak+=1
        else: break
    if streak<SW_STREAK: return None
    vma=float(np.mean(v[max(0,n-21):n-1])) if n>21 else 1
    vr=float(v[n-1])/vma if vma>0 else 1
    if vr<SW_VOL: return None
    cl=float(c[-1])
    return dict(streak=streak,vol_ratio=round(vr,2),close=round(cl,2),
                sl=round(cl*(1-SW_SL),2),tp=round(cl*(1+SW_TP),2))

@st.cache_data(ttl=3600,show_spinner=False)
def sw_scan():
    results=[]
    for t in SW_TICKERS:
        try:
            df=fetch_long(t,"2023-01-01"); sig=sw_detect(df)
            if sig: results.append({"ticker":t,**sig})
        except: pass
    return results


# ═════════════════════════════════════════════════════════════
# TAB 5 — DOUBLE BOTTOM
# ═════════════════════════════════════════════════════════════

DB_TICKERS=tuple([
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
])
DB_TOL=0.06; DB_MIN=7; DB_MAX=120; DB_SL=0.03; DB_TP=0.08

def db_pivots(s,left=2,right=2):
    x=s.values.astype(float); n=len(x); piv=np.zeros(n,dtype=bool)
    for i in range(left,n-right):
        w=x[i-left:i+right+1]
        if np.isfinite(x[i]) and x[i]==np.nanmin(w): piv[i]=True
    return piv

def db_pattern(df):
    if len(df)<60: return None
    lo=df["Low"]; hi=df["High"]
    piv=db_pivots(lo); idx=np.where(piv)[0]
    if len(idx)<2: return None
    best=None
    for a in range(len(idx)-1):
        i=idx[a]
        for b in range(a+1,len(idx)):
            j=idx[b]; sep=j-i
            if sep<DB_MIN: continue
            if sep>DB_MAX: break
            l1=float(lo.iloc[i]); l2=float(lo.iloc[j])
            if l1<=0 or l2<=0: continue
            avg=(l1+l2)/2
            if abs(l1-l2)/avg>DB_TOL: continue
            neck=float(hi.iloc[i:j+1].max())
            score=j-(abs(l1-l2)/avg)*1000
            if best is None or score>best[0]: best=(score,neck)
    return {"neckline":best[1]} if best else None

def db_check(df):
    if len(df)<61: return None
    w=df.iloc[-180:] if len(df)>180 else df
    p=db_pattern(w)
    if not p: return None
    neck=p["neckline"]
    if df["Close"].iloc[-2]<=neck and df["Close"].iloc[-1]>neck:
        cl=float(df["Close"].iloc[-1])
        return dict(neckline=round(neck,2),close=round(cl,2),
                    sl=round(cl*(1-DB_SL),2),tp=round(cl*(1+DB_TP),2),
                    strength_pct=round((cl-neck)/neck*100,2))
    return None

def db_check_signal(df, end_pos):
    """Exact logic from compute_breakout_signal in v4_4"""
    if end_pos <= 0: return None
    start = max(0, end_pos - 180 + 1)
    window = df.iloc[start:end_pos + 1]
    patt = db_pattern(window)
    if not patt: return None
    neck = float(patt["neckline"])
    ct = float(df["Close"].iloc[end_pos])
    cy = float(df["Close"].iloc[end_pos - 1])
    if not (cy <= neck and ct > neck): return None
    return {"neckline": neck, "strength": (ct - neck) / neck}


@st.cache_data(ttl=3600, show_spinner=False)
def db_scan():
    DB_MAX_POS = 10
    raw = yf.download(list(DB_TICKERS), period="5y", interval="1d",
                      group_by="ticker", progress=False, auto_adjust=False)
    data_by_ticker = {}; date_sets = []
    for t in DB_TICKERS:
        try:
            if hasattr(raw.columns, "levels") and len(raw.columns.levels) > 1:
                if t not in raw.columns.levels[0]: continue
                df = raw[t].copy()
            else:
                df = raw.copy()
            df = df.dropna()
            if isinstance(df.columns, pd.MultiIndex): df.columns = [c[0] for c in df.columns]
            needed = {"Open", "High", "Low", "Close"}
            if not needed.issubset(set(df.columns)): continue
            df.index = pd.to_datetime(df.index); df = df.sort_index()
            if len(df) < 200: continue
            data_by_ticker[t] = df[["Open", "High", "Low", "Close"]]
            date_sets.append(set(df.index))
        except: continue
    if not data_by_ticker or not date_sets: return []
    common_dates = sorted(list(set.intersection(*date_sets)))
    dates = pd.DatetimeIndex(common_dates)
    # Portfolio simulation to track open positions (matches original)
    open_positions = {}; pending = {}
    for di in range(1, len(dates)):
        date = dates[di]
        if date in pending:
            cands = pending.pop(date); cands.sort(key=lambda x: x[1], reverse=True)
            for tk, strength, neck in cands:
                if tk in open_positions or len(open_positions) >= DB_MAX_POS: break
                df = data_by_ticker.get(tk)
                if df is None or date not in df.index: continue
                o = float(df.loc[date, "Open"])
                if not np.isfinite(o) or o <= 0: continue
                open_positions[tk] = {"days_held": 0,
                    "sl_price": o * (1 - DB_SL), "tp_price": o * (1 + DB_TP)}
        to_close = []
        for tk, pos in open_positions.items():
            df = data_by_ticker.get(tk)
            if df is None or date not in df.index: continue
            bar = df.loc[date]; pos["days_held"] += 1
            lo = float(bar["Low"]); hi = float(bar["High"]); cl = float(bar["Close"])
            if (np.isfinite(lo) and lo <= pos["sl_price"]) or                (np.isfinite(hi) and hi >= pos["tp_price"]) or                pos["days_held"] >= 20:
                to_close.append(tk)
        for tk in to_close: open_positions.pop(tk, None)
        if di < len(dates) - 1:
            next_date = dates[di + 1]; cands = []
            for tk, df in data_by_ticker.items():
                if tk in open_positions: continue
                if date not in df.index or dates[di - 1] not in df.index: continue
                ep = df.index.get_loc(date)
                sig = db_check_signal(df, ep)
                if sig: cands.append((tk, sig["strength"], sig["neckline"]))
            if cands: pending.setdefault(next_date, []).extend(cands)
    # Fresh signals on latest close, respecting MaxPos slots
    latest = dates[-1]; prev = dates[-2] if len(dates) >= 2 else dates[-1]
    slots = DB_MAX_POS - len(open_positions); sigs = []
    if slots > 0:
        for tk, df in data_by_ticker.items():
            if tk in open_positions: continue
            if latest not in df.index or prev not in df.index: continue
            ep = df.index.get_loc(latest)
            sig = db_check_signal(df, ep)
            if sig:
                cl = float(df.loc[latest, "Close"])
                sigs.append({"ticker": tk, "signal_date": str(latest.date()),
                    "close": round(cl, 2), "neckline": round(sig["neckline"], 2),
                    "stop_loss": round(cl * (1 - DB_SL), 2),
                    "take_profit": round(cl * (1 + DB_TP), 2),
                    "strength_pct": round(sig["strength"] * 100, 2)})
        sigs.sort(key=lambda x: x["strength_pct"], reverse=True)
        sigs = sigs[:slots]
    return sigs


# ═════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═════════════════════════════════════════════════════════════

st.markdown("## 📊 QuantGaps Research Hub")
st.caption("⚠️ RESEARCH / FORWARD-TEST ONLY · Not financial advice · EOD data from Yahoo Finance")

t1,t2,t3,t4,t5 = st.tabs([
    "📡 Pattern Scanner",
    "🧪 Momentum Lab",
    "🔬 Signal Scanner",
    "〽️ Skewed-W",
    "📐 Double Bottom",
])

with t1:
    st.markdown("### Pattern Scanner — IWM · CRM · PYPL · UPS · COF · NTR · ALB")
    st.caption("DUD/DDD/DD/DU patterns · ATR exits · each strategy loads independently")
    subs=st.tabs([f"{c.ticker}·{c.pattern}" for c in PS])
    for sub,cfg in zip(subs,PS):
        with sub: render_ps(cfg)

with t2:
    st.markdown("### Momentum Sequence Lab — APPF · WDAY · HAL · CHRW")
    st.caption("Price-sequence rules · ATR exits")
    subs2=st.tabs([f"{c.ticker}·{c.direction}" for c in MS])
    for sub,cfg in zip(subs2,MS):
        with sub: render_ms(cfg)

with t3:
    st.markdown("### Signal Scanner — 22 Tickers")
    st.caption("Gap/streak/weekday-anchor patterns · FTNT SWKS ON PANW CRWD MU TSM AMD + 14 more")
    if st.button("▶ Run Signal Scan", type="primary", key="sig_run"):
        with st.spinner("Scanning 22 tickers…"):
            st.session_state["sig_res"] = run_signal_scan()
            st.session_state["sig_ts"]  = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    if "sig_res" in st.session_state:
        res=st.session_state["sig_res"]; active=[r for r in res if r.signal]
        st.caption(f"Last scan: {st.session_state.get('sig_ts','—')} · Signals: {len(active)}/{len(res)}")
        if active: st.success(f"🚨 {len(active)} signal(s): {', '.join(r.ticker for r in active)}")
        rows=[{"Ticker":r.ticker,"Strategy":r.strategy,"Dir":r.direction,
               "Signal":"✅" if r.signal else "—","Gap%":f"{r.gap_pct:+.2f}%",
               "C1":"✅" if r.cond1 else "❌","C2":"✅" if r.cond2 else "❌",
               "SL":f"-{r.sl_pct:.0f}%","TP":f"+{r.tp_pct:.0f}%","Hold":r.max_hold}
              for r in res]
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    else:
        st.info("Click ▶ Run Signal Scan to start (~20 seconds).")

with t4:
    st.markdown("### Skewed-W Scanner")
    st.caption(f"Down-streak≥{SW_STREAK} + vol surge≥{SW_VOL}x · SL {SW_SL*100:.0f}% · TP {SW_TP*100:.0f}% · {len(SW_TICKERS)} tickers")
    if st.button("▶ Run Skewed-W Scan", type="primary", key="sw_run"):
        st.cache_data.clear()
        with st.spinner("Scanning…"):
            sw_res=sw_scan()
        st.session_state["sw_res"]=sw_res
    if "sw_res" in st.session_state:
        sw_res=st.session_state["sw_res"]
        if sw_res:
            st.success(f"🟢 {len(sw_res)} signal(s)")
            st.dataframe(pd.DataFrame(sw_res),use_container_width=True,hide_index=True)
        else:
            st.info("No Skewed-W signals today.")
    else:
        st.info("Click ▶ Run Skewed-W Scan to start.")

with t5:
    st.markdown("### Double Bottom Scanner")
    st.caption(f"Pivot-low neckline breakout · SL {DB_SL*100:.0f}% · TP {DB_TP*100:.0f}% · {len(DB_TICKERS)} tickers")
    if st.button("▶ Run Double Bottom Scan", type="primary", key="db_run"):
        st.cache_data.clear()
        with st.spinner("Downloading and scanning 105 tickers (~30s)…"):
            db_res=db_scan()
        st.session_state["db_res"]=db_res
    if "db_res" in st.session_state:
        db_res=st.session_state["db_res"]
        if db_res:
            st.success(f"🟢 {len(db_res)} breakout signal(s)")
            st.dataframe(pd.DataFrame(db_res),use_container_width=True,hide_index=True)
        else:
            st.info("No double-bottom breakout signals on latest close.")
    else:
        st.info("Click ▶ Run Double Bottom Scan to start.")

st.markdown("---")
st.caption("QuantGaps Research · EOD data from Yahoo Finance · Not financial advice")
