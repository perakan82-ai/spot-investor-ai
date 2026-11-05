"""
🏦 Spot Investor AI — วิเคราะห์ลงทุน Spot ระยะกลาง & ระยะยาว
================================================================
ทีมวิเคราะห์ (Agents)
  1) Market Data Agent  : Binance (4H / 1D / 1W) → EMA21/50/200, RSI, MACD, ADX, ATR, OBV
  2) Structure Agent    : Market Structure (HH/HL), Volume Profile (POC / Value Area), แนวรับ-ต้าน, Fibonacci
  3) Quant Agent        : Backtest 1D + Conditional Forward Returns (ผลตอบแทนในอดีตเมื่อสภาพตลาดคล้ายวันนี้)
                          → Win Rate และระยะเวลาถือที่เหมาะ (วัน / สัปดาห์ / เดือน)
  4) Fundamental Agent  : CoinGecko (Market Cap, FDV, Supply, ATH) + DefiLlama (TVL, ค่าธรรมเนียม/รายได้)
  5) Macro Agent        : เทรนด์ BTC, BTC Dominance, Fear & Greed
  6) Research Agent (AI): Gemini + Google Search ค้นโปรเจกต์ Tokenomics ข่าวล่าสุด และกำหนดการ
                          (Unlock, Upgrade, Mainnet, ETF, Listing ฯลฯ)
  7) Strategist (AI)    : คัดเหรียญ ระยะกลาง (Swing) / ระยะยาว (สะสม) จัดสัดส่วนพอร์ต
  8) Risk Manager (AI)  : ตรวจความเสี่ยง ความสัมพันธ์ระหว่างเหรียญ และจำกัดสัดส่วน

ตัวเลขราคา (โซนซื้อ / SL / TP / DCA) คำนวณโดยระบบ ไม่ให้ AI แต่งเอง
รัน:  streamlit run spot_invest.py
⚠️ ไม่ใช่คำแนะนำการลงทุน
"""

import concurrent.futures as cf
import json
import re
import time
from datetime import datetime, timedelta, timezone

import ccxt
import feedparser
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

st.set_page_config(page_title="Spot Investor AI", page_icon="🏦", layout="wide")

# =============================================================================
# CONFIG
# =============================================================================
TIMEFRAMES = ["4h", "1d", "1w"]
TF_LABEL = {"4h": "4H", "1d": "1D", "1w": "1W"}
KLINE_LIMIT = {"4h": 500, "1d": 1000, "1w": 1000}
MED_FWD = [7, 14, 30, 60]          # ระยะกลาง: ดูผลตอบแทนล่วงหน้ากี่วัน
LONG_FWD = [90, 180, 365]          # ระยะยาว
MED_HOLD_LIMIT = (5, 90)           # วัน
LONG_HOLD_LIMIT = (60, 730)

BINANCE_HOSTS = ["https://data-api.binance.vision/api/v3", "https://api.binance.com/api/v3"]
EXCLUDE_BASES = {
    "USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "USDE", "USD1", "EUR", "EURI", "AEUR", "PYUSD",
    "RLUSD", "BFUSD", "XUSD", "USDS", "PAXG", "WBTC", "WBETH", "BNSOL",
}

DEFAULT_WATCHLIST = (
    "BTC ETH SOL XRP XLM ADA NEAR ENA HYPE AAVE ZEC BNB PUMP UNI HBAR\n"
    "DOGE LINK SUI AVAX TAO ONDO TRX LTC BCH WLD JUP PENDLE ARB PEPE RENDER"
)
MAX_WATCHLIST = 40

CG_IDS = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "XRP": "ripple", "XLM": "stellar", "ADA": "cardano",
    "NEAR": "near", "ENA": "ethena", "HYPE": "hyperliquid", "AAVE": "aave", "ZEC": "zcash", "BNB": "binancecoin",
    "PUMP": "pump-fun", "UNI": "uniswap", "HBAR": "hedera-hashgraph", "DOGE": "dogecoin", "LINK": "chainlink",
    "SUI": "sui", "AVAX": "avalanche-2", "TAO": "bittensor", "ONDO": "ondo-finance", "TRX": "tron",
    "LTC": "litecoin", "BCH": "bitcoin-cash", "WLD": "worldcoin-wld", "JUP": "jupiter-exchange-solana",
    "PENDLE": "pendle", "ARB": "arbitrum", "PEPE": "pepe", "RENDER": "render-token", "DOT": "polkadot",
    "TON": "the-open-network", "APT": "aptos", "OP": "optimism", "INJ": "injective-protocol",
    "SEI": "sei-network", "TIA": "celestia", "FET": "fetch-ai", "SHIB": "shiba-inu", "ETC": "ethereum-classic",
    "FIL": "filecoin", "ICP": "internet-computer", "ATOM": "cosmos", "ALGO": "algorand", "KAS": "kaspa",
}

RSS_FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
}

GEMINI_MODELS = ["gemini-3.8-flash", "gemini-3.1-flash-lite", "อื่นๆ (พิมพ์เอง)"]

STATUS_MED = {"BUY": "🟢 ซื้อได้", "WAIT": "🟡 รอย่อ/รอยืนยัน", "AVOID": "🔴 ยังไม่ควรซื้อ"}
STATUS_LONG = {"BUY": "🟢 สะสมได้", "WAIT": "🟡 ทยอยสะสมโซนล่าง", "AVOID": "🔴 รอสัญญาณกลับตัว"}
HZ_NAME = {"MEDIUM": "🌊 ระยะกลาง (Swing)", "LONG": "🏦 ระยะยาว (สะสม)"}


# =============================================================================
# HELPERS
# =============================================================================
def fmt_price(x) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(x):
        return "-"
    if x >= 1000:
        return f"{x:,.2f}"
    if x >= 1:
        return f"{x:,.4f}"
    if x >= 0.01:
        return f"{x:.5f}"
    return f"{x:.8f}"


def fmt_pct(x, plus=True) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(x):
        return "-"
    return f"{x:+.1f}%" if plus else f"{x:.1f}%"


def fmt_big(x) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(x) or x <= 0:
        return "-"
    for unit, d in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if x >= d:
            return f"${x / d:,.2f}{unit}"
    return f"${x:,.0f}"


def fmt_days(d) -> str:
    if d is None or not np.isfinite(d):
        return "-"
    if d < 14:
        return f"{d:.0f} วัน"
    if d < 60:
        return f"{d / 7:.0f} สัปดาห์"
    if d < 730:
        return f"{d / 30:.0f} เดือน"
    return f"{d / 365:.1f} ปี"


def range_days(a, b) -> str:
    """แปลงช่วงวันเป็นข้อความที่อ่านง่าย เช่น '3–6 สัปดาห์ (~21–42 วัน)'"""
    a, b = float(min(a, b)), float(max(a, b))
    if b < 14:
        return f"{a:.0f}–{b:.0f} วัน"
    if b < 70:
        return f"{max(1, round(a / 7))}–{max(1, round(b / 7))} สัปดาห์ (~{a:.0f}–{b:.0f} วัน)"
    lo, hi = max(1, round(a / 30)), max(1, round(b / 30))
    if lo == hi:
        hi += 1
    return f"{lo}–{hi} เดือน (~{a:.0f}–{b:.0f} วัน)"


def get_secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, ""))
    except Exception:  # noqa: BLE001
        return ""


def safe_float(v, default=np.nan):
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def parse_watchlist(text: str) -> list:
    out = []
    for tok in re.split(r"[\s,;]+", (text or "").upper()):
        tok = re.sub(r"[/\-_]", "", tok)
        if tok.endswith("USDT") and len(tok) > 4:
            tok = tok[:-4]
        if tok and tok not in out:
            out.append(tok)
    return out


def clip(x, lo, hi):
    return float(min(hi, max(lo, x)))


# =============================================================================
# 1) MARKET DATA AGENT
# =============================================================================
def connect_binance():
    last_err = None
    for host in BINANCE_HOSTS:
        try:
            ex = ccxt.binance({"enableRateLimit": True, "timeout": 20000})
            ex.urls["api"]["public"] = host
            ex.public_get_ping()
            return ex, host
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"เชื่อมต่อ Binance ไม่ได้: {last_err}")


def get_symbols(ex, watchlist: list, n_discovery: int) -> pd.DataFrame:
    tickers = ex.public_get_ticker_24hr()
    now_ms = time.time() * 1000
    rows = []
    for t in tickers:
        sym = t.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        price, qvol = safe_float(t.get("lastPrice"), 0), safe_float(t.get("quoteVolume"), 0)
        if price <= 0 or qvol <= 0 or now_ms - safe_float(t.get("closeTime"), 0) > 86_400_000:
            continue
        rows.append({"symbol": sym, "base": sym[:-4], "price": price,
                     "chg_24h": safe_float(t.get("priceChangePercent"), 0), "vol_24h": qvol})
    if not rows:
        raise RuntimeError("ไม่พบคู่เหรียญ USDT จาก Binance")
    all_df = pd.DataFrame(rows)
    wl = all_df[all_df["base"].isin(watchlist)].copy()
    missing = [c for c in watchlist if c not in set(wl["base"])]
    if missing:
        st.warning(f"ไม่พบคู่ Spot USDT บน Binance: {', '.join(missing)} (ข้ามไป)")
    order = {c: i for i, c in enumerate(watchlist)}
    wl = wl.sort_values("base", key=lambda s: s.map(order))
    wl["group"] = "Watchlist"
    parts = [wl]
    if n_discovery > 0:
        disc = (all_df[~all_df["base"].isin(set(watchlist) | EXCLUDE_BASES)]
                .sort_values("vol_24h", ascending=False).head(n_discovery).copy())
        disc["group"] = "Discovery 🔍"
        parts.append(disc)
    df = pd.concat(parts)
    if "BTC" not in set(df["base"]):
        btc = all_df[all_df["base"] == "BTC"].copy()
        btc["group"] = "Benchmark"
        df = pd.concat([df, btc])
    return df.reset_index(drop=True)


def fetch_klines(ex, symbol: str, tf: str) -> pd.DataFrame:
    raw = ex.public_get_klines({"symbol": symbol, "interval": tf, "limit": KLINE_LIMIT[tf]})
    cols = ["t", "open", "high", "low", "close", "volume", "ct", "qv", "n", "tbb", "tbq", "ig"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = pd.to_datetime(pd.to_numeric(df["t"]), unit="ms", utc=True)
    return df[["time", "open", "high", "low", "close", "volume"]].dropna().reset_index(drop=True)


# ---------- indicators ----------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def calc_rsi(close, n=14):
    d = close.diff()
    g = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    l_ = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return (100 - 100 / (1 + g / l_.replace(0, np.nan))).fillna(100.0).where(g.notna())


def calc_atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def calc_adx(df, n=14):
    up, dn = df["high"].diff(), -df["low"].diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    atr = calc_atr(df, n)
    pdi = 100 * pdm.ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * mdm.ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def add_ind(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    for n in (21, 50, 200):
        df[f"ema{n}"] = ema(c, n) if len(df) >= n else np.nan
    df["rsi"] = calc_rsi(c)
    macd = ema(c, 12) - ema(c, 26)
    df["hist"] = macd - ema(macd, 9)
    df["atr"] = calc_atr(df)
    df["adx"] = calc_adx(df)
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    df["obv"] = (np.sign(c.diff().fillna(0)) * df["volume"]).cumsum()
    return df


def score_row(r) -> float:
    """คะแนนเทคนิค -3 (ขาลงแรง) ถึง +3 (ขาขึ้นแรง)"""
    c, e21, e50, e200 = r["close"], r["ema21"], r["ema50"], r["ema200"]
    s = 0.0
    if pd.notna(e200) and pd.notna(e50):
        if c > e50 > e200:
            s += 1.5
        elif c < e50 < e200:
            s -= 1.5
        s += 0.25 if c > e200 else -0.25
    if pd.notna(e50):
        s += 0.5 if c > e50 else -0.5
    if pd.notna(e21):
        s += 0.25 if c > e21 else -0.25
    if pd.notna(r["hist"]):
        s += 0.5 if r["hist"] > 0 else -0.5
    rsi = r["rsi"]
    if pd.notna(rsi):
        if rsi >= 75:
            s -= 0.5
        elif rsi <= 25:
            s += 0.5
        elif 50 <= rsi <= 68:
            s += 0.25
        elif rsi < 45:
            s -= 0.25
    if pd.notna(r["adx"]):
        s *= 1.15 if r["adx"] >= 25 else (0.8 if r["adx"] < 15 else 1.0)
    return clip(s, -3, 3)


# =============================================================================
# 2) STRUCTURE AGENT
# =============================================================================
def pivots(df: pd.DataFrame, k: int):
    h, l_ = df["high"].values, df["low"].values
    ph, pl = [], []
    for i in range(k, len(df) - k):
        if h[i] == h[i - k:i + k + 1].max():
            ph.append((i, float(h[i])))
        if l_[i] == l_[i - k:i + k + 1].min():
            pl.append((i, float(l_[i])))
    return ph, pl


def market_structure(df: pd.DataFrame, k: int) -> str:
    ph, pl = pivots(df, k)
    if len(ph) < 2 or len(pl) < 2:
        return "ไม่ชัด"
    hh, hl = ph[-1][1] > ph[-2][1], pl[-1][1] > pl[-2][1]
    if hh and hl:
        return "ขาขึ้น (HH/HL)"
    if not hh and not hl:
        return "ขาลง (LH/LL)"
    return "Range/เปลี่ยนเทรนด์"


def volume_profile(df: pd.DataFrame, lookback=180, bins=40):
    d = df.tail(lookback)
    lo, hi = float(d["low"].min()), float(d["high"].max())
    if not np.isfinite(lo) or hi <= lo:
        return None
    edges = np.linspace(lo, hi, bins + 1)
    vol = np.zeros(bins)
    for l_, h_, v in zip(d["low"].values, d["high"].values, d["volume"].values):
        i0 = int(np.clip(np.searchsorted(edges, l_, side="right") - 1, 0, bins - 1))
        i1 = int(np.clip(np.searchsorted(edges, h_, side="right") - 1, 0, bins - 1))
        vol[i0:i1 + 1] += v / (i1 - i0 + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    order, cum, sel = np.argsort(vol)[::-1], 0.0, []
    for i in order:
        sel.append(i)
        cum += vol[i]
        if cum >= 0.7 * vol.sum():
            break
    hvn = [float(centers[i]) for i in range(1, bins - 1)
           if vol[i] >= vol[i - 1] and vol[i] >= vol[i + 1] and vol[i] > vol.mean() * 1.2]
    return {"poc": float(centers[vol.argmax()]), "val": float(centers[min(sel)]),
            "vah": float(centers[max(sel)]), "hvn": hvn}


def key_levels(d1: pd.DataFrame, w1: pd.DataFrame | None, vp: dict | None) -> list:
    cands = []
    ph, pl = pivots(d1.tail(250).reset_index(drop=True), 3)
    cands += [(p, "Swing High 1D", 1) for _, p in ph[-12:]] + [(p, "Swing Low 1D", 1) for _, p in pl[-12:]]
    if w1 is not None and len(w1) > 10:
        wph, wpl = pivots(w1.tail(160).reset_index(drop=True), 2)
        cands += [(p, "Swing High 1W", 2) for _, p in wph[-8:]] + [(p, "Swing Low 1W", 2) for _, p in wpl[-8:]]
    if vp:
        cands.append((vp["poc"], "POC", 2))
        cands += [(x, "HVN", 1) for x in vp["hvn"]]
    cands.sort(key=lambda x: x[0])
    clusters = []
    for p, kind, w in cands:
        if clusters and abs(p - clusters[-1][0][0]) / clusters[-1][0][0] < 0.012:
            clusters[-1].append((p, kind, w))
        else:
            clusters.append([(p, kind, w)])
    out = []
    for cl in clusters:
        ws = sum(m[2] for m in cl)
        out.append({"price": sum(m[0] * m[2] for m in cl) / ws, "strength": ws,
                    "kinds": ", ".join(sorted({m[1] for m in cl}))})
    return out


# =============================================================================
# 3) QUANT AGENT
# =============================================================================
def backtest_1d(df: pd.DataFrame) -> dict:
    """Setup ขาขึ้น 1D: close>EMA50, MACD hist>0, RSI 40-70 (+EMA50>EMA200 ถ้ามี)
    เข้า close, SL 2 ATR, TP1 2R, ถือสูงสุด 60 วัน → % ถึง TP1 ก่อน SL และระยะเวลาที่ใช้"""
    c, h, l_, atr = df["close"].values, df["high"].values, df["low"].values, df["atr"].values
    cond = (df["close"] > df["ema50"]) & (df["hist"] > 0) & df["rsi"].between(40, 70)
    if df["ema200"].notna().any():
        cond = cond & (df["ema50"] > df["ema200"])
    sig = np.array(cond.fillna(False), dtype=bool)  # สำเนาใหม่ (pandas 3 คืน array แบบอ่านอย่างเดียว)
    n, i = len(df), 60
    wins = losses = timeouts = 0
    r_sum, win_days = 0.0, []
    while i < n - 2:
        if not sig[i] or not np.isfinite(atr[i]) or atr[i] <= 0:
            i += 1
            continue
        e, risk = c[i], 2 * atr[i]
        sl, tp = e - risk, e + 2 * risk
        outcome, j = None, i
        for j in range(i + 1, min(n, i + 61)):
            if l_[j] <= sl:
                outcome = "loss"
                break
            if h[j] >= tp:
                outcome = "win"
                break
        if outcome is None:
            if j >= n - 1 and j - i < 60:
                break
            outcome = "timeout"
        if outcome == "win":
            wins += 1
            r_sum += 2
            win_days.append(j - i)
        elif outcome == "loss":
            losses += 1
            r_sum -= 1
        else:
            timeouts += 1
            r_sum += (c[j] - e) / risk
        i = j + 1
    total = wins + losses + timeouts
    out = {"n": total, "wins": wins, "wr_raw": wins / total * 100 if total else np.nan,
           "wr_smooth": (wins + 2) / (total + 4) * 100, "exp_r": r_sum / total if total else np.nan,
           "hold_p25": np.nan, "hold_p50": np.nan, "hold_p75": np.nan}
    if win_days:
        out["hold_p25"], out["hold_p50"], out["hold_p75"] = (float(x) for x in np.percentile(win_days, [25, 50, 75]))
    return out


def forward_returns(df: pd.DataFrame, horizons: list) -> dict:
    """Conditional forward returns: เมื่อในอดีตเหรียญนี้อยู่ในสภาพเดียวกับวันนี้ (Regime + RSI)
    ถือต่อ h วันแล้วได้ผลอย่างไร"""
    c = df["close"].values
    n = len(df)
    has200 = df["ema200"].notna().any()
    if has200:
        bull = ((df["close"] > df["ema200"]) & (df["ema50"] > df["ema200"])).values
        bear = ((df["close"] < df["ema200"]) & (df["ema50"] < df["ema200"])).values
        start = 200
    else:
        bull = (df["close"] > df["ema50"]).fillna(False).values
        bear = (df["close"] < df["ema50"]).fillna(False).values
        start = 50
    regime = np.where(bull, 2, np.where(bear, 0, 1))
    rsi_b = np.digitize(df["rsi"].fillna(50).values, [35, 50, 65])
    state = regime * 10 + rsi_b
    cur_state, cur_reg = state[-1], regime[-1]
    reg_name = {2: "ขาขึ้น", 1: "เปลี่ยนผ่าน", 0: "ขาลง"}[int(cur_reg)]
    rsi_name = ["RSI<35", "RSI 35-50", "RSI 50-65", "RSI>65"][int(rsi_b[-1])]
    res = []
    for h in horizons:
        if n - h <= start + 10:
            res.append({"h": h, "n": 0, "win": np.nan, "median": np.nan, "p25": np.nan, "p75": np.nan, "basis": "-"})
            continue
        idx = np.arange(start, n - h)
        sel, basis = idx[state[idx] == cur_state], f"{reg_name} + {rsi_name}"
        if len(sel) < 15:
            sel, basis = idx[regime[idx] == cur_reg], f"{reg_name}"
        if len(sel) < 8:
            sel, basis = idx, "ทุกช่วงเวลา"
        r = c[sel + h] / c[sel] - 1
        res.append({"h": h, "n": int(len(sel)), "win": float((r > 0).mean() * 100),
                    "median": float(np.median(r) * 100), "p25": float(np.percentile(r, 25) * 100),
                    "p75": float(np.percentile(r, 75) * 100), "basis": basis})
    valid = [x for x in res if x["n"] >= 10 and np.isfinite(x["median"])]
    best = None
    if valid:  # risk-adjusted: median / ความเสียหายช่วงแย่ (p25)
        best = max(valid, key=lambda x: (x["median"] + 0.1) / (abs(min(x["p25"], 0)) + 5) + x["win"] / 500)
    return {"rows": res, "best": best, "state": f"{reg_name} · {rsi_name}"}


def risk_metrics(d1: pd.DataFrame, btc_d1: pd.DataFrame | None) -> dict:
    ret = np.log(d1["close"]).diff()
    out = {"vol_ann": float(ret.tail(90).std() * np.sqrt(365) * 100)}
    last365 = d1["close"].tail(365)
    out["max_dd_1y"] = float(((last365 / last365.cummax()) - 1).min() * 100)
    out["beta"], out["corr_btc"], out["rs30"], out["rs90"] = np.nan, np.nan, np.nan, np.nan
    if btc_d1 is not None and len(btc_d1) > 100:
        a = pd.Series(ret.values, index=d1["time"]).tail(90)
        b = pd.Series(np.log(btc_d1["close"]).diff().values, index=btc_d1["time"])
        j = pd.concat([a, b], axis=1, join="inner").dropna()
        if len(j) > 30 and j.iloc[:, 1].var() > 0:
            out["beta"] = float(j.cov().iloc[0, 1] / j.iloc[:, 1].var())
            out["corr_btc"] = float(j.corr().iloc[0, 1])

        def r(df, n):
            return (df["close"].iloc[-1] / df["close"].iloc[-n - 1] - 1) * 100 if len(df) > n else np.nan

        out["rs30"] = r(d1, 30) - r(btc_d1, 30)
        out["rs90"] = r(d1, 90) - r(btc_d1, 90)
    return out


def tf_snapshot(df: pd.DataFrame) -> dict:
    last, prev = df.iloc[-1], df.iloc[-2]
    if pd.notna(last["ema200"]) and last["close"] > last["ema50"] > last["ema200"]:
        trend = "UP"
    elif pd.notna(last["ema200"]) and last["close"] < last["ema50"] < last["ema200"]:
        trend = "DOWN"
    elif pd.isna(last["ema200"]) and pd.notna(last["ema50"]):
        trend = "UP*" if last["close"] > last["ema50"] else "DOWN*"
    else:
        trend = "SIDEWAY"
    return {"score": round(score_row(last), 2), "trend": trend, "rsi": round(float(last["rsi"]), 1),
            "adx": round(safe_float(last["adx"], 0), 1), "atr": float(last["atr"]),
            "ema21": safe_float(last["ema21"]), "ema50": safe_float(last["ema50"]), "ema200": safe_float(last["ema200"]),
            "hist_up": bool(last["hist"] > prev["hist"]), "vol_ratio": round(safe_float(last["vol_ratio"], 1), 2),
            "bars": len(df)}


# ---------- plans ----------
def plan_medium(price, atr, sups, ress, status) -> dict:
    near = [s["price"] for s in sups if s["price"] >= price - 3.5 * atr]
    s1 = near[0] if near else price - 1.5 * atr
    if status == "BUY":
        lo, hi = max(s1, price - 1.2 * atr), price
    else:  # รอย่อมาที่แนวรับ
        hi = min(price, s1 + 0.5 * atr)
        lo = min(s1 - 0.2 * atr, hi - 0.3 * atr)
    entry = (lo + hi) / 2
    below = [s["price"] for s in sups if s["price"] < lo - 0.3 * atr]
    sl = below[0] - 0.5 * atr if below and lo - below[0] <= 2.0 * atr else lo - 1.5 * atr
    sl = min(sl, entry - 1.2 * atr)
    risk = entry - sl
    above = [r["price"] for r in ress if r["price"] > entry + 1.5 * risk]
    tp1 = min(above[0], entry + 3 * risk) if above else entry + 2 * risk
    above2 = [r for r in above if r > tp1 + risk]
    tp2 = min(above2[0], entry + 6 * risk) if above2 else entry + 4 * risk
    tp2 = max(tp2, tp1 + risk)
    pct = lambda x: (x / entry - 1) * 100  # noqa: E731
    return {"entry_low": lo, "entry_high": hi, "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2,
            "rr1": (tp1 - entry) / risk, "rr2": (tp2 - entry) / risk,
            "sl_pct": pct(sl), "tp1_pct": pct(tp1), "tp2_pct": pct(tp2), "dist_pct": (entry / price - 1) * 100}


def plan_long(price, atr_w, sups, ress, hi52, lo52, ath, emas_w, status) -> dict:
    fibs = {f"Fib {f}": hi52 - f * (hi52 - lo52) for f in (0.382, 0.5, 0.618, 0.786)} if hi52 > lo52 else {}
    lv = {s["price"]: s["kinds"] for s in sups}
    lv.update({v: k for k, v in fibs.items()})
    lv.update({v: k for k, v in emas_w.items() if np.isfinite(v)})
    below = sorted([p for p in lv if p < price * 0.995], reverse=True)

    def pick(target, lo_b, hi_b, default):
        c = [x for x in below if lo_b <= x <= hi_b]
        return (min(c, key=lambda x: abs(x - target)), lv[min(c, key=lambda x: abs(x - target))]) if c else (default, "ระยะห่างมาตรฐาน")

    if status == "BUY":
        t1, t1n = price, "ราคาปัจจุบัน"
    else:
        t1, t1n = pick(price * 0.95, price * 0.90, price * 0.99, price * 0.96)
    t2, t2n = pick(price * 0.85, price * 0.74, min(t1 * 0.97, price * 0.92), min(t1 * 0.9, price * 0.85))
    t3, t3n = pick(price * 0.70, max(lo52 * 0.85, price * 0.45), min(t2 * 0.95, price * 0.80),
                   min(t2 * 0.88, max(price * 0.70, lo52)))
    w = {"BUY": (0.40, 0.35, 0.25), "WAIT": (0.25, 0.40, 0.35), "AVOID": (0.15, 0.40, 0.45)}[status]
    avg = 1 / sum(wi / ti for wi, ti in zip(w, (t1, t2, t3)))  # ซื้อด้วยเงินเท่ากันตามน้ำหนัก
    invalid = t3 - clip(atr_w, 0.08 * t3, 0.20 * t3)  # ต่ำกว่า Tier 3 ไม่เกิน 8-20%
    tp1_c = [r["price"] for r in ress if r["price"] > price * 1.2]
    tp1 = min([x for x in (tp1_c[0] if tp1_c else None, hi52 if hi52 > price * 1.2 else None) if x] or [price * 1.35])
    ext = lo52 + 1.618 * (hi52 - lo52)
    if np.isfinite(ath) and tp1 * 1.15 < ath < price * 5:
        tp2, tp2n = ath, "ATH เดิม"
    elif ext > tp1 * 1.15:
        tp2, tp2n = ext, "Fib Extension 1.618"
    else:
        tp2, tp2n = tp1 * 1.5, "+50% จาก TP1"
    if tp2 > tp1 * 2.2:  # กันเป้าไกลเกินจริงในกรอบเวลาลงทุน
        tp2, tp2n = tp1 * 2.2, f"{tp2n} (จำกัดที่ 2.2x TP1)"
    pct = lambda x: (x / avg - 1) * 100  # noqa: E731
    return {"tiers": [{"price": t1, "weight": w[0], "basis": t1n}, {"price": t2, "weight": w[1], "basis": t2n},
                      {"price": t3, "weight": w[2], "basis": t3n}],
            "avg": avg, "invalid": invalid, "tp1": tp1, "tp2": tp2, "tp2_basis": tp2n,
            "tp1_pct": pct(tp1), "tp2_pct": pct(tp2), "invalid_pct": pct(invalid),
            "rr1": (tp1 - avg) / (avg - invalid) if avg > invalid else np.nan, "fibs": fibs}


def status_medium(s1d, s4h, rsi1d) -> str:
    if s1d >= 1 and s4h >= -0.5 and rsi1d < 72:
        return "BUY"
    if s1d >= 0 or (s1d >= -0.5 and s4h >= 1):
        return "WAIT"
    return "AVOID"


def status_long(sw, rsi_w, s1d, dd_ath) -> str:
    if sw >= 0.5 and rsi_w < 75:
        return "BUY"
    if sw >= -0.5 or (np.isfinite(dd_ath) and dd_ath <= -60 and rsi_w < 45 and s1d >= -0.5):
        return "WAIT"
    return "AVOID"


def analyze_coin(dfs: dict, btc_d1: pd.DataFrame | None) -> dict:
    d1 = add_ind(dfs["1d"])
    if len(d1) < 60:
        raise ValueError("ข้อมูล 1D น้อยกว่า 60 วัน")
    h4 = add_ind(dfs["4h"]) if dfs.get("4h") is not None and len(dfs["4h"]) > 60 else None
    w1 = add_ind(dfs["1w"]) if dfs.get("1w") is not None and len(dfs["1w"]) > 15 else None
    snap = {"1d": tf_snapshot(d1), "4h": tf_snapshot(h4) if h4 is not None else None,
            "1w": tf_snapshot(w1) if w1 is not None else None}
    price = float(d1["close"].iloc[-1])
    atr_d, atr_w = snap["1d"]["atr"], (snap["1w"]["atr"] if snap["1w"] else snap["1d"]["atr"] * 2.6)
    vp = volume_profile(d1)
    levels = key_levels(d1, w1, vp)
    sups = sorted([lv for lv in levels if lv["price"] < price * 0.995], key=lambda x: -x["price"])
    ress = sorted([lv for lv in levels if lv["price"] > price * 1.005], key=lambda x: x["price"])
    last365 = d1.tail(365)
    hi52, lo52 = float(last365["high"].max()), float(last365["low"].min())
    ath_bin = float((w1 if w1 is not None else d1)["high"].max())
    s4h = snap["4h"]["score"] if snap["4h"] else 0.0
    sw = snap["1w"]["score"] if snap["1w"] else snap["1d"]["score"] * 0.6
    rsi_w = snap["1w"]["rsi"] if snap["1w"] else 50.0
    dd_ath = (price / ath_bin - 1) * 100
    st_med = status_medium(snap["1d"]["score"], s4h, snap["1d"]["rsi"])
    st_long = status_long(sw, rsi_w, snap["1d"]["score"], dd_ath)
    emas_w = {"EMA21 1W": snap["1w"]["ema21"], "EMA50 1W": snap["1w"]["ema50"]} if snap["1w"] else {}
    if snap["1w"] and np.isfinite(snap["1w"]["ema200"]):
        emas_w["EMA200 1W"] = snap["1w"]["ema200"]
    bt = backtest_1d(d1)
    fwd_m, fwd_l = forward_returns(d1, MED_FWD), forward_returns(d1, LONG_FWD)
    return {
        "price": price, "d1": d1.tail(260).reset_index(drop=True),
        "w1": w1.tail(160).reset_index(drop=True) if w1 is not None else None,
        "snap": snap, "vp": vp, "sups": sups[:6], "ress": ress[:6], "hi52": hi52, "lo52": lo52,
        "ath_bin": ath_bin, "dd_ath": dd_ath, "struct_1d": market_structure(d1.tail(250).reset_index(drop=True), 3),
        "struct_1w": market_structure(w1.tail(160).reset_index(drop=True), 2) if w1 is not None else "ไม่ชัด",
        "status_med": st_med, "status_long": st_long,
        "plan_med": plan_medium(price, atr_d, sups, ress, st_med),
        "plan_long": plan_long(price, atr_w, sups, ress, hi52, lo52, ath_bin, emas_w, st_long),
        "bt": bt, "fwd_med": fwd_m, "fwd_long": fwd_l, "risk": risk_metrics(d1, btc_d1),
        "obv_up": bool(d1["obv"].iloc[-1] > d1["obv"].iloc[-30]) if len(d1) > 30 else None,
        "ret_series": np.log(d1["close"]).diff().tail(90).set_axis(d1["time"].tail(90)),
    }


def med_win_hold(a: dict) -> tuple:
    bt, sn = a["bt"], a["snap"]
    wr = bt["wr_smooth"]
    wr += {"BUY": 3, "WAIT": -2, "AVOID": -8}[a["status_med"]]
    if sn["4h"] and sn["4h"]["score"] >= 0.5:
        wr += 2
    if sn["1d"]["rsi"] > 72:
        wr -= 4
    if sn["1w"]:
        wr += 3 if sn["1w"]["score"] >= 0.5 else (-4 if sn["1w"]["score"] <= -1 else 0)
    if bt["n"] < 8:
        wr = 0.5 * wr + 0.5 * 45
    if np.isfinite(bt["hold_p25"]):
        lo, hi = bt["hold_p25"], max(bt["hold_p75"], bt["hold_p25"] + 3)
    elif a["fwd_med"]["best"]:
        h = a["fwd_med"]["best"]["h"]
        lo, hi = h * 0.5, h
    else:
        lo, hi = 14, 45
    lo, hi = clip(lo, *MED_HOLD_LIMIT), clip(hi, *MED_HOLD_LIMIT)
    return round(clip(wr, 15, 85), 1), (lo, max(hi, lo + 3))


def long_win_hold(a: dict, fund_score: float) -> tuple:
    best = a["fwd_long"]["best"]
    if best:
        wr = (best["win"] * best["n"] + 50 * 10) / (best["n"] + 10)
        h = best["h"]
    else:
        wr, h = 50.0, 180
    wr += {"BUY": 3, "WAIT": 0, "AVOID": -5}[a["status_long"]]
    wr += 3 if fund_score >= 65 else (-4 if fund_score < 35 else 0)
    span = {90: (60, 120), 180: (120, 240), 365: (240, 450)}.get(h, (h * 0.66, h * 1.33))
    return round(clip(wr, 15, 85), 1), (clip(span[0], *LONG_HOLD_LIMIT), clip(span[1], *LONG_HOLD_LIMIT))


# =============================================================================
# 4) FUNDAMENTAL AGENT
# =============================================================================
def cg_headers(key):
    return {"x-cg-demo-api-key": key} if key else {}


@st.cache_data(ttl=900, show_spinner=False)
def cg_markets(ids: tuple, key: str) -> dict:
    r = requests.get("https://api.coingecko.com/api/v3/coins/markets", headers=cg_headers(key), timeout=20,
                     params={"vs_currency": "usd", "ids": ",".join(ids), "per_page": 250,
                             "price_change_percentage": "7d,30d,200d,1y"})
    r.raise_for_status()
    return {c["id"]: c for c in r.json()}


@st.cache_data(ttl=86400, show_spinner=False)
def cg_by_symbols(symbols: tuple, key: str) -> dict:
    r = requests.get("https://api.coingecko.com/api/v3/coins/markets", headers=cg_headers(key), timeout=20,
                     params={"vs_currency": "usd", "symbols": ",".join(s.lower() for s in symbols),
                             "include_tokens": "top", "per_page": 250})
    r.raise_for_status()
    best = {}
    for c in r.json():
        s = c["symbol"].upper()
        if s not in best or (c.get("market_cap") or 0) > (best[s].get("market_cap") or 0):
            best[s] = c
    return {s: c["id"] for s, c in best.items()}


@st.cache_data(ttl=1800, show_spinner=False)
def cg_global(key: str) -> dict:
    r = requests.get("https://api.coingecko.com/api/v3/global", headers=cg_headers(key), timeout=15)
    r.raise_for_status()
    d = r.json()["data"]
    return {"btc_dom": d["market_cap_percentage"].get("btc"), "eth_dom": d["market_cap_percentage"].get("eth"),
            "mcap_chg_24h": d.get("market_cap_change_percentage_24h_usd"),
            "total_mcap": d.get("total_market_cap", {}).get("usd")}


@st.cache_data(ttl=43200, show_spinner=False)
def cg_detail(cid: str, key: str) -> dict:
    r = requests.get(f"https://api.coingecko.com/api/v3/coins/{cid}", headers=cg_headers(key), timeout=20,
                     params={"localization": "false", "tickers": "false", "market_data": "false",
                             "community_data": "false", "developer_data": "true", "sparkline": "false"})
    r.raise_for_status()
    d = r.json()
    return {"categories": [c for c in (d.get("categories") or []) if c][:5],
            "genesis": d.get("genesis_date"), "commits_4w": (d.get("developer_data") or {}).get("commit_count_4_weeks"),
            "desc": re.sub(r"<[^>]+>", "", (d.get("description") or {}).get("en", ""))[:400],
            "homepage": ((d.get("links") or {}).get("homepage") or [""])[0]}


@st.cache_data(ttl=21600, show_spinner=False)
def llama_data() -> dict:
    protos = requests.get("https://api.llama.fi/protocols", timeout=40).json()
    chains = requests.get("https://api.llama.fi/v2/chains", timeout=20).json()
    fees = requests.get("https://api.llama.fi/overview/fees", timeout=40,
                        params={"excludeTotalDataChart": "true", "excludeTotalDataChartBreakdown": "true"}).json()
    p_list = [{"name": (p.get("name") or "").lower(), "gecko": p.get("gecko_id"), "tvl": p.get("tvl") or 0,
               "cat": p.get("category") or "", "chg7": p.get("change_7d")} for p in protos if isinstance(p, dict)]
    c_map = {c.get("gecko_id"): c.get("tvl") or 0 for c in chains if isinstance(c, dict) and c.get("gecko_id")}
    f_list = []
    for f in fees.get("protocols", []):
        v30 = f.get("total30d")
        if v30 is None:
            v30 = (f.get("total7d") or 0) * 30 / 7 or (f.get("total24h") or 0) * 30
        f_list.append({"name": (f.get("displayName") or f.get("name") or "").lower(), "f30": v30 or 0,
                       "type": f.get("protocolType") or ""})
    return {"protocols": p_list, "chains": c_map, "fees": f_list}


def llama_match(cid: str, name: str, ll: dict) -> dict:
    key = (name or "").lower().strip()
    is_chain = cid in ll["chains"]

    def match(n):
        return n == key if is_chain else (n == key or n.startswith(key + " "))

    tvl = ll["chains"].get(cid) if is_chain else sum(
        p["tvl"] for p in ll["protocols"] if (p["gecko"] == cid or match(p["name"])) and p["cat"] not in ("CEX", "Chain"))
    fees30 = sum(f["f30"] for f in ll["fees"] if match(f["name"]))
    return {"tvl": tvl or np.nan, "fees_30d": fees30 or np.nan, "is_chain": is_chain}


def fundamental_score(f: dict) -> float:
    if not f:
        return 50.0
    s = 50.0
    rank = f.get("rank") or 999
    s += 12 if rank <= 10 else (8 if rank <= 30 else (3 if rank <= 100 else -5))
    fm = f.get("fdv_mcap")
    if fm and np.isfinite(fm):
        s += 8 if fm < 1.2 else (2 if fm < 2 else (-5 if fm < 3 else -12))
    cr = f.get("circ_ratio")
    if cr and np.isfinite(cr) and cr < 0.4:
        s -= 6
    vm = f.get("vol_mcap")
    if vm and np.isfinite(vm):
        s += 5 if vm > 0.1 else (-5 if vm < 0.02 else 0)
    cm = f.get("commits_4w")
    if cm:
        s += 5 if cm > 100 else (2 if cm > 20 else 0)
    pf = f.get("pf")
    if pf and np.isfinite(pf):
        s += 10 if pf < 20 else (5 if pf < 50 else (-5 if pf > 300 else 0))
    return clip(s, 0, 100)


def run_fundamentals(bases: list, cg_key: str) -> tuple:
    out, notes = {}, []
    ids = {b: CG_IDS[b] for b in bases if b in CG_IDS}
    unknown = [b for b in bases if b not in CG_IDS]
    if unknown:
        try:
            ids.update(cg_by_symbols(tuple(unknown), cg_key))
        except Exception as e:  # noqa: BLE001
            notes.append(f"หา CoinGecko ID ไม่ได้ ({str(e)[:60]})")
    if not ids:
        return {}, notes + ["ไม่พบ CoinGecko ID ของเหรียญใดเลย"]
    try:
        mk = cg_markets(tuple(sorted(set(ids.values()))), cg_key)
    except Exception as e:  # noqa: BLE001
        return {}, [f"CoinGecko ใช้ไม่ได้ ({str(e)[:80]}) → ข้ามข้อมูลพื้นฐาน"]
    try:
        ll = llama_data()
    except Exception as e:  # noqa: BLE001
        ll = None
        notes.append(f"DefiLlama ใช้ไม่ได้ ({str(e)[:60]})")
    for b, cid in ids.items():
        m = mk.get(cid)
        if not m:
            continue
        mcap, fdv = m.get("market_cap") or np.nan, m.get("fully_diluted_valuation") or np.nan
        circ, tot, mx = m.get("circulating_supply"), m.get("total_supply"), m.get("max_supply")
        f = {"id": cid, "name": m.get("name"), "rank": m.get("market_cap_rank"), "mcap": mcap, "fdv": fdv,
             "fdv_mcap": fdv / mcap if mcap and np.isfinite(fdv) else np.nan,
             "circ_ratio": (circ / (mx or tot)) if circ and (mx or tot) else np.nan,
             "vol_mcap": (m.get("total_volume") or 0) / mcap if mcap else np.nan,
             "ath": m.get("ath") or np.nan, "ath_chg": m.get("ath_change_percentage") or np.nan,
             "ath_date": (m.get("ath_date") or "")[:10],
             "chg_30d": m.get("price_change_percentage_30d_in_currency"),
             "chg_1y": m.get("price_change_percentage_1y_in_currency"),
             "tvl": np.nan, "fees_30d": np.nan, "pf": np.nan, "mcap_tvl": np.nan}
        if ll:
            lm = llama_match(cid, f["name"], ll)
            f.update({"tvl": lm["tvl"], "fees_30d": lm["fees_30d"]})
            if np.isfinite(lm["fees_30d"]) and lm["fees_30d"] > 0 and np.isfinite(mcap):
                f["pf"] = mcap / (lm["fees_30d"] * 12)
            if np.isfinite(lm["tvl"]) and lm["tvl"] > 0 and np.isfinite(mcap):
                f["mcap_tvl"] = mcap / lm["tvl"]
        out[b] = f
    return out, notes


def enrich_details(bases: list, fund: dict, cg_key: str):
    for i, b in enumerate(bases):
        f = fund.get(b)
        if not f:
            continue
        try:
            f.update(cg_detail(f["id"], cg_key))
        except Exception as e:  # noqa: BLE001
            if "429" in str(e):
                break  # ติด rate limit ของ CoinGecko ฟรี → ข้ามที่เหลือ
        if not cg_key and i < len(bases) - 1:
            time.sleep(1.5)


# =============================================================================
# 5) MACRO AGENT
# =============================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def fear_greed() -> dict:
    d = requests.get("https://api.alternative.me/fng/?limit=8", timeout=10).json()["data"]
    return {"value": int(d[0]["value"]), "label": d[0]["value_classification"],
            "week_ago": int(d[-1]["value"]) if len(d) > 1 else None}


def macro_view(analyses: dict, glob: dict | None, fng: dict | None) -> dict:
    btc = analyses.get("BTC")
    v = {"btc_1w": "-", "btc_1d": "-", "btc_sw": 0.0, "glob": glob, "fng": fng}
    if btc:
        v["btc_1d"] = btc["snap"]["1d"]["trend"]
        if btc["snap"]["1w"]:
            v["btc_1w"], v["btc_sw"] = btc["snap"]["1w"]["trend"], btc["snap"]["1w"]["score"]
    s = v["btc_sw"]
    v["regime"] = "Bull 🟢" if s >= 1 else ("Bear 🔴" if s <= -1 else "Neutral 🟡")
    return v


# =============================================================================
# NEWS (RSS) — ใช้เป็นข้อมูลประกอบให้ Research Agent
# =============================================================================
def fetch_rss(per_feed=25) -> list:
    items = []
    for src, url in RSS_FEEDS.items():
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            for e in feedparser.parse(resp.content).entries[:per_feed]:
                items.append({"source": src, "title": e.get("title", ""), "time": e.get("published", ""),
                              "url": e.get("link", "")})
        except Exception:  # noqa: BLE001
            continue
    return items


def headlines_for(base: str, name: str, items: list) -> list:
    pats = [rf"\b{re.escape(base)}\b"] + ([rf"\b{re.escape(name.lower())}\b"] if name else [])
    return [i["title"] for i in items
            if re.search(pats[0], i["title"]) or any(re.search(p, i["title"].lower()) for p in pats[1:])][:4]


# =============================================================================
# GEMINI
# =============================================================================
def extract_json(text: str) -> str:
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    a, b = t.find("{"), t.rfind("}")
    return t[a:b + 1] if a >= 0 and b > a else t


def extract_sources(resp) -> list:
    out = []
    try:
        gm = getattr(resp.candidates[0], "grounding_metadata", None)
        for ch in (getattr(gm, "grounding_chunks", None) or []):
            web = getattr(ch, "web", None)
            if web and getattr(web, "uri", None):
                out.append({"title": getattr(web, "title", None) or web.uri, "url": web.uri})
    except Exception:  # noqa: BLE001
        pass
    return out


BACKUP_MODELS = ["gemini-3.8-flash", "gemini-3.1-flash-lite", "gemini-3-flash-preview", "gemini-2.5-flash-lite"]


def find_replacement_model(client, err_msg: str, tried: set):
    for m in re.findall(r"models/([\w.\-]+)", err_msg):
        if m not in tried:
            return m
    for m in BACKUP_MODELS:  # รายชื่อสำรองที่รู้จัก (ไม่ต้องรอ list API)
        if m not in tried and not st.session_state.get("dead_models", {}).get(m):
            return m
    try:
        names = []
        for m in client.models.list():
            name = (m.name or "").replace("models/", "")
            acts = getattr(m, "supported_actions", None) or []
            if "flash" in name and (not acts or "generateContent" in acts) and not any(
                    k in name for k in ("image", "tts", "live", "audio", "embedding", "thinking")):
                names.append(name)

        def rank(n):
            return ("preview" not in n and "exp" not in n, "lite" not in n,
                    [int(x) for x in re.findall(r"\d+", n)[:2]] or [0])

        for n in sorted(names, key=rank, reverse=True):
            if n not in tried:
                return n
    except Exception:  # noqa: BLE001
        pass
    return None


def make_client(raw_key: str, model: str):
    """ตรวจ API Key ก่อนเริ่ม (กันสแกนไปครึ่งทางแล้วค่อยพัง)
    คีย์แบบใหม่ที่ขึ้นต้นด้วย 'AQ.' บางบัญชีใช้กับ endpoint ปกติไม่ได้ → ลองโหมด Vertex AI Express ให้อัตโนมัติ"""
    key = (raw_key or "").strip().strip('"').strip("'").strip()
    attempts = [("Gemini API", {"api_key": key})]
    if key.startswith("AQ."):
        attempts.append(("Vertex AI Express", {"api_key": key, "vertexai": True}))
    errors = []
    for label, kw in attempts:
        try:
            c = genai.Client(**kw)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{label}: {str(e)[:160]}")
            continue
        try:
            c.models.generate_content(model=model, contents="ping",
                                      config=types.GenerateContentConfig(max_output_tokens=5))
            return c, label
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if any(k in msg for k in ("401", "403", "UNAUTHENTICATED", "PERMISSION_DENIED", "API_KEY_INVALID",
                                       "API key not valid")):
                errors.append(f"{label}: {msg[:160]}")
                continue
            return c, label  # error อื่น (โมเดลไม่ว่าง/404) ไม่เกี่ยวกับคีย์ → ไปต่อได้ เดี๋ยว gemini_call จัดการ
    hint = ("คีย์ขึ้นต้นด้วย 'AQ.' ซึ่งเป็นรูปแบบใหม่ที่ยังใช้ไม่ได้กับบางบัญชี → สร้างคีย์แบบ 'AIza...' แทน"
            if key.startswith("AQ.") else
            "ตรวจว่าคัดลอกคีย์ครบ ไม่มีช่องว่าง/เครื่องหมายคำพูด และเป็นคีย์จาก aistudio.google.com/apikey")
    raise RuntimeError(f"GEMINI_API_KEY ใช้ไม่ได้ — {hint}\nรายละเอียด: " + " | ".join(errors))


def gemini_call(client, model, prompt, schema=None, search=False, temperature=0.3, max_retries=4):
    """คืน (ผลลัพธ์, แหล่งอ้างอิง) — schema=None คืนข้อความ"""
    kw = {"temperature": temperature}
    if schema is not None:
        kw.update(response_mime_type="application/json", response_schema=schema)
    if search:
        kw["tools"] = [types.Tool(google_search=types.GoogleSearch())]
    cfg = types.GenerateContentConfig(**kw)
    model = st.session_state.get("model_override", {}).get(model, model)
    requested, tried, busy = model, {model}, {}
    last_err, attempt = None, 0
    while attempt < max_retries:
        attempt += 1
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
            src = extract_sources(resp)
            if schema is None:
                if not (resp.text or "").strip():
                    raise ValueError("Gemini ตอบกลับว่าง")
                return resp.text, src
            parsed = getattr(resp, "parsed", None)
            if isinstance(parsed, schema):
                return parsed, src
            if isinstance(parsed, dict):
                return schema.model_validate(parsed), src
            if not (resp.text or "").strip():
                raise ValueError("Gemini ตอบกลับว่าง")
            return schema.model_validate_json(extract_json(resp.text)), src
        except Exception as e:  # noqa: BLE001
            last_err, msg = e, str(e)

            def switch(gone):
                new = find_replacement_model(client, msg if gone else "", tried)
                if new:
                    st.toast(f"{model} {'ใช้ไม่ได้แล้ว' if gone else 'ไม่ว่าง/โควต้าเต็ม'} → สลับไปใช้ {new}")
                    st.session_state.setdefault("model_override", {})[requested] = new
                    st.session_state["active_model"] = new
                    tried.add(new)
                return new

            if ("404" in msg or "NOT_FOUND" in msg) and len(tried) < 6:
                st.session_state.setdefault("dead_models", {})[model] = True
                new = switch(True)
                if new:
                    model, attempt = new, attempt - 1
                    continue
            if any(k in msg for k in ("429", "RESOURCE_EXHAUSTED", "503", "500", "UNAVAILABLE", "INTERNAL",
                                       "overloaded", "high demand", "DEADLINE")):
                busy[model] = busy.get(model, 0) + 1
                if busy[model] >= 2 and len(tried) < 6:
                    new = switch(False)
                    if new:
                        model = new
                        continue
                wait = min(60, 8 * 2 ** (busy[model] - 1)) + float(np.random.uniform(0, 3))  # 8, 16, 32, 60 วินาที
                st.toast(f"Gemini ไม่ว่างชั่วคราว ({model}) — รอ {wait:.0f} วินาที")
                time.sleep(wait)
                continue
            if isinstance(e, ValueError):
                time.sleep(3)
                continue
            raise
    raise RuntimeError(f"Gemini ไม่ตอบหลังลอง {max_retries} ครั้ง: {last_err}")


# =============================================================================
# 6) RESEARCH AGENT (Gemini + Google Search)
# =============================================================================
class Event(BaseModel):
    date: str = Field(description="วันที่ YYYY-MM-DD; ถ้าไม่รู้วันแน่นอนใช้ YYYY-MM หรือ 'Q1 2027'")
    title: str = Field(description="ชื่อเหตุการณ์ ภาษาไทยสั้นๆ")
    type: str = Field(description="UNLOCK, UPGRADE, MAINNET, LAUNCH, LISTING, ETF, GOVERNANCE, PARTNERSHIP, "
                                  "BURN, CONFERENCE, MACRO, OTHER")
    impact: str = Field(description="POSITIVE, NEGATIVE หรือ NEUTRAL")
    certainty: str = Field(description="CONFIRMED (ประกาศทางการ), EXPECTED (roadmap/คาดการณ์), RUMOR")
    source_url: str = Field(description="URL แหล่งที่มาที่ค้นเจอ")


class CoinResearch(BaseModel):
    symbol: str
    project_th: str = Field(description="โปรเจกต์ทำอะไร ใครใช้ ภาษาไทย 2-3 ประโยค")
    sector: str = Field(description="กลุ่ม เช่น Layer1, DeFi, AI, RWA, Meme, Payment, Privacy")
    moat_th: str = Field(description="จุดแข็ง/ความได้เปรียบทางการแข่งขัน")
    tokenomics_th: str = Field(description="อุปทาน เงินเฟ้อ การปลดล็อก buyback/burn ที่สำคัญ")
    recent_news_th: str = Field(description="ข่าวสำคัญล่าสุด 2-4 เรื่อง พร้อมวันที่")
    bull_case_th: list[str] = Field(description="เหตุผลขาขึ้น 2-4 ข้อ")
    bear_case_th: list[str] = Field(description="ความเสี่ยง 2-4 ข้อ")
    catalyst_score: int = Field(description="-100 (ลบมาก) ถึง 100 (บวกมาก) จากข่าว+กำหนดการ 1-6 เดือนข้างหน้า")
    news_sentiment: str = Field(description="BULLISH, BEARISH หรือ NEUTRAL")
    events: list[Event] = Field(description="กำหนดการ 6 เดือนข้างหน้า (และที่เพิ่งเกิดใน 30 วัน)")


class ResearchBatch(BaseModel):
    coins: list[CoinResearch]


def research_prompt(payload: list, today: str, json_hint: bool) -> str:
    lines = []
    for p in payload:
        lines.append(f"- {p['symbol']} ({p['name']}) ราคา ${fmt_price(p['price'])}, ห่าง ATH {fmt_pct(p['dd'])}, "
                     f"หมวด: {', '.join(p['cats']) or '-'}"
                     + (f"\n  พาดหัวล่าสุด: {' | '.join(p['heads'])}" if p["heads"] else ""))
    hint = ""
    if json_hint:
        hint = ("\n\nตอบเป็น JSON เท่านั้น รูปแบบ: {\"coins\":[{\"symbol\":\"\",\"project_th\":\"\",\"sector\":\"\","
                "\"moat_th\":\"\",\"tokenomics_th\":\"\",\"recent_news_th\":\"\",\"bull_case_th\":[],\"bear_case_th\":[],"
                "\"catalyst_score\":0,\"news_sentiment\":\"\",\"events\":[{\"date\":\"\",\"title\":\"\",\"type\":\"\","
                "\"impact\":\"\",\"certainty\":\"\",\"source_url\":\"\"}]}]}")
    return f"""คุณคือ Crypto Research Analyst ระดับสถาบัน วันนี้คือ {today}
ค้นข้อมูลล่าสุดจากเว็บสำหรับเหรียญต่อไปนี้ แล้ววิเคราะห์เชิงลึก:
{chr(10).join(lines)}

สำหรับแต่ละเหรียญ:
1. โปรเจกต์ทำอะไร จุดแข็ง (moat) และกลุ่มอุตสาหกรรม
2. Tokenomics: อุปทานหมุนเวียน เงินเฟ้อ การปลดล็อกโทเคน (Token Unlock) ครั้งใหญ่ที่กำลังมา buyback/burn
3. ข่าวสำคัญ 30 วันล่าสุด (ระบุวันที่)
4. กำหนดการ 6 เดือนข้างหน้า: Unlock, Upgrade/Hard fork, Mainnet, เปิดตัวผลิตภัณฑ์, ETF (วันตัดสิน), Listing,
   Governance vote, งานสำคัญ — ใส่วันที่ให้แม่นที่สุด และ source_url ที่ค้นเจอจริง
5. catalyst_score: น้ำหนักข่าว + กำหนดการต่อราคา 1-6 เดือน

กฎ: ห้ามแต่งข้อมูล ถ้าค้นไม่เจอให้ events เป็นรายการว่าง และบอกว่า "ไม่พบข้อมูล" ใช้ภาษาไทยกระชับ{hint}"""


def run_research(client, model, payload: list, use_search: bool, today: str) -> tuple:
    """คืน (ResearchBatch, sources, mode)"""
    if use_search:
        try:
            obj, src = gemini_call(client, model, research_prompt(payload, today, False), ResearchBatch,
                                   search=True, temperature=0.2)
            return obj, src, "Google Search"
        except Exception:  # noqa: BLE001  — บางโมเดลใช้ schema+search พร้อมกันไม่ได้
            pass
        try:
            text, src = gemini_call(client, model, research_prompt(payload, today, True), None,
                                    search=True, temperature=0.2)
            return ResearchBatch.model_validate_json(extract_json(text)), src, "Google Search"
        except Exception:  # noqa: BLE001
            pass
    obj, src = gemini_call(client, model, research_prompt(payload, today, False), ResearchBatch, temperature=0.2)
    return obj, src, "ความรู้ของโมเดล (ไม่ได้ค้นเว็บ)"


def parse_event_date(s: str):
    s = (s or "").strip()
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    try:
        if m:
            return datetime(int(m[1]), int(m[2]), int(m[3]), tzinfo=timezone.utc)
        m = re.search(r"(\d{4})-(\d{1,2})", s)
        if m:
            return datetime(int(m[1]), int(m[2]), 15, tzinfo=timezone.utc)
        m = re.search(r"Q([1-4])\s*(\d{4})", s, re.I)
        if m:
            return datetime(int(m[2]), int(m[1]) * 3 - 1, 15, tzinfo=timezone.utc)
        m = re.search(r"H([12])\s*(\d{4})", s, re.I)
        if m:
            return datetime(int(m[2]), 3 if m[1] == "1" else 9, 30, tzinfo=timezone.utc)
    except ValueError:
        return None
    return None


# =============================================================================
# 7) STRATEGIST  &  8) RISK MANAGER
# =============================================================================
class Pick(BaseModel):
    symbol: str
    horizon: str = Field(description="MEDIUM หรือ LONG")
    conviction: str = Field(description="สูง, กลาง หรือ ต่ำ")
    action_th: str = Field(description="สิ่งที่ควรทำตอนนี้ เช่น 'ซื้อ 40% ตอนนี้ + ตั้ง Limit ที่ Tier 2/3'")
    win_rate: int = Field(description="ปรับจาก SysWR ได้ไม่เกิน ±10")
    hold_min_days: int
    hold_max_days: int
    allocation_pct: float = Field(description="% ของพอร์ตที่แนะนำ")
    thesis_th: str = Field(description="เหตุผลหลัก 3-4 ประโยค รวมโปรเจกต์+เทคนิค+ข่าว+สถิติ")
    technical_th: str = Field(description="มุมมองเทคนิค อ้างอิงตัวเลขจริง")
    fundamental_th: str = Field(description="มุมมองพื้นฐาน/มูลค่า อ้างอิงตัวเลขจริง")
    catalyst_th: str = Field(description="ข่าว/กำหนดการที่ต้องจับตา พร้อมวันที่")
    entry_trigger_th: str = Field(description="เงื่อนไขก่อนซื้อ")
    exit_plan_th: str = Field(description="แผนขายทำกำไร/ทยอยขาย และสัญญาณที่ควรขายก่อนกำหนด")
    invalidation_th: str = Field(description="เมื่อไหร่แผนนี้ผิด ต้องขาย")


class Strategy(BaseModel):
    market_view_th: str = Field(description="ภาพรวมตลาด 3-5 ประโยค")
    regime_advice_th: str = Field(description="ตอนนี้เหมาะกับระยะกลางหรือยาว ควรถือเงินสดเท่าไหร่ เพราะอะไร")
    cash_reserve_pct: float = Field(description="% เงินสด USDT ที่ควรเก็บไว้")
    picks: list[Pick]


class Review(BaseModel):
    symbol: str
    horizon: str
    verdict: str = Field(description="APPROVE, CAUTION หรือ REJECT")
    final_win_rate: int
    max_allocation_pct: float
    note_th: str


class RiskOut(BaseModel):
    portfolio_note_th: str
    concentration_th: str = Field(description="ความเสี่ยงกระจุกตัว/ความสัมพันธ์สูงระหว่างเหรียญ")
    reviews: list[Review]


def coin_line(b: str, a: dict, f: dict, r) -> str:
    pm, pl, sn = a["plan_med"], a["plan_long"], a["snap"]
    fb, fl = a["fwd_med"]["best"], a["fwd_long"]["best"]
    t = pl["tiers"]
    s = (f"## {b} px={fmt_price(a['price'])} ATH{fmt_pct(a['dd_ath'])} struct1D={a['struct_1d']} struct1W={a['struct_1w']}\n"
         f"  TECH 4H={sn['4h']['score'] if sn['4h'] else '-'} 1D={sn['1d']['score']}({sn['1d']['trend']},RSI{sn['1d']['rsi']},ADX{sn['1d']['adx']}) "
         f"1W={sn['1w']['score'] if sn['1w'] else '-'}({sn['1w']['trend'] if sn['1w'] else '-'},RSI{sn['1w']['rsi'] if sn['1w'] else '-'}) "
         f"OBV30d={'↑' if a['obv_up'] else '↓'} RS30={fmt_pct(a['risk']['rs30'])} RS90={fmt_pct(a['risk']['rs90'])} "
         f"beta={a['risk']['beta']:.2f} vol={a['risk']['vol_ann']:.0f}% maxDD1y={a['risk']['max_dd_1y']:.0f}%\n"
         f"  MEDIUM {STATUS_MED[a['status_med']]} score={a['score_med']:.0f} SysWR={a['wr_med']}% hold={range_days(*a['hold_med'])} "
         f"zone={fmt_price(pm['entry_low'])}-{fmt_price(pm['entry_high'])} SL={fmt_price(pm['sl'])}({fmt_pct(pm['sl_pct'])}) "
         f"TP1={fmt_price(pm['tp1'])}({fmt_pct(pm['tp1_pct'])}) TP2={fmt_price(pm['tp2'])}({fmt_pct(pm['tp2_pct'])}) "
         f"RR={pm['rr1']:.1f} BT n={a['bt']['n']} expR={a['bt']['exp_r']:+.2f}"
         + (f" fwd{fb['h']}d win{fb['win']:.0f}% med{fb['median']:+.0f}%" if fb else "") + "\n"
         f"  LONG {STATUS_LONG[a['status_long']]} score={a['score_long']:.0f} SysWR={a['wr_long']}% hold={range_days(*a['hold_long'])} "
         f"DCA={'/'.join(fmt_price(x['price']) for x in t)} avg={fmt_price(pl['avg'])} inval={fmt_price(pl['invalid'])} "
         f"TP1={fmt_price(pl['tp1'])}({fmt_pct(pl['tp1_pct'])}) TP2={fmt_price(pl['tp2'])}({fmt_pct(pl['tp2_pct'])})"
         + (f" fwd{fl['h']}d win{fl['win']:.0f}% med{fl['median']:+.0f}% p25{fl['p25']:+.0f}% n={fl['n']}" if fl else ""))
    if f:
        s += (f"\n  FUND rank#{f.get('rank')} mcap={fmt_big(f.get('mcap'))} FDV/MC={f.get('fdv_mcap', np.nan):.2f} "
              f"circ={f.get('circ_ratio', np.nan) * 100:.0f}% vol/mc={f.get('vol_mcap', np.nan):.3f} "
              f"TVL={fmt_big(f.get('tvl'))} fees30d={fmt_big(f.get('fees_30d'))} P/F={f.get('pf', np.nan):.0f} "
              f"score={a['score_fund']:.0f}")
    if r:
        s += (f"\n  RESEARCH catalyst={r.catalyst_score} {r.news_sentiment} | {r.project_th[:120]} | "
              f"events: " + "; ".join(f"{e.date} {e.title}({e.impact})" for e in r.events[:5]))
    return s


def run_strategist(client, model, analyses, fund, research, macro, n_med, n_long, corr_txt) -> tuple:
    body = "\n".join(coin_line(b, a, fund.get(b), research.get(b)) for b, a in analyses.items())
    g, fg = macro["glob"], macro["fng"]
    mtxt = (f"BTC 1W={macro['btc_1w']} 1D={macro['btc_1d']} regime={macro['regime']}"
            + (f" | BTC.D={g['btc_dom']:.1f}% ETH.D={g['eth_dom']:.1f}% TotalMcap24h={g['mcap_chg_24h']:+.1f}%" if g else "")
            + (f" | Fear&Greed={fg['value']} ({fg['label']}) สัปดาห์ก่อน {fg['week_ago']}" if fg else ""))
    prompt = f"""คุณคือ Chief Investment Officer กองทุนคริปโต ผู้เชี่ยวชาญ Multi-timeframe, Market Structure,
Tokenomics และ Catalyst-driven investing วันนี้ {datetime.now(timezone.utc):%Y-%m-%d}

ผู้ใช้ซื้อ SPOT เท่านั้น (ไม่มี Short, ไม่มี Leverage) ถือ 2 ระยะ:
- MEDIUM (Swing): ถือหลายวันถึง ~3 เดือน ใช้แผน zone/SL/TP ของระบบ
- LONG (สะสม): ถือ 2-24 เดือน ใช้แผน DCA 3 Tier + จุดตัดขาดทุน (weekly close) ของระบบ

=== MACRO ===
{mtxt}

=== ความสัมพันธ์สูง (90 วัน) ===
{corr_txt}

=== ข้อมูลทุกเหรียญ (ตัวเลขทั้งหมดคำนวณจริง) ===
คำอธิบาย: score 0-100 | SysWR = Win Rate จากสถิติย้อนหลัง (MEDIUM: ถึง TP1 ก่อน SL; LONG: กำไรเมื่อถือครบ)
fwd = ผลตอบแทนในอดีตเมื่อสภาพตลาดเหมือนวันนี้ (win% / median / p25) | hold = ระยะถือจากสถิติ
{body}

=== งาน ===
1. คัด MEDIUM ไม่เกิน {n_med} เหรียญ และ LONG ไม่เกิน {n_long} เหรียญ (เหรียญเดียวกันอยู่ได้ทั้ง 2 ระยะถ้าเหมาะจริง)
2. MEDIUM: ห้ามเลือกเหรียญ 🔴 ยังไม่ควรซื้อ, ให้ความสำคัญกับ RS เทียบ BTC เป็นบวก, expR เป็นบวก, catalyst ใกล้ถึง
3. LONG: เน้นพื้นฐาน (rank, FDV/MC ต่ำ, รายได้จริง P/F), ระยะห่าง ATH, โครงสร้าง 1W และ catalyst 3-12 เดือน
   ระวังเหรียญที่มี Token Unlock ใหญ่ใกล้ถึง (ลด allocation/ win_rate)
4. win_rate ปรับจาก SysWR ได้ ±10 เท่านั้น พร้อมเหตุผล; hold_min/max_days อิงช่วง hold ของระบบ (ปรับได้ตาม catalyst)
5. allocation_pct รวมทุกเหรียญ + cash_reserve_pct = 100; ตลาด Bear/Fear สูง ให้ถือเงินสดมากขึ้น;
   เหรียญใหญ่ (BTC/ETH) เป็นแกนพอร์ตระยะยาวได้มากกว่า altcoin; altcoin ตัวเดียวไม่ควรเกิน 15%
6. ไม่ต้องเขียนราคาใหม่ ระบบใช้ตัวเลขของตัวเอง แต่ในคำอธิบายให้อ้างตัวเลขจริง
7. ถ้าไม่มีเหรียญที่ดีพอ เลือกน้อยลงได้ และบอกให้ถือเงินสด
ภาษาไทย กระชับ มืออาชีพ"""
    return gemini_call(client, model, prompt, Strategy, temperature=0.3, max_retries=6)


def run_risk(client, model, picks: list, macro: dict, corr_txt: str):
    rows = "\n".join(
        f"- {p['symbol']} {p['horizon']} WR={p['win_rate']}% alloc={p['allocation_pct']:.1f}% "
        + (f"SL={fmt_pct(p['plan']['sl_pct'])} TP1={fmt_pct(p['plan']['tp1_pct'])} RR={p['plan']['rr1']:.1f}"
           if p["horizon"] == "MEDIUM" else
           f"avg={fmt_price(p['plan']['avg'])} inval={fmt_pct(p['plan']['invalid_pct'])} TP1={fmt_pct(p['plan']['tp1_pct'])}")
        + f" vol={p['vol_ann']:.0f}% beta={p['beta']:.2f} status={p['status']} conviction={p['conviction']}"
        for p in picks)
    prompt = f"""คุณคือ Chief Risk Officer พอร์ตคริปโต SPOT (ไม่มี Leverage)
Macro: {macro['regime']}, Fear&Greed {macro['fng']['value'] if macro['fng'] else '-'}
ความสัมพันธ์สูง: {corr_txt}
แผน:
{rows}

ตรวจ: การกระจุกตัว (altcoin ที่วิ่งตาม BTC ด้วย beta สูงหลายตัว), ความผันผวน vs สัดส่วน, R:R, ความเสี่ยงปลดล็อกโทเคน,
ความสอดคล้องกับภาวะตลาด
- final_win_rate ปรับได้ ±10, max_allocation_pct ลดได้อย่างเดียว
- REJECT เมื่อแผนขัดภาวะตลาดรุนแรงหรือ R:R แย่ ภาษาไทย"""
    obj, _ = gemini_call(client, model, prompt, RiskOut, temperature=0.2, max_retries=4)
    return obj


def norm_hz(h: str) -> str:
    h = (h or "").upper()
    return "LONG" if any(k in h for k in ("LONG", "ยาว", "INVEST", "ACCUM", "สะสม")) else "MEDIUM"


def finalize(strategy: Strategy, analyses: dict, fund: dict) -> tuple:
    picks, dropped = [], []
    for p in strategy.picks:
        b = re.sub(r"[/\-_ ]|USDT$", "", p.symbol.upper().strip())
        a = analyses.get(b)
        if not a:
            continue
        hz = norm_hz(p.horizon)
        if any(x["symbol"] == b and x["horizon"] == hz for x in picks):
            continue
        if hz == "MEDIUM" and a["status_med"] == "AVOID":
            dropped.append(f"{b} (ระยะกลาง: ยังเป็นขาลง)")
            continue
        sys_wr = a["wr_med"] if hz == "MEDIUM" else a["wr_long"]
        lo_lim, hi_lim = MED_HOLD_LIMIT if hz == "MEDIUM" else LONG_HOLD_LIMIT
        h_sys = a["hold_med"] if hz == "MEDIUM" else a["hold_long"]
        h_lo = clip(p.hold_min_days or h_sys[0], max(lo_lim, h_sys[0] * 0.5), h_sys[1] * 1.5)
        h_hi = clip(p.hold_max_days or h_sys[1], h_lo + 3, min(hi_lim, h_sys[1] * 1.5))
        picks.append({
            "symbol": b, "horizon": hz, "conviction": p.conviction, "action": p.action_th,
            "win_rate": int(clip(p.win_rate, max(5, sys_wr - 10), min(85, sys_wr + 10))), "sys_wr": sys_wr,
            "hold": (h_lo, h_hi), "allocation_pct": clip(p.allocation_pct, 0, 40),
            "thesis": p.thesis_th, "technical": p.technical_th, "fundamental": p.fundamental_th,
            "catalyst": p.catalyst_th, "trigger": p.entry_trigger_th, "exit": p.exit_plan_th,
            "invalidation": p.invalidation_th,
            "status": (STATUS_MED[a["status_med"]] if hz == "MEDIUM" else STATUS_LONG[a["status_long"]]),
            "plan": a["plan_med"] if hz == "MEDIUM" else a["plan_long"],
            "vol_ann": a["risk"]["vol_ann"], "beta": safe_float(a["risk"]["beta"], 1.0),
            "verdict": "-", "risk_note": "",
        })
    cash = clip(strategy.cash_reserve_pct, 0, 90)
    tot = sum(p["allocation_pct"] for p in picks)
    if tot > 0 and tot + cash > 100.5:
        k = (100 - cash) / tot
        for p in picks:
            p["allocation_pct"] = round(p["allocation_pct"] * k, 1)
    elif tot + cash < 99.5:
        cash = 100 - tot
    return picks, round(cash, 1), dropped


def merge_risk(picks: list, ro: RiskOut | None):
    if not ro:
        return
    m = {(re.sub(r"USDT$", "", r.symbol.upper()), norm_hz(r.horizon)): r for r in ro.reviews}
    for p in picks:
        r = m.get((p["symbol"], p["horizon"]))
        if r:
            p["verdict"] = (r.verdict or "-").upper()
            p["win_rate"] = int(clip(r.final_win_rate, max(5, p["win_rate"] - 10), min(85, p["win_rate"] + 10)))
            if r.max_allocation_pct and r.max_allocation_pct > 0:
                p["allocation_pct"] = round(min(p["allocation_pct"], r.max_allocation_pct), 1)
            p["risk_note"] = r.note_th


# =============================================================================
# CHARTS
# =============================================================================
def chart_medium(a: dict, p: dict, sym: str) -> go.Figure:
    d = a["d1"].tail(160)
    fig = go.Figure(go.Candlestick(x=d["time"], open=d["open"], high=d["high"], low=d["low"], close=d["close"],
                                   name="1D", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"))
    for col, colr in (("ema21", "#f5c542"), ("ema50", "#42a5f5"), ("ema200", "#ab47bc")):
        if d[col].notna().any():
            fig.add_trace(go.Scatter(x=d["time"], y=d[col], name=col.upper(), line=dict(width=1.2, color=colr)))
    fig.add_hrect(y0=p["entry_low"], y1=p["entry_high"], fillcolor="#2196f3", opacity=0.18, line_width=0,
                  annotation_text="โซนซื้อ", annotation_position="left")
    for nm, y, colr in (("SL", p["sl"], "#ef5350"), ("TP1", p["tp1"], "#66bb6a"), ("TP2", p["tp2"], "#2e7d32")):
        fig.add_hline(y=y, line_dash="dash", line_color=colr, annotation_text=f"{nm} {fmt_price(y)}",
                      annotation_position="top left", annotation_font_color=colr)
    if a["vp"]:
        fig.add_hline(y=a["vp"]["poc"], line_dash="dot", line_color="#ff9800",
                      annotation_text=f"POC {fmt_price(a['vp']['poc'])}", annotation_position="bottom right")
    fig.update_layout(title=f"{sym}/USDT · 1D · แผนระยะกลาง", height=440, xaxis_rangeslider_visible=False,
                      margin=dict(l=8, r=8, t=44, b=8), legend=dict(orientation="h", y=1.08))
    return fig


def chart_long(a: dict, p: dict, sym: str) -> go.Figure:
    d = a["w1"] if a["w1"] is not None else a["d1"]
    tf = "1W" if a["w1"] is not None else "1D"
    fig = go.Figure(go.Candlestick(x=d["time"], open=d["open"], high=d["high"], low=d["low"], close=d["close"],
                                   name=tf, increasing_line_color="#26a69a", decreasing_line_color="#ef5350"))
    for col, colr in (("ema21", "#f5c542"), ("ema50", "#42a5f5"), ("ema200", "#ab47bc")):
        if d[col].notna().any():
            fig.add_trace(go.Scatter(x=d["time"], y=d[col], name=f"{col.upper()} {tf}", line=dict(width=1.2, color=colr)))
    for i, t in enumerate(p["tiers"], 1):
        fig.add_hline(y=t["price"], line_color="#2196f3", line_width=1.5,
                      annotation_text=f"Tier {i} ({t['weight'] * 100:.0f}%) {fmt_price(t['price'])}",
                      annotation_position="top left", annotation_font_color="#2196f3")
    for nm, y, colr in (("ตัดขาดทุน (Weekly close)", p["invalid"], "#ef5350"), ("TP1", p["tp1"], "#66bb6a"),
                        ("TP2", p["tp2"], "#2e7d32")):
        fig.add_hline(y=y, line_dash="dash", line_color=colr, annotation_text=f"{nm} {fmt_price(y)}",
                      annotation_position="top left", annotation_font_color=colr)
    fig.update_layout(title=f"{sym}/USDT · {tf} · แผนสะสมระยะยาว", height=440, xaxis_rangeslider_visible=False,
                      margin=dict(l=8, r=8, t=44, b=8), legend=dict(orientation="h", y=1.08), yaxis_type="log")
    return fig


def chart_fwd(fwd: dict, title: str) -> go.Figure:
    rows = [r for r in fwd["rows"] if r["n"] > 0]
    if not rows:
        return go.Figure()
    df = pd.DataFrame(rows)
    df["label"] = df["h"].map(lambda h: fmt_days(h))
    fig = go.Figure()
    fig.add_bar(x=df["label"], y=df["median"], name="ผลตอบแทนกลาง (median)",
                marker_color=["#26a69a" if v >= 0 else "#ef5350" for v in df["median"]],
                text=[f"ชนะ {w:.0f}%" for w in df["win"]], textposition="outside")
    fig.add_scatter(x=df["label"], y=df["p25"], mode="markers+lines", name="กรณีแย่ (p25)",
                    line=dict(color="#ef5350", dash="dot"))
    fig.add_scatter(x=df["label"], y=df["p75"], mode="markers+lines", name="กรณีดี (p75)",
                    line=dict(color="#66bb6a", dash="dot"))
    fig.update_layout(title=title, height=320, yaxis_title="%", margin=dict(l=8, r=8, t=40, b=8),
                      legend=dict(orientation="h", y=-0.2))
    return fig


def chart_radar(a: dict, name: str) -> go.Figure:
    cats = ["เทคนิคกลาง", "เทคนิคยาว", "พื้นฐาน", "มูลค่า", "แข็งแรง vs BTC", "Catalyst"]
    vals = [a["s_tech_med"], a["s_tech_long"], a["score_fund"], a["s_value"], a["s_rs"], a["s_cat"]]
    fig = go.Figure(go.Scatterpolar(r=vals + vals[:1], theta=cats + cats[:1], fill="toself", name=name))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), height=320, margin=dict(l=30, r=30, t=30, b=30),
                      showlegend=False)
    return fig


# =============================================================================
# SCORES
# =============================================================================
def compute_scores(a: dict, f: dict, r):
    sn = a["snap"]
    s4 = sn["4h"]["score"] if sn["4h"] else 0
    sw = sn["1w"]["score"] if sn["1w"] else sn["1d"]["score"] * 0.6
    struct_adj = 5 if "HH/HL" in a["struct_1d"] else (-5 if "LH/LL" in a["struct_1d"] else 0)
    a["s_tech_med"] = clip(50 + sn["1d"]["score"] * 10 + s4 * 4 + struct_adj, 0, 100)
    a["s_tech_long"] = clip(50 + sw * 12 + sn["1d"]["score"] * 4 + (5 if "HH/HL" in a["struct_1w"] else 0), 0, 100)
    rs30, rs90 = safe_float(a["risk"]["rs30"], 0), safe_float(a["risk"]["rs90"], 0)
    a["s_rs"] = clip(50 + clip(rs90, -50, 50) * 0.6 + clip(rs30, -30, 30) * 0.5, 0, 100)
    a["score_fund"] = fundamental_score(f)
    dd = safe_float(f.get("ath_chg") if f else np.nan, a["dd_ath"])
    rsi_w = sn["1w"]["rsi"] if sn["1w"] else 50
    v = 50 + (15 if -85 <= dd <= -60 else (8 if -60 < dd <= -30 else (-5 if dd > -10 else 0)))
    v += 8 if rsi_w < 35 else (-10 if rsi_w > 75 else 0)
    a["s_value"] = clip(v, 0, 100)
    a["s_cat"] = clip(50 + r.catalyst_score / 2, 0, 100) if r else 50.0
    a["score_med"] = 0.45 * a["s_tech_med"] + 0.2 * a["s_rs"] + 0.2 * a["s_cat"] + 0.15 * a["score_fund"]
    a["score_long"] = 0.35 * a["score_fund"] + 0.25 * a["s_tech_long"] + 0.2 * a["s_cat"] + 0.2 * a["s_value"]
    a["wr_med"], a["hold_med"] = med_win_hold(a)
    a["wr_long"], a["hold_long"] = long_win_hold(a, a["score_fund"])


def corr_pairs(analyses: dict, thr=0.85) -> tuple:
    rets = pd.DataFrame({b: a["ret_series"] for b, a in analyses.items()}).dropna(how="all")
    corr = rets.corr(min_periods=30)
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if pd.notna(v) and v >= thr and "BTC" not in (cols[i], cols[j]):
                pairs.append((cols[i], cols[j], v))
    pairs.sort(key=lambda x: -x[2])
    txt = ", ".join(f"{x}-{y} {v:.2f}" for x, y, v in pairs[:12]) or "ไม่มีคู่ที่สูงกว่า 0.85"
    return corr, txt


# =============================================================================
# FALLBACK: แผนจากระบบ (เมื่อ Gemini ไม่ว่าง)
# =============================================================================
def rule_strategy(A_: dict, F_: dict, RS_: dict, M_: dict, n_med: int, n_long: int) -> Strategy:
    reg = M_["regime"]
    fg = (M_["fng"] or {}).get("value", 50)
    cash = 20.0 if "Bull" in reg else (55.0 if "Bear" in reg else 35.0)
    cash += 10 if fg >= 75 else 0

    def ok_bt(a):
        return not (np.isfinite(a["bt"]["exp_r"]) and a["bt"]["exp_r"] < 0)

    med = sorted([b for b, a in A_.items() if a["status_med"] == "BUY" and ok_bt(a)], key=lambda b: -A_[b]["score_med"])
    med += sorted([b for b, a in A_.items() if a["status_med"] == "WAIT" and b not in med and a["score_med"] >= 55],
                  key=lambda b: -A_[b]["score_med"])
    med = [b for b in med if A_[b]["wr_med"] >= 45][:n_med]
    lng = sorted([b for b, a in A_.items()
                  if a["status_long"] != "AVOID" and a["score_long"] >= 50 and a["wr_long"] >= 45],
                 key=lambda b: -A_[b]["score_long"])[:n_long]
    invest = 100 - cash
    long_budget = invest * 0.65 if (lng and med) else (invest if lng else 0)
    med_budget = invest - long_budget if med else 0
    alloc = {}
    if lng:
        w = {b: 2.0 if b in ("BTC", "ETH") else 1.0 for b in lng}
        for b in lng:
            alloc[(b, "LONG")] = min(long_budget * w[b] / sum(w.values()), 40 if b in ("BTC", "ETH") else 15)
    for b in med:
        alloc[(b, "MEDIUM")] = min(med_budget / len(med), 15)

    def conv(s):
        return "สูง" if s >= 70 else ("กลาง" if s >= 58 else "ต่ำ")

    picks = []
    for hz, group in (("MEDIUM", med), ("LONG", lng)):
        for b in group:
            a, f, r = A_[b], F_.get(b) or {}, RS_.get(b)
            sn, rk = a["snap"], a["risk"]
            tech = (f"1D {sn['1d']['trend']} (score {sn['1d']['score']}, RSI {sn['1d']['rsi']}), "
                    f"โครงสร้าง {a['struct_1d']}; 1W {sn['1w']['trend'] if sn['1w'] else '-'} "
                    f"({a['struct_1w']}); RS90 vs BTC {fmt_pct(rk['rs90'])}")
            fund_t = (f"อันดับ #{f.get('rank')}, MCap {fmt_big(f.get('mcap'))}, FDV/MC {f.get('fdv_mcap', np.nan):.2f}, "
                      f"ห่าง ATH {fmt_pct(a['dd_ath'])}" if f else f"ห่าง ATH {fmt_pct(a['dd_ath'])} (ไม่มีข้อมูลพื้นฐาน)")
            cat = ("; ".join(f"{e.date} {e.title}" for e in r.events[:3]) if r and r.events
                   else "ไม่มีข้อมูลกำหนดการ (ตรวจข่าวเพิ่มเอง)")
            if hz == "MEDIUM":
                pl, fb = a["plan_med"], a["fwd_med"]["best"]
                stat = (f"Backtest {a['bt']['wr_raw']:.0f}% (n={a['bt']['n']})" if a["bt"]["n"] else "Backtest ไม่พอ")
                picks.append(Pick(
                    symbol=b, horizon="MEDIUM", conviction=conv(a["score_med"]),
                    action_th=("ตั้ง Limit Buy ในโซนซื้อได้เลย" if a["status_med"] == "BUY"
                               else "รอราคาย่อลงมาในโซนซื้อ แล้วค่อยเข้า"),
                    win_rate=int(round(a["wr_med"])), hold_min_days=int(a["hold_med"][0]),
                    hold_max_days=int(a["hold_med"][1]), allocation_pct=round(alloc[(b, hz)], 1),
                    thesis_th=f"คะแนนระยะกลาง {a['score_med']:.0f}/100 · {stat}"
                              + (f" · ในอดีตสภาพเดียวกันถือ {fmt_days(fb['h'])} ชนะ {fb['win']:.0f}%" if fb else ""),
                    technical_th=tech, fundamental_th=fund_t, catalyst_th=cat,
                    entry_trigger_th="แท่ง 4H ปิดยืนเหนือ EMA21 ขณะราคาอยู่ในโซนซื้อ",
                    exit_plan_th="ขาย 50% ที่ TP1 แล้วเลื่อน SL มาที่ทุน ขายที่เหลือที่ TP2",
                    invalidation_th=f"ราคาหลุด Stop Loss {fmt_price(pl['sl'])}"))
            else:
                pl, fb = a["plan_long"], a["fwd_long"]["best"]
                picks.append(Pick(
                    symbol=b, horizon="LONG", conviction=conv(a["score_long"]),
                    action_th=("ซื้อ Tier 1 ตอนนี้ + ตั้ง Limit Tier 2 และ 3 รอ" if a["status_long"] == "BUY"
                               else "ตั้ง Limit ทั้ง 3 Tier รอที่แนวรับ ไม่ไล่ราคา"),
                    win_rate=int(round(a["wr_long"])), hold_min_days=int(a["hold_long"][0]),
                    hold_max_days=int(a["hold_long"][1]), allocation_pct=round(alloc[(b, hz)], 1),
                    thesis_th=f"คะแนนระยะยาว {a['score_long']:.0f}/100 (พื้นฐาน {a['score_fund']:.0f}, "
                              f"มูลค่า {a['s_value']:.0f})"
                              + (f" · ในอดีตสภาพเดียวกันถือ {fmt_days(fb['h'])} ชนะ {fb['win']:.0f}% "
                                 f"กลาง {fmt_pct(fb['median'])}" if fb else ""),
                    technical_th=tech, fundamental_th=fund_t, catalyst_th=cat,
                    entry_trigger_th="ทยอยซื้อตามราคา Tier 1 / 2 / 3",
                    exit_plan_th="ขาย 30% ที่ TP1, 40% ที่ TP2, ที่เหลือถือต่อ ขายเมื่อ Weekly ปิดใต้ EMA21 1W",
                    invalidation_th=f"แท่ง Weekly ปิดต่ำกว่า {fmt_price(pl['invalid'])}"))
    btc_1w = M_["btc_1w"]
    return Strategy(
        market_view_th=f"[แผนจากระบบ] ภาวะตลาด {reg} · BTC 1W {btc_1w} · Fear & Greed {fg}",
        regime_advice_th=(f"ถือเงินสด ~{cash:.0f}% · คัดจากคะแนนรวม + สถานะเทคนิค + สถิติย้อนหลัง "
                          "(เลือกเฉพาะ Win Rate ≥ 45% · ระยะกลางต้องซื้อได้/รอย่อ · ระยะยาวคะแนน ≥ 50)"),
        cash_reserve_pct=cash, picks=picks)


def run_ai_stage(client, model, prep: dict, n_med: int, n_long: int, use_risk: bool):
    A_, F_, RS_, M_ = prep["analyses"], prep["fund"], prep["research"], prep["macro"]
    notes, fallback = list(prep["notes"]), False
    st.write("🧠 Strategist Agent: คัดเหรียญและจัดพอร์ต")
    try:
        strat, _ = run_strategist(client, model, A_, F_, RS_, M_, n_med, n_long, prep["corr_txt"])
    except Exception as e:  # noqa: BLE001 — AI ล่ม/ไม่ว่าง → ใช้แผนจากระบบ ไม่ทิ้งงานทั้งหมด
        notes.append(f"Strategist (AI) ไม่ว่าง: {str(e)[:90]} → ใช้แผนจากระบบ")
        st.write("⚙️ Gemini ไม่ว่าง → สร้างแผนจากระบบแทน")
        strat, fallback = rule_strategy(A_, F_, RS_, M_, n_med, n_long), True
    picks, cash, dropped = finalize(strat, A_, F_)
    if dropped:
        notes.append("ตัดออกเพราะยังเป็นขาลง: " + ", ".join(dropped))
    risk_out = None
    if use_risk and picks and not fallback:
        st.write("🛡️ Risk Manager Agent: ตรวจความเสี่ยงพอร์ต")
        try:
            risk_out = run_risk(client, model, picks, M_, prep["corr_txt"])
            merge_risk(picks, risk_out)
        except Exception as e:  # noqa: BLE001
            notes.append(f"Risk Manager ไม่ตอบ: {str(e)[:80]}")
    st.session_state["result"] = {
        "time": datetime.now(timezone.utc), "analyses": A_, "info": prep["info"], "fund": F_, "research": RS_,
        "sources": prep["sources"], "modes": prep["modes"], "macro": M_, "strat": strat, "picks": picks,
        "cash": cash, "risk": risk_out, "corr": prep["corr"], "notes": notes, "fallback": fallback,
        "model": "แผนจากระบบ" if fallback else st.session_state.get("active_model", model),
        "secs": time.time() - prep["t0"],
    }


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("⚙️ ตั้งค่า")
    gemini_key = st.text_input("GEMINI_API_KEY", value=get_secret("GEMINI_API_KEY"), type="password")
    model_choice = st.selectbox("Gemini Model", GEMINI_MODELS)
    if model_choice.startswith("อื่นๆ"):
        model_choice = st.text_input("ชื่อโมเดล", value="gemini-3.8-flash")
    use_search = st.toggle("🔎 Research Agent ค้น Google (ข่าว/กำหนดการ)", value=True,
                           help="ใช้ Grounding with Google Search ของ Gemini — ข้อมูลสดกว่า")
    use_risk = st.toggle("🛡️ Risk Manager Agent", value=True)

    st.divider()
    st.subheader("🪙 เหรียญ")
    wl = parse_watchlist(st.text_area("Watchlist", value=get_secret("WATCHLIST") or DEFAULT_WATCHLIST, height=130))
    wl = wl[:MAX_WATCHLIST]
    n_disc = st.slider("🔍 Discovery (Volume สูงที่ไม่อยู่ในลิสต์)", 0, 5, 2)
    n_research = st.slider("จำนวนเหรียญที่ให้ Research Agent ค้นลึก", 4, 12, 8,
                           help="เลือกจากคะแนนสูงสุด · 4 เหรียญ = 1 การเรียก Gemini")
    c1, c2 = st.columns(2)
    n_med = c1.number_input("ระยะกลาง (เหรียญ)", 0, 6, 3)
    n_long = c2.number_input("ระยะยาว (เหรียญ)", 0, 6, 4)

    st.divider()
    st.subheader("💰 พอร์ต")
    capital = st.number_input("เงินทุน (USDT)", min_value=50.0, value=1000.0, step=100.0)
    max_risk = st.slider("ระยะกลาง: ยอมขาดทุนต่อไม้สูงสุด (% ของทุน)", 0.5, 3.0, 1.5, 0.25)
    with st.expander("CoinGecko API Key (ไม่บังคับ)"):
        cg_key = st.text_input("Demo API key", value=get_secret("COINGECKO_API_KEY"), type="password",
                               help="สมัครฟรีที่ coingecko.com/en/api — ช่วยลดปัญหาโดนจำกัดความถี่")
    scan = st.button("🚀 วิเคราะห์ตลาด", type="primary", width="stretch")
    rerun_ai = st.button("🧠 ให้ AI วิเคราะห์ใหม่ (ใช้ข้อมูลเดิม)", width="stretch",
                         disabled=not st.session_state.get("prep"),
                         help="ไม่ดึงราคา/ข้อมูลใหม่ เรียกแค่ Strategist + Risk Manager — ใช้ตอน Gemini ไม่ว่าง")
    st.caption("⚠️ ไม่ใช่คำแนะนำการลงทุน · สถิติในอดีตไม่รับประกันอนาคต")

st.title("🏦 Spot Investor AI")
st.caption("วิเคราะห์ลงทุน Spot ระยะกลาง & ยาว · เทคนิค 4H/1D/1W + พื้นฐานโปรเจกต์ + ข่าวและกำหนดการ + สถิติย้อนหลัง")

# =============================================================================
# PIPELINE
# =============================================================================
if scan:
    if not gemini_key:
        st.error("กรุณาใส่ GEMINI_API_KEY")
        st.stop()
    if len(wl) < 3:
        st.error("ใส่เหรียญใน Watchlist อย่างน้อย 3 เหรียญ")
        st.stop()
    try:
        client, key_mode = make_client(gemini_key, model_choice)
    except RuntimeError as e:
        st.error(str(e))
        st.info("วิธีแก้: เปิด aistudio.google.com/apikey → Create API key → เลือก/สร้าง Project ใหม่ "
                "→ คัดลอกคีย์ที่ขึ้นต้นด้วย AIza มาวางใน Sidebar หรือ Settings → Secrets ของแอปนี้")
        st.stop()
    if key_mode != "Gemini API":
        st.toast(f"เชื่อมต่อผ่าน {key_mode}")
    st.session_state["active_model"] = st.session_state.get("model_override", {}).get(model_choice, model_choice)
    t0, notes = time.time(), []
    with st.status("กำลังวิเคราะห์...", expanded=True) as status:
        try:
            st.write("📊 Market Data Agent: ดึงราคา 4H / 1D / 1W")
            ex, host = connect_binance()
            syms = get_symbols(ex, wl, n_disc)
            prog = st.progress(0.0)
            jobs = [(s, tf) for s in syms["symbol"] for tf in TIMEFRAMES]
            raw = {s: {} for s in syms["symbol"]}
            with cf.ThreadPoolExecutor(max_workers=8) as pool:
                futs = {pool.submit(fetch_klines, ex, s, tf): (s, tf) for s, tf in jobs}
                for i, fut in enumerate(cf.as_completed(futs), 1):
                    s, tf = futs[fut]
                    try:
                        raw[s][tf] = fut.result()
                    except Exception:  # noqa: BLE001
                        pass
                    prog.progress(i / len(jobs), text=f"{i}/{len(jobs)}")

            st.write("📐 Structure + Quant Agent: แนวรับ-ต้าน, Volume Profile, Backtest, Forward Returns")
            btc_d1 = raw.get("BTCUSDT", {}).get("1d")
            analyses, info = {}, {}
            for _, row in syms.iterrows():
                dfs = raw.get(row["symbol"], {})
                if "1d" not in dfs:
                    continue
                try:
                    analyses[row["base"]] = analyze_coin(dfs, btc_d1)
                    info[row["base"]] = row.to_dict()
                except Exception as e:  # noqa: BLE001
                    notes.append(f"{row['base']}: {str(e)[:60]}")
            if not analyses:
                raise RuntimeError("วิเคราะห์ไม่ได้สักเหรียญ")

            st.write("🏛️ Fundamental Agent: CoinGecko + DefiLlama")
            fund, fnotes = run_fundamentals(list(analyses), cg_key)
            notes += fnotes

            st.write("🌍 Macro Agent: BTC / Dominance / Fear & Greed")
            try:
                glob = cg_global(cg_key)
            except Exception:  # noqa: BLE001
                glob = None
            try:
                fng = fear_greed()
            except Exception:  # noqa: BLE001
                fng = None
            macro = macro_view(analyses, glob, fng)

            for b, a in analyses.items():  # คะแนนรอบแรก (ยังไม่มี research)
                compute_scores(a, fund.get(b), None)

            shortlist = sorted([b for b in analyses if b != "BTC" or "BTC" in wl],
                               key=lambda b: -max(analyses[b]["score_med"], analyses[b]["score_long"]))[:n_research]
            st.write(f"🔎 Research Agent: ค้นข้อมูลเชิงลึก {', '.join(shortlist)}")
            enrich_details(shortlist, fund, cg_key)
            rss = fetch_rss()
            today = datetime.now(timezone(timedelta(hours=7))).strftime("%Y-%m-%d")
            cache = st.session_state.setdefault("research_cache", {})
            bucket = int(time.time() // (6 * 3600))
            research, sources, modes = {}, [], set()
            todo = []
            for b in shortlist:
                key = (b, bucket, use_search)
                if key in cache:
                    research[b] = cache[key]
                else:
                    todo.append(b)
            for i in range(0, len(todo), 4):
                batch = todo[i:i + 4]
                payload = [{"symbol": b, "name": (fund.get(b) or {}).get("name") or b,
                            "cats": (fund.get(b) or {}).get("categories", []), "price": analyses[b]["price"],
                            "dd": analyses[b]["dd_ath"],
                            "heads": headlines_for(b, (fund.get(b) or {}).get("name") or "", rss)} for b in batch]
                try:
                    rb, src, mode = run_research(client, model_choice, payload, use_search, today)
                    sources += src
                    modes.add(mode)
                    for cr in rb.coins:
                        sb = re.sub(r"USDT$", "", cr.symbol.upper().strip())
                        if sb in analyses:
                            research[sb] = cr
                            cache[(sb, bucket, use_search)] = cr
                except Exception as e:  # noqa: BLE001
                    notes.append(f"Research {', '.join(batch)} ไม่สำเร็จ: {str(e)[:80]}")
            for b, a in analyses.items():  # คะแนนรอบสอง (รวม catalyst)
                compute_scores(a, fund.get(b), research.get(b))
            corr, corr_txt = corr_pairs(analyses)

            prep = {"analyses": analyses, "info": info, "fund": fund, "research": research, "sources": sources,
                    "modes": modes, "macro": macro, "corr": corr, "corr_txt": corr_txt, "notes": notes, "t0": t0}
            st.session_state["prep"] = prep  # เก็บข้อมูลไว้ ถ้า AI ไม่ว่างจะได้ลองใหม่โดยไม่ต้องดึงข้อมูลซ้ำ
            run_ai_stage(client, model_choice, prep, int(n_med), int(n_long), use_risk)
            status.update(label=f"วิเคราะห์เสร็จ ✅ ({time.time() - t0:.0f} วินาที)", state="complete", expanded=False)
        except Exception as e:  # noqa: BLE001
            status.update(label="เกิดข้อผิดพลาด ❌", state="error")
            st.exception(e)
            st.stop()

if rerun_ai:
    prep = st.session_state.get("prep")
    if not prep or not gemini_key:
        st.warning("ยังไม่มีข้อมูลเดิม — กด 🚀 วิเคราะห์ตลาด ก่อน")
    else:
        try:
            client, _ = make_client(gemini_key, model_choice)
            with st.status("🧠 ให้ AI วิเคราะห์ใหม่ด้วยข้อมูลเดิม...", expanded=True) as status:
                prep["t0"] = time.time()
                prep["notes"] = [n for n in prep["notes"] if "แผนจากระบบ" not in n and "Strategist" not in n]
                run_ai_stage(client, model_choice, prep, int(n_med), int(n_long), use_risk)
                status.update(label="เสร็จ ✅", state="complete", expanded=False)
        except Exception as e:  # noqa: BLE001
            st.error(str(e)[:300])

R = st.session_state.get("result")
if not R:
    st.info("ใส่ GEMINI_API_KEY แล้วกด **🚀 วิเคราะห์ตลาด**")
    st.markdown("""
**ทีมวิเคราะห์ 8 แผนก**
| แผนก | ทำอะไร |
|---|---|
| 📊 Market Data | ราคา 4H / 1D / 1W ย้อนหลังสูงสุด 1,000 แท่ง |
| 📐 Structure | โครงสร้าง HH/HL, Volume Profile (POC), แนวรับ-ต้าน, Fibonacci |
| 🧮 Quant | Backtest + ผลตอบแทนในอดีตเมื่อสภาพตลาดเหมือนวันนี้ → Win Rate และระยะถือ |
| 🏛️ Fundamental | Market Cap, FDV, Supply, ATH, TVL, ค่าธรรมเนียมจริง (P/F) |
| 🌍 Macro | เทรนด์ BTC, BTC Dominance, Fear & Greed |
| 🔎 Research (AI) | ค้น Google: โปรเจกต์, Tokenomics, ข่าว, กำหนดการ Unlock/Upgrade/ETF |
| 🧠 Strategist (AI) | คัดเหรียญ ระยะกลาง/ยาว จัดสัดส่วนพอร์ต |
| 🛡️ Risk Manager (AI) | ตรวจความเสี่ยง ความสัมพันธ์ จำกัดสัดส่วน |
""")
    st.stop()

# =============================================================================
# DASHBOARD
# =============================================================================
A, F, RS, M = R["analyses"], R["fund"], R["research"], R["macro"]
picks, strat = R["picks"], R["strat"]
bkk = R["time"].astimezone(timezone(timedelta(hours=7)))
st.caption(f"วิเคราะห์เมื่อ {bkk:%d/%m/%Y %H:%M} (เวลาไทย) · {R['model']} · {R['secs']:.0f} วินาที · "
           f"Research: {', '.join(R['modes']) or '-'}")
if R.get("fallback"):
    st.warning("⚠️ Gemini ไม่ว่างตอนนี้ — แสดง **แผนจากระบบ (คำนวณล้วน)** แทน ตัวเลขราคา/Win Rate/ระยะถือครบเหมือนเดิม "
               "แต่ไม่มีบทวิเคราะห์ของ AI · กด **🧠 ให้ AI วิเคราะห์ใหม่ (ใช้ข้อมูลเดิม)** ใน Sidebar อีกครั้งใน 5-10 นาที")
for n in R["notes"][:6]:
    st.caption(f"ℹ️ {n}")

# เงินต่อเหรียญ
for p in picks:
    amt = capital * p["allocation_pct"] / 100
    pl = p["plan"]
    if p["horizon"] == "MEDIUM":
        cap_by_risk = capital * max_risk / 100 / (abs(pl["sl_pct"]) / 100) if pl["sl_pct"] else amt
        p["amount"] = min(amt, cap_by_risk)
        p["capped"] = cap_by_risk < amt
        p["qty"] = p["amount"] / pl["entry"]
    else:
        p["amount"], p["capped"] = amt, False
        p["tiers_amt"] = [(t["price"], amt * t["weight"], amt * t["weight"] / t["price"]) for t in pl["tiers"]]
        p["qty"] = sum(x[2] for x in p["tiers_amt"])

g, fg = M["glob"], M["fng"]
m = st.columns(5)
m[0].metric("ภาวะตลาด (BTC 1W)", M["regime"], f"1D: {M['btc_1d']}", delta_color="off")
m[1].metric("Fear & Greed", f"{fg['value']} · {fg['label']}" if fg else "-",
            f"{fg['value'] - fg['week_ago']:+d} จาก 7 วันก่อน" if fg and fg["week_ago"] is not None else None)
m[2].metric("BTC Dominance", f"{g['btc_dom']:.1f}%" if g else "-",
            f"ETH {g['eth_dom']:.1f}%" if g else None, delta_color="off")
m[3].metric("มูลค่าตลาดรวม", fmt_big(g["total_mcap"]) if g else "-",
            f"{g['mcap_chg_24h']:+.1f}% 24h" if g and g["mcap_chg_24h"] is not None else None)
m[4].metric("เงินสดที่แนะนำ", f"{R['cash']:.0f}%", f"{capital * R['cash'] / 100:,.0f} USDT", delta_color="off")

with st.container(border=True):
    st.markdown(f"**🧠 มุมมองตลาด:** {strat.market_view_th}")
    st.markdown(f"**🧭 กลยุทธ์ตอนนี้:** {strat.regime_advice_th}")
    if R["risk"]:
        st.markdown(f"**🛡️ Risk:** {R['risk'].portfolio_note_th}  \n**⚠️ การกระจุกตัว:** {R['risk'].concentration_th}")

VERDICT = {"APPROVE": "✅", "CAUTION": "⚠️", "REJECT": "❌", "-": "•"}
tabs = st.tabs(["🏆 แผนการลงทุน", "🔬 วิเคราะห์รายเหรียญ", "📅 ปฏิทินเหตุการณ์", "🗺️ Screener", "📘 วิธีอ่าน"])

# ---------------- TAB 1: PLAN ----------------
with tabs[0]:
    if not picks:
        st.warning("ตอนนี้ AI ไม่แนะนำให้ซื้อเหรียญใด — ถือเงินสดรอ")
    else:
        left, right = st.columns([3, 2])
        with left:
            alloc = pd.DataFrame([{"name": f"{p['symbol']} ({'กลาง' if p['horizon'] == 'MEDIUM' else 'ยาว'})",
                                   "pct": p["allocation_pct"]} for p in picks] + [{"name": "เงินสด USDT", "pct": R["cash"]}])
            fig = px.pie(alloc, names="name", values="pct", hole=0.55, title="สัดส่วนพอร์ตที่แนะนำ")
            fig.update_traces(textinfo="label+percent")
            fig.update_layout(height=360, margin=dict(l=8, r=8, t=40, b=8), showlegend=False)
            st.plotly_chart(fig, width="stretch")
        with right:
            wdf = pd.DataFrame([{"เหรียญ": f"{p['symbol']} {'🌊' if p['horizon'] == 'MEDIUM' else '🏦'}",
                                 "Win Rate": p["win_rate"]} for p in picks]).sort_values("Win Rate")
            fig = px.bar(wdf, x="Win Rate", y="เหรียญ", orientation="h", text="Win Rate", range_x=[0, 100],
                         color="Win Rate", color_continuous_scale="RdYlGn", range_color=[30, 75], title="Win Rate")
            fig.add_vline(x=50, line_dash="dot")
            fig.update_layout(height=360, margin=dict(l=8, r=8, t=40, b=8), coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch")

        for hz in ("MEDIUM", "LONG"):
            group = sorted([p for p in picks if p["horizon"] == hz], key=lambda p: -p["win_rate"])
            st.subheader(HZ_NAME[hz])
            if not group:
                st.caption("ไม่มีเหรียญที่เหมาะในระยะนี้ตอนนี้")
                continue
            if hz == "MEDIUM":
                tb = pd.DataFrame([{
                    "": VERDICT.get(p["verdict"], "•"), "เหรียญ": p["symbol"], "สถานะ": p["status"],
                    "Win Rate": p["win_rate"], "ถือ": range_days(*p["hold"]),
                    "ราคาตอนนี้": fmt_price(A[p["symbol"]]["price"]),
                    "โซนซื้อ": f"{fmt_price(p['plan']['entry_low'])} – {fmt_price(p['plan']['entry_high'])}",
                    "Stop Loss": f"{fmt_price(p['plan']['sl'])} ({fmt_pct(p['plan']['sl_pct'])})",
                    "TP1": f"{fmt_price(p['plan']['tp1'])} ({fmt_pct(p['plan']['tp1_pct'])})",
                    "TP2": f"{fmt_price(p['plan']['tp2'])} ({fmt_pct(p['plan']['tp2_pct'])})",
                    "R:R": f"1:{p['plan']['rr1']:.1f}", "พอร์ต %": p["allocation_pct"],
                    "ซื้อ (USDT)": f"{p['amount']:,.0f}", "มั่นใจ": p["conviction"]} for p in group])
            else:
                tb = pd.DataFrame([{
                    "": VERDICT.get(p["verdict"], "•"), "เหรียญ": p["symbol"], "สถานะ": p["status"],
                    "Win Rate": p["win_rate"], "ถือ": range_days(*p["hold"]),
                    "ราคาตอนนี้": fmt_price(A[p["symbol"]]["price"]),
                    "DCA Tier 1/2/3": " / ".join(fmt_price(t["price"]) for t in p["plan"]["tiers"]),
                    "ต้นทุนเฉลี่ย": fmt_price(p["plan"]["avg"]),
                    "ตัดขาดทุน (W close)": f"{fmt_price(p['plan']['invalid'])} ({fmt_pct(p['plan']['invalid_pct'])})",
                    "TP1": f"{fmt_price(p['plan']['tp1'])} ({fmt_pct(p['plan']['tp1_pct'])})",
                    "TP2": f"{fmt_price(p['plan']['tp2'])} ({fmt_pct(p['plan']['tp2_pct'])})",
                    "พอร์ต %": p["allocation_pct"], "ลงทุน (USDT)": f"{p['amount']:,.0f}",
                    "มั่นใจ": p["conviction"]} for p in group])
            st.dataframe(tb, width="stretch", hide_index=True, column_config={
                "Win Rate": st.column_config.ProgressColumn("Win Rate", min_value=0, max_value=100, format="%d%%"),
                "พอร์ต %": st.column_config.NumberColumn("พอร์ต %", format="%.1f%%")})

            for p in group:
                a, pl, r = A[p["symbol"]], p["plan"], RS.get(p["symbol"])
                with st.expander(f"{VERDICT.get(p['verdict'], '•')} {p['symbol']} · {p['status']} · Win {p['win_rate']}% · "
                                 f"ถือ {range_days(*p['hold'])}", expanded=p is group[0]):
                    st.markdown(f"**▶️ ทำอะไรตอนนี้:** {p['action']}")
                    c = st.columns(6)
                    c[0].metric("Win Rate", f"{p['win_rate']}%", f"ระบบ {p['sys_wr']}%", delta_color="off")
                    c[1].metric("ระยะถือ", range_days(*p["hold"]).split(" (")[0])
                    if hz == "MEDIUM":
                        c[2].metric("Entry", fmt_price(pl["entry"]), f"ตอนนี้ {fmt_price(a['price'])}", delta_color="off")
                        c[3].metric("Stop Loss", fmt_price(pl["sl"]), fmt_pct(pl["sl_pct"]))
                        c[4].metric("TP1", fmt_price(pl["tp1"]), fmt_pct(pl["tp1_pct"]))
                        c[5].metric("TP2", fmt_price(pl["tp2"]), fmt_pct(pl["tp2_pct"]))
                        st.plotly_chart(chart_medium(a, pl, p["symbol"]), width="stretch", key=f"cm_{p['symbol']}")
                        st.markdown(
                            f"**🧾 วิธีตั้งคำสั่ง (Spot):** Limit Buy {fmt_price(pl['entry_low'])}–{fmt_price(pl['entry_high'])} "
                            f"≈ **{p['amount']:,.0f} USDT** ({p['qty']:,.4g} {p['symbol']}) → ตั้ง **Stop-Limit ขาย** ที่ "
                            f"{fmt_price(pl['sl'])} → ขาย 50% ที่ TP1 {fmt_price(pl['tp1'])} แล้วเลื่อน SL มาที่ทุน → "
                            f"ขายที่เหลือที่ TP2 {fmt_price(pl['tp2'])}")
                        if p["capped"]:
                            st.caption(f"ลดขนาดไม้ให้ขาดทุนสูงสุดไม่เกิน {max_risk}% ของทุนถ้าโดน SL")
                    else:
                        c[2].metric("ต้นทุนเฉลี่ย", fmt_price(pl["avg"]), f"ตอนนี้ {fmt_price(a['price'])}", delta_color="off")
                        c[3].metric("ตัดขาดทุน", fmt_price(pl["invalid"]), fmt_pct(pl["invalid_pct"]))
                        c[4].metric("TP1", fmt_price(pl["tp1"]), fmt_pct(pl["tp1_pct"]))
                        c[5].metric("TP2", fmt_price(pl["tp2"]), fmt_pct(pl["tp2_pct"]))
                        st.plotly_chart(chart_long(a, pl, p["symbol"]), width="stretch", key=f"cl_{p['symbol']}")
                        st.markdown("**🧾 แผนทยอยสะสม (DCA)**")
                        st.dataframe(pd.DataFrame([{
                            "Tier": f"Tier {i}", "ราคา": fmt_price(px_), "อ้างอิง": t["basis"],
                            "ห่างจากราคาตอนนี้": fmt_pct((px_ / a["price"] - 1) * 100),
                            "น้ำหนัก": f"{t['weight'] * 100:.0f}%", "เงิน (USDT)": f"{usd:,.0f}",
                            f"จำนวน {p['symbol']}": f"{q:,.4g}"}
                            for i, (t, (px_, usd, q)) in enumerate(zip(pl["tiers"], p["tiers_amt"]), 1)]),
                            width="stretch", hide_index=True)
                        st.caption(f"ตัดขาดทุนเมื่อแท่ง **Weekly ปิด** ต่ำกว่า {fmt_price(pl['invalid'])} · "
                                   f"TP2 อ้างอิง {pl['tp2_basis']} · ทยอยขาย 30% ที่ TP1, 40% ที่ TP2, ที่เหลือถือต่อแบบ Trailing")
                    l_, r_ = st.columns(2)
                    with l_:
                        st.markdown(f"**💡 เหตุผลหลัก:** {p['thesis']}")
                        st.markdown(f"**📈 เทคนิค:** {p['technical']}")
                        st.markdown(f"**🏛️ พื้นฐาน:** {p['fundamental']}")
                    with r_:
                        st.markdown(f"**📅 Catalyst:** {p['catalyst']}")
                        st.markdown(f"**🎯 เงื่อนไขซื้อ:** {p['trigger']}")
                        st.markdown(f"**💰 แผนขาย:** {p['exit']}")
                        st.markdown(f"**🚫 แผนผิดเมื่อ:** {p['invalidation']}")
                        if p["risk_note"]:
                            st.markdown(f"**🛡️ Risk Manager ({p['verdict']}):** {p['risk_note']}")
                    fwd = a["fwd_med"] if hz == "MEDIUM" else a["fwd_long"]
                    st.plotly_chart(chart_fwd(fwd, f"ในอดีต เมื่อ {p['symbol']} อยู่ในสภาพ '{fwd['state']}' แล้วถือต่อ…"),
                                    width="stretch", key=f"fw_{p['symbol']}_{hz}")

# ---------------- TAB 2: DEEP DIVE ----------------
with tabs[1]:
    sym = st.selectbox("เลือกเหรียญ", list(A.keys()),
                       index=list(A.keys()).index(picks[0]["symbol"]) if picks else 0)
    a, f, r = A[sym], F.get(sym) or {}, RS.get(sym)
    st.markdown(f"### {sym} · {f.get('name', '')}  ·  ${fmt_price(a['price'])}")
    c = st.columns(6)
    c[0].metric("ระยะกลาง", STATUS_MED[a["status_med"]], f"Win {a['wr_med']}%", delta_color="off")
    c[1].metric("ระยะยาว", STATUS_LONG[a["status_long"]], f"Win {a['wr_long']}%", delta_color="off")
    c[2].metric("Market Cap", fmt_big(f.get("mcap")), f"อันดับ #{f.get('rank')}" if f.get("rank") else None, delta_color="off")
    c[3].metric("FDV / MCap", f"{f.get('fdv_mcap', np.nan):.2f}x" if f else "-",
                f"หมุนเวียน {f.get('circ_ratio', np.nan) * 100:.0f}%" if f and np.isfinite(f.get("circ_ratio", np.nan)) else None,
                delta_color="off")
    c[4].metric("ห่าง ATH", fmt_pct(f.get("ath_chg", a["dd_ath"]) if f else a["dd_ath"]),
                f"ATH {fmt_price(f.get('ath'))} ({f.get('ath_date', '')})" if f else None, delta_color="off")
    c[5].metric("RS vs BTC 90d", fmt_pct(a["risk"]["rs90"]), f"Beta {a['risk']['beta']:.2f}" if np.isfinite(a["risk"]["beta"]) else None,
                delta_color="off")

    l_, r_ = st.columns([2, 1])
    with l_:
        st.plotly_chart(chart_medium(a, a["plan_med"], sym), width="stretch", key="dd_med")
    with r_:
        st.plotly_chart(chart_radar(a, sym), width="stretch", key="dd_radar")
        st.caption(f"คะแนนรวม · ระยะกลาง {a['score_med']:.0f} · ระยะยาว {a['score_long']:.0f}")
    st.plotly_chart(chart_long(a, a["plan_long"], sym), width="stretch", key="dd_long")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📐 เทคนิคหลาย TF")
        st.dataframe(pd.DataFrame([{
            "TF": TF_LABEL[tf], "Score": s["score"], "Trend": s["trend"], "RSI": s["rsi"], "ADX": s["adx"],
            "EMA21": fmt_price(s["ema21"]), "EMA50": fmt_price(s["ema50"]), "EMA200": fmt_price(s["ema200"]),
            "Volume×": s["vol_ratio"]} for tf, s in a["snap"].items() if s]), width="stretch", hide_index=True)
        st.markdown(f"โครงสร้าง 1D: **{a['struct_1d']}** · 1W: **{a['struct_1w']}** · "
                    f"OBV 30 วัน: **{'สะสม ↑' if a['obv_up'] else 'ระบาย ↓'}**")
        if a["vp"]:
            st.markdown(f"Volume Profile 180 วัน: POC **{fmt_price(a['vp']['poc'])}** · Value Area "
                        f"{fmt_price(a['vp']['val'])} – {fmt_price(a['vp']['vah'])}")
        st.markdown("#### 🧱 แนวรับ / แนวต้าน")
        lv = ([{"ประเภท": "🔴 ต้าน", "ราคา": fmt_price(x["price"]), "ห่าง": fmt_pct((x["price"] / a["price"] - 1) * 100),
                "ที่มา": x["kinds"], "ความแข็งแรง": x["strength"]} for x in reversed(a["ress"][:4])]
              + [{"ประเภท": "🟢 รับ", "ราคา": fmt_price(x["price"]), "ห่าง": fmt_pct((x["price"] / a["price"] - 1) * 100),
                  "ที่มา": x["kinds"], "ความแข็งแรง": x["strength"]} for x in a["sups"][:4]])
        st.dataframe(pd.DataFrame(lv), width="stretch", hide_index=True)
    with c2:
        st.markdown("#### 🧮 สถิติย้อนหลัง")
        bt = a["bt"]
        st.markdown(f"Backtest Setup ขาขึ้น 1D: **{bt['wr_raw']:.0f}%** ถึง TP1 ก่อน SL (n={bt['n']}, "
                    f"Expectancy {bt['exp_r']:+.2f}R)" if bt["n"] else "Backtest: ไม่มีสัญญาณในอดีตพอ")
        fr = pd.DataFrame(a["fwd_med"]["rows"] + a["fwd_long"]["rows"])
        fr = fr[fr["n"] > 0]
        if not fr.empty:
            st.dataframe(pd.DataFrame({
                "ถือ": fr["h"].map(fmt_days), "โอกาสกำไร": fr["win"].map(lambda x: f"{x:.0f}%"),
                "กลาง": fr["median"].map(fmt_pct), "แย่ (p25)": fr["p25"].map(fmt_pct), "ดี (p75)": fr["p75"].map(fmt_pct),
                "ตัวอย่าง": fr["n"], "เทียบจากสภาพ": fr["basis"]}), width="stretch", hide_index=True)
        rk = a["risk"]
        st.markdown(f"ความผันผวน/ปี **{rk['vol_ann']:.0f}%** · Drawdown สูงสุด 1 ปี **{rk['max_dd_1y']:.0f}%** · "
                    f"Corr BTC **{rk['corr_btc']:.2f}**" if np.isfinite(rk["corr_btc"]) else
                    f"ความผันผวน/ปี **{rk['vol_ann']:.0f}%** · Drawdown สูงสุด 1 ปี **{rk['max_dd_1y']:.0f}%**")
        if f:
            st.markdown("#### 🏛️ พื้นฐาน")
            st.markdown(
                f"- FDV {fmt_big(f.get('fdv'))} · Volume/MCap {f.get('vol_mcap', np.nan):.3f}\n"
                f"- TVL {fmt_big(f.get('tvl'))} · ค่าธรรมเนียม 30 วัน {fmt_big(f.get('fees_30d'))}"
                + (f" · **P/F {f['pf']:.0f}x**" if np.isfinite(f.get("pf", np.nan)) else "") + "\n"
                f"- เปลี่ยนแปลง 30 วัน {fmt_pct(f.get('chg_30d'))} · 1 ปี {fmt_pct(f.get('chg_1y'))}\n"
                + (f"- หมวด: {', '.join(f.get('categories', []))}\n" if f.get("categories") else "")
                + (f"- Commit 4 สัปดาห์: {f['commits_4w']}\n" if f.get("commits_4w") else ""))
    if r:
        st.markdown("#### 🔎 Research")
        st.markdown(f"**โปรเจกต์ ({r.sector}):** {r.project_th}  \n**จุดแข็ง:** {r.moat_th}  \n"
                    f"**Tokenomics:** {r.tokenomics_th}  \n**ข่าวล่าสุด:** {r.recent_news_th}")
        b1, b2 = st.columns(2)
        b1.markdown("**🐂 Bull case**\n" + "\n".join(f"- {x}" for x in r.bull_case_th))
        b2.markdown("**🐻 Bear case / ความเสี่ยง**\n" + "\n".join(f"- {x}" for x in r.bear_case_th))
        st.caption(f"Catalyst score {r.catalyst_score:+d} · Sentiment {r.news_sentiment}")
    else:
        st.caption("เหรียญนี้ไม่อยู่ในกลุ่มที่ Research Agent ค้นลึก (เพิ่มจำนวนใน Sidebar ได้)")

# ---------------- TAB 3: CALENDAR ----------------
with tabs[2]:
    now = datetime.now(timezone.utc)
    ev = []
    for b, r in RS.items():
        for e in r.events:
            dt = parse_event_date(e.date)
            ev.append({"dt": dt, "วันที่": e.date, "อีก (วัน)": (dt - now).days if dt else None, "เหรียญ": b,
                       "เหตุการณ์": e.title, "ประเภท": e.type,
                       "ผลกระทบ": {"POSITIVE": "🟢 บวก", "NEGATIVE": "🔴 ลบ"}.get(e.impact.upper(), "⚪ กลาง"),
                       "ความแน่นอน": {"CONFIRMED": "ยืนยันแล้ว", "EXPECTED": "คาดการณ์", "RUMOR": "ข่าวลือ"}.get(
                           e.certainty.upper(), e.certainty), "แหล่งที่มา": e.source_url})
    if not ev:
        st.info("ยังไม่มีข้อมูลกำหนดการ (เปิด Research Agent ค้น Google แล้ววิเคราะห์ใหม่)")
    else:
        edf = pd.DataFrame(ev)
        edf["อีก (วัน)"] = pd.to_numeric(edf["อีก (วัน)"], errors="coerce")
        edf = edf[(edf["อีก (วัน)"].isna()) | (edf["อีก (วัน)"] >= -30)].sort_values("dt", na_position="last")
        plot_df = edf.dropna(subset=["dt"])
        if not plot_df.empty:
            fig = px.scatter(plot_df, x="dt", y="เหรียญ", color="ผลกระทบ", hover_data=["เหตุการณ์", "ประเภท", "ความแน่นอน"],
                             color_discrete_map={"🟢 บวก": "#26a69a", "🔴 ลบ": "#ef5350", "⚪ กลาง": "#9e9e9e"},
                             title="Timeline เหตุการณ์สำคัญ")
            fig.update_traces(marker=dict(size=14, symbol="diamond"))
            fig.add_vline(x=now, line_dash="dot")
            fig.update_layout(height=max(300, 34 * plot_df["เหรียญ"].nunique()), margin=dict(l=8, r=8, t=40, b=8))
            st.plotly_chart(fig, width="stretch")
        st.dataframe(edf.drop(columns=["dt"]), width="stretch", hide_index=True,
                     column_config={"แหล่งที่มา": st.column_config.LinkColumn("แหล่งที่มา")})
        st.caption("ตรวจสอบวันที่จากแหล่งที่มาก่อนตัดสินใจเสมอ — 'คาดการณ์' อาจเลื่อนได้")
    if R["sources"]:
        with st.expander(f"แหล่งอ้างอิงที่ Research Agent ค้นเจอ ({len(R['sources'])})"):
            seen = set()
            for s in R["sources"]:
                if s["url"] not in seen:
                    seen.add(s["url"])
                    st.markdown(f"- [{s['title']}]({s['url']})")

# ---------------- TAB 4: SCREENER ----------------
with tabs[3]:
    rows = []
    for b, a in A.items():
        f = F.get(b) or {}
        rows.append({
            "เหรียญ": b, "กลุ่ม": R["info"].get(b, {}).get("group", ""), "ราคา": fmt_price(a["price"]),
            "ระยะกลาง": STATUS_MED[a["status_med"]], "คะแนนกลาง": round(a["score_med"]), "Win กลาง": a["wr_med"],
            "ถือ (กลาง)": range_days(*a["hold_med"]).split(" (")[0],
            "ระยะยาว": STATUS_LONG[a["status_long"]], "คะแนนยาว": round(a["score_long"]), "Win ยาว": a["wr_long"],
            "ถือ (ยาว)": range_days(*a["hold_long"]).split(" (")[0],
            "พื้นฐาน": round(a["score_fund"]), "Catalyst": round(a["s_cat"]),
            "RS90 vs BTC": round(safe_float(a["risk"]["rs90"], 0), 1), "ห่าง ATH": round(a["dd_ath"], 0),
            "MCap": fmt_big(f.get("mcap")), "FDV/MC": round(f.get("fdv_mcap", np.nan), 2) if f else None,
            "Vol/ปี %": round(a["risk"]["vol_ann"]),
        })
    sdf = pd.DataFrame(rows).sort_values("คะแนนยาว", ascending=False)
    st.dataframe(sdf, width="stretch", hide_index=True, column_config={
        "คะแนนกลาง": st.column_config.ProgressColumn("คะแนนกลาง", min_value=0, max_value=100, format="%d"),
        "คะแนนยาว": st.column_config.ProgressColumn("คะแนนยาว", min_value=0, max_value=100, format="%d"),
        "Win กลาง": st.column_config.NumberColumn(format="%.0f%%"),
        "Win ยาว": st.column_config.NumberColumn(format="%.0f%%"),
        "RS90 vs BTC": st.column_config.NumberColumn(format="%+.1f%%"),
        "ห่าง ATH": st.column_config.NumberColumn(format="%.0f%%")})
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(sdf, x="คะแนนกลาง", y="คะแนนยาว", text="เหรียญ", color="พื้นฐาน",
                         color_continuous_scale="RdYlGn", title="แผนที่โอกาส (ขวาบน = ดีทั้ง 2 ระยะ)")
        fig.add_hline(y=50, line_dash="dot")
        fig.add_vline(x=50, line_dash="dot")
        fig.update_traces(textposition="top center")
        fig.update_layout(height=460, margin=dict(l=8, r=8, t=40, b=8))
        st.plotly_chart(fig, width="stretch")
    with c2:
        corr = R["corr"]
        if corr is not None and not corr.empty:
            fig = px.imshow(corr.round(2), color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                            title="ความสัมพันธ์ผลตอบแทน 90 วัน")
            fig.update_layout(height=460, margin=dict(l=8, r=8, t=40, b=8))
            st.plotly_chart(fig, width="stretch")

# ---------------- TAB 5: METHOD ----------------
with tabs[4]:
    st.markdown(f"""
### ระยะเวลาการถือ
| ระยะ | TF หลัก | ระยะถือ | วิธีซื้อ | วิธีออก |
|---|---|---|---|---|
| 🌊 ระยะกลาง (Swing) | 1D (จับจังหวะด้วย 4H) | {MED_HOLD_LIMIT[0]}–{MED_HOLD_LIMIT[1]} วัน (ปกติ 1–8 สัปดาห์) | Limit ในโซนซื้อ | SL / TP1 / TP2 |
| 🏦 ระยะยาว (สะสม) | 1W | 2–24 เดือน | DCA 3 Tier | ตัดขาดทุนเมื่อ Weekly ปิดต่ำกว่าจุดที่กำหนด / ทยอยขายที่ TP |

ระยะถือแต่ละเหรียญคำนวณจากสถิติจริงของเหรียญนั้น:
- **ระยะกลาง:** ใช้เวลาที่ setup แบบเดียวกันในอดีตใช้จริงจนถึงเป้า (ช่วง p25–p75)
- **ระยะยาว:** เลือกระยะ 3 / 6 / 12 เดือนที่ให้ผลตอบแทนเทียบความเสี่ยงดีที่สุด เมื่อเหรียญอยู่ในสภาพตลาดเดียวกับวันนี้

### Win Rate คืออะไร
- **ระยะกลาง:** โอกาสที่ราคาไปถึง TP1 ก่อนโดน Stop Loss คำนวณจาก Backtest 1D (SL 2 ATR, TP 2R) แล้วปรับตามความสอดคล้องของ 4H/1W
- **ระยะยาว:** โอกาสที่ถือครบระยะที่แนะนำแล้วมีกำไร จาก Conditional Forward Returns (ปรับเข้าหา 50% เมื่อตัวอย่างน้อย)
- AI ปรับได้ไม่เกิน ±10% จากค่าของระบบ (เช่น ตามข่าวหรือกำหนดการ) และ Risk Manager ปรับได้อีกไม่เกิน ±10%

### เครื่องมือวิเคราะห์
- **Market Structure:** HH/HL = ขาขึ้น, LH/LL = ขาลง จาก Swing Pivot
- **Volume Profile (180 วัน):** POC = ราคาที่ซื้อขายมากที่สุด (แนวรับ/ต้านแข็ง), Value Area = โซน 70% ของ Volume
- **แนวรับ-ต้าน:** รวม Swing 1D, Swing 1W, POC/HVN แล้วจัดกลุ่ม ยิ่งหลายแหล่งตรงกันยิ่งแข็ง
- **DCA Tier:** เลือกจากแนวรับจริง, Fibonacci 0.382–0.786 ของกรอบ 52 สัปดาห์ และ EMA 1W
- **พื้นฐาน:**
  - FDV/MCap สูง = มีเหรียญรอปลดล็อกมาก (แรงขายในอนาคต)
  - P/F = Market Cap ÷ ค่าธรรมเนียมรายปี (ยิ่งต่ำยิ่งถูกเทียบรายได้จริง)
  - Volume/MCap = สภาพคล่อง
- **RS vs BTC:** ผลตอบแทนส่วนเกินเทียบ BTC — ใน Spot ควรถือเหรียญที่แข็งกว่า BTC
- **Beta:** ถ้า BTC ขยับ 1% เหรียญนี้ขยับกี่ % (Beta สูง = ผันผวนกว่า)

### วิธีใช้กับ Spot
1. ระยะกลาง: ตั้ง Limit Buy ในโซน แล้วตั้ง **Stop-Limit ขาย** ที่ SL ทันที (Spot ไม่มี SL อัตโนมัติ)
2. ขาย 50% ที่ TP1 แล้วเลื่อน SL มาที่ทุน ขายที่เหลือที่ TP2
3. ระยะยาว: ตั้ง Limit ทั้ง 3 Tier ล่วงหน้า ตรวจแท่ง Weekly ทุกวันอาทิตย์ ถ้าปิดต่ำกว่าจุดตัดขาดทุนให้ขาย
4. เช็กปฏิทินเหตุการณ์: Token Unlock ใหญ่มักกดราคาก่อนและหลังวันปลดล็อก

⚠️ ข้อจำกัด:
- สถิติในอดีตไม่รับประกันอนาคต
- เหรียญใหม่มีข้อมูลน้อย ความแม่นยำจึงต่ำกว่า
- ข้อมูลกำหนดการจาก AI ต้องตรวจสอบกับแหล่งที่มาก่อนใช้เสมอ
""")
