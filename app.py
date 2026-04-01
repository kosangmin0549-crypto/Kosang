from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import math
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

try:
    import FinanceDataReader as fdr
except Exception as e:
    raise RuntimeError("FinanceDataReader is required. Install with pip install finance-datareader") from e

app = FastAPI(title="KRX Live Screener API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UNIVERSE_CACHE: dict[str, Any] = {"ts": None, "data": None}


def normalize_universe(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    code_col = cols.get("symbol") or cols.get("code")
    name_col = cols.get("name")
    market_col = cols.get("market")
    if not code_col or not name_col:
        raise RuntimeError(f"Unexpected listing columns: {list(df.columns)}")
    out = df[[code_col, name_col] + ([market_col] if market_col else [])].copy()
    out.columns = ["ticker", "name"] + (["market"] if market_col else [])
    if "market" not in out.columns:
        out["market"] = ""
    out["ticker"] = out["ticker"].astype(str).str.zfill(6)
    if out["market"].dtype != object:
        out["market"] = out["market"].astype(str)
    out = out[out["market"].isin(["KOSPI", "KOSDAQ"]) | out["market"].eq("")]
    return out.drop_duplicates(subset=["ticker"]).reset_index(drop=True)


def get_universe(force: bool = False) -> pd.DataFrame:
    now = datetime.utcnow()
    if not force and UNIVERSE_CACHE["data"] is not None and UNIVERSE_CACHE["ts"] is not None:
        if (now - UNIVERSE_CACHE["ts"]).seconds < 3600:
            return UNIVERSE_CACHE["data"]
    df = fdr.StockListing("KRX")
    uni = normalize_universe(df)
    UNIVERSE_CACHE["data"] = uni
    UNIVERSE_CACHE["ts"] = now
    return uni


def get_history(ticker: str, days: int = 365) -> pd.DataFrame:
    start = (date.today() - timedelta(days=days + 30)).isoformat()
    end = date.today().isoformat()
    df = fdr.DataReader(ticker, start, end)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    # normalize columns
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "date":
            rename_map[c] = "date"
        elif cl == "open":
            rename_map[c] = "open"
        elif cl == "high":
            rename_map[c] = "high"
        elif cl == "low":
            rename_map[c] = "low"
        elif cl == "close":
            rename_map[c] = "close"
        elif cl == "volume":
            rename_map[c] = "volume"
        elif cl in ("change", "changerate", "changes"):
            rename_map[c] = "change"
    df = df.rename(columns=rename_map)
    need = ["date", "open", "high", "low", "close", "volume"]
    for n in need:
        if n not in df.columns:
            raise RuntimeError(f"Missing expected column '{n}' for {ticker}. Columns: {list(df.columns)}")
    df["ticker"] = str(ticker).zfill(6)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    num_cols = ["open", "high", "low", "close", "volume"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["value"] = ((df["open"] + df["close"]) / 2.0) * df["volume"]
    df["change_rate"] = df["close"].pct_change() * 100.0
    return df[["date", "ticker", "open", "high", "low", "close", "volume", "value", "change_rate"]].dropna(subset=["close"]).reset_index(drop=True)


def sma(s: pd.Series, p: int) -> pd.Series:
    return s.rolling(p).mean()


def ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("date").reset_index(drop=True)
    out["ma5"] = sma(out["close"], 5)
    out["ma20"] = sma(out["close"], 20)
    out["ma60"] = sma(out["close"], 60)
    out["vol_ma20"] = sma(out["volume"], 20)
    out["ema12"] = ema(out["close"], 12)
    out["ema26"] = ema(out["close"], 26)
    out["macd"] = out["ema12"] - out["ema26"]
    out["signal"] = ema(out["macd"], 9)
    out["rsi14"] = rsi(out["close"], 14)
    out["bb_mid"] = sma(out["close"], 20)
    out["bb_std"] = out["close"].rolling(20).std(ddof=0)
    out["bb_upper"] = out["bb_mid"] + 2 * out["bb_std"]
    out["bb_lower"] = out["bb_mid"] - 2 * out["bb_std"]
    out["rolling_high20"] = out["high"].rolling(20).max()
    out["rolling_low20"] = out["low"].rolling(20).min()
    rng = (out["high"] - out["low"]).replace(0, np.nan)
    out["upper_wick"] = (out["high"] - out[["open", "close"]].max(axis=1)) / rng
    out["lower_wick"] = (out[["open", "close"]].min(axis=1) - out["low"]) / rng
    out["range_pct"] = (out["high"] - out["low"]) / out["close"] * 100.0
    return out


def analyze_history(df: pd.DataFrame, name: str = "", market: str = "") -> dict[str, Any]:
    if len(df) < 70:
        raise ValueError("Need at least ~70 trading days for stable analysis")
    e = enrich(df)
    last = e.iloc[-1]
    prev = e.iloc[-2]

    bull_align = pd.notna(last["ma5"]) and pd.notna(last["ma20"]) and pd.notna(last["ma60"]) and last["ma5"] > last["ma20"] > last["ma60"]
    golden = pd.notna(prev["ma5"]) and pd.notna(prev["ma20"]) and pd.notna(last["ma5"]) and pd.notna(last["ma20"]) and prev["ma5"] <= prev["ma20"] and last["ma5"] > last["ma20"]
    macd_cross = pd.notna(prev["macd"]) and pd.notna(prev["signal"]) and pd.notna(last["macd"]) and pd.notna(last["signal"]) and prev["macd"] <= prev["signal"] and last["macd"] > last["signal"]
    macd_up = pd.notna(last["macd"]) and pd.notna(last["signal"]) and last["macd"] > last["signal"]
    rsi_recover = pd.notna(prev["rsi14"]) and pd.notna(last["rsi14"]) and prev["rsi14"] < 40 and last["rsi14"] > 45
    rsi_sweet = pd.notna(last["rsi14"]) and 45 <= last["rsi14"] <= 68
    vol_surge = pd.notna(last["vol_ma20"]) and last["volume"] > last["vol_ma20"] * 1.8
    vol_confirm = pd.notna(last["vol_ma20"]) and last["volume"] > last["vol_ma20"] * 1.2
    bullish_engulf = (
        prev["close"] < prev["open"]
        and last["close"] > last["open"]
        and last["close"] >= prev["open"]
        and last["open"] <= prev["close"]
    )
    lower_wick_reversal = pd.notna(last["lower_wick"]) and last["lower_wick"] >= 0.45 and last["close"] > last["open"]
    breakout20 = pd.notna(last["rolling_high20"]) and last["close"] >= last["rolling_high20"] * 0.995
    close_above20 = pd.notna(last["ma20"]) and last["close"] > last["ma20"]
    close_above60 = pd.notna(last["ma60"]) and last["close"] > last["ma60"]
    liq = float(last["value"]) if pd.notna(last["value"]) else float(last["close"] * last["volume"])

    reversal = 0
    reversal += 18 if golden else 0
    reversal += 16 if macd_cross else 0
    reversal += 14 if rsi_recover else 0
    reversal += 12 if bullish_engulf else 0
    reversal += 10 if lower_wick_reversal else 0
    reversal += 8 if close_above20 else 0
    reversal += 8 if close_above60 else 0
    reversal += 10 if vol_surge else 0
    reversal += 8 if breakout20 else 0
    reversal += 10 if bull_align else 0

    prob = 0
    prob += 20 if bull_align else 0
    prob += 10 if golden else 0
    prob += 10 if macd_up else 0
    prob += 8 if macd_cross else 0
    prob += 12 if rsi_sweet else 0
    prob += 8 if vol_confirm else 0
    prob += 6 if vol_surge else 0
    prob += 12 if breakout20 else 0
    prob += 6 if close_above20 else 0
    prob += 6 if close_above60 else 0
    prob += 8 if liq > 30_000_000_000 else 4 if liq > 10_000_000_000 else 0
    prob -= 6 if pd.notna(last["change_rate"]) and last["change_rate"] > 12 else 0
    prob -= 8 if pd.notna(last["rsi14"]) and last["rsi14"] > 75 else 0
    est_prob = max(5, min(85, 25 + prob * 0.55))

    suspicion = 0
    suspicion += 18 if pd.notna(last["upper_wick"]) and last["upper_wick"] >= 0.45 else 0
    suspicion += 14 if pd.notna(last["range_pct"]) and last["range_pct"] > 9 else 0
    suspicion += 14 if vol_surge else 0
    suspicion += 10 if liq > 50_000_000_000 else 0
    suspicion += 12 if pd.notna(last["change_rate"]) and abs(last["change_rate"]) > 12 else 0

    patterns = []
    if golden: patterns.append("MA 골든크로스")
    if macd_cross: patterns.append("MACD 골든크로스")
    if rsi_recover: patterns.append("RSI 회복")
    if bullish_engulf: patterns.append("Bullish Engulfing")
    if lower_wick_reversal: patterns.append("아랫꼬리 반전")
    if vol_surge: patterns.append("거래량 급증")
    if breakout20: patterns.append("20일 돌파권")
    if bull_align: patterns.append("정배열")

    trend = "상승 추세" if bull_align else "상승 전환 시도" if close_above20 else "중립/약세"
    final_score = prob + reversal * 0.6 - suspicion * 0.5

    latest = {
        "date": str(last["date"]),
        "close": float(last["close"]),
        "open": float(last["open"]),
        "high": float(last["high"]),
        "low": float(last["low"]),
        "volume": float(last["volume"]),
        "value": liq,
        "change_rate": None if pd.isna(last["change_rate"]) else float(last["change_rate"]),
        "rsi14": None if pd.isna(last["rsi14"]) else float(last["rsi14"]),
        "macd": None if pd.isna(last["macd"]) else float(last["macd"]),
        "signal": None if pd.isna(last["signal"]) else float(last["signal"]),
        "ma5": None if pd.isna(last["ma5"]) else float(last["ma5"]),
        "ma20": None if pd.isna(last["ma20"]) else float(last["ma20"]),
        "ma60": None if pd.isna(last["ma60"]) else float(last["ma60"]),
        "upper_wick": None if pd.isna(last["upper_wick"]) else float(last["upper_wick"]),
        "range_pct": None if pd.isna(last["range_pct"]) else float(last["range_pct"]),
    }

    return {
        "ticker": str(df["ticker"].iloc[-1]).zfill(6),
        "name": name,
        "market": market,
        "latest": latest,
        "reversal_score": int(reversal),
        "prob_score": int(prob),
        "est_probability": round(est_prob, 1),
        "suspicion_score": int(suspicion),
        "patterns": patterns,
        "trend": trend,
        "summary": f"{trend} · 상승전환 {int(reversal)}점 · 상승확률 {round(est_prob,1)}%",
        "final_score": round(final_score, 2),
        "history_rows": len(df),
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "date": date.today().isoformat()}


@app.get("/api/universe")
def api_universe(limit: int = Query(200, ge=1, le=3000), market: str | None = None) -> dict[str, Any]:
    uni = get_universe()
    if market:
        uni = uni[uni["market"].str.upper() == market.upper()]
    rows = uni.head(limit).to_dict(orient="records")
    return {"count": len(rows), "items": rows}


@app.get("/api/history/{ticker}")
def api_history(ticker: str, days: int = Query(365, ge=90, le=1500)) -> dict[str, Any]:
    df = get_history(ticker, days=days)
    if df.empty:
        raise HTTPException(status_code=404, detail="No history found")
    return {"ticker": str(ticker).zfill(6), "rows": len(df), "items": df.to_dict(orient="records")}


@app.get("/api/analyze/{ticker}")
def api_analyze(ticker: str, days: int = Query(365, ge=120, le=1500)) -> dict[str, Any]:
    uni = get_universe()
    row = uni[uni["ticker"] == str(ticker).zfill(6)]
    name = row["name"].iloc[0] if not row.empty else ""
    market = row["market"].iloc[0] if not row.empty else ""
    df = get_history(ticker, days=days)
    if df.empty:
        raise HTTPException(status_code=404, detail="No history found")
    return analyze_history(df, name=name, market=market)


@app.get("/api/screen")
def api_screen(
    limit: int = Query(120, ge=10, le=400),
    days: int = Query(365, ge=120, le=1500),
    market: str | None = None,
) -> dict[str, Any]:
    uni = get_universe()
    if market:
        uni = uni[uni["market"].str.upper() == market.upper()]
    uni = uni.head(limit)
    analyses: list[dict[str, Any]] = []
    errors: list[str] = []
    for _, r in uni.iterrows():
        ticker = str(r["ticker"]).zfill(6)
        try:
            df = get_history(ticker, days=days)
            if len(df) < 70:
                continue
            analyses.append(analyze_history(df, name=str(r["name"]), market=str(r["market"])))
        except Exception as e:
            errors.append(f"{ticker}: {e}")
            continue

    reversal = sorted([a for a in analyses if a["reversal_score"] >= 25], key=lambda x: x["reversal_score"], reverse=True)[:15]
    probability = sorted(analyses, key=lambda x: x["prob_score"], reverse=True)[:15]
    suspicion = sorted([a for a in analyses if a["suspicion_score"] >= 18], key=lambda x: x["suspicion_score"], reverse=True)[:15]
    candidates = sorted(
        [a for a in analyses if a["prob_score"] >= 45 and a["reversal_score"] >= 20 and a["suspicion_score"] < 32 and a["latest"]["value"] >= 10_000_000_000],
        key=lambda x: x["final_score"],
        reverse=True,
    )
    final_pick = candidates[0] if candidates else (probability[0] if probability else None)

    return {
        "universe_checked": int(len(uni)),
        "analysed": int(len(analyses)),
        "errors": errors[:30],
        "latest_date": max((a["latest"]["date"] for a in analyses), default=None),
        "reversal": reversal,
        "probability": probability,
        "suspicion": suspicion,
        "final_pick": final_pick,
    }
