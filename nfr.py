"""Nifty Factor Rotation Dashboard — self-fetching stock-level version.

What this app does
- Pulls the current Nifty 200 universe from NSE automatically.
- Downloads daily price history from Yahoo Finance via yfinance.
- Pulls current public fundamentals from yfinance where available.
- Builds five stock-level factor sleeves:
    Alpha, Momentum, Quality, Value, Low Volatility
- Rotates capital across the five sleeves monthly.
- Produces exact share targets, rebalance logs, and a full audit trail.
- Shows dashboard charts and tables in Streamlit.

Important note
- Free public sources do not provide a perfect historical point-in-time record
  of every constituent/fundamental update. This app is designed as a practical
  research and deployment framework using live/public data, and it rebuilds
  factor sleeves from the current universe.

Run
    streamlit run nifty_factor_rotation_auto.py

Required packages
    streamlit, yfinance, pandas, numpy, requests, beautifulsoup4, plotly, scipy
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import warnings
from dataclasses import dataclass
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None  # type: ignore

from bs4 import BeautifulSoup
from scipy.stats import zscore

warnings.filterwarnings("ignore")

CACHE_DIR = Path.home() / ".cache" / "nifty_factor_rotation"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FACTOR_NAMES = ["Alpha", "Momentum", "Quality", "Value", "LowVol"]
FACTOR_DISPLAY = {
    "Alpha": "Nifty200 Alpha 30",
    "Momentum": "Nifty200 Momentum 30",
    "Quality": "Nifty200 Quality 30",
    "Value": "Nifty200 Value 30",
    "LowVol": "Nifty500 Low Volatility 50",
}

@dataclass
class Config:
    initial_capital: float = 100000.0
    start_date: str = "2018-01-01"
    end_date: Optional[str] = None
    top_n_alpha: int = 30
    top_n_momentum: int = 30
    top_n_quality: int = 30
    top_n_value: int = 30
    top_n_lowvol: int = 50
    max_universe: int = 220
    min_history_days: int = 252 * 2
    min_weight: float = 0.00
    max_weight: float = 0.10
    factor_min_weight: float = 0.10
    factor_max_weight: float = 0.40
    risk_free_annual: float = 0.06
    score_w12: float = 0.40
    score_w6: float = 0.30
    score_wsharpe: float = 0.30
    use_equal_weight_benchmark: bool = True


# -----------------------------
# Helpers
# -----------------------------

def _norm_col(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower().replace("_", " "))

def safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, str) and not x.strip():
            return default
        v = float(x)
        if math.isfinite(v):
            return v
        return default
    except Exception:
        return default

def sanitize_nav_series(nav: pd.Series, *, forward_fill: bool = True) -> pd.Series:
    """Clean a NAV series before return / drawdown calculations.

    Non-positive values are treated as data errors because a portfolio NAV should
    never be zero or negative in this framework. Keeping them would create fake
    100% drawdowns.
    """
    s = pd.Series(nav).copy()
    if s.empty:
        return s.astype(float)

    try:
        s.index = pd.to_datetime(s.index)
    except Exception:
        pass

    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="last")]

    # Negative/zero values are invalid for a NAV series. Treat as missing and
    # carry forward the last valid value instead of crashing the curve to zero.
    s = s.where(s > 0)
    if forward_fill:
        s = s.ffill()
    return s.dropna()

def softmax(x: pd.Series, temperature: float = 1.0) -> pd.Series:
    s = pd.Series(x, index=x.index, dtype=float)
    arr = s.values / max(temperature, 1e-9)
    arr = arr - np.nanmax(arr)
    ex = np.exp(np.clip(arr, -50, 50))
    out = ex / np.nansum(ex)
    return pd.Series(out, index=s.index)

def zscore_series(s: pd.Series) -> pd.Series:
    s = pd.Series(s, dtype=float)
    if s.dropna().nunique() <= 1:
        return pd.Series(0.0, index=s.index)
    z = (s - s.mean()) / (s.std(ddof=0) + 1e-12)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def annualized_sharpe(daily_returns: pd.Series, rf_annual: float) -> float:
    daily_returns = pd.Series(daily_returns).dropna().astype(float)
    if len(daily_returns) < 20:
        return np.nan
    mean_daily = daily_returns.mean()
    vol_daily = daily_returns.std(ddof=1)
    if vol_daily <= 0:
        return np.nan
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    return ((mean_daily - rf_daily) / vol_daily) * math.sqrt(252)

def max_drawdown(returns: pd.Series) -> float:
    s = (1 + pd.Series(returns).fillna(0.0)).cumprod()
    dd = s / s.cummax() - 1.0
    return float(dd.min())

def max_drawdown_nav(nav: pd.Series) -> float:
    """Compute max drawdown directly from a NAV/price series.

    A NAV series must stay positive. Any zero/negative/inf values are treated as
    data glitches and removed before calculating drawdown.
    """
    s = sanitize_nav_series(nav, forward_fill=True)
    if len(s) < 2:
        return 0.0
    dd = s / s.cummax() - 1.0
    return float(dd.min())

# -----------------------------
# NSE / Yahoo data fetch
# -----------------------------

class DataSourceError(RuntimeError):
    pass

class NSEDataSource:
    BASE = "https://www.nseindia.com"
    INDEX_PAGE = "https://www.nseindia.com/static/products-services/indices-nifty200-index"
    API_INDEX = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20200"

    @staticmethod
    def session() -> requests.Session:
        s = requests.Session()
        s.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.8",
            "Connection": "keep-alive",
            "Referer": NSEDataSource.BASE + "/",
        })
        return s

    @classmethod
    def fetch_nifty200_universe(cls) -> pd.DataFrame:
        cache_path = CACHE_DIR / "nifty200_universe.csv"
        if cache_path.exists() and time.time() - cache_path.stat().st_mtime < 7 * 24 * 3600:
            return pd.read_csv(cache_path)

        s = cls.session()
        # warm cookies
        try:
            s.get(cls.BASE, timeout=20)
        except Exception:
            pass

        # Try NSE JSON API first.
        for url in [cls.API_INDEX, cls.INDEX_PAGE]:
            try:
                r = s.get(url, timeout=30)
                if r.status_code == 200:
                    txt = r.text
                    # API JSON
                    if "data" in r.headers.get("content-type", "").lower() or txt.lstrip().startswith("{"):
                        try:
                            j = r.json()
                            rows = j.get("data", []) or j.get("metadata", {}).get("data", [])
                            if rows:
                                df = pd.DataFrame(rows)
                                sym_col = None
                                for c in df.columns:
                                    if _norm_col(c) in {"symbol", "underlying", "security symbol"}:
                                        sym_col = c
                                        break
                                if sym_col is None and "symbol" in df.columns:
                                    sym_col = "symbol"
                                if sym_col is not None:
                                    out = df[[sym_col]].rename(columns={sym_col: "Symbol"}).dropna().drop_duplicates()
                                    out["Symbol"] = out["Symbol"].astype(str).str.upper().str.strip()
                                    out = out[out["Symbol"].str.len() > 0]
                                    out["Ticker"] = out["Symbol"] + ".NS"
                                    out.to_csv(cache_path, index=False)
                                    return out.reset_index(drop=True)
                        except Exception:
                            pass

                    # HTML with csv link or constituents table
                    soup = BeautifulSoup(txt, "html.parser")
                    links = [a.get("href") for a in soup.find_all("a", href=True)]
                    csv_links = [h for h in links if h and ".csv" in h.lower()]
                    if csv_links:
                        csv_url = csv_links[0]
                        if csv_url.startswith("/"):
                            csv_url = cls.BASE + csv_url
                        c = s.get(csv_url, timeout=30)
                        c.raise_for_status()
                        from io import StringIO
                        df = pd.read_csv(StringIO(c.text))
                        # find symbol-like column
                        sym_col = None
                        for c0 in df.columns:
                            if _norm_col(c0) in {"symbol", "ticker", "security symbol"}:
                                sym_col = c0
                                break
                        if sym_col is None:
                            # take first col
                            sym_col = df.columns[0]
                        out = df[[sym_col]].rename(columns={sym_col: "Symbol"}).dropna().drop_duplicates()
                        out["Symbol"] = out["Symbol"].astype(str).str.upper().str.strip()
                        out = out[out["Symbol"].str.len() > 0]
                        out["Ticker"] = out["Symbol"] + ".NS"
                        out.to_csv(cache_path, index=False)
                        return out.reset_index(drop=True)

                    # fallback html tables
                    try:
                        tables = pd.read_html(txt)
                        for t in tables:
                            cols = [_norm_col(c) for c in t.columns]
                            cand = None
                            for c0 in t.columns:
                                if _norm_col(c0) in {"symbol", "security symbol", "underlying"}:
                                    cand = c0
                                    break
                            if cand is not None:
                                out = t[[cand]].rename(columns={cand: "Symbol"}).dropna().drop_duplicates()
                                out["Symbol"] = out["Symbol"].astype(str).str.upper().str.strip()
                                out = out[out["Symbol"].str.len() > 0]
                                out["Ticker"] = out["Symbol"] + ".NS"
                                out.to_csv(cache_path, index=False)
                                return out.reset_index(drop=True)
                    except Exception:
                        pass
            except Exception:
                continue

        raise DataSourceError("Unable to fetch Nifty 200 universe from NSE.")

    @staticmethod
    def fetch_index_history(index_name: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
        if yf is None:
            raise DataSourceError("yfinance is not installed.")
        # NSE historical index series is not always easy to fetch programmatically.
        # We use Yahoo Finance proxy tickers when available as a fallback.
        # For research the built-in benchmark uses equal-weighted Nifty 200 universe.
        raise DataSourceError("Index history fetch is not used in this implementation.")

class YahooDataSource:
    @staticmethod
    def fetch_prices(tickers: List[str], start: str, end: Optional[str]) -> pd.DataFrame:
        if yf is None:
            raise DataSourceError("yfinance is not installed.")
        tickers = [t for t in dict.fromkeys(tickers) if isinstance(t, str) and t]
        if not tickers:
            raise DataSourceError("No tickers provided.")
        cache_path = CACHE_DIR / f"prices_{start}_{end or 'latest'}.parquet"
        if cache_path.exists() and time.time() - cache_path.stat().st_mtime < 2 * 24 * 3600:
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                pass

        chunks = [tickers[i:i + 80] for i in range(0, len(tickers), 80)]
        frames = []
        for ch in chunks:
            data = yf.download(
                tickers=" ".join(ch),
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="column",
                actions=False,
            )
            if data is None or data.empty:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                # prefer Close
                close = data["Close"].copy()
                close.columns = [str(c).upper() for c in close.columns]
            else:
                # single ticker
                close = data[["Close"]].copy()
                close.columns = [ch[0].upper()]
            close = close.dropna(how="all")
            frames.append(close)
        if not frames:
            raise DataSourceError("No price data downloaded from Yahoo Finance.")
        px = pd.concat(frames, axis=1).sort_index()
        px = px.loc[:, ~px.columns.duplicated()]
        px = px.dropna(axis=1, how="all")
        px.index = pd.to_datetime(px.index)
        px = px.sort_index()
        px = px.apply(pd.to_numeric, errors="coerce")
        px = px.where(px > 0)  # guard against zero/negative bad ticks
        try:
            px.to_parquet(cache_path)
        except Exception:
            pass
        return px

    @staticmethod
    @lru_cache(maxsize=1024)
    def _ticker_info(ticker: str) -> Dict:
        if yf is None:
            return {}
        try:
            tk = yf.Ticker(ticker)
            info = tk.info or {}
            if not isinstance(info, dict):
                return {}
            return info
        except Exception:
            return {}

    @classmethod
    def fetch_fundamentals(cls, tickers: List[str], max_workers: int = 8) -> pd.DataFrame:
        cache_path = CACHE_DIR / "fundamentals.json"
        cache: Dict[str, Dict] = {}
        if cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text())
            except Exception:
                cache = {}
        results: Dict[str, Dict] = {}

        def worker(t: str):
            if t in cache:
                return t, cache[t]
            info = cls._ticker_info(t)
            f = {
                "Ticker": t,
                "shortName": info.get("shortName") or info.get("longName") or t,
                "marketCap": safe_float(info.get("marketCap")),
                "sharesOutstanding": safe_float(info.get("sharesOutstanding")),
                "trailingPE": safe_float(info.get("trailingPE")),
                "priceToBook": safe_float(info.get("priceToBook")),
                "priceToSalesTrailing12Months": safe_float(info.get("priceToSalesTrailing12Months")),
                "dividendYield": safe_float(info.get("dividendYield")),
                "returnOnEquity": safe_float(info.get("returnOnEquity")),
                "debtToEquity": safe_float(info.get("debtToEquity")),
                "profitMargins": safe_float(info.get("profitMargins")),
                "operatingMargins": safe_float(info.get("operatingMargins")),
                "earningsGrowth": safe_float(info.get("earningsGrowth")),
                "revenueGrowth": safe_float(info.get("revenueGrowth")),
                "beta": safe_float(info.get("beta")),
                "regularMarketPrice": safe_float(info.get("regularMarketPrice")),
                "currency": info.get("currency"),
            }
            return t, f

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(worker, t) for t in tickers]
            for fut in as_completed(futures):
                t, f = fut.result()
                results[t] = f

        try:
            cache.update(results)
            cache_path.write_text(json.dumps(cache))
        except Exception:
            pass
        df = pd.DataFrame(results.values())
        return df.drop_duplicates("Ticker").reset_index(drop=True)

    @staticmethod
    @lru_cache(maxsize=4096)
    def _live_price_cached(ticker: str) -> float:
        """Best-effort latest tradable price from Yahoo Finance.

        Uses yfinance's regularMarketPrice when available, then falls back to the
        most recent daily close from history.
        """
        if yf is None:
            return np.nan
        try:
            tk = yf.Ticker(ticker)
        except Exception:
            return np.nan

        # 1) regularMarketPrice from quote metadata
        try:
            info = tk.fast_info if hasattr(tk, "fast_info") else None
        except Exception:
            info = None
        if isinstance(info, dict):
            for key in ("lastPrice", "last_price", "regularMarketPrice", "regular_market_price", "previousClose"):
                px = safe_float(info.get(key))
                if np.isfinite(px) and px > 0:
                    return float(px)

        try:
            info = tk.info or {}
            if isinstance(info, dict):
                for key in ("regularMarketPrice", "currentPrice", "previousClose"):
                    px = safe_float(info.get(key))
                    if np.isfinite(px) and px > 0:
                        return float(px)
        except Exception:
            pass

        # 2) most recent close from history
        try:
            hist = tk.history(period="5d", interval="1d", auto_adjust=False)
            if hist is not None and not hist.empty:
                for col in ("Close", "Adj Close", "Open", "High", "Low"):
                    if col in hist.columns:
                        ser = pd.to_numeric(hist[col], errors="coerce").dropna()
                        if not ser.empty:
                            px = safe_float(ser.iloc[-1])
                            if np.isfinite(px) and px > 0:
                                return float(px)
        except Exception:
            pass

        return np.nan

    @classmethod
    def fetch_live_ltp(cls, tickers: List[str]) -> pd.Series:
        tickers = [t for t in dict.fromkeys(tickers) if isinstance(t, str) and t]
        if not tickers:
            return pd.Series(dtype=float)
        out = {}
        for t in tickers:
            out[t] = cls._live_price_cached(t)
        return pd.Series(out, dtype=float)

# -----------------------------
# Factor model
# -----------------------------

class FactorModel:
    def __init__(self, prices: pd.DataFrame, fundamentals: pd.DataFrame, cfg: Config):
        # Keep the raw download for auditing, but use a cleaned price panel for all calculations.
        self.raw_prices = prices.copy().sort_index()
        numeric = self.raw_prices.apply(pd.to_numeric, errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        numeric = numeric.mask(numeric <= 0)
        # Forward fill so held positions keep their last valid market price instead of
        # collapsing to zero when a quote is missing.
        self.price_panel = numeric.ffill()
        self.prices = self.raw_prices
        self.fundamentals = fundamentals.copy()
        self.cfg = cfg
        self.returns = self.price_panel.pct_change()
        self._score_cache: Dict[pd.Timestamp, pd.DataFrame] = {}
        self._basket_cache: Dict[Tuple[pd.Timestamp, float], Dict[str, pd.DataFrame]] = {}

    def universe(self) -> List[str]:
        return [c for c in self.price_panel.columns if c in set(self.fundamentals["Ticker"])]

    def benchmark_returns(self) -> pd.Series:
        # Equal-weighted universe benchmark
        uni = self.universe()
        if not uni:
            return pd.Series(dtype=float)
        return self.returns[uni].mean(axis=1)

    def factor_snapshots(self) -> pd.DataFrame:
        """Create static factor features from current fundamentals."""
        f = self.fundamentals.copy().set_index("Ticker")
        # normalize and proxy missing fields with medians
        for c in ["returnOnEquity", "debtToEquity", "trailingPE", "priceToBook", "dividendYield", "priceToSalesTrailing12Months", "profitMargins", "operatingMargins", "earningsGrowth", "revenueGrowth", "marketCap", "beta"]:
            if c not in f.columns:
                f[c] = np.nan
        # winsorize by quantile
        for c in ["returnOnEquity", "debtToEquity", "trailingPE", "priceToBook", "dividendYield", "priceToSalesTrailing12Months", "profitMargins", "operatingMargins", "earningsGrowth", "revenueGrowth", "marketCap", "beta"]:
            s = pd.to_numeric(f[c], errors="coerce")
            if s.notna().sum() > 5:
                lo, hi = s.quantile([0.01, 0.99])
                s = s.clip(lo, hi)
            f[c] = s
        return f

    def compute_stock_scores(self, as_of: pd.Timestamp) -> pd.DataFrame:
        as_of = pd.Timestamp(as_of)
        if as_of in self._score_cache:
            return self._score_cache[as_of].copy()
        idx = self.price_panel.index.get_indexer([as_of], method="pad")[0]
        if idx < self.cfg.min_history_days:
            raise DataSourceError(f"Not enough history available at {as_of.date()}")
        px = self.price_panel.iloc[: idx + 1].copy()
        rets = px.pct_change()

        universe = self.universe()
        bench = rets[universe].mean(axis=1).dropna()

        rows = []
        fund = self.factor_snapshots()
        for t in universe:
            s = px[t].dropna()
            r = rets[t].dropna()
            if len(s) < self.cfg.min_history_days or len(r) < self.cfg.min_history_days:
                continue

            # Momentum — risk-adjusted combined return using configured score weights.
            # We weight m12 and m6 first, then divide by vol once (not twice independently).
            p_t = s.iloc[-1]
            p_12 = s.iloc[-252]
            p_6 = s.iloc[-126]
            m12 = p_t / p_12 - 1.0
            m6 = p_t / p_6 - 1.0
            vol = r.iloc[-252:].std(ddof=1)
            w12_norm = self.cfg.score_w12 / (self.cfg.score_w12 + self.cfg.score_w6)
            w6_norm = self.cfg.score_w6 / (self.cfg.score_w12 + self.cfg.score_w6)
            combined_mom = w12_norm * m12 + w6_norm * m6
            mom_raw = combined_mom / (vol + 1e-12)

            # Alpha vs equal-weight benchmark
            rr = pd.concat([r.iloc[-252:], bench.iloc[-252:]], axis=1).dropna()
            rr.columns = ["stock", "bench"]
            if len(rr) > 30 and rr["bench"].var() > 0:
                beta = rr["stock"].cov(rr["bench"]) / rr["bench"].var()
                rf_daily = (1 + self.cfg.risk_free_annual) ** (1 / 252) - 1
                alpha = (rr["stock"].mean() - rf_daily) - beta * (rr["bench"].mean() - rf_daily)
                alpha_raw = alpha * 252
            else:
                alpha_raw = np.nan

            ff = fund.loc[t] if t in fund.index else pd.Series(dtype=float)
            roe = safe_float(ff.get("returnOnEquity"))
            debt = safe_float(ff.get("debtToEquity"))
            pb = safe_float(ff.get("priceToBook"))
            pe = safe_float(ff.get("trailingPE"))
            ps = safe_float(ff.get("priceToSalesTrailing12Months"))
            dy = safe_float(ff.get("dividendYield"))
            pm = safe_float(ff.get("profitMargins"))
            om = safe_float(ff.get("operatingMargins"))
            eg = safe_float(ff.get("earningsGrowth"))
            rg = safe_float(ff.get("revenueGrowth"))
            mcap = safe_float(ff.get("marketCap"))

            # Quality — each component is already a ratio/fraction; we z-score within the loop
            # but here we just store the raw components and let the cross-sectional zscore_series
            # normalise them. debt/equity from yfinance can be >100 (e.g. 150 = 150%),
            # so we scale it to a fraction first before log-transforming.
            debt_frac = max(debt, 0) / 100.0 if not np.isnan(debt) else 0.0
            quality_raw = np.nanmean([
                roe if not np.isnan(roe) else np.nan,
                pm if not np.isnan(pm) else np.nan,
                om if not np.isnan(om) else np.nan,
                -np.log1p(debt_frac),
            ])

            # Value — use negative logs of valuation multiples (lower multiple = better value).
            # dy from yfinance is already a decimal (0.03 = 3%), so log1p(dy) is tiny (~0.03).
            # We normalise each sub-signal to the same [-1, 1] order of magnitude by dividing
            # the log multiples by log(20) ≈ 3 (a typical mid-range PE) before averaging.
            _LOG20 = math.log(21.0)  # normaliser ≈ log1p(20)
            val_components = []
            if not np.isnan(pe) and pe > 0:
                val_components.append(-np.log1p(pe) / _LOG20)
            if not np.isnan(pb) and pb > 0:
                val_components.append(-np.log1p(pb) / _LOG20)
            if not np.isnan(ps) and ps > 0:
                val_components.append(-np.log1p(ps) / _LOG20)
            if not np.isnan(dy) and dy > 0:
                # Convert decimal yield to percentage-like scale (3% → 3) for comparability.
                val_components.append(np.log1p(dy * 100) / _LOG20)
            value_raw = float(np.nanmean(val_components)) if val_components else np.nan
            lowvol_raw = -r.iloc[-252:].std(ddof=1)
            rows.append({
                "Ticker": t,
                "MomRaw": mom_raw,
                "AlphaRaw": alpha_raw,
                "QualityRaw": quality_raw,
                "ValueRaw": value_raw,
                "LowVolRaw": lowvol_raw,
                "Mcap": mcap,
                "ROE": roe,
                "DebtToEquity": debt,
                "PE": pe,
                "PB": pb,
                "DivYield": dy,
                "ProfitMargins": pm,
                "OperatingMargins": om,
                "EarningsGrowth": eg,
                "RevenueGrowth": rg,
            })

        feat = pd.DataFrame(rows).set_index("Ticker")
        for c in ["MomRaw", "AlphaRaw", "QualityRaw", "ValueRaw", "LowVolRaw"]:
            feat[c] = zscore_series(feat[c].fillna(feat[c].median()))
        feat["MomentumScore"] = feat["MomRaw"]
        feat["AlphaScore"] = feat["AlphaRaw"]
        feat["QualityScore"] = feat["QualityRaw"]
        feat["ValueScore"] = feat["ValueRaw"]
        feat["LowVolScore"] = feat["LowVolRaw"]
        out = feat.replace([np.inf, -np.inf], np.nan).dropna(subset=["MomentumScore", "AlphaScore", "QualityScore", "ValueScore", "LowVolScore"], how="all")
        self._score_cache[as_of] = out.copy()
        return out

    def select_factor_baskets(self, as_of: pd.Timestamp, temperature: float = 1.0) -> Dict[str, pd.DataFrame]:
        as_of = pd.Timestamp(as_of)
        cache_key = (as_of, float(temperature))
        if cache_key in self._basket_cache:
            return {k: v.copy() for k, v in self._basket_cache[cache_key].items()}
        feat = self.compute_stock_scores(as_of)
        baskets = {}

        def topn(score_col: str, n: int) -> pd.DataFrame:
            df = feat[[score_col, "Mcap"]].copy().sort_values(score_col, ascending=False)
            df = df.head(n).copy()
            df["weight"] = softmax(df[score_col].fillna(df[score_col].median()), temperature=temperature)
            # capping at stock level
            df["weight"] = df["weight"].clip(self.cfg.min_weight, self.cfg.max_weight)
            df["weight"] = df["weight"] / df["weight"].sum()
            return df

        baskets["Alpha"] = topn("AlphaScore", self.cfg.top_n_alpha)
        baskets["Momentum"] = topn("MomentumScore", self.cfg.top_n_momentum)
        baskets["Quality"] = topn("QualityScore", self.cfg.top_n_quality)
        baskets["Value"] = topn("ValueScore", self.cfg.top_n_value)
        baskets["LowVol"] = topn("LowVolScore", self.cfg.top_n_lowvol)
        self._basket_cache[cache_key] = {k: v.copy() for k, v in baskets.items()}
        return {k: v.copy() for k, v in baskets.items()}

# -----------------------------
# Backtest and rotation
# -----------------------------


def value_sleeve_series(
    price_segment: pd.DataFrame,
    shares: pd.Series,
    cash: float,
    starting_value: float,
) -> pd.Series:
    """Value a fixed-share sleeve over time.

    Missing prices must never be interpreted as a zero portfolio value. We use
    forward fill within the segment and, if a row still has no valid prices for
    the held basket, we carry forward the last valid sleeve value.
    """
    if price_segment.empty:
        return pd.Series(dtype=float)

    sh = pd.Series(shares, dtype=float).copy()
    sh = sh.replace([np.inf, -np.inf], np.nan).dropna()
    if sh.empty:
        return pd.Series(float(starting_value), index=price_segment.index, dtype=float)

    px = price_segment.reindex(columns=sh.index).apply(pd.to_numeric, errors="coerce")
    px = px.replace([np.inf, -np.inf], np.nan)
    px = px.mask(px <= 0)
    px = px.ffill()

    sleeve_value = px.mul(sh, axis=1).sum(axis=1, min_count=1) + float(cash)
    sleeve_value = sleeve_value.astype(float).ffill()

    if sleeve_value.empty:
        return pd.Series(float(starting_value), index=price_segment.index, dtype=float)

    first_valid = sleeve_value.first_valid_index()
    if first_valid is None:
        return pd.Series(float(starting_value), index=price_segment.index, dtype=float)

    sleeve_value = sleeve_value.copy()
    sleeve_value.loc[:first_valid] = sleeve_value.loc[:first_valid].fillna(float(starting_value))
    sleeve_value = sleeve_value.ffill().fillna(float(starting_value))
    return sleeve_value



def chain_factor_index(
    raw_sleeve_value: pd.Series,
    prev_index_value: float,
    raw_base_value: float,
) -> pd.Series:
    """Convert a raw sleeve valuation path into a normalized factor index.

    The raw sleeve value includes changing allocated capital at each rebalance.
    For performance analytics and drawdown, we strip allocation size by chaining
    segment returns onto the prior factor index level.
    """
    raw = pd.Series(raw_sleeve_value).copy()
    if raw.empty:
        return pd.Series(dtype=float)

    try:
        raw.index = pd.to_datetime(raw.index)
    except Exception:
        pass

    raw = pd.to_numeric(raw, errors="coerce")
    raw = raw.replace([np.inf, -np.inf], np.nan)
    raw = raw.sort_index().ffill()
    raw = raw.where(raw > 0).dropna()
    if raw.empty:
        return pd.Series(float(prev_index_value), index=pd.Index(raw_sleeve_value.index), dtype=float)

    if not np.isfinite(raw_base_value) or raw_base_value <= 0:
        return pd.Series(float(prev_index_value), index=raw.index, dtype=float)

    idx = float(prev_index_value) * (raw / float(raw_base_value))
    idx = idx.replace([np.inf, -np.inf], np.nan).ffill()

    if idx.empty:
        return pd.Series(float(prev_index_value), index=raw.index, dtype=float)

    first_valid = idx.first_valid_index()
    if first_valid is None:
        return pd.Series(float(prev_index_value), index=raw.index, dtype=float)

    idx = idx.copy()
    idx.loc[:first_valid] = idx.loc[:first_valid].fillna(float(prev_index_value))
    idx = idx.ffill().fillna(float(prev_index_value))
    return idx

class Backtester:

    def __init__(self, prices: pd.DataFrame, model: FactorModel, cfg: Config):
        self.prices = prices.copy().sort_index()
        self.model = model
        self.cfg = cfg
        self.price_panel = getattr(model, "price_panel", self.prices.apply(pd.to_numeric, errors="coerce").where(self.prices > 0).ffill())
        self.returns = self.price_panel.pct_change()

    def month_ends(self) -> List[pd.Timestamp]:
        s = self.price_panel.index.to_series()
        return list(s.groupby(s.dt.to_period("M")).max().dropna().sort_values())

    def _factor_weights_from_history(self, hist: pd.DataFrame, as_of: pd.Timestamp, temperature: float = 1.0) -> pd.Series:
        if hist.empty:
            return pd.Series({f: 1 / 5 for f in FACTOR_NAMES}, dtype=float)

        h = hist.loc[hist.index <= pd.Timestamp(as_of)].copy()
        # Need at least 252 trading days of sleeve history before rotating away from equal-weight.
        if len(h) < 252:
            return pd.Series({f: 1 / 5 for f in FACTOR_NAMES}, dtype=float)

        metrics = []
        for f in FACTOR_NAMES:
            s = h[f"{f}NAV"].dropna().astype(float)
            if len(s) < 252:
                return pd.Series({f: 1 / 5 for f in FACTOR_NAMES}, dtype=float)
            m12 = s.iloc[-1] / s.iloc[-252] - 1.0
            m6 = s.iloc[-1] / s.iloc[-126] - 1.0
            r = s.pct_change().dropna().iloc[-252:]
            sharpe = annualized_sharpe(r, self.cfg.risk_free_annual)
            metrics.append({"Factor": f, "M12": m12, "M6": m6, "Sharpe": 0.0 if np.isnan(sharpe) else sharpe})

        mdf = pd.DataFrame(metrics).set_index("Factor")
        mdf["Z12"] = zscore_series(mdf["M12"])
        mdf["Z6"] = zscore_series(mdf["M6"])
        mdf["ZS"] = zscore_series(mdf["Sharpe"])
        mdf["Score"] = (
            self.cfg.score_w12 * mdf["Z12"]
            + self.cfg.score_w6 * mdf["Z6"]
            + self.cfg.score_wsharpe * mdf["ZS"]
        )
        w = softmax(mdf["Score"], temperature=temperature)
        w = w.clip(self.cfg.factor_min_weight, self.cfg.factor_max_weight)
        w = w / w.sum()
        return w

    @staticmethod
    def _combine_shares(factor_share_maps: Dict[str, pd.Series]) -> pd.Series:
        if not factor_share_maps:
            return pd.Series(dtype=float)
        parts = []
        for s in factor_share_maps.values():
            if s is not None and len(s) > 0:
                parts.append(s.astype(float))
        if not parts:
            return pd.Series(dtype=float)
        out = pd.concat(parts, axis=0).groupby(level=0).sum().sort_index()
        return out[out > 0]

    def factor_portfolio_navs(self, temperature: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        dates = self.month_ends()
        dates = [d for d in dates if d in self.prices.index and self.prices.index.get_loc(d) >= self.cfg.min_history_days]
        if len(dates) < 3:
            raise DataSourceError("Not enough rebalance dates.")

        daily_rows: List[Dict[str, object]] = []
        rebalance_rows: List[Dict[str, object]] = []
        factor_index_rows: List[Dict[str, object]] = []
        trade_rows: List[Dict[str, object]] = []

        portfolio_value = float(self.cfg.initial_capital)

        # Seed factor index history at a constant base so performance and drawdown
        # are not polluted by changing capital allocations.
        first_date = dates[0]
        factor_index_rows.append({"Date": first_date, **{f"{f}NAV": 100.0 for f in FACTOR_NAMES}})
        daily_rows.append({
            "Date": first_date,
            "PortfolioValue": portfolio_value,
            "Return": 0.0,
            **{f"{f}Weight": 1 / 5 for f in FACTOR_NAMES},
        })

        for i, dt in enumerate(dates):
            if dt not in self.prices.index:
                continue

            hist_df = pd.DataFrame(factor_index_rows)
            if "Date" in hist_df.columns:
                hist = hist_df.set_index("Date").sort_index()
            else:
                hist = pd.DataFrame()

            if hist.empty or len(hist) < 20:
                factor_weights = pd.Series({f: 1 / 5 for f in FACTOR_NAMES}, dtype=float)
            else:
                factor_weights = self._factor_weights_from_history(hist, dt, temperature=temperature)

            # Previous normalized factor index at the rebalance date.
            prev_index_map: Dict[str, float] = {}
            if hist.empty:
                prev_index_map = {f: 100.0 for f in FACTOR_NAMES}
            else:
                for f in FACTOR_NAMES:
                    col = f"{f}NAV"
                    if col in hist.columns and hist[col].dropna().empty is False:
                        prev_index_map[f] = float(hist[col].dropna().iloc[-1])
                    else:
                        prev_index_map[f] = 100.0

            # Record the rebalance-date index snapshot so the history is continuous.
            factor_index_rows.append({"Date": dt, **{f"{f}NAV": float(prev_index_map[f]) for f in FACTOR_NAMES}})

            baskets = self.model.select_factor_baskets(dt, temperature=temperature)
            price_row = self.price_panel.loc[dt]

            factor_share_maps: Dict[str, pd.Series] = {}
            factor_cash: Dict[str, float] = {}
            factor_start_values: Dict[str, float] = {}
            factor_target_caps: Dict[str, float] = {}

            for f in FACTOR_NAMES:
                sleeve_cap = float(portfolio_value * float(factor_weights[f]))
                factor_target_caps[f] = sleeve_cap
                basket = baskets.get(f, pd.DataFrame()).copy()

                # Ensure we only keep tickers with a valid price on the rebalance date.
                if not basket.empty and basket.index.name is None:
                    if "Ticker" in basket.columns:
                        basket = basket.set_index("Ticker")
                if basket.empty:
                    factor_share_maps[f] = pd.Series(dtype=float)
                    factor_cash[f] = sleeve_cap
                    factor_start_values[f] = sleeve_cap
                    continue

                tickers = [t for t in basket.index if t in price_row.index and pd.notna(price_row[t])]
                if not tickers:
                    factor_share_maps[f] = pd.Series(dtype=float)
                    factor_cash[f] = sleeve_cap
                    factor_start_values[f] = sleeve_cap
                    continue

                w = basket.loc[tickers, "weight"].astype(float)
                w = w / max(w.sum(), 1e-12)

                target_values = sleeve_cap * w
                px = price_row.reindex(tickers).astype(float)
                shares = np.floor(target_values.values / px.values)
                shares = pd.Series(shares, index=tickers, dtype=float)

                invested = float((shares * px).sum())
                cash = max(sleeve_cap - invested, 0.0)

                factor_share_maps[f] = shares
                factor_cash[f] = cash
                factor_start_values[f] = invested + cash

            # Build a per-lot trade log. Closed lots use the next rebalance date.
            # The final open lot is marked-to-market using live LTP from Yahoo Finance.
            exit_dt = dates[i + 1] if i + 1 < len(dates) else self.price_panel.index[-1]
            trade_status = "Closed" if i + 1 < len(dates) else "Open"

            live_ltp_map: Dict[str, float] = {}
            if trade_status == "Open":
                open_tickers = sorted({
                    ticker
                    for f in FACTOR_NAMES
                    for ticker, shares in factor_share_maps.get(f, pd.Series(dtype=float)).items()
                    if isinstance(ticker, str) and np.isfinite(float(shares)) and float(shares) > 0
                })
                if open_tickers:
                    live_ltp_map = YahooDataSource.fetch_live_ltp(open_tickers).to_dict()

            for f in FACTOR_NAMES:
                sh = factor_share_maps.get(f, pd.Series(dtype=float))
                if sh is None or len(sh) == 0:
                    continue
                for ticker, shares in sh.items():
                    shares = float(shares)
                    if not np.isfinite(shares) or shares <= 0:
                        continue
                    if ticker not in price_row.index or pd.isna(price_row[ticker]):
                        continue
                    buy_price = float(price_row[ticker])

                    if trade_status == "Open":
                        sell_price = safe_float(live_ltp_map.get(ticker))
                        if not np.isfinite(sell_price) or sell_price <= 0:
                            sell_price = buy_price
                    else:
                        sell_price = float(self.price_panel.loc[exit_dt, ticker]) if ticker in self.price_panel.columns else np.nan
                        if not np.isfinite(sell_price) or sell_price <= 0:
                            sell_price = buy_price

                    capital_used = buy_price * shares
                    pnl = (sell_price - buy_price) * shares
                    trade_rows.append({
                        "Date": dt,
                        "ExitDate": exit_dt,
                        "Factor": f,
                        "Ticker": ticker,
                        "Status": trade_status,
                        "BuyPrice": buy_price,
                        "SellPrice": sell_price,
                        "PnL": pnl,
                        "CapitalUsed": capital_used,
                        "Shares": shares,
                    })

            combined_shares = self._combine_shares(factor_share_maps)
            combined_value_at_rebal = float(sum(factor_start_values.values())) if factor_start_values else portfolio_value

            # Trade log summary.
            holdings_count = int((combined_shares > 0).sum())
            rebalance_rows.append({
                "Date": dt,
                "PortfolioValue": portfolio_value,
                **{f"{f}Weight": float(factor_weights[f]) for f in FACTOR_NAMES},
                **{f"{f}Capital": float(factor_target_caps[f]) for f in FACTOR_NAMES},
                **{f"{f}NAV": float(factor_start_values[f]) for f in FACTOR_NAMES},
                **{f"{f}Cash": float(factor_cash[f]) for f in FACTOR_NAMES},
                "HoldingsCount": holdings_count,
                "FactorScoreNote": "softmax over 12M/6M momentum + Sharpe",
            })

            # Simulate until next rebalance date.
            next_dt = dates[i + 1] if i + 1 < len(dates) else self.price_panel.index[-1]
            segment = self.price_panel.loc[(self.price_panel.index > dt) & (self.price_panel.index <= next_dt)].copy()

            # Build combined portfolio series including sleeve cash.
            if not segment.empty:
                sleeve_raw_series_map: Dict[str, pd.Series] = {}
                sleeve_index_series_map: Dict[str, pd.Series] = {}

                for f in FACTOR_NAMES:
                    sh = factor_share_maps.get(f, pd.Series(dtype=float))
                    cash = float(factor_cash.get(f, 0.0))
                    base_value = float(factor_start_values.get(f, 0.0))
                    prev_index = float(prev_index_map.get(f, 100.0))

                    if len(sh) > 0:
                        raw_value = value_sleeve_series(segment, sh, cash, base_value)
                    else:
                        raw_value = pd.Series(cash, index=segment.index, dtype=float)

                    sleeve_raw_series_map[f] = raw_value

                    if np.isfinite(base_value) and base_value > 0:
                        idx_value = chain_factor_index(raw_value, prev_index_value=prev_index, raw_base_value=base_value)
                    else:
                        idx_value = pd.Series(prev_index, index=segment.index, dtype=float)
                    sleeve_index_series_map[f] = idx_value

                combined_series = pd.concat(sleeve_raw_series_map.values(), axis=1).sum(axis=1, min_count=1)
                combined_series = combined_series.ffill()
            else:
                combined_series = pd.Series(dtype=float)
                sleeve_raw_series_map = {f: pd.Series(dtype=float) for f in FACTOR_NAMES}
                sleeve_index_series_map = {f: pd.Series(dtype=float) for f in FACTOR_NAMES}

            # Append a baseline row at the rebalance date for continuity.
            daily_rows.append({
                "Date": dt,
                "PortfolioValue": portfolio_value,
                "Return": 0.0,
                **{f"{f}Weight": float(factor_weights[f]) for f in FACTOR_NAMES},
            })

            # Daily path after rebalance.
            prev_port = portfolio_value
            for day in segment.index:
                if day in combined_series.index and pd.notna(combined_series.loc[day]):
                    pv = float(combined_series.loc[day])
                else:
                    pv = prev_port
                if not np.isfinite(pv) or pv <= 0:
                    pv = prev_port
                ret = (pv / prev_port - 1.0) if prev_port > 0 else 0.0
                row = {
                    "Date": day,
                    "PortfolioValue": pv,
                    "Return": ret,
                    **{f"{f}Weight": float(factor_weights[f]) for f in FACTOR_NAMES},
                }
                daily_rows.append(row)
                prev_port = pv

                hist_row = {"Date": day}
                for f in FACTOR_NAMES:
                    ser = sleeve_index_series_map.get(f, pd.Series(dtype=float))
                    val = float(ser.loc[day]) if day in ser.index and pd.notna(ser.loc[day]) else np.nan
                    hist_row[f"{f}NAV"] = val
                factor_index_rows.append(hist_row)

            if not segment.empty and len(combined_series) > 0:
                portfolio_value = float(combined_series.iloc[-1])
                if not np.isfinite(portfolio_value) or portfolio_value <= 0:
                    portfolio_value = prev_port
            else:
                portfolio_value = combined_value_at_rebal

        daily_df = pd.DataFrame(daily_rows).drop_duplicates("Date", keep="last").sort_values("Date").reset_index(drop=True)
        daily_df["Cumulative Return"] = daily_df["PortfolioValue"] / float(self.cfg.initial_capital) - 1.0

        factor_navs = pd.DataFrame(factor_index_rows).drop_duplicates("Date", keep="last").sort_values("Date").reset_index(drop=True)
        rebalance_log = pd.DataFrame(rebalance_rows).sort_values("Date").reset_index(drop=True)
        trade_log = pd.DataFrame(trade_rows).sort_values(["Date", "Factor", "Ticker"]).reset_index(drop=True)
        return daily_df, factor_navs, rebalance_log, trade_log

def compute_metrics(nav: pd.Series, rf_annual: float) -> Dict[str, float]:
    nav = sanitize_nav_series(nav, forward_fill=True)
    if len(nav) < 2:
        return {}
    ret = nav.pct_change().dropna()
    total = nav.iloc[-1] / nav.iloc[0] - 1.0
    # Use actual calendar years for CAGR so it is not distorted by holiday gaps.
    if isinstance(nav.index, pd.DatetimeIndex):
        try:
            years = (nav.index[-1] - nav.index[0]).days / 365.25
        except Exception:
            years = (len(nav) - 1) / 252.0
    else:
        years = (len(nav) - 1) / 252.0
    years = max(years, 1 / 252.0)
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0
    vol = ret.std(ddof=1) * math.sqrt(252)
    sharpe = annualized_sharpe(ret, rf_annual)
    # Pass the NAV series to max_drawdown so the running-peak is computed on actual prices.
    dd = max_drawdown_nav(nav)
    calmar = cagr / abs(dd) if (dd is not None and np.isfinite(dd) and dd < 0) else np.nan
    return {"TotalReturn": total, "CAGR": cagr, "Volatility": vol, "Sharpe": sharpe, "MaxDrawdown": dd, "Calmar": calmar}



# -----------------------------
# Reporting / plots
# -----------------------------

def fig_line(df: pd.DataFrame, cols: List[str], title: str, y_title: str = "Value") -> go.Figure:
    fig = go.Figure()
    for c in cols:
        if c in df.columns:
            fig.add_trace(go.Scatter(x=df.index if df.index.name else df["Date"], y=df[c], mode="lines", name=c))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title=y_title, template="plotly_white", height=420)
    return fig

def fig_stacked_weights(reb: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for f in FACTOR_NAMES:
        col = f"{f}Weight"
        if col in reb.columns:
            fig.add_trace(go.Scatter(x=reb["Date"], y=reb[col] * 100, mode="lines+markers", name=f))
    fig.update_layout(title="Factor Weights at Rebalance", xaxis_title="Date", yaxis_title="Weight %", template="plotly_white", height=420)
    return fig

# -----------------------------
# Streamlit app
# -----------------------------

def run_app():
    if st is None:
        raise RuntimeError("streamlit is not installed.")

    st.set_page_config(page_title="Nifty Factor Rotation", layout="wide")
    st.title("Nifty Factor Rotation — self-fetching stock-level dashboard")
    st.caption("Auto-downloads NSE universe + Yahoo Finance prices/fundamentals, then builds factor sleeves and monthly capital rotation.")

    with st.sidebar:
        st.header("Settings")
        capital = st.number_input("Initial capital (₹)", min_value=10000.0, value=100000.0, step=10000.0)
        start_date = st.text_input("Start date", value="2018-01-01")
        end_date = st.text_input("End date (blank = today)", value="")
        temp = st.slider("Softmax temperature", 0.25, 3.00, 1.00, 0.05)
        st.subheader("Factor sleeve bounds")
        fmin = st.slider("Min factor weight", 0.00, 0.30, 0.10, 0.01)
        fmax = st.slider("Max factor weight", 0.10, 0.80, 0.40, 0.01)
        st.subheader("Stock caps")
        smax = st.slider("Max stock weight inside sleeve", 0.02, 0.20, 0.10, 0.01)
        st.caption("This app rebuilds factor sleeves from public data and is suitable for research and deployment prototyping.")

    cfg = Config(
        initial_capital=float(capital),
        start_date=start_date,
        end_date=end_date or None,
        factor_min_weight=float(fmin),
        factor_max_weight=float(fmax),
        max_weight=float(smax),
    )

    @st.cache_data(show_spinner=False)
    def load_all(cfg: Config):
        universe = NSEDataSource.fetch_nifty200_universe()
        tickers = universe["Ticker"].tolist()
        prices = YahooDataSource.fetch_prices(tickers=tickers, start=cfg.start_date, end=cfg.end_date)
        # align to universe tickers only
        cols = [c for c in prices.columns if c in tickers]
        prices = prices[cols].dropna(axis=1, how="all")
        fundamentals = YahooDataSource.fetch_fundamentals(cols)
        return universe, prices, fundamentals

    if st.button("Fetch data and run backtest", type="primary"):
        progress = st.progress(0)
        status = st.empty()

        try:
            status.info("Step 1/4 — downloading universe, prices, and fundamentals...")
            progress.progress(15)
            universe, prices, fundamentals = load_all(cfg)

            st.success(f"Universe loaded: {len(universe)} stocks | Price matrix: {prices.shape[0]} days × {prices.shape[1]} tickers")
            progress.progress(35)

            status.info("Step 2/4 — building factor sleeves and monthly rotation...")
            model = FactorModel(prices=prices, fundamentals=fundamentals, cfg=cfg)
            bt = Backtester(prices=prices, model=model, cfg=cfg)
            daily_nav, sleeve_navs, rebalance_log, trade_log = bt.factor_portfolio_navs(temperature=temp)
            progress.progress(70)

            status.info("Step 3/3 — computing metrics and rendering charts...")
            progress.progress(85)
            # metrics
            portfolio_nav = daily_nav.set_index("Date")["PortfolioValue"]
            metrics = compute_metrics(portfolio_nav, cfg.risk_free_annual)
            sleeve_metrics = {}
            if not sleeve_navs.empty:
                for f in FACTOR_NAMES:
                    col = f"{f}NAV"
                    if col in sleeve_navs.columns:
                        sleeve_metrics[f] = compute_metrics(sleeve_navs.set_index("Date")[col], cfg.risk_free_annual)

            progress.progress(100)
            status.success("Backtest complete.")
        except Exception as e:
            status.error(f"Run failed: {e}")
            st.exception(e)
            st.stop()

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Ending value", f"₹{portfolio_nav.iloc[-1]:,.0f}")
        c2.metric("CAGR", f"{metrics.get('CAGR', np.nan)*100:.2f}%")
        c3.metric("Sharpe", f"{metrics.get('Sharpe', np.nan):.2f}")
        c4.metric("Max DD", f"{metrics.get('MaxDrawdown', np.nan)*100:.2f}%")
        c5.metric("Rebalances", f"{len(rebalance_log)}")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Factor Sleeves", "Rebalance Log", "Trade Log", "Data"])

        with tab1:
            col1, col2 = st.columns([1.15, 0.85])
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=daily_nav["Date"], y=daily_nav["PortfolioValue"], mode="lines", name="Portfolio"))
                fig.update_layout(title="Portfolio Equity Curve", template="plotly_white", height=420)
                st.plotly_chart(fig, use_container_width=True, key="overview_equity_curve")
                if not sleeve_navs.empty:
                    fig2 = go.Figure()
                    for f in FACTOR_NAMES:
                        fig2.add_trace(go.Scatter(x=sleeve_navs.index, y=sleeve_navs[f"{f}NAV"], mode="lines", name=f))
                    fig2.update_layout(title="Factor Sleeve NAVs", template="plotly_white", height=420)
                    st.plotly_chart(fig2, use_container_width=True, key="overview_sleeve_navs")
            with col2:
                met = pd.DataFrame([{"Metric": k, "Value": v} for k, v in metrics.items()])
                met["Value"] = met.apply(
                    lambda r: f"{r['Value']*100:.2f}%" if r["Metric"] in {"TotalReturn", "CAGR", "Volatility", "MaxDrawdown"} else f"{r['Value']:.2f}",
                    axis=1
                )
                st.dataframe(met, use_container_width=True, hide_index=True)
                st.subheader("Latest factor allocation")
                if not rebalance_log.empty:
                    last = rebalance_log.iloc[-1]
                    alloc = pd.DataFrame({
                        "Factor": FACTOR_NAMES,
                        "Weight": [last[f"{f}Weight"] for f in FACTOR_NAMES],
                    })
                    alloc["Weight"] = alloc["Weight"].map(lambda x: f"{x*100:.2f}%")
                    st.dataframe(alloc, use_container_width=True, hide_index=True)

        with tab2:
            st.subheader("Factor sleeve stats")
            rows = []
            for f in FACTOR_NAMES:
                mm = sleeve_metrics.get(f, {})
                rows.append({"Factor": f, **mm})
            sm = pd.DataFrame(rows)
            if not sm.empty:
                for c in ["TotalReturn", "CAGR", "Volatility", "MaxDrawdown"]:
                    if c in sm.columns:
                        sm[c] = sm[c].map(lambda x: f"{x*100:.2f}%")
                if "Sharpe" in sm.columns:
                    sm["Sharpe"] = sm["Sharpe"].map(lambda x: f"{x:.2f}")
                st.dataframe(sm, use_container_width=True, hide_index=True)
            if not sleeve_navs.empty:
                st.plotly_chart(go.Figure([
                    go.Scatter(x=sleeve_navs.index, y=sleeve_navs[f"{f}NAV"], mode="lines", name=f) for f in FACTOR_NAMES
                ]).update_layout(title="Sleeve NAVs", template="plotly_white", height=420), use_container_width=True, key="factor_sleeve_navs")

        with tab3:
            st.subheader("Rebalance log")
            if not rebalance_log.empty:
                show = rebalance_log.copy()
                for f in FACTOR_NAMES:
                    show[f"{f}Weight"] = show[f"{f}Weight"].map(lambda x: f"{x*100:.2f}%")
                    if f"{f}NAV" in show.columns:
                        show[f"{f}NAV"] = show[f"{f}NAV"].map(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "-")
                show["PortfolioValue"] = show["PortfolioValue"].map(lambda x: f"₹{x:,.0f}")
                st.dataframe(show, use_container_width=True, hide_index=True)
                st.plotly_chart(fig_stacked_weights(rebalance_log), use_container_width=True, key="rebalance_weight_chart")

        with tab4:
            st.subheader("Trade log")
            if not trade_log.empty:
                show = trade_log.copy()
                show["Date"] = pd.to_datetime(show["Date"]).dt.strftime("%Y-%m-%d")
                show["ExitDate"] = pd.to_datetime(show["ExitDate"]).dt.strftime("%Y-%m-%d")
                show["BuyPrice"] = show["BuyPrice"].map(lambda x: f"₹{x:,.2f}")
                show["SellPrice"] = show["SellPrice"].map(lambda x: f"₹{x:,.2f}")
                show["PnL"] = show["PnL"].map(lambda x: f"₹{x:,.2f}")
                show["CapitalUsed"] = show["CapitalUsed"].map(lambda x: f"₹{x:,.2f}")
                show["Shares"] = show["Shares"].map(lambda x: f"{x:,.0f}")
                st.dataframe(show, use_container_width=True, hide_index=True)
            else:
                st.info("No trades available for the selected period.")

        with tab5:
            st.subheader("Universe snapshot")
            st.dataframe(universe, use_container_width=True, hide_index=True)
            st.subheader("Fundamentals snapshot")
            st.dataframe(fundamentals, use_container_width=True, hide_index=True)
            st.download_button("Download universe CSV", universe.to_csv(index=False).encode("utf-8"), "nifty200_universe.csv", "text/csv")
            st.download_button("Download prices CSV", prices.reset_index().to_csv(index=False).encode("utf-8"), "prices.csv", "text/csv")

if __name__ == "__main__":
    if st is None:
        raise SystemExit("streamlit is not installed. Install streamlit and run: streamlit run nifty_factor_rotation_auto.py")
    run_app()