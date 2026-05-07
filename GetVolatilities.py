import math
import re
import json
import html as html_lib
import datetime as dt
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from scipy.optimize import brentq
from scipy.stats import norm


INPUT_FILE = "StockList.csv"
OUTPUT_FILE = "StockList_with_IV.csv"

RISK_FREE_RATE = 0.03
HIST_VOL_LOOKBACK_DAYS = 60
REQUEST_TIMEOUT = 20
MAX_STRIKES_PER_SIDE = 4

MIN_EXPIRY_QUALITY = 0.25
MIN_WEIGHT_TO_KEEP = 1e-6


@dataclass
class OptionCandidate:
    expiry: dt.date
    strike: float
    bid: float
    ask: float
    last: float
    mx_iv: Optional[float]
    kind: str
    bid_size: float = 0.0
    ask_size: float = 0.0


@dataclass
class IVResult:
    iv: Optional[float] = None
    expiry: str = ""
    forward_iv: Optional[float] = None
    forward_expiry: str = ""
    source: str = ""
    final_status: str = ""
    yf_status: str = ""
    mx_status: str = ""
    hv_status: str = ""


def bs_call_price(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return max(S - K * math.exp(-r * T), 0.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return max(K * math.exp(-r * T) - S, 0.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(price, S, K, T, r, is_call):
    if price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return None

    def f(sig):
        model = bs_call_price(S, K, T, r, sig) if is_call else bs_put_price(S, K, T, r, sig)
        return model - price

    try:
        return brentq(f, 1e-6, 5.0, maxiter=200)
    except Exception:
        return None


def year_fraction(expiry_date: dt.date) -> float:
    now = dt.datetime.now()
    expiry_dt = dt.datetime.combine(expiry_date, dt.time(hour=16, minute=0))
    seconds = (expiry_dt - now).total_seconds()
    return max(seconds / (365.25 * 24 * 60 * 60), 1e-6)


def forward_iv_from_two_expiries(iv1, expiry1, iv2, expiry2):
    t1 = year_fraction(expiry1)
    t2 = year_fraction(expiry2)

    if t2 <= t1:
        return None

    forward_var = (iv2**2 * t2 - iv1**2 * t1) / (t2 - t1)

    if forward_var <= 0 or not math.isfinite(forward_var):
        return None

    return math.sqrt(forward_var)


def annualized_hist_vol(prices):
    if prices is None or not isinstance(prices, (pd.Series, list, tuple, np.ndarray)):
        return None

    prices = pd.to_numeric(pd.Series(prices), errors="coerce").dropna()

    if len(prices) < HIST_VOL_LOOKBACK_DAYS + 1:
        return None

    r = np.log(prices).diff().dropna()

    if len(r) < 10:
        return None

    return float(r.std(ddof=1) * math.sqrt(252))


def clean_df(path):
    df = pd.read_csv(path)

    if "Symbol" not in df.columns:
        raise ValueError("Input file must contain a Symbol column.")

    if "Primary Exchange" in df.columns:
        df = df.drop(columns=["Primary Exchange"])

    numeric_cols = ["Implied Volatility", "Forward Implied Volatility"]
    text_cols = [
        "Expiry Date",
        "Forward Expiry Date",
        "IV Source",
        "Final Status",
        "YF Status",
        "MX Status",
        "HV Status",
    ]

    for col in numeric_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    for col in text_cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype("object")

    return df


def get_spot(symbol):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="6mo", auto_adjust=False)

        if hist.empty or "Close" not in hist.columns:
            return None, None, "no price history"

        close = pd.to_numeric(hist["Close"], errors="coerce").dropna()

        if close.empty:
            return None, None, "no usable close prices"

        return float(close.iloc[-1]), close, "ok"
    except Exception as e:
        return None, None, f"spot lookup failed: {e}"


def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def moneyness_weight(K, spot):
    m = abs(math.log(K / spot))
    return math.exp(-m / 0.05)


def liquidity_score(volume, open_interest):
    volume = max(safe_float(volume), 0.0)
    open_interest = max(safe_float(open_interest), 0.0)

    volume_quality = math.log1p(volume) / math.log1p(10)
    oi_quality = math.log1p(open_interest) / math.log1p(100)

    return min(1.0, 0.5 * volume_quality + 0.5 * oi_quality)


def size_score(bid_size, ask_size):
    bid_size = max(safe_float(bid_size), 0.0)
    ask_size = max(safe_float(ask_size), 0.0)

    bid_quality = math.log1p(bid_size) / math.log1p(10)
    ask_quality = math.log1p(ask_size) / math.log1p(10)

    return min(1.0, 0.5 * bid_quality + 0.5 * ask_quality)


def get_yf_expiries(ticker):
    today = dt.date.today()
    valid = []

    try:
        expiries = list(ticker.options)
    except Exception:
        return []

    for e in expiries:
        try:
            d = dt.datetime.strptime(str(e), "%Y-%m-%d").date()
            if d >= today:
                valid.append((d, str(e)))
        except Exception:
            continue

    valid.sort(key=lambda x: x[0])
    return valid


def yf_iv_for_expiry(ticker, expiry_date, expiry_str, spot):
    try:
        chain = ticker.option_chain(expiry_str)
        calls = chain.calls.copy()
        puts = chain.puts.copy()
    except Exception:
        return None, 0.0, "chain failed"

    T = year_fraction(expiry_date)
    rows = []

    for df, is_call in [(calls, True), (puts, False)]:
        option_rows = []

        if df.empty:
            continue

        for _, row in df.iterrows():
            K = safe_float(row.get("strike"))
            bid = safe_float(row.get("bid"))
            ask = safe_float(row.get("ask"))
            volume = safe_float(row.get("volume"))
            open_interest = safe_float(row.get("openInterest"))

            if K <= 0:
                continue
            if is_call and K < spot:
                continue
            if not is_call and K > spot:
                continue
            if ask <= 0:
                continue

            m = abs(math.log(K / spot))
            option_rows.append((m, K, bid, ask, volume, open_interest))

        option_rows.sort(key=lambda x: x[0])
        option_rows = option_rows[:MAX_STRIKES_PER_SIDE]

        for m, K, bid, ask, volume, open_interest in option_rows:
            if bid > 0 and ask > bid:
                price = 0.5 * (bid + ask)
                rel_spread = (ask - bid) / price
                spread_quality = 1 / (1 + 5 * rel_spread)
            else:
                price = 0.5 * ask
                spread_quality = 0.15

            iv = implied_vol(price, spot, K, T, RISK_FREE_RATE, is_call)

            if iv is None or not math.isfinite(iv) or iv <= 0:
                continue

            liq_quality = max(0.10, liquidity_score(volume, open_interest))
            quality = spread_quality * liq_quality
            weight = math.exp(-m / 0.05) * quality

            if weight > MIN_WEIGHT_TO_KEEP and math.isfinite(weight):
                rows.append((iv, weight))

    quality_score = float(sum(w for _, w in rows))

    if not rows:
        return None, quality_score, "no usable rows"

    if quality_score < MIN_EXPIRY_QUALITY:
        return None, quality_score, f"quality below threshold ({quality_score:.4f})"

    iv = float(np.average([x[0] for x in rows], weights=[x[1] for x in rows]))
    return iv, quality_score, f"ok quality={quality_score:.4f}, rows={len(rows)}"


def yf_weighted_iv(symbol, spot):
    try:
        ticker = yf.Ticker(symbol)
        expiries = get_yf_expiries(ticker)
    except Exception as e:
        return None, None, None, None, f"yfinance failed: {e}"

    if not expiries:
        return None, None, None, None, "no unexpired yfinance expiries"

    iv_results = []
    statuses = []

    for expiry_date, expiry_str in expiries[:4]:
        iv, quality, status = yf_iv_for_expiry(ticker, expiry_date, expiry_str, spot)
        statuses.append(f"{expiry_str}: {status}")

        if iv is not None:
            iv_results.append((expiry_date, expiry_str, iv, quality))

        if len(iv_results) >= 2:
            break

    if not iv_results:
        return None, None, None, None, "no usable yfinance IV; " + " | ".join(statuses)

    expiry1_date, expiry1_str, iv1, quality1 = iv_results[0]

    forward_iv = None
    forward_expiry = ""

    if len(iv_results) >= 2:
        expiry2_date, expiry2_str, iv2, quality2 = iv_results[1]
        forward_iv = forward_iv_from_two_expiries(iv1, expiry1_date, iv2, expiry2_date)
        forward_expiry = expiry2_str if forward_iv is not None else ""

    return iv1, expiry1_str, forward_iv, forward_expiry, " | ".join(statuses)


def fetch_mx_html(symbol):
    root = symbol[:-3].strip().upper()
    roots = []

    for r in [root, root.replace("-", "."), root.split("-")[0], root.split(".")[0]]:
        if r and r not in roots:
            roots.append(r)

    for r in roots:
        for url in [
            f"https://www.m-x.ca/en/trading/data/quotes?symbol={r}%2A",
            f"https://www.m-x.ca/en/trading/data/quotes?symbol={r}",
        ]:
            try:
                resp = requests.get(
                    url,
                    timeout=REQUEST_TIMEOUT,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                resp.raise_for_status()

                if resp.text:
                    return resp.text, r
            except Exception:
                continue

    return None, ""


def first_present_float(d, keys, default=0.0):
    for k in keys:
        if k in d:
            return safe_float(d.get(k), default)
    return default


def parse_mx(html):
    soup = BeautifulSoup(html, "html.parser")
    rows = []

    for tr in soup.find_all("tr"):
        data = tr.get("data-row")

        if not data:
            continue

        try:
            obj = json.loads(html_lib.unescape(data))

            for kind in ["call", "put"]:
                o = obj.get(kind, {})
                iv = safe_float(o.get("volatility")) / 100.0

                if iv <= 0:
                    continue

                rows.append(
                    OptionCandidate(
                        expiry=dt.datetime.strptime(o["expiry_date"], "%Y-%m-%d").date(),
                        strike=safe_float(o.get("strike_price")),
                        bid=safe_float(o.get("bid_price")),
                        ask=safe_float(o.get("ask_price")),
                        last=safe_float(o.get("last_price")),
                        mx_iv=iv,
                        kind=kind,
                        bid_size=first_present_float(o, ["bid_size", "bid_volume", "bid_qty", "bid_quantity"]),
                        ask_size=first_present_float(o, ["ask_size", "ask_volume", "ask_qty", "ask_quantity"]),
                    )
                )
        except Exception:
            continue

    return rows


def mx_iv_for_expiry(rows, expiry, spot):
    rows = [r for r in rows if r.expiry == expiry]
    T = year_fraction(expiry)
    weighted = []
    filtered = []

    for r in rows:
        if r.mx_iv is None or r.mx_iv <= 0 or r.strike <= 0:
            continue
        if r.kind == "call" and r.strike < spot:
            continue
        if r.kind == "put" and r.strike > spot:
            continue

        m = abs(math.log(r.strike / spot))
        filtered.append((m, r))

    calls_near = sorted(
        [(m, r) for m, r in filtered if r.kind == "call"],
        key=lambda x: x[0],
    )[:MAX_STRIKES_PER_SIDE]

    puts_near = sorted(
        [(m, r) for m, r in filtered if r.kind == "put"],
        key=lambda x: x[0],
    )[:MAX_STRIKES_PER_SIDE]

    for m, r in calls_near + puts_near:
        if m > 0.20:
            continue

        if r.bid > 0 and r.ask > r.bid:
            mid = 0.5 * (r.bid + r.ask)
            rel_spread = (r.ask - r.bid) / mid
            spread_quality = 1 / (1 + 5 * rel_spread)
        elif r.ask > 0:
            spread_quality = 0.15
        elif r.last > 0:
            spread_quality = 0.10
        else:
            continue

        if r.kind == "call":
            model = bs_call_price(spot, r.strike, T, RISK_FREE_RATE, r.mx_iv)
        else:
            model = bs_put_price(spot, r.strike, T, RISK_FREE_RATE, r.mx_iv)

        if r.last > 0:
            err = abs(model - r.last) / max(model, r.last, 0.01)
            stale_quality = math.exp(-5 * err)
        else:
            stale_quality = 0.50

        size_quality = max(0.10, size_score(r.bid_size, r.ask_size))
        quality = spread_quality * size_quality * stale_quality
        weight = math.exp(-m / 0.05) * quality

        if weight > MIN_WEIGHT_TO_KEEP and math.isfinite(weight):
            weighted.append((r.mx_iv, weight))

    quality_score = float(sum(w for _, w in weighted))

    if not weighted:
        return None, quality_score, "no usable rows"

    if quality_score < MIN_EXPIRY_QUALITY:
        return None, quality_score, f"quality below threshold ({quality_score:.4f})"

    iv = float(np.average([x[0] for x in weighted], weights=[x[1] for x in weighted]))
    return iv, quality_score, f"ok quality={quality_score:.4f}, rows={len(weighted)}"


def mx_weighted_iv(symbol, spot):
    if not symbol.endswith(".TO"):
        return None, None, None, None, "skipped: not .TO"

    if spot is None or spot <= 0:
        return None, None, None, None, "skipped: no usable spot"

    html, root = fetch_mx_html(symbol)

    if not html:
        return None, None, None, None, "MX fetch failed"

    rows = parse_mx(html)

    if not rows:
        return None, None, None, None, f"no MX rows parsed, root={root}"

    today = dt.date.today()
    expiries = sorted(set(r.expiry for r in rows if r.expiry >= today))

    if not expiries:
        return None, None, None, None, f"no future MX expiries, root={root}"

    iv_results = []
    statuses = []

    for expiry in expiries[:4]:
        iv, quality, status = mx_iv_for_expiry(rows, expiry, spot)
        statuses.append(f"{expiry.isoformat()}: {status}")

        if iv is not None:
            iv_results.append((expiry, expiry.isoformat(), iv, quality))

        if len(iv_results) >= 2:
            break

    if not iv_results:
        return None, None, None, None, f"no usable MX IV, root={root}; " + " | ".join(statuses)

    expiry1_date, expiry1_str, iv1, quality1 = iv_results[0]

    forward_iv = None
    forward_expiry = ""

    if len(iv_results) >= 2:
        expiry2_date, expiry2_str, iv2, quality2 = iv_results[1]
        forward_iv = forward_iv_from_two_expiries(iv1, expiry1_date, iv2, expiry2_date)
        forward_expiry = expiry2_str if forward_iv is not None else ""

    return iv1, expiry1_str, forward_iv, forward_expiry, f"root={root}; " + " | ".join(statuses)


def resolve(symbol):
    result = IVResult()

    try:
        spot, hist, spot_status = get_spot(symbol)
        result.yf_status = f"spot: {spot_status}"
    except Exception as e:
        spot, hist = None, None
        result.yf_status = f"spot lookup failed: {e}"

    try:
        if spot is not None and spot > 0:
            iv, expiry, fiv, fexp, status = yf_weighted_iv(symbol, spot)
            result.yf_status = status

            if iv is not None:
                result.iv = iv
                result.expiry = expiry or ""
                result.forward_iv = fiv
                result.forward_expiry = fexp or ""
                result.source = "yfinance_weighted"
                result.final_status = "ok"
                result.mx_status = "not attempted"
                result.hv_status = "not attempted"
                return result
        else:
            result.yf_status = "no usable yfinance spot"
    except Exception as e:
        result.yf_status = f"yfinance failed: {e}"

    try:
        iv, expiry, fiv, fexp, status = mx_weighted_iv(symbol, spot)
        result.mx_status = status

        if iv is not None:
            result.iv = iv
            result.expiry = expiry or ""
            result.forward_iv = fiv
            result.forward_expiry = fexp or ""
            result.source = "mx_weighted"
            result.final_status = "ok"
            result.hv_status = "not attempted"
            return result
    except Exception as e:
        result.mx_status = f"mx failed: {e}"

    try:
        hv = annualized_hist_vol(hist)
        result.hv_status = "ok" if hv is not None else "failed"

        if hv is not None:
            result.iv = hv
            result.expiry = ""
            result.forward_iv = None
            result.forward_expiry = ""
            result.source = "historical"
            result.final_status = "fallback used"
            return result
    except Exception as e:
        result.hv_status = f"hv failed: {e}"

    result.final_status = "all methods failed"
    return result


def reorder_columns(df):
    if "Primary Exchange" in df.columns:
        df = df.drop(columns=["Primary Exchange"])

    desired_order = [
        "Name",
        "Symbol",
        "Implied Volatility",
        "Forward Implied Volatility",
        "Expiry Date",
        "Forward Expiry Date",
        "IV Source",
        "Final Status",
        "YF Status",
        "MX Status",
        "HV Status",
    ]

    remaining = [c for c in df.columns if c not in desired_order]

    return df[desired_order + remaining]


def main():
    df = clean_df(INPUT_FILE)

    for i, row in df.iterrows():
        raw_symbol = row.get("Symbol", "")

        if pd.isna(raw_symbol) or str(raw_symbol).strip() == "":
            df.at[i, "Implied Volatility"] = np.nan
            df.at[i, "Forward Implied Volatility"] = np.nan
            df.at[i, "Expiry Date"] = ""
            df.at[i, "Forward Expiry Date"] = ""
            df.at[i, "IV Source"] = ""
            df.at[i, "Final Status"] = "blank symbol"
            df.at[i, "YF Status"] = "not attempted"
            df.at[i, "MX Status"] = "not attempted"
            df.at[i, "HV Status"] = "not attempted"
            print("blank symbol")
            continue

        symbol = str(raw_symbol).strip().upper()

        try:
            r = resolve(symbol)
        except Exception as e:
            r = IVResult(final_status=f"unexpected failure: {e}")

        df.at[i, "Implied Volatility"] = r.iv if r.iv is not None else np.nan
        df.at[i, "Forward Implied Volatility"] = r.forward_iv if r.forward_iv is not None else np.nan
        df.at[i, "Expiry Date"] = r.expiry
        df.at[i, "Forward Expiry Date"] = r.forward_expiry
        df.at[i, "IV Source"] = r.source
        df.at[i, "Final Status"] = r.final_status
        df.at[i, "YF Status"] = r.yf_status
        df.at[i, "MX Status"] = r.mx_status
        df.at[i, "HV Status"] = r.hv_status

        print(
            f"{symbol}: IV={r.iv} expiry={r.expiry} "
            f"forward_IV={r.forward_iv} forward_expiry={r.forward_expiry} "
            f"source={r.source} status={r.final_status}"
        )

    df = reorder_columns(df)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nWrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()