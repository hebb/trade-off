import math
import re
import json
import html as html_lib
import os
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

try:
    import pandas_market_calendars as mcal
except Exception:
    mcal = None


INPUT_FILE = "StockList_with_earnings.csv" if os.path.exists("StockList_with_earnings.csv") else "StockList.csv"
OUTPUT_FILE = "StockList_with_IV.csv"

RISK_FREE_RATE = 0.03
HIST_VOL_LOOKBACK_DAYS = 60
REQUEST_TIMEOUT = 20
MAX_STRIKES_PER_SIDE = 4
MIN_EXPIRY_QUALITY = 0.25
MIN_WEIGHT_TO_KEEP = 1e-6

MARKET_WINDOW_CACHE = {}


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
    override_iv: Optional[float] = None
    override_used: bool = False
    override_status: str = ""
    earnings_window_start: str = ""
    earnings_window_end: str = ""
    earnings_event_start: str = ""
    earnings_event_end: str = ""
    earnings_timing_normalized: str = ""
    earnings_window_overlap: str = "False"
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
        df["Primary Exchange"] = df["Primary Exchange"].astype("object")

    numeric_cols = [
        "Implied Volatility",
    ]

    text_cols = [
        "Expiry Date",
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


def exchange_calendar_name(primary_exchange: str) -> Optional[str]:
    x = str(primary_exchange or "").strip().upper()

    if x in {
        "NYSE",
        "NEW YORK STOCK EXCHANGE",
        "NASDAQ",
        "NASDAQGS",
        "NASDAQGM",
        "NASDAQCM",
        "AMEX",
        "NYSEAMERICAN",
        "NYSEARCA",
    }:
        return "XNYS"

    if x in {
        "TSX",
        "TSE",
        "TORONTO",
        "TORONTO STOCK EXCHANGE",
        "TSXV",
        "TSX VENTURE",
        "TSX VENTURE EXCHANGE",
    }:
        return "XTSE"

    return None


def exchange_timezone_name(primary_exchange: str) -> Optional[str]:
    cal_name = exchange_calendar_name(primary_exchange)

    if cal_name == "XNYS":
        return "America/New_York"

    if cal_name == "XTSE":
        return "America/Toronto"

    return None


def normalize_earnings_timing(x) -> str:
    s = str(x or "").strip().lower()
    s = re.sub(r"\s+", " ", s)

    if s == "unknown":
        return "Unknown"

    if s in {
        "after market close",
        "after close",
        "amc",
        "post-market",
        "post market",
    }:
        return "After Market Close"

    if s in {
        "before market open",
        "before open",
        "bmo",
        "pre-market",
        "pre market",
    }:
        return "Before Market Open"

    return str(x or "").strip()


def get_overnight_market_window(primary_exchange: str):
    """
    Correct override window:

        next_open = next market open after now
        window_start = market close immediately before that next_open
        window_end = next_open

    The returned window timestamps are timezone-aware UTC pandas Timestamps.
    local_tz_name is used only for constructing and displaying earnings intervals.
    """
    if mcal is None:
        return None, None, None, "pandas_market_calendars not installed"

    cal_name = exchange_calendar_name(primary_exchange)
    local_tz_name = exchange_timezone_name(primary_exchange)

    if cal_name is None or local_tz_name is None:
        return None, None, None, f"unknown exchange: {primary_exchange}"

    if cal_name in MARKET_WINDOW_CACHE:
        return MARKET_WINDOW_CACHE[cal_name]

    try:
        cal = mcal.get_calendar(cal_name)
        now_utc = pd.Timestamp.now(tz="UTC")

        sched = cal.schedule(
            start_date=(now_utc - pd.Timedelta(days=14)).date(),
            end_date=(now_utc + pd.Timedelta(days=21)).date(),
        )

        if sched.empty:
            result = (None, None, None, "empty market schedule")
            MARKET_WINDOW_CACHE[cal_name] = result
            return result

        next_opens = sched["market_open"][sched["market_open"] > now_utc]

        if next_opens.empty:
            result = (None, None, None, "could not identify next market open")
            MARKET_WINDOW_CACHE[cal_name] = result
            return result

        window_end = next_opens.iloc[0]

        prior_closes = sched["market_close"][sched["market_close"] < window_end]

        if prior_closes.empty:
            result = (None, None, None, "could not identify close before next market open")
            MARKET_WINDOW_CACHE[cal_name] = result
            return result

        window_start = prior_closes.iloc[-1]

        result = (window_start, window_end, local_tz_name, "ok")
        MARKET_WINDOW_CACHE[cal_name] = result
        return result

    except Exception as e:
        result = (None, None, None, f"calendar failed: {e}")
        MARKET_WINDOW_CACHE[cal_name] = result
        return result


def build_earnings_event_interval(row, local_tz_name: str):
    if "Next Earnings Date" not in row.index or "Earnings Timing" not in row.index:
        return None, None, "", "missing earnings columns"

    raw_date = row.get("Next Earnings Date")
    raw_timing = row.get("Earnings Timing")
    timing = normalize_earnings_timing(raw_timing)

    if pd.isna(raw_date) or str(raw_date).strip() == "":
        return None, None, timing, "no earnings date"

    try:
        earnings_date = pd.to_datetime(raw_date).date()
    except Exception:
        return None, None, timing, f"bad earnings date: {raw_date}"

    if timing == "After Market Close":
        start_time = dt.time(16, 0, 0)
        end_time = dt.time(23, 59, 59)

    elif timing == "Before Market Open":
        start_time = dt.time(0, 0, 0)
        end_time = dt.time(9, 30, 0)

    elif timing == "Unknown":
        start_time = dt.time(0, 0, 0)
        end_time = dt.time(23, 59, 59)

    else:
        return None, None, timing, f"unknown earnings timing: {raw_timing}"

    try:
        event_start_local = pd.Timestamp(
            dt.datetime.combine(earnings_date, start_time)
        ).tz_localize(local_tz_name)

        event_end_local = pd.Timestamp(
            dt.datetime.combine(earnings_date, end_time)
        ).tz_localize(local_tz_name)

        return (
            event_start_local.tz_convert("UTC"),
            event_end_local.tz_convert("UTC"),
            timing,
            "ok",
        )
    except Exception as e:
        return None, None, timing, f"could not build earnings interval: {e}"


def earnings_between_last_close_and_next_open(row):
    """
    Return:
        inside: bool
        status: str
        debug: dict

    Override condition:
        event_start < window_end and event_end > window_start
    """
    debug = {
        "Earnings Window Start": "",
        "Earnings Window End": "",
        "Earnings Event Start": "",
        "Earnings Event End": "",
        "Earnings Timing Normalized": "",
        "Earnings Window Overlap": "False",
    }

    if "Next Earnings Date" not in row.index or "Earnings Timing" not in row.index:
        return False, "missing earnings columns", debug

    raw_date = row.get("Next Earnings Date")

    if pd.isna(raw_date) or str(raw_date).strip() == "":
        return False, "no earnings date", debug

    primary_exchange = row.get("Primary Exchange", "")

    window_start, window_end, local_tz_name, cal_status = get_overnight_market_window(primary_exchange)

    if window_start is None or window_end is None or local_tz_name is None:
        return False, cal_status, debug

    event_start, event_end, timing, event_status = build_earnings_event_interval(row, local_tz_name)

    debug["Earnings Window Start"] = str(window_start.tz_convert(local_tz_name))
    debug["Earnings Window End"] = str(window_end.tz_convert(local_tz_name))
    debug["Earnings Timing Normalized"] = timing

    if event_start is None or event_end is None:
        return False, event_status, debug

    debug["Earnings Event Start"] = str(event_start.tz_convert(local_tz_name))
    debug["Earnings Event End"] = str(event_end.tz_convert(local_tz_name))

    overlaps = event_start < window_end and event_end > window_start
    debug["Earnings Window Overlap"] = str(bool(overlaps))

    status = (
        f"{'inside' if overlaps else 'outside'} overnight window; "
        f"window_start={debug['Earnings Window Start']}; "
        f"window_end={debug['Earnings Window End']}; "
        f"event_start={debug['Earnings Event Start']}; "
        f"event_end={debug['Earnings Event End']}; "
        f"timing={timing}; "
        f"event_status={event_status}"
    )

    return overlaps, status, debug


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

        model = (
            bs_call_price(spot, r.strike, T, RISK_FREE_RATE, r.mx_iv)
            if r.kind == "call"
            else bs_put_price(spot, r.strike, T, RISK_FREE_RATE, r.mx_iv)
        )

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


def apply_earnings_override(result, hist, earnings_inside):
    result.override_iv = None
    result.override_used = False
    result.override_status = "no earnings override"

    if not earnings_inside:
        return result

    result.override_used = True

    if result.forward_iv is not None:
        result.override_iv = result.forward_iv
        result.source = result.source + "_forward_earnings_override"
        result.override_status = "earnings override used forward IV"
        return result

    hv = annualized_hist_vol(hist)

    if hv is not None:
        result.override_iv = hv
        result.source = "historical_earnings_override"
        result.override_status = (
            "earnings override used historical volatility because forward IV unavailable"
        )

        if not result.hv_status or result.hv_status == "not attempted":
            result.hv_status = "ok"

        return result

    result.override_iv = None
    result.override_status = (
        "earnings override required but forward IV and historical volatility unavailable"
    )

    if not result.hv_status or result.hv_status == "not attempted":
        result.hv_status = "failed"

    return result


def attach_earnings_debug(result, earnings_debug):
    result.earnings_window_start = earnings_debug.get("Earnings Window Start", "")
    result.earnings_window_end = earnings_debug.get("Earnings Window End", "")
    result.earnings_event_start = earnings_debug.get("Earnings Event Start", "")
    result.earnings_event_end = earnings_debug.get("Earnings Event End", "")
    result.earnings_timing_normalized = earnings_debug.get("Earnings Timing Normalized", "")
    result.earnings_window_overlap = earnings_debug.get("Earnings Window Overlap", "False")
    return result


def resolve(symbol, row):
    result = IVResult()

    try:
        spot, hist, spot_status = get_spot(symbol)
        result.yf_status = f"spot: {spot_status}"
    except Exception as e:
        spot, hist = None, None
        result.yf_status = f"spot lookup failed: {e}"

    if "Next Earnings Date" in row.index and "Earnings Timing" in row.index:
        earnings_inside, earnings_status, earnings_debug = earnings_between_last_close_and_next_open(row)
    else:
        earnings_inside = False
        earnings_status = "no earnings input file"
        earnings_debug = {
            "Earnings Window Start": "",
            "Earnings Window End": "",
            "Earnings Event Start": "",
            "Earnings Event End": "",
            "Earnings Timing Normalized": "",
            "Earnings Window Overlap": "False",
        }

    attach_earnings_debug(result, earnings_debug)

    try:
        if spot is not None and spot > 0:
            iv, expiry, fiv, fexp, status = yf_weighted_iv(symbol, spot)
            result.yf_status = status + f"; earnings={earnings_status}"

            if iv is not None:
                result.iv = iv
                result.expiry = expiry or ""
                result.forward_iv = fiv
                result.forward_expiry = fexp or ""
                result.source = "yfinance_weighted"
                result.final_status = "ok"
                result.mx_status = "not attempted"
                result.hv_status = "not attempted"
                return apply_earnings_override(result, hist, earnings_inside)
        else:
            result.yf_status = "no usable yfinance spot" + f"; earnings={earnings_status}"
    except Exception as e:
        result.yf_status = f"yfinance failed: {e}; earnings={earnings_status}"

    try:
        iv, expiry, fiv, fexp, status = mx_weighted_iv(symbol, spot)
        result.mx_status = status + f"; earnings={earnings_status}"

        if iv is not None:
            result.iv = iv
            result.expiry = expiry or ""
            result.forward_iv = fiv
            result.forward_expiry = fexp or ""
            result.source = "mx_weighted"
            result.final_status = "ok"
            result.hv_status = "not attempted"
            return apply_earnings_override(result, hist, earnings_inside)
    except Exception as e:
        result.mx_status = f"mx failed: {e}; earnings={earnings_status}"

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
            return apply_earnings_override(result, hist, earnings_inside)
    except Exception as e:
        result.hv_status = f"hv failed: {e}"

    result.final_status = "all methods failed"
    return apply_earnings_override(result, hist, earnings_inside)


def reorder_columns(df):
    desired_order = [
        "Name",
        "Symbol",
        "Implied Volatility",
        "Expiry Date",
        "IV Source",
        "Final Status",
        "YF Status",
        "MX Status",
        "HV Status",
    ]

    for col in desired_order:
        if col not in df.columns:
            df[col] = ""

    return df[desired_order]


def main():
    df = clean_df(INPUT_FILE)

    for i, row in df.iterrows():
        raw_symbol = row.get("Symbol", "")

        if pd.isna(raw_symbol) or str(raw_symbol).strip() == "":
            df.at[i, "Implied Volatility"] = np.nan
            df.at[i, "Forward Implied Volatility"] = np.nan
            df.at[i, "Earnings Override Volatility"] = np.nan
            df.at[i, "Expiry Date"] = ""
            df.at[i, "Forward Expiry Date"] = ""
            df.at[i, "Earnings Override Used"] = "False"
            df.at[i, "Earnings Override Status"] = "blank symbol"
            df.at[i, "Earnings Window Start"] = ""
            df.at[i, "Earnings Window End"] = ""
            df.at[i, "Earnings Event Start"] = ""
            df.at[i, "Earnings Event End"] = ""
            df.at[i, "Earnings Timing Normalized"] = ""
            df.at[i, "Earnings Window Overlap"] = "False"
            df.at[i, "IV Source"] = ""
            df.at[i, "Final Status"] = "blank symbol"
            df.at[i, "YF Status"] = "not attempted"
            df.at[i, "MX Status"] = "not attempted"
            df.at[i, "HV Status"] = "not attempted"
            print("blank symbol")
            continue

        symbol = str(raw_symbol).strip().upper()

        try:
            r = resolve(symbol, row)
        except Exception as e:
            r = IVResult(final_status=f"unexpected failure: {e}")

        output_iv = r.override_iv if r.override_iv is not None else r.iv

        df.at[i, "Implied Volatility"] = output_iv if output_iv is not None else np.nan
        df.at[i, "Expiry Date"] = r.expiry
        df.at[i, "IV Source"] = r.source
        df.at[i, "Final Status"] = r.final_status
        df.at[i, "YF Status"] = r.yf_status
        df.at[i, "MX Status"] = r.mx_status
        df.at[i, "HV Status"] = r.hv_status

        print(
            f"{symbol}: IV={r.iv} expiry={r.expiry} "
            f"forward_IV={r.forward_iv} forward_expiry={r.forward_expiry} "
            f"override_IV={r.override_iv} override_used={r.override_used} "
            f"overlap={r.earnings_window_overlap} "
            f"source={r.source} status={r.final_status}"
        )

    df = reorder_columns(df)

    df = df.sort_values(
        by="Implied Volatility",
        ascending=False,
        na_position="last",
    )

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nWrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
