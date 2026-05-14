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


CONFIG_FILE = "config.yaml"

RISK_FREE_RATE = 0.03
HIST_VOL_LOOKBACK_DAYS = 60
REQUEST_TIMEOUT = 20
MAX_STRIKES_PER_SIDE = 4
MIN_EXPIRY_QUALITY = 0.25
MIN_WEIGHT_TO_KEEP = 1e-6
TRADING_DAYS_PER_YEAR = 252

MARKET_SCHEDULE_CACHE = {}

EARNINGS_NONE = "none"
EARNINGS_IMMEDIATE_OVERNIGHT = "immediate_overnight"
EARNINGS_OPEN_TO_OPEN = "open_to_open"
EARNINGS_FOLLOWING_OPEN_TO_EXPIRY = "following_open_to_expiry"


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
class MarketContext:
    cal_name: str
    local_tz_name: str
    schedule: pd.DataFrame
    now_utc: pd.Timestamp
    previous_close: pd.Timestamp
    next_open: pd.Timestamp
    following_open: pd.Timestamp


@dataclass
class IVResult:
    iv: Optional[float] = None
    expiry: str = ""
    forward_iv: Optional[float] = None
    forward_expiry: str = ""
    contest_iv: Optional[float] = None
    contest_expiry: str = ""
    contest_iv_status: str = ""
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


def trading_year_fraction(expiry_date: dt.date, primary_exchange: str, market_context=None) -> Optional[float]:
    trading_days = trading_sessions_to_expiry(expiry_date, primary_exchange, market_context=market_context)

    if trading_days is None or trading_days <= 0:
        return None

    return trading_days / TRADING_DAYS_PER_YEAR


def forward_iv_from_two_expiries(iv1, expiry1, iv2, expiry2, primary_exchange: str, market_context=None):
    t1 = trading_year_fraction(expiry1, primary_exchange, market_context=market_context)
    t2 = trading_year_fraction(expiry2, primary_exchange, market_context=market_context)

    if t1 is None or t2 is None:
        return None

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


def load_config_file(path: str = CONFIG_FILE):
    if not path or not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        return {}

    if path.lower().endswith(".json"):
        data = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                f"Configuration file {path!r} appears to be YAML, but PyYAML is not installed. "
                "Install it with: pip install pyyaml"
            ) from exc
        data = yaml.safe_load(text)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file {path!r} must contain a mapping at the top level.")
    return data


def cfg_get(config, section: str, key: str, default=None):
    sec = config.get(section, {})
    if isinstance(sec, dict) and key in sec:
        return sec[key]
    if key in config:
        return config[key]
    return default


def required_config_value(config, section: str, key: str) -> str:
    value = cfg_get(config, section, key, None)
    if value is None or not str(value).strip():
        raise ValueError(f"config.yaml must define {section}.{key}")
    return str(value)


def get_input_file(config) -> str:
    return required_config_value(config, "files", "stocklist_with_earnings_csv")


def get_output_file(config) -> str:
    return required_config_value(config, "files", "volatility_csv")


def parse_config_date(value, label: str = "date") -> Optional[dt.date]:
    if value is None:
        return None

    if isinstance(value, dt.datetime):
        return value.date()

    if isinstance(value, dt.date):
        return value

    if pd.isna(value) or str(value).strip() == "":
        return None

    try:
        return pd.to_datetime(value).date()
    except Exception as e:
        raise ValueError(f"Could not parse {label} {value!r} as a date.") from e


def get_contest_end_date(config) -> Optional[dt.date]:
    raw = cfg_get(config, "contest", "end_date", None)
    if raw is None:
        raw = cfg_get(config, "contest", "contest_end_date", None)
    if raw is None:
        raw = cfg_get(config, "run", "contest_end_date", None)
    if raw is None:
        raw = cfg_get(config, "model", "contest_end_date", None)
    return parse_config_date(raw, "contest end date")


def clean_df(path):
    df = pd.read_csv(path)

    if "Symbol" not in df.columns:
        raise ValueError("Input file must contain a Symbol column.")

    if "Primary Exchange" in df.columns:
        df["Primary Exchange"] = df["Primary Exchange"].astype("object")

    numeric_cols = [
        "Implied Volatility",
        "Contest Implied Volatility",
    ]

    text_cols = [
        "Expiry Date",
        "Contest Expiry Date",
        "Contest IV Status",
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
        "ARCA",
        "BATS",
        "CBOE",
        "CBOE BZX",
        "CBOE BZX EXCHANGE",
        "CBOE BYX",
        "CBOE EDGA",
        "CBOE EDGX",
        "CBOE GLOBAL MARKETS",
        "IEX",
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

    if s in {
        "during market hours",
        "during trading hours",
        "during market",
        "market hours",
        "trading hours",
        "dmh",
    }:
        return "During Trading Hours"

    return str(x or "").strip()


def to_utc_timestamp(value=None) -> pd.Timestamp:
    ts = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)

    if ts.tzinfo is None:
        return ts.tz_localize("UTC")

    return ts.tz_convert("UTC")


def empty_earnings_debug():
    return {
        "Earnings Window Start": "",
        "Earnings Window End": "",
        "Earnings Event Start": "",
        "Earnings Event End": "",
        "Earnings Timing Normalized": "",
        "Earnings Window Overlap": "False",
    }


def get_exchange_schedule(cal_name: str, start_date, end_date):
    key = (cal_name, str(start_date), str(end_date))

    if key in MARKET_SCHEDULE_CACHE:
        return MARKET_SCHEDULE_CACHE[key]

    cal = mcal.get_calendar(cal_name)
    sched = cal.schedule(start_date=start_date, end_date=end_date)
    MARKET_SCHEDULE_CACHE[key] = sched
    return sched


def get_market_context(primary_exchange: str, now_utc=None):
    """
    Build the exchange-session boundaries used for trading-day IV math and
    earnings-window classification.

    The returned timestamps are timezone-aware UTC pandas Timestamps.
    """
    if mcal is None:
        return None, "pandas_market_calendars not installed"

    cal_name = exchange_calendar_name(primary_exchange)
    local_tz_name = exchange_timezone_name(primary_exchange)

    if cal_name is None or local_tz_name is None:
        return None, f"unknown exchange: {primary_exchange}"

    try:
        now_utc = to_utc_timestamp(now_utc)

        sched = get_exchange_schedule(
            cal_name,
            start_date=(now_utc - pd.Timedelta(days=14)).date(),
            end_date=(now_utc + pd.Timedelta(days=450)).date(),
        )

        if sched.empty:
            return None, "empty market schedule"

        next_opens = sched["market_open"][sched["market_open"] > now_utc]

        if len(next_opens) < 2:
            return None, "could not identify next and following market opens"

        next_open = next_opens.iloc[0]
        following_open = next_opens.iloc[1]

        prior_closes = sched["market_close"][sched["market_close"] < next_open]

        if prior_closes.empty:
            return None, "could not identify close before next market open"

        previous_close = prior_closes.iloc[-1]

        return (
            MarketContext(
                cal_name=cal_name,
                local_tz_name=local_tz_name,
                schedule=sched,
                now_utc=now_utc,
                previous_close=previous_close,
                next_open=next_open,
                following_open=following_open,
            ),
            "ok",
        )

    except Exception as e:
        return None, f"calendar failed: {e}"


def expiry_close_for_context(expiry_date: dt.date, market_context: MarketContext) -> Optional[pd.Timestamp]:
    expiry_key = pd.Timestamp(expiry_date)

    if expiry_key not in market_context.schedule.index:
        return None

    return market_context.schedule.loc[expiry_key, "market_close"]


def is_usable_expiry(expiry_date: dt.date, market_context: MarketContext) -> bool:
    expiry_close = expiry_close_for_context(expiry_date, market_context)
    return expiry_close is not None and expiry_close > market_context.following_open


def trading_sessions_to_expiry(expiry_date: dt.date, primary_exchange: str, market_context=None) -> Optional[int]:
    market_context = market_context or get_market_context(primary_exchange)[0]

    if market_context is None:
        return None

    expiry_close = expiry_close_for_context(expiry_date, market_context)

    if expiry_close is None or expiry_close <= market_context.now_utc:
        return None

    closes = market_context.schedule["market_close"]
    count = int(((closes > market_context.now_utc) & (closes <= expiry_close)).sum())

    return count if count > 0 else None


def overlaps(a_start, a_end, b_start, b_end) -> bool:
    return a_start < b_end and a_end > b_start


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

    elif timing == "During Trading Hours":
        start_time = dt.time(9, 30, 0)
        end_time = dt.time(16, 0, 0)

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


def classify_earnings_override(row, primary_exchange: str, expiry_date, market_context=None):
    """
    Return:
        kind: one of the EARNINGS_* constants
        status: str
        debug: dict
        trading_days_to_expiry: Optional[int]
    """
    debug = empty_earnings_debug()

    if "Next Earnings Date" not in row.index or "Earnings Timing" not in row.index:
        return EARNINGS_NONE, "missing earnings columns", debug, None

    raw_date = row.get("Next Earnings Date")

    if pd.isna(raw_date) or str(raw_date).strip() == "":
        return EARNINGS_NONE, "no earnings date", debug, None

    if not expiry_date:
        return EARNINGS_NONE, "no first usable expiry for earnings override", debug, None

    if isinstance(expiry_date, str):
        try:
            expiry_date = dt.datetime.strptime(expiry_date, "%Y-%m-%d").date()
        except Exception:
            return EARNINGS_NONE, f"bad expiry date for earnings override: {expiry_date}", debug, None

    if market_context is None:
        market_context, cal_status = get_market_context(primary_exchange)
    else:
        cal_status = "ok"

    if market_context is None:
        return EARNINGS_NONE, cal_status, debug, None

    expiry_close = expiry_close_for_context(expiry_date, market_context)

    if expiry_close is None:
        return EARNINGS_NONE, f"expiry is not an exchange session: {expiry_date}", debug, None

    trading_days = trading_sessions_to_expiry(
        expiry_date,
        primary_exchange,
        market_context=market_context,
    )

    if trading_days is None:
        return EARNINGS_NONE, "no trading sessions remain to first usable expiry", debug, None

    event_start, event_end, timing, event_status = build_earnings_event_interval(
        row,
        market_context.local_tz_name,
    )

    debug["Earnings Window Start"] = str(
        market_context.previous_close.tz_convert(market_context.local_tz_name)
    )
    debug["Earnings Window End"] = str(expiry_close.tz_convert(market_context.local_tz_name))
    debug["Earnings Timing Normalized"] = timing

    if event_start is None or event_end is None:
        return EARNINGS_NONE, event_status, debug, trading_days

    debug["Earnings Event Start"] = str(event_start.tz_convert(market_context.local_tz_name))
    debug["Earnings Event End"] = str(event_end.tz_convert(market_context.local_tz_name))

    immediate_overlap = overlaps(
        event_start,
        event_end,
        market_context.previous_close,
        market_context.next_open,
    )
    open_to_open_overlap = overlaps(
        event_start,
        event_end,
        market_context.next_open,
        market_context.following_open,
    )
    following_to_expiry_overlap = overlaps(
        event_start,
        event_end,
        market_context.following_open,
        expiry_close,
    )

    if immediate_overlap:
        kind = EARNINGS_IMMEDIATE_OVERNIGHT
    elif open_to_open_overlap:
        kind = EARNINGS_OPEN_TO_OPEN
    elif following_to_expiry_overlap:
        kind = EARNINGS_FOLLOWING_OPEN_TO_EXPIRY
    else:
        kind = EARNINGS_NONE

    debug["Earnings Window Overlap"] = str(kind != EARNINGS_NONE)

    status = (
        f"{kind}; "
        f"window_start={debug['Earnings Window Start']}; "
        f"window_end={debug['Earnings Window End']}; "
        f"next_open={market_context.next_open.tz_convert(market_context.local_tz_name)}; "
        f"following_open={market_context.following_open.tz_convert(market_context.local_tz_name)}; "
        f"event_start={debug['Earnings Event Start']}; "
        f"event_end={debug['Earnings Event End']}; "
        f"timing={timing}; "
        f"trading_days_to_expiry={trading_days}; "
        f"event_status={event_status}"
    )

    return kind, status, debug, trading_days


def get_yf_expiries(ticker, market_context: MarketContext):
    valid = []

    try:
        expiries = list(ticker.options)
    except Exception:
        return []

    for e in expiries:
        try:
            d = dt.datetime.strptime(str(e), "%Y-%m-%d").date()
            if is_usable_expiry(d, market_context):
                valid.append((d, str(e)))
        except Exception:
            continue

    valid.sort(key=lambda x: x[0])
    return valid


def expiries_closest_to_target(expiries, target_date: dt.date):
    return sorted(
        expiries,
        key=lambda x: (
            abs((x[0] - target_date).days),
            x[0] < target_date,
            x[0],
        ),
    )


def yf_iv_for_expiry(ticker, expiry_date, expiry_str, spot, primary_exchange: str, market_context=None):
    try:
        chain = ticker.option_chain(expiry_str)
        calls = chain.calls.copy()
        puts = chain.puts.copy()
    except Exception:
        return None, 0.0, "chain failed"

    T = trading_year_fraction(expiry_date, primary_exchange, market_context=market_context)

    if T is None:
        return None, 0.0, "no trading sessions to expiry"

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


def yf_contest_iv(symbol, spot, primary_exchange: str, contest_end_date: Optional[dt.date]):
    if contest_end_date is None:
        return None, "", "skipped: contest.end_date not configured"

    if spot is None or spot <= 0:
        return None, "", "skipped: no usable spot"

    market_context, cal_status = get_market_context(primary_exchange)

    if market_context is None:
        return None, "", cal_status

    try:
        ticker = yf.Ticker(symbol)
        expiries = get_yf_expiries(ticker, market_context)
    except Exception as e:
        return None, "", f"yfinance failed: {e}"

    if not expiries:
        return None, "", (
            "no usable yfinance expiries closing after following market open; "
            f"contest_end_date={contest_end_date.isoformat()}; "
            f"following_open={market_context.following_open.tz_convert(market_context.local_tz_name)}"
        )

    statuses = []

    for expiry_date, expiry_str in expiries_closest_to_target(expiries, contest_end_date):
        iv, quality, status = yf_iv_for_expiry(
            ticker,
            expiry_date,
            expiry_str,
            spot,
            primary_exchange,
            market_context=market_context,
        )
        statuses.append(f"{expiry_str}: {status}")

        if iv is not None:
            return iv, expiry_str, (
                f"ok; contest_end_date={contest_end_date.isoformat()}; "
                f"selected={expiry_str}; source=yfinance; {status}"
            )

    return None, "", (
        f"no usable yfinance contest IV; contest_end_date={contest_end_date.isoformat()}; "
        + " | ".join(statuses)
    )


def yf_weighted_iv(symbol, spot, primary_exchange: str):
    market_context, cal_status = get_market_context(primary_exchange)

    if market_context is None:
        return None, None, None, None, cal_status

    try:
        ticker = yf.Ticker(symbol)
        expiries = get_yf_expiries(ticker, market_context)
    except Exception as e:
        return None, None, None, None, f"yfinance failed: {e}"

    if not expiries:
        return None, None, None, None, (
            "no usable yfinance expiries closing after following market open; "
            f"following_open={market_context.following_open.tz_convert(market_context.local_tz_name)}"
        )

    iv_results = []
    statuses = []

    for expiry_date, expiry_str in expiries[:4]:
        iv, quality, status = yf_iv_for_expiry(
            ticker,
            expiry_date,
            expiry_str,
            spot,
            primary_exchange,
            market_context=market_context,
        )
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
        forward_iv = forward_iv_from_two_expiries(
            iv1,
            expiry1_date,
            iv2,
            expiry2_date,
            primary_exchange,
            market_context=market_context,
        )
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


def mx_iv_for_expiry(rows, expiry, spot, primary_exchange: str, market_context=None):
    rows = [r for r in rows if r.expiry == expiry]
    T = trading_year_fraction(expiry, primary_exchange, market_context=market_context)

    if T is None:
        return None, 0.0, "no trading sessions to expiry"

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


def mx_contest_iv(symbol, spot, primary_exchange: str, contest_end_date: Optional[dt.date]):
    if contest_end_date is None:
        return None, "", "skipped: contest.end_date not configured"

    if not symbol.endswith(".TO"):
        return None, "", "skipped: not .TO"

    if spot is None or spot <= 0:
        return None, "", "skipped: no usable spot"

    market_context, cal_status = get_market_context(primary_exchange)

    if market_context is None:
        return None, "", cal_status

    html, root = fetch_mx_html(symbol)

    if not html:
        return None, "", "MX fetch failed"

    rows = parse_mx(html)

    if not rows:
        return None, "", f"no MX rows parsed, root={root}"

    expiries = [
        (e, e.isoformat())
        for e in sorted(set(r.expiry for r in rows if is_usable_expiry(r.expiry, market_context)))
    ]

    if not expiries:
        return None, "", (
            f"no usable MX expiries closing after following market open, root={root}; "
            f"contest_end_date={contest_end_date.isoformat()}; "
            f"following_open={market_context.following_open.tz_convert(market_context.local_tz_name)}"
        )

    statuses = []

    for expiry, expiry_str in expiries_closest_to_target(expiries, contest_end_date):
        iv, quality, status = mx_iv_for_expiry(
            rows,
            expiry,
            spot,
            primary_exchange,
            market_context=market_context,
        )
        statuses.append(f"{expiry_str}: {status}")

        if iv is not None:
            return iv, expiry_str, (
                f"ok; contest_end_date={contest_end_date.isoformat()}; "
                f"selected={expiry_str}; source=mx; root={root}; {status}"
            )

    return None, "", (
        f"no usable MX contest IV, root={root}; contest_end_date={contest_end_date.isoformat()}; "
        + " | ".join(statuses)
    )


def mx_weighted_iv(symbol, spot, primary_exchange: str):
    if not symbol.endswith(".TO"):
        return None, None, None, None, "skipped: not .TO"

    if spot is None or spot <= 0:
        return None, None, None, None, "skipped: no usable spot"

    market_context, cal_status = get_market_context(primary_exchange)

    if market_context is None:
        return None, None, None, None, cal_status

    html, root = fetch_mx_html(symbol)

    if not html:
        return None, None, None, None, "MX fetch failed"

    rows = parse_mx(html)

    if not rows:
        return None, None, None, None, f"no MX rows parsed, root={root}"

    expiries = sorted(set(r.expiry for r in rows if is_usable_expiry(r.expiry, market_context)))

    if not expiries:
        return None, None, None, None, (
            f"no usable MX expiries closing after following market open, root={root}; "
            f"following_open={market_context.following_open.tz_convert(market_context.local_tz_name)}"
        )

    iv_results = []
    statuses = []

    for expiry in expiries[:4]:
        iv, quality, status = mx_iv_for_expiry(
            rows,
            expiry,
            spot,
            primary_exchange,
            market_context=market_context,
        )
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
        forward_iv = forward_iv_from_two_expiries(
            iv1,
            expiry1_date,
            iv2,
            expiry2_date,
            primary_exchange,
            market_context=market_context,
        )
        forward_expiry = expiry2_str if forward_iv is not None else ""

    return iv1, expiry1_str, forward_iv, forward_expiry, f"root={root}; " + " | ".join(statuses)


def earnings_base_vol(result, hist):
    if result.forward_iv is not None:
        return result.forward_iv, "forward"

    hv = annualized_hist_vol(hist)

    if hv is not None:
        if not result.hv_status or result.hv_status == "not attempted":
            result.hv_status = "ok"

        return hv, "historical"

    if not result.hv_status or result.hv_status == "not attempted":
        result.hv_status = "failed"

    return None, ""


def append_override_source(source: str, suffix: str) -> str:
    return f"{source}_{suffix}" if source else suffix


def apply_base_earnings_override(result, hist, override_kind):
    base_iv, base_source = earnings_base_vol(result, hist)

    if base_iv is None:
        result.override_status = (
            f"{override_kind} earnings override required but forward IV and "
            "historical volatility unavailable"
        )
        return result

    result.override_iv = base_iv

    if base_source == "forward":
        result.source = append_override_source(result.source, "forward_earnings_override")
        result.override_status = f"{override_kind} earnings override used forward IV"
    else:
        result.source = append_override_source(result.source, "historical_earnings_override")
        result.override_status = (
            f"{override_kind} earnings override used historical volatility because "
            "forward IV unavailable"
        )

    return result


def apply_open_to_open_earnings_override(result, hist, first_expiry_trading_days):
    base_iv, base_source = earnings_base_vol(result, hist)

    if result.iv is None or first_expiry_trading_days is None or first_expiry_trading_days <= 0:
        result.override_status = "open-to-open earnings override required but first-expiry IV horizon unavailable"
        return result

    if base_iv is None:
        result.override_status = (
            "open-to-open earnings override required but forward IV and "
            "historical volatility unavailable"
        )
        return result

    event_var = (
        result.iv**2 * first_expiry_trading_days
        - base_iv**2 * (first_expiry_trading_days - 1)
    )

    if event_var > 0 and math.isfinite(event_var):
        result.override_iv = math.sqrt(event_var)
        result.source = append_override_source(result.source, "open_to_open_earnings_override")
        result.override_status = (
            "open-to-open earnings override calculated isolated one-trading-day IV "
            f"using {base_source} base volatility"
        )
        return result

    result.override_iv = base_iv
    result.source = append_override_source(
        result.source,
        f"open_to_open_{base_source}_fallback_earnings_override",
    )
    result.override_status = (
        "open-to-open earnings variance was non-positive; "
        f"used {base_source} base volatility"
    )
    return result


def apply_earnings_override(result, hist, override_kind, first_expiry_trading_days=None):
    result.override_iv = None
    result.override_used = False
    result.override_status = "no earnings override"

    if override_kind == EARNINGS_NONE:
        return result

    result.override_used = True

    if override_kind in {EARNINGS_IMMEDIATE_OVERNIGHT, EARNINGS_FOLLOWING_OPEN_TO_EXPIRY}:
        return apply_base_earnings_override(result, hist, override_kind)

    if override_kind == EARNINGS_OPEN_TO_OPEN:
        return apply_open_to_open_earnings_override(result, hist, first_expiry_trading_days)

    result.override_status = f"unknown earnings override kind: {override_kind}"
    return result


def attach_earnings_debug(result, earnings_debug):
    result.earnings_window_start = earnings_debug.get("Earnings Window Start", "")
    result.earnings_window_end = earnings_debug.get("Earnings Window End", "")
    result.earnings_event_start = earnings_debug.get("Earnings Event Start", "")
    result.earnings_event_end = earnings_debug.get("Earnings Event End", "")
    result.earnings_timing_normalized = earnings_debug.get("Earnings Timing Normalized", "")
    result.earnings_window_overlap = earnings_debug.get("Earnings Window Overlap", "False")
    return result


def finalize_with_earnings(result, hist, row, primary_exchange: str, status_attr: str):
    if "Next Earnings Date" in row.index and "Earnings Timing" in row.index:
        override_kind, earnings_status, earnings_debug, trading_days = classify_earnings_override(
            row,
            primary_exchange,
            result.expiry,
        )
    else:
        override_kind = EARNINGS_NONE
        earnings_status = "no earnings input file"
        earnings_debug = empty_earnings_debug()
        trading_days = None

    attach_earnings_debug(result, earnings_debug)

    current_status = getattr(result, status_attr)
    setattr(result, status_attr, current_status + f"; earnings={earnings_status}")

    return apply_earnings_override(result, hist, override_kind, trading_days)


def attach_contest_iv(result, symbol, spot, primary_exchange: str, contest_end_date: Optional[dt.date]):
    result.contest_iv = None
    result.contest_expiry = ""

    if contest_end_date is None:
        result.contest_iv_status = "skipped: contest.end_date not configured"
        return result

    iv, expiry, yf_status = yf_contest_iv(symbol, spot, primary_exchange, contest_end_date)

    if iv is not None:
        result.contest_iv = iv
        result.contest_expiry = expiry or ""
        result.contest_iv_status = yf_status
        return result

    mx_iv, mx_expiry, mx_status = mx_contest_iv(symbol, spot, primary_exchange, contest_end_date)

    if mx_iv is not None:
        result.contest_iv = mx_iv
        result.contest_expiry = mx_expiry or ""
        result.contest_iv_status = f"yfinance: {yf_status}; mx: {mx_status}"
        return result

    result.contest_iv_status = f"yfinance: {yf_status}; mx: {mx_status}"
    return result


def resolve(symbol, row, contest_end_date: Optional[dt.date] = None):
    result = IVResult()
    primary_exchange = row.get("Primary Exchange", "")

    try:
        spot, hist, spot_status = get_spot(symbol)
        result.yf_status = f"spot: {spot_status}"
    except Exception as e:
        spot, hist = None, None
        result.yf_status = f"spot lookup failed: {e}"

    try:
        if spot is not None and spot > 0:
            iv, expiry, fiv, fexp, status = yf_weighted_iv(symbol, spot, primary_exchange)
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
                attach_contest_iv(result, symbol, spot, primary_exchange, contest_end_date)
                return finalize_with_earnings(result, hist, row, primary_exchange, "yf_status")
        else:
            result.yf_status = "no usable yfinance spot"
    except Exception as e:
        result.yf_status = f"yfinance failed: {e}"

    try:
        iv, expiry, fiv, fexp, status = mx_weighted_iv(symbol, spot, primary_exchange)
        result.mx_status = status

        if iv is not None:
            result.iv = iv
            result.expiry = expiry or ""
            result.forward_iv = fiv
            result.forward_expiry = fexp or ""
            result.source = "mx_weighted"
            result.final_status = "ok"
            result.hv_status = "not attempted"
            attach_contest_iv(result, symbol, spot, primary_exchange, contest_end_date)
            return finalize_with_earnings(result, hist, row, primary_exchange, "mx_status")
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
            attach_contest_iv(result, symbol, spot, primary_exchange, contest_end_date)
            return finalize_with_earnings(result, hist, row, primary_exchange, "hv_status")
    except Exception as e:
        result.hv_status = f"hv failed: {e}"

    result.final_status = "all methods failed"
    attach_contest_iv(result, symbol, spot, primary_exchange, contest_end_date)
    attach_earnings_debug(result, empty_earnings_debug())
    return apply_earnings_override(result, hist, EARNINGS_NONE)


def reorder_columns(df):
    desired_order = [
        "Name",
        "Symbol",
        "Implied Volatility",
        "Expiry Date",
        "Contest Implied Volatility",
        "Contest Expiry Date",
        "Contest IV Status",
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
    config = load_config_file(CONFIG_FILE)
    contest_end_date = get_contest_end_date(config)
    input_file = get_input_file(config)
    output_file = get_output_file(config)
    print(
        "contest.end_date="
        f"{contest_end_date.isoformat() if contest_end_date is not None else 'not configured'}"
    )
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")

    df = clean_df(input_file)

    for i, row in df.iterrows():
        raw_symbol = row.get("Symbol", "")

        if pd.isna(raw_symbol) or str(raw_symbol).strip() == "":
            df.at[i, "Implied Volatility"] = np.nan
            df.at[i, "Contest Implied Volatility"] = np.nan
            df.at[i, "Forward Implied Volatility"] = np.nan
            df.at[i, "Earnings Override Volatility"] = np.nan
            df.at[i, "Expiry Date"] = ""
            df.at[i, "Contest Expiry Date"] = ""
            df.at[i, "Contest IV Status"] = "blank symbol"
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
            r = resolve(symbol, row, contest_end_date=contest_end_date)
        except Exception as e:
            r = IVResult(final_status=f"unexpected failure: {e}")

        output_iv = r.override_iv if r.override_iv is not None else r.iv

        df.at[i, "Implied Volatility"] = output_iv if output_iv is not None else np.nan
        df.at[i, "Expiry Date"] = r.expiry
        df.at[i, "Contest Implied Volatility"] = (
            r.contest_iv if r.contest_iv is not None else np.nan
        )
        df.at[i, "Contest Expiry Date"] = r.contest_expiry
        df.at[i, "Contest IV Status"] = r.contest_iv_status
        df.at[i, "IV Source"] = r.source
        df.at[i, "Final Status"] = r.final_status
        df.at[i, "YF Status"] = r.yf_status
        df.at[i, "MX Status"] = r.mx_status
        df.at[i, "HV Status"] = r.hv_status

        print(
            f"{symbol}: IV={r.iv} expiry={r.expiry} "
            f"contest_IV={r.contest_iv} contest_expiry={r.contest_expiry} "
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

    df.to_csv(output_file, index=False)
    print(f"\nWrote {output_file}")


if __name__ == "__main__":
    main()
