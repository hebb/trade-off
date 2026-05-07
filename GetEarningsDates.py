#!/usr/bin/env python3

import argparse
import os
import time
from datetime import date, datetime, timedelta

import pandas as pd
import requests
import yfinance as yf


FINNHUB_EARNINGS_URL = "https://finnhub.io/api/v1/calendar/earnings"

TIMING_MAP = {
    "bmo": "Before Market Open",
    "amc": "After Market Close",
    "dmh": "During Trading Hours",
}

DROP_OUTPUT_COLUMNS = ["Alternative Symbol"]


def normalize_symbol(symbol):
    return str(symbol).strip().upper()


def to_datetime(value):
    """
    Convert Finnhub/yfinance date values into a Python datetime where possible.
    Handles Timestamp, datetime, date, list, tuple, Series, Index, NumPy arrays,
    DatetimeIndex, and strings.
    """
    if value is None:
        return None

    if isinstance(value, (list, tuple, pd.Series, pd.Index)):
        if len(value) == 0:
            return None
        value = value[0]

    # Handles NumPy arrays without requiring an explicit numpy import.
    if hasattr(value, "shape") and hasattr(value, "__len__") and not isinstance(value, str):
        if len(value) == 0:
            return None
        value = value[0]

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.to_pydatetime()

    if isinstance(value, datetime):
        return value

    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())

    try:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return None

        if isinstance(parsed, pd.DatetimeIndex):
            if len(parsed) == 0:
                return None
            parsed = parsed[0]

        return parsed.to_pydatetime()
    except Exception:
        return None


def format_date(dt):
    if dt is None or pd.isna(dt):
        return ""
    return dt.strftime("%Y-%m-%d")


def is_successful_finnhub_row(row):
    """
    A cached row counts as successful if Finnhub supplied a usable earnings date.
    Timing may still be Unknown if Finnhub had no hour value, but the Finnhub poll
    itself was successful.
    """
    source = str(row.get("Earnings Source", "")).strip().lower()
    status = str(row.get("Earnings Status", "")).strip()
    earnings_date = str(row.get("Next Earnings Date", "")).strip()

    return (
        source == "finnhub"
        and earnings_date != ""
        and status.startswith("OK")
    )


def load_cached_finnhub_results(output_path):
    """
    Load successful Finnhub results from the existing output file, if it exists.
    """
    if not os.path.exists(output_path):
        return {}

    try:
        old = pd.read_csv(output_path, dtype=str).fillna("")
    except Exception as e:
        print(f"Warning: could not read existing output file for cache: {e}")
        return {}

    required = {
        "Symbol",
        "Next Earnings Date",
        "Earnings Timing",
        "Earnings Source",
        "Earnings Status",
    }

    if not required.issubset(set(old.columns)):
        print("Existing output file does not have the required cache columns; ignoring cache.")
        return {}

    cache = {}

    for _, row in old.iterrows():
        symbol = normalize_symbol(row.get("Symbol", ""))
        if not symbol:
            continue

        if is_successful_finnhub_row(row):
            cache[symbol] = {
                "date": str(row.get("Next Earnings Date", "")).strip(),
                "timing": str(row.get("Earnings Timing", "")).strip() or "Unknown",
                "source": "finnhub",
                "status": str(row.get("Earnings Status", "")).strip(),
            }

    return cache


def get_next_earnings_from_finnhub(
    symbol,
    api_key,
    days_ahead,
    max_429_retries,
    retry_sleep,
):
    """
    Query Finnhub for one symbol.

    Finnhub's 'hour' field:
        bmo = before market open
        amc = after market close
        dmh = during market hours
    """
    if not api_key:
        return {
            "date": None,
            "timing": "Unknown",
            "source": "",
            "status": "missing FINNHUB_API_KEY",
        }

    today = date.today()
    end_date = today + timedelta(days=days_ahead)

    params = {
        "symbol": symbol,
        "from": today.strftime("%Y-%m-%d"),
        "to": end_date.strftime("%Y-%m-%d"),
        "token": api_key,
    }

    attempt = 0

    while True:
        try:
            response = requests.get(FINNHUB_EARNINGS_URL, params=params, timeout=20)

            if response.status_code == 429:
                attempt += 1
                if attempt > max_429_retries:
                    return {
                        "date": None,
                        "timing": "Unknown",
                        "source": "",
                        "status": "finnhub HTTP 429: rate limit exceeded",
                    }

                print(f"  Finnhub 429 for {symbol}; sleeping {retry_sleep} seconds...")
                time.sleep(retry_sleep)
                continue

            if response.status_code != 200:
                return {
                    "date": None,
                    "timing": "Unknown",
                    "source": "",
                    "status": f"finnhub HTTP error: {response.status_code}",
                }

            data = response.json()
            break

        except Exception as e:
            return {
                "date": None,
                "timing": "Unknown",
                "source": "",
                "status": f"finnhub error: {type(e).__name__}: {e}",
            }

    earnings = data.get("earningsCalendar", [])
    if not earnings:
        return {
            "date": None,
            "timing": "Unknown",
            "source": "",
            "status": "finnhub no future earnings found",
        }

    rows = []

    for item in earnings:
        dt = to_datetime(item.get("date"))
        if dt is None:
            continue

        raw_hour = str(item.get("hour", "")).strip().lower()
        timing = TIMING_MAP.get(raw_hour, "Unknown")

        status = "OK"
        if raw_hour == "":
            status = "OK; Finnhub date found but hour missing"
        elif timing == "Unknown":
            status = f"OK; unknown Finnhub hour value: {raw_hour}"

        rows.append(
            {
                "date": dt,
                "timing": timing,
                "source": "finnhub",
                "status": status,
            }
        )

    if not rows:
        return {
            "date": None,
            "timing": "Unknown",
            "source": "",
            "status": "finnhub returned no usable earnings date",
        }

    rows.sort(key=lambda r: r["date"])
    return rows[0]


def get_from_yfinance_calendar(ticker):
    try:
        cal = ticker.calendar
    except Exception as e:
        return None, f"yfinance calendar failed: {e}"

    if cal is None:
        return None, "yfinance calendar empty"

    if isinstance(cal, pd.DataFrame):
        if cal.empty:
            return None, "yfinance calendar empty"

        possible_labels = [
            "Earnings Date",
            "Earnings",
            "Earnings Date Start",
            "Earnings Date End",
        ]

        for label in possible_labels:
            if label in cal.index:
                values = cal.loc[label].dropna().tolist()
                if values:
                    return to_datetime(values[0]), "yfinance_calendar"

        return None, "yfinance calendar has no earnings date"

    if isinstance(cal, dict):
        possible_keys = [
            "Earnings Date",
            "Earnings",
            "earningsDate",
            "earnings_date",
        ]

        for key in possible_keys:
            if key in cal:
                dt = to_datetime(cal[key])
                if dt is not None:
                    return dt, "yfinance_calendar"

        return None, "yfinance calendar has no earnings date"

    return None, f"yfinance calendar unexpected type: {type(cal).__name__}"


def get_from_yfinance_earnings_dates(ticker):
    try:
        ed = ticker.get_earnings_dates(limit=12)
    except Exception as e:
        return None, f"yfinance earnings_dates failed: {e}"

    if ed is None or len(ed) == 0:
        return None, "yfinance earnings_dates empty"

    if not isinstance(ed.index, pd.DatetimeIndex):
        return None, "yfinance earnings_dates index is not datetime"

    now = pd.Timestamp.now(tz=ed.index.tz) if ed.index.tz is not None else pd.Timestamp.now()

    future = ed[ed.index >= now.normalize()]
    if future.empty:
        return None, "yfinance earnings_dates has no future dates"

    return future.index[0].to_pydatetime(), "yfinance_earnings_dates"


def get_next_earnings_from_yfinance(symbol):
    """
    yfinance fallback. It may provide a date, but not a reliable explicit
    before-market / after-market / during-market field.
    """
    ticker = yf.Ticker(symbol)

    dt, source_or_status = get_from_yfinance_calendar(ticker)
    if dt is not None:
        return {
            "date": dt,
            "timing": "Unknown",
            "source": source_or_status,
            "status": "OK; timing not explicitly supplied",
        }

    calendar_status = source_or_status

    dt, source_or_status = get_from_yfinance_earnings_dates(ticker)
    if dt is not None:
        return {
            "date": dt,
            "timing": "Unknown",
            "source": source_or_status,
            "status": "OK; timing not explicitly supplied",
        }

    return {
        "date": None,
        "timing": "Unknown",
        "source": "",
        "status": f"{calendar_status}; {source_or_status}",
    }


def build_output_dataframe(input_df, results):
    """
    Preserve input columns except dropped columns, move Primary Exchange
    immediately after Symbol, insert earnings columns after Primary Exchange,
    and sort by ascending earnings date.
    """
    df = input_df.drop(
        columns=[c for c in DROP_OUTPUT_COLUMNS if c in input_df.columns]
    ).copy()

    if "Symbol" not in df.columns:
        raise ValueError("Input CSV must contain a 'Symbol' column.")

    # Move Primary Exchange immediately after Symbol, if present.
    if "Primary Exchange" in df.columns:
        cols = list(df.columns)
        cols.remove("Primary Exchange")

        symbol_pos = cols.index("Symbol")
        cols.insert(symbol_pos + 1, "Primary Exchange")

        df = df[cols]

    # Insert earnings columns immediately after Symbol, or after Primary Exchange
    # if Primary Exchange is present.
    if "Primary Exchange" in df.columns:
        insert_at = df.columns.get_loc("Primary Exchange") + 1
    else:
        insert_at = df.columns.get_loc("Symbol") + 1

    earnings_df = pd.DataFrame(
        {
            "Next Earnings Date": [
                results[s]["date"] for s in df["Symbol"].astype(str).str.strip()
            ],
            "Earnings Timing": [
                results[s]["timing"] for s in df["Symbol"].astype(str).str.strip()
            ],
            "Earnings Source": [
                results[s]["source"] for s in df["Symbol"].astype(str).str.strip()
            ],
            "Earnings Status": [
                results[s]["status"] for s in df["Symbol"].astype(str).str.strip()
            ],
        }
    )

    out = pd.concat(
        [
            df.iloc[:, :insert_at],
            earnings_df,
            df.iloc[:, insert_at:],
        ],
        axis=1,
    )

    out["_Sort Earnings Date"] = pd.to_datetime(
        out["Next Earnings Date"],
        errors="coerce",
    )

    out = out.sort_values(
        by=["_Sort Earnings Date", "Symbol"],
        ascending=[True, True],
        na_position="last",
    ).drop(columns=["_Sort Earnings Date"])

    return out

def should_skip_finnhub(symbol):
    """
    Skip Finnhub for symbols that are known to fail under the current account/API access.
    """
    s = str(symbol).strip().upper()
    return s.endswith(".TO")

def main():
    parser = argparse.ArgumentParser(
        description="Get next earnings dates and explicit release timing."
    )
    parser.add_argument("--input", default="StockList.csv", help="Input CSV file.")
    parser.add_argument(
        "--output",
        default="StockList_with_earnings.csv",
        help="Output CSV file.",
    )
    parser.add_argument(
        "--finnhub-key",
        default=os.getenv("FINNHUB_API_KEY", ""),
        help="Finnhub API key. Defaults to FINNHUB_API_KEY environment variable.",
    )
    parser.add_argument(
        "--days-ahead",
        type=int,
        default=180,
        help="How far ahead to search Finnhub's earnings calendar.",
    )
    parser.add_argument(
        "--finnhub-sleep",
        type=float,
        default=1.2,
        help="Delay between Finnhub ticker requests, in seconds.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=65.0,
        help="Delay after a Finnhub 429 response, in seconds.",
    )
    parser.add_argument(
        "--max-429-retries",
        type=int,
        default=2,
        help="Maximum number of retries after Finnhub 429 for each symbol.",
    )
    parser.add_argument(
        "--yfinance-sleep",
        type=float,
        default=0.25,
        help="Delay between yfinance fallback requests, in seconds.",
    )
    parser.add_argument(
        "--no-yfinance",
        action="store_true",
        help="Disable yfinance fallback.",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input, dtype=str).fillna("")

    if "Symbol" not in df.columns:
        raise ValueError("Input CSV must contain a 'Symbol' column.")

    cached = load_cached_finnhub_results(args.output)
    print(f"Loaded {len(cached)} successful cached Finnhub result(s) from {args.output}")

    results = {}

    symbols = df["Symbol"].astype(str).str.strip().tolist()
    total = len(symbols)

    for i, symbol in enumerate(symbols, start=1):
        norm_symbol = normalize_symbol(symbol)

        if not norm_symbol or norm_symbol == "NAN":
            results[symbol] = {
                "date": "",
                "timing": "Unknown",
                "source": "",
                "status": "missing symbol",
            }
            continue

        if norm_symbol in cached:
            print(f"[{i}/{total}] {symbol} - using cached Finnhub result")
            results[symbol] = cached[norm_symbol]
            continue

        if should_skip_finnhub(symbol):
            print(f"[{i}/{total}] {symbol} - skipping Finnhub for .TO symbol")
            finnhub_result = {
                "date": None,
                "timing": "Unknown",
                "source": "",
                "status": "Finnhub skipped for .TO symbol",
            }
        else:
            print(f"[{i}/{total}] {symbol} - polling Finnhub")

            finnhub_result = get_next_earnings_from_finnhub(
                symbol=symbol,
                api_key=args.finnhub_key,
                days_ahead=args.days_ahead,
                max_429_retries=args.max_429_retries,
                retry_sleep=args.retry_sleep,
            )

        if finnhub_result["source"] == "finnhub" and finnhub_result["date"] is not None:
            results[symbol] = {
                "date": format_date(finnhub_result["date"]),
                "timing": finnhub_result["timing"],
                "source": finnhub_result["source"],
                "status": finnhub_result["status"],
            }
            time.sleep(args.finnhub_sleep)
            continue

        finnhub_status = finnhub_result["status"]

        if args.no_yfinance:
            results[symbol] = {
                "date": "",
                "timing": "Unknown",
                "source": "",
                "status": f"Finnhub: {finnhub_status}; yfinance disabled",
            }
            time.sleep(args.finnhub_sleep)
            continue

        print(f"  Finnhub failed for {symbol}: {finnhub_status}")
        print(f"  Trying yfinance fallback for {symbol}")

        yf_result = get_next_earnings_from_yfinance(symbol)

        if yf_result["date"] is not None:
            results[symbol] = {
                "date": format_date(yf_result["date"]),
                "timing": yf_result["timing"],
                "source": yf_result["source"],
                "status": f"{yf_result['status']}; Finnhub fallback reason: {finnhub_status}",
            }
        else:
            results[symbol] = {
                "date": "",
                "timing": "Unknown",
                "source": "",
                "status": f"Finnhub: {finnhub_status}; yfinance: {yf_result['status']}",
            }

        if not should_skip_finnhub(symbol):
            time.sleep(args.finnhub_sleep)

        time.sleep(args.yfinance_sleep)

    out = build_output_dataframe(df, results)
    out.to_csv(args.output, index=False)

    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()