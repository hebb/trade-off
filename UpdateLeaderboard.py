#!/usr/bin/env python3
"""
UpdateLeaderboard.py

Project the current Trade Off leaderboard forward by one trading day.

For each holding, the script fetches the latest two closes from Yahoo Finance,
computes the most recent close-to-close return, applies CAD/USD exchange-rate
movement to USD-quoted holdings, and writes projected portfolio values plus new
rankings.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


CAD_MARKERS = (":CA", ".TO", ".V")
DEFAULT_FX_TICKER = "CAD=X"  # Yahoo's CAD-per-USD quote.


@dataclass
class ResolvedTicker:
    holding: str
    yahoo_symbol: str
    currency: str
    reason: str


@dataclass
class ReturnResult:
    holding: str
    yahoo_symbol: str
    currency: str
    start_date: str
    end_date: str
    start_close: float
    end_close: float
    local_return: float
    fx_return: float
    cad_return: float
    status: str


def canonical_ticker(t: object) -> str:
    if t is None or pd.isna(t):
        return ""
    return str(t).strip().upper()


def strip_canadian_suffix(t: str) -> str:
    t = canonical_ticker(t)
    for suffix in CAD_MARKERS:
        if t.endswith(suffix):
            return t[: -len(suffix)]
    return t


def is_canadian_symbol(t: str) -> bool:
    t = canonical_ticker(t)
    return t.endswith(CAD_MARKERS)


def is_canadian_exchange(exchange: object) -> bool:
    ex = str(exchange or "").strip().upper()
    return ex in {"TSE", "TSX", "TSXV", "TSX-V", "CVE", "NEO"}


def parse_holdings(value: object) -> List[str]:
    if value is None or pd.isna(value):
        return []
    return [canonical_ticker(x) for x in str(value).split(",") if str(x).strip()]


def parse_weights(value: object) -> Optional[List[float]]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None

    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        return None

    try:
        weights = [float(p.rstrip("%")) for p in parts]
    except ValueError:
        return None

    if any("%" in p for p in parts) or max(weights, default=0.0) > 1.5:
        weights = [w / 100.0 for w in weights]
    return weights


def weights_for_row(row: pd.Series) -> Tuple[List[str], np.ndarray, float, str]:
    holdings = parse_holdings(row.get("Holdings", ""))
    raw_weights = parse_weights(row.get("Weights", None))

    if not holdings:
        return holdings, np.array([], dtype=float), 1.0, "no holdings"

    if raw_weights is None or len(raw_weights) != len(holdings):
        weights = np.ones(len(holdings), dtype=float) / len(holdings)
        return holdings, weights, 0.0, "equal weights"

    weights = np.array(raw_weights, dtype=float)
    total = float(weights.sum())
    cash_weight = max(0.0, 1.0 - total)
    if total > 1.0 + 1e-8:
        weights = weights / total
        cash_weight = 0.0
        return holdings, weights, cash_weight, "weights normalized"
    return holdings, weights, cash_weight, "supplied weights"


def parse_value(value: object) -> float:
    if pd.isna(value):
        return float("nan")
    text = str(value).strip().replace("$", "").replace(",", "")
    return float(text)


def load_stock_list(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame(columns=["Symbol", "Alternative Symbol", "Primary Exchange"])
    df = pd.read_csv(path)
    for col in ["Symbol", "Alternative Symbol", "Primary Exchange"]:
        if col not in df.columns:
            df[col] = ""
    return df


def matching_stock_rows(stock_list: pd.DataFrame, holding: str) -> pd.DataFrame:
    if stock_list.empty:
        return stock_list

    holding = canonical_ticker(holding)
    base = strip_canadian_suffix(holding)
    symbol = stock_list["Symbol"].fillna("").map(canonical_ticker)
    alt = stock_list["Alternative Symbol"].fillna("").map(canonical_ticker)
    symbol_base = symbol.map(strip_canadian_suffix)
    alt_base = alt.map(strip_canadian_suffix)

    return stock_list[
        (symbol == holding)
        | (alt == holding)
        | (symbol_base == base)
        | (alt_base == base)
    ]


def add_unique(items: List[str], value: object) -> None:
    text = canonical_ticker(value)
    if text and text != "NAN" and text not in items:
        items.append(text)


def parse_ticker_overrides(values: Optional[List[str]]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    if isinstance(values, str):
        values = [values]
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Bad --ticker-override {item!r}. Use HOLDING=YAHOO_SYMBOL.")
        left, right = item.split("=", 1)
        left = canonical_ticker(left)
        right = canonical_ticker(right)
        if not left or not right:
            raise ValueError(f"Bad --ticker-override {item!r}. Use HOLDING=YAHOO_SYMBOL.")
        overrides[left] = right
    return overrides


def ticker_candidates(holding: str, stock_list: pd.DataFrame, overrides: Optional[Dict[str, str]] = None) -> List[Tuple[str, str, str]]:
    holding = canonical_ticker(holding)
    base = strip_canadian_suffix(holding)
    candidates: List[Tuple[str, str, str]] = []
    overrides = overrides or {}

    def add(symbol: str, currency: str, reason: str) -> None:
        symbol = canonical_ticker(symbol)
        if not symbol or symbol == "NAN":
            return
        item = (symbol, currency, reason)
        if item not in candidates:
            candidates.append(item)

    override = overrides.get(holding)
    if override:
        add(override, "CAD" if is_canadian_symbol(override) else "USD", "ticker override")

    rows = matching_stock_rows(stock_list, holding)
    for _, row in rows.iterrows():
        symbol = canonical_ticker(row.get("Symbol", ""))
        alt = canonical_ticker(row.get("Alternative Symbol", ""))
        canadian = is_canadian_exchange(row.get("Primary Exchange", "")) or is_canadian_symbol(symbol)

        if canadian:
            add(holding if is_canadian_symbol(holding) else f"{base}.TO", "CAD", "stock-list Canadian exchange")
            if alt:
                add(alt if is_canadian_symbol(alt) else f"{strip_canadian_suffix(alt)}.TO", "CAD", "stock-list alternate Canadian")
            if symbol:
                add(symbol if is_canadian_symbol(symbol) else f"{strip_canadian_suffix(symbol)}.TO", "CAD", "stock-list symbol Canadian")
        else:
            if symbol:
                add(symbol, "USD", "stock-list symbol")
            if alt:
                add(alt, "USD", "stock-list alternate")

    add(holding, "CAD" if is_canadian_symbol(holding) else "USD", "leaderboard symbol")
    if not is_canadian_symbol(holding) and rows.empty:
        add(f"{base}.TO", "CAD", "Canadian suffix fallback")
    return candidates


def normalize_download_frame(raw: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        field = "Adj Close" if "Adj Close" in raw.columns.get_level_values(0) else "Close"
        if field in raw.columns.get_level_values(0):
            return raw[field].copy()
        return pd.DataFrame()

    field = "Adj Close" if "Adj Close" in raw.columns else "Close"
    if field not in raw.columns:
        return pd.DataFrame()
    name = symbols[0] if symbols else "Close"
    return raw[[field]].rename(columns={field: name})


def fetch_price_history(symbols: Iterable[str], period: str) -> pd.DataFrame:
    symbols = sorted(set(s for s in symbols if s))
    if not symbols:
        return pd.DataFrame()

    raw = yf.download(
        tickers=symbols,
        period=period,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    closes = normalize_download_frame(raw, symbols)
    closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
    closes = closes.apply(pd.to_numeric, errors="coerce")
    return closes.dropna(axis=1, how="all")


def has_two_closes(closes: pd.DataFrame, symbol: str) -> bool:
    return symbol in closes.columns and closes[symbol].dropna().shape[0] >= 2


def resolve_ticker(holding: str, stock_list: pd.DataFrame, closes: pd.DataFrame, overrides: Optional[Dict[str, str]] = None) -> Optional[ResolvedTicker]:
    for symbol, currency, reason in ticker_candidates(holding, stock_list, overrides):
        if has_two_closes(closes, symbol):
            return ResolvedTicker(holding=holding, yahoo_symbol=symbol, currency=currency, reason=reason)
    return None


def latest_pair(series: pd.Series) -> Tuple[pd.Timestamp, pd.Timestamp, float, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if clean.shape[0] < 2:
        raise ValueError(f"Need at least two closes for {series.name}.")
    start_date, end_date = clean.index[-2], clean.index[-1]
    return start_date, end_date, float(clean.iloc[-2]), float(clean.iloc[-1])


def fx_gross_for_dates(fx: pd.Series, start_date: pd.Timestamp, end_date: pd.Timestamp) -> Tuple[float, str]:
    clean = pd.to_numeric(fx, errors="coerce").dropna().sort_index()
    start = clean[clean.index <= start_date]
    end = clean[clean.index <= end_date]
    if start.empty or end.empty:
        raise ValueError("FX history does not cover the stock return dates.")
    return float(end.iloc[-1] / start.iloc[-1]), clean.name or DEFAULT_FX_TICKER


def compute_holding_returns(
    holdings: Iterable[str],
    stock_list: pd.DataFrame,
    closes: pd.DataFrame,
    fx_ticker: str,
    overrides: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, ReturnResult], List[str]]:
    fx_series = closes[fx_ticker] if fx_ticker in closes.columns else pd.Series(dtype=float, name=fx_ticker)
    results: Dict[str, ReturnResult] = {}
    missing: List[str] = []

    for holding in sorted(set(holdings)):
        resolved = resolve_ticker(holding, stock_list, closes, overrides)
        if resolved is None:
            missing.append(holding)
            continue

        d0, d1, p0, p1 = latest_pair(closes[resolved.yahoo_symbol])
        local_gross = p1 / p0
        fx_gross = 1.0
        if resolved.currency == "USD":
            fx_gross, _ = fx_gross_for_dates(fx_series, d0, d1)

        cad_gross = local_gross * fx_gross
        results[holding] = ReturnResult(
            holding=holding,
            yahoo_symbol=resolved.yahoo_symbol,
            currency=resolved.currency,
            start_date=d0.date().isoformat(),
            end_date=d1.date().isoformat(),
            start_close=p0,
            end_close=p1,
            local_return=local_gross - 1.0,
            fx_return=fx_gross - 1.0,
            cad_return=cad_gross - 1.0,
            status=resolved.reason,
        )

    return results, missing


def rank_values(values: pd.Series) -> pd.Series:
    return values.rank(method="min", ascending=False).astype("Int64")


def project_leaderboard(
    leaderboard: pd.DataFrame,
    returns: Dict[str, ReturnResult],
    missing: List[str],
) -> pd.DataFrame:
    missing_set = set(missing)
    rows = []

    for _, row in leaderboard.iterrows():
        holdings, weights, cash_weight, weight_status = weights_for_row(row)
        value = parse_value(row.get("Value", np.nan))

        missing_for_row = [h for h in holdings if h in missing_set or h not in returns]
        if missing_for_row or not math.isfinite(value):
            portfolio_return = np.nan
            new_value = np.nan
            status = "missing returns: " + ", ".join(missing_for_row) if missing_for_row else "missing value"
        else:
            gross = cash_weight
            for holding, weight in zip(holdings, weights):
                gross += float(weight) * (1.0 + returns[holding].cad_return)
            portfolio_return = gross - 1.0
            new_value = value * gross
            status = weight_status

        rows.append(
            {
                "Name": row.get("Name", ""),
                "PreviousValue": value,
                "PreviousRank": row.get("Rank", np.nan),
                "NewValue": new_value,
                "DailyReturn": portfolio_return,
                "Status": status,
                "Holdings": row.get("Holdings", ""),
                "Weights": row.get("Weights", ""),
            }
        )

    out = pd.DataFrame(rows)
    out["NewRank"] = rank_values(out["NewValue"])
    out["RankChange"] = pd.to_numeric(out["PreviousRank"], errors="coerce") - pd.to_numeric(out["NewRank"], errors="coerce")
    cols = ["Name", "PreviousValue", "PreviousRank", "NewValue", "NewRank", "RankChange", "DailyReturn", "Status", "Holdings", "Weights"]
    return out[cols].sort_values(["NewRank", "NewValue"], ascending=[True, False], na_position="last")


def returns_to_frame(returns: Dict[str, ReturnResult], missing: List[str]) -> pd.DataFrame:
    rows = [r.__dict__ for r in returns.values()]
    for holding in missing:
        rows.append(
            {
                "holding": holding,
                "yahoo_symbol": "",
                "currency": "",
                "start_date": "",
                "end_date": "",
                "start_close": np.nan,
                "end_close": np.nan,
                "local_return": np.nan,
                "fx_return": np.nan,
                "cad_return": np.nan,
                "status": "missing price history",
            }
        )
    return pd.DataFrame(rows).sort_values(["status", "holding"])


def collect_all_candidate_symbols(holdings: Iterable[str], stock_list: pd.DataFrame, fx_ticker: str, overrides: Optional[Dict[str, str]] = None) -> List[str]:
    symbols: List[str] = []
    for holding in holdings:
        for symbol, _, _ in ticker_candidates(holding, stock_list, overrides):
            add_unique(symbols, symbol)
    add_unique(symbols, fx_ticker)
    return symbols


def load_config_file(path: str) -> Dict[str, object]:
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


def cfg_get(config: Dict[str, object], section: str, key: str, default):
    sec = config.get(section, {})
    if isinstance(sec, dict) and key in sec:
        return sec[key]
    if key in config:
        return config[key]
    return default


def choose(cli_value, config: Dict[str, object], section: str, key: str, default):
    return cli_value if cli_value is not None else cfg_get(config, section, key, default)


def choose_required(cli_value, config: Dict[str, object], section: str, key: str):
    value = choose(cli_value, config, section, key, None)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing required config value: {section}.{key}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Project leaderboard values using latest one-day stock and FX returns.")
    parser.add_argument("--config", default=None, help="Configuration file. Default: config.yaml if it exists.")
    parser.add_argument("--leaderboard", default=None, help="Input leaderboard CSV. Defaults to files.leaderboard_csv.")
    parser.add_argument("--stock-list", default=None, help="Stock list used to resolve Canadian tickers. Defaults to files.stocklist_csv.")
    parser.add_argument("--out", default=None, help="Projected leaderboard output CSV. Defaults to files.projected_leaderboard_csv.")
    parser.add_argument("--returns-out", default=None, help="Holding return audit output CSV. Defaults to files.last_day_returns_csv. Use '' to skip.")
    parser.add_argument("--period", default=None, help="Yahoo Finance lookback period for daily closes.")
    parser.add_argument("--fx-ticker", default=None, help="Yahoo ticker for CAD per USD exchange rate.")
    parser.add_argument("--ticker-override", action="append", default=None, help="Resolve a leaderboard holding to a Yahoo ticker, e.g. APPL=AAPL. Can be repeated.")
    parser.add_argument(
        "--allow-missing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write partial output instead of failing on missing holding returns.",
    )
    args = parser.parse_args()

    config_path = args.config
    if config_path is None and os.path.exists("config.yaml"):
        config_path = "config.yaml"
    config = load_config_file(config_path) if config_path else {}

    leaderboard_path = str(choose_required(args.leaderboard, config, "files", "leaderboard_csv"))
    stock_list_path = str(choose_required(args.stock_list, config, "files", "stocklist_csv"))
    out_path = str(choose_required(args.out, config, "files", "projected_leaderboard_csv"))
    returns_out = choose(args.returns_out, config, "files", "last_day_returns_csv", None)
    returns_out = "" if returns_out is None else str(returns_out)
    period = str(choose(args.period, config, "leaderboard_update", "period", "14d"))
    fx_ticker = str(choose(args.fx_ticker, config, "leaderboard_update", "fx_ticker", DEFAULT_FX_TICKER))
    ticker_overrides = choose(args.ticker_override, config, "leaderboard_update", "ticker_overrides", [])
    allow_missing = bool(choose(args.allow_missing, config, "leaderboard_update", "allow_missing", False))

    leaderboard = pd.read_csv(leaderboard_path)
    required = {"Name", "Value", "Holdings"}
    missing_cols = required - set(leaderboard.columns)
    if missing_cols:
        raise ValueError(f"Leaderboard missing columns: {sorted(missing_cols)}")
    if "Rank" not in leaderboard.columns:
        leaderboard["Rank"] = rank_values(leaderboard["Value"].map(parse_value))
    if "Weights" not in leaderboard.columns:
        leaderboard["Weights"] = ""

    stock_list = load_stock_list(stock_list_path)
    overrides = parse_ticker_overrides(ticker_overrides)
    holdings = sorted({h for value in leaderboard["Holdings"] for h in parse_holdings(value)})
    symbols = collect_all_candidate_symbols(holdings, stock_list, fx_ticker, overrides)
    closes = fetch_price_history(symbols, period)

    returns, missing = compute_holding_returns(holdings, stock_list, closes, fx_ticker, overrides)
    if missing and not allow_missing:
        raise ValueError(
            "Missing price history for holdings: "
            + ", ".join(missing)
            + ". Re-run with --allow-missing to write partial results."
        )

    projected = project_leaderboard(leaderboard, returns, missing)
    projected.to_csv(out_path, index=False)

    if returns_out:
        returns_to_frame(returns, missing).to_csv(returns_out, index=False)

    print(f"Wrote: {out_path}")
    if returns_out:
        print(f"Wrote: {returns_out}")
    complete = projected["NewValue"].notna().sum()
    print(f"Projected {complete} of {len(projected)} leaderboard rows.")
    if not projected.empty and projected["NewValue"].notna().any():
        top = projected.dropna(subset=["NewValue"]).iloc[0]
        print(f"Projected leader: {top['Name']} at {top['NewValue']:.2f}")


if __name__ == "__main__":
    main()
