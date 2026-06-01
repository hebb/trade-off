#!/usr/bin/env python3
"""
Evaluate a proposed Trade Off contest portfolio.

The script reports:
  - annualized portfolio volatility
  - tracking error versus the non-anchor leaderboard entries tied with or ahead
    of the anchor
  - tracking error versus the entire non-anchor leaderboard
  - simulated probability that the anchor wins with the proposed portfolio

Defaults come from config.yaml when present.  The proposed portfolio can be
passed as a command-line string or read from a small CSV.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from PortfolioOptimizer import (
    build_player_portfolios,
    canonical_ticker,
    choose,
    choose_required,
    ContestConstraints,
    cov_from_corr_vol,
    ensure_corr_and_vol_coverage,
    filter_portfolio_to_corr,
    is_legal_portfolio,
    load_config_file,
    load_corr_matrix,
    load_leaderboard,
    load_symbol_alias_map,
    load_vols,
    moe95,
    simulate_finish_probs,
    tracking_error_portfolios,
    vectorize_port,
)


def parse_date(value: object) -> dt.date:
    if value is None or str(value).strip() == "":
        raise ValueError("Date value is empty.")
    return dt.date.fromisoformat(str(value).strip())


def count_trading_days(as_of: dt.date, end_date: dt.date) -> int:
    """Count exchange sessions after as_of through end_date, inclusive."""
    if end_date <= as_of:
        raise ValueError(f"contest.end_date ({end_date}) must be after as-of date ({as_of}).")

    try:
        import pandas_market_calendars as mcal  # type: ignore

        cal = mcal.get_calendar("NYSE")
        schedule = cal.schedule(
            start_date=(as_of + dt.timedelta(days=1)).isoformat(),
            end_date=end_date.isoformat(),
        )
        return int(len(schedule))
    except Exception:
        days = pd.bdate_range(as_of + dt.timedelta(days=1), end_date)
        return int(len(days))


def resolve_days_remaining(args: argparse.Namespace, config: Dict[str, object]) -> int:
    days_raw = choose(args.days, config, "run", "days_remaining", None)
    if days_raw is not None and str(days_raw).strip() != "":
        days = int(days_raw)
        if days <= 0:
            raise ValueError("--days must be a positive integer.")
        return days

    end_raw = choose(None, config, "contest", "end_date", None)
    if end_raw is None:
        raise ValueError("Pass --days or set contest.end_date in config.yaml.")

    as_of = parse_date(args.as_of) if args.as_of else dt.date.today()
    return count_trading_days(as_of, parse_date(end_raw))


def parse_portfolio_string(text: str, symbol_alias_map: Optional[Dict[str, str]] = None) -> Dict[str, float]:
    text = str(text or "").strip()
    if not text:
        raise ValueError("Portfolio string is empty.")

    parts = [p.strip() for p in text.split(",") if p.strip()]
    keyed = [("=" in p) or (":" in p) for p in parts]
    if any(keyed) and not all(keyed):
        raise ValueError("Use either all weighted entries like AAPL=25,NVDA=20 or all bare tickers, not a mix.")

    if not any(keyed):
        if len(parts) == 0:
            raise ValueError("Portfolio string is empty.")
        weight = 1.0 / len(parts)
        return normalize_portfolio({canonical_ticker(p, symbol_alias_map): weight for p in parts})

    raw_weights: Dict[str, float] = {}
    parsed_items: List[tuple[str, float]] = []
    for part in parts:
        sep = "=" if "=" in part else ":"
        ticker, weight_text = part.split(sep, 1)
        ticker = canonical_ticker(ticker, symbol_alias_map)
        if not ticker:
            raise ValueError(f"Bad portfolio entry {part!r}: ticker is empty.")
        weight_text = weight_text.strip()
        value = parse_portfolio_weight(weight_text)
        parsed_items.append((ticker, value))

    for ticker, weight in parsed_items:
        if weight < 0:
            raise ValueError(f"Negative weight for {ticker}: {weight}")
        raw_weights[ticker] = raw_weights.get(ticker, 0.0) + float(weight)
    return normalize_portfolio(raw_weights)


def parse_portfolio_file(path: str, symbol_alias_map: Optional[Dict[str, str]] = None) -> Dict[str, float]:
    df = pd.read_csv(path)
    ticker_col = next((c for c in ["Ticker", "Symbol", "Holding", "Holdings"] if c in df.columns), None)
    weight_col = next((c for c in ["Weight", "Weights", "Pct", "Percent"] if c in df.columns), None)
    if ticker_col is None or weight_col is None:
        raise ValueError(
            f"{path} must contain ticker and weight columns, e.g. Ticker,Weight. Found: {list(df.columns)}"
        )

    weights: Dict[str, float] = {}
    for _, row in df.iterrows():
        ticker = canonical_ticker(row[ticker_col], symbol_alias_map)
        if not ticker:
            continue
        weight = parse_portfolio_weight(row[weight_col])
        if weight < 0:
            raise ValueError(f"Negative weight for {ticker}: {weight}")
        weights[ticker] = weights.get(ticker, 0.0) + float(weight)
    return normalize_portfolio(weights)


def parse_portfolio_weight(value: object) -> float:
    if pd.isna(value):
        raise ValueError("Cannot parse NaN portfolio weight")
    text = str(value).strip().replace(",", "")
    if not text:
        raise ValueError("Cannot parse empty portfolio weight")
    if text.endswith("%"):
        return float(text[:-1]) / 100.0
    parsed = float(text)
    if parsed > 1.0:
        return parsed / 100.0
    return parsed


def normalize_portfolio(port: Dict[str, float]) -> Dict[str, float]:
    out = {t: float(w) for t, w in port.items() if abs(float(w)) > 1e-12}
    total = sum(out.values())
    if total <= 0:
        raise ValueError("Portfolio weights sum to zero.")
    if total > 1.0 + 1e-8:
        raise ValueError(f"Portfolio weights sum to {total:.6f}; use weights summing to 100% or less.")
    if total < 1.0 - 1e-8:
        out["CASH"] = out.get("CASH", 0.0) + (1.0 - total)
    final_total = sum(out.values())
    for ticker in list(out.keys()):
        out[ticker] /= final_total
    return out


def parse_player_exclude_list(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw_items = value
    else:
        raw_items = str(value).split(",")
    return [str(item).strip() for item in raw_items if str(item).strip()]


def load_candidate_portfolio(args: argparse.Namespace, symbol_alias_map: Dict[str, str]) -> Dict[str, float]:
    if args.portfolio and args.portfolio_file:
        raise ValueError("Pass either --portfolio or --portfolio-file, not both.")
    if args.portfolio_file:
        return parse_portfolio_file(args.portfolio_file, symbol_alias_map)
    if args.portfolio:
        return parse_portfolio_string(args.portfolio, symbol_alias_map)
    raise ValueError('Pass a proposed portfolio with --portfolio "AAPL=25,NVDA=20,..." or --portfolio-file.')


def build_group_benchmark(
    group_players: Iterable[str],
    ports: Dict[str, Dict[str, float]],
    values: Dict[str, float],
    weighting: str,
) -> Dict[str, float]:
    players = list(group_players)
    if not players:
        raise ValueError("Cannot build a benchmark from an empty player group.")

    if weighting == "equal":
        alphas = {p: 1.0 / len(players) for p in players}
    elif weighting == "value":
        total_value = sum(float(values[p]) for p in players)
        if total_value <= 0:
            raise ValueError("Cannot value-weight a benchmark with non-positive total leaderboard value.")
        alphas = {p: float(values[p]) / total_value for p in players}
    else:
        raise ValueError("--benchmark-weighting must be 'equal' or 'value'.")

    bench: Dict[str, float] = {}
    for player in players:
        for ticker, weight in ports[player].items():
            bench[ticker] = bench.get(ticker, 0.0) + alphas[player] * float(weight)

    total = sum(bench.values())
    if total <= 0:
        raise ValueError("Benchmark weights sum to zero.")
    for ticker in list(bench.keys()):
        bench[ticker] /= total
    return bench


def portfolio_volatility(port: Dict[str, float], corr_u: pd.DataFrame, vol_vec: np.ndarray) -> float:
    universe_index = {ticker: i for i, ticker in enumerate(corr_u.index)}
    weights = vectorize_port(port, universe_index)
    sigma = cov_from_corr_vol(corr_u, vol_vec)
    return float(math.sqrt(max(weights @ sigma @ weights, 0.0)))


def format_pct(value: float) -> str:
    return f"{100.0 * value:.3f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a proposed Trade Off portfolio.")
    parser.add_argument("--config", type=str, default=None, help="Configuration file. Default: config.yaml if it exists.")
    parser.add_argument("--portfolio", type=str, default=None, help='Portfolio string, e.g. "AAPL=25,NVDA=20,SHOP=20,GSY=20,DOO=15".')
    parser.add_argument("--portfolio-file", type=str, default=None, help="CSV with Ticker,Weight columns.")
    parser.add_argument("--anchor", type=str, default=None, help="Anchor leaderboard name. Overrides config run.anchor.")
    parser.add_argument("--selected-value", type=float, default=None, help="Anchor starting value if the anchor is not in the leaderboard.")
    parser.add_argument("--leaderboard", type=str, default=None, help="Leaderboard CSV.")
    parser.add_argument("--corr-file", type=str, default=None, help="Correlation matrix CSV.")
    parser.add_argument("--vol-file", type=str, default=None, help="Volatility CSV.")
    parser.add_argument("--stocklist", type=str, default=None, help="Stock list CSV for ticker aliases.")
    parser.add_argument("--vol-source", choices=["short", "medium"], default=None, help="Vols used for volatility, TE, and simulation. Defaults to model.sim_vol_source.")
    parser.add_argument("--days", type=int, default=None, help="Trading days remaining. Overrides contest.end_date.")
    parser.add_argument("--as-of", type=str, default=None, help="YYYY-MM-DD date used when deriving days from contest.end_date. Default: today.")
    parser.add_argument("--sims", type=int, default=None, help="Number of Monte Carlo simulations.")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed.")
    parser.add_argument("--benchmark-weighting", choices=["equal", "value"], default="equal", help="How to aggregate leaderboard portfolios into group benchmarks.")
    parser.add_argument("--benchmark-exclude-players", type=str, default=None, help="Comma-separated leaderboard player names to exclude only from benchmark calculation.")
    parser.add_argument("--skip-win-probability", action="store_true", help="Skip Monte Carlo win-probability calculation.")
    parser.add_argument("--output-csv", type=str, default=None, help="Optional one-row metrics CSV to write.")

    args = parser.parse_args()

    config_path = args.config
    if config_path is None and os.path.exists("config.yaml"):
        config_path = "config.yaml"
    config = load_config_file(config_path) if config_path else {}

    anchor = choose(args.anchor, config, "run", "anchor", None)
    if anchor is None:
        anchor = choose(None, config, "model", "anchor", "ahebb")
    anchor = str(anchor)

    selected_value_raw = choose(args.selected_value, config, "run", "selected_value", None)
    selected_value = None if selected_value_raw is None or str(selected_value_raw).strip() == "" else float(selected_value_raw)

    leaderboard_csv = str(choose_required(args.leaderboard, config, "files", "leaderboard_csv"))
    stocklist_csv = str(choose_required(args.stocklist, config, "files", "stocklist_csv"))
    corr_csv = str(choose_required(args.corr_file, config, "files", "corr_file"))
    volatility_csv = str(choose_required(args.vol_file, config, "files", "volatility_csv"))
    vol_source = str(choose(args.vol_source, config, "model", "sim_vol_source", "medium"))
    benchmark_exclude_players = parse_player_exclude_list(
        choose(args.benchmark_exclude_players, config, "run", "benchmark_exclude_players", "")
    )
    sims: Optional[int] = None
    seed: Optional[int] = None
    days_remaining: Optional[int] = None
    if not args.skip_win_probability:
        sims = int(choose(args.sims, config, "final_simulation", "sims", 400_000))
        seed = int(choose(args.seed, config, "final_simulation", "seed", 0))
        days_remaining = resolve_days_remaining(args, config)

    symbol_alias_map = load_symbol_alias_map(stocklist_csv)
    proposed_port = load_candidate_portfolio(args, symbol_alias_map)

    corr = load_corr_matrix(corr_csv)
    vol_col = "Implied Volatility" if vol_source == "short" else "Contest Implied Volatility"
    vols = load_vols(volatility_csv, symbol_alias_map=symbol_alias_map, volatility_col=vol_col)

    lb = load_leaderboard(leaderboard_csv)
    players, values, ports = build_player_portfolios(lb, symbol_alias_map=symbol_alias_map)
    ports = {p: filter_portfolio_to_corr(ports[p], corr) for p in players}

    if anchor in values:
        anchor_value = float(values[anchor])
    elif selected_value is not None:
        anchor_value = float(selected_value)
        players.append(anchor)
        values[anchor] = anchor_value
        ports[anchor] = {"CASH": 1.0}
    else:
        raise ValueError(f"Anchor {anchor!r} is not in {leaderboard_csv}; pass --selected-value to add it synthetically.")

    all_non_anchor_players = [p for p in players if p != anchor]
    all_ahead_or_tied_players = [p for p in all_non_anchor_players if float(values[p]) >= anchor_value]
    requested_benchmark_exclude_set = set(benchmark_exclude_players)
    unknown_excluded_players = sorted(p for p in requested_benchmark_exclude_set if p not in players)
    if unknown_excluded_players:
        raise ValueError(f"--benchmark-exclude-players contains name(s) not found in the leaderboard: {unknown_excluded_players}")

    benchmark_exclude_set = requested_benchmark_exclude_set.intersection(all_non_anchor_players)
    non_anchor_players = [p for p in all_non_anchor_players if p not in benchmark_exclude_set]
    ahead_or_tied_players = [p for p in non_anchor_players if float(values[p]) >= anchor_value]
    if not all_non_anchor_players:
        raise ValueError("The leaderboard has no non-anchor players to compare against.")
    if not non_anchor_players:
        raise ValueError("Benchmark player exclusion removed every non-anchor player.")
    if not ahead_or_tied_players:
        raise ValueError(
            f"No benchmark-eligible non-anchor players are tied with or ahead of {anchor} "
            f"at value {anchor_value:.2f}."
        )

    proposed_noncash = [ticker for ticker in proposed_port if ticker != "CASH"]
    missing_corr = [ticker for ticker in proposed_noncash if ticker not in corr.index]
    missing_vol = [ticker for ticker in proposed_noncash if ticker not in vols]
    if missing_corr:
        raise ValueError(f"Proposed portfolio ticker(s) missing from correlation matrix: {missing_corr}")
    if missing_vol:
        raise ValueError(f"Proposed portfolio ticker(s) missing from {vol_col}: {missing_vol}")

    proposed_port_for_metrics = filter_portfolio_to_corr(proposed_port, corr)
    ahead_benchmark = build_group_benchmark(ahead_or_tied_players, ports, values, args.benchmark_weighting)
    full_benchmark = build_group_benchmark(non_anchor_players, ports, values, args.benchmark_weighting)

    metric_universe = sorted(
        set(proposed_port_for_metrics)
        | set(ahead_benchmark)
        | set(full_benchmark)
        | {"CASH"}
    )
    corr_u_metrics, vol_vec_metrics = ensure_corr_and_vol_coverage(metric_universe, corr, vols)
    vol = portfolio_volatility(proposed_port_for_metrics, corr_u_metrics, vol_vec_metrics)
    te_ahead = tracking_error_portfolios(proposed_port_for_metrics, ahead_benchmark, corr_u_metrics, vol_vec_metrics)
    te_full = tracking_error_portfolios(proposed_port_for_metrics, full_benchmark, corr_u_metrics, vol_vec_metrics)

    n_used: Optional[int] = None
    p_win: Optional[float] = None
    p_win_moe: Optional[float] = None
    if not args.skip_win_probability:
        if days_remaining is None or sims is None or seed is None:
            raise RuntimeError("Internal error: simulation settings were not resolved.")

        players_sim = list(players)
        values_sim = dict(values)
        ports_sim = dict(ports)
        ports_sim[anchor] = proposed_port_for_metrics

        universe_sim = sorted(set().union(*(set(ports_sim[p]) for p in players_sim), {"CASH"}))
        corr_u_sim, vol_vec_sim = ensure_corr_and_vol_coverage(universe_sim, corr, vols)
        sigma_ann_sim = cov_from_corr_vol(corr_u_sim, vol_vec_sim)
        u_index_sim = {ticker: i for i, ticker in enumerate(universe_sim)}
        weights_by_player = {p: vectorize_port(ports_sim[p], u_index_sim) for p in players_sim}
        w_mat = np.stack([weights_by_player[p] for p in players_sim], axis=0)
        finish, _, n_used = simulate_finish_probs(
            players_sim,
            values_sim,
            w_mat,
            sigma_ann_sim,
            days_remaining,
            n_sims=sims,
            seed=seed,
            return_counts=True,
        )
        p_win = float(finish[anchor]["P1st"])
        p_win_moe = moe95(p_win, n_used)

    constraints = ContestConstraints()
    legal_note = ""
    if not is_legal_portfolio(proposed_port, constraints):
        legal_note = " (not contest-legal under default contest constraints)"

    print(f"Anchor: {anchor} | anchor value: {anchor_value:.2f}")
    if args.skip_win_probability:
        print(f"Vol source: {vol_source} ({vol_col}) | win probability skipped")
    else:
        print(f"Vol source: {vol_source} ({vol_col}) | days: {days_remaining} | sims: {n_used:,}")
    print(f"Benchmark weighting: {args.benchmark_weighting}")
    if benchmark_exclude_set:
        excluded_found = [p for p in players if p in benchmark_exclude_set]
        print(f"Benchmark excludes: {', '.join(excluded_found)}")
    print("\nProposed portfolio:")
    for ticker, weight in sorted(proposed_port.items(), key=lambda item: item[1], reverse=True):
        print(f"  {ticker:8s} {format_pct(weight):>9s}")
    if legal_note:
        print(f"  NOTE:{legal_note}")

    print("\nMetrics:")
    ahead_player_count = f"{len(ahead_or_tied_players)} players"
    non_anchor_player_count = f"{len(non_anchor_players)} players"
    if benchmark_exclude_set:
        ahead_player_count = f"{len(ahead_or_tied_players)} of {len(all_ahead_or_tied_players)} players"
        non_anchor_player_count = f"{len(non_anchor_players)} of {len(all_non_anchor_players)} players"
    print(f"  Annualized volatility:                         {format_pct(vol)}")
    print(
        f"  TE vs tied/ahead non-anchor benchmark ({ahead_player_count}): "
        f"{format_pct(te_ahead)}"
    )
    print(
        f"  TE vs full non-anchor benchmark ({non_anchor_player_count}):      "
        f"{format_pct(te_full)}"
    )
    if benchmark_exclude_set:
        print(f"  Benchmark-only exclusions:                    {len(benchmark_exclude_set)} player(s)")
    if args.skip_win_probability:
        print(f"  Probability {anchor} wins:                     skipped")
    else:
        print(f"  Probability {anchor} wins:                     {format_pct(p_win)} (+/- {format_pct(p_win_moe)} 95% MOE)")

    if args.output_csv:
        row = {
            "Anchor": anchor,
            "AnchorValue": anchor_value,
            "VolSource": vol_source,
            "DaysRemaining": days_remaining,
            "Sims": n_used,
            "BenchmarkWeighting": args.benchmark_weighting,
            "BenchmarkExcludedPlayers": ",".join(p for p in players if p in benchmark_exclude_set),
            "AheadOrTiedPlayers": len(ahead_or_tied_players),
            "LeaderboardAheadOrTiedPlayers": len(all_ahead_or_tied_players),
            "NonAnchorPlayers": len(non_anchor_players),
            "LeaderboardNonAnchorPlayers": len(all_non_anchor_players),
            "AnnualizedVolatility": vol,
            "TrackingErrorAheadOrTiedBenchmark": te_ahead,
            "TrackingErrorFullNonAnchorBenchmark": te_full,
            "AnchorWinProbability": p_win,
            "AnchorWinProbabilityMOE95": p_win_moe,
        }
        pd.DataFrame([row]).to_csv(args.output_csv, index=False)
        print(f"\nWrote: {args.output_csv}")


if __name__ == "__main__":
    main()
