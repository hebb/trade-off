#!/usr/bin/env python3
"""
CombinedContestOptimizer.py

Single contest optimiser with selectable objective:

  --objective te       minimise tracking error to a win-probability-weighted benchmark
  --objective win      maximise the anchor's simulated probability of finishing first
  --objective vol      maximise portfolio volatility
  --objective current  use the anchor portfolio from Leaderboard.csv

Key points:
  - Default correlation file: correlation_matrix.csv
  - CorrelationMatrixDaily.csv and CorrelationMatrixMonthly.csv are not used.
  - --exclude applies to every optimisation objective.
  - --forced applies to every optimisation objective, e.g. --forced "AAPL=25,NVDA=10".
  - The weighted benchmark is calculated only for --objective te.
  - Contest rules: at least 5 stocks; each stock 5%-25%; 1% increments;
    cash allowed up to --max_cash for TE/WIN/current validation; no shorting.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set, Callable

import numpy as np
import pandas as pd


# -----------------------------
# Constants used by the volatility annealer
# -----------------------------

SHORTLIST_SIZE = 160
ELITE_POOL_SIZE = 260
N_STARTS = 160
ANNEAL_STEPS = 2400
LOCAL_PASSES = 25
TABU_LEN = 35
OUTSIDER_CHECKS = 120
SHORTLIST_START_FRAC = 0.45
ELITE_POOL_START_FRAC = 0.35
TEMP_START_MULT = 0.08
TEMP_END_MULT = 0.001
EPS = 1e-15


# -----------------------------
# Configuration loading
# -----------------------------

DEFAULT_CONFIG = {
    "objective": "te",
    "anchor": "ahebb",
    "days": None,
    "exclude": "",
    "forced": "",
    "files": {
        "leaderboard_csv": "Leaderboard.csv",
        "stocklist_csv": "StockList.csv",
        "corr_file": "correlation_matrix.csv",
        "vol_short_csv": "Volatilities.csv",
        "vol_medium_csv": "Volatilities2.csv",
        "winprob_path": "WinProbabilities_excl_ahebb.csv",
    },
    "model": {
        "wp_vol_source": "medium",
        "opt_vol_source": "medium",
        "sim_vol_source": "medium",
        "max_cash": 25,
    },
    "win_objective": {
        "inner_sims": 12_000,
        "rounds": 8,
        "samples": 450,
    },
    "final_simulation": {
        "sims": 400_000,
        "seed": 0,
    },
}


def deep_update(base: dict, updates: dict) -> dict:
    """Recursively merge updates into base and return a new dictionary."""
    out = dict(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config_file(path: Optional[str]) -> dict:
    """Load YAML or JSON config. Missing default config files are ignored."""
    if not path:
        return {}
    if not os.path.exists(path):
        if path == "config.yaml":
            return {}
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        if path.lower().endswith(".json"):
            data = json.load(f)
        else:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError("Reading YAML config files requires PyYAML. Install it or use a JSON config file.") from exc
            data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping at the top level.")
    return data


def cfg_get(cfg: dict, dotted: str, fallback=None):
    """Read nested config values, with a fallback to a flat key of the same final name."""
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            flat_key = dotted.split(".")[-1]
            return cfg.get(flat_key, fallback) if isinstance(cfg, dict) else fallback
        cur = cur[part]
    return cur


# -----------------------------
# Basic helpers
# -----------------------------

def norm_ticker(t: str) -> str:
    t = str(t).strip().upper()
    if t.endswith(":CA"):
        t = t[:-3]
    if t.endswith(".TO"):
        t = t[:-3]
    return t


def canonical_ticker(t: str, symbol_alias_map: Optional[Dict[str, str]] = None) -> str:
    t_norm = norm_ticker(t)
    if symbol_alias_map is None:
        return t_norm
    return symbol_alias_map.get(t_norm, t_norm)


def load_symbol_alias_map(path: str = "StockList.csv") -> Dict[str, str]:
    if not path or not os.path.exists(path):
        return {}

    stocklist = pd.read_csv(path)
    if "Symbol" not in stocklist.columns:
        return {}

    alias_map: Dict[str, str] = {}
    for _, row in stocklist.iterrows():
        sym_raw = row.get("Symbol", "")
        if pd.isna(sym_raw) or not str(sym_raw).strip():
            continue
        canonical = norm_ticker(sym_raw)
        alias_map[canonical] = canonical

        if "Alternative Symbol" in stocklist.columns:
            alt_raw = row.get("Alternative Symbol", "")
            if pd.notna(alt_raw) and str(alt_raw).strip():
                alias_map[norm_ticker(alt_raw)] = canonical
    return alias_map


def parse_pct(x) -> float:
    if pd.isna(x):
        raise ValueError("Cannot parse NaN percentage")
    s = str(x).strip().replace(",", "")
    if s.endswith("%"):
        return float(s[:-1]) / 100.0
    val = float(s)
    if val > 1.5:
        return val / 100.0
    return val


def parse_weights_string(s: str) -> Optional[List[float]]:
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        vals = [float(p.strip()) for p in s.split(",")]
    except Exception:
        return None
    # Leaderboard weights are usually decimal fractions, but tolerate percentages.
    if vals and max(vals) > 1.5:
        vals = [v / 100.0 for v in vals]
    return vals


def parse_forced_holdings(s: str, symbol_alias_map: Optional[Dict[str, str]] = None) -> Dict[str, int]:
    forced: Dict[str, int] = {}
    s = str(s or "").strip()
    if not s:
        return forced

    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Bad forced holding {part!r}. Use format like AAPL=25,NVDA=10.")
        ticker, weight = part.split("=", 1)
        t = canonical_ticker(ticker, symbol_alias_map)
        w = int(round(float(weight)))
        if w < 5 or w > 25:
            raise ValueError(f"Forced holding {t} has weight {w}%, outside the 5%-25% range.")
        if t in forced:
            raise ValueError(f"Duplicate forced ticker: {t}")
        forced[t] = w

    if sum(forced.values()) > 100:
        raise ValueError("Forced holdings exceed 100% total weight.")
    if sum(forced.values()) == 100 and len(forced) < 5:
        raise ValueError("Forced holdings sum to 100% but contain fewer than 5 stocks.")
    return forced


def portfolio_signature(port: Dict[str, float], round_to: int = 6) -> Tuple[Tuple[str, float], ...]:
    return tuple(sorted((k, round(v, round_to)) for k, v in port.items()))


# -----------------------------
# Matrix and input loading
# -----------------------------

def safe_float_matrix(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = (
            out[c].astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def repair_to_psd(matrix: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    matrix = (matrix + matrix.T) / 2.0
    vals, vecs = np.linalg.eigh(matrix)
    if vals.min() >= -1e-10:
        return matrix
    vals = np.clip(vals, floor, None)
    repaired = (vecs * vals) @ vecs.T
    return (repaired + repaired.T) / 2.0


def load_corr_matrix(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if raw.shape[1] < 2:
        raise ValueError("Correlation file must have one row-label column and at least one ticker column.")

    row_labels = raw.iloc[:, 0].apply(norm_ticker)
    col_labels = [norm_ticker(c) for c in raw.columns[1:]]
    corr_numeric = safe_float_matrix(raw.iloc[:, 1:])
    corr = pd.DataFrame(corr_numeric.values, index=row_labels, columns=col_labels)

    corr = corr.groupby(level=0).mean()
    corr = corr.T.groupby(level=0).mean().T

    common = sorted(set(corr.index) & set(corr.columns))
    if len(common) < 5:
        raise ValueError(f"Only {len(common)} common correlation tickers found; need at least 5.")

    corr = corr.loc[common, common].astype(float)
    finite_vals = corr.values[np.isfinite(corr.values)]
    if finite_vals.size == 0:
        raise ValueError("Correlation matrix contains no usable numeric values.")

    max_abs = float(np.nanmax(np.abs(finite_vals)))
    if max_abs > 1.000001:
        if max_abs <= 100.0:
            corr = corr / 100.0
        else:
            raise ValueError("Correlation matrix appears to contain values outside [-1, 1].")

    corr = (corr + corr.T) / 2.0
    corr = corr.clip(-1.0, 1.0).fillna(0.0)
    for t in corr.index:
        corr.loc[t, t] = 1.0

    arr = repair_to_psd(corr.values.astype(float))
    diag = np.sqrt(np.clip(np.diag(arr), 1e-12, None))
    arr = arr / np.outer(diag, diag)
    arr = np.clip(arr, -1.0, 1.0)
    np.fill_diagonal(arr, 1.0)
    return pd.DataFrame(arr, index=corr.index, columns=corr.columns)


def load_vols(path: str, symbol_alias_map: Optional[Dict[str, str]] = None) -> Dict[str, float]:
    vols = pd.read_csv(path)
    ticker_col = None
    for candidate in ["Ticker", "Symbol", "Underlying"]:
        if candidate in vols.columns:
            ticker_col = candidate
            break

    if ticker_col is None or "Implied Volatility" not in vols.columns:
        raise ValueError(
            f"Vol file {path} missing expected columns. Expected Implied Volatility and one of Ticker / Symbol / Underlying. "
            f"Found: {list(vols.columns)}"
        )

    out: Dict[str, float] = {}
    for _, row in vols.iterrows():
        raw = row.get(ticker_col, "")
        iv_raw = row.get("Implied Volatility", "")
        if pd.isna(raw) or not str(raw).strip() or pd.isna(iv_raw) or not str(iv_raw).strip():
            continue
        try:
            vol = parse_pct(iv_raw)
        except Exception:
            continue
        if vol <= 0:
            continue
        t_norm = norm_ticker(raw)
        t_canon = canonical_ticker(raw, symbol_alias_map)
        out[t_norm] = vol
        out[t_canon] = vol
    return out


def load_leaderboard(path: str) -> pd.DataFrame:
    lb = pd.read_csv(path)
    expected = {"Name", "Value", "Holdings", "Weights"}
    missing = expected - set(lb.columns)
    if missing:
        raise ValueError(f"Leaderboard missing columns: {missing}. Found: {list(lb.columns)}")
    lb = lb.dropna(subset=["Name", "Value", "Holdings"]).copy()
    lb["Name"] = lb["Name"].astype(str)
    lb["Value"] = lb["Value"].astype(float)
    return lb


# -----------------------------
# Portfolios and constraints
# -----------------------------

@dataclass
class ContestConstraints:
    min_stocks: int = 5
    min_weight: int = 5
    max_weight: int = 25
    step: int = 1
    allow_cash: bool = True
    max_cash: int = 25


def parse_portfolio_from_row(row: pd.Series, symbol_alias_map: Optional[Dict[str, str]] = None) -> Dict[str, float]:
    holdings = [canonical_ticker(x, symbol_alias_map) for x in str(row["Holdings"]).split(",") if str(x).strip()]
    w_list = parse_weights_string(row.get("Weights", ""))
    if (w_list is None) or (len(w_list) != len(holdings)):
        w = np.ones(len(holdings), dtype=float) / max(len(holdings), 1)
    else:
        w = np.array(w_list, dtype=float)

    port: Dict[str, float] = {}
    for t, wt in zip(holdings, w):
        port[t] = port.get(t, 0.0) + float(wt)

    rem = 1.0 - sum(port.values())
    if rem > 1e-8:
        port["CASH"] = port.get("CASH", 0.0) + float(rem)

    s = sum(port.values())
    if abs(s - 1.0) > 1e-8 and s > 0:
        for k in list(port.keys()):
            port[k] /= s
    return port


def build_player_portfolios(lb: pd.DataFrame, symbol_alias_map: Optional[Dict[str, str]] = None) -> Tuple[List[str], Dict[str, float], Dict[str, Dict[str, float]]]:
    ports: Dict[str, Dict[str, float]] = {}
    values: Dict[str, float] = {}
    for _, row in lb.iterrows():
        name = str(row["Name"])
        values[name] = float(row["Value"])
        ports[name] = parse_portfolio_from_row(row, symbol_alias_map=symbol_alias_map)

    seen = set()
    players: List[str] = []
    for name in lb["Name"].tolist():
        sig = (round(values[name], 6), portfolio_signature(ports[name], round_to=6))
        if sig in seen:
            continue
        seen.add(sig)
        players.append(name)
    return players, values, ports


def filter_portfolio_to_corr(port: Dict[str, float], corr: pd.DataFrame) -> Dict[str, float]:
    allowed = set(corr.index)
    out: Dict[str, float] = {}
    removed = 0.0
    for t, w in port.items():
        w = float(w)
        if t == "CASH" or t in allowed:
            out[t] = out.get(t, 0.0) + w
        else:
            removed += w
    if removed > 0:
        out["CASH"] = out.get("CASH", 0.0) + removed
    s = sum(out.values())
    if s <= 0:
        return {"CASH": 1.0}
    for k in list(out.keys()):
        out[k] /= s
    return out


def is_legal_portfolio(port: Dict[str, float], constraints: ContestConstraints, forced: Optional[Dict[str, int]] = None) -> bool:
    forced = forced or {}
    minw = constraints.min_weight / 100.0
    maxw = constraints.max_weight / 100.0
    step = constraints.step / 100.0
    max_cash = constraints.max_cash / 100.0

    if abs(sum(port.values()) - 1.0) > 5e-6:
        return False

    noncash = [(t, w) for t, w in port.items() if t != "CASH" and w > 1e-12]
    if len(noncash) < constraints.min_stocks:
        return False

    cash_w = float(port.get("CASH", 0.0))
    if cash_w < -1e-9:
        return False
    if constraints.allow_cash and cash_w > max_cash + 1e-12:
        return False
    if not constraints.allow_cash and cash_w > 1e-12:
        return False

    for t, w_pct in forced.items():
        if abs(port.get(t, 0.0) - w_pct / 100.0) > 5e-6:
            return False

    for _, w in noncash:
        if w < minw - 1e-12 or w > maxw + 1e-12:
            return False
        if abs(round(w / step) * step - w) > 5e-6:
            return False
    return True


def make_universe_from_players(players: List[str], ports: Dict[str, Dict[str, float]]) -> List[str]:
    tickers = set()
    for p in players:
        tickers.update(ports[p].keys())
    tickers.add("CASH")
    return sorted(tickers)


def ensure_corr_and_vol_coverage(universe: List[str], corr: pd.DataFrame, vol_dict: Dict[str, float]) -> Tuple[pd.DataFrame, np.ndarray]:
    n = len(universe)
    corr_u = pd.DataFrame(np.eye(n), index=universe, columns=universe, dtype=float)
    for i, a in enumerate(universe):
        for j, b in enumerate(universe):
            if a == b:
                corr_u.iat[i, j] = 1.0
            elif a == "CASH" or b == "CASH":
                corr_u.iat[i, j] = 0.0
            elif (a in corr.index) and (b in corr.columns):
                val = corr.loc[a, b]
                corr_u.iat[i, j] = float(val) if pd.notna(val) else 0.0
            else:
                corr_u.iat[i, j] = 0.0

    corr_u = (corr_u + corr_u.T) / 2.0
    for i in range(n):
        corr_u.iat[i, i] = 1.0

    vol_vec = np.zeros(n, dtype=float)
    for i, t in enumerate(universe):
        if t == "CASH":
            vol_vec[i] = 0.0
        else:
            if t not in vol_dict:
                close = sorted([k for k in vol_dict.keys() if str(k).replace(".", "-") == str(t).replace(".", "-")])
                extra = f" Close normalized matches: {close}" if close else ""
                raise ValueError(f"Missing volatility for ticker {t!r}. Add it to your vol file(s).{extra}")
            vol_vec[i] = float(vol_dict[t])

    C = repair_to_psd(corr_u.to_numpy(dtype=float, copy=True))
    d = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    C = C / (d[:, None] * d[None, :])
    np.fill_diagonal(C, 1.0)
    return pd.DataFrame(C, index=universe, columns=universe), vol_vec


def cov_from_corr_vol(corr_u: pd.DataFrame, vol_vec: np.ndarray) -> np.ndarray:
    D = np.diag(vol_vec)
    return D @ corr_u.to_numpy(dtype=float, copy=True) @ D


def vectorize_port(port: Dict[str, float], universe_index: Dict[str, int]) -> np.ndarray:
    w = np.zeros(len(universe_index), dtype=float)
    for t, wt in port.items():
        if t in universe_index:
            w[universe_index[t]] += float(wt)
    s = w.sum()
    if abs(s - 1.0) > 1e-8 and s > 0:
        w /= s
    return w


def dict_from_weight_vector(w: np.ndarray, universe: List[str]) -> Dict[str, float]:
    out = {t: float(w[i]) for i, t in enumerate(universe) if float(w[i]) > 1e-8}
    s = sum(out.values())
    if s > 0:
        for k in list(out.keys()):
            out[k] /= s
    return out


def tracking_error_portfolios(port_a: Dict[str, float], port_b: Dict[str, float], corr_u: pd.DataFrame, vol_vec: np.ndarray) -> float:
    universe = list(corr_u.index)
    u_index = {t: i for i, t in enumerate(universe)}
    Sigma_ann = cov_from_corr_vol(corr_u, vol_vec)
    diff = vectorize_port(port_a, u_index) - vectorize_port(port_b, u_index)
    return float(math.sqrt(max(diff @ Sigma_ann @ diff, 0.0)))


# -----------------------------
# Simulation and benchmark
# -----------------------------

def simulate_finish_probs(players: List[str], values: Dict[str, float], W_mat: np.ndarray, Sigma_ann: np.ndarray, days_remaining: int, n_sims: int = 400_000, seed: int = 0, return_counts: bool = False):
    rng = np.random.default_rng(seed)
    Sigma_day = Sigma_ann / 252.0
    mu_day = -0.5 * np.diag(Sigma_day)
    C = repair_to_psd(Sigma_day)
    L = np.linalg.cholesky(C + 1e-16 * np.eye(C.shape[0]))

    nP = len(players)
    logV0 = np.array([math.log(values[p]) for p in players], dtype=float)
    counts = {p: np.zeros(4, dtype=np.int64) for p in players}
    batch = 20_000

    for start in range(0, n_sims, batch):
        b = min(batch, n_sims - start)
        Z = rng.standard_normal(size=(b, days_remaining, C.shape[0]))
        R = mu_day + (Z @ L.T)
        port_r = np.einsum("bdk,pk->bdp", R, W_mat)
        logV = logV0 + port_r.sum(axis=1)
        order = np.argsort(-logV, axis=1)
        pos = np.full((b, nP), 3, dtype=np.int8)
        pos[np.arange(b), order[:, 0]] = 0
        pos[np.arange(b), order[:, 1]] = 1
        pos[np.arange(b), order[:, 2]] = 2
        for i, p in enumerate(players):
            vals, cts = np.unique(pos[:, i], return_counts=True)
            for v, ct in zip(vals, cts):
                counts[p][v] += ct

    probs = {}
    for p in players:
        probs[p] = {
            "P1st": counts[p][0] / n_sims,
            "P2nd": counts[p][1] / n_sims,
            "P3rd": counts[p][2] / n_sims,
            "PWorse": counts[p][3] / n_sims,
        }
    if return_counts:
        return probs, counts, n_sims
    return probs


def moe95(p_hat: float, n: int) -> float:
    return 1.96 * math.sqrt(max(p_hat * (1 - p_hat), 0.0) / n)


def build_weighted_benchmark(players: List[str], ports: Dict[str, Dict[str, float]], win_probs: Dict[str, float]) -> Dict[str, float]:
    total = sum(win_probs.get(p, 0.0) for p in players)
    if total <= 0:
        raise ValueError("Sum of win probabilities is <= 0.")
    bench: Dict[str, float] = {}
    for p in players:
        alpha = win_probs.get(p, 0.0) / total
        for t, wt in ports[p].items():
            bench[t] = bench.get(t, 0.0) + alpha * float(wt)
    s = sum(bench.values())
    for k in list(bench.keys()):
        bench[k] /= s
    return bench


def load_or_compute_win_probs_excl_anchor(winprob_path: str, players_ex: List[str], values_ex: Dict[str, float], W_mat_current_ex: np.ndarray, Sigma_ann_ex: np.ndarray, days_remaining: int, seed: int) -> Dict[str, float]:
    if os.path.exists(winprob_path):
        wp = pd.read_csv(winprob_path)
        if "Name" not in wp.columns or "WinProb" not in wp.columns:
            raise ValueError(f"{winprob_path} must contain columns Name, WinProb")
        probs = {str(n): float(w) for n, w in zip(wp["Name"], wp["WinProb"])}
        probs = {p: probs[p] for p in players_ex if p in probs}
        s = sum(probs.values())
        if s <= 0:
            raise ValueError(f"{winprob_path} sums to <= 0 over non-anchor players.")
        return {p: w / s for p, w in probs.items()}

    finish_ex = simulate_finish_probs(players_ex, values_ex, W_mat_current_ex, Sigma_ann_ex, days_remaining, n_sims=200_000, seed=seed)
    probs = {p: finish_ex[p]["P1st"] for p in players_ex}
    s = sum(probs.values())
    probs = {p: w / s for p, w in probs.items()}
    pd.DataFrame({"Name": list(probs.keys()), "WinProb": list(probs.values())}).sort_values("WinProb", ascending=False).to_csv(winprob_path, index=False)
    print(f"Wrote: {winprob_path} (created because it was missing)")
    return probs


# -----------------------------
# Generic legal sampler and local polish for TE/WIN
# -----------------------------

def legalise_int_port(weights: Dict[str, int], constraints: ContestConstraints, forced: Dict[str, int]) -> Dict[str, float]:
    port = {t: w / 100.0 for t, w in weights.items() if w > 0}
    cash = 1.0 - sum(port.values())
    if constraints.allow_cash and cash > 1e-12:
        port["CASH"] = cash
    s = sum(port.values())
    if abs(s - 1.0) > 1e-8 and s > 0:
        for k in list(port.keys()):
            port[k] /= s
    if not is_legal_portfolio(port, constraints, forced):
        raise ValueError("Internal error: integer portfolio is not legal.")
    return port


def sample_legal_portfolio(rng: np.random.Generator, tickers: List[str], ticker_probs: np.ndarray, constraints: ContestConstraints, forced: Dict[str, int]) -> Dict[str, float]:
    forced = forced or {}
    forced_weight = sum(forced.values())
    forced_count = len(forced)
    remaining_weight = 100 - forced_weight

    if remaining_weight == 0:
        return legalise_int_port(forced.copy(), constraints, forced)

    available = [t for t in tickers if t not in forced]
    if not available:
        raise ValueError("No available tickers remain after forced holdings.")

    min_extra_names = max(0, constraints.min_stocks - forced_count)
    max_extra_names = min(len(available), remaining_weight // constraints.min_weight)
    feasible_counts = [k for k in range(min_extra_names, max_extra_names + 1) if k == 0 or k * constraints.min_weight <= remaining_weight <= k * constraints.max_weight + constraints.max_cash]
    if not feasible_counts:
        raise ValueError("Forced holdings leave no feasible way to complete the portfolio.")

    k = int(rng.choice(feasible_counts))
    if k == 0:
        return legalise_int_port(forced.copy(), constraints, forced)

    p = np.array([ticker_probs[tickers.index(t)] if t in tickers else 0.0 for t in available], dtype=float)
    p = np.maximum(p, 1e-12)
    p /= p.sum()
    chosen = list(rng.choice(available, size=k, replace=False, p=p))

    min_stock_sum = forced_weight + k * constraints.min_weight
    max_stock_sum = forced_weight + k * constraints.max_weight
    if constraints.allow_cash:
        min_stock_sum = max(min_stock_sum, 100 - constraints.max_cash)
        max_stock_sum = min(max_stock_sum, 100)
    else:
        min_stock_sum = max_stock_sum = 100
    if min_stock_sum > max_stock_sum:
        return sample_legal_portfolio(rng, tickers, ticker_probs, constraints, forced)

    if min_stock_sum == max_stock_sum:
        stock_sum = int(min_stock_sum)
    else:
        mode = min(max_stock_sum, max(min_stock_sum, int(round((min_stock_sum + max_stock_sum) / 2))))
        stock_sum = int(round(rng.triangular(min_stock_sum, mode, max_stock_sum)))

    alloc = np.full(k, constraints.min_weight, dtype=int)
    remaining = stock_sum - forced_weight - int(alloc.sum())
    caps = np.full(k, constraints.max_weight - constraints.min_weight, dtype=int)

    alpha = np.array([max(ticker_probs[tickers.index(t)], 1e-6) for t in chosen], dtype=float)
    alpha = 0.25 + 5.0 * alpha / alpha.sum()
    extra = np.floor(max(remaining, 0) * rng.dirichlet(alpha)).astype(int)
    extra = np.minimum(extra, caps)
    alloc += extra

    rem2 = stock_sum - forced_weight - int(alloc.sum())
    tries = 0
    while rem2 > 0 and tries < 10_000:
        i = int(rng.integers(0, k))
        if alloc[i] < constraints.max_weight:
            alloc[i] += 1
            rem2 -= 1
        tries += 1

    weights = forced.copy()
    weights.update({t: int(w) for t, w in zip(chosen, alloc)})
    return legalise_int_port(weights, constraints, forced)


def cross_entropy_optimise(
    rng: np.random.Generator,
    tickers: List[str],
    constraints: ContestConstraints,
    forced: Dict[str, int],
    objective_fn: Callable[[np.ndarray], Tuple[float, float]],
    u_index: Dict[str, int],
    initial_bias: Optional[Dict[str, float]],
    n_rounds: int,
    samples_per_round: int,
    maximise: bool,
    label: str,
    polish_steps: int = 800,
) -> Dict[str, float]:
    probs = np.array([max((initial_bias or {}).get(t, 0.0), 0.0) for t in tickers], dtype=float)
    if probs.sum() <= 0:
        probs = np.ones(len(tickers), dtype=float) / len(tickers)
    else:
        probs /= probs.sum()

    best_port: Optional[Dict[str, float]] = None
    best_score = -float("inf") if maximise else float("inf")
    best_aux = 0.0
    elite_frac = 0.12
    elite_n = max(10, int(round(samples_per_round * elite_frac)))

    def to_vec(port: Dict[str, float]) -> np.ndarray:
        return vectorize_port(port, u_index)

    for r in range(n_rounds):
        scores = np.empty(samples_per_round, dtype=float)
        auxs = np.empty(samples_per_round, dtype=float)
        ports: List[Dict[str, float]] = []
        for i in range(samples_per_round):
            port = sample_legal_portfolio(rng, tickers, probs, constraints, forced)
            score, aux = objective_fn(to_vec(port))
            scores[i] = score
            auxs[i] = aux
            ports.append(port)
            improved = score > best_score if maximise else score < best_score
            if best_port is None or improved:
                best_port = port
                best_score = score
                best_aux = aux

        elite_idx = np.argsort(-scores)[:elite_n] if maximise else np.argsort(scores)[:elite_n]
        counts = np.zeros(len(tickers), dtype=float)
        wsum = np.zeros(len(tickers), dtype=float)
        for idx in elite_idx:
            port = ports[int(idx)]
            for j, t in enumerate(tickers):
                if t in port and port[t] > 1e-12:
                    counts[j] += 1.0
                    wsum[j] += port[t]
        freq = counts / counts.sum() if counts.sum() > 0 else np.ones(len(tickers)) / len(tickers)
        wfreq = wsum / wsum.sum() if wsum.sum() > 0 else freq
        new_probs = 0.65 * (0.55 * freq + 0.45 * wfreq) + 0.35 * probs
        new_probs = np.maximum(new_probs, 1e-9)
        probs = new_probs / new_probs.sum()
        direction = "best" if maximise else "lowest"
        print(f"Round {r + 1}/{n_rounds}: {direction} {label}={best_aux:.6f}  (objective={best_score:.6f})")

    if best_port is None:
        raise RuntimeError("Optimiser failed to find a feasible portfolio.")

    # Polish by 1% transfers among held names, cash, and high-probability outsiders.
    universe = list(u_index.keys())
    w = to_vec(best_port)
    forced_idx = {u_index[t] for t in forced if t in u_index}
    cash_i = u_index.get("CASH", None)
    step = constraints.step / 100.0
    minw = constraints.min_weight / 100.0
    maxw = constraints.max_weight / 100.0
    max_cash = constraints.max_cash / 100.0
    noncash_idx = [u_index[t] for t in tickers if t in u_index]

    cur_score, cur_aux = objective_fn(w)
    candidate_receivers = sorted(tickers, key=lambda t: probs[tickers.index(t)], reverse=True)[:max(40, constraints.min_stocks)]
    candidate_receiver_idx = [u_index[t] for t in candidate_receivers if t in u_index]
    if cash_i is not None:
        candidate_receiver_idx.append(cash_i)

    def count_noncash(wv: np.ndarray) -> int:
        return int(np.sum(wv[noncash_idx] > 1e-12))

    for _ in range(polish_steps):
        improved_any = False
        donors = [i for i in range(len(universe)) if w[i] > 1e-12 and i not in forced_idx]
        receivers = sorted(set(candidate_receiver_idx + [i for i in noncash_idx if w[i] > 1e-12]))
        for donor in donors:
            if donor == cash_i:
                if w[donor] - step < -1e-12:
                    continue
            else:
                if w[donor] - step < minw - 1e-12:
                    continue
                if w[donor] - step <= 1e-12 and count_noncash(w) - 1 < constraints.min_stocks:
                    continue
            for recv in receivers:
                if recv == donor or recv in forced_idx:
                    continue
                if recv == cash_i:
                    if w[recv] + step > max_cash + 1e-12:
                        continue
                else:
                    if w[recv] <= 1e-12 and step < minw - 1e-12:
                        continue
                    if w[recv] + step > maxw + 1e-12:
                        continue
                w2 = w.copy()
                w2[donor] -= step
                w2[recv] += step
                w2 = np.clip(w2, 0.0, None)
                s = w2.sum()
                if abs(s - 1.0) > 1e-8 and s > 0:
                    w2 /= s
                p2 = dict_from_weight_vector(w2, universe)
                if not is_legal_portfolio(p2, constraints, forced):
                    continue
                score2, aux2 = objective_fn(w2)
                improved = score2 > cur_score + 1e-12 if maximise else score2 < cur_score - 1e-12
                if improved:
                    w = w2
                    cur_score = score2
                    cur_aux = aux2
                    improved_any = True
                    break
            if improved_any:
                break
        if not improved_any:
            break

    out = dict_from_weight_vector(w, universe)
    if not is_legal_portfolio(out, constraints, forced):
        return best_port
    print(f"After polish: {label}={cur_aux:.6f}  (objective={cur_score:.6f})")
    return out


# -----------------------------
# WIN objective context
# -----------------------------

@dataclass
class MonteCarloObjectiveContext:
    r_T: np.ndarray
    max_logV_other: np.ndarray
    logV0_a: float


def make_mc_objective_context(players: List[str], values: Dict[str, float], W_full: Dict[str, np.ndarray], Sigma_ann: np.ndarray, days_remaining: int, inner_sims: int, seed: int, anchor: str) -> MonteCarloObjectiveContext:
    if anchor not in players:
        raise ValueError(f"Anchor {anchor!r} not found in Leaderboard.csv")

    Sigma_day = Sigma_ann / 252.0
    mu_day = -0.5 * np.diag(Sigma_day)
    T = int(days_remaining)
    mean_T = T * mu_day
    cov_T = repair_to_psd(T * Sigma_day)
    L = np.linalg.cholesky(cov_T + 1e-16 * np.eye(cov_T.shape[0]))

    rng = np.random.default_rng(seed + 12345)
    Z = rng.standard_normal(size=(inner_sims, cov_T.shape[0]))
    r_T = mean_T + (Z @ L.T)

    logV0_a = float(math.log(values[anchor]))
    players_other = [p for p in players if p != anchor]
    W_other = np.stack([W_full[p] for p in players_other], axis=0)
    logV0_other = np.array([math.log(values[p]) for p in players_other], dtype=float)
    logV_other = logV0_other[None, :] + (r_T @ W_other.T)
    max_logV_other = np.max(logV_other, axis=1)
    return MonteCarloObjectiveContext(r_T=r_T, max_logV_other=max_logV_other, logV0_a=logV0_a)


def mc_win_prob_first(w_anchor: np.ndarray, ctx: MonteCarloObjectiveContext) -> Tuple[float, float]:
    logV_anchor = ctx.logV0_a + (ctx.r_T @ w_anchor)
    p_hat = float(np.mean(logV_anchor > ctx.max_logV_other))
    return float(math.log(p_hat + 1e-12)), p_hat


# -----------------------------
# Volatility maximizer from VolatilityMaximizer.py, adapted for shared inputs
# -----------------------------

def int_portfolio_variance(port: Dict[str, int], cov: np.ndarray, ticker_to_idx: Dict[str, int]) -> float:
    names = list(port.keys())
    idx = np.array([ticker_to_idx[t] for t in names], dtype=int)
    w = np.array([port[t] / 100.0 for t in names], dtype=float)
    subcov = cov[np.ix_(idx, idx)]
    return float(w @ subcov @ w)


def clean_int_portfolio(port: Dict[str, int]) -> Dict[str, int]:
    return {k: int(v) for k, v in port.items() if int(v) > 0}


def legal_int_portfolio(port: Dict[str, int], forced: Optional[Dict[str, int]] = None) -> bool:
    forced = forced or {}
    if sum(port.values()) != 100 or len(port) < 5:
        return False
    for t, w in forced.items():
        if port.get(t) != w:
            return False
    for w in port.values():
        if not isinstance(w, int) or w < 5 or w > 25:
            return False
    return True


def random_int_portfolio_from_universe(universe: List[str], forced: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    forced = forced or {}
    missing = [t for t in forced if t not in universe]
    if missing:
        raise ValueError(f"Forced tickers not in this search universe: {missing}")

    forced_weight = sum(forced.values())
    remaining_weight = 100 - forced_weight
    forced_count = len(forced)
    if remaining_weight == 0:
        return forced.copy()

    min_extra_names = max(0, 5 - forced_count)
    max_extra_names = min(len(universe) - forced_count, remaining_weight // 5)
    feasible = [k for k in range(min_extra_names, max_extra_names + 1) if k == 0 or k * 5 <= remaining_weight <= k * 25]
    if not feasible:
        raise ValueError("Forced holdings leave no feasible way to complete the portfolio.")
    extra_k = random.choice(feasible)
    available = [t for t in universe if t not in forced]
    names = random.sample(available, extra_k) if extra_k > 0 else []
    weights = [5] * extra_k
    remaining = remaining_weight - extra_k * 5
    while remaining > 0:
        i = random.randrange(extra_k)
        if weights[i] < 25:
            weights[i] += 1
            remaining -= 1
    port = forced.copy()
    port.update(dict(zip(names, weights)))
    return port


def build_candidate_sets(tickers: List[str], cov: np.ndarray, vol_map: Dict[str, float], corr_df: pd.DataFrame, shortlist_size: int, elite_pool_size: int) -> Tuple[List[str], List[str], List[str], Dict[str, float]]:
    vols = {t: vol_map[t] for t in tickers}
    avg_pos_corr = {}
    cov_contrib = {}
    idx = {t: i for i, t in enumerate(tickers)}
    total_cov = cov.sum(axis=1)
    for t in tickers:
        row = corr_df.loc[t, tickers].values.astype(float)
        pos = row[row > 0]
        avg_pos_corr[t] = float(pos.mean()) if len(pos) > 0 else 0.0
        cov_contrib[t] = float(total_cov[idx[t]])

    def zscore_map(d: Dict[str, float]) -> Dict[str, float]:
        vals = np.array(list(d.values()), dtype=float)
        sd = vals.std()
        if sd == 0:
            return {k: 0.0 for k in d}
        return {k: (v - vals.mean()) / sd for k, v in d.items()}

    z_vol = zscore_map(vols)
    z_cov = zscore_map(cov_contrib)
    z_corr = zscore_map(avg_pos_corr)
    score = {t: 1.55 * z_vol[t] + 0.65 * z_cov[t] - 0.40 * z_corr[t] for t in tickers}
    ranked_all = sorted(tickers, key=lambda t: score[t], reverse=True)
    shortlist = ranked_all[: min(shortlist_size, len(ranked_all))]
    elite_pool = ranked_all[: min(elite_pool_size, len(ranked_all))]
    return shortlist, elite_pool, ranked_all, score


def make_seed_portfolios(shortlist: List[str], ranked_all: List[str], score: Dict[str, float], vol_map: Dict[str, float], corr_df: pd.DataFrame) -> List[Dict[str, int]]:
    ranked = sorted(shortlist, key=lambda t: score[t], reverse=True)
    vol_ranked = sorted(shortlist, key=lambda t: vol_map[t], reverse=True)
    seeds: List[Dict[str, int]] = []
    if len(ranked) >= 5:
        seeds.append({t: 20 for t in ranked[:5]})
    if len(vol_ranked) >= 5:
        seeds.append({t: 20 for t in vol_ranked[:5]})
    if len(ranked) >= 5:
        p = {t: 25 for t in ranked[:4]}
        p[ranked[0]] = 20
        p[ranked[4]] = 5
        seeds.append(p)
    if len(ranked) >= 6:
        seeds.append({ranked[0]: 25, ranked[1]: 20, ranked[2]: 20, ranked[3]: 15, ranked[4]: 10, ranked[5]: 10})
    if len(ranked) >= 8:
        seeds.append({ranked[0]: 20, ranked[1]: 15, ranked[2]: 15, ranked[3]: 15, ranked[4]: 10, ranked[5]: 10, ranked[6]: 10, ranked[7]: 5})
    if len(vol_ranked) >= 1:
        anchor = vol_ranked[0]
        pool = [t for t in ranked_all[: min(40, len(ranked_all))] if t != anchor]
        if len(pool) >= 4:
            corr_sorted = sorted(pool, key=lambda t: corr_df.loc[anchor, t], reverse=True)
            seeds.append({anchor: 25, corr_sorted[0]: 20, corr_sorted[1]: 20, corr_sorted[2]: 20, corr_sorted[3]: 15})
    if len(ranked) >= 10:
        seeds.append({ranked[0]: 15, ranked[1]: 15, ranked[2]: 15, ranked[3]: 10, ranked[4]: 10, ranked[5]: 10, ranked[6]: 10, ranked[7]: 5, ranked[8]: 5, ranked[9]: 5})

    out = []
    seen = set()
    for s in seeds:
        s = clean_int_portfolio(s)
        if legal_int_portfolio(s):
            key = tuple(sorted(s.items()))
            if key not in seen:
                out.append(s)
                seen.add(key)
    return out


def impose_forced_holdings(seed: Dict[str, int], forced: Dict[str, int], universe: List[str]) -> Dict[str, int]:
    if not forced:
        return seed
    port = {t: w for t, w in seed.items() if t not in forced}
    port.update(forced)
    total = sum(port.values())
    while total > 100:
        adjustable = [t for t in port if t not in forced and port[t] > 5]
        if not adjustable:
            raise ValueError("Cannot impose forced holdings on seed without violating constraints.")
        t = random.choice(adjustable)
        port[t] -= 1
        total -= 1
    available = [t for t in universe if t not in port]
    while total < 100:
        receivers = [t for t in port if t not in forced and port[t] < 25]
        if receivers:
            t = random.choice(receivers)
            port[t] += 1
            total += 1
        else:
            if not available:
                raise ValueError("Cannot complete seed after imposing forced holdings.")
            need = 100 - total
            if need < 5:
                adjustable = [t for t in port if t not in forced and port[t] > 5]
                if not adjustable:
                    raise ValueError("Cannot legally complete seed after imposing forced holdings.")
                t = random.choice(adjustable)
                port[t] -= 5 - need
                total -= 5 - need
                need = 5
            t = available.pop()
            add = min(25, need)
            port[t] = add
            total += add
    port = clean_int_portfolio(port)
    if not legal_int_portfolio(port, forced):
        raise ValueError("Failed to impose forced holdings while preserving contest rules.")
    return port


def top_k_outsiders_from_ranked(ranked_names: List[str], held: Set[str], k: int) -> List[str]:
    out = []
    for t in ranked_names:
        if t not in held:
            out.append(t)
            if len(out) >= k:
                break
    return out


def local_improvement_int(port: Dict[str, int], cov: np.ndarray, ticker_to_idx: Dict[str, int], ranked_names: List[str], forced: Optional[Dict[str, int]] = None, passes: int = LOCAL_PASSES, outsider_checks: int = OUTSIDER_CHECKS) -> Tuple[Dict[str, int], float]:
    forced = forced or {}
    forced_names = set(forced)
    best = clean_int_portfolio(port.copy())
    best_val = int_portfolio_variance(best, cov, ticker_to_idx)
    for _ in range(passes):
        improved = False
        held = list(best.keys())
        for a in held:
            if a in forced_names or a not in best or best[a] <= 5:
                continue
            for b in held:
                if b in forced_names or b not in best or a == b or best[b] >= 25:
                    continue
                cand = best.copy()
                cand[a] -= 1
                cand[b] += 1
                cand = clean_int_portfolio(cand)
                if not legal_int_portfolio(cand, forced):
                    continue
                val = int_portfolio_variance(cand, cov, ticker_to_idx)
                if val > best_val + EPS:
                    best, best_val = cand, val
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue

        held_set = set(best.keys())
        outsiders = top_k_outsiders_from_ranked(ranked_names, held_set, outsider_checks)
        current_names = sorted([t for t in best.keys() if t not in forced_names], key=lambda t: best[t])
        for old_name in current_names:
            wt = best[old_name]
            for new_name in outsiders:
                if new_name in best:
                    continue
                cand = best.copy()
                del cand[old_name]
                cand[new_name] = wt
                cand = clean_int_portfolio(cand)
                if not legal_int_portfolio(cand, forced):
                    continue
                val = int_portfolio_variance(cand, cov, ticker_to_idx)
                if val > best_val + EPS:
                    best, best_val = cand, val
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best, best_val


@dataclass
class MutationResult:
    portfolio: Dict[str, int]
    changed_names: Tuple[str, ...]


def mutate_transfer(port: Dict[str, int], forced: Dict[str, int]) -> MutationResult:
    p = port.copy()
    forced_names = set(forced)
    donors = [t for t, w in p.items() if t not in forced_names and w > 5]
    receivers = [t for t, w in p.items() if t not in forced_names and w < 25]
    if not donors or not receivers:
        return MutationResult(p, tuple())
    a = random.choice(donors)
    b = random.choice(receivers)
    tries = 0
    while a == b and tries < 10:
        b = random.choice(receivers)
        tries += 1
    if a == b:
        return MutationResult(p, tuple())
    move = random.choice([1, 1, 1, 2, 3])
    move = min(move, p[a] - 5, 25 - p[b])
    if move <= 0:
        return MutationResult(p, tuple())
    p[a] -= move
    p[b] += move
    return MutationResult(clean_int_portfolio(p), (a, b))


def mutate_replace_one(port: Dict[str, int], universe: List[str], taboo_names: Set[str], forced: Dict[str, int]) -> MutationResult:
    p = port.copy()
    forced_names = set(forced)
    held = set(p.keys())
    replaceable = [t for t in p if t not in forced_names]
    if not replaceable:
        return MutationResult(p, tuple())
    candidates = [t for t in universe if t not in held and t not in taboo_names]
    if not candidates:
        candidates = [t for t in universe if t not in held]
    if not candidates:
        return MutationResult(p, tuple())
    old_name = random.choice(replaceable)
    wt = p.pop(old_name)
    new_name = random.choice(candidates)
    p[new_name] = wt
    return MutationResult(clean_int_portfolio(p), (old_name, new_name))


def mutate_replace_two(port: Dict[str, int], universe: List[str], taboo_names: Set[str], forced: Dict[str, int]) -> MutationResult:
    p = port.copy()
    forced_names = set(forced)
    replaceable = [t for t in p if t not in forced_names]
    if len(replaceable) < 2:
        return MutationResult(port.copy(), tuple())
    old_a, old_b = random.sample(replaceable, 2)
    wt_a = p.pop(old_a)
    wt_b = p.pop(old_b)
    held = set(p.keys())
    candidates = [t for t in universe if t not in held and t not in taboo_names]
    if len(candidates) < 2:
        candidates = [t for t in universe if t not in held]
    if len(candidates) < 2:
        p[old_a] = wt_a
        p[old_b] = wt_b
        return MutationResult(clean_int_portfolio(p), tuple())
    new_a, new_b = random.sample(candidates, 2)
    p[new_a] = wt_a
    p[new_b] = wt_b
    return MutationResult(clean_int_portfolio(p), (old_a, old_b, new_a, new_b))


def mutate_split_merge(port: Dict[str, int], universe: List[str], taboo_names: Set[str], forced: Dict[str, int]) -> MutationResult:
    p = port.copy()
    forced_names = set(forced)
    if random.random() < 0.5:
        sources = [t for t, w in p.items() if t not in forced_names and w >= 10]
        held = set(p.keys())
        candidates = [t for t in universe if t not in held and t not in taboo_names]
        if not candidates:
            candidates = [t for t in universe if t not in held]
        if not sources or not candidates:
            return MutationResult(p, tuple())
        old_name = random.choice(sources)
        new_name = random.choice(candidates)
        max_split = min(p[old_name] - 5, 25)
        if max_split < 5:
            return MutationResult(p, tuple())
        split_amt = random.randint(5, max_split)
        p[old_name] -= split_amt
        p[new_name] = split_amt
        return MutationResult(clean_int_portfolio(p), (old_name, new_name))

    if len(p) <= 5:
        return MutationResult(p, tuple())
    smalls = [t for t, w in p.items() if t not in forced_names and w == 5]
    receivers = [t for t, w in p.items() if t not in forced_names and w <= 20]
    if not smalls or not receivers:
        return MutationResult(p, tuple())
    a = random.choice(smalls)
    b = random.choice(receivers)
    tries = 0
    while a == b and tries < 10:
        b = random.choice(receivers)
        tries += 1
    if a == b:
        return MutationResult(p, tuple())
    p[b] += p[a]
    del p[a]
    return MutationResult(clean_int_portfolio(p), (a, b))


def anneal_search_int(start_port: Dict[str, int], universe: List[str], cov: np.ndarray, ticker_to_idx: Dict[str, int], ranked_names: List[str], forced: Optional[Dict[str, int]] = None, steps: int = ANNEAL_STEPS, tabu_len: int = TABU_LEN) -> Tuple[Dict[str, int], float]:
    forced = forced or {}
    current = clean_int_portfolio(start_port.copy())
    current, current_val = local_improvement_int(current, cov, ticker_to_idx, ranked_names, forced=forced)
    best = current.copy()
    best_val = current_val
    base = max(current_val, 1e-8)
    temp_start = max(TEMP_START_MULT * base, 1e-6)
    temp_end = max(TEMP_END_MULT * base, 1e-8)
    tabu = deque(maxlen=tabu_len)
    for step_i in range(steps):
        frac = step_i / max(1, steps - 1)
        temp = temp_start * ((temp_end / temp_start) ** frac)
        taboo_names = set(tabu)
        r = random.random()
        if r < 0.38:
            mut = mutate_transfer(current, forced)
        elif r < 0.66:
            mut = mutate_replace_one(current, universe, taboo_names, forced)
        elif r < 0.84:
            mut = mutate_split_merge(current, universe, taboo_names, forced)
        else:
            mut = mutate_replace_two(current, universe, taboo_names, forced)
        cand = clean_int_portfolio(mut.portfolio)
        if not legal_int_portfolio(cand, forced):
            continue
        if step_i % 50 == 0:
            cand, cand_val = local_improvement_int(cand, cov, ticker_to_idx, ranked_names, forced=forced, passes=3, outsider_checks=min(40, OUTSIDER_CHECKS))
        else:
            cand_val = int_portfolio_variance(cand, cov, ticker_to_idx)
        delta = cand_val - current_val
        if delta >= 0 or random.random() < math.exp(delta / max(temp, 1e-12)):
            current = cand
            current_val = cand_val
            for name in mut.changed_names:
                tabu.append(name)
            if cand_val > best_val + EPS:
                best = cand.copy()
                best_val = cand_val
    best, best_val = local_improvement_int(best, cov, ticker_to_idx, ranked_names, forced=forced)
    return best, best_val


def solve_max_vol_portfolio(tickers: List[str], corr: pd.DataFrame, vol_map: Dict[str, float], forced: Dict[str, int], seed: int) -> Tuple[Dict[str, float], float]:
    random.seed(seed)
    np.random.seed(seed)

    usable = sorted([t for t in tickers if t in corr.index and t in corr.columns and t in vol_map])
    missing_forced = [t for t in forced if t not in usable]
    if missing_forced:
        raise ValueError(f"Forced tickers not found in usable volatility universe: {missing_forced}")
    if len(usable) < 5:
        raise ValueError(f"Only {len(usable)} usable tickers found for volatility maximization; need at least 5.")

    corr_u = corr.loc[usable, usable]
    vols = np.array([vol_map[t] for t in usable], dtype=float)
    cov = np.diag(vols) @ corr_u.values @ np.diag(vols)
    cov = repair_to_psd((cov + cov.T) / 2.0)
    ticker_to_idx = {t: i for i, t in enumerate(usable)}

    shortlist, elite_pool, ranked_all, score = build_candidate_sets(usable, cov, vol_map, corr_u, SHORTLIST_SIZE, ELITE_POOL_SIZE)
    for t in forced:
        if t not in shortlist:
            shortlist.append(t)
        if t not in elite_pool:
            elite_pool.append(t)

    print(f"Usable tickers: {len(usable)}")
    print(f"Shortlist size: {len(shortlist)}")
    print(f"Elite pool size: {len(elite_pool)}")
    print(f"Forced holdings: {forced if forced else 'none'}")

    seed_ports = make_seed_portfolios(shortlist, ranked_all, score, vol_map, corr_u)
    seed_ports = [impose_forced_holdings(seed_port, forced, elite_pool) for seed_port in seed_ports]
    seed_ports = [p for p in seed_ports if legal_int_portfolio(p, forced)]

    best_port: Optional[Dict[str, int]] = None
    best_val = -1.0
    seen_best_keys = set()

    for i, seed_port in enumerate(seed_ports, start=1):
        port, val = anneal_search_int(seed_port, elite_pool, cov, ticker_to_idx, ranked_all, forced=forced)
        if val > best_val + EPS:
            best_port, best_val = port, val
        seen_best_keys.add(tuple(sorted(port.items())))
        print(f"Seed {i}/{len(seed_ports)} complete | best vol so far = {math.sqrt(best_val):.4%}")

    for i in range(N_STARTS):
        u = random.random()
        if u < SHORTLIST_START_FRAC:
            search_universe = shortlist
        elif u < SHORTLIST_START_FRAC + ELITE_POOL_START_FRAC:
            search_universe = elite_pool
        else:
            search_universe = usable
        start = random_int_portfolio_from_universe(search_universe, forced=forced)
        port, val = anneal_search_int(start, search_universe, cov, ticker_to_idx, ranked_all, forced=forced)
        if val > best_val + EPS:
            best_port, best_val = port, val
        seen_best_keys.add(tuple(sorted(port.items())))
        if (i + 1) % 20 == 0:
            print(f"Random start {i + 1}/{N_STARTS} complete | best vol so far = {math.sqrt(best_val):.4%} | distinct local optima seen = {len(seen_best_keys)}")

    if best_port is None:
        raise RuntimeError("Volatility search failed to produce any legal portfolio.")
    out = {t: w / 100.0 for t, w in best_port.items()}
    return out, best_val


# -----------------------------
# Main orchestration
# -----------------------------

def get_days_remaining(days_cli: Optional[int]) -> int:
    if days_cli is not None:
        if days_cli <= 0:
            raise ValueError("--days must be a positive integer")
        return days_cli
    s = input("Enter number of trading days remaining: ").strip()
    d = int(s)
    if d <= 0:
        raise ValueError("Days remaining must be a positive integer")
    return d


def run_all(
    objective: str,
    leaderboard_csv: str,
    corr_csv: str,
    vol_short_csv: str,
    vol_medium_csv: str,
    days_remaining: int,
    n_sims: int,
    seed: int,
    winprob_path: str,
    wp_vol_source: str,
    opt_vol_source: str,
    sim_vol_source: str,
    inner_sims: int,
    rounds: int,
    samples: int,
    exclude_tickers: str,
    forced_holdings: str,
    max_cash: int,
    anchor_name: str,
    stocklist_csv: str,
) -> None:
    symbol_alias_map = load_symbol_alias_map(stocklist_csv)
    if symbol_alias_map:
        n_aliases = sum(1 for k, v in symbol_alias_map.items() if k != v)
        print(f"Loaded {n_aliases} alternative-symbol mapping(s) from {stocklist_csv}")

    corr = load_corr_matrix(corr_csv)
    vol_short = load_vols(vol_short_csv, symbol_alias_map=symbol_alias_map)
    vol_med = load_vols(vol_medium_csv, symbol_alias_map=symbol_alias_map)

    vols_wp = vol_short if wp_vol_source == "short" else vol_med
    vols_opt = vol_short if opt_vol_source == "short" else vol_med
    vols_sim = vol_short if sim_vol_source == "short" else vol_med

    forced = parse_forced_holdings(forced_holdings, symbol_alias_map=symbol_alias_map)
    raw_excludes = [canonical_ticker(x, symbol_alias_map) for x in exclude_tickers.split(",") if x.strip()]
    exclude_set = set(raw_excludes) - set(forced)
    if forced:
        print(f"Forced holdings: {forced}")
    if exclude_set:
        print(f"Excluding tickers from optimisation: {sorted(exclude_set)}")

    constraints = ContestConstraints(min_stocks=5, min_weight=5, max_weight=25, step=1, allow_cash=True, max_cash=max_cash)

    lb = load_leaderboard(leaderboard_csv)
    players, values, ports = build_player_portfolios(lb, symbol_alias_map=symbol_alias_map)
    if anchor_name not in players:
        raise ValueError(f"Anchor {anchor_name!r} not found in Leaderboard.csv")
    ports = {p: filter_portfolio_to_corr(ports[p], corr) for p in players}

    # General optimisation universe: all tickers in the selected objective volatility file and correlation matrix.
    opt_vols_for_universe = vols_opt
    available_tickers = sorted(set(corr.index) & set(opt_vols_for_universe.keys()))
    tickers_for_anchor = [t for t in available_tickers if t not in exclude_set]
    for t in forced:
        if t not in tickers_for_anchor:
            tickers_for_anchor.append(t)
    tickers_for_anchor = sorted(set(tickers_for_anchor))
    if len(tickers_for_anchor) < constraints.min_stocks:
        raise ValueError("After exclusions, fewer than min_stocks tickers remain in the optimisation universe.")

    anchor_port: Dict[str, float]

    if objective == "current":
        anchor_port = ports[anchor_name]
        if forced and not is_legal_portfolio(anchor_port, constraints, forced):
            raise ValueError("--objective current was selected, but the current anchor portfolio does not satisfy --forced.")
        print(f"\nUsing current {anchor_name} portfolio from Leaderboard.csv (no optimisation):")

    elif objective == "te":
        players_ex = [p for p in players if p != anchor_name]
        values_ex = {p: values[p] for p in players_ex}
        universe_ex = make_universe_from_players(players_ex, ports)
        corr_u_wp_ex, vol_vec_wp_ex = ensure_corr_and_vol_coverage(universe_ex, corr, vols_wp)
        Sigma_ann_wp_ex = cov_from_corr_vol(corr_u_wp_ex, vol_vec_wp_ex)
        u_index_wp_ex = {t: i for i, t in enumerate(universe_ex)}
        W_current_ex = {p: vectorize_port(ports[p], u_index_wp_ex) for p in players_ex}
        W_mat_current_ex = np.stack([W_current_ex[p] for p in players_ex], axis=0)
        win_probs = load_or_compute_win_probs_excl_anchor(winprob_path, players_ex, values_ex, W_mat_current_ex, Sigma_ann_wp_ex, days_remaining, seed)
        bench = build_weighted_benchmark(players_ex, ports, win_probs)
        bench = filter_portfolio_to_corr(bench, corr)
        for t in list(bench.keys()):
            if t in exclude_set:
                bench.pop(t)
        sb = sum(bench.values())
        if sb > 0:
            for k in list(bench.keys()):
                bench[k] /= sb
        pd.DataFrame([{"Ticker": k, "Weight": v} for k, v in sorted(bench.items(), key=lambda x: x[1], reverse=True)]).to_csv("weighted_benchmark.csv", index=False)
        print("Wrote: weighted_benchmark.csv")

        universe0 = sorted(set(tickers_for_anchor) | set(bench.keys()) | {"CASH"})
        corr_u_te, vol_vec_te = ensure_corr_and_vol_coverage(universe0, corr, vols_opt)
        Sigma_ann_te = cov_from_corr_vol(corr_u_te, vol_vec_te)
        u_index = {t: i for i, t in enumerate(universe0)}
        w_b = vectorize_port(bench, u_index)

        def te_obj(w: np.ndarray) -> Tuple[float, float]:
            d = w - w_b
            te = float(math.sqrt(max(d @ Sigma_ann_te @ d, 0.0)))
            return te, te

        print(f"\nOptimising for minimum tracking error using {opt_vol_source.upper()}-TERM vols.")
        anchor_port = cross_entropy_optimise(
            rng=np.random.default_rng(seed),
            tickers=[t for t in tickers_for_anchor if t != "CASH"],
            constraints=constraints,
            forced=forced,
            objective_fn=te_obj,
            u_index=u_index,
            initial_bias=bench,
            n_rounds=rounds,
            samples_per_round=samples,
            maximise=False,
            label="TE",
            polish_steps=1200,
        )
        te_vs_bench = tracking_error_portfolios(anchor_port, bench, corr_u_te, vol_vec_te)
        print(f"\nTE(selected {anchor_name}, benchmark) {opt_vol_source.upper()}-TERM vols (ann.): {te_vs_bench:.6f}")
        print(f"\nTracking-error-minimising portfolio for {anchor_name} (contest-legal, cash capped):")

    elif objective == "win":
        universe0 = sorted(set(tickers_for_anchor) | set(make_universe_from_players(players, ports)) | {"CASH"})
        corr_u_opt, vol_vec_opt = ensure_corr_and_vol_coverage(universe0, corr, vols_opt)
        Sigma_ann_opt = cov_from_corr_vol(corr_u_opt, vol_vec_opt)
        u_index = {t: i for i, t in enumerate(universe0)}
        W_full_current = {p: vectorize_port(ports[p], u_index) for p in players}
        ctx_mc = make_mc_objective_context(players, values, W_full_current, Sigma_ann_opt, days_remaining, inner_sims, seed, anchor_name)

        vol_bias = {t: opt_vols_for_universe.get(t, 0.0) for t in tickers_for_anchor}
        print(f"\nOptimising for win probability using {opt_vol_source.upper()}-TERM vols (inner MC objective, S={inner_sims}).")
        anchor_port = cross_entropy_optimise(
            rng=np.random.default_rng(seed),
            tickers=[t for t in tickers_for_anchor if t != "CASH"],
            constraints=constraints,
            forced=forced,
            objective_fn=lambda w: mc_win_prob_first(w, ctx_mc),
            u_index=u_index,
            initial_bias=vol_bias,
            n_rounds=rounds,
            samples_per_round=samples,
            maximise=True,
            label="P1st",
            polish_steps=800,
        )
        print(f"\nWin-probability-optimised portfolio for {anchor_name} (contest-legal, cash capped):")

    elif objective == "vol":
        print(f"\nOptimising for maximum volatility using {opt_vol_source.upper()}-TERM vols.")
        anchor_port, best_var = solve_max_vol_portfolio(tickers_for_anchor, corr, vols_opt, forced, seed)
        best_vol = math.sqrt(max(best_var, 0.0))
        print(f"\nMaximum-volatility portfolio for {anchor_name} (contest-legal):")
        print(f"Portfolio volatility: {best_vol:.4%}")
        print(f"Portfolio variance:   {best_var:.8f}")

    else:
        raise ValueError("--objective must be 'te', 'win', 'vol', or 'current'.")

    if not is_legal_portfolio(anchor_port, constraints, forced):
        raise RuntimeError(f"Internal error: chosen {anchor_name} portfolio violates constraints.")

    for t, wt in sorted(anchor_port.items(), key=lambda x: x[1], reverse=True):
        print(f"  {t:8s}  {wt * 100:6.2f}%")

    # Final full contest simulation including chosen anchor.
    ports_adj = dict(ports)
    ports_adj[anchor_name] = anchor_port
    universe_full = make_universe_from_players(players, ports_adj)
    corr_u_sim, vol_vec_sim = ensure_corr_and_vol_coverage(universe_full, corr, vols_sim)
    Sigma_ann_sim = cov_from_corr_vol(corr_u_sim, vol_vec_sim)
    u_index_sim = {t: i for i, t in enumerate(universe_full)}
    W_full = {p: vectorize_port(ports_adj[p], u_index_sim) for p in players}
    W_mat_full = np.stack([W_full[p] for p in players], axis=0)
    finish_full, _, n_used = simulate_finish_probs(players, values, W_mat_full, Sigma_ann_sim, days_remaining, n_sims=n_sims, seed=seed, return_counts=True)

    if anchor_name in finish_full:
        pa = finish_full[anchor_name]
        print(f"\n{anchor_name} finish probabilities (Monte Carlo) with ~95% MOE:")
        for k in ["P1st", "P2nd", "P3rd", "PWorse"]:
            p_hat = float(pa[k])
            moe = moe95(p_hat, n_used)
            print(f"  {k:6s}: {p_hat * 100:7.3f}%  (±{moe * 100:5.3f}%)")

    w_a = W_full[anchor_name]
    te_to_a = {}
    ann_vol = {}
    for p in players:
        w_p = W_full[p]
        diff = w_a - w_p
        te_to_a[p] = float(math.sqrt(max(diff @ Sigma_ann_sim @ diff, 0.0)))
        ann_vol[p] = float(math.sqrt(max(w_p @ Sigma_ann_sim @ w_p, 0.0)))

    out_rows = []
    for p in players:
        out_rows.append({
            "Player": p,
            "CurrentValue": values[p],
            "P1st": finish_full[p]["P1st"],
            "P2nd": finish_full[p]["P2nd"],
            "P3rd": finish_full[p]["P3rd"],
            "PWorse": finish_full[p]["PWorse"],
            f"TrackingError_to_{anchor_name}": te_to_a[p],
            "AnnualizedVolatility": ann_vol[p],
        })
    pd.DataFrame(out_rows).sort_values("P1st", ascending=False).to_csv("contest_results.csv", index=False)
    print("\nWrote: contest_results.csv")


if __name__ == "__main__":
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="YAML or JSON configuration file. Default: config.yaml if present.",
    )
    pre_args, _ = pre_parser.parse_known_args()

    file_config = load_config_file(pre_args.config)
    cfg = deep_update(DEFAULT_CONFIG, file_config)

    parser = argparse.ArgumentParser(
        description="Combined contest optimiser.",
        parents=[pre_parser],
    )
    parser.add_argument("--objective", choices=["te", "win", "vol", "current"], default=cfg_get(cfg, "objective"), help="Portfolio objective.")
    parser.add_argument("--anchor", type=str, default=cfg_get(cfg, "anchor"), help="Leaderboard player name to optimise/replace.")
    parser.add_argument("--stocklist", type=str, default=cfg_get(cfg, "files.stocklist_csv"), help="CSV containing Symbol and optional Alternative Symbol columns.")
    parser.add_argument("--leaderboard", type=str, default=cfg_get(cfg, "files.leaderboard_csv"), help="Leaderboard CSV.")
    parser.add_argument("--corr-file", type=str, default=cfg_get(cfg, "files.corr_file"), help="Correlation matrix CSV.")
    parser.add_argument("--vol-file", type=str, default=cfg_get(cfg, "files.vol_short_csv"), help="Short-term volatility CSV.")
    parser.add_argument("--vol-file2", type=str, default=cfg_get(cfg, "files.vol_medium_csv"), help="Medium-term volatility CSV.")

    parser.add_argument("--days", type=int, default=cfg_get(cfg, "days"), help="Trading days remaining.")
    parser.add_argument("--sims", type=int, default=cfg_get(cfg, "final_simulation.sims"), help="Number of Monte Carlo simulations for final reporting.")
    parser.add_argument("--seed", type=int, default=cfg_get(cfg, "final_simulation.seed"), help="RNG seed.")

    parser.add_argument("--wp_vol_source", choices=["short", "medium"], default=cfg_get(cfg, "model.wp_vol_source"), help="Vols used only when --objective te needs to compute the benchmark win-probability file.")
    parser.add_argument("--opt_vol_source", choices=["short", "medium"], default=cfg_get(cfg, "model.opt_vol_source"), help="Vols used for TE, WIN, or VOL optimisation.")
    parser.add_argument("--sim_vol_source", choices=["short", "medium"], default=cfg_get(cfg, "model.sim_vol_source"), help="Vols used for final full contest simulation.")

    parser.add_argument("--inner_sims", type=int, default=cfg_get(cfg, "win_objective.inner_sims"), help="Inner simulations for --objective win.")
    parser.add_argument("--rounds", type=int, default=cfg_get(cfg, "win_objective.rounds"), help="Optimisation rounds for --objective te or win.")
    parser.add_argument("--samples", type=int, default=cfg_get(cfg, "win_objective.samples"), help="Samples per round for --objective te or win.")

    parser.add_argument("--exclude", type=str, default=cfg_get(cfg, "exclude"), help="Comma-separated tickers to exclude from optimisation universe.")
    parser.add_argument("--forced", type=str, default=cfg_get(cfg, "forced"), help='Forced holdings, e.g. "AAPL=25,NVDA=15,SHOP=10".')
    parser.add_argument("--max_cash", type=int, default=cfg_get(cfg, "model.max_cash"), help="Maximum CASH percent allowed in anchor portfolio.")
    parser.add_argument("--winprob-file", type=str, default=cfg_get(cfg, "files.winprob_path"), help="Win-probability CSV used/created only by --objective te.")

    args = parser.parse_args()
    days = get_days_remaining(args.days)

    print(
        f"Objective: {args.objective} | anchor={args.anchor} | corr={args.corr_file} | "
        f"config={args.config} | "
        f"Vol sources: win-prob={args.wp_vol_source}, opt={args.opt_vol_source}, sim={args.sim_vol_source} | "
        f"max_cash={args.max_cash}%"
    )

    run_all(
        objective=args.objective,
        leaderboard_csv=args.leaderboard,
        corr_csv=args.corr_file,
        vol_short_csv=args.vol_file,
        vol_medium_csv=args.vol_file2,
        days_remaining=days,
        n_sims=args.sims,
        seed=args.seed,
        winprob_path=args.winprob_file,
        wp_vol_source=args.wp_vol_source,
        opt_vol_source=args.opt_vol_source,
        sim_vol_source=args.sim_vol_source,
        inner_sims=args.inner_sims,
        rounds=args.rounds,
        samples=args.samples,
        exclude_tickers=args.exclude,
        forced_holdings=args.forced,
        max_cash=args.max_cash,
        anchor_name=args.anchor,
        stocklist_csv=args.stocklist,
    )