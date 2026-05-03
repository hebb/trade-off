#!/usr/bin/env python3
"""
ContestAnalytics.py

This script analyses a stock-picking contest and constructs a contest-legal portfolio
for the anchor player ("ahebb") that minimises tracking error (TE) to a benchmark
portfolio. It does not optimise win probability for ahebb; it only minimises TE to
a benchmark built from other players.

The benchmark is a probability-weighted average of the other players’ portfolios,
where the weights come from WinProbabilities_excl_ahebb.csv (computed separately, or
loaded if the file already exists).

Tracking error is defined as:
    TE = sqrt((w - w_b)^T Σ (w - w_b))
where Σ is the stock-level covariance matrix built from implied volatilities and a
combined correlation matrix.

Contest constraints enforced for ahebb:
  - At least 5 stocks
  - Each stock weight between 5% and 25%
  - 1% increments
  - No shorting
  - Cash allowed

Volatility-source options (new; defaults preserve prior behaviour):
  --wp_vol_source {short,medium}
      Volatilities used when (re)computing WinProbabilities_excl_ahebb.csv (default: medium).
  --te_vol_source {short,medium}
      Volatilities used for TE optimisation and TE(selected ahebb, benchmark) reporting (default: short).
  --sim_vol_source {short,medium}
      Volatilities used for the final full contest simulation after replacing ahebb (default: medium).

Arguments:
  --days N   Trading days remaining (if omitted, prompts interactively)
  --sims N   Monte Carlo simulations (default: 400000)
  --seed N   RNG seed (default: 0)

Outputs (written to current directory):
  - WinProbabilities_excl_ahebb.csv (created if missing)
  - weighted_benchmark.csv
  - contest_results.csv

Note:
This version does not write the selected ahebb TE-minimising portfolio to a file.
"""

from __future__ import annotations

import math
import os
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------

def norm_ticker(t: str) -> str:
    t = str(t).strip()
    t = t.replace(":CA", "")
    t = t.replace(".TO", "")
    return t


def parse_pct(x) -> float:
    if isinstance(x, str):
        s = x.strip()
        if s.endswith("%"):
            return float(s[:-1]) / 100.0
        return float(s)
    return float(x)


def parse_weights_string(s: str) -> Optional[List[float]]:
    if not isinstance(s, str) or not s.strip():
        return None
    parts = [p.strip() for p in s.split(",")]
    try:
        return [float(p) for p in parts]
    except Exception:
        return None


def portfolio_signature(port: Dict[str, float], round_to: int = 6) -> Tuple[Tuple[str, float], ...]:
    items = sorted((k, round(v, round_to)) for k, v in port.items())
    return tuple(items)


def filter_portfolio_to_corr(port: Dict[str, float], corr: pd.DataFrame) -> Dict[str, float]:
    """
    Remove tickers not in corr (except CASH), and renormalize to sum to 1.
    Any removed weight becomes CASH.
    """
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


# -----------------------------
# Loaders
# -----------------------------

def load_corr_matrix(path: str) -> pd.DataFrame:
    corr = pd.read_csv(path, index_col=0)
    corr.index = [norm_ticker(i) for i in corr.index]
    corr.columns = [norm_ticker(c) for c in corr.columns]
    corr = corr.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    corr = (corr + corr.T) / 2.0
    n = len(corr)
    for i in range(n):
        corr.iat[i, i] = 1.0
    return corr


def combine_monthly_daily_corr(corr_monthly: pd.DataFrame, corr_daily: pd.DataFrame) -> pd.DataFrame:
    cm = corr_monthly.copy(deep=True)
    common = cm.index.intersection(corr_daily.index)
    if len(common) != corr_daily.shape[0]:
        missing = sorted(set(corr_daily.index) - set(cm.index))
        raise ValueError(
            "Daily matrix tickers not fully contained in monthly matrix. "
            f"Missing: {missing}"
        )
    cm.loc[common, common] = corr_daily.loc[common, common]
    cm = (cm + cm.T) / 2.0
    n = len(cm)
    for i in range(n):
        cm.iat[i, i] = 1.0
    return cm


def load_vols(path: str) -> Dict[str, float]:
    vols = pd.read_csv(path)
    if "Ticker" not in vols.columns or "Implied Volatility" not in vols.columns:
        raise ValueError(f"Vol file {path} missing expected columns Ticker / Implied Volatility")
    vols["Ticker_norm"] = vols["Ticker"].apply(norm_ticker)
    vols["vol"] = vols["Implied Volatility"].apply(parse_pct)
    return dict(zip(vols["Ticker_norm"], vols["vol"]))


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
# Portfolios
# -----------------------------

def parse_portfolio_from_row(row: pd.Series) -> Dict[str, float]:
    holdings = [norm_ticker(x) for x in str(row["Holdings"]).split(",") if str(x).strip()]
    w_list = parse_weights_string(row.get("Weights", ""))

    if (w_list is None) or (len(w_list) != len(holdings)):
        w = np.ones(len(holdings), dtype=float) / len(holdings)
    else:
        w = np.array(w_list, dtype=float)

    port: Dict[str, float] = {}
    for t, wt in zip(holdings, w):
        port[t] = port.get(t, 0.0) + float(wt)

    rem = 1.0 - sum(port.values())
    if rem > 1e-8:
        port["CASH"] = port.get("CASH", 0.0) + float(rem)

    s = sum(port.values())
    if abs(s - 1.0) > 1e-8:
        for k in list(port.keys()):
            port[k] /= s

    return port


def build_player_portfolios(
    lb: pd.DataFrame,
    drop_duplicates: bool = True
) -> Tuple[List[str], Dict[str, float], Dict[str, Dict[str, float]]]:
    ports: Dict[str, Dict[str, float]] = {}
    values: Dict[str, float] = {}

    for _, row in lb.iterrows():
        name = str(row["Name"])
        values[name] = float(row["Value"])
        ports[name] = parse_portfolio_from_row(row)

    if not drop_duplicates:
        return list(ports.keys()), values, ports

    seen = set()
    players: List[str] = []
    for name in lb["Name"].tolist():
        sig = (round(values[name], 6), portfolio_signature(ports[name], round_to=6))
        if sig in seen:
            continue
        seen.add(sig)
        players.append(name)

    return players, values, ports


# -----------------------------
# Risk model
# -----------------------------

def make_universe(players: List[str], ports: Dict[str, Dict[str, float]]) -> List[str]:
    tickers = set()
    for p in players:
        tickers.update(ports[p].keys())
    tickers.add("CASH")
    return sorted(tickers)


def ensure_corr_and_vol_coverage(
    universe: List[str],
    corr: pd.DataFrame,
    vol_dict: Dict[str, float],
) -> Tuple[pd.DataFrame, np.ndarray]:
    n = len(universe)
    corr_u = pd.DataFrame(np.eye(n), index=universe, columns=universe, dtype=float)

    for i, a in enumerate(universe):
        for j, b in enumerate(universe):
            if a == b:
                corr_u.iat[i, j] = 1.0
            elif a == "CASH" or b == "CASH":
                corr_u.iat[i, j] = 0.0
            else:
                if (a in corr.index) and (b in corr.columns):
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
                raise ValueError(f"Missing volatility for ticker '{t}'. Add it to your vol file(s).")
            vol_vec[i] = float(vol_dict[t])

    # PSD-fix correlation
    C = corr_u.to_numpy(dtype=float, copy=True)
    C = (C + C.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.clip(eigvals, 1e-10, None)
    C_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(C_psd))
    C_psd = C_psd / (d[:, None] * d[None, :])

    corr_u_psd = pd.DataFrame(C_psd, index=universe, columns=universe)
    return corr_u_psd, vol_vec


def cov_from_corr_vol(corr_u: pd.DataFrame, vol_vec: np.ndarray) -> np.ndarray:
    D = np.diag(vol_vec)
    return D @ corr_u.to_numpy(dtype=float, copy=True) @ D


def vectorize_port(port: Dict[str, float], universe_index: Dict[str, int]) -> np.ndarray:
    w = np.zeros(len(universe_index), dtype=float)
    for t, wt in port.items():
        w[universe_index[t]] += float(wt)
    s = w.sum()
    if abs(s - 1.0) > 1e-8:
        w /= s
    return w


def tracking_error_portfolios(
    port_a: Dict[str, float],
    port_b: Dict[str, float],
    corr_u: pd.DataFrame,
    vol_vec: np.ndarray,
) -> float:
    universe = list(corr_u.index)
    u_index = {t: i for i, t in enumerate(universe)}
    Sigma_ann = cov_from_corr_vol(corr_u, vol_vec)
    w_a = vectorize_port(port_a, u_index)
    w_b = vectorize_port(port_b, u_index)
    diff = w_a - w_b
    return float(math.sqrt(max(diff @ Sigma_ann @ diff, 0.0)))


# -----------------------------
# Benchmark
# -----------------------------

def build_weighted_benchmark(
    players: List[str],
    ports: Dict[str, Dict[str, float]],
    win_probs: Dict[str, float],
) -> Dict[str, float]:
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


# -----------------------------
# TE-min portfolio for ahebb
# -----------------------------

@dataclass
class ContestConstraints:
    min_stocks: int = 5
    min_weight: int = 5
    max_weight: int = 25
    step: int = 1
    allow_cash: bool = True



def greedy_te_min_portfolio(
    benchmark: Dict[str, float],
    corr_u: pd.DataFrame,
    vol_vec: np.ndarray,
    constraints: ContestConstraints,
) -> Dict[str, float]:
    """
    Improved TE optimiser (drop-in replacement).

    Strategy:
      1) Propose several candidate stock sets (top benchmark weights + weighted random restarts).
      2) For each set, build a contest-legal integer portfolio by rounding benchmark weights
         and assigning the residual to CASH (if allowed).
      3) Run a local search that performs 1% transfers between holdings (and CASH) to reduce TE.

    Returns a contest-legal portfolio dict over corr_u.index.
    """
    universe = list(corr_u.index)
    u_index = {t: i for i, t in enumerate(universe)}
    if constraints.allow_cash and "CASH" not in u_index:
        raise ValueError("CASH must be present in corr_u index when allow_cash=True.")

    Sigma_ann = cov_from_corr_vol(corr_u, vol_vec)
    Sigma_ann = (Sigma_ann + Sigma_ann.T) / 2.0  # numerical safety

    w_b = vectorize_port(benchmark, u_index)

    # Objective is annualised TE (sqrt of quadratic form), matching existing reporting.
    def te_of(w: np.ndarray) -> float:
        d = w - w_b
        q = float(d @ Sigma_ann @ d)
        return float(math.sqrt(max(q, 0.0)))

    # Quadratic-form delta for f(d)=d^T S d (no sqrt); we use it inside local search.
    def delta_q(d: np.ndarray, delta: np.ndarray) -> float:
        Sd = Sigma_ann @ d
        return float(2.0 * (delta @ Sd) + (delta @ (Sigma_ann @ delta)))

    minw, maxw, step = constraints.min_weight, constraints.max_weight, constraints.step
    if step != 1:
        if 100 % step != 0:
            raise ValueError("constraints.step must divide 100.")
    step_pct = step
    step_w = step_pct / 100.0
    min_wu = minw / 100.0
    max_wu = maxw / 100.0

    # Candidate tickers: only those in corr_u (derived from correlation coverage)
    candidates = [(t, float(benchmark.get(t, 0.0))) for t in universe if t != "CASH"]
    candidates.sort(key=lambda x: x[1], reverse=True)

    if len(candidates) < constraints.min_stocks:
        raise ValueError("Not enough candidate tickers to satisfy min_stocks.")

    # Build benchmark-aware integer-feasible weights for a chosen set (excluding CASH).
    def make_feasible(chosen: List[str]) -> Optional[np.ndarray]:
        chosen = sorted(set(chosen))
        if len(chosen) < constraints.min_stocks:
            return None
        if (len(chosen) * minw) > 100:
            return None

        targets = np.array([max(0.0, float(benchmark.get(t, 0.0))) for t in chosen], dtype=float)
        if targets.sum() <= 0:
            targets = np.ones(len(chosen), dtype=float)
        targets = targets / targets.sum()

        # Round to integer percent increments, then clip to bounds.
        w_int = np.rint((100.0 * targets) / step_pct).astype(int) * step_pct
        w_int = np.clip(w_int, minw, maxw)

        def ssum() -> int:
            return int(w_int.sum())

        cash_pct = 0

        # Reduce if overweight: peel from the most "over-target" names, respecting minw.
        while ssum() > 100:
            over = (w_int / 100.0) - targets
            can = np.where(w_int > minw)[0]
            if len(can) == 0:
                return None
            idx = int(can[np.argmax(over[can])])
            w_int[idx] -= step_pct

        # Add if underweight: assign to CASH if allowed, else add to most "under-target" names.
        while ssum() < 100:
            if constraints.allow_cash:
                cash_pct = 100 - ssum()
                break
            under = targets - (w_int / 100.0)
            can = np.where(w_int < maxw)[0]
            if len(can) == 0:
                return None
            idx = int(can[np.argmax(under[can])])
            w_int[idx] += step_pct

        w = np.zeros(len(universe), dtype=float)
        for t, pct in zip(chosen, w_int):
            w[u_index[t]] = pct / 100.0
        if constraints.allow_cash:
            w[u_index["CASH"]] = cash_pct / 100.0

        s = float(w.sum())
        if abs(s - 1.0) > 1e-12:
            w /= s

        # Validate bounds (non-cash) and min holdings.
        noncash = [t for t in chosen]
        for t in noncash:
            wt = w[u_index[t]]
            if wt + 1e-12 < min_wu or wt - 1e-12 > max_wu:
                return None
        return w

    # Local search: repeatedly apply the single best 1% transfer (i -> j) to reduce quadratic TE.
    def local_improve(w: np.ndarray, max_iters: int = 25000) -> np.ndarray:
        cash_i = u_index.get("CASH", None) if constraints.allow_cash else None
        d = w - w_b
        q_cur = float(d @ Sigma_ann @ d)

        noncash_idx = [u_index[t] for t, _ in candidates]
        noncash_set = set(noncash_idx)

        def count_noncash_holdings(wv: np.ndarray) -> int:
            c = 0
            for i in noncash_set:
                if wv[i] > 1e-12:
                    c += 1
            return c

        for _ in range(max_iters):
            best_dq = 0.0
            best_move = None  # (i, j)

            for i in range(len(universe)):
                wi = w[i]
                if wi <= 1e-15:
                    continue

                # Check if we can take 1% from i.
                if (cash_i is not None) and (i == cash_i):
                    if wi - step_w < -1e-15:
                        continue
                else:
                    if wi - step_w < min_wu - 1e-15:
                        continue

                for j in range(len(universe)):
                    if j == i:
                        continue

                    wj = w[j]
                    if (cash_i is not None) and (j == cash_i):
                        # cash has no upper bound
                        pass
                    else:
                        if wj <= 1e-15:
                            # creating a new non-cash holding: must land >= min_wu
                            if step_w + 1e-15 < min_wu:
                                continue
                        if wj + step_w > max_wu + 1e-15:
                            continue

                    # Keep at least min_stocks non-cash holdings.
                    if (cash_i is None) or (i != cash_i):
                        if (i in noncash_set) and (w[i] - step_w <= 1e-12):
                            if count_noncash_holdings(w) - 1 < constraints.min_stocks:
                                continue

                    delta = np.zeros(len(universe), dtype=float)
                    delta[i] -= step_w
                    delta[j] += step_w
                    dq = delta_q(d, delta)
                    if dq < best_dq - 1e-18:
                        best_dq = dq
                        best_move = (i, j)

            if best_move is None:
                break

            i, j = best_move
            w[i] -= step_w
            w[j] += step_w

            delta = np.zeros(len(universe), dtype=float)
            delta[i] -= step_w
            delta[j] += step_w
            d = d + delta
            q_cur = q_cur + best_dq

        # Snap to grid.
        w_pct = np.rint(w * 100.0 / step_pct) * step_pct
        w = (w_pct / 100.0).astype(float)

        # Reconcile sum to 1.0 via cash if allowed, else normalise.
        if constraints.allow_cash:
            w[u_index["CASH"]] = 1.0 - float(np.sum([w[u_index[t]] for t, _ in candidates if w[u_index[t]] > 1e-12]))
            if w[u_index["CASH"]] < -1e-9:
                w = np.clip(w, 0.0, None)
                s = float(w.sum())
                if s > 0:
                    w /= s
        else:
            w = np.clip(w, 0.0, None)
            s = float(w.sum())
            if s > 0:
                w /= s

        return w

    pool_size = min(40, len(candidates))
    pool = candidates[:pool_size]
    pool_tickers = [t for t, _ in pool]
    pool_weights = np.array([max(w, 0.0) for _, w in pool], dtype=float)
    if pool_weights.sum() <= 0:
        pool_weights = np.ones(len(pool_weights), dtype=float) / len(pool_weights)
    else:
        pool_weights /= pool_weights.sum()

    rng = np.random.default_rng(0)

    def propose_set(k: int, mode: str) -> List[str]:
        k = max(k, constraints.min_stocks)
        k = min(k, max(constraints.min_stocks, 12))
        if mode == "top":
            return pool_tickers[:k]
        idx = rng.choice(len(pool_tickers), size=k, replace=False, p=pool_weights)
        return [pool_tickers[int(i)] for i in idx]

    best_w = None
    best_te = float("inf")

    for k in [constraints.min_stocks, 6, 7, 8, 9, 10, 12]:
        if k < constraints.min_stocks:
            continue
        w0 = make_feasible(propose_set(k, "top"))
        if w0 is None:
            continue
        w1 = local_improve(w0)
        te1 = te_of(w1)
        if te1 < best_te:
            best_te = te1
            best_w = w1

    n_restarts = 60
    for _ in range(n_restarts):
        k = int(rng.integers(constraints.min_stocks, 13))
        w0 = make_feasible(propose_set(k, "rand"))
        if w0 is None:
            continue
        w1 = local_improve(w0)
        te1 = te_of(w1)
        if te1 < best_te:
            best_te = te1
            best_w = w1

    assert best_w is not None

    port: Dict[str, float] = {}
    for t in universe:
        wt = float(best_w[u_index[t]])
        if wt > 1e-8:
            port[t] = wt
    s = sum(port.values())
    for k in list(port.keys()):
        port[k] /= s
    return port


# -----------------------------
# Simulation
# -----------------------------

def simulate_finish_probs(
    players: List[str],
    values: Dict[str, float],
    W_mat: np.ndarray,
    Sigma_ann: np.ndarray,
    days_remaining: int,
    n_sims: int = 400_000,
    seed: int = 0,
    return_counts: bool = False,
):
    rng = np.random.default_rng(seed)
    Sigma_day = Sigma_ann / 252.0
    mu_day = -0.5 * np.diag(Sigma_day)

    C = (Sigma_day + Sigma_day.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.clip(eigvals, 1e-14, None)
    C_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    L = np.linalg.cholesky(C_psd + 1e-16 * np.eye(C_psd.shape[0]))

    nP = len(players)
    logV0 = np.array([math.log(values[p]) for p in players], dtype=float)

    counts = {p: np.zeros(4, dtype=np.int64) for p in players}
    batch = 20_000

    for start in range(0, n_sims, batch):
        b = min(batch, n_sims - start)
        Z = rng.standard_normal(size=(b, days_remaining, C_psd.shape[0]))
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


# -----------------------------
# Days input
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


# -----------------------------
# Win probs caching
# -----------------------------

def load_or_compute_win_probs_excl_ahebb(
    winprob_path: str,
    players_ex: List[str],
    values_ex: Dict[str, float],
    W_mat_current_ex: np.ndarray,
    Sigma_ann_med_ex: np.ndarray,
    days_remaining: int,
    seed: int,
) -> Dict[str, float]:
    if os.path.exists(winprob_path):
        wp = pd.read_csv(winprob_path)
        if "Name" not in wp.columns or "WinProb" not in wp.columns:
            raise ValueError(f"{winprob_path} must contain columns Name, WinProb")
        probs = {str(n): float(w) for n, w in zip(wp["Name"], wp["WinProb"])}
        probs = {p: probs[p] for p in players_ex if p in probs}
        s = sum(probs.values())
        if s <= 0:
            raise ValueError(f"{winprob_path} sums to <= 0 over non-ahebb players.")
        return {p: w / s for p, w in probs.items()}

    finish_ex = simulate_finish_probs(
        players_ex, values_ex, W_mat_current_ex, Sigma_ann_med_ex,
        days_remaining, n_sims=200_000, seed=seed
    )
    probs = {p: finish_ex[p]["P1st"] for p in players_ex}
    s = sum(probs.values())
    probs = {p: w / s for p, w in probs.items()}

    pd.DataFrame({"Name": list(probs.keys()), "WinProb": list(probs.values())}) \
      .sort_values("WinProb", ascending=False) \
      .to_csv(winprob_path, index=False)

    print(f"Wrote: {winprob_path} (created because it was missing)")
    return probs


# -----------------------------
# Main
# -----------------------------

def run_all(
    leaderboard_csv: str,
    corr_monthly_csv: str,
    corr_daily_csv: str,
    vol_short_csv: str,
    vol_medium_csv: str,
    days_remaining: int,
    n_sims: int = 400_000,
    seed: int = 0,
    drop_duplicates: bool = True,
    winprob_path: str = "WinProbabilities_excl_ahebb.csv",
    wp_vol_source: str = "medium",
    te_vol_source: str = "short",
    sim_vol_source: str = "medium",
) -> None:
    lb = load_leaderboard(leaderboard_csv)
    players, values, ports = build_player_portfolios(lb, drop_duplicates=drop_duplicates)

    corr_m = load_corr_matrix(corr_monthly_csv)
    corr_d = load_corr_matrix(corr_daily_csv)
    corr_combined = combine_monthly_daily_corr(corr_m, corr_d)

    # Filter portfolios to correlation coverage (plus CASH)
    ports = {p: filter_portfolio_to_corr(ports[p], corr_combined) for p in players}

    vol_short = load_vols(vol_short_csv)
    vol_med = load_vols(vol_medium_csv)

    vols_wp = vol_short if wp_vol_source == "short" else vol_med
    vols_te = vol_short if te_vol_source == "short" else vol_med
    vols_sim = vol_short if sim_vol_source == "short" else vol_med

    # --- Benchmark win probs: ahebb excluded entirely ---
    players_ex = [p for p in players if p != "ahebb"]
    values_ex = {p: values[p] for p in players_ex}

    universe_ex = make_universe(players_ex, ports)
    corr_u_med_ex, vol_vec_med_ex = ensure_corr_and_vol_coverage(universe_ex, corr_combined, vols_wp)
    Sigma_ann_med_ex = cov_from_corr_vol(corr_u_med_ex, vol_vec_med_ex)
    u_index_med_ex = {t: i for i, t in enumerate(universe_ex)}

    W_current_ex = {p: vectorize_port(ports[p], u_index_med_ex) for p in players_ex}
    W_mat_current_ex = np.stack([W_current_ex[p] for p in players_ex], axis=0)

    win_probs = load_or_compute_win_probs_excl_ahebb(
        winprob_path,
        players_ex,
        values_ex,
        W_mat_current_ex,
        Sigma_ann_med_ex,
        days_remaining,
        seed,
    )

    bench = build_weighted_benchmark(players_ex, ports, win_probs)
    bench = filter_portfolio_to_corr(bench, corr_combined)

    pd.DataFrame([{"Ticker": k, "Weight": v} for k, v in sorted(bench.items(), key=lambda x: x[1], reverse=True)]) \
      .to_csv("weighted_benchmark.csv", index=False)
    print("Wrote: weighted_benchmark.csv")

    # --- Optimize ahebb vs benchmark using chosen TE vols ---
    universe0 = make_universe(players, ports)
    corr_u_short, vol_vec_short = ensure_corr_and_vol_coverage(universe0, corr_combined, vols_te)

    constraints = ContestConstraints(min_stocks=5, min_weight=5, max_weight=25, step=1, allow_cash=True)
    ahebb_te_min = greedy_te_min_portfolio(bench, corr_u_short, vol_vec_short, constraints)

    te_vs_bench_short = tracking_error_portfolios(ahebb_te_min, bench, corr_u_short, vol_vec_short)
    print(f"\nTE(selected ahebb, benchmark) {te_vol_source.upper()}-TERM vols (ann.): {te_vs_bench_short:.6f}")

    corr_u_med0, vol_vec_med0 = ensure_corr_and_vol_coverage(universe0, corr_combined, vol_med)
    te_vs_bench_med = tracking_error_portfolios(ahebb_te_min, bench, corr_u_med0, vol_vec_med0)
    print(f"TE(selected ahebb, benchmark) MEDIUM-TERM vols (ann.): {te_vs_bench_med:.6f}")

    print("\nTracking-error-minimizing portfolio for ahebb (contest-legal, cash allowed):")
    for t, wt in sorted(ahebb_te_min.items(), key=lambda x: x[1], reverse=True):
        print(f"  {t:6s}  {wt*100:6.2f}%")

    # --- Final full contest sim including ahebb using chosen sim vols ---
    ports_adj = dict(ports)
    ports_adj["ahebb"] = ahebb_te_min

    universe_full = make_universe(players, ports_adj)
    corr_u_med_full, vol_vec_med_full = ensure_corr_and_vol_coverage(universe_full, corr_combined, vols_sim)
    Sigma_ann_med_full = cov_from_corr_vol(corr_u_med_full, vol_vec_med_full)
    u_index_full = {t: i for i, t in enumerate(universe_full)}

    W_full = {p: vectorize_port(ports_adj[p], u_index_full) for p in players}
    W_mat_full = np.stack([W_full[p] for p in players], axis=0)

    finish_full, counts_full, n_used = simulate_finish_probs(
        players, values, W_mat_full, Sigma_ann_med_full,
        days_remaining, n_sims=n_sims, seed=seed, return_counts=True
    )

    if "ahebb" in finish_full:
        pa = finish_full["ahebb"]
        print("\nAHEBB finish probabilities (Monte Carlo) with ~95% MOE:")
        for k in ["P1st", "P2nd", "P3rd", "PWorse"]:
            p_hat = float(pa[k])
            moe = moe95(p_hat, n_used)
            print(f"  {k:6s}: {p_hat*100:7.3f}%  (±{moe*100:5.3f}%)")

    w_a = W_full["ahebb"]
    te_to_a = {}
    ann_vol = {}
    for p in players:
        w_p = W_full[p]
        diff = w_a - w_p
        te_to_a[p] = float(math.sqrt(max(diff @ Sigma_ann_med_full @ diff, 0.0)))
        ann_vol[p] = float(math.sqrt(max(w_p @ Sigma_ann_med_full @ w_p, 0.0)))

    out_rows = []
    for p in players:
        out_rows.append({
            "Player": p,
            "CurrentValue": values[p],
            "P1st": finish_full[p]["P1st"],
            "P2nd": finish_full[p]["P2nd"],
            "P3rd": finish_full[p]["P3rd"],
            "PWorse": finish_full[p]["PWorse"],
            "TrackingError_to_ahebb": te_to_a[p],
            "AnnualizedVolatility": ann_vol[p],
        })

    pd.DataFrame(out_rows).sort_values("P1st", ascending=False).to_csv("contest_results.csv", index=False)
    print("\nWrote: contest_results.csv")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=None, help="Trading days remaining (e.g., 13)")
    parser.add_argument("--sims", type=int, default=400_000, help="Number of Monte Carlo simulations")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--wp_vol_source", choices=["short", "medium"], default="medium",
                        help="Volatilities used when (re)computing WinProbabilities_excl_ahebb.csv (default: medium)")
    parser.add_argument("--te_vol_source", choices=["short", "medium"], default="short",
                        help="Volatilities used for TE optimisation and TE(selected ahebb, benchmark) reporting (default: short)")
    parser.add_argument("--sim_vol_source", choices=["short", "medium"], default="medium",
                        help="Volatilities used for the final full contest simulation after replacing ahebb (default: medium)")
    args = parser.parse_args()

    days = get_days_remaining(args.days)

    print(f"Vol sources: win-prob={args.wp_vol_source}, TE={args.te_vol_source}, sim={args.sim_vol_source}")

    run_all(
        leaderboard_csv="Leaderboard.csv",
        corr_monthly_csv="CorrelationMatrixMonthly.csv",
        corr_daily_csv="CorrelationMatrixDaily.csv",
        vol_short_csv="Volatilities.csv",
        vol_medium_csv="Volatilities2.csv",
        days_remaining=days,
        n_sims=args.sims,
        seed=args.seed,
        drop_duplicates=True,
        winprob_path="WinProbabilities_excl_ahebb.csv",
        wp_vol_source=args.wp_vol_source,
        te_vol_source=args.te_vol_source,
        sim_vol_source=args.sim_vol_source,
    )

