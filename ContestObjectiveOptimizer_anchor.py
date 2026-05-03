#!/usr/bin/env python3
"""
ContestObjectiveOptimizer.py

Integrated contest optimiser with selectable objective:

  --objective te   : choose anchor portfolio by minimising tracking error (TE) to benchmark
  --objective win  : choose anchor portfolio by maximising P(anchor finishes 1st)

Shared contest constraints for anchor:
  - at least 5 stocks
  - each stock weight between 5% and 25%
  - 1% increments
  - cash allowed, but CASH <= --max_cash (default 25%)
  - no shorting
  - universe restricted to tickers in combined correlation matrix
  - unknown tickers dropped and weight goes to CASH (then constraints applied)

Inputs (current directory):
  Leaderboard.csv
  Volatilities.csv             (short)
  Volatilities2.csv            (medium)
  CorrelationMatrixMonthly.csv
  CorrelationMatrixDaily.csv
  WinProbabilities_excl_ahebb.csv  (loaded if present; created if missing)

Outputs (current directory):
  weighted_benchmark.csv
  contest_results.csv
  WinProbabilities_excl_ahebb.csv  (only created if missing)

Notes:
  - TE uses the same improved greedy/local-search optimiser you had, but now enforces CASH cap.
  - WIN uses a cheap inner Monte Carlo objective with common random numbers:
        R_T ~ N(T*mu_day, T*Sigma_day)
    Opponents’ terminal log-values are precomputed; each candidate only computes anchor logV and
    compares to the per-sim opponent max.
"""

from __future__ import annotations

import argparse
import math
import os
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
    cd = corr_daily.copy(deep=True)

    # Force row/column labels to be unique. Duplicate labels can make
    # loc-assignment behave badly even when the visible ticker sets match.
    cm = cm.loc[~cm.index.duplicated(keep="first"), ~cm.columns.duplicated(keep="first")]
    cd = cd.loc[~cd.index.duplicated(keep="first"), ~cd.columns.duplicated(keep="first")]

    daily_tickers = set(cd.index).union(set(cd.columns))
    monthly_tickers = set(cm.index).intersection(set(cm.columns))

    missing = sorted(daily_tickers - monthly_tickers)
    if missing:
        raise ValueError(
            "Daily matrix tickers not fully contained in monthly matrix. "
            f"Missing: {missing}"
        )

    common = sorted(set(cm.index).intersection(cm.columns).intersection(cd.index).intersection(cd.columns))

    cm.loc[common, common] = cd.loc[common, common]

    cm = cm.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    cm = cm.reindex(index=cm.index, columns=cm.index, fill_value=0.0)
    cm = (cm + cm.T) / 2.0

    for i in range(len(cm)):
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
    if abs(s - 1.0) > 1e-8 and s > 0:
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
# Constraints
# -----------------------------

@dataclass
class ContestConstraints:
    min_stocks: int = 5
    min_weight: int = 5
    max_weight: int = 25
    step: int = 1
    allow_cash: bool = True
    max_cash: int = 25  # CASH cap


def is_legal_portfolio(port: Dict[str, float], constraints: ContestConstraints) -> bool:
    minw, maxw = constraints.min_weight / 100.0, constraints.max_weight / 100.0
    step = constraints.step / 100.0
    max_cash = constraints.max_cash / 100.0

    noncash = [(t, w) for t, w in port.items() if t != "CASH" and w > 1e-12]
    if len(noncash) < constraints.min_stocks:
        return False

    s = sum(port.values())
    if abs(s - 1.0) > 5e-6:
        return False

    cash_w = float(port.get("CASH", 0.0))
    if cash_w < -1e-9:
        return False
    if constraints.allow_cash and cash_w > max_cash + 1e-12:
        return False

    for _, w in noncash:
        if w < minw - 1e-12 or w > maxw + 1e-12:
            return False
        if abs(round(w / step) * step - w) > 5e-6:
            return False

    return True


# -----------------------------
# TE-min portfolio for anchor (with CASH cap)
# -----------------------------

def greedy_te_min_portfolio(
    benchmark: Dict[str, float],
    corr_u: pd.DataFrame,
    vol_vec: np.ndarray,
    constraints: ContestConstraints,
    exclude_set: Optional[set] = None,
) -> Dict[str, float]:
    universe = list(corr_u.index)
    u_index = {t: i for i, t in enumerate(universe)}
    if constraints.allow_cash and "CASH" not in u_index:
        raise ValueError("CASH must be present in corr_u index when allow_cash=True.")

    Sigma_ann = cov_from_corr_vol(corr_u, vol_vec)
    Sigma_ann = (Sigma_ann + Sigma_ann.T) / 2.0

    w_b = vectorize_port(benchmark, u_index)

    def te_of(w: np.ndarray) -> float:
        d = w - w_b
        q = float(d @ Sigma_ann @ d)
        return float(math.sqrt(max(q, 0.0)))

    def delta_q(d: np.ndarray, delta: np.ndarray) -> float:
        Sd = Sigma_ann @ d
        return float(2.0 * (delta @ Sd) + (delta @ (Sigma_ann @ delta)))

    minw, maxw, step = constraints.min_weight, constraints.max_weight, constraints.step
    max_cash_pct = constraints.max_cash
    if step != 1 and (100 % step != 0):
        raise ValueError("constraints.step must divide 100.")
    step_pct = step
    step_w = step_pct / 100.0
    min_wu = minw / 100.0
    max_wu = maxw / 100.0
    max_cash = max_cash_pct / 100.0

    candidates = []
    for t in universe:
        if t == "CASH":
            continue
        if exclude_set and t in exclude_set:
            continue
        candidates.append((t, float(benchmark.get(t, 0.0))))
    candidates.sort(key=lambda x: x[1], reverse=True)

    if len(candidates) < constraints.min_stocks:
        raise ValueError("Not enough candidate tickers to satisfy min_stocks after exclusions.")

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

        w_int = np.rint((100.0 * targets) / step_pct).astype(int) * step_pct
        w_int = np.clip(w_int, minw, maxw)

        def ssum() -> int:
            return int(w_int.sum())

        while ssum() > 100:
            over = (w_int / 100.0) - targets
            can = np.where(w_int > minw)[0]
            if len(can) == 0:
                return None
            idx = int(can[np.argmax(over[can])])
            w_int[idx] -= step_pct

        if not constraints.allow_cash:
            while ssum() < 100:
                under = targets - (w_int / 100.0)
                can = np.where(w_int < maxw)[0]
                if len(can) == 0:
                    return None
                idx = int(can[np.argmax(under[can])])
                w_int[idx] += step_pct

        # Enforce CASH cap: require stock sum >= 100 - max_cash_pct
        if constraints.allow_cash:
            min_stock_sum = 100 - max_cash_pct
            # If stock sum too low, push weight into stocks with headroom.
            while ssum() < min_stock_sum:
                headroom = maxw - w_int
                can = np.where(headroom > 0)[0]
                if len(can) == 0:
                    return None
                # add to most under-target among those with headroom
                under = targets - (w_int / 100.0)
                idx = int(can[np.argmax(under[can])])
                w_int[idx] += step_pct

        cash_pct = 0
        if constraints.allow_cash:
            cash_pct = 100 - ssum()
            if cash_pct < 0 or cash_pct > max_cash_pct:
                return None

        w = np.zeros(len(universe), dtype=float)
        for t, pct in zip(chosen, w_int):
            w[u_index[t]] = pct / 100.0
        if constraints.allow_cash:
            w[u_index["CASH"]] = cash_pct / 100.0

        s = float(w.sum())
        if abs(s - 1.0) > 1e-12 and s > 0:
            w /= s

        for t in chosen:
            wt = w[u_index[t]]
            if wt + 1e-12 < min_wu or wt - 1e-12 > max_wu:
                return None
        if constraints.allow_cash and w[u_index["CASH"]] > max_cash + 1e-12:
            return None

        return w

    def local_improve(w: np.ndarray, max_iters: int = 25000) -> np.ndarray:
        cash_i = u_index.get("CASH", None) if constraints.allow_cash else None
        d = w - w_b
        q_cur = float(d @ Sigma_ann @ d)

        noncash_idx = [u_index[t] for t, _ in candidates]
        noncash_set = set(noncash_idx)

        def count_noncash_holdings(wv: np.ndarray) -> int:
            return int(np.sum(wv[list(noncash_set)] > 1e-12))

        for _ in range(max_iters):
            best_dq = 0.0
            best_move = None

            for i in range(len(universe)):
                wi = w[i]
                if wi <= 1e-15:
                    continue

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
                        # Enforce cash cap on recipient
                        if wj + step_w > max_cash + 1e-15:
                            continue
                    else:
                        if wj <= 1e-15:
                            if step_w + 1e-15 < min_wu:
                                continue
                        if wj + step_w > max_wu + 1e-15:
                            continue

                    if (cash_i is None) or (i != cash_i):
                        if (i in noncash_set) and (w[i] - step_w <= 1e-12):
                            if count_noncash_holdings(w) - 1 < constraints.min_stocks:
                                continue

                    delta = np.zeros(len(universe), dtype=float)
                    delta[i] -= step_w
                    delta[j] += step_w

                    # If delta adds to cash, enforce cap
                    if cash_i is not None and delta[cash_i] > 0:
                        if w[cash_i] + delta[cash_i] > max_cash + 1e-15:
                            continue

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

        w_pct = np.rint(w * 100.0 / step_pct) * step_pct
        w = (w_pct / 100.0).astype(float)

        if constraints.allow_cash:
            # Reconcile by forcing cash to residual, but cap it; if residual exceeds cap, push into stocks.
            w[u_index["CASH"]] = 1.0 - float(np.sum(w[list(noncash_set)]))
            if w[u_index["CASH"]] > max_cash + 1e-12:
                overflow = w[u_index["CASH"]] - max_cash
                w[u_index["CASH"]] = max_cash
                # Distribute overflow into stocks with headroom.
                headroom = []
                for t, _ in candidates:
                    idx = u_index[t]
                    if w[idx] > 1e-12:
                        headroom.append((idx, max_wu - w[idx]))
                headroom = [(idx, hr) for idx, hr in headroom if hr > 1e-12]
                if not headroom:
                    w = np.clip(w, 0.0, None)
                    s = float(w.sum())
                    if s > 0:
                        w /= s
                    return w
                # Greedy distribute in 1% steps.
                steps = int(round(overflow / step_w))
                for _ in range(steps):
                    headroom.sort(key=lambda x: x[1], reverse=True)
                    idx, hr = headroom[0]
                    if hr < step_w - 1e-12:
                        break
                    w[idx] += step_w
                    w[u_index["CASH"]] -= step_w
                    headroom[0] = (idx, hr - step_w)
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
        w0 = make_feasible(propose_set(k, "top"))
        if w0 is None:
            continue
        w1 = local_improve(w0)
        te1 = te_of(w1)
        if te1 < best_te:
            best_te = te1
            best_w = w1

    for _ in range(60):
        k = int(rng.integers(constraints.min_stocks, 13))
        w0 = make_feasible(propose_set(k, "rand"))
        if w0 is None:
            continue
        w1 = local_improve(w0)
        te1 = te_of(w1)
        if te1 < best_te:
            best_te = te1
            best_w = w1

    if best_w is None:
        raise RuntimeError("TE optimiser failed to find a feasible portfolio.")

    port: Dict[str, float] = {}
    for t in universe:
        wt = float(best_w[u_index[t]])
        if wt > 1e-8:
            port[t] = wt
    s = sum(port.values())
    for k in list(port.keys()):
        port[k] /= s
    if not is_legal_portfolio(port, constraints):
        raise RuntimeError("TE optimiser produced an illegal portfolio (after CASH cap enforcement).")
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
# Win probs caching (benchmark weights)
# -----------------------------

def load_or_compute_win_probs_excl_ahebb(
    winprob_path: str,
    players_ex: List[str],
    values_ex: Dict[str, float],
    W_mat_current_ex: np.ndarray,
    Sigma_ann_ex: np.ndarray,
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
            raise ValueError(f"{winprob_path} sums to <= 0 over non-anchor players.")
        return {p: w / s for p, w in probs.items()}

    finish_ex = simulate_finish_probs(
        players_ex, values_ex, W_mat_current_ex, Sigma_ann_ex,
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
# WIN objective (inner MC + outer search)
# -----------------------------

@dataclass
class MonteCarloObjectiveContext:
    r_T: np.ndarray
    max_logV_other: np.ndarray
    logV0_a: float


def make_mc_objective_context(
    players: List[str],
    values: Dict[str, float],
    W_full: Dict[str, np.ndarray],
    Sigma_ann: np.ndarray,
    days_remaining: int,
    inner_sims: int,
    seed: int,
    anchor: str = "ahebb",
) -> MonteCarloObjectiveContext:
    if anchor not in players:
        raise ValueError(f"Anchor '{anchor}' not found in Leaderboard.csv")

    Sigma_day = Sigma_ann / 252.0
    mu_day = -0.5 * np.diag(Sigma_day)
    T = int(days_remaining)

    mean_T = T * mu_day
    cov_T = T * Sigma_day

    C = (cov_T + cov_T.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.clip(eigvals, 1e-14, None)
    C_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    L = np.linalg.cholesky(C_psd + 1e-16 * np.eye(C_psd.shape[0]))

    rng = np.random.default_rng(seed + 12345)
    Z = rng.standard_normal(size=(inner_sims, C_psd.shape[0]))
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
    wins = (logV_anchor > ctx.max_logV_other)
    p_hat = float(np.mean(wins))
    obj = float(math.log(p_hat + 1e-12))
    return obj, p_hat


def sample_legal_portfolio(
    rng: np.random.Generator,
    tickers: List[str],                 # excludes CASH
    ticker_probs: np.ndarray,
    constraints: ContestConstraints,
    bench: Dict[str, float],
) -> Dict[str, float]:
    minw, maxw, step = constraints.min_weight, constraints.max_weight, constraints.step
    if step != 1:
        raise ValueError("This sampler assumes 1% increments.")

    k = int(rng.integers(constraints.min_stocks, 13))
    k = max(constraints.min_stocks, min(k, len(tickers)))

    p = np.maximum(ticker_probs, 1e-12)
    p = p / p.sum()
    idx = rng.choice(len(tickers), size=k, replace=False, p=p)
    chosen = [tickers[int(i)] for i in idx]

    min_stock_sum = minw * k
    if constraints.allow_cash:
        min_stock_sum = max(min_stock_sum, 100 - constraints.max_cash)
        stock_sum = int(round(rng.triangular(min_stock_sum, 92.0, 100.0)))
        stock_sum = max(min_stock_sum, min(stock_sum, 100))
    else:
        stock_sum = 100

    alloc = np.full(k, minw, dtype=int)
    remaining = stock_sum - int(np.sum(alloc))
    caps = np.full(k, maxw - minw, dtype=int)

    b = np.array([max(bench.get(t, 0.0), 1e-6) for t in chosen], dtype=float)
    alpha = 0.25 + 3.0 * (b / b.sum())
    x = rng.dirichlet(alpha)

    extra = np.floor(max(remaining, 0) * x).astype(int)
    extra = np.minimum(extra, caps)
    alloc = alloc + extra

    rem2 = stock_sum - int(np.sum(alloc))
    tries = 0
    while rem2 > 0 and tries < 10_000:
        i = int(rng.integers(0, k))
        if alloc[i] < maxw:
            alloc[i] += 1
            rem2 -= 1
        tries += 1

    port: Dict[str, float] = {t: w / 100.0 for t, w in zip(chosen, alloc)}
    cash = 1.0 - sum(port.values())
    if constraints.allow_cash and cash > 1e-12:
        port["CASH"] = cash

    s = sum(port.values())
    if abs(s - 1.0) > 1e-8 and s > 0:
        for t in list(port.keys()):
            port[t] /= s

    if not is_legal_portfolio(port, constraints):
        # Should be rare; resample by returning a trivial safe port.
        # (Outer loop will ignore poor objective.)
        return {"CASH": min(constraints.max_cash / 100.0, 1.0)}
    return port


def optimise_for_win_probability(
    rng: np.random.Generator,
    tickers: List[str],                # excludes CASH
    bench: Dict[str, float],
    constraints: ContestConstraints,
    u_index: Dict[str, int],
    ctx_mc: MonteCarloObjectiveContext,
    n_rounds: int,
    samples_per_round: int,
    elite_frac: float = 0.12,
    polish_steps: int = 600,
) -> Dict[str, float]:
    bench_w = np.array([max(bench.get(t, 0.0), 0.0) for t in tickers], dtype=float)
    if bench_w.sum() > 0:
        p = bench_w / bench_w.sum()
    else:
        p = np.ones(len(tickers), dtype=float) / len(tickers)

    best_port: Dict[str, float] = {"CASH": min(constraints.max_cash / 100.0, 1.0)}
    best_obj = -float("inf")
    best_p = 0.0

    elite_n = max(10, int(round(samples_per_round * elite_frac)))
    max_cash = constraints.max_cash / 100.0

    for r in range(n_rounds):
        objs = np.empty(samples_per_round, dtype=float)
        ps = np.empty(samples_per_round, dtype=float)
        ports: List[Dict[str, float]] = []

        for i in range(samples_per_round):
            port = sample_legal_portfolio(rng, tickers, p, constraints, bench)

            w = np.zeros(len(u_index), dtype=float)
            for t, wt in port.items():
                if t in u_index:
                    w[u_index[t]] += float(wt)
            s = w.sum()
            if abs(s - 1.0) > 1e-8 and s > 0:
                w /= s

            obj, p_hat = mc_win_prob_first(w, ctx_mc)
            objs[i] = obj
            ps[i] = p_hat
            ports.append(port)

            if obj > best_obj:
                best_obj = obj
                best_port = port
                best_p = p_hat

        elite_idx = np.argsort(-objs)[:elite_n]

        counts = np.zeros(len(tickers), dtype=float)
        wsum = np.zeros(len(tickers), dtype=float)
        for idx in elite_idx:
            port = ports[int(idx)]
            for j, t in enumerate(tickers):
                if t in port and port[t] > 1e-12:
                    counts[j] += 1.0
                    wsum[j] += port[t]

        if counts.sum() > 0:
            freq = counts / counts.sum()
        else:
            freq = np.ones(len(tickers), dtype=float) / len(tickers)

        if wsum.sum() > 0:
            wfreq = wsum / wsum.sum()
        else:
            wfreq = freq

        new_p = 0.65 * (0.55 * freq + 0.45 * wfreq) + 0.35 * p
        new_p = np.maximum(new_p, 1e-9)
        p = new_p / new_p.sum()

        print(f"Round {r+1}/{n_rounds}: best inner-MC P1st={best_p:.6f}  (logP={best_obj:.6f})")

    # Polish: accept 1% transfers that improve inner-MC objective, enforcing CASH cap.
    w = np.zeros(len(u_index), dtype=float)
    for t, wt in best_port.items():
        if t in u_index:
            w[u_index[t]] += float(wt)
    if abs(w.sum() - 1.0) > 1e-8 and w.sum() > 0:
        w /= w.sum()

    minw = constraints.min_weight / 100.0
    maxw_stock = constraints.max_weight / 100.0
    step = constraints.step / 100.0

    cash_i = u_index.get("CASH", None)
    noncash_idx = [u_index[t] for t in tickers if t in u_index]

    def noncash_count(wv: np.ndarray) -> int:
        return int(np.sum(wv[noncash_idx] > 1e-12))

    cur_obj, cur_p = mc_win_prob_first(w, ctx_mc)

    for _ in range(polish_steps):
        donor = int(rng.integers(0, len(u_index)))
        recv = int(rng.integers(0, len(u_index)))
        if donor == recv:
            continue
        if w[donor] <= 1e-15:
            continue

        if cash_i is not None and donor != cash_i:
            if w[donor] - step < minw - 1e-12:
                continue
            if donor in noncash_idx and (w[donor] - step <= 1e-12) and noncash_count(w) - 1 < constraints.min_stocks:
                continue
        else:
            if w[donor] - step < -1e-12:
                continue

        if cash_i is not None and recv != cash_i:
            if w[recv] <= 1e-15 and step + 1e-12 < minw:
                continue
            if w[recv] + step > maxw_stock + 1e-12:
                continue

        w2 = w.copy()
        w2[donor] -= step
        w2[recv] += step

        if cash_i is not None:
            w2 = np.clip(w2, 0.0, None)
            noncash_sum = float(np.sum(w2[noncash_idx]))
            w2[cash_i] = max(0.0, 1.0 - noncash_sum)
            if w2[cash_i] > max_cash + 1e-12:
                continue
            s = w2.sum()
            if abs(s - 1.0) > 1e-8 and s > 0:
                w2 /= s
        else:
            w2 = np.clip(w2, 0.0, None)
            s = w2.sum()
            if s > 0:
                w2 /= s

        if noncash_count(w2) < constraints.min_stocks:
            continue

        ok = True
        for i in noncash_idx:
            if w2[i] > 1e-12:
                if w2[i] < minw - 1e-12 or w2[i] > maxw_stock + 1e-12:
                    ok = False
                    break
                if abs(round(w2[i] / step) * step - w2[i]) > 5e-6:
                    ok = False
                    break
        if not ok:
            continue

        if cash_i is not None and w2[cash_i] > max_cash + 1e-12:
            continue

        obj2, p2 = mc_win_prob_first(w2, ctx_mc)
        if obj2 > cur_obj + 1e-12:
            w = w2
            cur_obj, cur_p = obj2, p2

    out: Dict[str, float] = {}
    for t, i in u_index.items():
        wt = float(w[i])
        if wt > 1e-8:
            out[t] = wt
    s = sum(out.values())
    for k in list(out.keys()):
        out[k] /= s

    if not is_legal_portfolio(out, constraints):
        return best_port

    print(f"After polish: best inner-MC P1st={cur_p:.6f}  (logP={cur_obj:.6f})")
    return out


# -----------------------------
# Main
# -----------------------------

def run_all(
    objective: str,
    leaderboard_csv: str,
    corr_monthly_csv: str,
    corr_daily_csv: str,
    vol_short_csv: str,
    vol_medium_csv: str,
    days_remaining: int,
    n_sims: int,
    seed: int,
    winprob_path: str,
    wp_vol_source: str,
    te_vol_source: str,
    opt_vol_source: str,
    sim_vol_source: str,
    inner_sims: int,
    rounds: int,
    samples: int,
    exclude_tickers: str,
    max_cash: int,
    anchor_name: str,
) -> None:
    lb = load_leaderboard(leaderboard_csv)
    players, values, ports = build_player_portfolios(lb, drop_duplicates=True)

    if anchor_name not in players:
        raise ValueError(f"Anchor {anchor_name!r} not found in Leaderboard.csv")

    corr_m = load_corr_matrix(corr_monthly_csv)
    corr_d = load_corr_matrix(corr_daily_csv)
    corr_combined = combine_monthly_daily_corr(corr_m, corr_d)

    ports = {p: filter_portfolio_to_corr(ports[p], corr_combined) for p in players}

    vol_short = load_vols(vol_short_csv)
    vol_med = load_vols(vol_medium_csv)

    vols_wp = vol_short if wp_vol_source == "short" else vol_med
    vols_te = vol_short if te_vol_source == "short" else vol_med
    vols_opt = vol_short if opt_vol_source == "short" else vol_med
    vols_sim = vol_short if sim_vol_source == "short" else vol_med

    raw_excludes = [norm_ticker(x) for x in exclude_tickers.split(",") if x.strip()]
    exclude_set = set(raw_excludes)
    if exclude_set:
        print(f"Excluding tickers from optimisation: {sorted(exclude_set)}")

    constraints = ContestConstraints(
        min_stocks=5,
        min_weight=5,
        max_weight=25,
        step=1,
        allow_cash=True,
        max_cash=max_cash,
    )

    # --- Benchmark win probs: anchor excluded entirely ---
    players_ex = [p for p in players if p != anchor_name]
    values_ex = {p: values[p] for p in players_ex}

    universe_ex = make_universe(players_ex, ports)
    corr_u_wp_ex, vol_vec_wp_ex = ensure_corr_and_vol_coverage(universe_ex, corr_combined, vols_wp)
    Sigma_ann_wp_ex = cov_from_corr_vol(corr_u_wp_ex, vol_vec_wp_ex)
    u_index_wp_ex = {t: i for i, t in enumerate(universe_ex)}

    W_current_ex = {p: vectorize_port(ports[p], u_index_wp_ex) for p in players_ex}
    W_mat_current_ex = np.stack([W_current_ex[p] for p in players_ex], axis=0)

    win_probs = load_or_compute_win_probs_excl_ahebb(
        winprob_path,
        players_ex,
        values_ex,
        W_mat_current_ex,
        Sigma_ann_wp_ex,
        days_remaining,
        seed,
    )

    bench = build_weighted_benchmark(players_ex, ports, win_probs)
    bench = filter_portfolio_to_corr(bench, corr_combined)

    # Remove excluded tickers from benchmark bias (recommended)
    if exclude_set:
        for t in list(bench.keys()):
            if t in exclude_set:
                bench.pop(t)
        sb = sum(bench.values())
        if sb > 0:
            for k in list(bench.keys()):
                bench[k] /= sb

    pd.DataFrame([{"Ticker": k, "Weight": v} for k, v in sorted(bench.items(), key=lambda x: x[1], reverse=True)]) \
      .to_csv("weighted_benchmark.csv", index=False)
    print("Wrote: weighted_benchmark.csv")

    # --- Universe for anchor optimisation ---
    universe0 = make_universe(players, ports)
    tickers_for_anchor = [t for t in universe0 if t != "CASH" and t not in exclude_set]

    if len(tickers_for_anchor) < constraints.min_stocks:
        raise ValueError("After exclusions, fewer than min_stocks tickers remain in the universe.")

    # --- Choose anchor portfolio ---
    if objective == "current":
        # ports[...] has already been processed the same way as every other player:
        # parsed, unknown tickers dropped to CASH, renormalised, and filtered to the correlation universe.
        anchor_port = ports[anchor_name]

        print(f"\nUsing current {anchor_name} portfolio from Leaderboard.csv (no optimisation):")

    elif objective == "te":
        corr_u_te, vol_vec_te = ensure_corr_and_vol_coverage(universe0, corr_combined, vols_te)
        anchor_port = greedy_te_min_portfolio(bench, corr_u_te, vol_vec_te, constraints, exclude_set=exclude_set)

        te_vs_bench_te = tracking_error_portfolios(anchor_port, bench, corr_u_te, vol_vec_te)
        print(f"\nTE(selected {anchor_name}, benchmark) {te_vol_source.upper()}-TERM vols (ann.): {te_vs_bench_te:.6f}")

        corr_u_med0, vol_vec_med0 = ensure_corr_and_vol_coverage(universe0, corr_combined, vol_med)
        te_vs_bench_med = tracking_error_portfolios(anchor_port, bench, corr_u_med0, vol_vec_med0)
        print(f"TE(selected {anchor_name}, benchmark) MEDIUM-TERM vols (ann.): {te_vs_bench_med:.6f}")

        print(f"\nTracking-error-minimizing portfolio for {anchor_name} (contest-legal, cash capped):")

    elif objective == "win":
        corr_u_opt, vol_vec_opt = ensure_corr_and_vol_coverage(universe0, corr_combined, vols_opt)
        Sigma_ann_opt = cov_from_corr_vol(corr_u_opt, vol_vec_opt)
        u_index_full = {t: i for i, t in enumerate(universe0)}

        W_full_current = {p: vectorize_port(ports[p], u_index_full) for p in players}
        ctx_mc = make_mc_objective_context(
            players=players,
            values=values,
            W_full=W_full_current,
            Sigma_ann=Sigma_ann_opt,
            days_remaining=days_remaining,
            inner_sims=inner_sims,
            seed=seed,
            anchor=anchor_name,
        )

        print(f"\nOptimising for win probability using {opt_vol_source.upper()}-TERM vols (inner MC objective, S={inner_sims}).")
        rng = np.random.default_rng(seed)

        anchor_port = optimise_for_win_probability(
            rng=rng,
            tickers=tickers_for_anchor,
            bench=bench,
            constraints=constraints,
            u_index=u_index_full,
            ctx_mc=ctx_mc,
            n_rounds=rounds,
            samples_per_round=samples,
            elite_frac=0.12,
            polish_steps=600,
        )

        print(f"\nWin-probability-optimised portfolio for {anchor_name} (contest-legal, cash capped):")

    else:
        raise ValueError("--objective must be 'te', 'win', or 'current'.")

    if not is_legal_portfolio(anchor_port, constraints):
        raise RuntimeError(f"Internal error: chosen {anchor_name} portfolio violates constraints (including CASH cap).")

    for t, wt in sorted(anchor_port.items(), key=lambda x: x[1], reverse=True):
        print(f"  {t:6s}  {wt*100:6.2f}%")

    # --- Final full contest sim including chosen anchor using sim vols ---
    ports_adj = dict(ports)
    ports_adj[anchor_name] = anchor_port

    universe_full = make_universe(players, ports_adj)
    corr_u_sim, vol_vec_sim = ensure_corr_and_vol_coverage(universe_full, corr_combined, vols_sim)
    Sigma_ann_sim = cov_from_corr_vol(corr_u_sim, vol_vec_sim)
    u_index_sim = {t: i for i, t in enumerate(universe_full)}

    W_full = {p: vectorize_port(ports_adj[p], u_index_sim) for p in players}
    W_mat_full = np.stack([W_full[p] for p in players], axis=0)

    finish_full, _, n_used = simulate_finish_probs(
        players, values, W_mat_full, Sigma_ann_sim,
        days_remaining, n_sims=n_sims, seed=seed, return_counts=True
    )

    if anchor_name in finish_full:
        pa = finish_full[anchor_name]
        print(f"\n{anchor_name} finish probabilities (Monte Carlo) with ~95% MOE:")
        for k in ["P1st", "P2nd", "P3rd", "PWorse"]:
            p_hat = float(pa[k])
            moe = moe95(p_hat, n_used)
            print(f"  {k:6s}: {p_hat*100:7.3f}%  (±{moe*100:5.3f}%)")

    # contest_results.csv
    Sigma_ann_sim = (Sigma_ann_sim + Sigma_ann_sim.T) / 2.0
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
    parser = argparse.ArgumentParser()

    parser.add_argument("--objective", choices=["te", "win", "current"], default="te",
                        help="Portfolio objective: 'te' minimises tracking error; 'win' maximises P(1st); 'current' uses the anchor portfolio from Leaderboard.csv.")
    parser.add_argument("--anchor", type=str, default="ahebb",
                        help="Leaderboard player name to optimise/replace. Default: ahebb.")

    parser.add_argument("--days", type=int, default=None, help="Trading days remaining (e.g. 13)")
    parser.add_argument("--sims", type=int, default=400_000, help="Number of Monte Carlo simulations (final reporting)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")

    parser.add_argument("--wp_vol_source", choices=["short", "medium"], default="medium",
                        help="Vols used when (re)computing WinProbabilities_excl_ahebb.csv (default: medium)")
    parser.add_argument("--te_vol_source", choices=["short", "medium"], default="short",
                        help="Vols used for TE optimisation and TE(selected anchor, benchmark) reporting (default: short)")
    parser.add_argument("--opt_vol_source", choices=["short", "medium"], default="medium",
                        help="Vols used for WIN optimisation objective (default: medium)")
    parser.add_argument("--sim_vol_source", choices=["short", "medium"], default="medium",
                        help="Vols used for the final full contest simulation after replacing anchor (default: medium)")

    parser.add_argument("--inner_sims", type=int, default=12000,
                        help="Inner simulations for WIN objective (only used when --objective win)")
    parser.add_argument("--rounds", type=int, default=8,
                        help="Optimisation rounds (only used when --objective win)")
    parser.add_argument("--samples", type=int, default=450,
                        help="Samples per round (only used when --objective win)")

    parser.add_argument("--exclude", type=str, default="",
                        help="Comma-separated tickers to exclude from anchor optimisation universe (applies to both objectives).")

    parser.add_argument("--max_cash", type=int, default=25,
                        help="Maximum CASH percent allowed in anchor portfolio (default: 25).")

    args = parser.parse_args()

    days = get_days_remaining(args.days)

    print(
        f"Objective: {args.objective} | anchor={args.anchor} | "
        f"Vol sources: win-prob={args.wp_vol_source}, TE={args.te_vol_source}, opt={args.opt_vol_source}, sim={args.sim_vol_source} | "
        f"max_cash={args.max_cash}%"
    )

    run_all(
        objective=args.objective,
        leaderboard_csv="Leaderboard.csv",
        corr_monthly_csv="CorrelationMatrixMonthly.csv",
        corr_daily_csv="CorrelationMatrixDaily.csv",
        vol_short_csv="Volatilities.csv",
        vol_medium_csv="Volatilities2.csv",
        days_remaining=days,
        n_sims=args.sims,
        seed=args.seed,
        winprob_path="WinProbabilities_excl_ahebb.csv",
        wp_vol_source=args.wp_vol_source,
        te_vol_source=args.te_vol_source,
        opt_vol_source=args.opt_vol_source,
        sim_vol_source=args.sim_vol_source,
        inner_sims=args.inner_sims,
        rounds=args.rounds,
        samples=args.samples,
        exclude_tickers=args.exclude,
        max_cash=args.max_cash,
        anchor_name=args.anchor,
    )

