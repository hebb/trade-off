import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
import argparse

import numpy as np
import pandas as pd


# ============================================================
# User settings
# ============================================================
VOL_FILE = "Volatilities.csv"
CORR_FILE = "correlation_matrix.csv"

RANDOM_SEED = 42

# Example: "AAPL=25,NVDA=15,SHOP=10"
DEFAULT_FORCED_HOLDINGS = ""

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

MIN_WEIGHT = 5
MAX_WEIGHT = 25
TOTAL_WEIGHT = 100
MIN_NAMES = 5

EPS = 1e-15

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# Helpers
# ============================================================
def normalize_ticker(t: str) -> str:
    t = str(t).strip().upper()
    if t.endswith(".TO"):
        t = t[:-3]
    if t.endswith(":CA"):
        t = t[:-3]
    return t


def parse_percent_like(x) -> float:
    if pd.isna(x):
        return np.nan

    s = str(x).strip()
    if s == "":
        return np.nan

    had_percent = "%" in s
    s = s.replace("%", "").replace(",", "")
    val = pd.to_numeric(s, errors="coerce")

    if pd.isna(val):
        return np.nan

    val = float(val)
    if val < 0:
        return np.nan

    if had_percent:
        return val / 100.0

    if val > 1.5:
        return val / 100.0

    return val


def safe_float_matrix(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = (
            out[c]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def clean_portfolio(port: Dict[str, int]) -> Dict[str, int]:
    return {k: int(v) for k, v in port.items() if int(v) > 0}


def parse_forced_holdings(s: str) -> Dict[str, int]:
    forced = {}
    s = str(s).strip()

    if not s:
        return forced

    for part in s.split(","):
        part = part.strip()
        if not part:
            continue

        if "=" not in part:
            raise ValueError(
                f"Bad forced holding '{part}'. Use format like AAPL=25,NVDA=10."
            )

        ticker, weight = part.split("=", 1)
        ticker = normalize_ticker(ticker)
        weight = int(weight)

        if weight < MIN_WEIGHT or weight > MAX_WEIGHT:
            raise ValueError(
                f"Forced holding {ticker} has weight {weight}%, "
                f"but each stock must be between {MIN_WEIGHT}% and {MAX_WEIGHT}%."
            )

        if ticker in forced:
            raise ValueError(f"Duplicate forced ticker: {ticker}")

        forced[ticker] = weight

    if sum(forced.values()) > TOTAL_WEIGHT:
        raise ValueError("Forced holdings exceed 100% total weight.")

    if sum(forced.values()) == TOTAL_WEIGHT and len(forced) < MIN_NAMES:
        raise ValueError("Forced holdings sum to 100% but contain fewer than 5 names.")

    return forced


def legal_portfolio(port: Dict[str, int], forced: Optional[Dict[str, int]] = None) -> bool:
    forced = forced or {}

    if sum(port.values()) != TOTAL_WEIGHT:
        return False

    if len(port) < MIN_NAMES:
        return False

    for t, w in forced.items():
        if port.get(t) != w:
            return False

    for w in port.values():
        if not isinstance(w, int):
            return False
        if w < MIN_WEIGHT or w > MAX_WEIGHT:
            return False

    return True


def portfolio_variance(port: Dict[str, int], cov: np.ndarray, ticker_to_idx: Dict[str, int]) -> float:
    names = list(port.keys())
    idx = np.array([ticker_to_idx[t] for t in names], dtype=int)
    w = np.array([port[t] / 100.0 for t in names], dtype=float)
    subcov = cov[np.ix_(idx, idx)]
    return float(w @ subcov @ w)


def portfolio_volatility(port: Dict[str, int], cov: np.ndarray, ticker_to_idx: Dict[str, int]) -> float:
    return math.sqrt(max(portfolio_variance(port, cov, ticker_to_idx), 0.0))


def repair_to_psd(matrix: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    vals, vecs = np.linalg.eigh(matrix)

    if vals.min() >= -1e-10:
        return matrix

    vals = np.clip(vals, floor, None)
    repaired = (vecs * vals) @ vecs.T
    return (repaired + repaired.T) / 2.0


def top_k_outsiders_from_ranked(ranked_names: List[str], held: Set[str], k: int) -> List[str]:
    out = []
    for t in ranked_names:
        if t not in held:
            out.append(t)
            if len(out) >= k:
                break
    return out


# ============================================================
# Random portfolio generation
# ============================================================
def random_portfolio_from_universe(
    universe: List[str],
    k: Optional[int] = None,
    forced: Optional[Dict[str, int]] = None
) -> Dict[str, int]:
    forced = forced or {}

    missing = [t for t in forced if t not in universe]
    if missing:
        raise ValueError(f"Forced tickers not in this search universe: {missing}")

    forced_weight = sum(forced.values())
    remaining_weight = TOTAL_WEIGHT - forced_weight
    forced_count = len(forced)

    if remaining_weight == 0:
        if forced_count < MIN_NAMES:
            raise ValueError("Forced holdings sum to 100% but have fewer than 5 names.")
        return forced.copy()

    min_extra_names = max(0, MIN_NAMES - forced_count)
    max_extra_names = min(len(universe) - forced_count, remaining_weight // MIN_WEIGHT)

    feasible_extra_counts = []
    for extra_k in range(min_extra_names, max_extra_names + 1):
        if extra_k == 0:
            feasible_extra_counts.append(extra_k)
        elif extra_k * MIN_WEIGHT <= remaining_weight <= extra_k * MAX_WEIGHT:
            feasible_extra_counts.append(extra_k)

    if not feasible_extra_counts:
        raise ValueError("Forced holdings leave no feasible way to complete the portfolio.")

    if k is None:
        extra_k = random.choice(feasible_extra_counts)
    else:
        extra_k = k - forced_count
        if extra_k not in feasible_extra_counts:
            raise ValueError(f"Requested k={k} is infeasible with forced holdings.")

    available = [t for t in universe if t not in forced]
    names = random.sample(available, extra_k) if extra_k > 0 else []

    if extra_k == 0:
        return forced.copy()

    weights = [MIN_WEIGHT] * extra_k
    remaining = remaining_weight - extra_k * MIN_WEIGHT

    while remaining > 0:
        i = random.randrange(extra_k)
        if weights[i] < MAX_WEIGHT:
            weights[i] += 1
            remaining -= 1

    port = forced.copy()
    port.update(dict(zip(names, weights)))
    return port


# ============================================================
# Data loading
# ============================================================
def load_inputs(
    vol_file: str,
    corr_file: str
) -> Tuple[List[str], np.ndarray, Dict[str, int], Dict[str, float], pd.DataFrame]:
    vol_df = pd.read_csv(vol_file)
    corr_df_raw = pd.read_csv(corr_file)

    required_vol_cols = {"Ticker", "Implied Volatility"}
    missing_vol_cols = required_vol_cols - set(vol_df.columns)

    if missing_vol_cols:
        raise ValueError(f"Volatility file is missing required columns: {sorted(missing_vol_cols)}")

    if corr_df_raw.shape[1] < 2:
        raise ValueError("Correlation file must have one row-label column and at least one ticker column.")

    vol_df["TickerNorm"] = vol_df["Ticker"].apply(normalize_ticker)
    vol_df["Vol"] = vol_df["Implied Volatility"].apply(parse_percent_like)

    vol_df = vol_df.dropna(subset=["TickerNorm", "Vol"])
    vol_df = vol_df[vol_df["Vol"] > 0].copy()
    vol_df = vol_df.drop_duplicates(subset=["TickerNorm"], keep="last")

    vol_map = vol_df.set_index("TickerNorm")["Vol"].to_dict()

    if len(vol_map) < MIN_NAMES:
        raise ValueError(f"Only {len(vol_map)} usable volatilities found; need at least {MIN_NAMES}.")

    row_labels = corr_df_raw.iloc[:, 0].apply(normalize_ticker)
    col_labels = [normalize_ticker(c) for c in corr_df_raw.columns[1:]]
    corr_numeric = safe_float_matrix(corr_df_raw.iloc[:, 1:])

    corr_raw = pd.DataFrame(corr_numeric.values, index=row_labels, columns=col_labels)

    corr_raw = corr_raw.groupby(level=0).mean()
    corr_raw = corr_raw.T.groupby(level=0).mean().T

    common = sorted(set(corr_raw.index) & set(corr_raw.columns) & set(vol_map.keys()))

    if len(common) < MIN_NAMES:
        raise ValueError(f"Only {len(common)} common tickers found; need at least {MIN_NAMES}.")

    corr = corr_raw.loc[common, common].astype(float)

    finite_vals = corr.values[np.isfinite(corr.values)]
    if finite_vals.size == 0:
        raise ValueError("Correlation matrix contains no usable numeric values.")

    max_abs = float(np.nanmax(np.abs(finite_vals)))
    if max_abs > 1.000001:
        if max_abs <= 100.0:
            corr = corr / 100.0
        else:
            raise ValueError(
                "Correlation matrix appears to contain values outside [-1, 1] "
                "and not in percentage form."
            )

    corr = (corr + corr.T) / 2.0
    corr = corr.clip(-1.0, 1.0)

    if corr.isna().values.any():
        corr = corr.fillna(0.0)

    for t in common:
        corr.loc[t, t] = 1.0

    corr_values = corr.values.astype(float)
    corr_values = repair_to_psd(corr_values)
    corr_values = (corr_values + corr_values.T) / 2.0

    diag = np.sqrt(np.clip(np.diag(corr_values), 1e-12, None))
    corr_values = corr_values / np.outer(diag, diag)
    corr_values = np.clip(corr_values, -1.0, 1.0)
    np.fill_diagonal(corr_values, 1.0)

    corr = pd.DataFrame(corr_values, index=common, columns=common)

    vols = np.array([vol_map[t] for t in common], dtype=float)
    cov = np.diag(vols) @ corr.values @ np.diag(vols)
    cov = (cov + cov.T) / 2.0
    cov = repair_to_psd(cov)

    ticker_to_idx = {t: i for i, t in enumerate(common)}

    return common, cov, ticker_to_idx, vol_map, corr


# ============================================================
# Candidate universe selection
# ============================================================
def build_candidate_sets(
    tickers: List[str],
    cov: np.ndarray,
    vol_map: Dict[str, float],
    corr_df: pd.DataFrame,
    shortlist_size: int,
    elite_pool_size: int
) -> Tuple[List[str], List[str], List[str], Dict[str, float]]:
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
        mu = vals.mean()
        sd = vals.std()

        if sd == 0:
            return {k: 0.0 for k in d}

        return {k: (v - mu) / sd for k, v in d.items()}

    z_vol = zscore_map(vols)
    z_cov = zscore_map(cov_contrib)
    z_corr = zscore_map(avg_pos_corr)

    score = {}
    for t in tickers:
        score[t] = 1.55 * z_vol[t] + 0.65 * z_cov[t] - 0.40 * z_corr[t]

    ranked_all = sorted(tickers, key=lambda t: score[t], reverse=True)
    shortlist = ranked_all[: min(shortlist_size, len(ranked_all))]
    elite_pool = ranked_all[: min(elite_pool_size, len(ranked_all))]

    return shortlist, elite_pool, ranked_all, score


# ============================================================
# Seed portfolios
# ============================================================
def make_seed_portfolios(
    shortlist: List[str],
    ranked_all: List[str],
    score: Dict[str, float],
    vol_map: Dict[str, float],
    corr_df: pd.DataFrame
) -> List[Dict[str, int]]:
    ranked = sorted(shortlist, key=lambda t: score[t], reverse=True)
    vol_ranked = sorted(shortlist, key=lambda t: vol_map[t], reverse=True)

    seeds = []

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
        seeds.append({
            ranked[0]: 25,
            ranked[1]: 20,
            ranked[2]: 20,
            ranked[3]: 15,
            ranked[4]: 10,
            ranked[5]: 10,
        })

    if len(ranked) >= 8:
        seeds.append({
            ranked[0]: 20,
            ranked[1]: 15,
            ranked[2]: 15,
            ranked[3]: 15,
            ranked[4]: 10,
            ranked[5]: 10,
            ranked[6]: 10,
            ranked[7]: 5,
        })

    if len(vol_ranked) >= 1:
        anchor = vol_ranked[0]
        pool = ranked_all[: min(40, len(ranked_all))]
        pool = [t for t in pool if t != anchor]

        if len(pool) >= 4:
            corr_sorted = sorted(pool, key=lambda t: corr_df.loc[anchor, t], reverse=True)
            seeds.append({
                anchor: 25,
                corr_sorted[0]: 20,
                corr_sorted[1]: 20,
                corr_sorted[2]: 20,
                corr_sorted[3]: 15,
            })

    if len(ranked) >= 10:
        seeds.append({
            ranked[0]: 15,
            ranked[1]: 15,
            ranked[2]: 15,
            ranked[3]: 10,
            ranked[4]: 10,
            ranked[5]: 10,
            ranked[6]: 10,
            ranked[7]: 5,
            ranked[8]: 5,
            ranked[9]: 5,
        })

    out = []
    seen = set()

    for s in seeds:
        s = clean_portfolio(s)

        if legal_portfolio(s):
            key = tuple(sorted(s.items()))
            if key not in seen:
                out.append(s)
                seen.add(key)

    return out


def impose_forced_holdings(
    seed: Dict[str, int],
    forced: Dict[str, int],
    universe: List[str]
) -> Dict[str, int]:
    if not forced:
        return seed

    port = {t: w for t, w in seed.items() if t not in forced}
    port.update(forced)

    total = sum(port.values())

    while total > TOTAL_WEIGHT:
        adjustable = [t for t in port if t not in forced and port[t] > MIN_WEIGHT]

        if not adjustable:
            raise ValueError("Cannot impose forced holdings on seed without violating constraints.")

        t = random.choice(adjustable)
        port[t] -= 1
        total -= 1

    available = [t for t in universe if t not in port]

    while total < TOTAL_WEIGHT:
        receivers = [t for t in port if t not in forced and port[t] < MAX_WEIGHT]

        if receivers:
            t = random.choice(receivers)
            port[t] += 1
            total += 1
        else:
            if not available:
                raise ValueError("Cannot complete seed after imposing forced holdings.")

            need = TOTAL_WEIGHT - total
            if need < MIN_WEIGHT:
                adjustable = [t for t in port if t not in forced and port[t] > MIN_WEIGHT]
                if not adjustable:
                    raise ValueError("Cannot legally complete seed after imposing forced holdings.")

                t = random.choice(adjustable)
                port[t] -= MIN_WEIGHT - need
                total -= MIN_WEIGHT - need
                need = MIN_WEIGHT

            t = available.pop()
            add = min(MAX_WEIGHT, need)
            port[t] = add
            total += add

    port = clean_portfolio(port)

    if not legal_portfolio(port, forced):
        raise ValueError("Failed to impose forced holdings while preserving contest rules.")

    return port


# ============================================================
# Local deterministic improvement
# ============================================================
def local_improvement(
    port: Dict[str, int],
    cov: np.ndarray,
    ticker_to_idx: Dict[str, int],
    ranked_names: List[str],
    forced: Optional[Dict[str, int]] = None,
    passes: int = LOCAL_PASSES,
    outsider_checks: int = OUTSIDER_CHECKS
) -> Tuple[Dict[str, int], float]:
    forced = forced or {}
    forced_names = set(forced)

    best = clean_portfolio(port.copy())
    best_val = portfolio_variance(best, cov, ticker_to_idx)

    for _ in range(passes):
        improved = False

        held = list(best.keys())

        for a in held:
            if a in forced_names or a not in best or best[a] <= MIN_WEIGHT:
                continue

            for b in held:
                if b in forced_names or b not in best or a == b or best[b] >= MAX_WEIGHT:
                    continue

                cand = best.copy()
                cand[a] -= 1
                cand[b] += 1
                cand = clean_portfolio(cand)

                if not legal_portfolio(cand, forced):
                    continue

                val = portfolio_variance(cand, cov, ticker_to_idx)

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

        current_names = [t for t in best.keys() if t not in forced_names]
        current_names = sorted(current_names, key=lambda t: best[t])

        for old_name in current_names:
            if old_name not in best:
                continue

            wt = best[old_name]

            for new_name in outsiders:
                if new_name in best:
                    continue

                cand = best.copy()
                del cand[old_name]
                cand[new_name] = wt
                cand = clean_portfolio(cand)

                if not legal_portfolio(cand, forced):
                    continue

                val = portfolio_variance(cand, cov, ticker_to_idx)

                if val > best_val + EPS:
                    best, best_val = cand, val
                    improved = True
                    break

            if improved:
                break

        if not improved:
            break

    return best, best_val


# ============================================================
# Mutation operators
# ============================================================
@dataclass
class MutationResult:
    portfolio: Dict[str, int]
    changed_names: Tuple[str, ...]


def mutate_transfer(port: Dict[str, int], forced: Dict[str, int]) -> MutationResult:
    p = port.copy()
    forced_names = set(forced)

    donors = [t for t, w in p.items() if t not in forced_names and w > MIN_WEIGHT]
    receivers = [t for t, w in p.items() if t not in forced_names and w < MAX_WEIGHT]

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
    move = min(move, p[a] - MIN_WEIGHT, MAX_WEIGHT - p[b])

    if move <= 0:
        return MutationResult(p, tuple())

    p[a] -= move
    p[b] += move

    return MutationResult(clean_portfolio(p), (a, b))


def mutate_replace_one(
    port: Dict[str, int],
    universe: List[str],
    taboo_names: Set[str],
    forced: Dict[str, int]
) -> MutationResult:
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

    return MutationResult(clean_portfolio(p), (old_name, new_name))


def mutate_replace_two(
    port: Dict[str, int],
    universe: List[str],
    taboo_names: Set[str],
    forced: Dict[str, int]
) -> MutationResult:
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
        return MutationResult(clean_portfolio(p), tuple())

    new_a, new_b = random.sample(candidates, 2)
    p[new_a] = wt_a
    p[new_b] = wt_b

    return MutationResult(clean_portfolio(p), (old_a, old_b, new_a, new_b))


def mutate_split_merge(
    port: Dict[str, int],
    universe: List[str],
    taboo_names: Set[str],
    forced: Dict[str, int]
) -> MutationResult:
    p = port.copy()
    forced_names = set(forced)
    do_split = random.random() < 0.5

    if do_split:
        sources = [t for t, w in p.items() if t not in forced_names and w >= 10]
        held = set(p.keys())

        candidates = [t for t in universe if t not in held and t not in taboo_names]
        if not candidates:
            candidates = [t for t in universe if t not in held]

        if not sources or not candidates:
            return MutationResult(p, tuple())

        old_name = random.choice(sources)
        new_name = random.choice(candidates)

        max_split = min(p[old_name] - MIN_WEIGHT, MAX_WEIGHT)
        if max_split < MIN_WEIGHT:
            return MutationResult(p, tuple())

        split_amt = random.randint(MIN_WEIGHT, max_split)

        p[old_name] -= split_amt
        p[new_name] = split_amt

        return MutationResult(clean_portfolio(p), (old_name, new_name))

    if len(p) <= MIN_NAMES:
        return MutationResult(p, tuple())

    smalls = [t for t, w in p.items() if t not in forced_names and w == MIN_WEIGHT]
    receivers = [t for t, w in p.items() if t not in forced_names and w <= MAX_WEIGHT - MIN_WEIGHT]

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

    return MutationResult(clean_portfolio(p), (a, b))


# ============================================================
# Simulated annealing + tabu memory
# ============================================================
def anneal_search(
    start_port: Dict[str, int],
    universe: List[str],
    cov: np.ndarray,
    ticker_to_idx: Dict[str, int],
    ranked_names: List[str],
    forced: Optional[Dict[str, int]] = None,
    steps: int = ANNEAL_STEPS,
    tabu_len: int = TABU_LEN
) -> Tuple[Dict[str, int], float]:
    forced = forced or {}

    current = clean_portfolio(start_port.copy())
    current, current_val = local_improvement(
        current,
        cov,
        ticker_to_idx,
        ranked_names,
        forced=forced
    )

    best = current.copy()
    best_val = current_val

    base = max(current_val, 1e-8)
    temp_start = max(TEMP_START_MULT * base, 1e-6)
    temp_end = max(TEMP_END_MULT * base, 1e-8)

    tabu = deque(maxlen=tabu_len)

    for step in range(steps):
        frac = step / max(1, steps - 1)
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

        cand = clean_portfolio(mut.portfolio)

        if not legal_portfolio(cand, forced):
            continue

        if step % 50 == 0:
            cand, cand_val = local_improvement(
                cand,
                cov,
                ticker_to_idx,
                ranked_names,
                forced=forced,
                passes=3,
                outsider_checks=min(40, OUTSIDER_CHECKS)
            )
        else:
            cand_val = portfolio_variance(cand, cov, ticker_to_idx)

        delta = cand_val - current_val

        if delta >= 0:
            accept = True
        else:
            prob = math.exp(delta / max(temp, 1e-12))
            accept = random.random() < prob

        if accept:
            current = cand
            current_val = cand_val

            for name in mut.changed_names:
                tabu.append(name)

            if cand_val > best_val + EPS:
                best = cand.copy()
                best_val = cand_val

    best, best_val = local_improvement(
        best,
        cov,
        ticker_to_idx,
        ranked_names,
        forced=forced
    )

    return best, best_val


# ============================================================
# Main solve
# ============================================================
def solve_max_vol_portfolio(
    vol_file: str,
    corr_file: str,
    forced_holdings: str = DEFAULT_FORCED_HOLDINGS
) -> Tuple[Dict[str, int], float]:
    forced = parse_forced_holdings(forced_holdings)

    tickers, cov, ticker_to_idx, vol_map, corr_df = load_inputs(vol_file, corr_file)

    missing_forced = [t for t in forced if t not in ticker_to_idx]
    if missing_forced:
        raise ValueError(f"Forced tickers not found in usable universe: {missing_forced}")

    shortlist, elite_pool, ranked_all, score = build_candidate_sets(
        tickers=tickers,
        cov=cov,
        vol_map=vol_map,
        corr_df=corr_df,
        shortlist_size=SHORTLIST_SIZE,
        elite_pool_size=ELITE_POOL_SIZE
    )

    for t in forced:
        if t not in shortlist:
            shortlist.append(t)
        if t not in elite_pool:
            elite_pool.append(t)

    print(f"Usable tickers: {len(tickers)}")
    print(f"Shortlist size: {len(shortlist)}")
    print(f"Elite pool size: {len(elite_pool)}")
    print(f"Forced holdings: {forced if forced else 'none'}")

    seed_ports = make_seed_portfolios(shortlist, ranked_all, score, vol_map, corr_df)

    seed_ports = [
        impose_forced_holdings(seed, forced, elite_pool)
        for seed in seed_ports
    ]

    seed_ports = [
        seed
        for seed in seed_ports
        if legal_portfolio(seed, forced)
    ]

    best_port: Optional[Dict[str, int]] = None
    best_val = -1.0
    seen_best_keys = set()

    for i, seed in enumerate(seed_ports, start=1):
        port, val = anneal_search(
            start_port=seed,
            universe=elite_pool,
            cov=cov,
            ticker_to_idx=ticker_to_idx,
            ranked_names=ranked_all,
            forced=forced
        )

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
            search_universe = tickers

        start = random_portfolio_from_universe(search_universe, forced=forced)

        port, val = anneal_search(
            start_port=start,
            universe=search_universe,
            cov=cov,
            ticker_to_idx=ticker_to_idx,
            ranked_names=ranked_all,
            forced=forced
        )

        if val > best_val + EPS:
            best_port, best_val = port, val

        seen_best_keys.add(tuple(sorted(port.items())))

        if (i + 1) % 20 == 0:
            print(
                f"Random start {i + 1}/{N_STARTS} complete | "
                f"best vol so far = {math.sqrt(best_val):.4%} | "
                f"distinct local optima seen = {len(seen_best_keys)}"
            )

    if best_port is None:
        raise RuntimeError("Search failed to produce any legal portfolio.")

    return best_port, best_val


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find the contest-legal portfolio with maximum volatility."
    )

    parser.add_argument(
        "--vol-file",
        default=VOL_FILE,
        help=f"Volatility CSV file. Default: {VOL_FILE}"
    )

    parser.add_argument(
        "--corr-file",
        default=CORR_FILE,
        help=f"Correlation matrix CSV file. Default: {CORR_FILE}"
    )

    parser.add_argument(
        "--forced",
        default=DEFAULT_FORCED_HOLDINGS,
        help='Forced holdings, e.g. "AAPL=25,NVDA=15,SHOP=10". Weights are percentages.'
    )

    args = parser.parse_args()

    best_port, best_var = solve_max_vol_portfolio(
        vol_file=args.vol_file,
        corr_file=args.corr_file,
        forced_holdings=args.forced
    )

    tickers, cov, ticker_to_idx, vol_map, corr_df = load_inputs(
        args.vol_file,
        args.corr_file
    )

    best_vol = portfolio_volatility(best_port, cov, ticker_to_idx)

    result = (
        pd.DataFrame(
            [
                {
                    "Ticker": t,
                    "Weight (%)": w,
                    "Volatility": vol_map[t],
                }
                for t, w in best_port.items()
            ]
        )
        .sort_values(["Weight (%)", "Ticker"], ascending=[False, True])
        .reset_index(drop=True)
    )

    print("\nBest portfolio found")
    print(result.to_string(index=False))
    print(f"\nPortfolio volatility: {best_vol:.4%}")
    print(f"Portfolio variance:   {best_var:.8f}")
