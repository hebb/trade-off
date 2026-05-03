#!/usr/bin/env python3
"""
infer_weights.py

Infer portfolio weights for The Globe and Mail “Trade Off” contest.

This version keeps your strict date policy and user-friendly output, but adds a
non-brute-force solver for larger portfolios by formulating the problem as a
mixed-integer linear programme (MILP).

Behaviour summary
- Strict dates: for each player, uses that player's last N+1 portfolio dates (if --max_intervals N),
  never substitutes dates, and reports explicit missing (date,ticker) pairs.
- Currency: CAD tickers end with .TO or :CA; all other tickers treated as USD.
- FX: taken from stock_closes as a ticker (default CAD_USD). This script uses an end-date keyed
  change factor map as before.
- Solving:
    * For small portfolios (default <= 5 holdings): brute-force enumeration with UNIQUE / NOT UNIQUE / NO SOLUTIONS.
    * For larger portfolios: MILP:
        - If feasible within tolerance: reports a feasible solution.
        - Otherwise: reports best solution found minimising max absolute dollar error (E).
  (Counting uniqueness for large n is not attempted.)

Requirements for MILP
- Tries PuLP first; if unavailable, tries scipy.optimize.milp; otherwise errors with guidance.
"""

import argparse
import itertools
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd

CAD_MARKERS = (":CA", ".TO")
FX_TICKER_DEFAULT = "CAD_USD"


# -------------------------
# Helpers
# -------------------------

def canonical_ticker(t: str) -> str:
    return str(t).strip().upper()


def is_cad_ticker(t: str) -> bool:
    return canonical_ticker(t).endswith(CAD_MARKERS)


def ticker_base(t: str) -> str:
    t = canonical_ticker(t)
    for suf in CAD_MARKERS:
        if t.endswith(suf):
            return t[:-len(suf)]
    return t


def candidate_variants(t: str) -> List[str]:
    t = canonical_ticker(t)
    base = ticker_base(t)
    variants = [t]
    if t != base:
        variants.append(base)
    variants += [base + ".TO", base + ":CA"]
    out, seen = [], set()
    for v in variants:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def parse_holdings(s: Any) -> List[str]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    return [canonical_ticker(x) for x in str(s).split(",") if x.strip()]


def has_missing_weights(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    s = str(x).strip()
    return s == "" or s.lower() == "nan"


def resolve_holdings_to_prices(
    holdings: List[str],
    available_tickers: set,
    verbose: bool = False,
) -> Tuple[Optional[List[str]], Dict[str, str], Optional[str]]:
    """
    Resolve each holding to an existing ticker in stock_closes using suffix variants.
    Returns (resolved, mapping, err). If unresolved, resolved=None and err is non-empty.
    """
    mapping: Dict[str, str] = {}
    resolved: List[str] = []

    for h in holdings:
        h = canonical_ticker(h)
        existing = [v for v in candidate_variants(h) if v in available_tickers]
        if not existing:
            return None, {}, f"Unresolvable ticker {h}"
        choice = existing[0]
        if verbose and choice != h and is_cad_ticker(choice) and not is_cad_ticker(h):
            print(f"NOTE: {h} not found; using {choice} from prices (appears CAD).")
        mapping[h] = choice
        resolved.append(choice)

    return resolved, mapping, None


# -------------------------
# Strict date alignment
# -------------------------

def required_dates(pv: pd.DataFrame, player: str, max_intervals: Optional[int]) -> List[pd.Timestamp]:
    pv_p = pv[pv["player"] == player].sort_values("date")
    if len(pv_p) < 2:
        return []
    if max_intervals is None:
        return list(pv_p["date"])
    return list(pv_p["date"].iloc[-(max_intervals + 1):])


def align_player_data_strict(
    player: str,
    holdings: List[str],
    pv: pd.DataFrame,
    px: pd.DataFrame,
    max_intervals: Optional[int],
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], List[Tuple[str, str]]]:
    dates = required_dates(pv, player, max_intervals)
    if len(dates) < 2:
        return None, None, [("INSUFFICIENT_PLAYER_DATES", "")]

    pv_p = pv[(pv["player"] == player) & (pv["date"].isin(dates))].sort_values("date")

    px_piv = (
        px[(px["ticker"].isin(holdings)) & (px["date"].isin(dates))]
        .pivot(index="date", columns="ticker", values="close")
        .sort_index()
    )

    missing: List[Tuple[str, str]] = []
    for d in dates:
        ds = pd.Timestamp(d).date().isoformat()
        if d not in px_piv.index:
            for t in holdings:
                missing.append((ds, t))
            continue

        row = px_piv.loc[d]
        for t in holdings:
            if t not in px_piv.columns or pd.isna(row.get(t, np.nan)):
                missing.append((ds, t))

    if missing:
        return None, None, missing

    return pv_p.reset_index(drop=True), px_piv.loc[dates], []


def gross_returns(px_piv: pd.DataFrame, holdings: List[str]) -> np.ndarray:
    prices = np.column_stack([px_piv[t].to_numpy(float) for t in holdings])
    return prices[1:] / prices[:-1]


# -------------------------
# Output formatting
# -------------------------

def format_missing(missing: List[Tuple[str, str]], max_show: int = 10) -> str:
    if not missing:
        return ""
    shown = missing[:max_show]
    s = ", ".join(f"({d},{t})" for d, t in shown)
    if len(missing) > max_show:
        s += f", ... (+{len(missing) - max_show} more)"
    return s


def format_weights(w_pct: np.ndarray, cash: int, tickers: List[str]) -> str:
    parts = [f"{tickers[i]}:{int(w_pct[i])}%" for i in range(len(w_pct))]
    parts.append(f"CASH:{int(cash)}%")
    return ", ".join(parts)


# -------------------------
# Brute force (small n)
# -------------------------

def enumerate_weights_allow_cash(n: int, min_w: int = 5, max_w: int = 25):
    out = []
    for combo in itertools.product(range(min_w, max_w + 1), repeat=n):
        s = sum(combo)
        if s > 100:
            continue
        cash = 100 - s
        out.append((np.array(combo, dtype=np.int16), int(cash)))
    return out


def solve_bruteforce(
    V0: np.ndarray,
    V1: np.ndarray,
    R: np.ndarray,                 # shape (T, n)
    tol_dollars: float,
    max_solutions: int,
) -> Dict[str, Any]:
    n = R.shape[1]
    candidates = enumerate_weights_allow_cash(n)

    best = None
    best_err = float("inf")
    count = 0
    unique = None

    for w_pct, cash in candidates:
        w = w_pct.astype(float) / 100.0
        pred = V0 * (1.0 + (R @ w))
        err_vec = pred - V1
        maxerr = float(np.max(np.abs(err_vec)))

        if maxerr < best_err:
            best_err = maxerr
            best = (w_pct.copy(), int(cash), best_err)

        if maxerr <= tol_dollars:
            count += 1
            if count == 1:
                unique = (w_pct.copy(), int(cash))
            else:
                unique = None
            if count >= max_solutions:
                break

    if count == 0:
        return {"status": "no_solutions", "closest": best}
    if unique is not None:
        return {"status": "unique", "solution": unique}
    return {"status": "not_unique", "count": count}


# -------------------------
# MILP (large n)
# -------------------------

def solve_milp(
    V0: np.ndarray,
    V1: np.ndarray,
    R: np.ndarray,                 # shape (T, n)
    tol_dollars: float,
    time_limit_s: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Minimise E subject to -E <= error_t <= E for all t, and weights integer 5..25, sum<=100.
    If the optimal E <= tol_dollars, we treat as feasible within tolerance.
    """
    T, n = R.shape
    # error_t = V0_t*(R_t @ w) + V0_t - V1_t, with w in fractions.
    # Use p_i as integer percentage points, w_i = p_i/100.
    # error_t = V0_t*(R_t @ p)/100 + (V0_t - V1_t).

    # Prefer PuLP if available.
    try:
        import pulp  # type: ignore
        use_pulp = True
    except Exception:
        use_pulp = False

    if use_pulp:
        import pulp  # type: ignore

        prob = pulp.LpProblem("infer_weights", pulp.LpMinimize)
        p = [pulp.LpVariable(f"p_{i}", lowBound=5, upBound=25, cat="Integer") for i in range(n)]
        E = pulp.LpVariable("E", lowBound=0, cat="Continuous")

        prob += E
        prob += pulp.lpSum(p) <= 100

        for t in range(T):
            expr = pulp.lpSum((float(V0[t]) * float(R[t, i]) / 100.0) * p[i] for i in range(n)) + float(V0[t] - V1[t])
            prob += expr <= E
            prob += -expr <= E

        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s) if time_limit_s else pulp.PULP_CBC_CMD(msg=False)
        status = prob.solve(solver)

        if pulp.LpStatus[status] not in ("Optimal", "Not Solved", "Undefined", "Infeasible"):
            return {"status": "insufficient", "reason": f"MILP solver status {pulp.LpStatus[status]}"}

        if pulp.LpStatus[status] == "Infeasible":
            return {"status": "insufficient", "reason": "MILP infeasible"}

        p_sol = np.array([int(round(v.value())) for v in p], dtype=np.int16)
        cash = int(100 - int(np.sum(p_sol)))
        E_val = float(E.value()) if E.value() is not None else float("inf")

        if E_val <= tol_dollars:
            return {"status": "feasible", "solution": (p_sol, cash), "E": E_val}
        return {"status": "no_solutions", "closest": (p_sol, cash, E_val), "milp": True}

    # Fall back to SciPy milp if available.
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds  # type: ignore
    except Exception:
        return {
            "status": "cannot_solve",
            "reason": "No MILP backend available. Install 'pulp' (recommended) or SciPy >= 1.9 for scipy.optimize.milp."
        }

    # Variables: [p_0..p_{n-1}, E]
    m = n + 1
    c = np.zeros(m, dtype=float)
    c[-1] = 1.0  # minimise E

    integrality = np.zeros(m, dtype=int)
    integrality[:n] = 1  # integer p_i
    integrality[-1] = 0  # continuous E

    lb = np.zeros(m, dtype=float)
    ub = np.zeros(m, dtype=float)
    lb[:n] = 5
    ub[:n] = 25
    lb[-1] = 0
    ub[-1] = np.inf
    bounds = Bounds(lb, ub)

    A = []
    lo = []
    hi = []

    # sum p_i <= 100
    row = np.zeros(m, dtype=float)
    row[:n] = 1.0
    A.append(row)
    lo.append(-np.inf)
    hi.append(100.0)

    # For each t: expr <= E and -expr <= E
    # expr = sum_i (V0*R/100)*p_i + (V0 - V1)
    # expr - E <= 0
    # -expr - E <= 0
    for t in range(T):
        coef = (float(V0[t]) * R[t, :] / 100.0).astype(float)
        const = float(V0[t] - V1[t])

        row1 = np.zeros(m, dtype=float)
        row1[:n] = coef
        row1[-1] = -1.0
        A.append(row1)
        lo.append(-np.inf)
        hi.append(-const)

        row2 = np.zeros(m, dtype=float)
        row2[:n] = -coef
        row2[-1] = -1.0
        A.append(row2)
        lo.append(-np.inf)
        hi.append(const)

    lin = LinearConstraint(np.vstack(A), np.array(lo), np.array(hi))
    res = milp(c=c, constraints=[lin], integrality=integrality, bounds=bounds)

    if not res.success or res.x is None:
        return {"status": "insufficient", "reason": "SciPy MILP failed", "detail": str(res.message)}

    x = res.x
    p_sol = np.array([int(round(v)) for v in x[:n]], dtype=np.int16)
    cash = int(100 - int(np.sum(p_sol)))
    E_val = float(x[-1])

    if E_val <= tol_dollars:
        return {"status": "feasible", "solution": (p_sol, cash), "E": E_val}
    return {"status": "no_solutions", "closest": (p_sol, cash, E_val), "milp": True}


# -------------------------
# Main solve per player
# -------------------------

def solve_player_weights(
    player: str,
    holdings_raw: List[str],
    holdings_raw_str: str,
    pv: pd.DataFrame,
    px: pd.DataFrame,
    available_tickers: set,
    fx_map: Dict[pd.Timestamp, float],
    tol_dollars: float,
    max_solutions: int,
    max_intervals: Optional[int],
    brute_force_max_n: int,
    milp_time_limit_s: Optional[int],
) -> Dict[str, Any]:

    resolved, _, err = resolve_holdings_to_prices(holdings_raw, available_tickers, verbose=False)
    if resolved is None:
        return {"status": "insufficient", "reason": err, "holdings_raw": holdings_raw_str}

    pv_p, px_piv, missing = align_player_data_strict(player, resolved, pv, px, max_intervals)
    if pv_p is None:
        if missing and missing[0][0] == "INSUFFICIENT_PLAYER_DATES":
            return {"status": "insufficient", "reason": "INSUFFICIENT_PLAYER_DATES", "holdings_raw": holdings_raw_str}
        return {"status": "insufficient", "reason": "missing_prices", "missing": missing, "holdings_raw": holdings_raw_str}

    V0 = pv_p["close_value"].to_numpy(float)[:-1]
    V1 = pv_p["close_value"].to_numpy(float)[1:]
    T = len(V0)

    G_local = gross_returns(px_piv, resolved)  # shape (T, n)
    end_dates = list(px_piv.index[1:])

    missing_fx = [pd.Timestamp(d).date().isoformat() for d in end_dates if d not in fx_map]
    if missing_fx:
        return {"status": "insufficient", "reason": "missing_fx", "missing_fx": missing_fx, "holdings_raw": holdings_raw_str}

    usd_mask = np.array([not is_cad_ticker(t) for t in resolved], dtype=bool)
    g_vec = np.array([fx_map[d] for d in end_dates], dtype=float)

    G_cad = G_local.copy()
    if np.any(usd_mask):
        G_cad[:, usd_mask] *= g_vec[:, None]

    R = G_cad - 1.0  # shape (T, n)
    n = R.shape[1]

    if n <= brute_force_max_n:
        out = solve_bruteforce(V0=V0, V1=V1, R=R, tol_dollars=tol_dollars, max_solutions=max_solutions)
        out["resolved"] = resolved
        out["holdings_raw"] = holdings_raw_str
        out["intervals_used"] = T
        return out

    out = solve_milp(V0=V0, V1=V1, R=R, tol_dollars=tol_dollars, time_limit_s=milp_time_limit_s)
    out["resolved"] = resolved
    out["holdings_raw"] = holdings_raw_str
    out["intervals_used"] = T
    return out


# -------------------------
# Entry point
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--portfolio_values", required=True)
    ap.add_argument("--stock_closes", required=True)
    ap.add_argument("--leaderboard", required=True)
    ap.add_argument("--tol_dollars", type=float, default=0.75)
    ap.add_argument("--max_intervals", type=int, default=None)
    ap.add_argument("--max_solutions", type=int, default=50000)
    ap.add_argument("--fx_ticker", type=str, default=FX_TICKER_DEFAULT)
    ap.add_argument("--bruteforce_max_n", type=int, default=5, help="Use brute force only when holdings <= this.")
    ap.add_argument("--milp_time_limit_s", type=int, default=None, help="Optional MILP time limit in seconds.")
    args = ap.parse_args()

    pv = pd.read_csv(args.portfolio_values, parse_dates=["date"])
    px = pd.read_csv(args.stock_closes, parse_dates=["date"])
    lb = pd.read_csv(args.leaderboard)

    pv["player"] = pv["player"].astype(str).str.strip()
    pv["close_value"] = pd.to_numeric(pv["close_value"], errors="coerce")

    px["ticker"] = px["ticker"].map(canonical_ticker)
    px["close"] = pd.to_numeric(px["close"], errors="coerce")

    available = set(px["ticker"].unique())

    fx_ticker = canonical_ticker(args.fx_ticker)
    fx_px = px[px["ticker"] == fx_ticker].set_index("date").sort_index()
    if fx_px.empty:
        raise ValueError(f"FX series {fx_ticker} not found in stock_closes.")

    fx = fx_px["close"].to_numpy(float)
    fx_dates = list(fx_px.index)

    # Map by interval end-date: factor to apply to USD gross returns to express in CAD.
    # This preserves your existing convention. If you change FX definition, change here.
    fx_map: Dict[pd.Timestamp, float] = {fx_dates[i + 1]: fx[i] / fx[i + 1] for i in range(len(fx) - 1)}

    printed = 0
    skipped_has_weights = 0
    skipped_no_holdings = 0

    for idx, row in lb.iterrows():
        player = str(row.get("Name", "")).strip()
        holdings_raw_str = str(row.get("Holdings", "")).strip()
        holdings_raw = parse_holdings(holdings_raw_str)

        if not holdings_raw:
            skipped_no_holdings += 1
            continue

        if not has_missing_weights(row.get("Weights", None)):
            skipped_has_weights += 1
            continue

        res = solve_player_weights(
            player=player,
            holdings_raw=holdings_raw,
            holdings_raw_str=holdings_raw_str,
            pv=pv,
            px=px,
            available_tickers=available,
            fx_map=fx_map,
            tol_dollars=args.tol_dollars,
            max_solutions=args.max_solutions,
            max_intervals=args.max_intervals,
            brute_force_max_n=args.bruteforce_max_n,
            milp_time_limit_s=args.milp_time_limit_s,
        )

        prefix = f"{player} (row {idx})"
        status = res.get("status", "")

        if status == "insufficient":
            reason = res.get("reason", "")
            if reason == "missing_prices":
                msg = format_missing(res.get("missing", []))
                print(f"{prefix}: INSUFFICIENT DATA (missing prices): {msg}")
            elif reason == "missing_fx":
                miss_fx = ", ".join(res.get("missing_fx", []))
                print(f"{prefix}: INSUFFICIENT DATA (missing FX for interval end-date(s)): {miss_fx}")
            else:
                print(f"{prefix}: INSUFFICIENT DATA ({reason}). Holdings=\"{holdings_raw_str}\"")

        elif status == "cannot_solve":
            print(f"{prefix}: CANNOT SOLVE ({res.get('reason','')}). Holdings=\"{holdings_raw_str}\"")

        elif status == "unique":
            w_pct, cash = res["solution"]
            resolved = res["resolved"]
            print(f"{prefix}: UNIQUE (intervals_used={res['intervals_used']}) -> {format_weights(w_pct, cash, resolved)}")

        elif status == "not_unique":
            print(f"{prefix}: NOT UNIQUE (intervals_used={res['intervals_used']}) -> {res['count']} solutions")

        elif status == "feasible":
            w_pct, cash = res["solution"]
            resolved = res["resolved"]
            E = float(res.get("E", float("nan")))
            print(
                f"{prefix}: FEASIBLE (intervals_used={res['intervals_used']}, tol=${args.tol_dollars:.2f}). "
                f"Max-abs error=${E:.2f}. Weights: {format_weights(w_pct, cash, resolved)}"
            )

        else:  # no_solutions
            w_pct, cash, err = res["closest"]
            resolved = res["resolved"]
            tag = " (MILP)" if res.get("milp") else ""
            print(
                f"{prefix}: NO SOLUTIONS{tag} (intervals_used={res['intervals_used']}, tol=${args.tol_dollars:.2f}). "
                f"Closest max-abs error=${float(err):.2f}. Closest weights: {format_weights(w_pct, cash, resolved)}"
            )

        printed += 1

    print(f"\nSummary: printed={printed}, skipped_has_weights={skipped_has_weights}, skipped_no_holdings={skipped_no_holdings}")


if __name__ == "__main__":
    main()


