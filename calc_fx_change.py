#!/usr/bin/env python3
"""
calc_fx_change.py

Compute implied FX change factors g_t = (USD/CAD)_{t+1}/(USD/CAD)_t from ONE anchor portfolio.

Inputs
------
portfolio_values.csv: date, player, close_value
stock_closes.csv:     date, ticker, close
leaderboard.csv:      Name, Holdings, Weights  (anchor must have Weights)

Currency rule
-------------
- Tickers ending with .TO or :CA are CAD-quoted.
- Everything else is USD-quoted.

Weights
-------
Anchor weights may be given either as percentage points (e.g. 18,17,...) or fractions (0.18,0.17,...).
The script auto-detects based on sum(weights): if > 1.5, it divides by 100.

Date rule
---------
- Uses the anchor player's last N+1 dates if --max_intervals N is set; otherwise uses all available dates.
- Never substitutes dates; if any required price is missing, it errors listing missing (date,ticker).

Output
------
One line per interval: start_date -> end_date, g (USD/CAD factor), and 1/g (CAD/USD factor).
Optionally writes CSV via --out_csv.
"""

import argparse
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

CAD_MARKERS = (":CA", ".TO")


def canonical_ticker(t: str) -> str:
    return str(t).strip().upper()


def is_cad_ticker(t: str) -> bool:
    t = canonical_ticker(t)
    return t.endswith(CAD_MARKERS)


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


def parse_holdings(s: str) -> List[str]:
    if pd.isna(s):
        return []
    return [canonical_ticker(x) for x in str(s).split(",") if x.strip()]


def parse_weights(s: str) -> Optional[List[float]]:
    if s is None:
        return None
    if isinstance(s, float) and np.isnan(s):
        return None
    st = str(s).strip()
    if st == "" or st.lower() == "nan":
        return None
    parts = [p.strip() for p in st.split(",") if p.strip() != ""]
    try:
        return [float(x) for x in parts]
    except ValueError:
        return None


def weights_to_fractions(weights_raw: List[float]) -> np.ndarray:
    w = np.array(weights_raw, dtype=float)
    s = float(np.sum(w))
    if s > 1.5:
        w = w / 100.0
    return w


def resolve_holdings_to_prices(holdings: List[str], available_tickers: set) -> Tuple[List[str], Dict[str, str]]:
    mapping: Dict[str, str] = {}
    resolved: List[str] = []

    for h in holdings:
        h = canonical_ticker(h)
        existing = [v for v in candidate_variants(h) if v in available_tickers]
        if not existing:
            raise ValueError(f"Unresolvable ticker: {h}")
        choice = h if h in existing else existing[0]
        if choice != h and (choice.endswith(".TO") or choice.endswith(":CA")) and not (h.endswith(".TO") or h.endswith(":CA")):
            print(f"NOTE: {h} not found; using {choice} from prices (appears CAD).")
        mapping[h] = choice
        resolved.append(choice)

    return resolved, mapping


def required_anchor_dates(pv: pd.DataFrame, player: str, max_intervals: Optional[int]) -> List[pd.Timestamp]:
    pv_p = pv[pv["player"] == player].dropna(subset=["close_value"]).sort_values("date")
    if len(pv_p) < 2:
        raise ValueError("Anchor has fewer than 2 portfolio close values.")
    if max_intervals is None:
        return list(pv_p["date"].to_list())
    need = min(len(pv_p), max_intervals + 1)
    return list(pv_p["date"].iloc[-need:].to_list())


def strict_price_pivot(px: pd.DataFrame, tickers: List[str], dates: List[pd.Timestamp]) -> pd.DataFrame:
    dates_sorted = sorted(pd.Timestamp(d) for d in dates)
    px_sub = px[(px["ticker"].isin(tickers)) & (px["date"].isin(dates_sorted))]
    piv = px_sub.pivot(index="date", columns="ticker", values="close").sort_index()

    missing = []
    for d in dates_sorted:
        ds = pd.Timestamp(d).date().isoformat()
        if d not in piv.index:
            for t in tickers:
                missing.append((ds, t))
            continue
        row = piv.loc[d]
        for t in tickers:
            if (t not in piv.columns) or pd.isna(row.get(t, np.nan)):
                missing.append((ds, t))

    if missing:
        s = ", ".join([f"({d},{t})" for d, t in missing[:80]])
        if len(missing) > 80:
            s += f", ... (+{len(missing)-80} more)"
        raise ValueError("Missing stock closes for required dates: " + s)

    return piv.loc[dates_sorted]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--portfolio_values", required=True)
    ap.add_argument("--stock_closes", required=True)
    ap.add_argument("--leaderboard", required=True)
    ap.add_argument("--anchor_name", required=True)
    ap.add_argument("--max_intervals", type=int, default=None)
    ap.add_argument("--out_csv", default=None, help="Optional CSV output path.")
    args = ap.parse_args()

    pv = pd.read_csv(args.portfolio_values)
    px = pd.read_csv(args.stock_closes)
    lb = pd.read_csv(args.leaderboard)

    pv["date"] = pd.to_datetime(pv["date"])
    pv["player"] = pv["player"].astype(str).str.strip()
    pv["close_value"] = pd.to_numeric(pv["close_value"], errors="coerce")

    px["date"] = pd.to_datetime(px["date"])
    px["ticker"] = px["ticker"].map(canonical_ticker)
    px["close"] = pd.to_numeric(px["close"], errors="coerce")
    available_tickers = set(px["ticker"].unique())

    lb["Name_up"] = lb["Name"].astype(str).str.strip().str.upper()
    anchor_up = str(args.anchor_name).strip().upper()
    row = lb[lb["Name_up"] == anchor_up]
    if row.empty:
        raise ValueError("Anchor name not found in leaderboard.")
    row = row.iloc[0]
    anchor_name = str(row["Name"]).strip()

    holdings_raw = parse_holdings(row.get("Holdings", ""))
    weights_raw = parse_weights(row.get("Weights", None))
    if weights_raw is None:
        raise ValueError("Anchor weights are missing/empty in leaderboard.")
    if len(weights_raw) != len(holdings_raw):
        raise ValueError("Anchor weights count does not match anchor holdings count.")

    resolved, _ = resolve_holdings_to_prices(holdings_raw, available_tickers)

    w = weights_to_fractions(weights_raw)
    w_cash = max(0.0, 1.0 - float(np.sum(w)))

    dates = required_anchor_dates(pv, anchor_name, args.max_intervals)
    dates = sorted(pd.Timestamp(d) for d in dates)

    pv_a = pv[(pv["player"] == anchor_name) & (pv["date"].isin(dates))].dropna(subset=["close_value"]).sort_values("date")
    if list(pv_a["date"].to_list()) != dates:
        raise ValueError("Anchor portfolio_values missing required date(s).")

    piv = strict_price_pivot(px, resolved, dates)

    usd_mask = np.array([not is_cad_ticker(t) for t in resolved], dtype=bool)
    cad_mask = ~usd_mask
    if not np.any(usd_mask):
        raise ValueError("Anchor resolves to all-CAD tickers; cannot infer FX.")

    out_rows = []
    V = pv_a["close_value"].to_numpy(float)

    for i in range(len(dates) - 1):
        d0, d1 = dates[i], dates[i + 1]
        V0, V1 = float(V[i]), float(V[i + 1])
        Gp = V1 / V0

        p0 = piv.loc[d0, resolved].to_numpy(float)
        p1 = piv.loc[d1, resolved].to_numpy(float)
        G = p1 / p0

        cad_term = float(G[cad_mask] @ w[cad_mask]) if np.any(cad_mask) else 0.0
        usd_term = float(G[usd_mask] @ w[usd_mask])

        if np.isclose(usd_term, 0.0):
            raise ValueError(f"USD term is zero for interval ending {d1.date().isoformat()}.")

        g = (Gp - w_cash - cad_term) / usd_term
        inv = 1.0 / g

        out_rows.append({
            "start_date": d0.date().isoformat(),
            "end_date": d1.date().isoformat(),
            "Gp": Gp,
            "cash_weight": w_cash,
            "cad_term": cad_term,
            "usd_term": usd_term,
            "g_usd_per_cad_factor": g,
            "g_usd_per_cad_pct": (g - 1.0) * 100.0,
            "g_cad_per_usd_factor": inv,
            "g_cad_per_usd_pct": (inv - 1.0) * 100.0,
        })

    df_out = pd.DataFrame(out_rows)
    for _, r in df_out.iterrows():
        print(
            f"{r['start_date']} -> {r['end_date']}: "
            f"g(USD/CAD)={r['g_usd_per_cad_factor']:.8f} ({r['g_usd_per_cad_pct']:+.4f}%), "
            f"1/g(CAD/USD)={r['g_cad_per_usd_factor']:.8f} ({r['g_cad_per_usd_pct']:+.4f}%)"
        )

    if args.out_csv:
        df_out.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()

