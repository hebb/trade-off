import pandas as pd
import yfinance as yf
import argparse
import numpy as np

def load_tickers_from_csv(file_path):
    df = pd.read_csv(file_path)

    if "Symbol" not in df.columns:
        raise ValueError("CSV must contain a 'Symbol' column")

    tickers = (
        df["Symbol"]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    return tickers

def download_prices(tickers, start=None, end=None):
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=True,
        threads=True,
        group_by="column"
    )

    if data.empty:
        raise ValueError("No price data was downloaded.")

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            raise ValueError("Downloaded data does not contain Close prices.")
        prices = data["Close"].copy()
    else:
        if "Close" not in data.columns:
            raise ValueError("Downloaded data does not contain Close prices.")
        if len(tickers) != 1:
            raise ValueError("Unexpected single-level columns for multiple tickers.")
        prices = data[["Close"]].copy()
        prices.columns = [tickers[0]]

    prices = prices.dropna(axis=1, how="all")
    return prices

def compute_returns(price_df):
    returns = price_df.pct_change(fill_method=None)
    returns = returns.iloc[1:]  # drop first row only
    returns = returns.dropna(axis=1, how="all")  # drop only stocks with no usable returns
    return returns

def compute_correlation(returns_df, min_periods=30):
    if returns_df.empty:
        raise ValueError("Returns table is empty after cleaning.")
    return returns_df.corr(min_periods=min_periods)

def choose_factor_count(n_assets, n_obs, requested=None):
    rank_limit = max(1, min(n_assets, n_obs) - 1)
    if requested is not None:
        if requested < 1:
            raise ValueError("--factors must be at least 1.")
        if requested > rank_limit:
            raise ValueError(
                f"--factors={requested} is too high for {n_assets} assets and {n_obs} observations. "
                f"Use {rank_limit} or fewer."
            )
        return requested

    return max(
        1,
        min(
            20,
            int(np.sqrt(max(n_assets, 1))),
            max(1, (n_obs - 1) // 3),
            rank_limit,
        ),
    )

def low_rank_approximation(matrix, factor_count):
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    return (u[:, :factor_count] * s[:factor_count]) @ vt[:factor_count, :]

def repair_to_psd(matrix, floor=1e-12):
    matrix = (matrix + matrix.T) / 2.0
    vals, vecs = np.linalg.eigh(matrix)
    vals = np.clip(vals, floor, None)
    repaired = (vecs * vals) @ vecs.T
    return (repaired + repaired.T) / 2.0

def covariance_to_correlation(cov):
    cov = repair_to_psd(cov)
    diag = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    corr = cov / np.outer(diag, diag)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr

def estimate_factor_covariance(
    returns_df,
    min_periods=30,
    factors=None,
    em_iterations=25,
    em_tolerance=1e-6,
    idio_floor=1e-8,
):
    if returns_df.empty:
        raise ValueError("Returns table is empty after cleaning.")

    usable = returns_df.loc[:, returns_df.count() >= min_periods].copy()
    usable = usable.dropna(axis=0, how="all")
    if usable.shape[1] < 2:
        raise ValueError(
            f"Only {usable.shape[1]} ticker has at least {min_periods} return observations; need at least 2."
        )
    if usable.shape[0] < 3:
        raise ValueError("Need at least 3 return rows to estimate a factor covariance matrix.")

    tickers = list(usable.columns)
    means = usable.mean(axis=0, skipna=True)
    demeaned = usable.subtract(means, axis=1)
    missing = demeaned.isna().to_numpy()
    x = demeaned.fillna(0.0).to_numpy(dtype=float, copy=True)

    factor_count = choose_factor_count(x.shape[1], x.shape[0], factors)

    if missing.any() and em_iterations > 0:
        previous_missing = x[missing].copy()
        for _ in range(em_iterations):
            low_rank = low_rank_approximation(x, factor_count)
            x[missing] = low_rank[missing]

            current_missing = x[missing]
            denom = max(float(np.linalg.norm(previous_missing)), 1e-12)
            if float(np.linalg.norm(current_missing - previous_missing)) / denom < em_tolerance:
                break
            previous_missing = current_missing.copy()

    sample_cov = (x.T @ x) / max(x.shape[0] - 1, 1)
    sample_cov = repair_to_psd(sample_cov)

    vals, vecs = np.linalg.eigh(sample_cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    factor_vals = np.clip(vals[:factor_count], 0.0, None)
    exposures = vecs[:, :factor_count]
    factor_cov = (exposures * factor_vals) @ exposures.T

    residual_var = np.diag(sample_cov - factor_cov).copy()
    positive_vars = np.diag(sample_cov)
    scale = float(np.nanmedian(positive_vars[positive_vars > 0])) if np.any(positive_vars > 0) else 1.0
    residual_floor = max(float(idio_floor) * scale, 1e-18)
    residual_var = np.clip(residual_var, residual_floor, None)

    cov = factor_cov + np.diag(residual_var)
    cov = repair_to_psd(cov)
    return pd.DataFrame(cov, index=tickers, columns=tickers), factor_count

def compute_factor_correlation(
    returns_df,
    min_periods=30,
    factors=None,
    em_iterations=25,
    em_tolerance=1e-6,
    idio_floor=1e-8,
):
    cov, factor_count = estimate_factor_covariance(
        returns_df,
        min_periods=min_periods,
        factors=factors,
        em_iterations=em_iterations,
        em_tolerance=em_tolerance,
        idio_floor=idio_floor,
    )
    corr = covariance_to_correlation(cov.to_numpy(dtype=float))
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns), factor_count

def main():
    parser = argparse.ArgumentParser(description="Compute correlation matrix of daily returns.")
    parser.add_argument("stock_list_csv", help="Path to StockList.csv")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", default="correlation_matrix.csv")
    parser.add_argument("--method", choices=["factor", "raw"], default="factor",
                        help="Correlation estimator. 'factor' uses PCA factors plus idiosyncratic variance.")
    parser.add_argument("--factors", type=int, default=None,
                        help="Number of PCA factors for --method factor. Defaults to a data-size-aware value.")
    parser.add_argument("--em-iterations", type=int, default=25,
                        help="Maximum EM-style iterations used to impute missing returns for --method factor.")
    parser.add_argument("--em-tolerance", type=float, default=1e-6,
                        help="Relative convergence tolerance for missing-return imputation.")
    parser.add_argument("--idio-floor", type=float, default=1e-8,
                        help="Minimum idiosyncratic variance as a fraction of median asset variance.")
    parser.add_argument("--min-periods", type=int, default=30,
                        help="Minimum overlapping return observations required for a correlation")

    args = parser.parse_args()

    tickers = load_tickers_from_csv(args.stock_list_csv)
    print(f"Loaded {len(tickers)} tickers")

    prices = download_prices(tickers, args.start, args.end)
    print(f"Price table shape: {prices.shape}")

    returns = compute_returns(prices)
    print(f"Returns table shape: {returns.shape}")

    if args.method == "raw":
        corr = compute_correlation(returns, min_periods=args.min_periods)
    else:
        corr, factor_count = compute_factor_correlation(
            returns,
            min_periods=args.min_periods,
            factors=args.factors,
            em_iterations=args.em_iterations,
            em_tolerance=args.em_tolerance,
            idio_floor=args.idio_floor,
        )
        print(f"Estimated factor covariance with {factor_count} factor(s)")
    print(f"Correlation matrix shape: {corr.shape}")

    corr.to_csv(args.output)
    print(f"Saved correlation matrix to {args.output}")

if __name__ == "__main__":
    main()
