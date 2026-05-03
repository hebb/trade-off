import pandas as pd
import yfinance as yf
import argparse

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

def main():
    parser = argparse.ArgumentParser(description="Compute correlation matrix of daily returns.")
    parser.add_argument("stock_list_csv", help="Path to StockList.csv")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", default="correlation_matrix.csv")
    parser.add_argument("--min-periods", type=int, default=30,
                        help="Minimum overlapping return observations required for a correlation")

    args = parser.parse_args()

    tickers = load_tickers_from_csv(args.stock_list_csv)
    print(f"Loaded {len(tickers)} tickers")

    prices = download_prices(tickers, args.start, args.end)
    print(f"Price table shape: {prices.shape}")

    returns = compute_returns(prices)
    print(f"Returns table shape: {returns.shape}")

    corr = compute_correlation(returns, min_periods=args.min_periods)
    print(f"Correlation matrix shape: {corr.shape}")

    corr.to_csv(args.output)
    print(f"Saved correlation matrix to {args.output}")

if __name__ == "__main__":
    main()
