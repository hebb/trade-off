import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm

# -----------------------
# Settings
# -----------------------

TICKER_B = "TCL-B.TO"
TICKER_A = "TCL-A.TO"

PERIOD = "2y"

START_DATE = "2026-03-23"

WINDOWS = [20, 40, 60, 120]

# -----------------------
# Download data
# -----------------------

data = yf.download(
    [TICKER_B, TICKER_A],
    period=PERIOD,
    interval="1d",
    auto_adjust=False,
    progress=False,
    group_by="column"
)

df = pd.DataFrame({
    "open_b": data["Open"][TICKER_B],
    "close_b": data["Close"][TICKER_B],
    "open_a": data["Open"][TICKER_A],
    "close_a": data["Close"][TICKER_A],
}).dropna()

# -----------------------
# Build base variables
# -----------------------

df["spread"] = np.log(df["close_b"]) - np.log(df["close_a"])
df["next_oo_b"] = df["open_b"].shift(-2) / df["open_b"].shift(-1) - 1

# -----------------------
# Loop over windows
# -----------------------

results = []

for window in WINDOWS:

    temp = df.copy()

    # Rolling stats
    temp["mean"] = temp["spread"].rolling(window).mean()
    temp["std"] = temp["spread"].rolling(window).std()
    temp["z"] = (temp["spread"] - temp["mean"]) / temp["std"]

    # Post-event regression sample
    reg_df = temp[temp.index >= START_DATE].dropna(subset=["next_oo_b", "z"])

    if len(reg_df) < 10:
        print(f"\nWindow {window}: Not enough data")
        continue

    # Regression
    X = sm.add_constant(reg_df["z"])
    y = reg_df["next_oo_b"]

    model = sm.OLS(y, X).fit()

    alpha = model.params["const"]
    beta = model.params["z"]

    # Latest signal
    latest = temp.dropna(subset=["z"]).iloc[-1]
    latest_z = latest["z"]

    expected = alpha + beta * latest_z

    results.append({
        "window": window,
        "n_obs": int(model.nobs),
        "beta": beta,
        "p_value": model.pvalues["z"],
        "r2": model.rsquared,
        "latest_z": latest_z,
        "expected_return": expected
    })

# -----------------------
# Display results
# -----------------------

results_df = pd.DataFrame(results)

print("\nWindow comparison (post-March 23 regime):\n")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
