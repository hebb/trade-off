import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm

TICKER_B = "TCL-B.TO"
TICKER_A = "TCL-A.TO"

PERIOD = "2y"
START_DATE = "2026-03-23"
ROLLING_WINDOW = 20

# If True, rolling mean/std can use pre-START_DATE data.
# Regression still uses only rows on/after START_DATE.
USE_PRE_START_FOR_ZSCORE = True

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
    "volume_b": data["Volume"][TICKER_B],
}).dropna()

df["next_oo_b"] = df["open_b"].shift(-2) / df["open_b"].shift(-1) - 1
df["spread"] = np.log(df["close_b"]) - np.log(df["close_a"])

if USE_PRE_START_FOR_ZSCORE:
    z_base = df.copy()
else:
    z_base = df[df.index >= START_DATE].copy()

z_base["spread_mean"] = z_base["spread"].rolling(ROLLING_WINDOW).mean()
z_base["spread_std"] = z_base["spread"].rolling(ROLLING_WINDOW).std()
z_base["spread_z"] = (
    (z_base["spread"] - z_base["spread_mean"]) / z_base["spread_std"]
)

df["spread_z"] = z_base["spread_z"]

reg_df = df[df.index >= START_DATE].dropna(subset=["next_oo_b", "spread_z"]).copy()

if len(reg_df) < 5:
    raise ValueError(
        f"Not enough usable observations. Only {len(reg_df)} rows. "
        f"Set USE_PRE_START_FOR_ZSCORE=True, reduce ROLLING_WINDOW, or use a later date when more data exists."
    )

X = sm.add_constant(reg_df["spread_z"])
y = reg_df["next_oo_b"]

model = sm.OLS(y, X).fit()

alpha = model.params["const"]
beta = model.params["spread_z"]

# Residual daily volatility: actual return minus expected return
reg_df["expected_next_oo"] = model.predict(X)
reg_df["residual"] = reg_df["next_oo_b"] - reg_df["expected_next_oo"]

residual_daily_std = reg_df["residual"].std(ddof=1)
raw_daily_std = reg_df["next_oo_b"].std(ddof=1)

model = sm.OLS(y, X).fit()

alpha = model.params["const"]
beta = model.params["spread_z"]

latest = df.dropna(subset=["spread_z"]).iloc[-1]

expected_next_oo = alpha + beta * latest["spread_z"]

print("Regression: next open-to-open return ~ spread_z")
print()
print(f"Start date:                 {START_DATE}")
print(f"Rolling window:             {ROLLING_WINDOW}")
print(f"Use pre-start for z-score:   {USE_PRE_START_FOR_ZSCORE}")
print(f"Observations:               {int(model.nobs)}")
print()
print(f"Alpha:                      {alpha:.4%}")
print(f"Beta:                       {beta:.4%}")
print(f"Beta p-value:               {model.pvalues['spread_z']:.4f}")
print(f"R-squared:                  {model.rsquared:.4f}")
print(f"Raw daily OO std:           {raw_daily_std:.4%}")
print(f"Residual daily OO std:      {residual_daily_std:.4%}")
print()
print("Latest signal")
print()
print(f"Date:                       {latest.name.date()}")
print(f"TCL-B close:                 {latest['close_b']:.4f}")
print(f"TCL-A close:                 {latest['close_a']:.4f}")
print(f"Spread z-score:              {latest['spread_z']:.3f}")
print()
print("Expected next open-to-open return")
print()
print(f"{expected_next_oo:.4%}")

if expected_next_oo > 0:
    print("Signal direction: positive for TCL-B")
else:
    print("Signal direction: negative for TCL-B")
