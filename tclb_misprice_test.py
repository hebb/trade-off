import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm

tickers = ["TCL-B.TO", "TCL-A.TO"]

data = yf.download(
    tickers,
    period="2y",
    interval="1d",
    auto_adjust=False,
    progress=False,
    group_by="column"
)

# Extract fields
open_b = data["Open"]["TCL-B.TO"]
close_b = data["Close"]["TCL-B.TO"]
vol_b = data["Volume"]["TCL-B.TO"]

open_a = data["Open"]["TCL-A.TO"]
close_a = data["Close"]["TCL-A.TO"]

df = pd.DataFrame({
    "open_b": open_b,
    "close_b": close_b,
    "vol_b": vol_b,
    "open_a": open_a,
    "close_a": close_a
}).dropna()

# The tradable return: buy at next open, sell at following open
df["next_oo_b"] = df["open_b"].shift(-2) / df["open_b"].shift(-1) - 1

# Signals known at today's close
df["cc_b"] = df["close_b"].pct_change()
df["spread_ba"] = np.log(df["close_b"]) - np.log(df["close_a"])
df["spread_z"] = (df["spread_ba"] - df["spread_ba"].rolling(60).mean()) / df["spread_ba"].rolling(60).std()

df["b_vs_5d_median"] = df["close_b"] / df["close_b"].rolling(5).median() - 1
df["low_volume"] = df["vol_b"] < df["vol_b"].rolling(60).median()

df = df.dropna()

def run_reg(signal):
    x = sm.add_constant(df[signal])
    y = df["next_oo_b"]
    model = sm.OLS(y, x).fit()
    print("\nSignal:", signal)
    print(model.summary())

for signal in ["cc_b", "spread_z", "b_vs_5d_median"]:
    run_reg(signal)

# Extreme spread test
df["spread_bucket"] = pd.qcut(df["spread_z"], 5, labels=False)

print("\nAverage next open-to-open return by TCL-B/TCL-A spread bucket:")
print(df.groupby("spread_bucket")["next_oo_b"].agg(["count", "mean", "median", "std"]))

# Large up/down close test
threshold = 0.05

print("\nAfter large TCL-B up closes:")
print(df.loc[df["cc_b"] > threshold, "next_oo_b"].agg(["count", "mean", "median", "std"]))

print("\nAfter large TCL-B down closes:")
print(df.loc[df["cc_b"] < -threshold, "next_oo_b"].agg(["count", "mean", "median", "std"]))

# Low-volume distorted close test
distorted = df["low_volume"] & (df["b_vs_5d_median"].abs() > 0.05)

print("\nAfter low-volume distorted closes:")
print(df.loc[distorted, "next_oo_b"].agg(["count", "mean", "median", "std"]))

# Directional hit rates
print("\nDirectional hit rates:")
print("After high spread_z > 1:", (df.loc[df["spread_z"] > 1, "next_oo_b"] < 0).mean())
print("After low spread_z < -1:", (df.loc[df["spread_z"] < -1, "next_oo_b"] > 0).mean())
