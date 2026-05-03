import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

ticker = "TCL-B.TO"
start = "2026-03-23"
end = "2026-04-06"

df = yf.download(ticker, start=start, end=end, auto_adjust=True)

prices = df["Close"].dropna()
log_ret = np.log(prices).diff().dropna()

print("Number of price observations:", len(prices))
print("Number of return observations:", len(log_ret))

adf_price = adfuller(prices)
adf_ret = adfuller(log_ret)

print("ADF p-value, prices:    ", adf_price[1])
print("ADF p-value, log return:", adf_ret[1])

max_pacf_lag = max(1, min(20, len(log_ret) // 2 - 1))
max_acf_lag = max(1, min(20, len(log_ret) - 1))

print("Using ACF lags:", max_acf_lag)
print("Using PACF lags:", max_pacf_lag)

acf_vals = acf(log_ret, nlags=max_acf_lag, fft=True)
pacf_vals = pacf(log_ret, nlags=max_pacf_lag, method="ywadjusted")

plot_acf(log_ret, lags=max_acf_lag)
plt.title(f"{ticker} log returns ACF")
plt.savefig("acf.png")
plt.close()

plot_pacf(log_ret, lags=max_pacf_lag, method="ywadjusted")
plt.title(f"{ticker} log returns PACF")
plt.savefig("pacf.png")
plt.close()
