"""Econometric analysis: returns, stationarity, ARIMA, GARCH(1,1).

Design notes
------------
* Log returns are computed from adjusted-agnostic closes (index level, so
  no adjustment question arises for ^GSPC).
* The out-of-sample window is the last 252 trading days (~1 year).
* The GARCH(1,1) is fitted with the `arch` package (mean = constant,
  distribution = normal). Returns are scaled by 100 for estimation
  (the `arch` optimiser is unstable on raw daily returns, whose scale is
  ~1e-2; unscaled fits produced a spurious mean of about -0.10 on one
  window). Reported mu and omega are converted back to raw-return units
  (mu / 100, omega / 10000); alpha, beta and persistence are
  scale-invariant.
* One-step-ahead volatility forecasts are produced on a rolling basis,
  refitting every day on an expanding window, so each forecast uses a
  model fitted only on data strictly before the forecast day.
* Baselines: (a) constant-volatility forecast = in-sample return
  standard deviation; (b) naive forecast = trailing 21-day realised
  volatility. Forecast accuracy is scored against two realised proxies:
  squared returns (for variance forecasts) and absolute returns (for
  volatility forecasts), plus a Mincer-Zarnowitz regression of realised
  variance on forecast variance.
"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

TRADING_DAYS_YEAR = 252
TEST_DAYS = 252
RV_WINDOW = 21


def log_returns(prices: pd.DataFrame) -> pd.Series:
    r = np.log(prices["Close"] / prices["Close"].shift(1)).dropna()
    r.name = "log_return"
    return r


def adf_report(series: pd.Series, label: str) -> dict:
    stat, pval, lags, nobs, crit, _ = adfuller(series.dropna(), autolag="AIC")
    return {
        "series": label,
        "adf_statistic": float(stat),
        "p_value": float(pval),
        "lags_used": int(lags),
        "n_obs": int(nobs),
        "critical_values": {k: float(v) for k, v in crit.items()},
    }


def fit_arima_grid(returns: pd.Series) -> dict:
    """Grid-search ARIMA(p, 0, q) on returns by AIC; returns are stationary."""
    best = None
    for p in range(0, 4):
        for q in range(0, 4):
            try:
                res = ARIMA(returns, order=(p, 0, q)).fit()
            except Exception:  # noqa: BLE001 - some orders fail to converge
                continue
            if best is None or res.aic < best["aic"]:
                best = {
                    "order": (p, 0, q),
                    "aic": float(res.aic),
                    "bic": float(res.bic),
                    "params": {k: float(v) for k, v in res.params.items()},
                    "sigma2": float(res.params["sigma2"]),
                }
    return best


def fit_garch(returns: pd.Series):
    """Fit GARCH(1,1) on returns x 100 (numerical stability; see notes)."""
    from arch import arch_model

    model = arch_model(
        returns * 100, mean="Constant", vol="Garch", p=1, q=1, dist="normal"
    )
    return model.fit(disp="off")


def garch_summary(fit) -> dict:
    params = fit.params
    return {
        # converted back to raw-return units
        "mu": float(params["mu"]) / 100,
        "omega": float(params["omega"]) / 10000,
        "alpha1": float(params["alpha[1]"]),
        "beta1": float(params["beta[1]"]),
        "persistence": float(params["alpha[1]"] + params["beta[1]"]),
        # likelihood criteria are on the x100 scale; see REPORT.md
        "loglikelihood_scaled": float(fit.loglikelihood),
        "aic_scaled": float(fit.aic),
        "bic_scaled": float(fit.bic),
        "n_obs": int(fit.nobs),
    }


def one_step_var(fit) -> float:
    """1-step-ahead variance forecast in raw-return units."""
    fc = fit.forecast(horizon=1, reindex=False)
    return float(fc.variance.iloc[0, 0]) / 10000


def rolling_garch_forecasts(returns: pd.Series, test_days: int = TEST_DAYS) -> pd.DataFrame:
    """True 1-step-ahead GARCH(1,1) variance forecasts over the test window.

    Refit daily on an expanding window. Each forecast uses a model fitted
    only on data strictly before the forecast day.
    """
    train = returns.iloc[:-test_days]
    test = returns.iloc[-test_days:]
    forecasts: list[float] = []
    for i in range(len(test)):
        window = pd.concat([train, test.iloc[:i]])
        fit = fit_garch(window)
        forecasts.append(one_step_var(fit))
    out = pd.DataFrame(
        {
            "return": test.values,
            "garch_var_forecast": forecasts,
        },
        index=test.index,
    )
    out["garch_vol_forecast"] = np.sqrt(out["garch_var_forecast"])
    out["realised_var_proxy"] = out["return"] ** 2
    out["realised_vol_proxy"] = out["return"].abs()
    return out


def naive_forecasts(returns: pd.Series, test_days: int = TEST_DAYS) -> pd.DataFrame:
    """Baselines: constant-vol and trailing-21-day realised vol."""
    train = returns.iloc[:-test_days]
    test = returns.iloc[-test_days:]
    const_var = float(train.var())
    rows = []
    for i, (date, r) in enumerate(test.items()):
        history = pd.concat([train, test.iloc[:i]])
        trail = history.iloc[-RV_WINDOW:]
        rows.append(
            {
                "return": r,
                "const_var_forecast": const_var,
                "const_vol_forecast": float(np.sqrt(const_var)),
                "naive_var_forecast": float(trail.var()),
                "naive_vol_forecast": float(trail.std()),
            }
        )
    out = pd.DataFrame(rows, index=test.index)
    out["realised_var_proxy"] = out["return"] ** 2
    out["realised_vol_proxy"] = out["return"].abs()
    return out


def score_forecasts(df: pd.DataFrame, var_col: str, vol_col: str) -> dict:
    err_var = df[var_col] - df["realised_var_proxy"]
    err_vol = df[vol_col] - df["realised_vol_proxy"]
    return {
        "rmse_var": float(np.sqrt(np.mean(err_var**2))),
        "mae_var": float(np.mean(np.abs(err_var))),
        "rmse_vol": float(np.sqrt(np.mean(err_vol**2))),
        "mae_vol": float(np.mean(np.abs(err_vol))),
    }


def mincer_zarnowitz(df: pd.DataFrame, var_col: str) -> dict:
    """Regress realised variance on forecast variance (OLS, with intercept)."""
    import statsmodels.api as sm

    x = sm.add_constant(df[var_col].values, has_constant="add")
    res = sm.OLS(df["realised_var_proxy"].values, x).fit()
    return {
        "intercept": float(res.params[0]),
        "slope": float(res.params[1]),
        "r_squared": float(res.rsquared),
        "intercept_p": float(res.pvalues[0]),
        "slope_p": float(res.pvalues[1]),
    }


def diagnostics(returns: pd.Series, fit) -> dict:
    std_resid = fit.std_resid.dropna()
    lb = acorr_ljungbox(std_resid, lags=[10], return_df=True)
    lb2 = acorr_ljungbox(std_resid**2, lags=[10], return_df=True)
    lm_stat, lm_p, _, _ = het_arch(std_resid, nlags=10)
    jb_stat, jb_p, _, _ = jarque_bera(std_resid)
    return {
        "ljung_box_std_resid_lag10_stat": float(lb["lb_stat"].iloc[0]),
        "ljung_box_std_resid_lag10_p": float(lb["lb_pvalue"].iloc[0]),
        "ljung_box_sq_std_resid_lag10_stat": float(lb2["lb_stat"].iloc[0]),
        "ljung_box_sq_std_resid_lag10_p": float(lb2["lb_pvalue"].iloc[0]),
        "arch_lm_stat": float(lm_stat),
        "arch_lm_p": float(lm_p),
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_p": float(jb_p),
        "std_resid_skew": float(std_resid.skew()),
        "std_resid_kurtosis": float(std_resid.kurtosis()),
        "returns_skew": float(returns.skew()),
        "returns_kurtosis": float(returns.kurtosis()),
    }
