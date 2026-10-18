"""Value at Risk and Expected Shortfall of the inventory, four ways.

All functions take the factor log returns R (DataFrame, columns = commodities + ['fx'], one row per day), the
inventory and the current prices, and return a dict with the loss distribution summary in EUR (losses positive):
    VaR_a = a-quantile of the loss,   ES_a = mean loss beyond VaR_a.
Horizons above one day: parametric and Monte Carlo methods scale the covariance with h (square-root-of-time);
historical methods use overlapping h-day returns.
"""
import numpy as np
import pandas as pd
from scipy import stats

from .data import log_returns


def _summary(loss, alpha):
    loss = np.sort(np.asarray(loss).ravel())
    var = np.quantile(loss, alpha, method="higher")
    return dict(VaR=var, ES=loss[loss >= var].mean())


def _cols(inv):
    return inv.names + ["fx"]


# ---------------------------------------------------------------- 1. parametric (delta-normal)
def parametric(R, inv, prices, alpha=0.95, h=1):
    """Loss = -delta . dx with dx ~ N(0, h * Sigma); zero mean (standard for short horizons)."""
    d = inv.delta(prices)
    S = R[_cols(inv)].cov().values * h
    sd = np.sqrt(d @ S @ d)
    z = stats.norm.ppf(alpha)
    return dict(VaR=z * sd, ES=sd * stats.norm.pdf(z) / (1 - alpha), sd=sd)


def contributions(R, inv, prices, alpha=0.95, h=1):
    """Euler allocation of the parametric VaR to the positions (each position carries its own FX exposure):
    contribution_i = z * cov(L_i, L) / sd(L); they add up to the VaR."""
    v = inv.values_eur(prices)
    X = R[_cols(inv)].values
    Li = -(v * (X[:, :-1] - X[:, [-1]]))                     # daily delta P&L per position, sign: loss
    C = np.cov(Li, rowvar=False) * h
    sd = np.sqrt(C.sum())
    z = stats.norm.ppf(alpha)
    stand_alone = z * np.sqrt(np.diag(C))
    return pd.DataFrame({"value_EUR": v, "stand_alone_VaR": stand_alone, "contribution": z * C.sum(1) / sd},
                        index=inv.names)


# ---------------------------------------------------------------- 2. historical simulation
def historical(prices_hist, inv, prices, alpha=0.95, h=1):
    """Full revaluation of today's book under every observed (overlapping) h-day factor change."""
    dx = log_returns(prices_hist[_cols(inv)], h).values
    return _summary(-inv.pnl(prices, dx).sum(1), alpha)


# ---------------------------------------------------------------- 3. Monte Carlo
def monte_carlo(R, inv, prices, alpha=0.95, h=1, n=200_000, dist="normal", df=None, seed=0):
    """Correlated factor changes with the sample covariance (Cholesky), full revaluation.

    dist="t": multivariate Student t with `df` degrees of freedom (fitted to the portfolio's daily delta P&L by
    maximum likelihood if None), scaled to the same covariance: fat tails at equal variance."""
    rng = np.random.default_rng(seed)
    S = R[_cols(inv)].cov().values * h
    L = np.linalg.cholesky(S)
    Z = rng.standard_normal((n, S.shape[0]))
    if dist == "t":
        if df is None:
            df = fit_t_df(R, inv, prices)
        w = rng.chisquare(df, size=(n, 1)) / df
        Z = Z / np.sqrt(w) * np.sqrt((df - 2) / df)          # unit variance
    out = _summary(-inv.pnl(prices, Z @ L.T).sum(1), alpha)
    if dist == "t":
        out["df"] = df
    return out


def fit_t_df(R, inv, prices):
    """Degrees of freedom of a Student t fitted to the daily delta P&L of the book (location and scale free)."""
    pnl = R[_cols(inv)].values @ inv.delta(prices)
    df, _, _ = stats.t.fit(pnl)
    return float(np.clip(df, 2.5, 100))


# ---------------------------------------------------------------- 4. filtered historical simulation (EWMA)
def ewma_vol(R, lam=0.94):
    """RiskMetrics EWMA volatility of each column; row t uses returns up to t-1, the last row is the forecast."""
    var = np.empty((len(R) + 1, R.shape[1]))
    var[0] = R.var().values
    for t, r in enumerate(R.values):
        var[t + 1] = lam * var[t] + (1 - lam) * r ** 2
    return pd.DataFrame(np.sqrt(var[1:]), index=R.index, columns=R.columns), np.sqrt(var[-1])


def filtered_historical(R, inv, prices, alpha=0.95, h=1, lam=0.94):
    """Hull-White: devolatilise each factor's returns by their EWMA volatility, rescale by today's forecast.
    Volatility clustering is taken into account while the empirical joint shape of the shocks is kept."""
    cols = _cols(inv)
    vol, now = ewma_vol(R[cols], lam)
    prev = vol.shift(1).fillna(R[cols].std())                  # volatility known before each return
    dx = (R[cols] / prev).values * now * np.sqrt(h)
    return _summary(-inv.pnl(prices, dx).sum(1), alpha)


METHODS = {
    "parametric (delta-normal)": lambda R, P, inv, p, a, h: parametric(R, inv, p, a, h),
    "historical simulation": lambda R, P, inv, p, a, h: historical(P, inv, p, a, h),
    "Monte Carlo, normal": lambda R, P, inv, p, a, h: monte_carlo(R, inv, p, a, h),
    "Monte Carlo, Student t": lambda R, P, inv, p, a, h: monte_carlo(R, inv, p, a, h, dist="t"),
    "filtered HS (EWMA)": lambda R, P, inv, p, a, h: filtered_historical(R, inv, p, a, h),
}
