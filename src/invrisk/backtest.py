"""Backtesting: does the 1-day VaR forecast hold out of sample?

Rolling window of `window` days: each day t the VaR is estimated from the previous window and compared with the
realised full-revaluation loss of the (fixed) inventory from t-1 to t. Exceptions are days with loss > VaR.
Kupiec (1995) tests the number of exceptions (proportion of failures), Christoffersen (1998) their independence.
"""
import numpy as np
import pandas as pd
from scipy import stats

from .data import log_returns


def kupiec(n_exc, n, alpha):
    p = 1 - alpha
    x = n_exc
    phat = x / n
    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll1 = (n - x) * np.log(1 - phat) + x * np.log(phat) if 0 < x < n else 0.0
    lr = -2 * (ll0 - ll1)
    return lr, 1 - stats.chi2.cdf(lr, 1)


def christoffersen(exc):
    """Independence test of the exception sequence (first-order Markov alternative)."""
    e = np.asarray(exc, int)
    a, b = e[:-1], e[1:]
    n00, n01 = np.sum((a == 0) & (b == 0)), np.sum((a == 0) & (b == 1))
    n10, n11 = np.sum((a == 1) & (b == 0)), np.sum((a == 1) & (b == 1))
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / max(len(a), 1)

    def ll(k, q):
        return k * np.log(q) if k > 0 and q > 0 else 0.0
    l0 = ll(n00 + n10, 1 - p) + ll(n01 + n11, p)
    l1 = ll(n00, 1 - p01) + ll(n01, p01) + ll(n10, 1 - p11) + ll(n11, p11)
    lr = max(-2 * (l0 - l1), 0.0)
    return lr, 1 - stats.chi2.cdf(lr, 1)


def rolling(prices, inv, method, alpha=0.95, window=125):
    """method(R_window, P_window, inv, prices_today, alpha, h=1) -> dict with 'VaR'. Returns a DataFrame."""
    R = log_returns(prices)
    rows = []
    for t in range(window, len(R)):
        Rw = R.iloc[t - window:t]
        Pw = prices.loc[Rw.index[0] - pd.Timedelta(days=7):Rw.index[-1]].iloc[-(window + 1):]
        p_now = prices.loc[Rw.index[-1]]                       # book valued at yesterday's close
        var = method(Rw, Pw, inv, p_now, alpha, 1)["VaR"]
        loss = -inv.pnl(p_now, R.iloc[[t]][inv.names + ["fx"]].values).sum()
        rows.append(dict(date=R.index[t], VaR=var, loss=loss))
    df = pd.DataFrame(rows).set_index("date")
    df["exception"] = df.loss > df.VaR
    return df


def summarize(bt, alpha):
    n, x = len(bt), int(bt.exception.sum())
    lr_k, p_k = kupiec(x, n, alpha)
    lr_c, p_c = christoffersen(bt.exception.values)
    return dict(days=n, expected=(1 - alpha) * n, exceptions=x, kupiec_p=p_k, christoffersen_p=p_c)


def kupiec_power(n, alpha, true_rate, level=0.05):
    """Probability that the Kupiec test (at `level`) rejects a VaR model whose true exception rate is
    `true_rate`, with n backtest days: how much a passed backtest is worth."""
    return float(sum(stats.binom.pmf(x, n, true_rate) for x in range(n + 1) if kupiec(x, n, alpha)[1] < level))
