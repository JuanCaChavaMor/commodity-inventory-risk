import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from invrisk import backtest as bt  # noqa: E402
from invrisk.data import log_returns  # noqa: E402
from invrisk.measures import contributions, filtered_historical, historical, monte_carlo, parametric  # noqa: E402
from invrisk.portfolio import TROY_OUNCES_PER_TONNE, Inventory  # noqa: E402

SIG = np.array([0.017, 0.012, 0.005])          # daily vols: silver, copper, fx
RHO = np.array([[1, 0.2, 0.5], [0.2, 1, 0.3], [0.5, 0.3, 1]])


def synthetic_prices(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    cov = RHO * np.outer(SIG, SIG)
    dx = rng.multivariate_normal(np.zeros(3), cov, size=n)
    x = np.log([22.0, 8000.0, 1.07]) + np.cumsum(dx, axis=0)
    return pd.DataFrame(np.exp(x), columns=["silver", "copper", "fx"],
                        index=pd.bdate_range("2010-01-01", periods=n))


@pytest.fixture(scope="module")
def case():
    P = synthetic_prices()
    return P, log_returns(P), Inventory.course_case(), P.iloc[-1]


def test_units_and_values():
    assert TROY_OUNCES_PER_TONNE == pytest.approx(32150.75, rel=1e-6)
    inv = Inventory({"silver": 100.0})
    p = pd.Series({"silver": 20.0, "fx": 1.25})
    assert inv.values_eur(p)[0] == pytest.approx(1600.0)
    # full revaluation: silver +10 % in USD and euro +10 % against the dollar leave the euro value unchanged
    assert inv.pnl(p, [[np.log(1.1), np.log(1.1)]])[0, 0] == pytest.approx(0.0, abs=1e-9)


def test_parametric_matches_closed_form(case):
    P, R, inv, now = case
    v = inv.values_eur(now)
    # loss = -(v1 (x1 - x3) + v2 (x2 - x3)); variance from the true covariance
    a = np.array([v[0], v[1], -v.sum()])
    sd_true = np.sqrt(a @ (RHO * np.outer(SIG, SIG)) @ a)
    r = parametric(R, inv, now, 0.99)
    assert r["VaR"] == pytest.approx(stats.norm.ppf(0.99) * sd_true, rel=0.05)
    assert r["ES"] > r["VaR"]


def test_methods_agree_on_gaussian_data(case):
    P, R, inv, now = case
    ref = parametric(R, inv, now, 0.95)["VaR"]
    for r in (historical(P, inv, now, 0.95), monte_carlo(R, inv, now, 0.95, n=100_000),
              filtered_historical(R, inv, now, 0.95)):
        assert r["VaR"] == pytest.approx(ref, rel=0.08)


def test_euler_contributions_add_up(case):
    P, R, inv, now = case
    c = contributions(R, inv, now)
    assert c.contribution.sum() == pytest.approx(parametric(R, inv, now)["VaR"], rel=1e-9)
    assert c.contribution.sum() < c.stand_alone_VaR.sum()


def test_student_t_has_fatter_tail(case):
    P, R, inv, now = case
    n = monte_carlo(R, inv, now, 0.99, n=200_000)
    t = monte_carlo(R, inv, now, 0.99, n=200_000, dist="t", df=4)
    assert t["ES"] > n["ES"]


def test_kupiec_and_christoffersen():
    assert bt.kupiec(5, 100, 0.95)[1] == pytest.approx(1.0)
    assert bt.kupiec(15, 100, 0.95)[1] < 0.01
    clustered = np.r_[np.zeros(90), np.ones(10)]
    spread = np.zeros(100); spread[::10] = 1
    assert bt.christoffersen(clustered)[1] < bt.christoffersen(spread)[1]
    assert bt.kupiec_power(250, 0.99, 0.01) < 0.1 < bt.kupiec_power(250, 0.99, 0.04)
