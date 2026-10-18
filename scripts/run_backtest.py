"""Rolling out-of-sample backtest of the 1-day VaR (Table 4 of the report).

    python scripts/run_backtest.py [prices.csv] [--window 125]
"""
import argparse

import pandas as pd

from _path import ROOT
from invrisk import backtest as bt
from invrisk.data import load_course_data, load_csv
from invrisk.measures import METHODS
from invrisk.portfolio import Inventory

ap = argparse.ArgumentParser()
ap.add_argument("csv", nargs="?")
ap.add_argument("--window", type=int, default=125)
a = ap.parse_args()
OUT = ROOT / "results" / ("public" if a.csv else "")   # public data never overwrite the course results
OUT.mkdir(parents=True, exist_ok=True)
P = load_csv(a.csv) if a.csv else load_course_data()
inv = Inventory.course_case()
rows, series = [], {}
for alpha in (0.95, 0.99):
    for name, f in METHODS.items():
        if name == "Monte Carlo, Student t":
            continue                        # slow in a rolling loop and close to the normal Monte Carlo here
        if name == "Monte Carlo, normal":
            f = (lambda R, P_, inv_, p, al, h: __import__("invrisk.measures", fromlist=["x"]).monte_carlo(
                R, inv_, p, al, h, n=20_000))
        res = bt.rolling(P, inv, f, alpha, a.window)
        series[(name, alpha)] = res
        rows.append(dict(method=name, alpha=alpha, **bt.summarize(res, alpha)))
        print(rows[-1], flush=True)
out = pd.DataFrame(rows)
n = int(out.days.iloc[0])
power = pd.DataFrame([dict(alpha=al, true_rate=k * (1 - al), power=bt.kupiec_power(n, al, k * (1 - al)))
                      for al in (0.95, 0.99) for k in (1.5, 2, 3)])
power.to_csv(OUT / "backtest_power.csv", index=False, float_format="%.3f")
print(power.round(3).to_string(index=False))
out.to_csv(OUT / "backtest.csv", index=False, float_format="%.4f")
pd.concat({f"{k[0]}|{k[1]}": v for k, v in series.items()}).to_csv(OUT / "backtest_series.csv",
                                                                     float_format="%.0f")
