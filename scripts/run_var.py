"""VaR and ES of the inventory (Tables 1-3 of the report).

    python scripts/run_var.py [prices.csv]      # default: the course data in data/raw (see README)

1. the course exercise: 1-day VaR(95%) of the EUR value of 500 t of silver, with and without the silver-FX
   correlation;
2. the book (silver + copper): five methods x confidence 95 / 99 % x horizon 1 / 10 days, and the Euler
   contributions of the positions.
"""
import json
import sys

import numpy as np
import pandas as pd

from _path import ROOT
from invrisk.data import load_course_data, load_csv, log_returns
from invrisk.measures import METHODS, contributions, fit_t_df, parametric
from invrisk.portfolio import Inventory

P = load_csv(sys.argv[1]) if len(sys.argv) > 1 else load_course_data()
R = log_returns(P)
now = P.iloc[-1]
out = ROOT / "results"
out.mkdir(exist_ok=True)

# ---------------------------------------------------------------- 1. the course exercise
silver = Inventory({"silver": Inventory.course_case().quantities["silver"]})
Rs = R[["silver", "fx"]]
base = parametric(Rs, silver, now)
# exact: zero correlation with the same variances
d = silver.delta(now)
sd0 = np.sqrt(d ** 2 @ Rs.var().values)
exercise = dict(value_EUR=float(silver.values_eur(now)[0]), VaR95_1d=base["VaR"], ES95_1d=base["ES"],
                VaR95_1d_without_correlation=1.6448536 * sd0, correlation_silver_fx=Rs.corr().iloc[0, 1],
                silver_USD_only_VaR95_1d=1.6448536 * silver.values_eur(now)[0] * Rs.silver.std())
print(json.dumps(exercise, indent=2))

# ---------------------------------------------------------------- 2. the book
inv = Inventory.course_case()
rows = []
for alpha in (0.95, 0.99):
    for h in (1, 10):
        for name, f in METHODS.items():
            r = f(R, P, inv, now, alpha, h)
            rows.append(dict(method=name, alpha=alpha, horizon=h, VaR=r["VaR"], ES=r["ES"]))
table = pd.DataFrame(rows)
table.round({"VaR": 0, "ES": 0}).to_csv(out / "var_table.csv", index=False)
contrib = contributions(R, inv, now)
contrib.round(0).to_csv(out / "contributions.csv")
stats = dict(n_days=len(R), start=str(P.index[0].date()), end=str(P.index[-1].date()),
             value_EUR=dict(zip(inv.names, inv.values_eur(now).tolist())),
             vol_annual=(R.std() * np.sqrt(252)).to_dict(), corr=R.corr().round(3).to_dict(),
             excess_kurtosis=R.kurt().to_dict(), t_df=fit_t_df(R, inv, now), exercise=exercise)
(out / "summary.json").write_text(json.dumps(stats, indent=2, default=float))
print(table.assign(VaR=table.VaR / 1e6, ES=table.ES / 1e6).round(2).to_string(index=False))
print((contrib / 1e6).round(2))
