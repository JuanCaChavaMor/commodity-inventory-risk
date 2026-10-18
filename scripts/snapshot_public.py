"""Snapshot of a backtest on public data: the summary tables and a figure, without the prices themselves.

    python scripts/fetch_data.py --start 2015-01-01
    python scripts/run_var.py data/prices.csv
    python scripts/run_backtest.py data/prices.csv --window 250
    python scripts/snapshot_public.py        # -> results/public_snapshot/, figures/backtest_public.(png|pdf)

results/public/ (with the daily series) stays local; the snapshot keeps only statistics, which the report quotes.
"""
import json
import shutil

import numpy as np
import pandas as pd

from _path import ROOT
from invrisk.plotting import C, INK, plt, save

SRC, DST, FIG = ROOT / "results" / "public", ROOT / "results" / "public_snapshot", ROOT / "figures"
DST.mkdir(exist_ok=True)
for f in ("backtest.csv", "backtest_power.csv", "summary.json"):
    shutil.copy(SRC / f, DST / f)

# cumulative exceptions minus their expected number: a correct VaR wanders around zero, a VaR that is too low drifts
# upwards, and clustered exceptions show up as steps
s = pd.read_csv(SRC / "backtest_series.csv")
s.columns = ["key", "date", "VaR", "loss", "exception"]
s["date"] = pd.to_datetime(s.date)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharex=True)
for ax, alpha in zip(axes, (0.95, 0.99)):
    for k, m in enumerate(["parametric (delta-normal)", "historical simulation", "Monte Carlo, normal",
                           "filtered HS (EWMA)"]):
        g = s[s.key == f"{m}|{alpha}"].sort_values("date")
        excess = np.cumsum(g.exception.astype(float)) - (1 - alpha) * np.arange(1, len(g) + 1)
        ax.plot(g.date, excess, color=C[k], lw=1.2, label=m)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_title(f"VaR {int(100 * alpha)}%", loc="left", fontsize=9)
    ax.set_ylabel("exceptions minus expected")
axes[0].legend(fontsize=7)
save(fig, FIG / "backtest_public.png")
meta = json.loads((DST / "summary.json").read_text())
print("snapshot", meta["start"], "..", meta["end"], "->", DST)
