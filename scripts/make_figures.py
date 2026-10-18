"""Figures of the report and the README.

    python scripts/make_figures.py      # after run_var.py and run_backtest.py
"""
import numpy as np
import pandas as pd

from _path import ROOT
from invrisk.data import load_course_data, log_returns
from invrisk.plotting import C, INK, MUTED, plt, save
from invrisk.portfolio import Inventory

FIG, RES = ROOT / "figures", ROOT / "results"
P = load_course_data()
R = log_returns(P)
inv = Inventory.course_case()
now = P.iloc[-1]
M = 1e6

# ---------------------------------------------------------------- 1. prices and the natural hedge
fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
for ax, (c, lab) in zip(axes, [("silver", "silver [USD/oz]"), ("copper", "copper [USD/t]"), ("fx", "EUR/USD")]):
    ax.plot(P.index, P[c], color=C[["silver", "copper", "fx"].index(c)], lw=1.2)
    ax.set_title(lab, loc="left", fontsize=9, color=INK)
    ax.tick_params(axis="x", labelrotation=30, labelsize=7)
save(fig, FIG / "prices.png")

# ---------------------------------------------------------------- 2. loss distribution and the measures
hist_loss = -inv.pnl(now, R[inv.names + ["fx"]].values).sum(1) / M
t = pd.read_csv(RES / "var_table.csv")
t1 = t[(t.horizon == 1) & (t.alpha == 0.95)].set_index("method")
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.hist(hist_loss, bins=40, color=MUTED, alpha=0.5, label="historical daily losses (today's book)")
for k, (name, r) in enumerate(t1.iterrows()):
    ax.axvline(r.VaR / M, color=C[k], lw=1.4, label=f"VaR95 {name}: {r.VaR / M:.1f}")
ax.set_xlabel("1-day loss [EUR m]  (gains negative)"); ax.set_ylabel("days")
ax.legend(fontsize=7, loc="upper left")
save(fig, FIG / "loss_distribution.png")

# ---------------------------------------------------------------- 3. backtest
s = pd.read_csv(RES / "backtest_series.csv")
s.columns = ["key", "date", "VaR", "loss", "exception"]
s["date"] = pd.to_datetime(s.date)
fig, ax = plt.subplots(figsize=(8, 3.4))
first = s[s.key.str.startswith("parametric") & s.key.str.endswith("|0.95")]
ax.bar(first.date, first.loss / M, color=np.where(first.loss > first.VaR, C[1], MUTED), width=1.0,
       label="realised loss (exceptions in orange)")
for k, (key, lab) in enumerate([("parametric (delta-normal)|0.95", "parametric 95%"),
                                ("filtered HS (EWMA)|0.95", "filtered HS 95%"),
                                ("parametric (delta-normal)|0.99", "parametric 99%")]):
    g = s[s.key == key]
    ax.plot(g.date, g.VaR / M, color=[C[0], C[2], INK][k], lw=1.3, label=lab)
ax.axhline(0, color=INK, lw=0.6)
ax.set_ylabel("EUR m"); ax.legend(fontsize=7, ncol=2, loc="lower left")
save(fig, FIG / "backtest.png")

# ---------------------------------------------------------------- 4. diversification
c = pd.read_csv(RES / "contributions.csv", index_col=0)
fig, ax = plt.subplots(figsize=(5.2, 3.0))
ax.bar([0, 1], c.stand_alone_VaR / M, color=MUTED, width=0.35, label="stand-alone VaR")
ax.bar([0.38, 1.38], c.contribution / M, color=C[0], width=0.35, label="contribution to book VaR")
ax.set_xticks([0.19, 1.19], c.index)
ax.set_ylabel("1-day VaR95 [EUR m]")
ax.set_title(f"sum stand-alone {c.stand_alone_VaR.sum() / M:.1f}  vs  book {c.contribution.sum() / M:.1f}",
             loc="left", fontsize=8, color=MUTED)
ax.legend(fontsize=8); ax.grid(axis="x", visible=False)
save(fig, FIG / "diversification.png")
print("figures written to", FIG)
