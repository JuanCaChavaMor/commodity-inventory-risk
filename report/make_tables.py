"""LaTeX tables of the report from results/.

    python report/make_tables.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", Path(__file__).resolve().parent / "tables"
OUT.mkdir(exist_ok=True)
M = 1e6


def write(name, body):
    (OUT / f"{name}.tex").write_text(body)
    print("wrote", name)


def tab(spec, header, rows):
    return f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header} \\\\\n\\midrule\n" + "\n".join(rows) + \
        "\n\\bottomrule\n\\end{tabular}\n"


t = pd.read_csv(RES / "var_table.csv")
rows = []
for m, g in t.groupby("method", sort=False):
    g = g.set_index(["alpha", "horizon"])
    cells = [f"{g.loc[(a, h), 'VaR'] / M:.1f} & {g.loc[(a, h), 'ES'] / M:.1f}"
             for a in (0.95, 0.99) for h in (1, 10)]
    rows.append(f"{m} & " + " & ".join(cells) + " \\\\")
write("var", tab("l" + "rr" * 4,
                 r"& \multicolumn{4}{c}{95\%} & \multicolumn{4}{c}{99\%} \\ \cmidrule(lr){2-5}\cmidrule(lr){6-9}"
                 r"& \multicolumn{2}{c}{1 day} & \multicolumn{2}{c}{10 days} & \multicolumn{2}{c}{1 day} & "
                 r"\multicolumn{2}{c}{10 days} \\" "\nmethod & VaR & ES & VaR & ES & VaR & ES & VaR & ES", rows))

b = pd.read_csv(RES / "backtest.csv")
rows = [f"{r.method} & {int(100 * r.alpha)}\\% & {r.expected:.1f} & {r.exceptions} & {r.kupiec_p:.2f} & "
        f"{r.christoffersen_p:.2f} \\\\" for r in b.itertuples()]
write("backtest", tab("lrrrrr", "method & level & expected & exceptions & Kupiec $p$ & Christoffersen $p$", rows))

pw = pd.read_csv(RES / "backtest_power.csv")
rows = [f"{int(100 * r.alpha)}\\% & {100 * r.true_rate:.1f}\\% & {100 * r.power:.0f}\\% \\\\" for r in pw.itertuples()]
write("power", tab("rrr", "VaR level & true exception rate & probability of rejection", rows))

c = pd.read_csv(RES / "contributions.csv", index_col=0)
rows = [f"{k} & {r.value_EUR / M:.0f} & {r.stand_alone_VaR / M:.2f} & {r.contribution / M:.2f} \\\\"
        for k, r in c.iterrows()]
rows.append(f"\\midrule book & {c.value_EUR.sum() / M:.0f} & {c.stand_alone_VaR.sum() / M:.2f} & "
            f"{c.contribution.sum() / M:.2f} \\\\")
write("contributions", tab("lrrr", r"position & value [EUR m] & stand-alone VaR & contribution", rows))

# backtest on public data (snapshot written by scripts/snapshot_public.py)
SNAP = RES / "public_snapshot"
if (SNAP / "backtest.csv").exists():
    b = pd.read_csv(SNAP / "backtest.csv")
    rows = [f"{r.method} & {int(100 * r.alpha)}\\% & {r.expected:.1f} & {r.exceptions} & {r.kupiec_p:.3f} & "
            f"{r.christoffersen_p:.3f} \\\\" for r in b.itertuples()]
    write("backtest_public", tab("lrrrrr", "method & level & expected & exceptions & Kupiec $p$ & "
                                 "Christoffersen $p$", rows))
    pw = pd.read_csv(SNAP / "backtest_power.csv")
    n = int(b.days.iloc[0])
    old = pd.read_csv(RES / "backtest_power.csv")
    n_old = int(pd.read_csv(RES / "backtest.csv").days.iloc[0])
    rows = [f"{int(100 * r.alpha)}\\% & {100 * r.true_rate:.1f}\\% & {100 * o.power:.0f}\\% & {100 * r.power:.0f}\\% \\\\"
            for r, o in zip(pw.itertuples(), old.itertuples())]
    write("power_public", tab("rrrr", f"VaR level & true exception rate & {n_old} days & {n} days", rows))
