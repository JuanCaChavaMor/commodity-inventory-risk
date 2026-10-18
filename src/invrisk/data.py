"""Price data: the course exports (finanzen.net, Nov 2022 - Nov 2023) or the public series of scripts/fetch_data.py.

Convention: `fx` is the EUR/USD rate quoted as USD per EUR (1.07 = one euro costs 1.07 dollars), so the euro value
of a quantity q priced at S USD per unit is q * S / fx.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


def read_finanzen(path, sheet):
    """One sheet of a finanzen.net export: two title rows, a header row, then date / close / ... (newest first)."""
    v = pd.read_excel(path, sheet_name=sheet, header=None).iloc[3:, :2]
    v.columns = ["date", "close"]
    v = v.dropna(subset=["date"])
    v["date"] = pd.to_datetime(v["date"]).dt.normalize()
    return v.set_index("date").close.astype(float).sort_index()


def load_course_data(raw=RAW):
    """Silver (USD/oz), copper (USD/t) and EUR/USD closes on the trading days common to all three.

    The silver export also contains weekend quotes and the FX export a few; aligning on dates keeps weekdays
    on which all three markets closed."""
    ag = read_finanzen(Path(raw) / "MC_Simulation.xlsx", "Silber_USD")
    fx = read_finanzen(Path(raw) / "MC_Simulation.xlsx", "USD_EUR")
    cu = read_finanzen(Path(raw) / "Historische Simulation - Kupfer USD.xlsx", "Kupfer USD")
    df = pd.concat([ag.rename("silver"), cu.rename("copper"), fx.rename("fx")], axis=1, join="inner")
    return df[df.index.dayofweek < 5]


def load_csv(path):
    """Prices saved by scripts/fetch_data.py (columns date, silver, copper, fx)."""
    return pd.read_csv(path, parse_dates=["date"], index_col="date").dropna()


def log_returns(prices, horizon=1):
    """Log returns over `horizon` trading days (overlapping for horizon > 1)."""
    return np.log(prices).diff(horizon).dropna()
