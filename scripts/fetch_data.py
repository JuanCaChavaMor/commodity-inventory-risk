"""Download public daily prices to data/prices.csv (date, silver [USD/oz], copper [USD/t], fx [USD per EUR]).

    python scripts/fetch_data.py --start 2015-01-01

Sources: EUR/USD reference rate of the European Central Bank (data-api.ecb.europa.eu, free reuse with
attribution); silver spot (XAG/USD) and COMEX copper futures (HG, USD/lb, converted to USD/t) from stooq.com.
Run it on a machine with internet access; the course data (finanzen.net) are not redistributed.
"""
import argparse
import io
import urllib.request

import pandas as pd

from _path import ROOT

LB_PER_TONNE = 2204.62262


def _get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as r:
        return r.read().decode()


def ecb_eurusd(start):
    url = ("https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A"
           f"?format=csvdata&startPeriod={start}")
    df = pd.read_csv(io.StringIO(_get(url)))
    return df.set_index(pd.to_datetime(df.TIME_PERIOD)).OBS_VALUE.rename("fx")


def stooq(symbol, start):
    df = pd.read_csv(io.StringIO(_get(f"https://stooq.com/q/d/l/?s={symbol}&i=d")))
    s = df.set_index(pd.to_datetime(df.Date)).Close
    return s[s.index >= start]


ap = argparse.ArgumentParser()
ap.add_argument("--start", default="2015-01-01")
a = ap.parse_args()
prices = pd.concat([stooq("xagusd", a.start).rename("silver"),
                    (stooq("hg.f", a.start) * LB_PER_TONNE).rename("copper"),
                    ecb_eurusd(a.start)], axis=1, join="inner")
prices.index.name = "date"
out = ROOT / "data" / "prices.csv"
prices.to_csv(out)
print(f"{len(prices)} days {prices.index[0].date()} .. {prices.index[-1].date()} -> {out}")
