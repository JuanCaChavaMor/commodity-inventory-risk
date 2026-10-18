"""Download public daily prices to data/prices.csv (date, silver [USD/oz], copper [USD/t], fx [USD per EUR]).

    python scripts/fetch_data.py --start 2015-01-01                 # Yahoo Finance (needs: pip install yfinance)
    python scripts/fetch_data.py --source stooq --stooq-key KEY      # stooq.com (CSV download now needs an API key)

Sources: EUR/USD reference rate of the European Central Bank (data-api.ecb.europa.eu, free reuse with
attribution); silver and copper from COMEX futures (SI, USD/oz; HG, USD/lb, converted to USD/t) via Yahoo Finance,
or silver spot (XAG/USD) and HG futures from stooq.com. Run it on a machine with internet access; the course data
(finanzen.net) are not redistributed.
"""
import argparse
import io
import os
import sys
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
    s = df.set_index(pd.to_datetime(df.TIME_PERIOD)).OBS_VALUE.rename("fx")
    return s.dropna()


def yahoo(ticker, start):
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance is not installed: pip install yfinance  (or use --source stooq --stooq-key KEY)")
    df = yf.download(ticker, start=start, auto_adjust=False, progress=False)
    if df.empty:
        sys.exit(f"Yahoo Finance returned no data for {ticker}")
    s = df["Close"]
    if isinstance(s, pd.DataFrame):          # newer yfinance: one column per ticker
        s = s.iloc[:, 0]
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.dropna()


def stooq(symbol, start, key=None):
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d" + (f"&apikey={key}" if key else "")
    text = _get(url)
    df = pd.read_csv(io.StringIO(text))
    if "Date" not in df.columns or "Close" not in df.columns:
        sys.exit(f"stooq did not return a CSV for {symbol}; it answered:\n  {text[:300].strip()}\n"
                 "stooq now requires an API key for downloads (--stooq-key or STOOQ_APIKEY), "
                 "or use the default source: --source yahoo")
    s = df.set_index(pd.to_datetime(df.Date)).Close
    return s[s.index >= start]


ap = argparse.ArgumentParser()
ap.add_argument("--start", default="2015-01-01")
ap.add_argument("--source", choices=["yahoo", "stooq"], default="yahoo")
ap.add_argument("--stooq-key", default=os.environ.get("STOOQ_APIKEY"))
a = ap.parse_args()
if a.source == "yahoo":
    silver, copper_lb = yahoo("SI=F", a.start), yahoo("HG=F", a.start)
else:
    silver, copper_lb = stooq("xagusd", a.start, a.stooq_key), stooq("hg.f", a.start, a.stooq_key)
prices = pd.concat([silver.rename("silver"), (copper_lb * LB_PER_TONNE).rename("copper"), ecb_eurusd(a.start)],
                   axis=1, join="inner").dropna()
prices.index.name = "date"
out = ROOT / "data" / "prices.csv"
out.parent.mkdir(exist_ok=True)
prices.to_csv(out)
print(f"{len(prices)} days {prices.index[0].date()} .. {prices.index[-1].date()} -> {out}")
