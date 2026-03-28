"""
05_download_realized.py -- Download realized macroeconomic series.

Fetches HICP inflation and real GDP growth from the ECB Statistical Data
Warehouse (SDW) and Eurostat APIs.

Author: Eric Vansteenberghe
Reference:
    Vansteenberghe, E. (2026). "Uncertain and Asymmetric Forecasts."
    Banque de France Working Paper.

Output:
    data/realized/inflation.csv -- monthly HICP YoY inflation (%)
    data/realized/gdp.csv       -- quarterly real GDP YoY growth (%)
"""
import sys
import os
import io
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import REALIZED_DIR


ECB_API_BASE = "https://data-api.ecb.europa.eu/service/data"


def _download_ecb_series(flow, key, start="1990-01"):
    """Download a series from ECB SDW as CSV, return DataFrame."""
    url = f"{ECB_API_BASE}/{flow}/{key}"
    params = {
        "format": "csvdata",
        "startPeriod": start,
    }
    print(f"  Fetching ECB: {flow}/{key} ...")
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    return df


def download_inflation():
    """
    HICP — Year-on-year rate of change, Euro Area.
    ECB series: ICP.M.U2.N.000000.4.ANR
    """
    REALIZED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REALIZED_DIR / "inflation.csv"

    df = _download_ecb_series("ICP", "M.U2.N.000000.4.ANR")

    # The CSV has TIME_PERIOD and OBS_VALUE columns
    df = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
    df.rename(columns={"TIME_PERIOD": "period", "OBS_VALUE": "inflation"}, inplace=True)
    df["Date"] = pd.to_datetime(df["period"], format="mixed")
    df = df[["Date", "inflation"]].sort_values("Date").reset_index(drop=True)

    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows → {out_path}")
    return df


def download_gdp():
    """
    Real GDP — Quarter-on-quarter growth (seasonally adjusted), Euro Area.
    ECB series: MNA.Q.Y.I8.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.GY
    (year-on-year growth rate)
    """
    REALIZED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REALIZED_DIR / "gdp.csv"

    try:
        df = _download_ecb_series(
            "MNA", "Q.Y.I8.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.GY"
        )
        df = df[["TIME_PERIOD", "OBS_VALUE"]].copy()
        df.rename(columns={"TIME_PERIOD": "period", "OBS_VALUE": "gdp_growth"}, inplace=True)
        df["Date"] = pd.to_datetime(df["period"], format="mixed")
        df = df[["Date", "gdp_growth"]].sort_values("Date").reset_index(drop=True)
    except Exception as e:
        print(f"  ECB MNA download failed ({e}), trying Eurostat ...")
        df = _download_eurostat_gdp()

    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows → {out_path}")
    return df


def _download_eurostat_gdp():
    """Fallback: Eurostat NAMQ_10_GDP for Euro Area GDP growth."""
    url = (
        "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
        "NAMQ_10_GDP?lang=EN&geo=EA20&unit=CLV_PCH_SM&s_adj=SCA&na_item=B1GQ"
    )
    print(f"  Fetching Eurostat GDP ...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # Parse Eurostat JSON response
    time_idx = data["dimension"]["time"]["category"]["index"]
    values = data["value"]

    rows = []
    for time_key, pos in time_idx.items():
        val = values.get(str(pos))
        if val is not None:
            rows.append({"period": time_key, "gdp_growth": float(val)})

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["period"].str.replace("Q", "-Q").apply(
        lambda x: pd.Period(x, freq="Q").start_time
    ))
    df = df[["Date", "gdp_growth"]].sort_values("Date").reset_index(drop=True)
    return df


def download_all():
    """Download all realized series used by the project."""
    print("Downloading realized macroeconomic series ...")
    download_inflation()
    download_gdp()
    print("All realized series downloaded.")


if __name__ == "__main__":
    download_all()
