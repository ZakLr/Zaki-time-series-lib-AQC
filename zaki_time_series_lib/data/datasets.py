import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from zaki_time_series_lib.data.base_loader import BaseDatasetLoader
from zaki_time_series_lib.utils.logger import get_logger

logger = get_logger(__name__)


ETT_BASE_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"


class ETTh1Loader(BaseDatasetLoader):
    def __init__(self, cache_dir: Optional[str] = None, **kwargs):
        super().__init__("ETTh1", cache_dir=cache_dir, **kwargs)

    def _download(self) -> pd.DataFrame:
        url = f"{ETT_BASE_URL}/ETTh1.csv"
        df = pd.read_csv(url)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df.astype(np.float32)
        return df


class ETTh2Loader(BaseDatasetLoader):
    def __init__(self, cache_dir: Optional[str] = None, **kwargs):
        super().__init__("ETTh2", cache_dir=cache_dir, **kwargs)

    def _download(self) -> pd.DataFrame:
        url = f"{ETT_BASE_URL}/ETTh2.csv"
        df = pd.read_csv(url)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df.astype(np.float32)
        return df


class ETTm1Loader(BaseDatasetLoader):
    def __init__(self, cache_dir: Optional[str] = None, **kwargs):
        super().__init__("ETTm1", cache_dir=cache_dir, **kwargs)

    def _download(self) -> pd.DataFrame:
        url = f"{ETT_BASE_URL}/ETTm1.csv"
        df = pd.read_csv(url)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df.astype(np.float32)
        return df


class WeatherLoader(BaseDatasetLoader):
    def __init__(self, cache_dir: Optional[str] = None, **kwargs):
        super().__init__("Weather", cache_dir=cache_dir, **kwargs)

    def _download(self) -> pd.DataFrame:
        url = ("https://raw.githubusercontent.com/thuml/Time-Series-Library/main/"
               "dataset/weather/weather.csv")
        df = pd.read_csv(url)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df.astype(np.float32)
        return df


class ElectricityLoader(BaseDatasetLoader):
    def __init__(self, cache_dir: Optional[str] = None, **kwargs):
        super().__init__("Electricity", cache_dir=cache_dir, **kwargs)

    def _download(self) -> pd.DataFrame:
        url = ("https://raw.githubusercontent.com/thuml/Time-Series-Library/main/"
               "dataset/electricity/electricity.csv")
        df = pd.read_csv(url)
        df = df.rename(columns={df.columns[0]: "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df.astype(np.float32)
        return df


class TrafficLoader(BaseDatasetLoader):
    def __init__(self, cache_dir: Optional[str] = None, **kwargs):
        super().__init__("Traffic", cache_dir=cache_dir, **kwargs)

    def _download(self) -> pd.DataFrame:
        url = ("https://raw.githubusercontent.com/thuml/Time-Series-Library/main/"
               "dataset/traffic/traffic.csv")
        df = pd.read_csv(url)
        df = df.rename(columns={df.columns[0]: "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df.astype(np.float32)
        return df


class ExchangeRateLoader(BaseDatasetLoader):
    def __init__(self, cache_dir: Optional[str] = None, **kwargs):
        super().__init__("ExchangeRate", cache_dir=cache_dir, **kwargs)

    def _download(self) -> pd.DataFrame:
        url = ("https://raw.githubusercontent.com/thuml/Time-Series-Library/main/"
               "dataset/exchange_rate/exchange_rate.csv")
        df = pd.read_csv(url)
        df = df.rename(columns={df.columns[0]: "date"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df.astype(np.float32)
        return df


class CSVLoader(BaseDatasetLoader):
    def __init__(self, file_path: str, name: Optional[str] = None,
                 date_col: Optional[str] = None, **kwargs):
        self.file_path = file_path
        self.date_col = date_col
        fname = Path(file_path).stem
        super().__init__(name or f"CSV_{fname}", **kwargs)

    def _download(self) -> pd.DataFrame:
        df = pd.read_csv(self.file_path)
        if self.date_col and self.date_col in df.columns:
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            df = df.set_index(self.date_col).sort_index()
        return df


class GSODLoader(BaseDatasetLoader):
    r"""
    NOAA ISD-Lite weather data for station 725300-94846 (KORD Chicago O'Hare).
    Matches the exact pipeline from qrc-complete.ipynb:
      - Downloads hourly weather data from NOAA for given years
      - Parses ISD-Lite fixed-width format
      - Computes physics-derived features (RH_pct, theta_K, VPD_hPa, u_ms, v_ms)
      - Applies quality control (range clipping)

    Parameters
    ----------
    years : list, optional
        Years to download. Default [2019, 2020, 2021, 2022, 2023, 2024].
    cache_dir : str, optional
        Cache directory. Defaults to settings.DATA_CACHE_DIR.

    The returned DataFrame has columns:
        T_db_C, T_dew_C, SLP_hPa, WS_ms, WD_deg, RH_pct,
        theta_K, VPD_hPa, u_ms, v_ms
    """
    BASE_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite"
    STATION_USAF = "725300"
    STATION_WBAN = "94846"

    R_DRY = 287.05
    CP_DRY = 1004.0
    P0 = 101325.0
    G_FORCE = 9.80665
    KORD_ELEV_M = 201.0
    T_STD = 288.15

    FEATURE_COLS = [
        "T_db_C", "T_dew_C", "SLP_hPa", "WS_ms", "WD_deg",
        "RH_pct", "theta_K", "VPD_hPa", "u_ms", "v_ms",
    ]

    def __init__(self, years: Optional[list] = None,
                 cache_dir: Optional[str] = None, **kwargs):
        if years is None:
            years = [2019, 2020, 2021, 2022, 2023, 2024]
        self.years = sorted(years)
        name = f"GSOD_KORD_{min(years)}-{max(years)}"
        super().__init__(name, cache_dir=cache_dir, **kwargs)

    def _download(self) -> pd.DataFrame:
        import gzip
        import shutil
        import urllib.request
        import urllib.error
        from pathlib import Path

        frames = []
        for year in self.years:
            gz_path = Path(self.cache_dir) / f"kord_{year}.gz"
            txt_path = Path(self.cache_dir) / f"kord_{year}.txt"

            url = (f"{self.BASE_URL}/{year}/"
                   f"{self.STATION_USAF}-{self.STATION_WBAN}-{year}.gz")

            if not (txt_path.exists() and txt_path.stat().st_size > 10000):
                try:
                    self.dlog.get_logger().info(f"Downloading {year}...")
                    req = urllib.request.Request(
                        url, headers={"User-Agent": "Mozilla/5.0"}
                    )
                    with urllib.request.urlopen(req, timeout=60) as r:
                        with open(gz_path, "wb") as f:
                            f.write(r.read())
                    with gzip.open(gz_path, "rb") as f_in, open(txt_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    gz_path.unlink(missing_ok=True)
                except Exception as e:
                    self.dlog.get_logger().warning(
                        f"Year {year} download failed: {e}"
                    )
                    continue

            if txt_path.exists():
                df_raw = self._parse_isd_file(txt_path)
                self.dlog.get_logger().info(
                    f"  {year}: {len(df_raw)} records"
                )
                frames.append(df_raw)

        if not frames:
            raise RuntimeError(
                "No GSOD data downloaded. Check internet or years."
            )

        df = pd.concat(frames).sort_index()
        full_idx = pd.date_range(
            df.index.min(), df.index.max(), freq="h"
        )
        df = df.reindex(full_idx)
        df.index.name = "timestamp"

        self.dlog.get_logger().info(
            f"Total: {len(df)} hourly slots across {len(self.years)} years"
        )

        df = self._compute_physics_features(df)
        df = self._quality_control(df)
        df = df[self.FEATURE_COLS]

        return df

    def _parse_isd_file(self, txt_path: str) -> pd.DataFrame:
        rows = []
        import codecs
        with codecs.open(txt_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 10:
                    continue
                try:
                    year = int(parts[0])
                    month = int(parts[1])
                    day = int(parts[2])
                    hour = int(parts[3])
                    if not (1 <= month <= 12 and 1 <= day <= 31 and 0 <= hour <= 23):
                        continue

                    def parse_field(idx):
                        if idx >= len(parts) or parts[idx] == "-9999":
                            return np.nan
                        return float(parts[idx])

                    rows.append({
                        "year": year, "month": month, "day": day, "hour": hour,
                        "T_db": parse_field(4), "T_dew": parse_field(5),
                        "SLP": parse_field(6), "WD": parse_field(7),
                        "WS": parse_field(8), "sky": parse_field(9),
                        "precip_1h": parse_field(10) if len(parts) > 10 else np.nan,
                        "precip_6h": parse_field(11) if len(parts) > 11 else np.nan,
                    })
                except (ValueError, IndexError):
                    continue

        raw = pd.DataFrame(rows)
        raw["timestamp"] = pd.to_datetime(
            raw[["year", "month", "day", "hour"]], errors="coerce"
        )
        raw = raw.dropna(subset=["timestamp"])
        raw = raw.set_index("timestamp").sort_index()
        raw = raw.rename(columns={
            "T_db": "T_db_C", "T_dew": "T_dew_C",
            "SLP": "SLP_hPa", "WD": "WD_deg", "WS": "WS_ms",
        })
        for col in ["T_db_C", "T_dew_C", "SLP_hPa", "WS_ms"]:
            if col in raw.columns:
                raw[col] = raw[col] / 10.0
        return raw

    def _compute_physics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        a, b = 17.625, 243.04
        T_db = df["T_db_C"]
        T_dew = df["T_dew_C"]

        gamma_T = (a * T_db) / (b + T_db)
        gamma_Td = (a * T_dew) / (b + T_dew)
        df["RH_pct"] = (100.0 * np.exp(gamma_Td - gamma_T)).clip(0, 100)

        elev_factor = np.exp(
            -self.G_FORCE * self.KORD_ELEV_M / (self.R_DRY * self.T_STD)
        )
        df["P_stn_Pa"] = df["SLP_hPa"] * 100.0 * elev_factor

        T_K = T_db + 273.15
        df["theta_K"] = T_K * (self.P0 / df["P_stn_Pa"]) ** (self.R_DRY / self.CP_DRY)

        es = 6.112 * np.exp((a * T_db) / (b + T_db))
        ea = 6.112 * np.exp((a * T_dew) / (b + T_dew))
        df["VPD_hPa"] = (es - ea).clip(lower=0)

        wd_rad = np.deg2rad(df["WD_deg"])
        df["u_ms"] = df["WS_ms"] * np.sin(wd_rad)
        df["v_ms"] = df["WS_ms"] * np.cos(wd_rad)

        return df

    def _quality_control(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.loc[~df["T_db_C"].between(-60, 50), "T_db_C"] = np.nan
        df.loc[~df["T_dew_C"].between(-70, 40), "T_dew_C"] = np.nan
        df.loc[~df["SLP_hPa"].between(870, 1084), "SLP_hPa"] = np.nan
        df.loc[~df["RH_pct"].between(0, 100), "RH_pct"] = np.nan
        df.loc[df["WS_ms"] < 0, "WS_ms"] = np.nan
        return df


def list_available_datasets() -> list:
    return [
        "ETTh1 - Electricity Transformer Temperature (hourly, 7 features)",
        "ETTh2 - Electricity Transformer Temperature (hourly, 7 features)",
        "ETTm1 - Electricity Transformer Temperature (minutely, 7 features)",
        "Weather - 21 weather indicators (10-min frequency)",
        "Electricity - 321 customer electricity consumption (hourly)",
        "Traffic - 862 road occupancy rates (hourly)",
        "ExchangeRate - 8 currency exchange rates (daily)",
        "GSOD_KORD - NOAA ISD-Lite KORD Chicago (hourly, 10 physics features)",
    ]
