from zaki_time_series_lib.data.base_loader import BaseDatasetLoader
from zaki_time_series_lib.data.datasets import (
    ETTh1Loader, ETTh2Loader, ETTm1Loader,
    WeatherLoader, ElectricityLoader, TrafficLoader, ExchangeRateLoader,
    GSODLoader,
)

DATASET_REGISTRY = {
    "ETTh1": ETTh1Loader,
    "ETTh2": ETTh2Loader,
    "ETTm1": ETTm1Loader,
    "Weather": WeatherLoader,
    "Electricity": ElectricityLoader,
    "Traffic": TrafficLoader,
    "ExchangeRate": ExchangeRateLoader,
    "GSOD_KORD": GSODLoader,
}
