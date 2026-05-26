from zaki_time_series_lib.models.statistical.persistence import PersistenceModel, SeasonalNaiveModel
from zaki_time_series_lib.models.statistical.arima import ARIMAModel, SARIMAModel, AutoARIMAModel
from zaki_time_series_lib.models.statistical.garch import GARCHModel
from zaki_time_series_lib.models.statistical.exponential_smoothing import ExponentialSmoothingModel, HoltWintersModel
from zaki_time_series_lib.models.statistical.theta import ThetaModel

STATISTICAL_MODEL_REGISTRY = {
    "Persistence": PersistenceModel,
    "SeasonalNaive": SeasonalNaiveModel,
    "ARIMA": ARIMAModel,
    "SARIMA": SARIMAModel,
    "AutoARIMA": AutoARIMAModel,
    "GARCH": GARCHModel,
    "ExponentialSmoothing": ExponentialSmoothingModel,
    "HoltWinters": HoltWintersModel,
    "Theta": ThetaModel,
}
