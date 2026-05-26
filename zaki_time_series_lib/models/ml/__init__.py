from zaki_time_series_lib.models.ml.sklearn_models import (
    LinearModel, RidgeModel, LassoModel, ElasticNetModel,
    RandomForestModel, XGBoostModel, LightGBMModel,
    SVRModel, GaussianProcessModel, KNNModel
)

ML_MODEL_REGISTRY = {
    "LinearRegression": LinearModel,
    "Ridge": RidgeModel,
    "Lasso": LassoModel,
    "ElasticNet": ElasticNetModel,
    "RandomForest": RandomForestModel,
    "XGBoost": XGBoostModel,
    "LightGBM": LightGBMModel,
    "SVR": SVRModel,
    "GaussianProcess": GaussianProcessModel,
    "KNN": KNNModel,
}
