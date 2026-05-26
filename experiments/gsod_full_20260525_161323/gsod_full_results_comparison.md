# GSOD Full Model Experiment Results

**Run ID**: 20260525_161323  

**Window**: 24  

**Horizons**: [1, 6]  

**Models**: 30  


## Model Ranking by VPT


| Model | VPT | FSDH | Time(s) |
|-------|-----|------|---------|
| L63_Ridge                 |   2 |    2 |     nan |
| L63_RandomForest          |   2 |    2 |     nan |
| Ridge(alpha=1)            |   2 |    1 |    0.03 |
| LinearRegression          |   2 |    1 |    0.06 |
| Persistence               |   2 |    0 |    0.00 |
| XGBoost                   |   2 |    0 |    1.08 |
| Lasso(alpha=0.01)         |   2 |    0 |    1.45 |
| KNN                       |   2 |    0 |    2.54 |
| ElasticNet(alpha=0.01)    |   2 |    0 |    2.57 |
| SVR                       |   2 |    0 |   87.86 |
| RandomForest              |   2 |    0 |   93.03 |
| L63_Persistence           |   1 |    0 |     nan |
| SeasonalNaive             |   0 |    0 |    0.00 |
| ExpSmoothing              |   0 |    0 |    0.12 |
| Theta                     |   0 |    0 |    1.05 |
| ESN500                    |   0 |    0 |    3.70 |
| HoltWinters               |   0 |    0 |   11.96 |
| ESN1000                   |   0 |    0 |   13.90 |
| ARIMA(2,1,2)              |   0 |    0 |   24.04 |
| GRU                       |   0 |    0 |   58.46 |
| LSTM                      |   0 |    0 |   68.70 |
| CNN                       |   0 |    0 |   69.82 |
| Transformer               |   0 |    0 |  176.39 |
| NBeats                    |   0 |    0 |  265.80 |
| L63_LSTM64                |   0 |    0 |     nan |
| L63_ESN500                |   0 |    0 |     nan |

## Per-Horizon Metrics

| Model | Horizon | RMSE | MAE | NRMSE | Skill |
|-------|---------|------|-----|-------|-------|
| Persistence               | 1       | nan | nan | nan | nan |
| Persistence               | 6       | nan | nan | nan | nan |
| SeasonalNaive             | 1       | nan | nan | nan | nan |
| SeasonalNaive             | 6       | nan | nan | nan | nan |
| ARIMA(2,1,2)              | 1       | nan | nan | nan | nan |
| ARIMA(2,1,2)              | 6       | nan | nan | nan | nan |
| ExpSmoothing              | 1       | nan | nan | nan | nan |
| ExpSmoothing              | 6       | nan | nan | nan | nan |
| HoltWinters               | 1       | nan | nan | nan | nan |
| HoltWinters               | 6       | nan | nan | nan | nan |
| Theta                     | 1       | nan | nan | nan | nan |
| Theta                     | 6       | nan | nan | nan | nan |
| LinearRegression          | 1       | nan | nan | nan | nan |
| LinearRegression          | 6       | nan | nan | nan | nan |
| Ridge(alpha=1)            | 1       | nan | nan | nan | nan |
| Ridge(alpha=1)            | 6       | nan | nan | nan | nan |
| Lasso(alpha=0.01)         | 1       | nan | nan | nan | nan |
| Lasso(alpha=0.01)         | 6       | nan | nan | nan | nan |
| ElasticNet(alpha=0.01)    | 1       | nan | nan | nan | nan |
| ElasticNet(alpha=0.01)    | 6       | nan | nan | nan | nan |
| RandomForest              | 1       | nan | nan | nan | nan |
| RandomForest              | 6       | nan | nan | nan | nan |
| XGBoost                   | 1       | nan | nan | nan | nan |
| XGBoost                   | 6       | nan | nan | nan | nan |
| SVR                       | 1       | nan | nan | nan | nan |
| SVR                       | 6       | nan | nan | nan | nan |
| KNN                       | 1       | nan | nan | nan | nan |
| KNN                       | 6       | nan | nan | nan | nan |
| LSTM                      | 1       | nan | nan | nan | nan |
| LSTM                      | 6       | nan | nan | nan | nan |
| GRU                       | 1       | nan | nan | nan | nan |
| GRU                       | 6       | nan | nan | nan | nan |
| CNN                       | 1       | nan | nan | nan | nan |
| CNN                       | 6       | nan | nan | nan | nan |
| Transformer               | 1       | nan | nan | nan | nan |
| Transformer               | 6       | nan | nan | nan | nan |
| NBeats                    | 1       | nan | nan | nan | nan |
| NBeats                    | 6       | nan | nan | nan | nan |
| ESN500                    | 1       | nan | nan | nan | nan |
| ESN500                    | 6       | nan | nan | nan | nan |
| ESN1000                   | 1       | nan | nan | nan | nan |
| ESN1000                   | 6       | nan | nan | nan | nan |
| L63_Persistence           | 1       | nan | nan | nan | nan |
| L63_Persistence           | 6       | nan | nan | nan | nan |
| L63_Ridge                 | 1       | nan | nan | nan | nan |
| L63_Ridge                 | 6       | nan | nan | nan | nan |
| L63_RandomForest          | 1       | nan | nan | nan | nan |
| L63_RandomForest          | 6       | nan | nan | nan | nan |
| L63_LSTM64                | 1       | nan | nan | nan | nan |
| L63_LSTM64                | 6       | nan | nan | nan | nan |
| L63_ESN500                | 1       | nan | nan | nan | nan |
| L63_ESN500                | 6       | nan | nan | nan | nan |