from zaki_time_series_lib.models.deep_learning.lstm_models import LSTMModel, BiLSTMModel, GRUModel
from zaki_time_series_lib.models.deep_learning.cnn_models import CNNModel, TCNModel
from zaki_time_series_lib.models.deep_learning.transformer_models import TransformerModel, InformerModel
from zaki_time_series_lib.models.deep_learning.nbeats import NBeatsModel
from zaki_time_series_lib.models.deep_learning.esn import ESNModel, ESN500Model, ESN1000Model

DL_MODEL_REGISTRY = {
    "LSTM": LSTMModel,
    "BiLSTM": BiLSTMModel,
    "GRU": GRUModel,
    "CNN": CNNModel,
    "TCN": TCNModel,
    "Transformer": TransformerModel,
    "Informer": InformerModel,
    "NBeats": NBeatsModel,
    "ESN": ESNModel,
    "ESN500": ESN500Model,
    "ESN1000": ESN1000Model,
}
