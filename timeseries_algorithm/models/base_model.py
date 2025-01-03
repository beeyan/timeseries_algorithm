import abc

class BaseTimeSeriesModel(abc.ABC):
    """
    
    """

    @abc.abstractmethod
    def fit(self, X, y):
        NotImplementedError

    @abc.abstractmethod
    def predict(self, X):
        NotImplementedError

    @abc.abstractmethod
    def save_model(self, filepath: str):
        NotImplementedError

    @abc.abstractmethod
    def load_model(self, filepath: str):
        NotImplementedError