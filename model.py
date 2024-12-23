from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def fit(self):
        raise NotImplementedError
