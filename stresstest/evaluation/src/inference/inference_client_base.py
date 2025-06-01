from abc import ABC, abstractmethod

class InferenceClientBase(ABC):

    @abstractmethod
    def prepare(self, *args, **kwargs) -> dict:
        """
        Prepare method to be implemented by subclasses. 
        This method should return a dictionary with the necessary inputs for the predict method.
        The returned ditionary is handled by the evaluation script.
        """
        pass

    @abstractmethod
    def predict(self, *args, **kwargs) -> str:
        """Predict method to be implemented by subclasses."""
        pass
