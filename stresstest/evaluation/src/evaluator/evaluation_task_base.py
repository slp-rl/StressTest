from abc import ABC, abstractmethod


class EvaluationTaskBase(ABC):
    """
    Base class for evaluators.
    """

    def __init__(self, logger=None, **kwargs):
        self.logger = logger

    @abstractmethod
    def evaluate(self):
        """
        Evaluate the answer based on the provided arguments.
        """
        pass

    @abstractmethod
    def make_inference(self):
        """
        Make inference using the evaluator.
        """
        pass