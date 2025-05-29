from abc import ABC, abstractmethod
from ..data_models import EvaluationTaskOutput


class EvaluatorBase(ABC):

    @abstractmethod
    def evaluate_answer(
        self, **kwargs
    ) -> EvaluationTaskOutput:
        pass
