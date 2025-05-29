from pydantic import BaseModel
from typing import Optional, List


class EvaluationTaskOutput(BaseModel):
    # answer is either an int or a list of strings
    answer: int | List[str]
