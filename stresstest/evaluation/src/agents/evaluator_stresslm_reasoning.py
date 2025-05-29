import re
from infra.logger import Logger
from ..data_models import EvaluationTaskOutput
from .evaluator_agent_base import EvaluatorBase


class EvaluatorStresSLMReasoning(EvaluatorBase):
    def __init__(self, logger: Logger):
        self.logger = logger
        self.logger.info("Initialized EvaluatorStresSLMReasoning answer parser.")

    def evaluate_answer(
        self, input_prompt: str, audio_llm_output: str
    ) -> EvaluationTaskOutput:
        self.logger.info(f"Starting evaluation for prompt: {input_prompt}")
        try:
            # audio_llm_output is of the format: number. textual answer. example: "1. This is the intention.",
            self.logger.info(f"Audio LLM output: {audio_llm_output}")
            match = re.match(r"(\d+)", audio_llm_output)
            if not match:
                self.logger.error("Failed to extract integer from audio LLM output.")
                return EvaluationTaskOutput(answer=-1)
            extracted_number = match.group(1)
            self.logger.info(f"Extracted number: {extracted_number}")
            # Convert the extracted number to an integer
            answer = int(extracted_number)
            self.logger.info(f"Final answer extracted: {answer}")
            return EvaluationTaskOutput(answer=answer)
            
        except Exception as e:
            self.logger.error(
                f"An error occurred during evaluation, error: {str(e)}"
            )
        return EvaluationTaskOutput(answer=-1)
