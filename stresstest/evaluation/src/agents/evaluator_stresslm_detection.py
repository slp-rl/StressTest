import re
from infra.logger import Logger
from ..data_models import EvaluationTaskOutput
from .evaluator_agent_base import EvaluatorBase


class EvaluatorStresSLMDetection(EvaluatorBase):
    def __init__(self, logger: Logger):
        self.logger = logger
        self.logger.info("Initialized EvaluatorStresSLMDetection answer parser.")

    def evaluate_answer(
        self, input_prompt: str, audio_llm_output: str, **kwargs
    ) -> EvaluationTaskOutput:
        self.logger.info(f"Starting evaluation for prompt: {input_prompt}")
        try:
            # audio_llm_output is of the format: ['word1', 'word2'] or ['word'] etc, given as a string. 
            # example: "['this', 'and']"
            # answer should be a list of words: ['this', 'and']
            self.logger.info(f"Audio LLM output: {audio_llm_output}")
            match = re.match(r"\[('.*?')\]", audio_llm_output)
            if not match:
                self.logger.error("Failed to extract word from audio LLM output.")
                return EvaluationTaskOutput(answer=-1)
            extracted_words = match.group(1)
            self.logger.info(f"Extracted words: {extracted_words}")
            # Convert the extracted words to a list
            answer = [word.strip("'") for word in extracted_words.split(", ")]
            self.logger.info(f"Final answer extracted: {answer}")
            return EvaluationTaskOutput(answer=answer)
            
        except Exception as e:
            self.logger.error(
                f"An error occurred during evaluation, error: {str(e)}"
            )
        return EvaluationTaskOutput(answer=-1)
