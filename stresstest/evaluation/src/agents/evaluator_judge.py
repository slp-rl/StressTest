import openai
import yaml
import json
import re
from pathlib import Path
from infra.logger import Logger
from ...configs import configs
from ..data_models import EvaluationTaskOutput
from .evaluator_agent_base import EvaluatorBase


class OpenAIEvaluatorAgent(EvaluatorBase):
    def __init__(self, logger: Logger, prompt_path: str = None):
        assert prompt_path is not None, "Prompt path must be provided"
        self.logger = logger
        self.system_prompt, self.user_prompt_template = self._load_prompts(prompt_path)
        self.logger.info("Initialized OpenAIEvaluatorAgent judge.")

    def _load_prompts(self, prompt_path: str):
        path = Path(__file__).parent / prompt_path
        with open(path, "r") as f:
            content = yaml.safe_load(f)
            return content["system_prompt"], content["user_prompt"]

    def evaluate_answer(self, input_prompt: str, audio_llm_output: str, num_retries: int = 3) -> EvaluationTaskOutput:
        user_message = self.user_prompt_template.format(
            input_prompt=input_prompt,
            audio_llm_output=audio_llm_output
        )
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]

        while num_retries > 0:
            try:
                response = openai.chat.completions.create(
                    model=configs.JUDGE_MODEL_NAME,
                    messages=messages,
                    temperature=0,
                )
                content = response.choices[0].message.content
                self.logger.info(f"OpenAI response: {content}")
                parsed = eval(content.strip())  # Replace with `json.loads` for safety if strict JSON
                return EvaluationTaskOutput(**parsed)
            except Exception as e:
                self.logger.warning(
                    f"Evaluation failed: {e}. Retries left: {num_retries - 1}"
                )
                num_retries -= 1

        self.logger.error("Evaluation failed after all retries. Returning default.")
        return EvaluationTaskOutput(answer=['Error Occured'])
