import yaml
from typing import List
from ..data_models import AudioLLMPromptPool, AudioLLMPrompt


class Prompter:

    def __init__(self, prompt_pool_path: str, evaluator_type: str):
        self.prompt_config_path = prompt_pool_path
        self.prompt_config_raw = self._load_prompt_pool()
        self.evaluator_type = evaluator_type

    def _load_prompt_pool(self):
        with open(self.prompt_config_path, "r") as file:
            prompt_pool = yaml.safe_load(file)
        return prompt_pool

    def create_prompt_template_pool(self):
        audio_llm_prompts = []
        for prompt in self.prompt_config_raw["EVALUATION_PROMPT_POOL"]:
            audio_llm_prompt = AudioLLMPrompt(
                id=prompt["prompt_id"], template=prompt["prompt_template"]
            )
            audio_llm_prompts.append(audio_llm_prompt)
        return AudioLLMPromptPool(prompts=audio_llm_prompts)

    def get_model_evaluation_input_prompt(self, input_prompt, audio_llm_output):
        return { 'input_prompt': input_prompt, 'audio_llm_output': audio_llm_output }

    def create_model_input_prompt(self, 
            prompt_template: str, 
            answers: List[str], 
            transcription: str = None
        ):

        transcription_kwarg = {
            "transcription": transcription
        } if transcription else {}
        
        return prompt_template.format(
            answer_1=answers[0],
            answer_2=answers[1],
            **transcription_kwarg
        )

    def create_model_stress_prediction_prompt(self, 
            prompt_template: str, 
            transcription: str
        ):
        
        return prompt_template.format(
            transcription=transcription
        )