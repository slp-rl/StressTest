from pydantic import BaseModel
from typing import List


class AudioLLMPrompt(BaseModel):
    id: int
    template: str


class AudioLLMPromptPool(BaseModel):
    prompts: List[AudioLLMPrompt]

    def get_prompt_by_id(self, prompt_id: int) -> AudioLLMPrompt:
        for prompt in self.prompts:
            if prompt.id == prompt_id:
                return prompt
        raise ValueError(f"Prompt with id {prompt_id} not found in the pool.")
