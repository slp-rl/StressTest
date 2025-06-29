import yaml


class PromptTemplateManager:
    def __init__(self, yml_path):
        with open(yml_path, "r") as f:
            self.prompts = yaml.safe_load(f)

    def get_prompt_by_id(self, prompt_id):
        if not (0 <= prompt_id < len(self.prompts) + 1):
            raise IndexError(f"Prompt ID {prompt_id} is out of range.")
        for prompt in self.prompts:
            if prompt["id"] == prompt_id:
                return prompt
        raise ValueError(f"Prompt ID {prompt_id} not found.")

    def get_keys_in_prompts(self):
        return ['answer_1', 'answer_2', 'answer_label', 'correct_answer', 'description', 'emphasized_words', 'transcription']

    def render(self, example, prompt_id):
        """
        Renders the question and answer for a given example and prompt_id.
        `example` should be a dict with all required variables filled in.
        """
        prompt = self.get_prompt_by_id(prompt_id)

        try:
            question = prompt["question"].format(**example).strip()
            
            if prompt['id'] == 4:
                words = [f"'{w.strip()}'" for w in example['emphasized_words'].split(",")]
                example['emphasized_words'] = f"[{', '.join(words)}]"
            answer = prompt["answer"].format(**example).strip()
        except KeyError as e:
            raise ValueError(f"Missing variable in example: {e}")

        return {
            "task": prompt["TASK"],
            "prompt_id": prompt_id,
            "question": question,
            "answer": answer,
            "audio": example["audio"],
            "audio_id": example["audio_id"]
        }

    def get_tasks(self):
        return list(set(p["TASK"] for p in self.prompts))

    def get_prompts_by_tasks(self, task_names):
        return [p['id'] for p in self.prompts if p["TASK"] in task_names]