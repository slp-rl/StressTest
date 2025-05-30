import time
import torch
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from peft import PeftModel, PeftConfig
from infra.logger import Logger
from ...configs import configs
from .inference_client_base import InferenceClientBase


class InferenceClientQwen2Audio(InferenceClientBase):

    model_name = "Qwen/Qwen2-Audio-7B-Instruct"
    stresslm_model_name = "slprl/StresSLM"

    def __init__(self, logger: Logger, stresslm: bool = False):
        self.logger = logger

        self.processor = AutoProcessor.from_pretrained(self.model_name)

        if stresslm:
            self.logger.info(f"Loading {self.stresslm_model_name} model from hub")
            peft_config = PeftConfig.from_pretrained(self.stresslm_model_name)
            base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
                peft_config.base_model_name_or_path,
                device_map="cuda"
            )
            self.model = PeftModel.from_pretrained(base_model, self.stresslm_model_name)
        else:
            self.logger.info(f"Loading model {self.model_name}")
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self.model_name,
                device_map="cuda"
            )

        self.model.eval()

    def predict(self, text, audio, **kwargs):
        inputs = self.processor(text=text, audios=audio, return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        start = time.time()
        generate_ids = self.model.generate(**inputs, max_length=512)
        self.logger.info(f"Predicted in {time.time() - start:.2f}s")
        generate_ids = generate_ids[:, inputs['input_ids'].size(1):]
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return response[0]

    def model_conversation_template(self, text_prompt, audio_path): 
        return [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path},
                {"type": "text", "text": text_prompt},
            ]},
        ]

    def prepapre(self, text_prompt, audio, **kwargs):
        conversation = self.model_conversation_template(text_prompt, audio['path'])
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audio_resample = librosa.resample(
            y=audio['array'], 
            orig_sr=audio['sampling_rate'], 
            target_sr=self.processor.feature_extractor.sampling_rate
        )
        pred_input = {
            "text": [text],
            "audio": [audio_resample]
        }

        return pred_input
