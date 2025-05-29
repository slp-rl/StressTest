from openai import OpenAI
import soundfile as sf
import io
import base64
from ...configs import configs
from .inference_client_base import InferenceClientBase


class InferenceClientGPTAudio(InferenceClientBase):

    model_name = "gpt-4o-audio-preview-2024-12-17"

    def __init__(self):
        self.client = OpenAI(api_key=configs.OPENAI_API_KEY)

    def predict(self, text, **kwargs):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=text,
            **kwargs
        )
        return completion.choices[0].message.content

    def model_conversation_template(self, text_prompt, encoded_audio): 
        return [
        {
            "role": "user",
            "content": [
                { 
                    "type": "text",
                    "text": text_prompt
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": encoded_audio,
                        "format": "wav"
                    }
                }
            ]
        },
    ]

    def prepare_audio(self, audio):
        # Load audio sample
        audio_array = audio["array"]
        sampling_rate = audio["sampling_rate"]
        # Save audio to WAV in-memory buffer
        buffer = io.BytesIO()
        sf.write(buffer, audio_array, sampling_rate, format='WAV')
        wav_bytes = buffer.getvalue()
        # Base64 encode
        encoded_audio = base64.b64encode(wav_bytes).decode("utf-8")
        return encoded_audio

    def prepapre(self, text_prompt, audio, **kwargs):
        # Prepare the audio
        encoded_audio = self.prepare_audio(audio)
        conversation = self.model_conversation_template(text_prompt, encoded_audio)
        return {
            "text": conversation,
            "max_tokens": 256,
            'modalities': ['text'],
        }