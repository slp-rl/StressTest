from .inference_client_base import InferenceClientBase

class MockInferenceClient(InferenceClientBase):

    def predict(self, *args, **kwargs) -> str:
        return "2. The answer is two."
    
    def prepare(self, *args, **kwargs) -> dict:
        return {
            "text": """What is the answer to this question? 
                1. The answer is one. 
                2. The answer is two. 
                Answer: """, 
            "audio": "audio.wav"
        }
    