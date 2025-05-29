from infra.storage import FileStorage
from infra.logger import Logger
from .configs import configs
from .src.inference import InferenceClientBase, MockInferenceClient

logger = Logger(context={"service": "evaluator"}, use_context_var=True)
storage_client = FileStorage(storage_path=configs.RESULTS_PATH)

inference_client : InferenceClientBase = MockInferenceClient()
if configs.MODEL_TO_EVALUATE == "qwen2audio":
    from .src.inference import InferenceClientQwen2Audio
    logger.info("Using Qwen2Audio Inference Client")
    inference_client = InferenceClientQwen2Audio(logger=logger)

elif configs.MODEL_TO_EVALUATE == "stresslm":
    from .src.inference import InferenceClientQwen2Audio
    logger.info("Using Qwen2Audio Inference Client for StressLM", context={"model": "stresslm"})
    inference_client = InferenceClientQwen2Audio(logger=logger, stresslm=True)

elif configs.MODEL_TO_EVALUATE == "gpt-4o-audio":
    from .src.inference import InferenceClientGPTAudio
    logger.info("Using GPT-4o Audio Inference Client")
    inference_client = InferenceClientGPTAudio()

elif configs.MODEL_TO_EVALUATE == "mock":
    logger.info("Using Mock Inference Client")
    inference_client = MockInferenceClient()
