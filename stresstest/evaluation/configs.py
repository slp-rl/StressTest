from pathlib import Path
from typing import Literal
import argparse
from pydantic_settings import BaseSettings

CURRENT_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    OPENAI_API_KEY: str = "your_openai_api_key"
    PROMPT_CONFIG_PATH: str = str(CURRENT_DIR / "src/evaluator/evaluation_prompts.yml")
    JUDGE_MODEL_NAME: str = "gpt-4o"
    RESULTS_PATH: str = str(CURRENT_DIR.parent.parent / "results")
    MODEL_TO_EVALUATE: Literal["stresslm", "qwen2audio", "gpt-4o-audio", "mock"] = "mock"
    TASK: Literal["ssr", "ssd"] = "ssr"
    STRESS_TEST_DS: str = "slprl/StressTest"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["ssr", "ssd"], help="Which task to run")
    parser.add_argument("--model_to_evaluate", choices=["stresslm", "qwen2audio", "gpt-4o-audio", "mock"], help="Model name")
    args = parser.parse_args()
    return args

args = parse_args()

configs = Settings(**{
    "TASK": args.task,
    "MODEL_TO_EVALUATE": args.model_to_evaluate
})
