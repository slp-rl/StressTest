from pathlib import Path
from typing import Literal
import argparse
from pydantic_settings import BaseSettings

CURRENT_DIR = Path(__file__).resolve().parent

STRESS_DS_MAP = {
    "stresstest": "slprl/StressTest",
    "stresspresso": "slprl/StressPresso",
}

class Settings(BaseSettings):
    OPENAI_API_KEY: str = "your_openai_api_key" # Set your OpenAI API key here or through environment variable
    PROMPT_CONFIG_PATH: str = str(CURRENT_DIR / "src/evaluator/evaluation_prompts.yml")
    JUDGE_MODEL_NAME: str = "gpt-4o"
    RESULTS_PATH: str = str(CURRENT_DIR.parent.parent / "results")
    MODEL_TO_EVALUATE: Literal["stresslm", "qwen2audio", "gpt-4o-audio", "mock"] = "mock"
    TASK: Literal["ssr_accuracy", "open_ssr", "ssd"] = "ssr_accuracy"
    STRESS_TEST_DS: str = "slprl/StressTest"
    EVALUATOR_TYPE: Literal["judge", "stresslm_custom"] = "judge"
    STRESSLM_MODEL_CHECKPOINT: str = "slprl/StresSLM"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["ssr_accuracy", "open_ssr", "ssd"], help="Which task to run")
    parser.add_argument("--model_to_evaluate", choices=["stresslm", "qwen2audio", "gpt-4o-audio", "mock"], help="Model name")
    parser.add_argument("--ds_name", type=str, choices=["stresstest", "stresspresso"], default="stresstest", help="Dataset name")
    parser.add_argument("--evaluator_type", choices=["judge", "stresslm_custom"], default="judge", help="Type of evaluator agent to use")
    parser.add_argument("--stresslm_model_checkpoint", type=str, default="slprl/StresSLM", help="Path to the StressLM model checkpoint")
    parser.add_argument("--results_path", type=str, default=str(CURRENT_DIR.parent.parent / "results"), help="Results output directory")
    args = parser.parse_args()
    return args

args = parse_args()

configs = Settings(**{
    "TASK": args.task,
    "MODEL_TO_EVALUATE": args.model_to_evaluate,
    "STRESS_TEST_DS": STRESS_DS_MAP[args.ds_name],
    "EVALUATOR_TYPE": args.evaluator_type,
    "STRESSLM_MODEL_CHECKPOINT": args.stresslm_model_checkpoint,
    "RESULTS_PATH": args.results_path,
})
