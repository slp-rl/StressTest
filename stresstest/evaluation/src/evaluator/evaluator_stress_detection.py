import re
from tqdm import tqdm
import evaluate
from datasets import Dataset, load_dataset
from ...configs import configs
from ...clients import storage_client, logger, inference_client
from ..data_models import AudioLLMPrompt, AudioLLMPromptPool
from ..agents import EvaluatorBase, OpenAIEvaluatorAgent
from .prompter import Prompter



class EvaluatorStressDetection:

    evaluation_agents = {
        "judge": lambda logger: OpenAIEvaluatorAgent(logger=logger, prompt_path="evaluator_stress_detection.yml"),
    }
    prompt_id = 2

    def __init__(self, evaluator_type: str = 'judge'):
        self.inference_client = inference_client
        self.evaluator_type = evaluator_type
        self.prompter: Prompter = Prompter(prompt_pool_path=configs.PROMPT_CONFIG_PATH, evaluator_type=self.evaluator_type)
        self.dataset: Dataset = self.prepare_dataset()
        self.prompt_template_pool: AudioLLMPromptPool = self.prompter.create_prompt_template_pool()
        self.prompt_template: AudioLLMPrompt = self.prompt_template_pool.get_prompt_by_id(prompt_id=self.prompt_id)
        self.logger = logger
        self.evaluator_agent: EvaluatorBase = self.evaluation_agents[self.evaluator_type](logger=logger)
        self.precision_metric = evaluate.load("precision")
        self.recall_metric = evaluate.load("recall")
        self.f1_metric = evaluate.load("f1")
        self.results_file_path = f"{configs.MODEL_TO_EVALUATE}_evaluation_ssd.json"

    def prepare_dataset(self):
        dataset: Dataset = load_dataset(
            configs.STRESS_TEST_DS,
            split="test",
        )
        dataset = dataset.map(
            lambda x: {"stress_labels": x['stress_pattern']['binary']}
        )
        return dataset


    def _evaluate_model_answer(self, input_prompt: str, audio_llm_output: str):
        evaluation_prompt_kwargs = self.prompter.get_model_evaluation_input_prompt(
            input_prompt=input_prompt, audio_llm_output=audio_llm_output
        )
        return self.evaluator_agent.evaluate_answer(**evaluation_prompt_kwargs)
    
    def process_agent_answer(self, predicted_words: list, transcription):
        # return binary version of the predicted words according to the transcription
        words = transcription.split()
        normalized_predicted_words = self.normalize_words_list(predicted_words)
        binary_predicted_words = [1 if word in normalized_predicted_words else 0 for word in words]
        return binary_predicted_words
    
    def normalize_sentence(self, text):
        # Lowercase
        text = text.lower()
        # Remove punctuation (keep only letters, numbers, and whitespace)
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def normalize_words_list(self, lst):
        # Normalize each string in the list
        normalized_list = [self.normalize_sentence(item) for item in lst]
        return normalized_list


    def make_inference(self):
        self.logger.info(f"Making inference for prompt: {self.prompt_template.id}")
        initial_data = {
            "task": "stress_detection",
            "dataset": configs.STRESS_TEST_DS,
            "model": configs.MODEL_TO_EVALUATE,
            "evaluator_model": configs.JUDGE_MODEL_NAME,
            "evaluations": []
        }
        storage_client.save_json(
            file_name=self.results_file_path,
            data=initial_data
        )
        
        for sample_idx, sample in tqdm(enumerate(self.dataset)):
            transcription = self.normalize_sentence(sample["transcription"])

            self.logger.info(
                f"Evaluating {sample_idx=}, {transcription=}"
            )
            input_prompt = self.prompter.create_model_stress_prediction_prompt(
                prompt_template=self.prompt_template.template,
                transcription=transcription
            )

            pred_input = self.inference_client.prepapre(
                text_prompt=input_prompt,
                audio=sample["audio"],
                sample_idx=sample_idx,
            )

            audio_llm_output = self.inference_client.predict(
                **pred_input
            )
            evaluation_output = self._evaluate_model_answer(
                input_prompt=input_prompt, audio_llm_output=audio_llm_output
            )
            binary_predicted_words = self.process_agent_answer(predicted_words=evaluation_output.answer, transcription=transcription)
            # Save results
            results_data = storage_client.load_json(file_name=self.results_file_path)
            results_data["evaluations"].append(
                {
                    "sample_index": sample_idx,
                    "intonation": sample["intonation"],
                    "prompt_id": self.prompt_template.id,
                    "audio_path": sample["audio"]["path"],
                    "input_prompt": input_prompt,
                    "model_answer": audio_llm_output,
                    "agent_prediction": {"answer": evaluation_output.answer},
                    "stress_pred": binary_predicted_words,
                    "stress_labels": sample["stress_labels"],
                }
            )
            storage_client.save_json(file_name=self.results_file_path, data=results_data)


    def compute_prf_metrics(self, predictions, references, average="binary"):
        """
        Computes precision, recall, and F1 using Hugging Face's `evaluate`.
        Args:
            predictions (List[int]): Model's predicted labels.
            references  (List[int]): True labels.
            average     (str): "binary", "macro", "micro", or "weighted".
                            Use "binary" for two-class tasks.
        Returns:
            Dict[str, float]: e.g. {"precision": 0.8, "recall": 0.75, "f1": 0.77}
        """
        p = self.precision_metric.compute(predictions=predictions, references=references, average=average)["precision"]
        r = self.recall_metric.compute(predictions=predictions, references=references, average=average)["recall"]
        f = self.f1_metric.compute(predictions=predictions, references=references, average=average)["f1"]

        return {"precision": p, "recall": r, "f1": f}

    def evaluate(self):
        self.logger.info(f"Evaluating for prompt: {self.prompt_template.id}")
        evaluations_data = storage_client.load_json(file_name=self.results_file_path)
        preds = []
        labels = []
        for eval in evaluations_data["evaluations"]:
            pred = eval["stress_pred"]
            label = eval["stress_labels"]
            assert len(pred) == len(label), f"Predictions and labels length mismatch: {len(pred)} vs {len(label)} for {pred=} and {label=}, {eval['sample_index']=}"
            preds.extend(pred)
            labels.extend(label)
        ssd_metrics = self.compute_prf_metrics(predictions=preds, references=labels)

        results = {
            "task": "SSD",
            "dataset": configs.STRESS_TEST_DS,
            "prompt_id": self.prompt_template.id,
            "description": f"StessTest results for {configs.MODEL_TO_EVALUATE} on {configs.STRESS_TEST_DS} with evaluator {self.evaluator_type}",
            "ssd_metrics": ssd_metrics,
        }
        self.logger.info(f"Results: {results}")
        file_path = f"{configs.MODEL_TO_EVALUATE}_evaluation_metrics_ssd.json"
        storage_client.save_json(file_name=file_path, data=results)
        return results
