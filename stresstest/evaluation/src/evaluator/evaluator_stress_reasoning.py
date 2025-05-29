from tqdm import tqdm
from datasets import Dataset, load_dataset
from ...configs import configs
from ...clients import storage_client, logger, inference_client
from ..data_models import AudioLLMPrompt, AudioLLMPromptPool
from ..agents import EvaluatorBase, OpenAIEvaluatorAgent, EvaluatorStresSLMReasoning
from .prompter import Prompter
from .evaluation_task_base import EvaluationTaskBase



class EvaluatorStressReasoning(EvaluationTaskBase):

    evaluation_agents = {
        "judge": lambda logger: OpenAIEvaluatorAgent(logger=logger, prompt_path="evaluator_stress_reasoning.yml"),
        "stresslm_custom": lambda logger: EvaluatorStresSLMReasoning(logger=logger),
    }
    prompt_id = 1

    def __init__(self, evaluator_type: str = 'judge'):
        self.evaluator_type = evaluator_type
        self.inference_client = inference_client
        self.logger = logger
        self.prompter: Prompter = Prompter(prompt_pool_path=configs.PROMPT_CONFIG_PATH, evaluator_type=self.evaluator_type)
        self.dataset: Dataset = load_dataset(
            configs.STRESS_TEST_DS,
            split="test",
            token=configs.HF_API_TOKEN,
        )
        self.prompt_template_pool: AudioLLMPromptPool = self.prompter.create_prompt_template_pool()
        self.prompt_template: AudioLLMPrompt = self.prompt_template_pool.get_prompt_by_id(prompt_id=self.prompt_id)
        self.evaluator_agent: EvaluatorBase = self.evaluation_agents[self.evaluator_type](logger=logger)
        self.results_file_path = f"{configs.MODEL_TO_EVALUATE}_evaluation_ssr.json"

    def _evaluate_model_answer(self, input_prompt: str, audio_llm_output: str):
        evaluation_prompt_kwargs = self.prompter.get_model_evaluation_input_prompt(
            input_prompt=input_prompt, audio_llm_output=audio_llm_output
        )
        return self.evaluator_agent.evaluate_answer(**evaluation_prompt_kwargs)

    def make_inference(self):
        self.logger.info(f"Making inference for ssr prompt, prompt id: {self.prompt_template.id}")
        initial_data = {
            'dataset': "slprl/StressTest",
            "model": f"{configs.MODEL_TO_EVALUATE}",
            "evaluator_model": configs.JUDGE_MODEL_NAME,
            "evaluations": []
        }

        # Save initial data to results file
        storage_client.save_json(
            file_name=self.results_file_path,
            data=initial_data
        )
        
        for sample_idx, sample in tqdm(enumerate(self.dataset)):
            self.logger.info(
                f"Evaluating intonation: {sample['intonation']}, {sample_idx=}"
            )
            answers = sample['possible_answers']
            gt_label_index = sample['label']

            # get input prompt
            input_prompt = self.prompter.create_model_input_prompt(
                prompt_template=self.prompt_template.template,
                answers=answers,
            )

            # prepare input for inference
            pred_input = self.inference_client.prepapre(
                text_prompt=input_prompt,
                audio=sample["audio"],
                sample_idx=sample_idx,
            )

            # Make inference
            audio_llm_output = self.inference_client.predict(
                **pred_input
            )

            # Evaluate with Judge
            evaluation_output = self._evaluate_model_answer(
                input_prompt=input_prompt, audio_llm_output=audio_llm_output
            )
            # Save results
            results_data = storage_client.load_json(file_name=self.results_file_path)
            results_data["evaluations"].append(
                {
                    "sample_index": sample_idx,
                    "intonation": sample["intonation"],
                    "prompt_id": self.prompt_template.id,
                    "sample_id": sample["transcription_id"],
                    "interpretation_id": sample["interpretation_id"],
                    "audio_path": sample["audio"]["path"],
                    "input_prompt": input_prompt,
                    "model_answer": audio_llm_output,
                    "agent_prediction": {"answer": evaluation_output.answer},
                    "label_index": gt_label_index,
                    "label_answer": gt_label_index + 1
                }
            )
            storage_client.save_json(file_name=self.results_file_path, data=results_data)

    def _calculate_accuracy(self, preds, labels):
        correct = 0
        for pred, label in zip(preds, labels):
            if pred == label:
                correct += 1
        accuracy = correct / len(labels)
        return accuracy

    def evaluate(self):
        self.logger.info(f"Evaluating for prompt: {self.prompt_template.id}")
        evaluations_data = storage_client.load_json(file_name=self.results_file_path)
        predictions_data = evaluations_data["evaluations"]
        preds_and_labels = [
            (evaluation["agent_prediction"]["answer"], evaluation["label_answer"])
            for evaluation in predictions_data
        ]

        preds, labels = zip(*preds_and_labels)
        accuracy = self._calculate_accuracy(preds=preds, labels=labels)

        results = {
            "task": "SSR",
            "dataset": configs.STRESS_TEST_DS,
            "prompt_id": self.prompt_template.id,
            "n_samples": len(preds),
            "description": f"StessTest Evaluation results for {configs.MODEL_TO_EVALUATE} with evaluator {self.evaluator_type}",
            "ssr_accuracy": accuracy,
        }
        self.logger.info(f"Results: {results}")
        file_path = f"{configs.MODEL_TO_EVALUATE}_stresstest_ssr_evaluation_results.json"
        storage_client.save_json(file_name=file_path, data=results)
        return results
