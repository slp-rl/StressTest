from tqdm import tqdm
from datasets import Dataset, load_dataset, Audio
from ...configs import configs
from ...clients import storage_client, logger, inference_client
from ..data_models import AudioLLMPrompt, AudioLLMPromptPool
from ..agents import EvaluatorBase, OpenAIEvaluatorAgent, EvaluatorStresSLMReasoning
from .prompter import Prompter
from .evaluation_task_base import EvaluationTaskBase

STRESS_DS_MAP = {
    "slprl/StressTest": "StressTest",
    "slprl/StressPresso": "StressPresso",
}


class EvaluatorStressReasoning(EvaluationTaskBase):

    evaluation_agents = {
        "judge": {
            "ssr_accuracy": lambda logger: OpenAIEvaluatorAgent(logger=logger, prompt_path="evaluator_ssr_accuracy.yml"),
            "open_ssr": lambda logger: OpenAIEvaluatorAgent(logger=logger, prompt_path="evaluator_open_ssr.yml"),
        },
        "stresslm_custom": {"ssr_accuracy": lambda logger: EvaluatorStresSLMReasoning(logger=logger)},
    }
    task_to_prompt_id = {
        "ssr_accuracy": 1,
        "open_ssr": 3,
    }
    

    def __init__(self, evaluator_type: str = 'judge'):
        self.evaluator_type = evaluator_type
        self.inference_client = inference_client
        self.logger = logger
        self.prompter: Prompter = Prompter(prompt_pool_path=configs.PROMPT_CONFIG_PATH, evaluator_type=self.evaluator_type)
        self.dataset: Dataset = load_dataset(
            configs.STRESS_TEST_DS,
            split="test",
        )
        self.prompt_id = self.task_to_prompt_id.get(configs.TASK)
        self.dataset = self.dataset.cast_column("audio", Audio(sampling_rate=16000))
        self.prompt_template_pool: AudioLLMPromptPool = self.prompter.create_prompt_template_pool()
        self.prompt_template: AudioLLMPrompt = self.prompt_template_pool.get_prompt_by_id(prompt_id=self.prompt_id)
        self.evaluator_agent: EvaluatorBase = self.evaluation_agents[self.evaluator_type][configs.TASK](logger=logger)
        self.results_file_path = f"{configs.MODEL_TO_EVALUATE}_{STRESS_DS_MAP[configs.STRESS_TEST_DS]}_evaluation_{configs.TASK}.json"

    def _evaluate_model_answer(self, input_prompt: str, audio_llm_output: str, open_ended_kwargs: dict = None):
        evaluation_prompt_kwargs = self.prompter.get_model_evaluation_input_kwargs(
            input_prompt=input_prompt, audio_llm_output=audio_llm_output, open_ended_kwargs=open_ended_kwargs
        )
        return self.evaluator_agent.evaluate_answer(**evaluation_prompt_kwargs)

    def make_inference(self):
        self.logger.info(f"Making inference for ssr prompt, prompt id: {self.prompt_template.id}")
        initial_data = {
            'dataset': configs.STRESS_TEST_DS,
            "model": f"{configs.MODEL_TO_EVALUATE}",
            "checkpoint": configs.STRESSLM_MODEL_CHECKPOINT if configs.MODEL_TO_EVALUATE == "stresslm" else None,
            "evaluator_model": configs.JUDGE_MODEL_NAME if self.evaluator_type == "judge" else "custom",
            "evaluator_type": self.evaluator_type,
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
            gt_intended_meaning = answers[gt_label_index]
            gt_transcription = sample["transcription"]
            gt_stressed_words = ", ".join(sample['stress_pattern']['words'])
            
            # get input prompt
            input_prompt = self.prompter.create_model_input_prompt(
                prompt_template=self.prompt_template.template,
                answers=answers,
            )

            # prepare input for inference
            pred_input = self.inference_client.prepare(
                text_prompt=input_prompt,
                audio=sample["audio"],
                sample_idx=sample_idx,
            )

            # Make inference
            audio_llm_output = self.inference_client.predict(
                **pred_input
            )
            # Evaluate with Judge
            open_ended_kwargs = None if configs.TASK != "open_ssr" else {"gt_transcription": gt_transcription, "gt_stressed_words": gt_stressed_words, "gt_intended_meaning": gt_intended_meaning}
            evaluation_output = self._evaluate_model_answer(
                input_prompt=input_prompt, audio_llm_output=audio_llm_output, open_ended_kwargs=open_ended_kwargs
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
    
    def _calculate_avarage_score(self, preds):
        return sum(preds) / len(preds)
    
    def _get_final_score_by_task(self, preds, labels):
        if configs.TASK == "ssr_accuracy":
            return self._calculate_accuracy(preds=preds, labels=labels)
        elif configs.TASK == "open_ssr":
            return self._calculate_avarage_score(preds=preds)
        else:
            raise NotImplementedError(f"Task {configs.TASK} not implemented in result calculation.")

    def evaluate(self):
        self.logger.info(f"Evaluating task {configs.TASK} with prompt: {self.prompt_template.id}")
        evaluations_data = storage_client.load_json(file_name=self.results_file_path)
        predictions_data = evaluations_data["evaluations"]
        preds_and_labels = [
            (evaluation["agent_prediction"]["answer"], evaluation["label_answer"])
            for evaluation in predictions_data
        ]

        preds, labels = zip(*preds_and_labels)
        final_score = self._get_final_score_by_task(preds=preds, labels=labels)

        results = {
            "task": configs.TASK,
            "dataset": configs.STRESS_TEST_DS,
            "prompt_id": self.prompt_template.id,
            "n_samples": len(preds),
            "description": f"{configs.TASK} on {STRESS_DS_MAP[configs.STRESS_TEST_DS]} evaluation results for {configs.MODEL_TO_EVALUATE} with evaluator {self.evaluator_type}",
            "ssr_score": final_score,
        }
        self.logger.info(f"Results: {results}")
        file_path = f"{configs.MODEL_TO_EVALUATE}_{STRESS_DS_MAP[configs.STRESS_TEST_DS]}_{configs.TASK}_evaluation_results.json"
        storage_client.save_json(file_name=file_path, data=results)
        return results
