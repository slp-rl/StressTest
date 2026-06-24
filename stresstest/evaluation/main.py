


if __name__ == "__main__":
    from .configs import configs
    from .clients import logger
    from .src.evaluator import EvaluatorStressReasoning, EvaluatorStressDetection, EvaluationTaskBase

    task_to_evaluator_class = {
        "ssr_accuracy": EvaluatorStressReasoning,
        "open_ssr": EvaluatorStressReasoning,
        "ssd": EvaluatorStressDetection
    }
    evaluator: EvaluationTaskBase = task_to_evaluator_class[configs.TASK](evaluator_type=configs.EVALUATOR_TYPE)     
    logger.info(f"Evaluating task: {configs.TASK}")
    
    evaluator.make_inference()
    evaluator.evaluate()

