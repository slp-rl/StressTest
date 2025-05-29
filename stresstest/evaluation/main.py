


if __name__ == "__main__":
    from .configs import configs
    from .clients import logger
    from .src.evaluator import EvaluatorStressReasoning, EvaluatorStressDetection, EvaluationTaskBase

    task_to_evaluator_class = {
        "ssr": EvaluatorStressReasoning,
        "ssd": EvaluatorStressDetection
    }
    evaluator_type = 'judge' # "judge", "stresslm_custom"
    evaluator: EvaluationTaskBase = task_to_evaluator_class[configs.TASK](evaluator_type=evaluator_type)     
    logger.info(f"Evaluating task: {configs.TASK}")
    
    evaluator.make_inference()
    evaluator.evaluate()

