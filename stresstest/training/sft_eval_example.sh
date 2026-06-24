module load cuda
module load nvidia

set -e

# LOAD YOUR PYTHON ENV:
source path/to/StressTest/.venv/bin/activate

# LOAD YOUR ENV VARIABLES:
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY="optional: your wandb api key"
# Set working directory to project root
cd path/to/StressTest

RUN_NAME=stresslm-sft-eval-example
EXPERIMENT_NAME=sft_and_eval_example
PYTHON=path/to/StressTest/.venv/bin/python

$PYTHON -m stresstest.training.sft_example \
  --experiment-name ${EXPERIMENT_NAME} \
  --lr 7e-5 \
  --run-name ${RUN_NAME} \
  --report-to wandb
  # report to wandb is optional


CHECKPOINT_DIR=path/to/StressTest/stresstest/training/experiments/${EXPERIMENT_NAME}/${RUN_NAME}
CHECKPOINT=$(cat ${CHECKPOINT_DIR}/best_checkpoint.txt)
echo "Evaluating best checkpoint: ${CHECKPOINT}"

for TASK in ssd ssr_accuracy; do
  for DS in stresstest stresspresso; do
    echo "Running eval: task=${TASK} ds=${DS}"
    $PYTHON -m stresstest.evaluation.main \
      --task ${TASK} \
      --model_to_evaluate stresslm \
      --ds_name ${DS} \
      --evaluator_type stresslm_custom \
      --stresslm_model_checkpoint ${CHECKPOINT} \
      --results_path ${CHECKPOINT_DIR}/evaluation
  done
done