import os
import wandb
import argparse
import math
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments, AutoProcessor, Qwen2AudioForConditionalGeneration, TrainerCallback, TrainerState, TrainerControl
from infra.storage import FileStorage
from .utils.collator import StressDataCollator
from .stress_17k import DatasetAugmentation


CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))


class StopAfterNStepsCallback(TrainerCallback):
    def __init__(self, stop_step: int):
        self.stop_step = stop_step

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step >= self.stop_step:
            print(f"Stopping at step {state.global_step}")
            control.should_training_stop = True
            control.should_save = True
        return control


class EvalAndSaveAtEndCallback(TrainerCallback):
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step == state.max_steps:
            control.should_evaluate = True
            control.should_save = True
        return control


def compute_stage_training_steps(full_size, fine_size, batch_size):
    steps_full = math.ceil(full_size / batch_size)
    steps_fine = math.ceil(fine_size / batch_size)
    total_steps = steps_full + steps_fine
    return steps_full, steps_fine, total_steps


def train():
    parser = argparse.ArgumentParser(description="Train a model with LoRA")
    parser.add_argument('--experiment-name', type=str, default='sft_example', help='Name of the experiment')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate for optimizer')
    parser.add_argument('--report-to', type=str, default='none', choices=['wandb', 'none'], help='Whether to report training metrics to wandb or not')
    parser.add_argument('--run-name', type=str, default='stresslm-sft', help='Run name for wandb')
    args = parser.parse_args()
    report_to = args.report_to
    wandb_api_key = os.environ.get('WANDB_API_KEY')
    if report_to == 'wandb' and not wandb_api_key:
        raise ValueError("Wandb reporting selected but WANDB_API_KEY environment variable is not set.")

    run_name = args.run_name
    print(f"Experiment name: {args.experiment_name}")
    print(f"Run name: {run_name}")
    print(f"Reporting to: {report_to}")

    stresslm_dataset_handler = DatasetAugmentation(n_proc=8
                                ).train_test_split(test_size=0.15
                                    ).prepare_structure_for_augmentation(
                                        ).augment_with_training_prompts(tasks='all'
                                            ).shuffle_augmented_dataset()
    # DatasetDict({
    #     train_full: Dataset({
    #         features: ['transcription_id', 'interpretation_id', 'audio', 'audio_id', 'ds_name', 'task', 'prompt_id', 'question', 'answer'],
    #         num_rows: 16812
    #     })
    #     train_fine: Dataset({
    #         features: ['transcription_id', 'interpretation_id', 'audio', 'audio_id', 'ds_name', 'task', 'prompt_id', 'question', 'answer'],
    #         num_rows: 4456
    #     })
    #     test: Dataset({
    #         features: ['transcription_id', 'interpretation_id', 'audio', 'audio_id', 'ds_name', 'task', 'prompt_id', 'question', 'answer'],
    #         num_rows: 197
    #     })
    # })
    dataset_full = stresslm_dataset_handler.get_full_dataset() # train_full and test
    dataset_fine = stresslm_dataset_handler.get_fine_dataset() # train_fine and test

    ## Load model
    print('Loading model...')
    model_name = 'Qwen/Qwen2-Audio-7B-Instruct'
    model = Qwen2AudioForConditionalGeneration.from_pretrained(model_name, trust_remote_code=True).to('cuda')
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    configs = {
        'model_name': model_name,
        'lr': args.lr,
        'scheduler': 'cosine',
        'batch_size': 8,
        'gradient_accumulation_steps': 2,
        'target_modules': ['q_proj', 'v_proj'],
        'experiment_name': args.experiment_name,
    }

    steps_full, steps_fine, total_steps = compute_stage_training_steps(
        full_size=len(dataset_full['train']),
        fine_size=len(dataset_fine['train']),
        batch_size=configs['batch_size'] * configs['gradient_accumulation_steps'],
    )
    print(f"Steps for stage 1 (full dataset): {steps_full}")
    print(f"Steps for stage 2 (fine dataset): {steps_fine}")
    print(f"Total steps: {total_steps}")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        use_rslora=True,
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=0.1,
        bias='none',
        task_type='CAUSAL_LM'
    )

    model = get_peft_model(model, lora_config)

    if report_to == 'wandb':
        wandb.login(key=wandb_api_key)
        # wandb.init(project='stresSLM-sft', name=args.experiment_name)
        wandb.init(project='stressLM-qwen-sft', entity='iy-space', name=args.experiment_name)
    
    # print trainable parameters
    print(f'Model: {model}')
    model.print_trainable_parameters()
    # storage_client
    experiment_folder = f'{CURRENT_FOLDER}/experiments/{args.experiment_name}'
    storage_client = FileStorage(storage_path=experiment_folder)
    storage_client.create_path_if_not_exists(run_name)

    def create_trainer(train_dataset, max_steps, save_steps=50, eval_steps=50, callback=None, save_total_limit=2):
        training_args = TrainingArguments(
            output_dir=f'{experiment_folder}/{run_name}',
            per_device_train_batch_size=configs['batch_size'],
            per_device_eval_batch_size=configs['batch_size'],
            gradient_accumulation_steps=configs['gradient_accumulation_steps'],
            max_steps=max_steps,
            warmup_steps=int(0.05 * total_steps),
            weight_decay=0.01,
            logging_dir=f'{experiment_folder}/{run_name}/logs',
            logging_steps=50,
            eval_on_start=True,
            eval_strategy='steps',
            eval_steps=eval_steps,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            learning_rate=configs['lr'],
            lr_scheduler_type=configs['scheduler'],
            remove_unused_columns=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False},
            run_name=run_name,
            report_to=report_to,
        )
        callbacks_kwarg = {'callbacks': [callback]} if callback else {}

        return Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dataset_full['test'],
            data_collator=StressDataCollator(processor),
            tokenizer=processor.feature_extractor,
            **callbacks_kwarg
        )
    
    print('Training...')
    total_times_to_eval = 20

    # Stage 1
    print('Stage 1')
    eval_steps = total_steps // total_times_to_eval
    trainer_stage_1 = create_trainer(
        train_dataset=dataset_full['train'], 
        max_steps=total_steps, 
        save_steps=eval_steps,
        eval_steps=eval_steps,
        save_total_limit=3,
        callback=StopAfterNStepsCallback(steps_full)
    )
    trainer_stage_1.train()
    print(f'steps_full: {steps_full}')
    
    # Stage 2 — resume from checkpoint for continued cosine schedule
    print('Stage 2')
    # resume from checkpoint
    checkpoint_path = os.path.join(trainer_stage_1.args.output_dir, f"checkpoint-{steps_full}")
    assert os.path.exists(checkpoint_path), f"Checkpoint path {checkpoint_path} does not exist"
    trainer_stage_2 = create_trainer(
        train_dataset=dataset_fine['train'],
        max_steps=total_steps,
        save_steps=eval_steps,
        eval_steps=eval_steps,
        save_total_limit=3,
        callback=EvalAndSaveAtEndCallback(),
    )
    print(f'Resuming from checkpoint path: {checkpoint_path}')
    trainer_stage_2.train(resume_from_checkpoint=checkpoint_path)

    best_checkpoint = trainer_stage_2.state.best_model_checkpoint
    checkpoint_pointer = os.path.join(experiment_folder, run_name, "best_checkpoint.txt")
    with open(checkpoint_pointer, "w") as f:
        f.write(best_checkpoint)
    print(f"Best checkpoint: {best_checkpoint}")

    if report_to == 'wandb':
        wandb.finish()
   

if __name__=='__main__':
    train()
