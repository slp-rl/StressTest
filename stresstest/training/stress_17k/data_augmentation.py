from datasets import load_dataset, concatenate_datasets, DatasetDict
import re
from typing import Self
from pathlib import Path
from training_prompter import PromptTemplateManager

CURRENT_DIR = Path(__file__).resolve().parent
TRAINING_PROMPTS_PATH = f"{CURRENT_DIR}/training_prompts.yml"
DATASET_NAME = "slprl/Stress-17K-raw"

class DatasetAugmentation:
    def __init__(self, n_proc=1):
        self.dataset: DatasetDict = load_dataset(DATASET_NAME)
        self.augmented_dataset = DatasetDict()
        self.prompter = PromptTemplateManager(TRAINING_PROMPTS_PATH)
        self.n_proc = n_proc  # Number of processes for parallel processing
    
    def prepare_structure_for_augmentation(self) -> Self:
        def convert_example(example):
            # Extract individual answers
            answer_1 = example['possible_answers'][0]
            answer_2 = example['possible_answers'][1]
            
            # Determine the correct label and correct answer
            label = example['label']
            correct_answer = example['possible_answers'][label]
            answer_label = label + 1  # because prompt uses 1-based indexing

            # Tokenize the transcription
            transcription = re.sub(r'\b(\w+)-(\w+)\b', r'\1 \2', example['transcription'])
            words = transcription.strip().split()

            # Get emphasized words based on `gt_stress_indices`
            stress_indices = example['gt_stress_indices']
            try:
                emphasized_words = [words[i] for i in stress_indices]
            except IndexError as e:
                print(f"IndexError: {stress_indices} out of range for words: {words}")
                print(f'Example: {example}')
                raise e
            emphasized_string = ', '.join(emphasized_words)

            return {
                'ds_name': 'stress-ds',
                'answer_1': answer_1,
                'answer_2': answer_2,
                'answer_label': answer_label,
                'correct_answer': correct_answer,
                'description': example['description'],
                'emphasized_words': emphasized_string,
                'transcription': example['transcription'],
                'audio': example['audio'],
                'audio_id': example['audio_id'],
                'metadata': example['metadata'],
                'interpretation_id': example['interpretation_id'],
                'transcription_id': example['transcription_id']
            }
        self.dataset = self.dataset.map(convert_example, desc='Preparing dataset sample structure for training prompts', num_proc=self.n_proc)
        return self
            
    def augment_with_training_prompts(self, tasks='all', cols_to_keep=['ds_name', 'task', 'prompt_id', 'question', 'answer', 'audio', 'audio_id', 'interpretation_id', 'transcription_id']) -> Self:
        all_tasks = self.prompter.get_tasks()
        if tasks == 'all':
            tasks = all_tasks
        elif not all(task in all_tasks for task in tasks):
            raise ValueError(f'Task not found in the prompt templates')
        prompt_ids = self.prompter.get_prompts_by_tasks(tasks)
        # concatenate the dataset with the augmented dataset for each prompt
        self.augmented_dataset['train_full'] = concatenate_datasets([self.dataset['train_full'].map(
            self.prompter.render, fn_kwargs={"prompt_id": prompt_id}, desc=f'mapping prompt {prompt_id}', num_proc=self.n_proc) for prompt_id in prompt_ids])
        self.augmented_dataset['train_fine'] = concatenate_datasets([self.dataset['train_fine'].map(
            self.prompter.render, fn_kwargs={"prompt_id": prompt_id}, desc=f'mapping prompt {prompt_id}', num_proc=self.n_proc) for prompt_id in prompt_ids])
        # use only end 2 end prompt for test set
        self.augmented_dataset['test'] = concatenate_datasets([self.dataset['test'].map(
            self.prompter.render, fn_kwargs={"prompt_id": prompt_id}, desc=f'mapping prompt {prompt_id}', num_proc=self.n_proc) for prompt_id in [1]]
        )
        
        remove_columns = [col for col in self.augmented_dataset['train_full'].column_names if col not in cols_to_keep]
        self.augmented_dataset['train_full'] = self.augmented_dataset['train_full'].remove_columns(remove_columns)
        self.augmented_dataset['train_fine'] = self.augmented_dataset['train_fine'].remove_columns(remove_columns)
        self.augmented_dataset['test'] = self.augmented_dataset['test'].remove_columns(remove_columns)
        return self
    
    def get_dataset(self) -> DatasetDict:
        return self.dataset
    
    def get_augmented_dataset(self) -> DatasetDict:
        return self.augmented_dataset
    
    def get_full_dataset(self) -> DatasetDict:
        return DatasetDict({
            'train': self.augmented_dataset['train_full'],
            'test': self.augmented_dataset['test']
        })

    def get_fine_dataset(self) -> DatasetDict:
        return DatasetDict({
            'train': self.augmented_dataset['train_fine'],
            'test': self.augmented_dataset['test']
        })
    
    def train_test_split(self, test_size=0.2, shuffle=True, **train_test_split_kwargs) -> Self:
        dataset_full = DatasetDict({
            'train': self.dataset['train_full']
        })
        dataset_filtered = DatasetDict({
            'train': self.dataset['train_fine']
        })
        # Split the filtered dataset into train and test sets
        filtered_dataset = dataset_filtered['train'].train_test_split(test_size=test_size, shuffle=shuffle, seed=42, **train_test_split_kwargs)
        test_audio_ids = filtered_dataset['test']['audio_id']
        # Filter out test audio_ids from full train dataset
        dataset_full = dataset_full['train'].filter(lambda x: x['audio_id'] not in test_audio_ids, desc="Filtering full dataset by test audio_ids")
        self.dataset = DatasetDict({
            'train_full': dataset_full,
            'train_fine': filtered_dataset['train'],
            'test': filtered_dataset['test']
        })
        return self
    
    def shuffle_augmented_dataset(self) -> Self:
        self.augmented_dataset['train_full'] = self.augmented_dataset['train_full'].shuffle(seed=42)
        self.augmented_dataset['train_fine'] = self.augmented_dataset['train_fine'].shuffle(seed=42)
        return self
