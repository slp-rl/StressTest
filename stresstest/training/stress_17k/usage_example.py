from data_augmentation import DatasetAugmentation

if __name__=="__main__":
    # Initialize the DataAugmentation class with the number of processes.
    # This will download the pre-augmented dataset from hf
    data_augmentation = DatasetAugmentation(n_proc=8)
    
    print("Pre-augmented dataset loaded.")
    print(data_augmentation.dataset)
    
    # Train test split the dataset to create train_full, train_fine, and test (validation) datasets.
    data_augmentation.train_test_split(test_size=0.15)
    
    # Prepare the structure of the data to match the training prompts variables
    data_augmentation.prepare_structure_for_augmentation()
    
    # Augment the dataset with SSR and SSD training prompts.
    # The training prompts are loaded from the "training_prompts.yml" YAML file.
    # Each training prompt is associated with a task -
    # ["e2e_stress_meaning", "elaborated_explanation", "cascade_reasoning", "stress_detection"] 
    # You can specify which tasks to augment with by passing a list of task names,
    # e.g. tasks=['e2e_stress_meaning', 'elaborated_explanation']
    # Or use 'all' to augment with all tasks.
    # You can also update the YML file to add new tasks or modify existing ones.
    data_augmentation.augment_with_training_prompts(tasks='all')
    
    print("Dataset augmented with training prompts.")
    print("Augmented dataset:")
    print(data_augmentation.get_augmented_dataset())

