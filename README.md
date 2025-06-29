# StressTest
**Official repository of the paper:**

*StressTest: Can YOUR Speech LM Handle the Stress?*

<p align="center">
    🌐 <a href="https://pages.cs.huji.ac.il/adiyoss-lab/stresstest/" target="_blank">Project</a> | 📃 <a href="https://arxiv.org/abs/2505.22765" target="_blank">Paper</a> | 🤗 <a href="https://huggingface.co/datasets/slprl/StressTest" target="_blank">StressTest Dataset </a><br> | 🤗 <a href="https://huggingface.co/slprl/StresSLM" target="_blank">StresSLM Model</a>
</p>




This repository provides code for evaluating **Sentence Stress Detection (SSD)** and **Sentence Stress Reasoning (SSR)** on ***StressTest*** benchmark. 

It includes:

* Evaluation of our proposed model **StresSLM**.
* Examples to run evaluation with two additional models.

It also includes **Stress-17K** training data loading and augmentation script used to train **StresSLM**.

<p align="center">
  <img src="imgs/main_fig1.png" alt="StressTest Overview" width="100%" />
</p>

---

## 🚀 Getting Started

### 🔧 Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/slp-rl/StressTest.git
cd StressTest
pip install -r requirements.txt
```

---

## 📊 Evaluation


### ✅ Running the Evaluations

We evaluate models using our judgment-based protocol. You’ll need an OpenAI API key for the judge (e.g., GPT-4) evaluation. Set the key as an environment variable:

```bash
export OPENAI_API_KEY=your_openai_api_key
```

altenatively, you can set the key in the `stresstest/evaluation/configs.py` file:

```python
class Settings(BaseSettings):
    OPENAI_API_KEY: str = "your_openai_api_key"
```

Then run the evaluation script:

```bash
python -m stresstest.evaluation.main \
    --task ssr \
    --model_to_evaluate stresslm
```

You can change the `--task` flag to `ssd` for the Sentence Stress Detection task.
`--model_to_evaluate` can be one of the following `["stresslm", "qwen2audio", "gpt-4o-audio", "mock"]`.

the script will create a `results/` directory at the project root to store evaluation outputs.
The expected project structure is:

```
StressTest
├── infra
├── stresstest
│   └── evaluation
└── results
```


---

### 🤔 Evaluating Your Own Model

To evaluate your own model, implement it using the following interface and place it under the stresstest/evaluation/src/inference directory:
```python
from abc import ABC, abstractmethod

class InferenceClientBase(ABC):

    @abstractmethod
    def prepare(self, *args, **kwargs) -> dict:
        """
        Prepare method to be implemented by subclasses. 
        This method should return a dictionary with the necessary inputs for the predict method.
        The returned ditionary is handled by the evaluation script.
        """
        pass

    @abstractmethod
    def predict(self, *args, **kwargs) -> str:
        """Predict method to be implemented by subclasses."""
        pass
```

Then, register your model by updating the configs.py and clients.py files in the stresstest/evaluation folder. Make sure your new model is included as a valid option for the --model_to_evaluate argument.

---

## 🏋️‍♂️ Training

We release:

* The synthetic training data `Stress-17K` used to train StresSLM (released).
* The training script for finetuning on SSD and SSR (coming soon).

Stay tuned!

### 🧪 Synthetic Training Data — `Stress-17K`

We release `Stress-17K`, a synthetic dataset generated via our proposed pipeline. It supports multi-task instruction tuning across four task types to improve performance on SSD and SSR tasks.

The raw pre-augmented dataset is available on 🤗 Hugging Face under: [`slprl/Stress-17K-raw`](https://huggingface.co/datasets/slprl/Stress-17K-raw-pre-augmentation) and is automatically downloaded by the augmentation script.

#### 🔄 Usage Example

You can use the `DatasetAugmentation` class to load, structure, and augment the data:

```python
from data_augmentation import DatasetAugmentation

data_augmentation = DatasetAugmentation(n_proc=8)
data_augmentation.train_test_split(test_size=0.15)
data_augmentation.prepare_structure_for_augmentation()
data_augmentation.augment_with_training_prompts(tasks='all')
augmented_dataset = data_augmentation.get_augmented_dataset()
```

The augmentation utilities are available under:

```
StressTest
├── infra
├── stresstest
│   └── training
│       └── stress_17k
```

Each sample can be augmented into multiple instruction-following formats defined in a YAML configuration. This YAML file is also located in the `stress_17k` directory and can be edited to add new tasks or modify existing ones.

---

## 📖 Citation

If you use this work, please cite our paper:

```bibtex
@misc{yosha2025stresstest,
      title={StressTest: Can YOUR Speech LM Handle the Stress?}, 
      author={Iddo Yosha and Gallil Maimon and Yossi Adi},
      year={2025},
      eprint={2505.22765},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.22765}, 
}
```
