# QwenMed

QwenMed is a Large Language Model (LLM) (Qwen3-1.7B) fine-tuned on the medical datasets for medical QA tasks.

---

## Features

* **Medical Question Answering**: Provides responses to medical queries.
* **Reasoning Capability**: Can generate detailed thought processes (Chain-of-Thought) before providing a concise answer.
* **Efficient Fine-tuning**: Unsloth library for PEFT.
* **Hugging Face Integration**: Models are pushed to and loaded from Hugging Face Hub. [HERE](https://huggingface.co/Thiraput01/QwenMed-1.7B-Reasoning) (Note: Only adapters)


## Usage
The `notebooks/QwenMed_inference.ipynb` notebook demonstrates how to load the model and perform inference. 
You can switch between non-thinking and reasoning modes to observe different response styles.

**For Non-Thinking**
```python
from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Thiraput01/QwenMed-1.7B-Reasoning",
    max_seq_length = 2048,
    load_in_4bit = True,
)

messages = [
    {"role" : "user", "content" : "I'm having chest pain, what could be the cause?"}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True,
    enable_thinking = False, # Disable thinking
)

_ = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),
    max_new_tokens = 256,
    temperature = 0.7, top_p = 0.8, top_k = 20,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)
```


**For Thinking (Reasoning) inference**
```python
from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Thiraput01/QwenMed-1.7B-Reasoning",
    max_seq_length = 2048,
    load_in_4bit = True,
)

messages = [
    {"role" : "user", "content" : "I'm having a headace, what do you think?"}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize = False,
    add_generation_prompt = True,
    enable_thinking = True, # Enable thinking
)

_ = model.generate(
    **tokenizer(text, return_tensors = "pt").to("cuda"),
    max_new_tokens = 2048,
    temperature = 0.6, top_p = 0.95, top_k = 20,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)
```


# Training
The `notebooks/QwenMed_train.ipynb` notebook provides the process for fine-tuning the QwenMed model.


# Dataset Used

- [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
- [Laurent1/MedQuad-MedicalQnADataset_128tokens_max](https://huggingface.co/datasets/Laurent1/MedQuad-MedicalQnADataset_128tokens_max)


# Training Configuration
- Base Model: `unsloth/Qwen3-1.7B-unsloth-bnb-4bit`
- Epochs: `2`
- Learning Rate: `5e-5`
- Scheduler: `CosineAnnealingLR`
- LoRA Parameters: `r=32, alpha=64`
- Batch Size: `2`
- Gradient Accumulation Steps = `8`


# Results
**Training results**

![train](https://raw.githubusercontent.com/Thiraput01/QwenMed/main/result/train_graph.png)

**Validation results**

![eval](https://raw.githubusercontent.com/Thiraput01/QwenMed/main/result/eval_graph.png)
