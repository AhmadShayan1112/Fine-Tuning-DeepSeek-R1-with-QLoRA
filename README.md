# 🔧 Fine-Tuning DeepSeek R1 with QLoRA

This project demonstrates how to fine-tune the **DeepSeek R1 language model** using **QLoRA**, a memory-efficient fine-tuning technique. The goal is to enable scalable and cost-effective fine-tuning of large language models with limited hardware resources like a single NVIDIA P100 GPU.

---

## 🚀 Overview

DeepSeek R1 is a powerful open-source LLM. In this notebook, we:
- Load and configure the DeepSeek R1 base model.
- Prepare a custom dataset.
- Apply quantization-aware fine-tuning using QLoRA.
- Train the model efficiently using PEFT (Parameter-Efficient Fine-Tuning).
- Evaluate and save the fine-tuned model.

---

## 🛠️ Features

- ✅ Uses Hugging Face Transformers and PEFT libraries.
- ✅ Supports quantized training with `bitsandbytes`.
- ✅ Streamlined training pipeline for fast prototyping.
- ✅ Tokenization using DeepSeek tokenizer.
- ✅ Trainer integration with mixed precision (fp16).

---

## 📂 Project Structure

```
📁 fine-tuning-deepseek
│
├── fine-tunning-deep-seek-r1.ipynb   # Jupyter notebook for fine-tuning
├── README.md                         # Project documentation (this file)
```

---

## 📦 Dependencies

Install required packages with:

```bash
pip install torch datasets transformers accelerate peft bitsandbytes
```

Or, use the provided environment configuration from the notebook.

---

## 📊 Dataset

The notebook uses a simple synthetic dataset for demonstration:

```python
{
  "prompt": "What is 2 + 2?",
  "completion": "2 + 2 = 4"
}
```

To fine-tune on your own data, format your dataset similarly and update the loading section accordingly.

---

## 🏋️‍ Training

We use Hugging Face's `Trainer` API with QLoRA for efficient fine-tuning. Key configurations:
- LoRA with rank `r=8`, alpha `16`
- `bnb_4bit_quant_type='nf4'`
- `bnb_4bit_use_double_quant=True`

Training can be launched via:

```python
trainer.train()
```

The model checkpoints are saved to `./deepseek-r1-finetuned`.

---

## 📊 Evaluation

After training, the model is tested with:

```python
model.eval()
input_ids = tokenizer("Test prompt", return_tensors="pt").input_ids.to(device)
output = model.generate(input_ids, max_new_tokens=100)
```

---

## 💾 Saving & Loading the Model

```python
model.save_pretrained('./deepseek-r1-finetuned')
tokenizer.save_pretrained('./deepseek-r1-finetuned')
```

You can later reload with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('./deepseek-r1-finetuned')
tokenizer = AutoTokenizer.from_pretrained('./deepseek-r1-finetuned')
```

---

## 📌 Notes

- Ensure your GPU supports 4-bit quantization (e.g., NVIDIA P100 or better).
- For better performance, consider using a larger dataset or running longer epochs.
- You can switch to DeepSpeed or FSDP for distributed training in multi-GPU setups.

---

## 🧠 References

- [DeepSeek R1 Model Card](https://huggingface.co/deepseek-ai/deepseek-llm-7b-base)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

---

## 📬 Contact

Feel free to open an issue or reach out if you have any questions or suggestions!

---

