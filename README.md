# TinyLlama Finance Assistant using LoRA Fine-Tuning

A domain-adapted Small Language Model (SLM) fine-tuned for Financial Question Answering using TinyLlama and the FIQA dataset.

---

# Overview

This project demonstrates parameter-efficient fine-tuning of a Small Language Model (SLM) using LoRA (Low-Rank Adaptation) on the FIQA financial question-answering dataset.

The objective was to adapt TinyLlama toward finance-related conversational understanding while maintaining lightweight training suitable for Google Colab environments.

---

# Features

* Financial Question Answering
* TinyLlama 1.1B Fine-Tuning
* LoRA-based Parameter Efficient Training
* Hugging Face Transformers Workflow
* Google Colab Compatible
* Lightweight Inference Pipeline
* NLP Evaluation Metrics

---

# Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* PEFT / LoRA
* Hugging Face Datasets
* Evaluate
* Google Colab

---

# Model Information

| Component          | Details                      |
| ------------------ | ---------------------------- |
| Base Model         | TinyLlama-1.1B-Chat-v1.0     |
| Fine-Tuning Method | LoRA                         |
| Dataset            | FIQA                         |
| Domain             | Financial Question Answering |
| Training Platform  | Google Colab                 |

---

# Project Workflow

FIQA Dataset
↓
Tokenization
↓
TinyLlama Base Model
↓
LoRA Fine-Tuning
↓
Finance Domain SLM
↓
Evaluation (ROUGE / BERTScore)

---

# Evaluation Metrics

| Metric       | Score |
| ------------ | ----- |
| ROUGE-1      | 0.208 |
| ROUGE-2      | 0.023 |
| ROUGE-L      | 0.104 |
| BERTScore F1 | 0.827 |

---

# Key Observations

* The model successfully adapted toward financial question answering.
* Strong semantic similarity was achieved according to BERTScore evaluation.
* LoRA enabled efficient fine-tuning on limited hardware.
* Some repetition was observed during long-form generation due to small model size.

---

# Sample Output

### Question

What is inflation?

### Generated Answer

Inflation is the increase in the price of goods and services over time, reducing purchasing power and affecting economic stability.
---

# Installation

```bash
pip install -r requirements.txt
```

---

# Run Inference

```bash
python inference.py
```

---

# Future Improvements

* Retrieval-Augmented Generation (RAG)
* QLoRA Optimization
* Larger Financial Datasets
* Better Decoding Strategies
* API Deployment using FastAPI
* Gradio Web Interface

---

# Author
Yuva Shankar Narayana
