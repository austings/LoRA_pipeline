# LoRA_pipeline
The goal of this project is to provide a lightweight, modular system for fine-tuning and serving LLMs using techniques such as LoRA and QLoRA, enabling efficient training and deployment without full model retraining.

##  Scope
Implementation of LoRA-based fine-tuning workflows using Hugging Face PEFT
Support for quantized training approaches (QLoRA) to reduce memory footprint
Exploration of high-throughput and single-GPU fine-tuning strategies
Integration considerations for serving multiple adapters (e.g. via vLLM)
Early-stage research into scalable adapter usage and deployment patterns
Purpose

This project is part of an ongoing effort to:

Reduce compute and memory requirements for LLM fine-tuning
Enable rapid experimentation with domain-specific adaptations
Bridge research (LoRA, QLoRA, S-LoRA, etc.) with practical production pipelines
Status

## Early-stage / research-driven.
The repository currently serves as a foundation for experimentation, benchmarking, and iterative development of efficient LLM fine-tuning systems.

## Research
### 🤗 PEFT: State-of-the-art Parameter-Efficient Fine-Tuning. 
https://github.com/huggingface/peft
https://huggingface.co/docs/peft/en/package_reference/lora

### Fine-tuning LLMs with PEFT and LoRA
https://www.youtube.com/watch?v=Us5ZFp16PaU

### LORA: LOW-RANK ADAPTATION OF LARGE  LANGUAGE MODELS	6 Oct 2021	
https://arxiv.org/pdf/2106.09685
### QLORA: Efficient Finetuning of Quantized LLMs	23 May 2023	
https://arxiv.org/pdf/2305.14314
### ASPEN: High-Throughput LoRA Fine-Tuning of Large  Language Models with a Single GPU 	5 Dec 2023	
https://arxiv.org/pdf/2312.02515

### LoRA+: Efficient Low Rank Adaptation of Large Models 19 Feb 2024
https://arxiv.org/abs/2402.12354

### S-LORA: SERVING THOUSANDS OF CONCURRENT LORA ADAPTERS 7 Nov 2023
https://arxiv.org/pdf/2311.03285

### Using LoRA adapters
This document shows you how to use LoRA adapters with vLLM 
https://docs.vllm.ai/en/latest/models/lora.html

# wasm
https://github.com/ktock/container2wasm
