LLM Fine-Tuning: Sarcastic Resume Roaster

Fine-tunes a Large Language Model (LLM) to generate sarcastic responses for resumes.

Dataset included (CSV with Context and Response)

Uses PEFT + QLoRA (4-bit quantization)

Model: unsloth/llama-3-8b-bnb-4bit

Training parameters:

Batch size: 1

Gradient accumulation: 4

Epochs: 2

Learning rate: 2e-4

Optimizer: paged_adamw_8bit

Gradient checkpointing: enabled

Output: fine-tuned model saved for inference

Requirements
pip install torch transformers peft bitsandbytes accelerate datasets trl

