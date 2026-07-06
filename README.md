# Resume Roast Studio

A polished, opinionated project for generating savage, technically aware roasts of resumes and job descriptions using a fine-tuned Llama-based model.

## What this project does

This repository packages a LoRA-adapted language model that can critique resumes with sharp humor and technical specificity. It is designed for experimentation, local inference, and future expansion into a web app or API.

## Repository structure

- [README.md](README.md) — project overview and usage
- [requirements.txt](requirements.txt) — Python dependencies
- [data/](data/) — dataset files such as [data/Dataset.csv](data/Dataset.csv)
- [notebooks/](notebooks/) — training and exploration notebooks
- [scripts/](scripts/) — runnable inference scripts
- [roaster_v1/](roaster_v1/) — fine-tuned adapter and tokenizer assets

## Quick start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the inference script:
   ```bash
   python scripts/roast_resume.py "I built a React app, used Docker, and claimed to know AWS."
   ```

3. For notebook-based exploration, open the files in [notebooks/](notebooks/).

## Model details

- Base model: unsloth/llama-3-8b-bnb-4bit
- Fine-tuning approach: LoRA / PEFT
- Quantization: 4-bit
- Intended use: humorous, critical resume analysis

## Notes

This project is intentionally stylized and should be used responsibly. The model is meant for entertainment, experimentation, and creative evaluation rather than professional hiring decisions.
