#!/usr/bin/env python3
"""Generate a sharp roast for a resume or job description."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


MODEL_DIR = Path(__file__).resolve().parents[1] / "roaster_v1"


def build_pipeline(model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Roast a resume or job description with the fine-tuned model")
    parser.add_argument("text", nargs="+", help="The resume text or job description to roast")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    prompt = "You are an elite technical resume roast engine. Critique the following resume or job description with wit, precision, and unfiltered honesty.\n\n" + " ".join(args.text)

    generator = build_pipeline(MODEL_DIR)
    output = generator(
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=True,
        top_p=0.95,
        return_full_text=False,
    )
    print(output[0]["generated_text"].strip())


if __name__ == "__main__":
    main()
