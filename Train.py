
!pip install -U bitsandbytes
!pip install -U transformers
!pip install -U peft
!pip install -U accelerate
!pip install -U trl
!pip install -U datasets
!pip install pypdf
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import os
os.environ["ACCELERATE_MIXED_PRECISION"] = "fp16"
import os
os.environ["ACCELERATE_MIXED_PRECISION"] = "no"




csv_file = "/content/Dataset.csv"
cv_col = "Context"
roast_col = "Response"
model_id = "unsloth/llama-3-8b-bnb-4bit"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ------------------------------------------
# 2. TOKENIZER
# ------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)


# ------------------------------------------
# 4. PREPARE MODEL FOR QLoRA
# ------------------------------------------
model.config.use_cache = False
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True
)



peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
)

model = get_peft_model(model, peft_config)

dataset = load_dataset(
    "csv",
    data_files=csv_file,
    split="train",
)

def format_prompt(example):
    return (
        "### CV Context:\n"
        + example[cv_col]
        + "\n\n### Roast:\n"
        + example[roast_col]
    )


from trl import SFTConfig

sft_config = SFTConfig(
    output_dir="roaster_v1",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=2,
    logging_steps=1,
    save_strategy="no",

    fp16=False,   # ❌ disable AMP
    bf16=False,   # ❌ disable BF16

    optim="paged_adamw_8bit",
    report_to="none",
)


trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    formatting_func=format_prompt,
)


print("🚀 Starting Training...")
trainer.train()
output_dir = "roaster_v1"
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("✅ Training Finished & Saved!")


print("✅ Training Finished!")
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pypdf import PdfReader

# ==========================================
# CONFIGURATION
# ==========================================
BASE_MODEL = "unsloth/llama-3-8b-bnb-4bit"
LORA_PATH = "roaster_v1"
CV_PDF_PATH = ""

# ==========================================
# 1. LOAD MODEL (GPU FORCED)
# ==========================================
print("⏳ Loading model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map={"": 0},
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()


print("📄 Reading PDF...")
cv_text = ""
try:
    reader = PdfReader(CV_PDF_PATH)
    for page in reader.pages:
        text = page.extract_text()
        if text:
            cv_text += text + "\n"
except Exception as e:
    print(f"❌ Error: {e}")


if len(cv_text) > 2000:
    start_idx = random.randint(500, len(cv_text) - 1500) # Skip header, pick middle
    end_idx = start_idx + 1500 # Grab 1500 chars (enough for context)
    selected_context = cv_text[start_idx:end_idx]
    print(f"🎲 Randomly selected text from char {start_idx} to {end_idx}...")
else:
    selected_context = cv_text # Use whole thing if short

print("-" * 30)
print(f"👀 MODEL FOCUSING ON:\n{selected_context[:200]}...") # Preview
print("-" * 30)


prompt = f"""### CV Context:
...{selected_context}...

### Roast:
Here is a brutal critique of the specific projects and skills mentioned above:
1."""

# ==========================================
# 4. GENERATE
# ==========================================
print("🔥 Generating Roast...")

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=2048,
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        min_new_tokens=100,
        max_new_tokens=512,
        temperature=0.7, # Lower temp = sticks to facts better
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

print("\n" + "="*30)
print("       THE RANDOMIZED ROAST")
print("="*30)
print("1." + response)
print("="*30)