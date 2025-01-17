!pip install -q transformers datasets accelerate bitsandbytes wandb
!pip install -q git+https://github.com/huggingface/peft.git
!pip install -q huggingface_hub
!pip install trl

import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTTrainer
from huggingface_hub import notebook_login
import shutil

print("Please enter your Hugging Face token when prompted.")
notebook_login()

MODEL_NAME = "microsoft/phi-2"
DATASET_NAME = "Cynaptics/persona-chat"
OUTPUT_DIR = "phi2_finetuned"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def preprocess_data(dataset):
    def format_dialog(row):
        persona = " ".join(row["persona_b"])
        dialogue = " ".join(row["dialogue"])
        
        # Format for Phi-2 loading
        return {
            "text": f"Instruct: Given this persona: {persona[:256]}\nand this dialogue: {dialogue[:256]}\ngenerate a response.\nOutput: {row['reference']}"
        }
    
    processed = [format_dialog(row) for row in dataset["train"]]
    return Dataset.from_list(processed)

dataset = load_dataset(DATASET_NAME)
train_dataset = preprocess_data(dataset)

train_dataset

train_dataset['text'][:5]

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors=None
    )

tokenized_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.config.use_cache = False

lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "dense"
    ]
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    num_train_epochs=1,
    warmup_ratio=0.1,
    logging_steps=10,
    save_steps=100,
    fp16=True,
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    lr_scheduler_type="cosine",
    evaluation_strategy="no",
    save_total_limit=2,
    report_to="tensorboard",
    max_steps=500
)

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    tokenizer=tokenizer,
    peft_config=lora_config,
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

test_prompt = (
    "Persona: I love hiking and outdoor activities.\n"
    "Dialogue: What's your favorite hiking trail?\n"
    "Response:"
)

inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=50, temperature=1.0,top_k=50,top_p=0.9,repetition_penalty=1.2)
print("Generated response:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

folder_to_zip = '/kaggle/working/phi2_finetuned'
zip_path = '/kaggle/working/phi2_finetuned.zip'
shutil.make_archive(zip_path.replace('.zip', ''), 'zip', folder_to_zip)

print(f"Folder zipped at: {zip_path}")