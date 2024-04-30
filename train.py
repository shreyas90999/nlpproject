from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch
from huggingface_hub import notebook_login
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training,PeftModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import DatasetDict
import random
import argparse

#Load Dataset

dataset = load_dataset("ruslanmv/ai-medical-chatbot") # Medical QA dataset

#get args from command line

args = argparse.ArgumentParser()
args.add_argument("--sample", type=int, default=2000) # Number of samples to train on

sample = args.sample
#generate random indices for train and test dataset
indices = random.sample(range(sample), sample)
test_indices = random.sample(range(sample), sample*0.2)

dataset_dict = {"train": dataset["train"].select(indices),
                "test": dataset["train"].select(test_indices)}

raw_datasets = DatasetDict(dataset_dict)

################################################################################
# bitsandbytes parameters
################################################################################
bnb_4bit_compute_dtype = "float16"
bnb_config = BitsAndBytesConfig(
    # Activate 4-bit precision base model loading
    load_in_4bit=True,
    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type="nf4",
    # Load tokenizer and model with QLoRA configuration
    
    bnb_4bit_compute_dtype=getattr(torch, bnb_4bit_compute_dtype),
    # Activate nested quantization for 4-bit base models (double quantization)
    bnb_4bit_use_double_quant=True,
)

# The model to train from the Hugging Face hub
model_name = "mistralai/Mistral-7B-Instruct-v0.2" # After multiple experiments this was giving best results

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = True
model.config.pretraining_tp = 1

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#Apply Chat Template to data
def apply_chat_template(example, tokenizer):

    messages = [
        {"role": "user", "content": example['Patient']},
          {"role": "assistant", "content":example["Doctor"]}

    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)

    return {"text":text}

column_names = list(raw_datasets["train"].features)
raw_datasets= raw_datasets.map(apply_chat_template,
                                fn_kwargs={"tokenizer": tokenizer},
                                remove_columns=column_names,
                                desc="Applying chat template",)
#Load train and test dataset
train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["test"]

# Prepare the model for training
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
)
model = get_peft_model(model, peft_config)
# Model Training Arguments 
training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=250,
    logging_steps=250,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="wandb"
)
# SFT Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    max_seq_length= None,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
)
#Start Training
trainer.train()
trainer.save_model("./fine_tuned_model")