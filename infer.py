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
import pandas as pd
args = argparse.ArgumentParser()
args.add_argument("--modelPath", type=str, default="mistralai/Mistral-7B-Instruct-v0.2") # Number of samples to train on

#Load Dataset
dataset = load_dataset("ruslanmv/ai-medical-chatbot") # Medical QA dataset

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
#load Model
# The model to train from the Hugging Face hub

model_name = args.modelPath

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

#Generate response
def generate(prompts):
    messages = [
    {"role": "user", "content":prompts}]
    model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
    generated_ids = model.generate(model_inputs, max_new_tokens=1024, do_sample=True)
    results = tokenizer.batch_decode(generated_ids)
    return results

# Use mistral model withut fine tuning
csv  = {"id":[],
        "patient":[],
        "doctor":[],
        "model_response":[]}
count = 0
for i in range(500):
    csv["id"].append(count)
    csv["patient"].append(dataset["Patient"][i])
    csv["doctor"].append(dataset["Doctor"][i])
    csv["model_response"].append(generate(dataset["Patient"][i]))
    count+=1
    print(count)
    df = pd.DataFrame(csv)
    df.to_csv("data.csv")