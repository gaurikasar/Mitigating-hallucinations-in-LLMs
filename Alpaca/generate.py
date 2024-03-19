import argparse
import os

import sys

import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig,BitsAndBytesConfig, LlamaForCausalLM, LlamaTokenizer,AutoModelForCausalLM
import sys
sys.path.append('/content/drive/MyDrive/Hallucination_LLMs/alpaca-lora-main')
from utils.prompter import Prompter
#from utils import prompter

import jsonlines
from tqdm import tqdm
import time
n = os.getcwd().split('/')[2]


def generate_step(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        num_beams=1,
        early_stopping=True,
    )

    # Without streaming
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=512,
            
        )
    s = generation_output.sequences[0][len(input_ids[0]):]
    output = tokenizer.decode(s)
    return output

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str)
parser.add_argument("--out_file", type=str)
args = parser.parse_args()

device = "cuda"
load_8bit = True
base_model = 'baffo32/decapoda-research-llama-7B-hf'
lora_weights = 'tloen/alpaca-lora-7b'
tokenizer = LlamaTokenizer.from_pretrained(base_model)

'''model = LlamaForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=load_8bit,
    torch_dtype=torch.float16,
    device_map="auto",
)
print("MODEL",model)

model = PeftModel.from_pretrained(
    model,
    lora_weights,
    torch_dtype=torch.float16,
)'''
m = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            ),    torch_dtype=torch.bfloat16,
    device_map={"": 0}
)
'''model = PeftModel.from_pretrained(
    m,
    lora_weights,
    torch_dtype=torch.float16,
)'''
model = PeftModel.from_pretrained(m, 'riverallzero/alpaca-lora-7b')
# unwind broken decapoda-research config
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2
# if load_8bit:
#     model.half()
model.eval()
model = torch.compile(model)

out_file = args.out_file

# Read input data in batches of 10 records
batch_size = 10
with jsonlines.open(args.input_file) as reader:
    records = list(reader)
    for i in tqdm(range(0, len(records), batch_size)):
        batch = records[i:i+batch_size]
        for line in batch:
            question = line['question']
            prompter_instance = Prompter('')
            instruction = "Answer the following question."
            prompt = prompter_instance.generate_prompt(instruction, question)
            response = generate_step(prompt)
            line.update({'generated_answer': response})

        # Write batch to output file
        with jsonlines.open(out_file, mode='a') as writer:
            for line in batch:
                writer.write(line)
