import jsonlines
import time
import openai
from tqdm import tqdm
import argparse

# Set your API key
openai.api_key = ""

# Set the model you want to use
model_engine = "gpt-3.5-turbo"

# Function to read data from a JSONL file
def read_jsonl(file_path):
    data = []
    with jsonlines.open(file_path, 'r') as reader:
        for line in reader:
            data.append(line)
    return data

# Generate a response from the model
def get_response(prompt):
    completion = openai.chat.completions.create(
        model=model_engine,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    # Print the response
    response = completion.choices[0].message.content
    return response

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str)
parser.add_argument("--out_file", type=str)
args = parser.parse_args()

# Read input data from JSONL file
source = read_jsonl(args.input_file)

# Generate responses and write to output file
with jsonlines.open(args.out_file, 'w') as writer:
    for line in tqdm(source, desc="Generating responses"):
        try:
            response = get_response(line['question'])
            line['generated_answer'] = response
            writer.write(line)
        except Exception as e:
            print(f"Error generating response for question: {line['question']}. Error: {e}")