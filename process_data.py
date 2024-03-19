import json
import jsonlines
import random    

dataset1_questions = set()
with jsonlines.open('/content/drive/MyDrive/Self_Reflection_Medical-main/dataset/wiki_dataset/qa_data.json', 'r') as f:
    for record in f:
        dataset1_questions.add(record['question'])

# Load the JSON file
with open('/content/drive/MyDrive/Self_Reflection_Medical-main/dataset/wiki_dataset/hotpot_train_v1.1.json') as f:
    train_data = json.load(f)

# Shuffle the dataset
random.shuffle(train_data)

# Select the first 1000 records
selected_records = train_data[:100]

# Write the selected records to the JSONL file
with jsonlines.open('/content/drive/MyDrive/Self_Reflection_Medical-main/dataset/wiki_dataset/val_data.jsonl', mode='w') as writer:
    for record in selected_records:
        writer.write({
            'id': record['_id'],
            'url': '',
            'generated_answer': '',
            'topic': '',
            'context': record['context'],
            'question': record['question'],
            'answer': record['answer'],
        })

print("100 records added to val_data.jsonl")
