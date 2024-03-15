Steps:
1. Clone this repository - https://github.com/gaurikasar/Mitigating-hallucinations-in-LLMs.git
2. Implemented all the commands and files to run on collab- open colabi on google colab
3. Follow the steps mentioned in the collab:
    1. Mount your drive:
    from google.colab import drive
       drive.mount('/content/drive')
    2. Navigate to ChatGPT folder
       import os
       os.chdir('/content/drive/MyDrive/Self_Reflection_Medical-main/ChatGPT')
    3. Install the necessary libraries
    4. Prepare the Dataset
       !python /content/drive/MyDrive/Self_Reflection_Medical-main/dataset/wiki_dataset/process_data.py
    5. Install Pytorch libraries
       !pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
    6. Run the Baseline Model
       !CUDA_VISIBLE_DEVICES=0 python generate.py  --input_file '/content/drive/MyDrive/Self_Reflection_Medical-main/dataset/wiki_dataset/val_data.jsonl' --out_file output.jsonl
    7. Run the Self-Reflection model
       !CUDA_VISIBLE_DEVICES=0 python3 /content/drive/MyDrive/Self_Reflection_Medical-main/ChatGPT/loop.py --input-file /content/drive/MyDrive/Self_Reflection_Medical-main/dataset/wiki_dataset/val_data.jsonl --    sources 'wiki' --out-dir /content/drive/MyDrive/Self_Reflection_Medical-main/ChatGPT/loop_output --max-loop 3 --max-knowledge-loop 3 --max-response-loop 3 --demo-num 0 --threshold-entailment 0.8 --threshold-fact -1 --threshold-consistency -5

   


