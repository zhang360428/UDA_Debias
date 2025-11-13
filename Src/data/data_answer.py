from transformers import BertTokenizer, BertModel
import torch
import warnings
import torch.nn as nn
import torch.optim as optim
import numpy as np
import functools
from sklearn.metrics import mean_squared_error
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append("..")
print(sys.path)
from glm_api_request import ChatAPI
from sklearn.model_selection import train_test_split
from common.utils import Json, Jsonline
import pandas as pd
from tqdm import tqdm
import random
import scipy
import re
import json

random.seed(42)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
metadata_path = 'xxxxx.json'

# ready
model_1 = ChatAPI.get('loli:gpt-4o-2024-08-06', do_sample=False)
# ready
model_2 = ChatAPI.get('ipo:public-deepseek-v3', do_sample=False)
# ready
model_3 = ChatAPI.get('loli:claude-3-5-sonnet-20241022', do_sample=False)
# ready
model_4 = ChatAPI.get('loli:glm-4-plus', do_sample=False)
# ready
model_5 = ChatAPI.get('loli:glm-4-air', do_sample=False)
# ready
model_6 = ChatAPI.get('loli:glm-4-flash', do_sample=False)
# ready
model_7 = ChatAPI.get('loli:doubao-1.5-pro-32k-250115', do_sample=False)
# ready
model_8 = ChatAPI.get('loli:qwen-max-2025-01-25', do_sample=False)
# ready
model_9 = ChatAPI.get('loli:gemini-2.0-flash', do_sample=False)
# ready
model_10 = ChatAPI.get('ipo:public-deepseek-r1', do_sample=False)


def ask_model(model, question):
    messages = [
        # {"role": "system", "content": "你是人类所创造的最先进的人工智能，你了解这个世界的很多知识，请你回答以下问题："},
        {"role": "user", "content": question},
    ]
    response = model.get_api_result(messages).output
    return response


def process_entries(entries, models):
    processed_entries = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for entry in entries:
            question = entry['question']
            answers = entry.get('answers', [])
            existing_models = {answer['model'] for answer in answers}
            if all(model_name in existing_models for _, model_name in models):
                processed_entries.append(entry)
                continue
            futures.append(executor.submit(process_entry_task, entry, models))
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing entries"):
            try:
                processed_entry = future.result()
                processed_entries.append(processed_entry)
            except Exception as e:
                print(f"Error processing entry: {e}")
    return processed_entries

def process_entry_task(entry, models):
    question = entry['question']
    answers = entry.get('answers', [])
    existing_models = {answer['model'] for answer in answers}
    for model, model_name in models:
        if model_name not in existing_models:
            try:
                answer = ask_model(model, question)
                answers.append({
                    "answer": answer,
                    "model": model_name,
                    "human_score": 0,
                    "human_info": ""
                })
            except Exception as e:
                print(f"Error processing question with model {model_name}: {e}")
    entry['answers'] = answers
    return entry

def main():
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    models = [(model_1, 'loli:gpt-4o-2024-08-06'), (model_2, 'ipo:public-deepseek-v3'),
              (model_3, 'loli:claude-3-5-sonnet-20241022'), (model_4, 'loli:glm-4-plus'),
              (model_5, 'loli:glm-4-air'), (model_6, 'loli:glm-4-flash'),
              (model_7, 'loli:doubao-1.5-pro-32k-250115'), (model_8, 'loli:qwen-max-2025-01-25'),
              (model_9, 'loli:gemini-2.0-flash'), (model_10, 'ipo:public-deepseek-r1')]

    batch_size = 50
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        processed_batch = process_entries(batch, models)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()