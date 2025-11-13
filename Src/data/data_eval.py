import sys

sys.path.append("..")
from glm_api_request import ChatAPI
from common.utils import Json, Jsonline
from concurrent.futures import ThreadPoolExecutor
import ast
import pandas as pd
import functools
from tqdm import tqdm
import random
import scipy
import re
import json
import os

random.seed(42)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_response(payload: dict, model_name: str) -> dict:
    payload = payload.copy()
    model = ChatAPI.get(model_name, do_sample=False, temperature=0)
    i = 4
    while i > 0:
        try:
            messages = payload['messages']
            response = model.get_api_result(messages).output
            payload['judge_response'] = response

            json_score, score = process_score(response)
            score = score * 2.5 if score is not None else None
            if score is not None:
                payload['json_score'] = json_score
                payload['judge_score'] = score
                payload['diff_score'] = abs(score - payload['human_score'])
                payload['judge_model_name'] = model_name
                del payload['messages']
                return payload
            else:
                i -= 1
                print(f"Failed to get valid score for payload {payload['qid']}, retrying...")
                continue
        except Exception as e:
            i -= 1
            print(f"Error occurred: {e}, retrying...")
    del payload['messages']
    payload['judge_response'] = None
    payload['judge_score'] = None
    payload['diff_score'] = 100
    payload['judge_model_name'] = model_name
    return payload


def process_score(judge_response: str):
    try:
        json_str = re.findall(r'\{.*?\}', judge_response, re.DOTALL)[-1]
        json_str = json_str.replace("\\n", "")
        json_str = json_str.replace("\\\"", "\"")
        json_scores = ast.literal_eval(json_str)
        score = json_scores['overall']
        return json_scores, score
    except Exception as e:
        print(f"Failed to process score: {e}")
        return None, None


def prepare_payloads(category: str, prompt_version: str, metadata_path: str) -> list:
    questions = Json.load(metadata_path)
    payloads = []

    questions_filtered = [q for q in questions if category in q['categories']]

    prompts = Json.load('test.json')[prompt_version.split('-')[0]][prompt_version.split('-')[1]]
    sys_template_reference, user_prompt_reference = prompts['system_prompt'], prompts["user_prompt"]
    sys_template_no_reference, user_prompt_no_reference = prompts.get('system_prompt_no_reference',
                                                                      sys_template_reference), prompts.get(
        'user_prompt_no_reference', user_prompt_reference)

    for line in questions_filtered:
        qid = line['qid']
        sys_prompt = line.get('system_prompt', '')
        question = line['question']
        if sys_prompt:
            question = sys_prompt + "\n\n" + question
        reference = line['reference']
        if reference:
            sys_template = sys_template_reference
            user_prompt = user_prompt_reference
        else:
            sys_template = sys_template_no_reference
            user_prompt = user_prompt_no_reference

        for answer in line['answers']:
            content = user_prompt.replace('{prompt}', question).replace('{response}', answer['answer']).replace(
                '{reference}', reference)
            messages = [
                {"role": "user", "content": sys_template + content},
            ]
            item = {
                "qid": qid,
                "question": question,
                "reference": reference,
                "messages": messages,
                "human_score": answer['human_score'],
                "category": category,
                "prompt_version": prompt_version,
                "answer_model": answer['model']
            }
            item.update(answer)
            payloads.append(item)

    print(f"Total {len(payloads)} payloads prepared for category {category} and prompt_version {prompt_version}.")
    return payloads


def run_task_for_category(category: str, prompt_version: str, judge_models: list, metadata_path: str, num_threads: int):
    file_name = metadata_path.split('/')[-1].split('.')[0]
    payloads = prepare_payloads(category, prompt_version, metadata_path)
    results = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for model_name in judge_models:
            for payload in payloads:
                futures.append(executor.submit(get_response, payload.copy(), model_name))

        for future in tqdm(futures, desc=f"Processing models for category {category}", ncols=100):
            results.append(future.result())

    save_path = f'xxxxx/results_{file_name}_{prompt_version}.json'
    Json.dump(results, save_path)
    print(f"Results saved to {save_path}.")
    return results


def run_all_tasks(categories: list, prompt_versions: list, judge_models: list, metadata_path: str, num_threads: int):
    for i in range(len(categories)):
        category = categories[i]
        prompt_version = prompt_versions[i]
        print(f"Processing {category} with prompt {prompt_version}...")
        run_task_for_category(category, prompt_version, judge_models, metadata_path, num_threads)
        print(f"Finished {category}.")


if __name__ == "__main__":
    categories = ['arena-hard-v0.1']
    prompt_versions = ['default-v1_no_ref']
    judge_models = ['loli:gpt-4o-2024-08-06',
                    'loli:deepseek-chat',
                    'loli:claude-3-5-sonnet-20241022',
                    'loli:glm-4-plus',
                    'loli:glm-4-air',
                    'loli:glm-4-flash',
                    'loli:doubao-1.5-pro-32k-250115',
                    'loli:qwen-max-2025-01-25',
                    'loli:gemini-2.0-flash',
                    'loli:deepseek-reasoner']
    metadata_path = 'xxxxx.json'
    num_threads = 8

    run_all_tasks(categories, prompt_versions, judge_models, metadata_path, num_threads)