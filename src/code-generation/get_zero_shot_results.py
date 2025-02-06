from reasoners import ActiveCodeReasoner, OracleCodeAnswerer
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils import chat_gpt
from tqdm import tqdm
import pandas as pd

dataset_path = './data/code-generation/APPS_codewars.jsonl'

def gpt_llm_call(user_prompt, system_prompt, n_used, seed, logprobs=False):
    return chat_gpt(
        user_prompt=user_prompt, 
        system_prompt=system_prompt,
        n_used=n_used,
        logprobs=logprobs,
        seed=seed,
        temperature=1,
        top_p=0.95,
        model_name='gpt-4o-mini'
    )
llm_call = gpt_llm_call

data = {}
with open(dataset_path, 'r') as file:
    for line in file:
        content = json.loads(line)
        id = content['task_id']
        content.pop('task_id')
        data[id] = content

results_dict = {}

for task_id in tqdm(list(data.keys())):
    task_data = data[task_id]
    reasoner = ActiveCodeReasoner(
        llm_call, 
        task_data, 
        seed=0, 
        mode="run", 
        logprobs=False, 
        unique_hs=False,
        total_hypothesis=10,
        total_questions=5
    )
    oracle = OracleCodeAnswerer(
        llm_call, 
        task_data, 
        seed=0
    )

    hypothesis = reasoner.generate_hypothesis({})

    if len(hypothesis) < 1:
        continue
    
    results = oracle.get_ground_truth_test_results(hypothesis)
    results = [1 if r == True else 0 for r in results]
    if len(results) < 1:
        acc = 0
    else:
        acc = sum(results) / len(results)
    
    results_dict[task_id] = acc

pd.DataFrame(results_dict.items(), columns=['task_id', 'accuracy']).to_csv(
    './data/code-generation/APPS_codewars_zero_shot_results.csv', index=False
)