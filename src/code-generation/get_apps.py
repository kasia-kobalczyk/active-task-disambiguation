#%%
from datasets import load_dataset, concatenate_datasets
import json
import re
import pandas as pd
import sys
from code_utils import _get_program_str
from _execution import get_test_case_result, get_expected_outputs
from tqdm import tqdm

sys.set_int_max_str_digits(100000)

dataset = load_dataset("codeparrot/apps")
dataset = concatenate_datasets([dataset["train"], dataset["test"]])

#%%
processed_dataset = []

for sample in dataset:
    if sample['starter_code'] != '':
        task_id = sample['problem_id']
        prompt = sample['question'] 
        source = sample['url']
        # match between www. and .com
        try:
            source = re.search('www\.(.*)\.com', source).group(1)
        except:
            source = re.search('//(.*)\.com', source).group(1)
        entry_point = sample['starter_code']

        solutions = json.loads(sample['solutions'])

        if 'class Solution' in solutions[0]:
            pass

        try:
            entry_point = re.search('def\s+(\w+)\s*\(', entry_point).group(1)
        except:
            pass

        if sample['input_output'] != '':
            test_inputs = json.loads(sample['input_output'])['inputs']
            test_outputs = json.loads(sample['input_output'])['outputs']
        else:
            test_inputs = None
            test_outputs = None

        processed_dataset.append({
            'task_id': task_id,
            'prompt': prompt,
            'starter_code': sample['starter_code'],
            'entry_point': entry_point,
            'solutions': solutions,
            'source': source,
            'test_inputs': test_inputs,
            'test_outputs': test_outputs,
        })

df = pd.DataFrame(processed_dataset)
#%%
df[(~df.prompt.str.lower().str.contains('example'))]['source'].value_counts()

# %%
sample_df = df[
    (~df.prompt.str.lower().str.contains('example'))
    & (df.source == 'codewars')
].dropna().copy()
sample_df = sample_df[
    sample_df.test_inputs.apply(lambda x: len(x) >= 5)
]
sample_df['prompt'] = '"""\n' "=====Problem statement====\n\n"+ sample_df['prompt'] + '\n"""\n\n' + '# ====== Solution ======\n\n'
#%%
# construct test cases
def construct_test(test_inputs, test_outputs):
    test = "def check(candidate):"
    for i, o in zip(test_inputs, test_outputs):
        if type(o[0]) == str:
            o = f"'{o[0]}'"
        else:
            o = o[0]
        i = [str(x) if type(x) != str else f"'{x}'" for x in i]
        test += f"\n    assert candidate({','.join(i)}) == {o}"
    test += '\n\n'
    return test

sample_df['test'] = sample_df.apply(lambda x: construct_test(x['test_inputs'], x['test_outputs']), axis=1)

#%%
# select canonical solutions that pass all test cases
def get_canonical_solution(solutions, entry_point, test):
    for solution in solutions:
        passed = get_test_case_result(
            program_str=solution, 
            test_case=test, 
            entry_point=entry_point, 
            timeout=1,
        )
        if passed == True:
            return solution
    return None

# #%%
# i = 3
# sample_solution = sample_df.iloc[i]['solutions'][1]
# program_str = sample_solution
# test = sample_df.iloc[i]['test']
# print(sample_solution)
# print(program_str)
# print(test)
# get_test_case_result(
#     program_str=program_str,
#     test_case=test,
#     entry_point=sample_df.iloc[i]['entry_point'],
#     timeout=1
# )


#%%
tqdm.pandas()

sample_df['canonical_solution'] = sample_df.progress_apply(lambda x: get_canonical_solution(x['solutions'], x['entry_point'], x['test']), axis=1)

#%%
sample_df['canonical_solution'].notnull().sum()
#%%
sample_df.dropna(inplace=True)
#%%
def get_context(prompt, entry_point, canonical_solution):
    expr = rf"def {entry_point}(.*)\n"
    completion = re.split(expr, canonical_solution)[-1]
    context = canonical_solution.replace(completion, '')
    return prompt + context 

sample_df['prompt_with_context'] = sample_df.apply(
    lambda x: get_context(x['prompt'], x['entry_point'], x['canonical_solution']), 
    axis=1
)

def check_canonical_solution(solution, entry_point, test, prompt):
    program_str = _get_program_str(
        code_sample=solution,
        entry_point=entry_point,
        prompt=prompt,
    )
    passed = get_test_case_result(
        program_str=program_str,
        test_case=test, 
        entry_point=entry_point, 
        timeout=1,
    )
    return passed

sample_df['canonical_solution_check'] = sample_df.progress_apply(
    lambda x: check_canonical_solution(x['canonical_solution'], x['entry_point'], x['test'], x['prompt_with_context']), axis=1
)

(sample_df['canonical_solution_check'] == True).sum()

sample_df = sample_df[sample_df['canonical_solution_check'] == True]
# %%
save_df = sample_df[[
    'task_id', 'prompt_with_context', 'entry_point', 'test', 'canonical_solution'
]].dropna().rename({
    'prompt_with_context': 'prompt',
}, axis=1)

save_df.to_json('../../data/code-generation/APPS_codewars.jsonl', orient='records', lines=True)


# %%
program_str = _get_program_str(
    code_sample=sample_df.iloc[0]['canonical_solution'],
    entry_point=sample_df.iloc[0]['entry_point'],
    prompt=sample_df.iloc[0]['prompt_with_context']
)

inputs_str_ls = ['who_is_winner(["A_Yellow", "B_Red", "C_Yellow", "C_Red", "C_Yellow", "D_Yellow", "A_Red", "E_Red"])']

get_expected_outputs(
    program_str=program_str,
    inputs_str_ls=inputs_str_ls,
    timeout=1
)
# %%
get_test_case_result(
    program_str=program_str,
    test_case=sample_df.iloc[0]['test'],
    entry_point=sample_df.iloc[0]['entry_point'],
    timeout=1
)
# %%
' '.join([str(x) for x in save_df['task_id'].unique()[:20]])
# %%
list(save_df['task_id'].unique()[:20])
# %%

df = pd.read_csv('../../data/code-generation/APPS_codewars_zero_shot_results.csv')
df.accuracy.hist()
# %%
import numpy as np

task_ids = df[(df.accuracy <= 0.6) & (df.accuracy > 0.0)]['task_id'].values
len(task_ids)

# %%
' '.join([str(t) for t in task_ids])
# %%
