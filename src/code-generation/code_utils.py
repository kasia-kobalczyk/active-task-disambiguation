import re
from _execution import time_limit
import numpy as np

def _extract_bracketed_string(s):
    stack = []
    start = -1
    for i, c in enumerate(s):
        if c == '(':
            stack.append(i)
            if len(stack) == 1:
                start = i
        elif c == ')':
            stack.pop()
            if len(stack) == 0:
                return s[start+1:i]
    return ''

def _get_function_input(s, entry_point, include_entry_point=True):
    expr = re.compile(f'{entry_point}\((.*)') 
    if re.search(expr, s):
        inputs = re.search(expr, s).group(0)
        # extract everything until the closing bracket 
        inputs = _extract_bracketed_string(inputs)
        if include_entry_point:
            inputs = f'{entry_point}({inputs})'
    else:
        inputs = "error"
    return inputs

def _extract_inputs(assertions, entry_point, include_entry_point=True):
    inputs = [
        _get_function_input(s, entry_point, include_entry_point=include_entry_point)
        for s in assertions
    ]
    return inputs

def _get_program_str(code_sample, entry_point, prompt):
    if '```python' in code_sample:
        code_sample = code_sample.split('```python')[1]
        code_sample = code_sample.split('```')[0]

    expr = rf"def {entry_point}(.*)\n"
    completion = re.split(expr, code_sample)[-1]
    program_str = prompt + completion
    return program_str


def to_str(o):
    try:
        with time_limit(1):
            str_o = str(o)
            return str_o
    except:
        return 'to_str timed out'

def get_entropy_from_counts(counts):
    print(counts)
    counts = np.array(counts)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs))
    return entropy

def remove_spaces(word):
    word = re.sub(r'^\s+', '', word)
    word = re.sub(r'\s+$', '', word)
    return word