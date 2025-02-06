import openai
import numpy as np
import os
import re
import json
import time
import random
import os

OPENAI_API_TYPE = "azure"
OPENAI_API_VERSION = "2024-02-15-preview"

# For GPT-3.5-turbo
OPENAI_API_BASE =  ""
OPENAI_API_KEY =  ""
OPENAI_API_ENGINE = ""

# For GPT-4o-mini
OPEN_API_BASE_GPT4o_mini    = ""
OPEN_API_KEY_GPT4o_mini     = ""
OPEN_API_ENGINE_GPT4o_mini  = ""

MAX_RETRY = 5

def get_dir_used(strategy, hypothesis_type, hypothesis_decision, SEED):
    this_dir = f"./results/20Q/{hypothesis_type}/{hypothesis_decision}/{strategy}/{OPENAI_API_ENGINE}/iter_{SEED}"
    create_directory(this_dir)
    return this_dir

def create_directory(directory_path):
    """
    Create directory or multiple directories given a string representing the path.
    
    Args:
        directory_path (str): The directory path to be created.
    
    Returns:
        bool: True if directory creation successful, False otherwise.
    """
    try:
        # Create the directory or directories
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
        return True
    except OSError as e:
        # Directory already exists or permission error
        print(f"Failed to create directory '{directory_path}'. Error: {e}")
        return False

def save_json(data, file_path):
    # Writing JSON data to the file in a readable format
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)  # The `indent` parameter formats the JSON with indentation for readability

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def remove_special_characters(word):
        return re.sub(r'[^\w\s]', '', word)

def chat_gpt(
        user_prompt=None, 
        system_prompt=None, 
        n_used=1,
        logprobs=False,
        seed=None,
        model_name=['gpt-3.5-turbo', 'gpt-4', 'gpt-4o-mini'],
        temperature=0.7,
        top_p=0.95
    ):
    if model_name == 'gpt-3.5-turbo':
        model = OPENAI_API_ENGINE
        api_key = OPENAI_API_KEY
        api_base = OPENAI_API_BASE
        api_version = OPENAI_API_VERSION
    elif model_name == 'gpt-4o-mini':
        model = OPEN_API_ENGINE_GPT4o_mini
        api_key = OPEN_API_KEY_GPT4o_mini
        api_base = OPEN_API_BASE_GPT4o_mini
        api_version = OPENAI_API_VERSION

    success = False
    it = 0
    if seed is None:
        seed = np.random.randint(0, 100000)
    while not success and it < MAX_RETRY:
        it += 1
        client = openai.AzureOpenAI(
            azure_endpoint=api_base,
            api_key=api_key,
            api_version=api_version
        )
        if system_prompt is None:
            system_prompt = "You are an AI assistant that helps people find information."
        message_text = [{"role":"system","content": system_prompt}]
        if user_prompt:
            message_text.append({"role":"user", "content": user_prompt})

        response = client.chat.completions.create(
            model=model,
            messages = message_text,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            n=n_used,
            seed=seed
        )
        try:
            response.choices[0].message.content is not None
            success = True
            return response
        
        except:
            sleep_time = random.uniform(0.5, 1.5)
            time.sleep(sleep_time)

    return response


def llama(
        user_prompt=None, 
        system_prompt=None, 
        n_used=1,
        logprobs=False,
        seed=None,
        temperature=0.7,
        top_p=0.95,
        llm_name='llama-3-70B'
    ):
    success = False
    it = 0

    if llm_name == 'llama-3-70B':
        llm_name_used = 'meta-llama/Meta-Llama-3-70B-Instruct'
    elif llm_name == 'llama-3-8B':
        llm_name_used = 'meta-llama/Meta-Llama-3-8B-Instruct'

    while not success and it < MAX_RETRY:
        it += 1
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8000/v1"

        client = openai.OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )

        if system_prompt is None:
            system_prompt = "You are an AI assistant that helps people find information."
        message_text = [{"role":"system","content": system_prompt}]
        if user_prompt:
            message_text.append({"role":"user", "content": user_prompt})
        print("---------- Waiting-------------")
        response = client.chat.completions.create(
                model=llm_name_used,
                messages=message_text,
                n=n_used,
                temperature=temperature,
                top_p=top_p,
                logprobs=logprobs,
                seed=seed,
            )
        try:
            response.choices[0].message.content is not None
            success = True
            print("---------- PASS -------------")
        except:
            print("---------- NOT PASS -------------")
            pass

    return response



def obtain_cost(resp):
    if OPENAI_API_ENGINE == "gpt4-32k_20230830":
        prompt_cost = 0.01
        completion_cost = 0.03
    elif OPENAI_API_ENGINE == "gpt4-32k_20230815":
        prompt_cost = 0.0015
        completion_cost = 0.002
    elif OPENAI_API_ENGINE == "SWNorth-gpt-35-turbo-0613-20231016":
        prompt_cost = 0.0005
        completion_cost = 0.0015
    elif OPENAI_API_ENGINE == "SWNorth-gpt-4-0613-20231016":
        prompt_cost = 0.01
        completion_cost = 0.03
    else:
        prompt_cost = 0
        completion_cost = 0
    
    return prompt_cost*(resp.usage.prompt_tokens/1000) + completion_cost*(resp.usage.completion_tokens/1000)


