# Active Task Disambiguation

## Code Generation

This repository contains the code used for the code generation experiments presented in the paper.

`run_human_eval.sh` and `run_apps.sh` contain the command that should be run to reproduce the experimental results on the HumanEval and APPS benchmarks

We also provide the generated programs, queries, and their evaluation results with GPT-3.5-turbo and GPT-4o-mini in `results`. Results can be analysed in `analyze_code_results.ipynb`

In order to run code generation with OpenAI models, the openai API keys should be first inserted in `src/utils.py`

In order to run code generation with open source models from the huggingface library, we recommend setting up a local entrypoint via vllm, e.g.
`python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --tensor-parallel-size=1 --enforce-eager --disable-custom-all-reduce  --host localhost --port 8000`  

`src/code-generation/active_code_generation.py` is the main file used for running the experiments

`src/code-generation/reasoners.py` implements two classess of Base and Active problem-solving agents generating binary or open queries.

`src/code-generation/_execution.py` contains code for exectuing LLM-generated programs in a sandbox enivronment

