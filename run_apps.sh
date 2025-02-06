llm='llama-3-8B' # choose from 'gpt-4o-mini', 'gpt-3.5-turbo' 'llama-3-8B' 'llama-3-70B'
strategy='active-reasoning' # choose from 'baseline', 'baseline-binary', 'active-reasoning', 'active-reasoning-binary'
dataset_path='./data/code-generation/APPS_codewars.jsonl'
save_dir='code-generation/APPS-codewars'

for seed in 0 1 2
do
    for i in 4317 2717 2939 3072 3452 3477 3536 3715 4084 4190 4293 4315 4513 1614 2664 2671 2681 2717 2728 2881 2927 2939 2991 3016 3068 3072 3079 3220 3248 3278 3366 3443 3452 3477 3536 3589 3594 3598 3689 3706 3715 3786 3822 3856 3958 4084 4128 4182 4190 4240 4293 4315 4317 4326 4353 4360 4414 4438 4513 4546
    do 
    python src/code-generation/active_code_generation.py 'seed='${seed} 'strategy='${strategy} 'task_id='${i} 'llm='${llm} 'dataset_path='${dataset_path} 'save_dir='${save_dir}
    done
done