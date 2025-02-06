llm='gpt-4o-mini' # choose from 'gpt-4o-mini', 'gpt-3.5-turbo'
strategy='baseline' # choose from 'baseline', 'baseline-binary', 'active-reasoning', 'active-reasoning-binary'

for seed in 0 1 2
do
    for i in 64 154 5 6 17 26 33 36 38 39 41 50 54 55 64 70 81 96 106 109 147 93 118 101 143 121 134 139 141 122 82 115 77 98 90 138 74 95 110 123 111 154 114 91 103 107 76 159 73
    do 
    python src/code-generation/active_code_generation.py 'seed='${seed} 'strategy='${strategy} 'task_id=HumanEval/'${i} 'llm='${llm}
    done
done