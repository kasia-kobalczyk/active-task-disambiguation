import json
import sys
import os
currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
rootdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir) 
from utils import chat_gpt, llama
import hydra
from omegaconf import DictConfig, OmegaConf
from utils import save_json, create_directory
from _execution import *
from reasoners import *


class ReasonerHandler():
    def __init__(self, Reasoner, OracleAnswerer, max_iter, seed, save_path):
        self.Reasoner = Reasoner
        self.OracleAnswerer = OracleAnswerer
    
        self.max_iter = max_iter
        self.seed = seed

        self.all_requirements = {}
        self.all_questions = {}
        self.selected_questions = {}
        self.listed_hypothesis = {}
        self.restricted_questions = [] 
        
        self.iter = 0
        self.save_path = save_path

        
    def run(self):
        self.Reasoner.total_cost = 0
        self.OracleAnswerer.total_cost = 0

        while True:
            if self.iter == self.max_iter:
                break
        
            # Step 1. Generate hypothesis
            print('Sampling hypothesis...')
            hypothesis = self.Reasoner.generate_hypothesis(self.all_requirements)
            self.listed_hypothesis[self.iter] = [h.__dict__ for h in hypothesis]
            print(f'Sampled {len(hypothesis)} hypothesis.')

            # Step 2. Generate questions
            print('Sampling questions...')
            questions = self.Reasoner.generate_questions(
                requirements=self.all_requirements, 
                restricted_questions=self.restricted_questions + list(self.selected_questions.values()), 
            )
            self.all_questions[self.iter] = questions
            print(f'Candidate questions: {questions}')

            # Step 3. Select best question
            best_question = self.Reasoner.select_best_question(questions, hypothesis)
            print(f'Selected question: {best_question}')
            self.selected_questions[self.iter] = best_question

            # Step 4. Answer question
            answer = self.OracleAnswerer.answer(best_question)
            print(f"Oracle answer: {answer}")

            if answer == '"error"' or answer == "error":
                self.restricted_questions.append(best_question)
             
            self.all_requirements[self.iter] = self.Reasoner.q_a_to_requirement(best_question, answer)
            
            self.iter += 1


        # Save the results
        save_json(self.all_requirements, self.save_path + '/requirements.json')
        save_json(self.all_questions, self.save_path + '/questions.json')
        save_json(self.selected_questions, self.save_path + '/questions_selected.json')
        save_json(self.listed_hypothesis, self.save_path + '/listed_hypothesis.json')

        print("Total cost:", self.Reasoner.total_cost + self.OracleAnswerer.total_cost)   


    def evaluate(self):
        self.all_requirements = json.load(open(f"{self.save_path}/requirements.json"))
        program_correctness = {}
        all_eval_program_samples = {}
        self.Reasoner.total_hypothesis = 20
        self.Reasoner.total_cost = 0
        self.OracleAnswerer.total_cost = 0
        self.Reasoner.mode = 'eval'
        self.OracleAnswerer.mode = 'eval'
        
        for seed in range(3):
            all_eval_program_samples[seed] = {}
            program_correctness[seed] = {}
            requirements = {}
            print("========================")
            print(f"Starting evaluation no. {seed}")
            iter = 0    
            self.Reasoner.seed = seed    
            h_ls = self.Reasoner.generate_hypothesis(requirements)
            result = self.OracleAnswerer.get_ground_truth_test_results(h_ls)
            program_correctness[seed][iter] = [to_str(r) for r in result]
            n_correct = len([r for r in result if r == True])
            print("Iteration 0:")
            print(f"Correct / Total: {n_correct}/{len(h_ls)} \n")
            all_eval_program_samples[seed][iter] = [h.content for h in h_ls]
            
            for iter in self.all_requirements.keys():
                print("Iteration:", int(iter) + 1)
                requirements[str(int(iter) + 1)] = self.all_requirements[iter]
                h_ls = self.Reasoner.generate_hypothesis(requirements)
                result = self.OracleAnswerer.get_ground_truth_test_results(h_ls)
                program_correctness[seed][str(int(iter) + 1)] = [to_str(r) for r in result]
                n_correct = len([r for r in result if r == True])
                print(f"Correct / Total: {n_correct}/{len(h_ls)} \n")
                all_eval_program_samples[seed][str(int(iter) + 1)] = [h.content for h in h_ls]

         
                save_json(all_eval_program_samples, f"{self.save_path}/eval_program_samples.json")
                save_json(program_correctness, f"{self.save_path}/eval_program_correctness.json")

        print("==============================")
        print("Total evaluation cost:", self.Reasoner.total_cost + self.OracleAnswerer.total_cost)


@hydra.main(version_base=None, config_path=f"{rootdir}/config", config_name="main_code_generation")
def main(cfg: DictConfig) -> None:

    data = {}
    with open(cfg.dataset_path, 'r') as file:
        for line in file:
            content = json.loads(line)
            id = content['task_id']
            content.pop('task_id')
            data[id] = content

    task_data = data[cfg.task_id]
    
    if cfg.llm in ['gpt-3.5-turbo', 'gpt-4o-mini']:
        def gpt_llm_call(user_prompt, system_prompt, n_used, seed, logprobs=False):
            return chat_gpt(
                user_prompt=user_prompt, 
                system_prompt=system_prompt,
                n_used=n_used,
                logprobs=logprobs,
                seed=seed,
                temperature=1,
                top_p=0.95,
                model_name=cfg.llm
            )
        llm_call = gpt_llm_call
    elif cfg.llm in ['llama-3-70B', 'llama-3-8B']:
        def llama_llm_call(user_prompt, system_prompt, n_used, seed, logprobs=False):
            return llama(
                user_prompt=user_prompt, 
                system_prompt=system_prompt,
                n_used=n_used,
                seed=seed,
                logprobs=logprobs,
                temperature=1,
                top_p=0.95,
                llm_name=cfg.llm
            )
        llm_call = llama_llm_call

    else:
        raise ValueError('Invalid LLM model')

    if cfg.strategy == 'active-reasoning':
        Reasoner = ActiveCodeReasoner(
            llm_call=llm_call,
            task_data=task_data,
            total_questions=cfg.total_questions,
            total_hypothesis=cfg.total_hypothesis,
            seed=cfg.seed,
            logprobs=False,
            unique_hs=True
        )
        OracleAnswerer = OracleCodeAnswerer(
            llm_call=llm_call,
            task_data=task_data,
            seed=cfg.seed
        )
    
    elif cfg.strategy == 'active-reasoning-binary':
        Reasoner = ActiveBinaryCodeReasoner(
            llm_call=llm_call,
            task_data=task_data,
            total_questions=cfg.total_questions,
            total_hypothesis=cfg.total_hypothesis,
            seed=cfg.seed,
            logprobs=False,
            unique_hs=True,
            
        )
        OracleAnswerer = OracleBinaryCodeAnswerer(
            llm_call=llm_call,
            task_data=task_data,
            seed=cfg.seed
        )
    
    elif cfg.strategy == 'baseline':
        Reasoner = ActiveCodeReasonerBase(
            llm_call=llm_call,
            task_data=task_data,
            seed=cfg.seed
        )
        OracleAnswerer = OracleCodeAnswerer(
            llm_call=llm_call,
            task_data=task_data,
            seed=cfg.seed
        )
    
        
    elif cfg.strategy == 'baseline-binary':
        Reasoner = ActiveBinaryCodeReasonerBase(
            llm_call=llm_call,
            task_data=task_data,
            seed=cfg.seed
        )
        OracleAnswerer = OracleBinaryCodeAnswerer(
            llm_call=llm_call,
            task_data=task_data,
            seed=cfg.seed
        )

    
    else:
        raise ValueError(f"Invalid strategy {cfg.strategy}")



    save_path=f"./results/{cfg.save_dir}/{cfg.task_id}/{cfg.strategy}/{cfg.llm}/iter_{cfg.seed}"

    handler = ReasonerHandler(
        Reasoner=Reasoner,
        OracleAnswerer=OracleAnswerer,
        max_iter=cfg.max_iter,
        seed=cfg.seed,
        save_path=save_path,
    )

    create_directory(save_path)
    with open(f"{save_path}/config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    handler.run()
    handler.evaluate()


if __name__ == '__main__':
    os.environ['HYDRA_FULL_ERROR'] = '1'
    main()


    
