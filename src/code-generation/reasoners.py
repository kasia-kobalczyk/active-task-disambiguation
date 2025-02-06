import sys
import os
import re
import multiprocess as mp
import pandas as pd
import numpy as np

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
rootdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir)
from utils import chat_gpt
from code_utils import (
    _extract_inputs,
    _get_program_str,
    to_str,
    get_entropy_from_counts,
    remove_spaces,
)
from utils import obtain_cost
from _execution import *


class Hypothesis:
    def __init__(self, content, logp=None):
        self.content = content
        self.logp = logp


class OracleCodeAnswerer:
    def __init__(self, llm_call, task_data, seed, mode="run"):
        self.llm_call = llm_call
        self.task_data = task_data
        self.timeout = 0.1
        self.total_cost = 0
        self.seed = seed
        self.mode = mode

    def answer(self, question):
        program_str = _get_program_str(
            self.task_data["canonical_solution"],
            self.task_data["entry_point"],
            self.task_data["prompt"],
        )
        #program_str = self.task_data["prompt"] + completion
        outputs = get_expected_outputs(
            program_str, [question], timeout=self.timeout
        )
        output = outputs[0]
        if type(output) == str:
            true_output = f'"{output}"'
        else:
            true_output = to_str(output)
        return true_output

    def get_ground_truth_test_results(self, hypothesis):
        test = self.task_data["test"]
        entry_point = self.task_data["entry_point"]
        results = []
        for h in hypothesis:
            program_str = _get_program_str(
                h.content, entry_point, self.task_data["prompt"]
            )
            result = get_test_case_result(program_str, test, entry_point, 0.5)
            results.append(result)

        return results


class OracleBinaryCodeAnswerer(OracleCodeAnswerer):
    def __init__(self, llm_call, task_data, seed, mode="run"):
        super().__init__(llm_call, task_data, seed, mode)

    def answer(self, question):
        #completion = self.task_data["canonical_solution"]
        #program_str = self.task_data["prompt"] + completion
        program_str = _get_program_str(
            self.task_data["canonical_solution"],
            self.task_data["entry_point"],
            self.task_data["prompt"],
        )
        check_program = test_case_ls_to_check([question])
        result = get_test_case_result(
            program_str, check_program, self.task_data["entry_point"], 0.5
        )
        return result


class ActiveCodeReasonerBase:
    def __init__(
        self, llm_call, task_data, seed, mode="run", logprobs=False, unique_hs=False
    ):
        self.llm_call = llm_call
        self.entry_point = task_data["entry_point"]
        self.context = task_data["prompt"]
        self.total_cost = 0
        self.timeout = 0.1
        self.max_retry = 5
        self.total_questions = 1
        self.total_hypothesis = 0
        self.seed = seed
        self.mode = mode
        self.logprobs = logprobs
        self.unique_hs = unique_hs

    def generate_hypothesis(self, requirements):
        if self.mode == "run":
            return []
        else:
            return self._generate_hypothesis(requirements)

    def _generate_hypothesis(self, requirements):
        all_hypothesis = []
        it = 0
        total_cost = 0
        while len(all_hypothesis) < self.total_hypothesis and it < self.max_retry:
            hypothesis, cost = self._sample_raw_hypothesis(
                requirements, self.total_hypothesis, self.seed + it
            )
            total_cost += cost
            if len(requirements) > 0:
                hypothesis = self._filter_hypothesis(hypothesis, requirements)
            all_hypothesis += hypothesis
            it += 1

        self.total_cost += total_cost
        return all_hypothesis

    def _sample_raw_hypothesis(self, requirements, n_samples, seed):
        system_prompt = (
            "You are an expert Python programmer that specializes in coding tasks. "
            + "Complete the function given by the user. Do not change the function signature."
            + "Do not add any additional commentary. Do not import any additional libraries."
            + "Start your answer with the function signature."
        )
        response = self.llm_call(
            system_prompt=system_prompt,
            user_prompt=self._get_problem_statement(requirements),
            n_used=max(1, n_samples // 2),
            seed=seed,
            logprobs=self.logprobs,
        )
        cost = obtain_cost(response)
        hypothesis = [
            response.choices[i].message.content for i in range(len(response.choices))
        ]
        if self.logprobs:
            logprobs = [
                sum(x.logprob for x in response.choices[i].logprobs.content)
                for i in range(len(response.choices))
            ]
        else:
            logprobs = [None] * len(response.choices)

        hypothesis_ls = []
        for h, logp in zip(hypothesis, logprobs):
            hypothesis = Hypothesis(h, logp)
            hypothesis_ls.append(hypothesis)

        return hypothesis_ls, cost

    def _filter_hypothesis(self, hypothesis, requirements):
        valid_requirements = [r for r in requirements.values() if r[1] != '"error"' and r[1] != '"timed out"']
        if len(valid_requirements) == 0:
            return hypothesis
        
        tests = []
        for q, a in valid_requirements:
            test = "assert " + q + " == " + a
            tests.append(test)
        check_program = test_case_ls_to_check(tests)

        valid_hypothesis = []
        for h in hypothesis:
            program_str = _get_program_str(h.content, self.entry_point, self.context)
            result = get_test_case_result(
                program_str, check_program, self.entry_point, 0.5
            )
            if result == True:
                if self.unique_hs:
                    unique = h.content not in [h.content for h in valid_hypothesis]
                else:
                    unique = True
                if unique:
                    valid_hypothesis.append(h)

        return valid_hypothesis

    def generate_questions(self, requirements, restricted_questions):
        q_ls = []
        it = 0
        while len(q_ls) < self.total_questions and it < self.max_retry:
            q_ls += self._generate_questions(requirements, 3, 1, self.seed + it)
            it += 1
            q_ls = list(set(q_ls) - set(restricted_questions))

        return q_ls

    def _get_problem_statement(self, requirements):
        valid_requirements = [r for r in requirements.values() if r[1] != '"error"' and r[1] != '"timed out"']
        if len(valid_requirements) == 0:
            return self.context
        elif '# ====== Solution ======' in self.context:
            solution_divider = '# ====== Solution ======'
            problem_statement = self.context.split(solution_divider)[0]
            code = self.context.split(solution_divider)[-1]
            problem_statement += '"""\n====== Examples ======='
            for r in valid_requirements:
                q, a = r
                problem_statement += f"\n{q} -> {a}"
            problem_statement += '\n"""\n\n'
            problem_statement += solution_divider + '\n\n' + code
        else:
            problem_statement = (
                self.context.strip('"""\n') + "\n" + "    Examples:" + "\n"
            )
            for r in valid_requirements:
                q, a = r
                problem_statement += f"\n    {_extract_inputs([q], self.entry_point, include_entry_point=False)[0]} -> {a}"
            problem_statement += '\n    """'
        return problem_statement

    def _generate_questions(self, requirements, n_samples, n_used, seed):
        system_prompt = (
            "You are an expert Python programmer that specializes in solving user-specified coding tasks"
            + "To ensure you correctly understand user specifications, you can query the user for expected program outputs of sample inputs."
            + f"Given the function signature, generate {n_samples} sample inputs that will be most helpful in formalizing user intent."
            + "Structure your response as a list of function calls:\n "
            + f"1. {self.entry_point}(inputs)\n"
            + f"2. {self.entry_point}(inputs)\n"
            + "...\n"
            + f"{n_samples}. {self.entry_point}(inputs)\n"
            + f"Do not generate any additional content beyond the numbered list of function calls."
        )
        if len(requirements) > 0:
            system_prompt += "Do not repeat the same inputs as in the Examples given."
        response = self.llm_call(
            system_prompt=system_prompt,
            user_prompt=self._get_problem_statement(requirements),
            n_used=n_used,
            seed=seed,
        )
        cost = obtain_cost(response)
        contents = [response.choices[i].message.content for i in range(n_used)]
        questions = []
        for content in contents:
            question_str_ls = content.split(self.entry_point)
            questions_str_ls = [self.entry_point + t for t in question_str_ls if t != ""]
            if len(questions_str_ls) == 0:
                return []
            questions_str_ls = _extract_inputs(questions_str_ls, self.entry_point)
            questions_str_ls = [i for i in questions_str_ls if i != "error"]
            questions.append(questions_str_ls)
        questions = [item for sublist in questions for item in sublist]

        self.total_cost += cost

        return questions

    def select_best_question(self, questions, hypothesis):
        if len(questions) > 0:
            return questions[0]
        else:
            return "None"

    def q_a_to_requirement(self, question, answer):
        return (question, answer)


class ActiveCodeReasoner(ActiveCodeReasonerBase):
    def __init__(
        self,
        llm_call,
        task_data,
        total_questions,
        total_hypothesis,
        seed,
        mode="run",
        logprobs=False,
        unique_hs=False,
    ):
        super().__init__(llm_call, task_data, seed, mode, logprobs, unique_hs)
        self.total_questions = total_questions
        self.total_hypothesis = total_hypothesis

    def generate_hypothesis(self, requirements):
        return self._generate_hypothesis(requirements)

    def generate_questions(self, requirements, restricted_questions):
        q_ls = []
        it = 0
        while len(q_ls) < self.total_questions and it < self.max_retry:
            q_ls += self._generate_questions(
                requirements, self.total_questions, 2, self.seed + it
            )
            it += 1
            q_ls = list(set(q_ls) - set(restricted_questions))

        return q_ls

    def answer_questions(self, program, questions):
        program_str = _get_program_str(program, self.entry_point, self.context)
        return get_expected_outputs(program_str, questions, self.timeout)

    def select_best_question(self, questions, hypothesis):
        if self.logprobs == False:
            logprobs = [0 for h in hypothesis]
        else:
            logprobs = [h.logp for h in hypothesis]

        print("Selecting best question")
        if len(questions) == 1 or len(hypothesis) < 1:
            return questions[0]

        answers_df = pd.DataFrame()
        for h_id, h in enumerate(hypothesis):
            answers = [to_str(o) for o in self.answer_questions(h.content, questions)]
            q_id = list(range(len(questions)))
            df = pd.DataFrame(
                {
                    "program_id": h_id,
                    "q_id": q_id,
                    "output": answers,
                    "logprob": logprobs[h_id],
                }
            )
            answers_df = pd.concat([answers_df, df])

        print(answers_df.pivot(index="program_id", columns="q_id", values="output"))

        valid_qs = answers_df.groupby("q_id")["output"].apply(
            lambda x: len(x[(x != "error") & (x != "timed out")]) > 0
        )
        valid_qs = valid_qs[valid_qs == True]
        answers_df = answers_df[answers_df["q_id"].isin(valid_qs.index)]
        print(answers_df.head())
        if len(answers_df) == 0:
            return np.random.choice(questions)
        answers_df['output'] = answers_df['output'].str.replace('timed out', 'error')
        output_counts = (
            answers_df.fillna("None")
            .groupby(["q_id", "output"])["logprob"]
            .apply(lambda x: np.sum([1 / np.exp(y) for y in x]))
            .reset_index()
        )
        output_counts.columns = ["q_id", "output", "count"]
        print(output_counts)

        def _get_entropy(probs):
            probs = np.array(probs)
            probs = probs / probs.sum()
            entropy = -(probs * np.log(probs)).sum()
            return entropy

        entropies = output_counts.groupby("q_id")["count"].apply(_get_entropy)
        print(entropies)
        if len(entropies[entropies > 0]) == 0:
            return questions[entropies.index[0]]

        else:
            best_question_idx = entropies[entropies == entropies.max()].index[0]
            print(f"Best question id: {best_question_idx}")
            return questions[best_question_idx]


class ActiveBinaryCodeReasonerBase(ActiveCodeReasonerBase):
    def __init__(self, llm_call, task_data, seed, mode="run"):
        super().__init__(llm_call, task_data, seed, mode)

    def _generate_questions(self, requirements, n_samples, n_used, seed):
        system_prompt = (
            "You are an expert Python programmer that specializes in solving user-specified coding tasks"
            + "To ensure you correctly understand user specifications, you can write additional test cases."
            + f"Given the function signature, generate {n_samples} sample test cases that will be most helpful in formalizing user intent."
            + "Structure your response as a list of assertions:\n "
            + f"1. assert {self.entry_point}(inputs) == \n"
            + f"2. assert {self.entry_point}(inputs) == \n"
            + "...\n"
            + f"{n_samples}. assert {self.entry_point}(inputs) == \n"
            + f"Do not generate any additional content or comments beyond the list of assertions."
        )
        if len(requirements) > 0:
            system_prompt += "Do not repeat the same tests as in the Examples given."
        response = self.llm_call(
            system_prompt=system_prompt,
            user_prompt=self._get_problem_statement(requirements),
            n_used=n_used,
            seed=seed,
        )
        cost = obtain_cost(response)
        contents = [response.choices[i].message.content for i in range(n_used)]
        questions = []
        for content in contents:
            content =  re.sub("#.+\\n", "", content)
            questions_str_ls = [
                "assert " + t
                for t in re.split("assert ", content)
                if t.startswith(self.entry_point)
            ]
            if len(questions_str_ls) == 0:
                return [] 
            questions_str_ls[0] = re.sub("^1. ", "", questions_str_ls[0]).strip()
            questions_str_ls = [re.sub(r"\n\d. ", "", q) for q in questions_str_ls]
            questions_str_ls = [
                q.strip("\n").strip("\n").rstrip(',')
                for q in questions_str_ls
            ]
            questions.append(questions_str_ls)
        questions = [item for sublist in questions for item in sublist]
        questions = [q for q in questions if self._check_question(q)]
        self.total_cost += cost
        return questions
    
    def _check_question(self, q):
        check_program = self.context
        check_program += f'    pass\n'
        check_program += f"\n\ndef check({self.entry_point}):"
        check_program += f"\n    try:"
        check_program += f"\n        {q}"
        check_program += f"\n    except AssertionError as e:"
        check_program += f"\n        print(e)"
        check_program += f"\n        return True"
        check_program += f"\n    except Exception as e:"
        check_program += f"\n        print(e)"
        check_program += f"\n        return False"
        check_program += f"\n    return True"
        check_program += f"\n\nresult=check({self.entry_point})"
        exec_globals = {}
        blockPrint()
        result = False
        try:
            exec(check_program, exec_globals)
            result = exec_globals["result"]
        except Exception as e:
            result = False
        enablePrint()
        return result

    def _filter_hypothesis(self, hypothesis, requirements):
        questions = [t[0] for t in requirements.values()]
        answers = [t[1] for t in requirements.values()]

        check_program = test_case_ls_to_check_with_answers(questions, answers)
        print(check_program)
        valid_hypothesis = []
        for h in hypothesis:
            program_str = _get_program_str(h.content, self.entry_point, self.context)
            result = get_test_case_result(
                program_str, check_program, self.entry_point, 0.5
            )
            if result == True:
                if self.unique_hs:
                    unique = h.content not in [h.content for h in valid_hypothesis]
                else:
                    unique = True
                if unique:
                    valid_hypothesis.append(h)

        return valid_hypothesis

    def _get_problem_statement(self, requirements):
        answers = [t[1] for t in requirements.values()]
        true_answers = [a for a in answers if a == True]
        if len(true_answers) == 0:
            return self.context
        elif '# ====== Solution ======' in self.context:
            solution_divider = '# ====== Solution ======'
            problem_statement = self.context.split(solution_divider)[0]
            code = self.context.split(solution_divider)[-1]
            problem_statement += '"""\n====== Examples ======='
            for q, a in requirements.values():
                if a == True:
                    problem_statement +=  f"\n    {q}"
            problem_statement += '\n"""\n\n'
            problem_statement += solution_divider + '\n\n' + code
        else:
            problem_statement = (
                self.context.strip('"""\n') + "\n" + "    Examples:" + "\n"
            )
            for q, a in requirements.values():
                if a == True:
                    problem_statement += f"\n    {q}"

            problem_statement += '\n    """'
        return problem_statement


class ActiveBinaryCodeReasoner(ActiveBinaryCodeReasonerBase, ActiveCodeReasoner):
    def __init__(
        self,
        llm_call,
        task_data,
        total_questions,
        total_hypothesis,
        seed,
        logprobs=False,
        unique_hs=False,
    ):
        ActiveCodeReasoner.__init__(
            self,
            llm_call=llm_call,
            task_data=task_data,
            total_questions=total_questions,
            total_hypothesis=total_hypothesis,
            seed=seed,
            logprobs=logprobs,
            unique_hs=unique_hs
        )


    def answer_questions(self, program, questions):
        program_str = _get_program_str(program, self.entry_point, self.context)
        answers = []
        for q in questions:
            check_program = test_case_ls_to_check([q])
            result = get_test_case_result(
                program_str, check_program, self.entry_point, 0.5
            )
            answers.append(result)
        return answers

