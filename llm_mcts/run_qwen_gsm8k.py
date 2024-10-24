import os
import sys
import logging
from dotenv import load_dotenv

import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.evaluate import Evaluate
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.getcwd()))
from llm_mcts.mcts import MCTSr, ZeroShotCoT, MultipleTurnSelfRefine, Policy

load_dotenv()

logging.basicConfig(level=logging.WARNING)
logging.getLogger("mcts-llm").setLevel(logging.INFO)

np.random.seed(42)


# load llm
llm = dspy.HFClientVLLM(model='/data0/ckpt/qwen/Qwen2.5-7B-Instruct',
                    model_type='chat',
                    port=8001)

dspy.settings.configure(lm=llm, experimental=True)

# load dataset
gsm8k = GSM8K()
gsm8k_trainset = [
    dspy.Example(
        problem=example['question'], 
        gold_reasoning=example['gold_reasoning'],
        answer=example['answer']
    ).with_inputs("problem") for example in gsm8k.train
]

gsm8k_testset = [
    dspy.Example(
        problem=example['question'], 
        gold_reasoning=example['gold_reasoning'],
        answer=example['answer']
    ).with_inputs("problem") for example in gsm8k.test
]

evaluate = Evaluate(devset=gsm8k_testset[:36], metric=gsm8k_metric, num_threads=3, display_progress=True, display_table=False, 
                    return_outputs=True, return_all_scores=True, provide_traceback=True)

## method 1: zeroshotcot
cot_score, cot_results, cot_answ_label = evaluate(ZeroShotCoT())
cot_data = [(dict(example) | {"prediction": prediction, "correct": score}) for example, prediction, score in cot_results]
cotresult_df = pd.DataFrame(cot_data)
cotresult_df.to_excel("cot_result.xlsx")


## method 2: MultipleTurnSelfRefine
mtrf_score, mtrf_results, mtrf_answ_label = evaluate(MultipleTurnSelfRefine(num_turns=1))
mtrf_data = [(dict(example) | {"prediction": prediction, "correct": score}) for example, prediction, score in mtrf_results]
mtrfresult_df = pd.DataFrame(mtrf_data)
mtrfresult_df.to_excel("mtrf_result.xlsx")



## method 3: MCTSr
mctsr_score, mctsr_results, mctsr_answ_label = evaluate(MCTSr(samples_per_node=5, default_uct_score=1e8))
mctsr_data = [(dict(example) | {"prediction": prediction, "correct": score}) for example, prediction, score in mctsr_results]
mctsrresult_df = pd.DataFrame(mctsr_data)
mctsrresult_df.to_excel("mctsr_result.xlsx")



## method 4: MCTSr with importance sampling
mctsr_is_score, mctsr_is_results, mctsr_is_answ_label = evaluate(MCTSr(policy=Policy.IMPORTANCE_SAMPLING, samples_per_node=5, default_uct_score=1e8))
mctsr_is_data = [(dict(example) | {"prediction": prediction, "correct": score}) for example, prediction, score in mctsr_is_results]
mctsr_isresult_df = pd.DataFrame(mctsr_is_data)
mctsr_isresult_df.to_excel("mctsr_is_result.xlsx")


## method 5: MCTSr with max_rollouts=6
mctsr_maxr_score, mctsr_maxr_results, mctsr_maxr_answ_label = evaluate(MCTSr(max_rollouts=6))
mctsr_maxr_data = [(dict(example) | {"prediction": prediction, "correct": score}) for example, prediction, score in mctsr_maxr_results]
mctsr_maxrresult_df = pd.DataFrame(mctsr_maxr_data)
mctsr_maxrresult_df.to_excel("mctsr_maxr_result.xlsx")

