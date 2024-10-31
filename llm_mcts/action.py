# Copyright (c) 2024 Magik Compute Pte. Ltd. All rights reserved.
from openai import OpenAI
from typing import Dict, Iterator, List, Optional, Union
from qwen_agent.agent import BasicAgent
from qwen_agent.agents import TIRMathAgent
from qwen_agent.llm import BaseChatModel


def load_llm_config(base_url='http://localhost:8000/v1', api_key='token-abc123'):
    client = OpenAI(api_key=api_key, base_url=base_url)
    model_name = client.models.list().data[0].id

    llm_cfg = {
            # Use a model service compatible with the OpenAI API
            'model': model_name,
            'model_server': client.base_url.__str__(),  # base_url, also known as api_base
            'api_key': client.api_key,
            # (Optional) LLM hyperparameters for generation:
            'generate_cfg': {
                # 'temperature': 0.9,
                'max_input_tokens': 30000
            }
        }
    return llm_cfg, model_name.split("/")[-1]



class MCTSAgent(BasicAgent):
    def __init__(self, llm_cfg: Optional[Union[Dict, BaseChatModel]] = None):
        super().__init__(llm=llm_cfg)
        self.cot_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
        self.tir_prompt = """Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."""

        self.tir_agent = TIRMathAgent(llm=llm_cfg, name='Qwen2.5-Math', system_message=self.tir_prompt)

    def zero_shot(self, question):
        messages = [
            {"role": "system", "content": self.cot_prompt},
            {'role': 'user', 'content': question}
        ]
        *_, last = self.run(messages)
        return last[-1]['content']

    def tir_answer(self, question):
        messages = [{'role': 'user', 'content': question}]
        *_, last = self.tir_agent.run(messages, function_list=['code_interpreter'])
        return last[-1]['content']



# if __name__ == '__main__':
#     llm_cfg, model_name = load_llm_config('http://localhost:8000/v1')
#     agent = MCTSAgent(llm_cfg)
#     print(agent.zero_shot("斐波那契数列前10项"))
    # print(agent.tir_answer("斐波那契数列前10项"))

