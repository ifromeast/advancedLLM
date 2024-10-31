import os
import sys
import json
import asyncio
import argparse
from collections import OrderedDict
from typing import Optional, Set, List
from contextlib import asynccontextmanager
import fastapi
import uvicorn
from fastapi import Request

import torch
from transformers import AutoModel, AutoTokenizer
from transformers import BitsAndBytesConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"  # For 4090*6

class MathRMEvaluator:
    def __init__(self, model_name: str) -> None:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        self.model = AutoModel.from_pretrained(
            model_name, 
            device_map="auto", 
            torch_dtype=torch.float16,
            trust_remote_code=True,
            quantization_config=quantization_config
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.max_length = 1024

    def calc_score(self, sys_prompt: str = None, question: str = "", response: str = "") -> float:
        if not sys_prompt:
            sys_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response}
                ]
        conversation_str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        input_ids = self.tokenizer.encode(conversation_str, return_tensors="pt", add_special_tokens=False).to(self.model.device)

        outputs = self.model(input_ids=input_ids[:,:self.max_length])
        torch.cuda.empty_cache()
        score = outputs[0].cpu().detach().numpy()[0][0]
        return score


_running_tasks: Set[asyncio.Task] = set()

@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    async def _force_log():
        while True:
            await asyncio.sleep(10)

    task = asyncio.create_task(_force_log())
    _running_tasks.add(task)
    task.add_done_callback(_running_tasks.remove)
    yield

app = fastapi.FastAPI(lifespan=lifespan)

evaluator = MathRMEvaluator(model_name='/data0/ckpt/Qwen/Qwen2.5-Math-RM-72B')

@app.post("/get_score")
async def chat(request: Request):
    params = await request.json()
    system_prompt = params.get("sys_prompt", "")
    question = params.get("question", "")
    response = params.get("response", "")
    score = evaluator.calc_score(sys_prompt=system_prompt, question=question, response=response)
    return {"score": float(score)}

if __name__ == '__main__':
    
    uvicorn.run(app, host='0.0.0.0', port=8110)