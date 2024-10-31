import torch
from transformers import AutoModel, AutoTokenizer
from transformers import BitsAndBytesConfig

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