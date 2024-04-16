from utils.logger import setup_logger
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.config import LLM_MODEL
import torch
from typing import List


class LLM:
    def __init__(self, device: torch.device, model_name: str = LLM_MODEL) -> None:
        self.logger = setup_logger("llm", "logs/llm.log")
        self.model_name = model_name
        self.device = device

        if self.device.type == "mps":
            from mlx_lm import load

            self.model, self.tokenizer = load(model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # TODO: Quantize the model!
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto").to(self.device)

        self.logger.info(f"Initialized LLM with model: {model_name} on device: {self.device}")

    def generate_definition(self, word: str, prompt_template: str) -> str:
        self.logger.info(f"Generating definition for word: {word} using model: {self.model_name}")
        prompt = prompt_template.format(word=word)
        if self.device.type == "mps":
            from mlx_lm import generate

            return generate(
                self.model, self.tokenizer, prompt, verbose=True, max_tokens=512, temp=0.7
            ).strip()
        else:
            model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
            definition = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return definition.strip()

    def vote_definition(self, word, definition, definitions: List[str], prompt_template: str) -> int:
        self.logger.info(f"Voting on definitions: {definitions} using model: {self.model_name}")
        prompt = prompt_template.format(word=word, definition=definition, definitions="\n".join(definitions))
        if self.device.type == "mps":
            from mlx_lm import generate

            return int(
                generate(self.model, self.tokenizer, prompt, verbose=True, max_tokens=512, temp=0.7).strip()
            )
        else:
            model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=10, do_sample=True)
            vote = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return int(vote.strip())

    def judge_decision(
        self, word: str, correct_definition: str, definition: str, prompt_template: str
    ) -> bool:
        self.logger.info(f"Judging decision for word: {word} using model: {self.model_name}")
        prompt = prompt_template.format(
            word=word, correct_definition=correct_definition, definition=definition
        )
        if self.device.type == "mps":
            from mlx_lm import generate

            return (
                generate(self.model, self.tokenizer, prompt, verbose=True, max_tokens=512, temp=0.7).strip()
                == "True"
            )
        else:
            model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=10, do_sample=True)
            decision = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return decision.strip() == "True"
