from utils.logger import setup_logger
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.config import LLM_MODEL
import torch
from typing import List
import os


class LLM:
    def __init__(
        self, device: torch.device, model_name: str = LLM_MODEL, temp: float = 0.7, max_tokens: int = 512
    ):
        self.logger = setup_logger("llm", "logs/llm.log")
        self.model_name = model_name
        self.device = device
        self.temp = temp
        self.max_tokens = max_tokens

        if self.device.type == "mps":
            from mlx_lm import load

            self.model, self.tokenizer = load(model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HF_TOKEN"))
            # TODO: Quantize the model!
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, token=os.getenv("HF_TOKEN"), device_map="auto"
            ).to(self.device)

        self.logger.info(f"Initialized LLM with model: {model_name} on device: {self.device}")

    def generate_definition(self, word: str, prompt_template: str) -> str:
        self.logger.info(f"Generating definition for word: {word} using model: {self.model_name}")
        prompt = prompt_template.format(word=word)
        if self.device.type == "mps":
            messages = [
                {"role": "user", "content": prompt},
            ]
            encoded_messages = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
            from mlx_lm import generate

            return generate(
                self.model,
                self.tokenizer,
                encoded_messages,
                verbose=True,
                max_tokens=self.max_tokens,
                temp=self.temp,
            ).strip()
        else:
            messages = [
                {"role": "user", "content": prompt},
            ]
            model_inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.device)
            # model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(model_inputs, max_new_tokens=100, do_sample=True)
            definition = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return definition.strip()

    def vote_definition(self, word, definition, definitions: List[str], prompt_template: str) -> int:
        self.logger.info(f"Voting on definitions: {definitions} using model: {self.model_name}")
        prompt = prompt_template.format(word=word, definition=definition, definitions="\n".join(definitions))
        if self.device.type == "mps":
            from mlx_lm import generate

            return int(
                generate(
                    self.model,
                    self.tokenizer,
                    prompt,
                    verbose=True,
                    max_tokens=self.max_tokens,
                    temp=self.temp,
                ).strip()
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

            # TODO: Is this the best way to handle this?
            # We expect the output to be "True" or "False" but in some cases the model gives longer unnecessary outputs. So if the first 5 characters contain the word "True" or "False" we consider it as the output.
            return (
                generate(
                    self.model,
                    self.tokenizer,
                    prompt,
                    verbose=True,
                    max_tokens=self.max_tokens,
                    temp=self.temp,
                )
                .strip()
                .lower()[0:4]
                == "true"
            )
        else:
            model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=10, do_sample=True)
            decision = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return decision.strip().lower()[0:4] == "true"
