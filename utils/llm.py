from utils.logger import setup_logger
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import List
import os


class LLM:
    def __init__(self, device: torch.device, temp: float, model_name: str, max_tokens: int = 512):
        self.logger = setup_logger("llm", "logs/llm.log")
        self.model_name = model_name
        self.device = device
        self.temp = temp
        self.max_tokens = max_tokens

        if self.device.type == "mps":
            from mlx_lm import load

            self.model, self.tokenizer = load(model_name)
        else:
            if self.model_name == "meta-llama/Meta-Llama-3-8B-Instruct":
                self.pipe = pipeline(
                    "text-generation",
                    model=self.model_name,
                    # model_kwargs={"torch_dtype": torch.bfloat16},
                    device=self.device,
                    temperature=self.temp,
                    max_new_tokens=self.max_tokens,
                    do_sample=True,
                )
            elif self.model_name == "mistralai/Mistral-7B-Instruct-v0.2":
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HF_TOKEN"))
                self.model = AutoModelForCausalLM.from_pretrained(model_name, token=os.getenv("HF_TOKEN")).to(
                    self.device
                )
                self.pipe = pipeline(
                    task="text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    temperature=self.temp,
                    max_new_tokens=self.max_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                )

    def generate_answer(self, messages: List[dict]) -> str:
        if self.device.type == "mps":
            # TODO: For Mac it doesn't work
            from mlx_lm import generate

            return generate(
                self.model,
                self.tokenizer,
                messages,
                verbose=True,
                max_tokens=self.max_tokens,
                temp=self.temp,
            )
        else:
            if self.model_name == "meta-llama/Meta-Llama-3-8B-Instruct":
                prompt = self.pipe.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                self.logger.info(f"Message: {messages} changed to prompt: {prompt}")
                terminators = [
                    self.pipe.tokenizer.eos_token_id,
                    self.pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                ]
                outputs = self.pipe(
                    prompt,
                    do_sample=True,
                    eos_token_id=terminators,
                    temperature=self.temp,
                )
                return outputs[0]["generated_text"][len(prompt) :]
            elif self.model_name == "mistralai/Mistral-7B-Instruct-v0.2":
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                self.logger.info(f"Message: {messages} changed to prompt: {prompt}")
                outputs = self.pipe(
                    prompt,
                    do_sample=True,
                    pad_token_id=self.pipe.tokenizer.eos_token_id,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temp,
                )
                return outputs[0]["generated_text"][len(outputs) :]
                # output = self.pipe(messages)[0]["generated_text"][-1]["content"].strip()

    def generate_definition(self, word: str, messages: List[dict]) -> str:
        self.logger.info(f"Generating definition for word: {word} using model: {self.model_name}")
        model_output = self.generate_answer(messages)
        self.logger.info(f"Model output for generating definition: {model_output}")
        # If model_output has double quotes at the start and end, remove them
        return model_output.strip().strip('"')

    def vote_definition(self, messages: List[dict]) -> int:
        self.logger.info(f"Voting on definitions using model: {self.model_name}")
        model_output = self.generate_answer(messages)
        self.logger.info(f"Model output for voting a definition: {model_output}")
        try:
            # if output starts with a number, convert to int and return
            if model_output[0].isdigit():
                return int(model_output[0])
            # if output has a single digit in it, find it using regex and return it
            elif any(char.isdigit() for char in model_output):
                return int("".join(filter(str.isdigit, model_output)))
            else:
                # raise ValueError
                raise ValueError()
        except ValueError:
            self.logger.error(f"Error: {model_output} does not start with a number")
            exit()

    def judge_decision(self, word: str, messages: List[dict]) -> bool:
        self.logger.info(f"Judging decision for word: {word} using model: {self.model_name}")
        model_output = self.generate_answer(messages)
        self.logger.info(f"Model output for judge decision: {model_output}")
        return model_output.strip().lower()[0:4] == "true"

    def know_one_of_the_definitions(self, word: str, messages: List[dict]) -> bool:
        self.logger.info(
            f"Checking if LLM knows one of the definitions for word: {word} using model: {self.model_name}"
        )
        model_output = self.generate_answer(messages)
        self.logger.info(f"Model output for knowing one of the definitions: {model_output}")
        return model_output.strip().lower()[0:4] == "true"
