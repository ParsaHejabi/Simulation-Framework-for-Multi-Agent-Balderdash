from utils.logger import setup_logger
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import List
import os


class LLM:
    def __init__(
        self, device: torch.device, temp: float, model_name: str, max_tokens: int = 512, verbose: bool = False
    ):
        self.logger = setup_logger("llm", "logs/llm.log", verbose=verbose)
        self.model_name = model_name
        self.device = device
        self.temp = temp
        self.max_tokens = max_tokens

        if self.device.type == "mps":
            from mlx_lm import load

            self.model, self.tokenizer = load(model_name)
        else:
            if self.model_name == "meta-llama/Meta-Llama-3-8B-Instruct":
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HF_TOKEN"))
                """
                Based on the information at https://huggingface.co/docs/transformers/main/en/model_doc/llama3
                The Llama3 models were trained using bfloat16, but the original inference uses float16.
                The original model uses pad_id = -1 which means that there is no padding token.
                We can’t have the same logic, make sure to add a padding token using tokenizer.add_special_tokens({"pad_token":"<pad>"}) and resize the token embedding accordingly.
                You should also set the model.config.pad_token_id.
                """
                if self.tokenizer.pad_token is None:
                    self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
                    self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids("<pad>")

                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, token=os.getenv("HF_TOKEN"), torch_dtype=torch.float16
                ).to(self.device)
                self.model.resize_token_embeddings(len(self.tokenizer))
                self.model.config.pad_token_id = self.tokenizer.pad_token_id

                self.pipe = pipeline(
                    task="text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temp,
                    do_sample=True,
                    return_full_text=False,
                )
            elif self.model_name == "google/gemma-1.1-7b-it":
                self.pipe = pipeline(
                    "text-generation",
                    model=self.model_name,
                    # model_kwargs={"torch_dtype": torch.bfloat16},
                    device=self.device,
                    temperature=self.temp,
                    max_new_tokens=self.max_tokens,
                    do_sample=True,
                    # return_full_text=False,
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
            elif self.model_name == "microsoft/Phi-3-small-8k-instruct":
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True, token=os.getenv("HF_TOKEN")
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    token=os.getenv("HF_TOKEN"),
                    torch_dtype="auto",
                ).to(self.device)
                self.pipe = pipeline(
                    task="text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temp,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_full_text=False,
                )
            elif self.model_name == "mistralai/Mistral-7B-Instruct-v0.3":
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HF_TOKEN"))
                self.pipe = pipeline(
                    task="text-generation",
                    model=self.model_name,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    temperature=self.temp,
                    max_new_tokens=self.max_tokens,
                    do_sample=True,
                    return_full_text=False,
                    token=os.getenv("HF_TOKEN"),
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            else:
                raise ValueError(f"Error: Model {self.model_name} not supported")

    def check_input_with_context_length(self, messages: List[dict]) -> None:
        """
        Check if the input prompt is within the context length of the model. For now, we are assuming that the context
        length is 8192 tokens. If the prompt is greater than 8192 tokens, raise an error.
        """
        input_prompt = self.pipe.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        if len(input_prompt[0]) > 8192:
            raise ValueError(
                f"Error: Input {messages[0]['content'][:100]} is greater than 8192 tokens\n{messages}"
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
            try:
                if self.model_name == "meta-llama/Meta-Llama-3-8B-Instruct":
                    prompt = self.pipe.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    self.logger.info(f"Message: {messages} changed to prompt: {prompt}")
                    self.check_input_with_context_length(messages)
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
                    return outputs[0]["generated_text"]
                elif self.model_name == "mistralai/Mistral-7B-Instruct-v0.2":
                    prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    self.logger.info(f"Message: {messages} changed to prompt: {prompt}")
                    self.check_input_with_context_length(messages)
                    outputs = self.pipe(
                        prompt,
                        do_sample=True,
                        pad_token_id=self.pipe.tokenizer.eos_token_id,
                        max_new_tokens=self.max_tokens,
                        temperature=self.temp,
                    )
                    return outputs[0]["generated_text"][len(outputs) :]
                    # output = self.pipe(messages)[0]["generated_text"][-1]["content"].strip()
                elif self.model_name == "google/gemma-1.1-7b-it":
                    # This model does not support "system" messages.
                    # Get the content of the "system" message and put it at the beginning of the "user" message
                    # First, assert that messages has only two elements, one has role "system" and the other has role "user"
                    assert len(messages) == 2
                    assert messages[0]["role"] == "system"
                    assert messages[1]["role"] == "user"
                    # Get the content of the "user" message
                    user_message = messages[1]["content"]
                    # Get the content of the "system" message
                    system_message = messages[0]["content"]
                    messages = [{"role": "user", "content": "\n".join([system_message, user_message])}]
                    prompt = self.pipe.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    self.logger.info(f"Message: {messages} changed to prompt: {prompt}")
                    self.check_input_with_context_length(messages)
                    outputs = self.pipe(
                        prompt,
                        do_sample=True,
                        temperature=self.temp,
                    )
                    return outputs[0]["generated_text"][len(prompt) :]
                elif self.model_name == "microsoft/Phi-3-small-8k-instruct":
                    prompt = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    self.logger.info(f"Message: {messages} changed to prompt: {prompt}")
                    self.check_input_with_context_length(messages)
                    outputs = self.pipe(
                        prompt,
                        do_sample=True,
                        pad_token_id=self.pipe.tokenizer.eos_token_id,
                        max_new_tokens=self.max_tokens,
                        temperature=self.temp,
                    )
                    return outputs[0]["generated_text"]
                elif self.model_name == "mistralai/Mistral-7B-Instruct-v0.3":
                    # This model does not support "system" messages and only has "user" and "assistant" roles.
                    # Get the content of the "system" message and put it at the beginning of the "user" message
                    # First, assert that messages has only two elements, one has role "system" and the other has role "user".
                    assert len(messages) == 2
                    assert messages[0]["role"] == "system"
                    assert messages[1]["role"] == "user"
                    # Get the content of the "user" message.
                    user_message = messages[1]["content"]
                    # Get the content of the "system" message.
                    system_message = messages[0]["content"]
                    messages = [{"role": "user", "content": "\n".join([system_message, user_message])}]
                    prompt = self.pipe.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    self.logger.info(f"Message: {messages} changed to prompt: {prompt}")
                    self.check_input_with_context_length(messages)
                    outputs = self.pipe(
                        prompt,
                        do_sample=True,
                        temperature=self.temp,
                        max_new_tokens=self.max_tokens,
                    )
                    return outputs[0]["generated_text"]
                else:
                    raise ValueError(f"Error: Model {self.model_name} not supported")

            except ValueError as e:
                self.logger.critical(e)
                exit()

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
                raise ValueError(f"Error: {model_output} does not start with a number")
        except ValueError as e:
            self.logger.critical(e)
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
