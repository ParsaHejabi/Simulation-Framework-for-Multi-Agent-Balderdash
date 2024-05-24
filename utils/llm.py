from utils.logger import setup_logger
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import List, Optional
import os
from openai import OpenAI
import tiktoken


class LLM:
    def __init__(
        self,
        device: torch.device,
        temp: float,
        model_name: str,
        max_tokens: int = 512,
        verbose: bool = False,
        random_seed: Optional[int] = None,
    ):
        self.logger = setup_logger("llm", "logs/llm.log", verbose=verbose)
        self.model_name = model_name
        self.device = device
        self.temp = temp
        self.max_tokens = max_tokens
        self.random_seed = random_seed
        self.is_api_model = is_api_model(model_name)

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
                    model_name, token=os.getenv("HF_TOKEN"), torch_dtype=torch.bfloat16
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
                    token=os.getenv("HF_TOKEN"),
                )
            elif self.model_name == "google/gemma-1.1-7b-it":
                self.pipe = pipeline(
                    "text-generation",
                    model=self.model_name,
                    model_kwargs={"torch_dtype": torch.bfloat16},
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
                    torch_dtype=torch.bfloat16,
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
                    token=os.getenv("HF_TOKEN"),
                )
            elif self.model_name == "mistralai/Mistral-7B-Instruct-v0.3":
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.getenv("HF_TOKEN"))
                self.pipe = pipeline(
                    task="text-generation",
                    model=self.model_name,
                    tokenizer=self.tokenizer,
                    model_kwargs={"torch_dtype": torch.bfloat16},
                    device=self.device,
                    temperature=self.temp,
                    max_new_tokens=self.max_tokens,
                    do_sample=True,
                    return_full_text=False,
                    token=os.getenv("HF_TOKEN"),
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            elif self.is_api_model:
                if self.model_name.startswith("gpt-"):
                    self.client = OpenAI(
                        api_key=os.getenv("OPENAI_API_KEY"),
                        project=os.getenv("OPENAI_PROJECT_ID"),
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
                        max_new_tokens=self.max_tokens,
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
                elif self.is_api_model:
                    if self.model_name.startswith("gpt-"):
                        self.logger.info(
                            f"Counting tokens for messages: {num_tokens_from_messages(messages, self.model_name)}"
                        )
                        self.logger.info(f"Calling OpenAI chat/completion with messages: {messages}")
                        completion = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=messages,
                            max_tokens=self.max_tokens,
                            seed=self.random_seed,
                        )
                        if completion.choices[0].finish_reason == "length":
                            raise ValueError(
                                f"Error: OpenAI model {self.model_name} output for messages: {completion.choices[0].message.content} has reached max tokens {self.max_tokens}"
                            )
                        elif completion.choices[0].finish_reason == "null":
                            raise ValueError(
                                f"Error: OpenAI model {self.model_name} output for messages: API response still in progress or incomplete"
                            )
                        elif completion.choices[0].finish_reason == "stop":
                            return completion.choices[0].message.content

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


def is_api_model(model_name: str) -> bool:
    if model_name in ["gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-4-turbo", "gpt-4o"] or model_name in [
        "gemini-pro",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-latest",
    ]:
        return True
    return False


def num_tokens_from_messages(messages: List[dict], model_name: str) -> int:
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model_name in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model_name == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model_name:
        print(
            "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
        )
        return num_tokens_from_messages(messages, model_name="gpt-3.5-turbo-0613")
    elif "gpt-4" in model_name:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model_name="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model_name}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
