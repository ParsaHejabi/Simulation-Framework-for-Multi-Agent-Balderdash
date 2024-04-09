from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.config import LLM_MODEL
from utils.logger import setup_logger
import torch


class Player:
    def __init__(self, player_id: int, name: str, device: torch.device) -> None:
        self.logger = setup_logger(f"player_{player_id}", f"logs/player_{player_id}.log")
        self.logger.info(f"Initializing Player: {player_id} - {name}")
        self.player_id = player_id
        self.name = name
        self.score = 0
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        # TODO: Quantize the model!
        self.model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, device_map="auto").to(self.device)

    def update_score(self, points: int) -> None:
        self.score += points

    def generate_definition(self, word: str, prompt_template: str) -> str:
        self.logger.info(f"Generating definition for word: {word}")
        prompt = prompt_template.format(word=word)
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        definition = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return definition.strip()

    def vote_definition(self, definitions: dict, prompt_template: str) -> int:
        self.logger.info(f"Voting on definitions: {definitions}")
        prompt = prompt_template.format(definitions="\n".join(definitions.values()))
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=10, do_sample=True)
        vote = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return int(vote.strip())
