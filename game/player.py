from utils.logger import setup_logger
from utils.llm import LLM
from typing import List


class Player:
    def __init__(self, player_id: int, name: str, llm: LLM) -> None:
        self.logger = setup_logger(f"player_{player_id}", f"logs/player_{player_id}.log")
        self.logger.info(f"Initializing Player: {player_id} - Name: {name} - LLM: {llm.model_name}")
        self.player_id = player_id
        self.name = name
        self.score = 0
        self.llm = llm

    def update_score(self, points: int) -> None:
        self.logger.info(f"Player {self.player_id} - {self.name} scored: {points} points")
        self.score += points

    def generate_definition(self, word: str, prompt_template: str) -> str:
        self.logger.info(f"Generating definition for word: {word} for player: {self.player_id} - {self.name}")
        return self.llm.generate_definition(word, prompt_template)

    def vote_definition(
        self, word: str, definition: str, definitions: List[str], prompt_template: str
    ) -> int:
        self.logger.info(f"Voting on definitions for player: {self.player_id} - {self.name}")
        return self.llm.vote_definition(word, definition, definitions, prompt_template)
