from utils.logger import setup_logger
from utils.llm import LLM
from typing import List


class Player:
    def __init__(self, player_id: int, name: str, game_id, llm: LLM) -> None:
        self.logger = setup_logger(f"player_{player_id}", f"logs/player_{player_id}.log")
        self.logger.info(f"Initializing Player: {player_id} - Name: {name} - LLM: {llm.model_name}")
        self.player_id = player_id
        self.llm_name = llm.model_name
        self.game_id = game_id
        self.name = name
        self.score = 0
        self.score_history = {}
        self.rank_history = {}
        self.llm = llm

    def update_score_and_rank(self, round_id: int, points: int, rank: int) -> None:
        self.logger.info(
            f"Player {self.player_id} - {self.name} scored: {points} points in round: {round_id}"
        )
        self.score += points
        self.score_history[round_id] = self.score
        self.rank_history[round_id] = rank

    def generate_definition(self, word: str, messages: List[dict]) -> str:
        self.logger.info(f"Generating definition for word: {word} for player: {self.player_id} - {self.name}")
        return self.llm.generate_definition(word, messages)

    def vote_definition(self, messages: List[dict]) -> int:
        self.logger.info(f"Voting on definitions for player: {self.player_id} - {self.name}")
        return self.llm.vote_definition(messages)
