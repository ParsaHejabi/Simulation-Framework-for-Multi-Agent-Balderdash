from utils.logger import setup_logger
from typing import List
import random


class Round:
    def __init__(self, round_id: int, word: str, correct_definition: str, pos: str) -> None:
        self.logger = setup_logger(f"round_{round_id}", f"logs/round_{round_id}.log")
        self.logger.info(f"Initializing Round: {round_id} - Word: {word}")
        self.round_id = round_id
        self.word = word
        self.correct_definition = correct_definition
        self.pos = pos
        self.player_definitions = {}
        self.definitions_permutation = []
        self.votes = {}

    def add_player_definition(self, player_id: int, definition: str, judge_decision: bool) -> None:
        self.logger.info(
            f"Adding player definition: {player_id} - {definition}, judge decision: {judge_decision}"
        )
        self.player_definitions[player_id] = (definition, judge_decision)

    def add_vote(self, player_id: int, target_permuted_player_id: int) -> None:
        self.logger.info(f"Adding vote: {player_id} - {target_permuted_player_id}")
        target_player_id = self.definitions_permutation[target_permuted_player_id - 1]
        self.votes[player_id] = target_player_id

    def get_eligible_voting_players_definitions(self) -> List[str]:
        self.logger.info("Getting eligible voting players definitions")
        all_definitions = {
            player_id: definition
            for player_id, (definition, judge_decision) in self.player_definitions.items()
            if not judge_decision
        }
        # Add the correct definition to the list of definitions
        all_definitions[-1] = self.correct_definition
        # A random permutation of shuffled all_definitions.keys()
        self.definitions_permutation = random.sample(all_definitions.keys(), len(all_definitions))
        return [
            f"{index + 1}: {all_definitions[player_id]}"
            for index, player_id in enumerate(self.definitions_permutation)
        ]

    def calculate_scores(self):
        self.logger.info("Calculating scores")
        scores = {}
        for player_id in self.player_definitions.keys():
            scores[player_id] = 0

        for player_id, target_player_id in self.votes.items():
            if target_player_id == -1:
                scores[player_id] += 2
            else:
                scores[target_player_id] += 1

        # Assign scores to players
        for player_id, (definition, judge_decision) in self.player_definitions.items():
            if judge_decision:
                scores[player_id] += 3

        self.logger.info(f"Calculated scores: {scores}")
        return scores
