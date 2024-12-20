from utils.logger import setup_logger
from typing import List
import random


class Round:
    def __init__(
        self, game_id: int, round_id: int, word: str, correct_definition: str, pos: str, players: List[int]
    ) -> None:
        self.logger = setup_logger(f"round_{round_id}", f"logs/round_{round_id}.log")
        self.logger.info(
            f"Initializing Round: {round_id} - Word: {word}, Correct Definition: {correct_definition}, POS: {pos}"
        )
        self.game_id = game_id
        self.players = players
        self.round_id = round_id
        self.word = word
        self.correct_definition = correct_definition
        self.pos = pos
        self.scores = {}
        self.player_definitions = {}
        self.definitions_permutation = []
        self.votes = {}

    def add_player_definition(
        self, player_id: int, definition: str, judge_decision: bool, llm_knows_one_of_the_defs: bool
    ) -> None:
        self.logger.info(
            f"Adding player definition: {player_id} - {definition}, judge decision: {judge_decision}, Does LLM know at least one of the definitions: {llm_knows_one_of_the_defs}"
        )
        self.player_definitions[player_id] = {
            "definition": definition,
            "judge_decision": judge_decision,
            "llm_knows_one_of_the_defs": llm_knows_one_of_the_defs,
        }

    def add_vote(self, player_id: int, target_permuted_player_id: int) -> None:
        self.logger.info(f"Adding vote: {player_id} - {target_permuted_player_id}")
        target_player_id = self.definitions_permutation[target_permuted_player_id - 1]
        self.votes[player_id] = target_player_id

    def get_eligible_voting_players_definitions(self) -> List[str]:
        self.logger.info("Getting eligible voting players definitions")
        all_definitions = {
            player_id: player_definition_dict["definition"]
            for player_id, player_definition_dict in self.player_definitions.items()
            if not player_definition_dict["judge_decision"]
        }
        # Add the correct definition to the list of definitions
        all_definitions[-1] = self.correct_definition
        # A random permutation of shuffled all_definitions.keys()
        self.definitions_permutation = random.sample(all_definitions.keys(), len(all_definitions))
        return [
            f"{index + 1}. {all_definitions[player_id]}"
            for index, player_id in enumerate(self.definitions_permutation)
        ]

    def calculate_scores(
        self, receiving_vote_points: int, correct_vote_points: int, correct_definition_points: int
    ) -> dict:
        self.logger.info("Calculating scores")
        scores = {}
        for player_id in self.player_definitions.keys():
            scores[player_id] = 0

        for player_id, target_player_id in self.votes.items():
            if target_player_id == -1:
                scores[player_id] += correct_vote_points
            else:
                scores[target_player_id] += receiving_vote_points

        # Assign scores to players
        for player_id, player_definition_dict in self.player_definitions.items():
            if player_definition_dict["judge_decision"]:
                scores[player_id] += correct_definition_points

        self.logger.info(f"Calculated scores: {scores}")
        self.scores = scores
        return scores
