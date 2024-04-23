from game.player import Player
from game.round import Round
from database.mongodb import MongoDB
from utils.config import NUM_ROUNDS, LLM_MODEL
import random
from utils.logger import setup_logger
import pandas as pd
from ast import literal_eval
from typing import Optional
import torch
from utils.llm import LLM


class GameManager:
    def __init__(self, db_connection_string: str, llm_model_name: str = LLM_MODEL) -> None:
        self.logger = setup_logger("game_manager", "logs/game_manager.log")
        self.logger.info("Initializing GameManager")
        self.db = MongoDB(db_connection_string)
        self.players = []
        self.current_round = 0
        self.device = self.get_device()
        self.logger.info(f"Using device: {self.device}")

        self.llms = {}
        self.judge_llm = self.get_or_load_llm(llm_model_name)

    def get_or_load_llm(self, model_name: str = LLM_MODEL) -> LLM:
        if model_name not in self.llms:
            self.logger.info(f"Loading LLM: {model_name} on device: {self.device}")
            self.llms[model_name] = LLM(self.device, model_name)
        return self.llms[model_name]

    def create_player(self, name: str, model_name: Optional[str] = None) -> None:
        llm = self.get_or_load_llm(model_name)
        last_player_id = self.db.get_last_player_id()
        player_id = last_player_id + 1
        self.logger.info(f"Creating player: {player_id} - {name} with LLM: {llm.model_name}")
        player = Player(player_id, name, llm)
        self.players.append(player)
        self.db.insert_player(player.__dict__)

    def load_word_data(self, file_path: str) -> None:
        self.logger.info(f"Loading word data from: {file_path}")
        self.word_data = []
        data = pd.read_csv(file_path)
        for _, row in data.iterrows():
            word = row["words"]
            definition = literal_eval(row["def"])[0]
            pos = literal_eval(row["POS"])[0]
            self.word_data.append((word, definition, pos))

    def start_game(self, words_file: str = "data/words.csv") -> None:
        self.logger.info("Starting the game")
        self.load_word_data(words_file)
        self.logger.info(f"Loaded {len(self.word_data)} words and playing {NUM_ROUNDS} rounds")
        last_game_id = self.db.get_last_game_id()
        game_id = last_game_id + 1
        for round_id in range(1, NUM_ROUNDS + 1):
            self.play_round(game_id, round_id)

    def create_round(
        self, game_id: int, round_id: int, word: str, correct_definition: str, pos: str
    ) -> Round:
        self.logger.info(f"Creating round: {round_id}/10 in game: {game_id} with word: {word}")
        return Round(
            game_id=game_id,
            round_id=round_id,
            word=word,
            correct_definition=correct_definition,
            pos=pos,
            players=[player.player_id for player in self.players],
        )

    def play_round(self, game_id: int, round_id: int) -> None:
        self.logger.info(f"Playing round: {round_id}")
        # Select a random word and its definition
        word, correct_definition, pos = self.get_random_word()
        self.logger.info(f"Selected word: {word} with definition: {correct_definition}, and POS: {pos}")

        # Create a new round
        round = self.create_round(game_id, round_id, word, correct_definition, pos)

        # Generate definitions from players
        for player in self.players:
            with open("prompts/generate_definition.txt", "r") as file:
                prompt_template = file.read()
            definition = player.generate_definition(word, prompt_template)
            self.logger.info(
                f"Player {player.player_id} - {player.name} defined the word {word} as: {definition}"
            )

            with open("prompts/judge.txt", "r") as file:
                judge_prompt_template = file.read()
            judge_decision = self.judge_llm.judge_decision(
                word, correct_definition, definition, judge_prompt_template
            )
            self.logger.info(
                f"Judge decision for player {player.player_id} - {player.name} with definition: {definition} is: {judge_decision}"
            )
            round.add_player_definition(player.player_id, definition, judge_decision=judge_decision)

        eligible_voting_players_definitions = round.get_eligible_voting_players_definitions()
        self.logger.info(f"Eligible voting players definitions: {eligible_voting_players_definitions}")
        # Perform voting
        for player in self.players:
            # Skip players who defined the word correctly
            if round.player_definitions[player.player_id][1]:
                self.logger.info(
                    f"Skipping player {player.player_id} - {player.name} from voting as they defined the word correctly"
                )
                continue
            with open("prompts/vote_definition.txt", "r") as file:
                prompt_template = file.read()
            target_permuted_player_id = player.vote_definition(
                word,
                round.player_definitions[player.player_id][0],
                eligible_voting_players_definitions,
                prompt_template,
            )
            self.logger.info(
                f"Player {player.player_id} - {player.name} voted for definition: {target_permuted_player_id}"
            )
            round.add_vote(player.player_id, target_permuted_player_id)

        # Calculate scores
        scores = round.calculate_scores()
        self.logger.info(f"Round scores: {scores}")
        for player_id, score in scores.items():
            player = self.get_player_by_id(player_id)
            player.update_score(score)
            self.db.update_player(player_id, {"score": player.score})

        # Store round data in the database
        self.db.insert_round(round.__dict__)

    def get_random_word(self) -> tuple:
        return random.choice(self.word_data)

    def get_player_by_id(self, player_id: int) -> Player:
        for player in self.players:
            if player.player_id == player_id:
                return player
        raise ValueError(f"Player with ID {player_id} not found")

    def get_device(self) -> torch.device:
        if not torch.backends.mps.is_available():
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        else:
            return torch.device("mps")
