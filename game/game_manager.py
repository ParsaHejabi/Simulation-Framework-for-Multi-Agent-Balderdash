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


class GameManager:
    def __init__(self, db_connection_string: str) -> None:
        self.logger = setup_logger("game_manager", "logs/game_manager.log")
        self.logger.info("Initializing GameManager")
        self.db = MongoDB(db_connection_string)
        self.players = []
        self.current_round = 0
        self.device = self.get_device()
        self.logger.info(f"Using device: {self.device}")

    def create_player(self, player_id: int, name: str) -> None:
        self.logger.info(f"Creating player: {player_id} - {name}")
        player = Player(player_id, name, self.device)
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
        for round_id in range(1, NUM_ROUNDS + 1):
            self.play_round(round_id)

    def play_round(self, round_id: int) -> None:
        self.logger.info(f"Playing round: {round_id}")
        # Select a random word and its definition
        word, correct_definition, pos = self.get_random_word()

        # Create a new round
        round = Round(round_id, word, correct_definition, pos)

        # Generate definitions from players
        for player in self.players:
            with open("prompts/generate_definition.txt", "r") as file:
                prompt_template = file.read()
            definition = player.generate_definition(word, prompt_template)
            round.add_player_definition(player.player_id, definition)

        # Perform voting
        for player in self.players:
            with open("prompts/vote_definition.txt", "r") as file:
                prompt_template = file.read()
            vote = player.vote_definition(round.player_definitions, prompt_template)
            round.add_vote(player.player_id, vote)

        # Calculate scores
        scores = round.calculate_scores()
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
