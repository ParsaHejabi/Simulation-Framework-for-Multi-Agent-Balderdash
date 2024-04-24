from game.player import Player
from game.round import Round
from game.game import Game
from database.mongodb import MongoDB
from utils.config import NUM_ROUNDS, LLM_MODEL
import random
from utils.logger import setup_logger
import pandas as pd
from ast import literal_eval
from typing import Optional, List, Tuple
import torch
from utils.llm import LLM
import pandas as pd


class GameManager:
    def __init__(
        self,
        db_connection_string: str,
        game_description: str,
        random_seed: int,
        judge_llm_model_name: str = LLM_MODEL,
    ) -> None:
        self.logger = setup_logger("game_manager", "logs/game_manager.log")
        self.logger.info("Initializing GameManager")
        self.db = MongoDB(db_connection_string)
        self.players = []
        self.device = self.get_device()
        self.logger.info(f"Using device: {self.device}")

        self.llms = {}
        self.game = Game(
            self.db.get_last_game_id() + 1,
            game_description=game_description,
            number_of_rounds=NUM_ROUNDS,
            judge_llm_model_name=judge_llm_model_name,
        )
        self.db.insert_game(self.game.__dict__)
        self.random_seed = random_seed
        self.set_random_seed(self.random_seed)
        self.judge_llm = self.get_or_load_llm(judge_llm_model_name)

    def set_random_seed(self, random_seed: int) -> None:
        random.seed(random_seed)

    def get_or_load_llm(self, model_name: str) -> LLM:
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
            definition = literal_eval(row["def"])[0].strip()
            pos = literal_eval(row["POS"])[0]
            if pos:
                pos = pos.strip()
            self.word_data.append((word, definition, pos))

    def start_game(self, words_file: str = "data/words.csv") -> None:
        self.logger.info("Starting the game")
        self.load_word_data(words_file)
        self.logger.info(
            f"Loaded {len(self.word_data)} words and playing {self.game.number_of_rounds} rounds"
        )
        for round_id in range(1, self.game.number_of_rounds + 1):
            self.play_round(self.game.game_id, round_id)

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

    def get_round_winners_strategies(self, round: dict) -> List[Tuple[str, str]]:
        """
        Get the strategies of the player(s) who got the highest scores in the corresponding round.
        """
        scores = round["scores"]
        max_score = max(scores.values())
        round_winners = [player_id for player_id, score in scores.items() if score == max_score]
        round_winners_strategies = []
        for player_id in round_winners:
            player_definition = round["player_definitions"][str(player_id)][0]
            is_true_definition = str(round["player_definitions"][str(player_id)][1]).lower() == "true"
            if is_true_definition:
                outcome = f"This definition was semantically equal to the true dictionary definition"
            else:
                number_of_received_votes = len(
                    [k for k, v in round["votes"].items() if v == player_id and k != player_id]
                )
                has_chosen_true_definition = round["votes"][str(player_id)] == -1
                if has_chosen_true_definition:
                    outcome = f"This definition was chosen by {number_of_received_votes} players excluding the player himself and the player also chose the true dictionary definition"
                else:
                    outcome = f"This definition was chosen by {number_of_received_votes} players excluding the player himself"

            round_winners_strategies.append((player_definition, outcome))

        return round_winners_strategies

    def get_player_history_csv(self, player_id: int) -> str:
        player_rounds = self.db.get_player_rounds(player_id)
        history = []
        for round in player_rounds:
            round_id = round["round_id"]
            word = round["word"]
            definition = round["correct_definition"]
            generated_definition = round["player_definitions"][str(player_id)][0]
            wrote_true_definition = round["player_definitions"][str(player_id)][1]
            if wrote_true_definition:
                guessed_correct_definiton = False
                deception_ratio = -1
            else:
                guessed_correct_definiton = round["votes"][str(player_id)] == -1
                number_of_received_votes = len(
                    [k for k, v in round["votes"].items() if v == player_id and k != player_id]
                )
                deception_ratio = number_of_received_votes / len(round["votes"])
            round_winners_strategies = self.get_round_winners_strategies(round)
            history.append(
                {
                    "round_id": round_id,
                    "word": word,
                    "definition": definition,
                    "generated_definition": generated_definition,
                    "wrote_true_definition": wrote_true_definition,
                    "guessed_correct_definiton": guessed_correct_definiton,
                    "deception_ratio": deception_ratio,
                    "round_winners_strategies": round_winners_strategies,
                }
            )

        df = pd.DataFrame(history)
        return df.to_csv()

    def play_round(self, game_id: int, round_id: int) -> None:
        self.logger.info(f"Playing round: {round_id}")
        # Select a random word and its definition
        word, correct_definition, pos = self.get_random_word()
        self.logger.info(f"Selected word: {word} with definition: {correct_definition}, and POS: {pos}")

        # Create a new round
        round = self.create_round(game_id, round_id, word, correct_definition, pos)

        with open("prompts/default_game_rules.txt", "r") as game_rules_prompt_template_file:
            game_rules_prompt_template = game_rules_prompt_template_file.read()

        # Generate definitions from players
        for player in self.players:
            generate_definition_messages = [{"role": "system", "content": game_rules_prompt_template}]
            with open("prompts/history.txt", "r") as history_prompt_template_file:
                history_prompt_template = history_prompt_template_file.read()
                history_csv = self.get_player_history_csv(player.player_id)
                history_prompt = history_prompt_template.format(history_csv=history_csv)
            with open("prompts/generate_definition.txt", "r") as generate_definition_prompt_template_file:
                generate_definition_prompt_template = generate_definition_prompt_template_file.read()
                generate_definition_prompt = generate_definition_prompt_template.format(word=word)
            generate_definition_messages.append(
                {"role": "user", "content": "\n".join([history_prompt, generate_definition_prompt])}
            )
            definition = player.generate_definition(word, generate_definition_messages)
            self.logger.info(
                f"Player {player.player_id} - {player.name} defined the word {word} as: {definition}"
            )

            judge_decision_messages = []
            with open("prompts/judge.txt", "r") as judge_prompt_template_file:
                judge_prompt_template = judge_prompt_template_file.read()
                judge_prompt = judge_prompt_template.format(
                    word=word, correct_definition=correct_definition, definition=definition
                )
            judge_decision_messages.append({"role": "user", "content": judge_prompt})
            judge_decision = self.judge_llm.judge_decision(word, judge_decision_messages)
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
            vote_definition_messages = [{"role": "system", "content": game_rules_prompt_template}]
            with open("prompts/history.txt", "r") as history_prompt_template_file:
                history_prompt_template = history_prompt_template_file.read()
                history_csv = self.get_player_history_csv(player.player_id)
                history_prompt = history_prompt_template.format(history_csv=history_csv)
            with open("prompts/vote_definition.txt", "r") as vote_definition_prompt_template_file:
                vote_definition_prompt_template = vote_definition_prompt_template_file.read()
                vote_definition_prompt = vote_definition_prompt_template.format(
                    word=word,
                    definition=round.player_definitions[player.player_id][0],
                    definitions="\n".join(eligible_voting_players_definitions),
                )
            vote_definition_messages.append(
                {"role": "user", "content": "\n".join([history_prompt, vote_definition_prompt])}
            )
            target_permuted_player_id = player.vote_definition(
                eligible_voting_players_definitions,
                vote_definition_messages,
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
