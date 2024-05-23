from game.player import Player
from game.round import Round
from game.game import Game
from database.mongodb import MongoDB
import random
from utils.logger import setup_logger
import pandas as pd
from ast import literal_eval
from typing import List, Tuple
import torch
from utils.llm import LLM, is_api_model
import pandas as pd


class GameManager:
    def __init__(
        self,
        db_connection_string: str,
        game_description: str,
        random_seed: int,
        llms_temperature: float,
        history_window_size: int,
        receiving_vote_points: int,
        correct_vote_points: int,
        correct_definition_points: int,
        num_rounds: int,
        judge_llm_model_name: str,
        judge_llm_gpu: int,
        words_file: str,
        filter_words: str,
        dry_run: bool = False,
    ) -> None:
        self.dry_run = dry_run
        self.logger = setup_logger("game_manager", "logs/game_manager.log", verbose=self.dry_run)
        self.logger.info("Initializing GameManager")
        self.db = MongoDB(db_connection_string)
        self.players = []

        self.llms = {}
        self.receiving_vote_points = receiving_vote_points
        self.correct_vote_points = correct_vote_points
        self.correct_definition_points = correct_definition_points
        self.filter_words = filter_words
        self.game = Game(
            self.db.get_last_game_id() + 1,
            game_description=game_description,
            number_of_rounds=num_rounds,
            judge_llm_model_name=judge_llm_model_name,
            random_seed=random_seed,
            receiving_vote_points=self.receiving_vote_points,
            correct_vote_points=self.correct_vote_points,
            correct_definition_points=self.correct_definition_points,
            history_window_size=history_window_size,
            llms_temperature=llms_temperature,
            words_file=words_file,
            filter_words=filter_words,
        )
        # If it is not a dry run, save the game to the database
        if not self.dry_run:
            self.db.insert_game(self.game.__dict__)

        self.random_seed = random_seed
        self.set_random_seed(self.random_seed)
        self.llms_temperature = llms_temperature
        self.history_window_size = history_window_size
        self.judge_llm = self.get_or_load_llm(judge_llm_model_name, gpu_index=judge_llm_gpu)

    def set_random_seed(self, random_seed: int) -> None:
        random.seed(random_seed)
        torch.random.manual_seed(random_seed)

    def get_or_load_llm(self, model_name: str, gpu_index: int) -> LLM:
        if model_name not in self.llms:
            this_llm_device = self.get_device(model_name=model_name, gpu_index=gpu_index)
            self.logger.info(f"Loading LLM: {model_name} on device: {this_llm_device}")
            self.llms[model_name] = LLM(
                device=this_llm_device,
                model_name=model_name,
                temp=self.llms_temperature,
                verbose=self.dry_run,
                random_seed=self.random_seed,
            )
        return self.llms[model_name]

    def create_player(self, name: str, game_id: int, model_name: str, gpu_index: int) -> None:
        llm = self.get_or_load_llm(model_name, gpu_index=gpu_index)
        last_player_id = self.db.get_last_player_id()
        player_id = last_player_id + 1
        self.logger.info(f"Creating player: {player_id} - {name} with LLM: {llm.model_name}")
        player = Player(player_id, name, game_id, llm)
        self.players.append(player)
        if not self.dry_run:
            self.db.insert_player(player.__dict__)

    def load_word_data(self, file_path: str, filter_known_words: str) -> None:
        self.logger.info(f"Loading word data from: {file_path}")
        self.word_data = {}
        data = pd.read_csv(file_path, engine="python")
        # if data contains a column "llm_knows_word" then filter the data based on the value of this column
        if "llm_knows_word" in data.columns:
            if filter_known_words == "known":
                # filter the data using the column "llm_knows_word" in the dataframe and keep only the rows where the value is True
                data = data[data["llm_knows_word"] == True].reset_index(drop=True)
                self.logger.info(f"Filtered known words and loaded {len(data)} words")
            elif filter_known_words == "unknown":
                # filter the data using the column "llm_knows_word" in the dataframe and keep only the rows where the value is False
                data = data[data["llm_knows_word"] == False].reset_index(drop=True)
                self.logger.info(f"Filtered unknown words and loaded {len(data)} words")
        data = data.sample(self.game.number_of_rounds, random_state=self.random_seed)
        for _, row in data.iterrows():
            word = row["words"].strip().lower()
            # if row["def"] first character starts with '[' then use literal_eval to convert it to a list
            # if not, just strip the string
            if row["def"].strip()[0] == "[":
                definition = literal_eval(row["def"])
                pos = literal_eval(row["POS"])
                # assert len(definition) == len(pos)
                # for each definition strip the string
                definition = [defn.strip() for defn in definition]
                pos = [p.strip() for p in pos if p is not None]
                self.word_data[len(self.word_data) + 1] = {
                    "word": word,
                    "def": definition,
                    "POS": pos,
                    "type": "multiple_definitions",
                }
            else:
                definition = row["def"].strip()
                pos = row["POS"]
                if pos:
                    pos = pos.strip()
                self.word_data[len(self.word_data) + 1] = {
                    "word": word,
                    "def": definition,
                    "POS": pos,
                    "type": "single_definition",
                }

    def start_game(self, words_file: str) -> None:
        self.logger.info("Starting the game")
        self.load_word_data(words_file, filter_known_words=self.filter_words)
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
            player_definition = round["player_definitions"][str(player_id)]["definition"]
            is_true_definition = (
                str(round["player_definitions"][str(player_id)]["judge_decision"]).lower() == "true"
            )
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

    def get_player_history_csv(self, player_id: int, window_size: int) -> str:
        player_rounds = self.db.get_player_rounds(player_id=player_id, window_size=window_size)
        history = []
        for round in player_rounds:
            round_id = round["round_id"]
            word = round["word"]
            definition = round["correct_definition"]
            generated_definition = round["player_definitions"][str(player_id)]["definition"]
            wrote_true_definition = round["player_definitions"][str(player_id)]["judge_decision"]
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
            # rank_among_players is an integer indicating your rank among all players up to that round. It can be calculated using self.players list and each player.score attribute.
            # If players have the same score, they should have the same rank.
            rank_among_players = (
                len(
                    [
                        player
                        for player in self.players
                        if player.score > self.get_player_by_id(player_id).score
                    ]
                )
                + 1
            )
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
                    "rank_among_players": rank_among_players,
                }
            )

        df = pd.DataFrame(history)
        return df.to_csv(index=False)

    def play_round(self, game_id: int, round_id: int) -> None:
        self.logger.info(f"Playing round: {round_id}")
        # Select a random word and its definition
        word, all_correct_definitions, all_poss = (
            self.word_data[round_id]["word"],
            self.word_data[round_id]["def"],
            self.word_data[round_id]["POS"],
        )
        if self.word_data[round_id]["type"] == "single_definition":
            correct_definition = all_correct_definitions
            if all_poss is None:
                pos = "N/A"
            else:
                pos = all_poss
        else:
            correct_definition = all_correct_definitions[0]
            if len(all_poss) == 0 or all_poss[0] is None:
                pos = "N/A"
            else:
                pos = all_poss[0]
        self.logger.info(f"Selected word: {word} with definition: {correct_definition}, and POS: {pos}")

        # Create a new round
        round = self.create_round(game_id, round_id, word, correct_definition, pos)

        with open("prompts/game_rules.txt", "r") as game_rules_prompt_template_file:
            game_rules_prompt_template = game_rules_prompt_template_file.read()
            game_rules_prompt = game_rules_prompt_template.format(
                receiving_vote_points=self.receiving_vote_points,
                correct_vote_points=self.correct_vote_points,
                correct_definition_points=self.correct_definition_points,
            )

        with open("prompts/system_judge.txt", "r") as system_judge_prompt_template_file:
            system_judge_prompt_template = system_judge_prompt_template_file.read()

        # Generate definitions from players
        for player in self.players:
            generate_definition_messages = [{"role": "system", "content": game_rules_prompt}]
            with open("prompts/history.txt", "r") as history_prompt_template_file:
                history_prompt_template = history_prompt_template_file.read()
                history_csv = self.get_player_history_csv(
                    player_id=player.player_id,
                    window_size=self.history_window_size,
                )
                history_prompt = history_prompt_template.format(history_csv=history_csv)
            with open(
                "prompts/user_generate_definition.txt", "r"
            ) as user_generate_definition_prompt_template_file:
                user_generate_definition_prompt_template = (
                    user_generate_definition_prompt_template_file.read()
                )
                user_generate_definition_prompt = user_generate_definition_prompt_template.format(word=word)
            generate_definition_messages.append(
                {"role": "user", "content": "\n".join([history_prompt, user_generate_definition_prompt])}
            )
            definition = player.generate_definition(word, generate_definition_messages)
            self.logger.info(
                f"Player {player.player_id} - {player.name} defined the word {word} as: {definition}"
            )

            judge_decision_messages = [{"role": "system", "content": system_judge_prompt_template}]
            with open("prompts/user_judge.txt", "r") as user_judge_prompt_template_file:
                user_judge_prompt_template = user_judge_prompt_template_file.read()
                user_judge_prompt = user_judge_prompt_template.format(
                    word=word, correct_definition=correct_definition, definition=definition
                )
            judge_decision_messages.append({"role": "user", "content": user_judge_prompt})
            # First check if the player knows the meaning of the main word used in this round
            judge_decision_on_round_word = self.judge_llm.judge_decision(word, judge_decision_messages)
            self.logger.info(
                f"Judge decision for player {player.player_id} - {player.name} with definition: {definition} is: {judge_decision_on_round_word}"
            )

            llm_knows_one_of_the_defs = False
            if judge_decision_on_round_word:
                llm_knows_one_of_the_defs = True
            else:
                for another_correct_def in all_correct_definitions:
                    judge_decision_messages = [{"role": "system", "content": system_judge_prompt_template}]
                    user_judge_prompt = user_judge_prompt_template.format(
                        word=word, correct_definition=another_correct_def, definition=definition
                    )
                    judge_decision_messages.append({"role": "user", "content": user_judge_prompt})
                    judge_decision_on_another_correct_def = self.judge_llm.judge_decision(
                        word, judge_decision_messages
                    )
                    if judge_decision_on_another_correct_def:
                        llm_knows_one_of_the_defs = True
                        break

            round.add_player_definition(
                player.player_id,
                definition,
                judge_decision=judge_decision_on_round_word,
                llm_knows_one_of_the_defs=llm_knows_one_of_the_defs,
            )

        eligible_voting_players_definitions = round.get_eligible_voting_players_definitions()
        self.logger.info(f"Eligible voting players definitions: {eligible_voting_players_definitions}")
        # Perform voting
        for player in self.players:
            # Skip players who defined the word correctly
            if round.player_definitions[player.player_id]["judge_decision"]:
                self.logger.info(
                    f"Skipping player with database ID {player.player_id} - {player.name} from voting as they defined the word correctly"
                )
                continue
            vote_definition_messages = [{"role": "system", "content": game_rules_prompt}]
            with open("prompts/history.txt", "r") as history_prompt_template_file:
                history_prompt_template = history_prompt_template_file.read()
                history_csv = self.get_player_history_csv(
                    player_id=player.player_id, window_size=self.history_window_size
                )
                history_prompt = history_prompt_template.format(history_csv=history_csv)

            # get the index of this player's definition in the round.definitions_permutation list
            player_index_in_the_permuted_list = round.definitions_permutation.index(player.player_id)

            # create a list of all indexes excluding the player's index
            all_indexes_excluding_player = [
                str(index + 1)
                for index in range(len(round.definitions_permutation))
                if index != player_index_in_the_permuted_list
            ]

            if len(all_indexes_excluding_player) == 1:
                all_indexes_excluding_player_text = f"which is {all_indexes_excluding_player[0]}"
            else:
                all_indexes_excluding_player_text = f"among {', '.join(all_indexes_excluding_player)}"

            # find this player's definition in the eligible_voting_players_definitions list and remove it from the list
            definitions_passed_for_voting = eligible_voting_players_definitions.copy()
            # pop the player's definition from definitions_passed_for_voting.
            # The player's definition is the one that starts with player_index_in_the_permuted_list + 1
            definitions_passed_for_voting.pop(player_index_in_the_permuted_list)
            with open("prompts/vote_definition.txt", "r") as vote_definition_prompt_template_file:
                vote_definition_prompt_template = vote_definition_prompt_template_file.read()
                vote_definition_prompt = vote_definition_prompt_template.format(
                    word=word,
                    definition=round.player_definitions[player.player_id]["definition"],
                    definitions="\n".join(definitions_passed_for_voting),
                    all_indexes_excluding_player=all_indexes_excluding_player_text,
                )
            vote_definition_messages.append(
                {"role": "user", "content": "\n".join([history_prompt, vote_definition_prompt])}
            )
            target_permuted_player_id = player.vote_definition(
                vote_definition_messages,
            )
            self.logger.info(
                f"Player {player.player_id} - {player.name} voted for definition: {target_permuted_player_id}"
            )
            round.add_vote(player.player_id, target_permuted_player_id)

        # Calculate scores
        scores = round.calculate_scores(
            receiving_vote_points=self.receiving_vote_points,
            correct_vote_points=self.correct_vote_points,
            correct_definition_points=self.correct_definition_points,
        )
        self.logger.info(f"Round scores: {scores}")
        for player_id, score in scores.items():
            player = self.get_player_by_id(player_id)
            player.update_score(score)
            if not self.dry_run:
                self.db.update_player(player_id, {"score": player.score})

        if not self.dry_run:
            # Store round data in the database
            self.db.insert_round(round.__dict__)

    def get_player_by_id(self, player_id: int) -> Player:
        try:
            for player in self.players:
                if player.player_id == player_id:
                    return player
            raise ValueError(f"Player with ID {player_id} not found")
        except ValueError as e:
            self.logger.critical(e)
            exit()

    def get_device(self, model_name: str, gpu_index: int) -> torch.device:
        try:
            if gpu_index < 0 or is_api_model(model_name):
                return torch.device("cpu")

            if not torch.backends.mps.is_available():
                if torch.cuda.is_available():
                    # Check if that GPU index is valid
                    if gpu_index >= torch.cuda.device_count():
                        raise ValueError(
                            f"GPU index {gpu_index} is not valid. There are only {torch.cuda.device_count()} GPUs available"
                        )
                    # Check if that GPU index is free
                    if torch.cuda.memory_allocated(torch.device(f"cuda:{gpu_index}")) > 0:
                        raise ValueError(f"GPU index {gpu_index} is not free")
                    return torch.device(f"cuda:{gpu_index}")
                else:
                    return torch.device("cpu")
            else:
                return torch.device("mps")
        except ValueError as e:
            self.logger.critical(e)
            exit()
