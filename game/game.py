from utils.logger import setup_logger


class Game:
    def __init__(
        self,
        game_id: int,
        game_description: str,
        number_of_rounds: int,
        judge_llm_model_name: str,
        random_seed: int,
        receiving_vote_points: int,
        correct_vote_points: int,
        correct_definition_points: int,
        history_window_size: int,
        llms_temperature: float,
        words_file: str,
        filter_words: str,
    ) -> None:
        self.logger = setup_logger(f"game_{game_id}", f"logs/game_{game_id}.log")
        self.logger.info(
            f"Initializing Game: {game_id} - Description: {game_description} - Judge LLM: {judge_llm_model_name}"
        )
        self.game_id = game_id
        self.game_description = game_description
        self.number_of_rounds = number_of_rounds
        self.judge_llm_model_name = judge_llm_model_name
        self.random_seed = random_seed
        self.receiving_vote_points = receiving_vote_points
        self.correct_vote_points = correct_vote_points
        self.correct_definition_points = correct_definition_points
        self.history_window_size = history_window_size
        self.llms_temperature = llms_temperature
        self.words_file = words_file
        self.filter_words = filter_words
