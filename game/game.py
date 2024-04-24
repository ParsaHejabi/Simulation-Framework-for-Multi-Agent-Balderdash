from utils.logger import setup_logger


class Game:
    def __init__(
        self, game_id: int, game_description: str, number_of_rounds: int, judge_llm_model_name: str
    ) -> None:
        self.logger = setup_logger(f"game_{game_id}", f"logs/game_{game_id}.log")
        self.logger.info(
            f"Initializing Game: {game_id} - Description: {game_description} - Judge LLM: {judge_llm_model_name}"
        )
        self.game_id = game_id
        self.game_description = game_description
        self.number_of_rounds = number_of_rounds
        self.judge_llm_model_name = judge_llm_model_name
