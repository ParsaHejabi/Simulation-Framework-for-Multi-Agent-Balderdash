from game.game_manager import GameManager
from dotenv import load_dotenv
import os
from utils.config import LLM_MODEL, RANDOM_SEED, TEMP, HISTORY_WINDOW_SIZE

load_dotenv()


def main() -> None:
    game_manager = GameManager(
        os.getenv("MONGODB_CONNECTION_STRING"),
        game_description=f"with unlimited history, default rules, known words, no communication, no seed stories",
        random_seed=RANDOM_SEED,
        judge_llm_model_name=LLM_MODEL,
        llms_temperature=TEMP,
        history_window_size=HISTORY_WINDOW_SIZE,
    )

    # Create players
    game_manager.create_player(f"Player 1 - {LLM_MODEL}", LLM_MODEL)
    game_manager.create_player(f"Player 2 - {LLM_MODEL}", LLM_MODEL)
    game_manager.create_player(f"Player 3 - {LLM_MODEL}", LLM_MODEL)

    # Start the game
    game_manager.start_game(
        os.path.join("data", "meta-llama_Meta-Llama-3-8B-Instruct_balderdash_words1.csv"),
        filter_known_words=True,
    )


if __name__ == "__main__":
    main()
