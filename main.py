from game.game_manager import GameManager
from dotenv import load_dotenv
import os
from utils.config import LLM_MODEL

load_dotenv()


def main() -> None:
    game_manager = GameManager(os.getenv("MONGODB_CONNECTION_STRING"))

    # Create players
    game_manager.create_player(f"Player 1 - {LLM_MODEL}", LLM_MODEL)
    game_manager.create_player(f"Player 2 - {LLM_MODEL}", LLM_MODEL)
    game_manager.create_player(f"Player 3 - {LLM_MODEL}", LLM_MODEL)

    # Start the game
    game_manager.start_game(os.path.join("data", "modified_balderdash_words1.csv"))


if __name__ == "__main__":
    main()
