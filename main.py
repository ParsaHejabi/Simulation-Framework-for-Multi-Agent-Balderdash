from game.game_manager import GameManager
from dotenv import load_dotenv
import os
import argparse

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, help="Random seed value")
    parser.add_argument("--history_window_size", type=int, help="History window size")
    parser.add_argument("--player_llm_model", type=str, help="Player LLM model name")
    parser.add_argument("--judge_llm_model", type=str, help="Judge LLM model name")
    parser.add_argument("--correct_vote_points", type=int, help="Correct vote points")
    parser.add_argument("--correct_definition_points", type=int, help="Correct definition points")
    parser.add_argument("--receiving_vote_points", type=int, help="Receiving vote points")
    parser.add_argument("--llms_temperature", type=float, help="LLMs temperature")
    parser.add_argument("--num_rounds", type=int, help="Number of rounds")
    parser.add_argument("--words_file", type=str, help="Words file")
    parser.add_argument("--filter_words", type=str, help="Filter words", choices=["known", "unknown", "all"])
    args = parser.parse_args()

    game_manager = GameManager(
        db_connection_string=os.getenv("MONGODB_CONNECTION_STRING"),
        game_description=f"Correct definition rule experiment with default scoring rules but correct definition score changing from 0 to 6 and five different seeds, no communication, no seed stories.",
        random_seed=args.random_seed,
        judge_llm_model_name=args.judge_llm_model,
        llms_temperature=args.llms_temperature,
        history_window_size=args.history_window_size,
        receiving_vote_points=args.receiving_vote_points,
        correct_vote_points=args.correct_vote_points,
        correct_definition_points=args.correct_definition_points,
        num_rounds=args.num_rounds,
        words_file=args.words_file,
        filter_words=args.filter_words,
    )

    # Create players
    game_manager.create_player(f"Player 1", game_manager.game.game_id, args.player_llm_model)
    game_manager.create_player(f"Player 2", game_manager.game.game_id, args.player_llm_model)
    game_manager.create_player(f"Player 3", game_manager.game.game_id, args.player_llm_model)

    # Start the game
    game_manager.start_game(os.path.join("data", args.words_file))


if __name__ == "__main__":
    main()
