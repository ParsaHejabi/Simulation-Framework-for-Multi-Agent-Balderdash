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
    args = parser.parse_args()

    game_manager = GameManager(
        os.getenv("MONGODB_CONNECTION_STRING"),
        game_description=f"modified prompts for definition and voting, with unlimited history, default rules, unknown words, no communication, no seed stories",
        random_seed=args.random_seed,
        judge_llm_model_name=args.judge_llm_model,
        llms_temperature=args.llms_temperature,
        history_window_size=args.history_window_size,
        receiving_vote_points=args.receiving_vote_points,
        correct_vote_points=args.correct_vote_points,
        correct_definition_points=args.correct_definition_points,
    )

    # Create players
    game_manager.create_player(f"Player 1 - {args.player_llm_model}", args.player_llm_model)
    game_manager.create_player(f"Player 2 - {args.player_llm_model}", args.player_llm_model)
    game_manager.create_player(f"Player 3 - {args.player_llm_model}", args.player_llm_model)

    # Start the game
    game_manager.start_game(
        os.path.join("data", "meta-llama_Meta-Llama-3-8B-Instruct_balderdash_words1.csv"),
        filter_known_words="unknown",
    )


if __name__ == "__main__":
    main()
