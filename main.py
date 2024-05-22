from game.game_manager import GameManager
from dotenv import load_dotenv
import os
import argparse

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_description", type=str, help="Game description")
    parser.add_argument("--random_seed", type=int, help="Random seed value")
    parser.add_argument("--history_window_size", type=int, help="History window size")
    parser.add_argument("--player_llm_models", type=str, nargs="+", help="Player LLM model names")
    parser.add_argument("--num_players", type=int, help="Number of players")
    parser.add_argument("--judge_llm_model", type=str, help="Judge LLM model name")
    parser.add_argument("--llm_gpu_mapping", type=int, nargs="+", help="LLM GPU mapping")
    parser.add_argument("--correct_vote_points", type=int, help="Correct vote points")
    parser.add_argument("--correct_definition_points", type=int, help="Correct definition points")
    parser.add_argument("--receiving_vote_points", type=int, help="Receiving vote points")
    parser.add_argument("--llms_temperature", type=float, help="LLMs temperature")
    parser.add_argument("--num_rounds", type=int, help="Number of rounds")
    parser.add_argument("--words_file", type=str, help="Words file")
    parser.add_argument("--filter_words", type=str, help="Filter words", choices=["known", "unknown", "all"])
    parser.add_argument("--dry_run", action="store_true", help="Dry run")
    args = parser.parse_args()

    if args.dry_run:
        # It is a dry run, just one round with one word, not saving to the database
        if args.num_rounds != 1:
            raise ValueError("Number of rounds must be 1 for dry run")

    game_manager = GameManager(
        db_connection_string=os.getenv("MONGODB_CONNECTION_STRING"),
        game_description=args.game_description,
        random_seed=args.random_seed,
        judge_llm_model_name=args.judge_llm_model,
        judge_llm_gpu=args.llm_gpu_mapping[0],
        llms_temperature=args.llms_temperature,
        history_window_size=args.history_window_size,
        receiving_vote_points=args.receiving_vote_points,
        correct_vote_points=args.correct_vote_points,
        correct_definition_points=args.correct_definition_points,
        num_rounds=args.num_rounds,
        words_file=args.words_file,
        filter_words=args.filter_words,
        dry_run=args.dry_run,
    )

    # For each model, we want at least a single GPU
    all_llm_models = set(args.player_llm_models + [args.judge_llm_model])
    if len(all_llm_models) > len(args.llm_gpu_mapping):
        raise ValueError(
            "Number of LLM GPU mappings must be equal or greater than the number of unique LLM models"
        )

    if len(args.player_llm_models) != 1:
        if len(args.player_llm_models) != args.num_players:
            raise ValueError("Number of player LLM models must be 1 or equal to the number of players")
        else:
            # Number of player LLM models is equal to the number of players, create players with different models
            for index, player_llm_model in enumerate(args.player_llm_models):
                # First GPU is for the judge, so we start from the second GPU
                game_manager.create_player(
                    f"Player {index + 1}",
                    game_manager.game.game_id,
                    player_llm_model,
                    args.llm_gpu_mapping[index + 1],
                )
    else:
        # Number of player LLM models is 1, create players with the same model and the gpu mapping is not important
        for index in range(args.num_players):
            game_manager.create_player(
                f"Player {index + 1}",
                game_manager.game.game_id,
                args.player_llm_models[0],
                args.llm_gpu_mapping[index + 1],
            )

    # Start the game
    game_manager.start_game(os.path.join("data", args.words_file))


if __name__ == "__main__":
    main()
