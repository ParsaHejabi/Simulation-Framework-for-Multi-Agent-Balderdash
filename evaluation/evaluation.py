import pandas as pd
from eval_tools import evaluation_tools
import os
import argparse
import scipy.stats as stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_game_id", type=int, help="Starting game id of the experiment")
    parser.add_argument("--end_game_id", type=int, help="Ending game id of the experiment")
    parser.add_argument(
        "--experiment_type", type=str, help="Type of experiment can be 'benchmark' or 'regression'"
    )
    parser.add_argument("--output_dir", type=str, help="Output directory to save the results", default=".")

    args = parser.parse_args()

    etools = evaluation_tools(os.getenv("MONGODB_CONNECTION_STRING"), "balderdash_game")

    experiments = etools.get_experiment_data(args.start_game_id, args.end_game_id)
    if args.experiment_type == "benchmark":
        for key, experiment in experiments.items():
            _, overal_avg = etools.experiment_game_average(experiment)
            table = {}
            for llm, metrics in overal_avg.items():
                table[llm] = {}
                for metric, values in metrics.items():
                    mean = round(values["mean"], 2)
                    std = round(values["std"], 2)
                    table[llm][metric] = f"{mean} ± {std}"

            df = pd.DataFrame(table).T
            df.index.name = "LLM"
            print(df)
            df.to_csv(
                os.path.join(
                    args.output_dir,
                    f"benchmark_experiment_{args.start_game_id}_{args.end_game_id}_{key[1]}_{key[0]}.csv",
                )
            )
    elif args.experiment_type == "regression":
        regression_stats = {}
        for key, experiment in experiments.items():
            _, overal_avg = etools.experiment_game_average(experiment)
            table = {}
            for llm, metrics in overal_avg.items():
                regression_stats.setdefault(llm, {})[key[4]] = metrics["true_def_ratio"]["mean"]
        regression_results = {}
        for llm, data in regression_stats.items():
            regression_results[llm] = {}
            (
                regression_results[llm]["slope"],
                regression_results[llm]["intercept"],
                regression_results[llm]["r_value"],
                regression_results[llm]["p_value"],
                regression_results[llm]["std_err"],
            ) = stats.linregress(list(data.keys()), list(data.values()))
        df = pd.DataFrame(regression_results).T
        df.index.name = "LLM"
        print(df)
        df.to_csv(
            os.path.join(
                args.output_dir, f"regression_experiment_{args.start_game_id}_{args.end_game_id}.csv"
            )
        )


if __name__ == "__main__":
    main()
