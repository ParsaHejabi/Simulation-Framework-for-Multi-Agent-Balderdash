# Evaluating Creativity and Deception in Large Language Models: A Simulation Framework for Multi-Agent Balderdash

By Parsa Hejabi, Elnaz Rahmati, Alireza Salkhordeh Ziabari, Preni Golazizian, Jesse Thomason, Morteza Dehghani - University of Southern California

## Abstract

Large Language Models (LLMs) have shown impressive capabilities in complex tasks and interactive environments, yet their creativity remains underexplored. This paper introduces a simulation framework utilizing the game Balderdash to evaluate both the creativity and logical reasoning of LLMs. In Balderdash, players generate fictitious definitions for obscure terms to deceive others while identifying correct definitions. Our framework enables multiple LLM agents to participate in this game, assessing their ability to produce plausible definitions and strategize based on game rules and history. We implemented a centralized game engine featuring various LLMs as participants and a judge LLM to evaluate semantic equivalence. Through a series of experiments, we analyzed the performance of different LLMs, examining metrics such as True Definition Ratio, Deception Ratio, and Correct Guess Ratio. The results provide insights into the creative and deceptive capabilities of LLMs, highlighting their strengths and areas for improvement. Specifically, the study reveals that infrequent vocabulary in LLMs' input leads to poor reasoning on game rules and historical context.

## Prerequisites

- python 3.10+
- Ubuntu GPU-enabled server with CUDA 12.1+
  - Check your GPUs with `nvidia-smi`
- python environment with packages installed as in requirements.txt
- A local [MongoDB server](https://www.mongodb.com/docs/manual/installation/) or Atlas cluster, and a connection string
- An [OpenAI account](https://platform.openai.com/docs/overview), with an API key and a project ID
- [Hugging Face](https://huggingface.co/) account, with an access token for the gated models

## Setup Environment

```bash
# Clone the repository
git clone git@github.com:ParsaHejabi/Simulation-Framework-for-Multi-Agent-Balderdash.git
# Change directory to the root of the repository
cd ROOT_OF_THE_REPO
# Create an empty directory for data
mkdir data
# Create a .env file and fill in the required fields
echo -e "OPENAI_API_KEY=\nOPENAI_PROJECT_ID=\nMONGODB_CONNECTION_STRING=\nMONGODB_USERNAME=\nMONGODB_PASSWORD=" > .env
# Create a virtual environment
python3 -m venv venv
# Activate the virtual environment
source venv/bin/activate
# Install the required packages
pip install -r requirements.txt
```

## Datasets

Please contact the corresponding author for access to the datasets used in this study.

## Running the Experiments

- `run_first_experiment.py`: This script runs the leaderboard experiment on two datasets: "Basic Frequent English Words" and "All Balderdash," five different random seeds, four different players (LLMs) in each game, and three different settings based on history type (HT) with the ruling score of 1, 2, and 3 for receiving votes, correct votes, and correct definitions, respectively.
- `run_second_experiment.py`: This script runs the convergence experiment on the known words dataset for each LLM in `all` and `mini` history types with three players of the same LLM in each game, and the ruling score of 1, 2, and 3 for receiving votes, correct votes, and correct definitions, respectively.
- `run_third_experiment.py`: This script runs the game rules experiment on the known words dataset for each LLM in `none` history type with one player (LLM) in each game, and the ruling score of 1, 2, 0 in one setting and 1, 2, 50 in the other setting for receiving votes, correct votes, and correct definitions, respectively.

Here are the variables that you can set along with their descriptions:

| **Variable**                              | **Description**                                                                                                 | **Possible Values**                                                                                           |
|-------------------------------------------|-----------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `dry_run`                                 | Flag to determine if the script is in dry run mode.                                                             | `true`, `false`                                                                                               |
| `words_files_list`                        | List of words files to be used in the experiments.                                                              | `"meta-llama_Meta-Llama-3-8B-Instruct_balderdash_words1.csv"`, `"basic_english_words.csv"`                    |
| `history_types_list`                      | List of history types to be used in the experiments.                                                            | `"full"`, `"mini"`, `"none"`                                                                                  |
| `current_run`                             | Counter for the current run of the experiment.                                                                  | Any integer starting from `0`                                                                                 |
| `player_llm_models`                       | List of player LLM models to be used in the experiment.                                                         | `"meta-llama/Meta-Llama-3-8B-Instruct"`, `"microsoft/Phi-3-small-8k-instruct"`, `"google/gemma-1.1-7b-it"`, `"mistralai/Mistral-7B-Instruct-v0.3"` |
| `num_players`                             | Number of players in the experiment.                                                                            | Any integer (example: `4`)                                                                                    |
| `judge_llm_model`                         | LLM model to be used as the judge.                                                                              | `"meta-llama/Meta-Llama-3-8B-Instruct"`                                                                       |
| `llm_gpu_mapping`                         | Mapping of GPUs to be used by LLMs. The first one is used for the judge LLM and starting the second one, the numbers are used for loading nth player to the nth GPU. | Array of integers (example: `(0 0 1 2 0)`)                                                                     |
| `gpu_indexes`                             | Indexes of GPUs available on the device.                                                                        | Comma-separated list of integers (example: `"0,1,2"`)                                                         |
| `history_window_size`                     | Size of the history window for the experiment.                                                                  | Any integer (example: `20`)                                                                                   |
| `receiving_vote_points`                   | Points awarded for receiving votes.                                                                             | Any integer (example: `1`)                                                                                    |
| `correct_vote_points`                     | Points awarded for correct votes.                                                                               | Any integer (example: `2`)                                                                                    |
| `correct_definition_points`               | Points awarded for correct definitions.                                                                         | Any integer (example: `3`)                                                                                    |
| `llms_temperature`                        | Temperature setting for the LLMs.                                                                               | Any float (example: `0.9`)                                                                                    |
| `filter_words`                            | Filter applied to the words for the experiment.                                                                 | `"all"`, `"known"`, `"unknown"`                                                                               |
| `system_judge_prompt_file`                | Path to the system judge prompt file.                                                                           | `"prompts/system_judge.txt"`                                                                                   |
| `user_judge_prompt_file`                  | Path to the user judge prompt file.                                                                             | `"prompts/user_judge.txt"`                                                                                     |
| `game_rules_prompt_file`                  | Path to the game rules prompt file.                                                                             | `"prompts/game_rules.txt"`, `"prompts/game_rules_no_history.txt"`                                              |
| `user_generate_definition_prompt_file`    | Path to the user generate definition prompt file.                                                              | `"prompts/user_generate_definition.txt"`, `"prompts/user_generate_definition_no_history.txt"`                  |
| `vote_definition_prompt_file`             | Path to the vote definition prompt file.                                                                        | `"prompts/vote_definition.txt"`, `"prompts/vote_definition_no_history.txt"`                                    |
| `num_rounds`                              | Number of rounds in the experiment.                                                                             | Any integer (example: `20`)                                                                                   |
| `random_seed`                             | Random seed value for reproducibility.                                                                          | Any integer (example: `5`)                                                                                    |
| `game_description`                        | Description of the game being run.                                                                              | Any string (example: `"Leaderboard experiment with words file basic_english_words.csv, history type full."`)   |

## Evaluation

All functions needed for retrieving game data from the database, analyzing, and visualizing model performance are provided in `evaluation/eval_tools.py`. Additionally, a Python script (`evaluation/evaluation.py`) is available to help run evaluations on LLMs using the previously mentioned tools. The type of evaluation can be determined using input arguments. Below are the variables you can set along with their descriptions:

| **Variable**       | **Description**                                                | **Possible Values**                    |
|--------------------|----------------------------------------------------------------|----------------------------------------|
| `start_game_id`    | Integer to specify the game ID of the first game in the experiment. | Any integer, e.g., `1`                 |
| `end_game_id`      | Integer to specify the game ID of the last game in the experiment.  | Any integer, e.g., `1`                 |
| `experiment_type`  | The type of experiment being evaluated.                           | `"benchmark"`, `"regression"`          |
| `output_dir`       | Path to save results.                                             | A string, e.g., `"./results"`          |

## Prompts

All prompts are stored in the `prompts` directory. You can change the prompts by editing the files in this directory. The prompts are being rendered using jinja2 templates, so you can use variables in the prompts.

## Leaderboard Experiment Results

| **HT** | **LLM**   | **LKR**          | **TDR**          | **DR**           | **CGR**          | **AS**           |
|--------|-----------|------------------|------------------|------------------|------------------|------------------|
| none       | Llama     | 0.59 ± 0.10      | 0.40 ± 0.04      | 0.19 ± 0.12      | 0.56 ± 0.11      | 2.08 ± 0.19      |
|    | Phi       | 0.49 ± 0.12      | 0.33 ± 0.10      | 0.24 ± 0.13      | **0.77 ± 0.12**  | 2.21 ± 0.30      |
|        | Gemma     | **0.93 ± 0.02**  | **0.77 ± 0.08**  | **0.25 ± 0.13**  | 0.47 ± 0.19      | **2.65 ± 0.13**  |
|        | Mistral   | 0.59 ± 0.13      | 0.48 ± 0.17      | 0.15 ± 0.07      | 0.74 ± 0.13      | 2.35 ± 0.29      |
| mini       | Llama     | 0.89 ± 0.12      | 0.78 ± 0.14      | 0.28 ± 0.18      | 0.49 ± 0.20      | 2.68 ± 0.15      |
|    | Phi       | 0.74 ± 0.18      | 0.60 ± 0.18      | 0.30 ± 0.17      | **0.74 ± 0.07**  | 2.52 ± 0.22      |
|        | Gemma     | 0.92 ± 0.09      | 0.78 ± 0.12      | 0.30 ± 0.28      | 0.43 ± 0.16      | 2.63 ± 0.21      |
|        | Mistral   | **0.94 ± 0.07**  | **0.83 ± 0.07**  | **0.34 ± 0.18**  | 0.35 ± 0.32      | **2.76 ± 0.06**  |
| full       | Llama     | 0.93 ± 0.06      | 0.78 ± 0.16      | 0.61 ± 0.07      | **0.52 ± 0.31**  | 2.72 ± 0.18      |
|    | Phi       | 0.98 ± 0.04      | 0.85 ± 0.05      | 0.53 ± 0.34      | 0.42 ± 0.26      | 2.79 ± 0.07      |
|        | Gemma     | 0.97 ± 0.04      | 0.84 ± 0.07      | 0.52 ± 0.29      | 0.44 ± 0.34      | 2.69 ± 0.21      |
|        | Mistral   | **1.00 ± 0.00**  | **0.90 ± 0.05**  | **0.62 ± 0.35**  | 0.05 ± 0.10      | **2.81 ± 0.06**  |

**Table 1**: Leaderboard experiment results on "Basic Frequent English Words," evaluating each LLM in three different settings based on history type (HT) using the average of LKR, TDR, DR, CGR, and AS metrics over all rounds and games. The highest value of each metric for different game settings is in bold. However, based on the standard deviation, this does not represent absolute superiority.

| **HT**  | **LLM**   | **LKR**          | **TDR**          | **DR**           | **CGR**          | **AS**           |
|---------|-----------|------------------|------------------|------------------|------------------|------------------|
| none        | Llama     | 0.27 ± 0.09      | 0.22 ± 0.11      | 0.26 ± 0.06      | 0.57 ± 0.12      | 2.00 ± 0.25      |
|     | Phi       | 0.31 ± 0.13      | 0.25 ± 0.11      | 0.19 ± 0.05      | **0.62 ± 0.05**  | **2.08 ± 0.22**  |
|         | Gemma     | 0.18 ± 0.10      | 0.15 ± 0.08      | 0.12 ± 0.03      | 0.35 ± 0.08      | 1.26 ± 0.15      |
|         | Mistral   | **0.44 ± 0.17**  | **0.33 ± 0.17**  | **0.27 ± 0.04**  | 0.45 ± 0.08      | 2.05 ± 0.28      |
| mini        | Llama     | 0.29 ± 0.15      | 0.24 ± 0.12      | **0.30 ± 0.07**  | 0.54 ± 0.12      | 2.04 ± 0.17      |
|     | Phi       | **0.45 ± 0.16**  | **0.35 ± 0.15**  | 0.26 ± 0.09      | **0.56 ± 0.13**  | **2.28 ± 0.19**  |
|         | Gemma     | 0.07 ± 0.07      | 0.05 ± 0.05      | 0.10 ± 0.06      | 0.32 ± 0.12      | 0.96 ± 0.35      |
|         | Mistral   | 0.44 ± 0.17      | **0.35 ± 0.13**  | **0.30 ± 0.07**  | 0.41 ± 0.11      | 2.05 ± 0.25      |
| full        | Llama     | 0.33 ± 0.16      | 0.24 ± 0.15      | 0.20 ± 0.06      | 0.39 ± 0.12      | 1.70 ± 0.27      |
|     | Phi       | **0.40 ± 0.14**  | **0.37 ± 0.14**  | **0.33 ± 0.08**  | **0.66 ± 0.13**  | **2.52 ± 0.25**  |
|         | Gemma     | 0.17 ± 0.10      | 0.13 ± 0.10      | 0.19 ± 0.09      | 0.26 ± 0.13      | 1.19 ± 0.48      |
|         | Mistral   | 0.36 ± 0.18      | 0.31 ± 0.15      | 0.28 ± 0.06      | 0.36 ± 0.12      | 1.89 ± 0.27      |

**Table 2**: Leaderboard experiment results on "All Balderdash," evaluating each LLM in three different settings based on history type (HT) using the average of LKR, TDR, DR, CGR, and AS metrics over all rounds and games. The highest value of each metric for different game settings is in bold. However, based on the standard deviation, this does not represent absolute superiority.

## Reference

If you use this framework in your research, please cite the following paper:

```
@inproceedings{hejabievaluating,
  title={Evaluating Creativity and Deception in Large Language Models: A Simulation Framework for Multi-Agent Balderdash},
  author={Hejabi, Parsa and Rahmati, Elnaz and Ziabari, Alireza S and Golazizian, Preni and Thomason, Jesse and Dehghani, Morteza},
  booktitle={The 4th Wordplay: When Language Meets Games @ ACL 2024},
  year={2024},
  organization={Association for Computational Linguistics},
  address={Mexico City, Mexico}
}
```
