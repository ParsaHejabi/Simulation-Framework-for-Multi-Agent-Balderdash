All functions needed for retrieving game data from the database, analyzing, and visualizing model performance are provided in `eval_tools.py`. Additionally, a Python script (`evaluation.py`) is available to help run evaluations on LLMs using the previously mentioned tools. The type of evaluation can be determined using input arguments. Below are the variables you can set along with their descriptions:

| **Variable**       | **Description**                                                | **Possible Values**                    |
|--------------------|----------------------------------------------------------------|----------------------------------------|
| `start_game_id`    | Integer to specify the game ID of the first game in the experiment. | Any integer, e.g., `1`                 |
| `end_game_id`      | Integer to specify the game ID of the last game in the experiment.  | Any integer, e.g., `1`                 |
| `experiment_type`  | The type of experiment being evaluated.                           | `"benchmark"`, `"regression"`          |
| `output_dir`       | Path to save results.                                             | A string, e.g., `"./results"`          |
