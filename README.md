# Evaluating Creativity and Deception in Large Language Models: A Simulation Framework for Multi-Agent Balderdash  
Large Language Models (LLMs) have shown impressive capabilities in complex tasks and interactive environments, yet their creativity remains underexplored. This paper introduces a simulation framework utilizing the game Balderdash to evaluate both the creativity and logical reasoning of LLMs. In Balderdash, players generate fictitious definitions for obscure terms to deceive others while identifying correct definitions. Our framework enables multiple LLM agents to participate in this game, assessing their ability to produce plausible definitions and strategize based on game rules and history. We implemented a centralized game engine featuring various LLMs as participants and a judge LLM to evaluate semantic equivalence. Through a series of experiments, we analyzed the performance of different LLMs, examining metrics such as True Definition Ratio, Deception Ratio, and Correct Guess Ratio. The results provide insights into the creative and deceptive capabilities of LLMs, highlighting their strengths and areas for improvement. Specifically, the study reveals that infrequent vocabulary in LLMs' input leads to poor reasoning on game rules and historical context.

## Leaderboard 
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

**Table**: Leaderboard experiment results on "Basic Frequent English Words," evaluating each LLM in three different settings based on history type (HT) using the average of LKR, TDR, DR, CGR, and AS metrics over all rounds and games. The highest value of each metric for different game settings is in bold. However, based on the standard deviation, this does not represent absolute superiority.

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

**Table**: Leaderboard experiment results on "All Balderdash," evaluating each LLM in three different settings based on history type (HT) using the average of LKR, TDR, DR, CGR, and AS metrics over all rounds and games. The highest value of each metric for different game settings is in bold. However, based on the standard deviation, this does not represent absolute superiority.


## Datasets
The datasets proposed and used in this work can be found in this [link](https://drive.google.com/drive/folders/18PBf4619-d532kj1I-GscntPLQNLFrb6?usp=sharing).
