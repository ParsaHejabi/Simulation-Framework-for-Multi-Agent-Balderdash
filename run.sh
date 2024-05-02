#!/bin/bash

source venv/bin/activate

gpus_indexes="0 1"
filter_words_configs="all known"
# models tried:
# "meta-llama/Meta-Llama-3-8B-Instruct",
# "meta-llama/Meta-Llama-3-8B-Instruct-v0.2",
# "mlx-community/Mistral-7B-Instruct-v0.2",
# "mlx-community/NeuralBeagle14-7B"
# "mistralai/Mistral-7B-Instruct-v0.2"
player_llm_model="meta-llama/Meta-Llama-3-8B-Instruct"
judge_llm_model="meta-llama/Meta-Llama-3-8B-Instruct"
history_window_size=10
random_seed=5
correct_vote_points=2
correct_definition_points=3
receiving_vote_points=1
llms_temperature=0.9
num_rounds=50
# words_file can be "basic_english_words.csv", or "meta-llama_Meta-Llama-3-8B-Instruct_balderdash_words1.csv"
words_file="meta-llama_Meta-Llama-3-8B-Instruct_balderdash_words1.csv"

# for 0 to len(gpus_indexes) - 1
for i in $(seq 0 $((${#gpus_indexes[@]} - 1))); do
    CUDA_VISIBLE_DEVICES=${gpus_indexes[i]} python3 main.py \
        --player_llm_model ${player_llm_model} \
        --judge_llm_model ${judge_llm_model} \
        --history_window_size ${history_window_size} \
        --random_seed ${random_seed} \
        --correct_vote_points ${correct_vote_points} \
        --correct_definition_points ${correct_definition_points} \
        --receiving_vote_points ${receiving_vote_points} \
        --llms_temperature ${llms_temperature} \
        --num_rounds ${num_rounds} \
        --words_file ${words_file} \
        --filter_words ${filter_words_configs[i]}
done
