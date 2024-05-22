#!/bin/bash

source venv/bin/activate

dry_run=false

# player_llm_models can be a string with multiple models separated by whitestpace
player_llm_models="meta-llama/Meta-Llama-3-8B-Instruct microsoft/Phi-3-small-8k-instruct google/gemma-1.1-7b-it"
num_players=3
judge_llm_model="meta-llama/Meta-Llama-3-8B-Instruct"
# the first gpu is for the judge, and the rest are for the players
llm_gpu_mapping=(0 0 0 1)
# first make a unique set of indexes from llm_gpu_mapping array
llm_gpu_mapping_set=($(echo "${llm_gpu_mapping[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))
# join the elements of llm_gpu_mapping_set array with comma and store it in gpu_indexes
gpu_indexes=$(IFS=,; echo "${llm_gpu_mapping_set[*]}")
# join the elements of llm_gpu_mapping array with whitespace and store it in llm_gpu_mapping
llm_gpu_mapping=$(IFS=' '; echo "${llm_gpu_mapping[*]}")
history_window_size=10
correct_vote_points=2
receiving_vote_points=1
llms_temperature=0.9

# In the second experiment, we will only use "meta-llama_Meta-Llama-3-8B-Instruct_balderdash_words1.csv" wordlist.
words_file="meta-llama_Meta-Llama-3-8B-Instruct_balderdash_words1.csv"
filter_words="known"

# if dry_run is true, then num_rounds has to be 1, and random_seeds has to be of length 1
if [ $dry_run = true ]; then
    num_rounds=1
    random_seed=5
    correct_definition_points=6
    game_description="Dry run."
    # echo the full command to be run
    echo "CUDA_VISIBLE_DEVICES=${gpu_indexes} python3 main.py \
        --game_description \"${game_description}\" \
        --player_llm_models ${player_llm_models} \
        --num_players ${num_players} \
        --judge_llm_model ${judge_llm_model} \
        --llm_gpu_mapping ${llm_gpu_mapping} \
        --history_window_size ${history_window_size} \
        --random_seed ${random_seed} \
        --correct_vote_points ${correct_vote_points} \
        --correct_definition_points ${correct_definition_points} \
        --receiving_vote_points ${receiving_vote_points} \
        --llms_temperature ${llms_temperature} \
        --num_rounds ${num_rounds} \
        --words_file ${words_file} \
        --filter_words ${filter_words} \
        --dry_run"
    CUDA_VISIBLE_DEVICES="${gpu_indexes}" python3 main.py \
        --game_description "${game_description}" \
        --player_llm_models ${player_llm_models} \
        --num_players ${num_players} \
        --judge_llm_model ${judge_llm_model} \
        --llm_gpu_mapping ${llm_gpu_mapping} \
        --history_window_size ${history_window_size} \
        --random_seed ${random_seed} \
        --correct_vote_points ${correct_vote_points} \
        --correct_definition_points ${correct_definition_points} \
        --receiving_vote_points ${receiving_vote_points} \
        --llms_temperature ${llms_temperature} \
        --num_rounds ${num_rounds} \
        --words_file ${words_file} \
        --filter_words ${filter_words} \
        --dry_run &
    echo "-----------------------------------"
    wait
else
    game_description="Correct definition rule experiment with default scoring rules but correct definition score changing from 0 to 6 and five different seeds, no communication, no seed stories."
    num_rounds=50
    random_seeds=(5 10 15 20 25)
    # correct definition points will be changed from 0 to 6
    correct_definition_points=(0 1 2 3 4 5 6)

# Calculate total number of runs upfront
total_runs=$((${#random_seeds[@]} * ${#correct_definition_points[@]}))

current_run=0

for random_seed in "${random_seeds[@]}"; do
    for i in $(seq 0 $((${#correct_definition_points[@]} - 1))); do
        current_run=$((current_run + 1))
        echo "Running ${current_run}/${total_runs} with random seed ${random_seed} and correct definition points ${correct_definition_points[$i]}"
        correct_definition_points=${correct_definition_points[$i]}

        # echo the full command to be run
        echo "CUDA_VISIBLE_DEVICES=${gpu_indexes} python3 main.py \
            --game_description \"${game_description}\" \
            --player_llm_models ${player_llm_models} \
            --num_players ${num_players} \
            --judge_llm_model ${judge_llm_model} \
            --llm_gpu_mapping ${llm_gpu_mapping} \
            --history_window_size ${history_window_size} \
            --random_seed ${random_seed} \
            --correct_vote_points ${correct_vote_points} \
            --correct_definition_points ${correct_definition_points} \
            --receiving_vote_points ${receiving_vote_points} \
            --llms_temperature ${llms_temperature} \
            --num_rounds ${num_rounds} \
            --words_file ${words_file} \
            --filter_words ${filter_words}"

        # Start the command in the background
        CUDA_VISIBLE_DEVICES=${gpu_indexes} python3 main.py \
            --game_description "${game_description}" \
            --player_llm_models ${player_llm_models} \
            --num_players ${num_players} \
            --judge_llm_model ${judge_llm_model} \
            --llm_gpu_mapping ${llm_gpu_mapping} \
            --history_window_size ${history_window_size} \
            --random_seed ${random_seed} \
            --correct_vote_points ${correct_vote_points} \
            --correct_definition_points ${correct_definition_points} \
            --receiving_vote_points ${receiving_vote_points} \
            --llms_temperature ${llms_temperature} \
            --num_rounds ${num_rounds} \
            --words_file ${words_file} \
            --filter_words ${filter_words} &
        echo "-----------------------------------"
        # Wait for the background job to complete
        wait
    done
done

# Wait for any remaining background jobs to complete
wait
