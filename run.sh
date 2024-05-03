#!/bin/bash

source venv/bin/activate

# gpus_indexes is an array of indexes of the GPUs to be used
gpus_indexes=(0 1 2)
filter_words_configs=("all" "known")
player_llm_model="meta-llama/Meta-Llama-3-8B-Instruct"
judge_llm_model="meta-llama/Meta-Llama-3-8B-Instruct"
history_window_size=10
random_seeds=(5 10 15 20 25)
correct_vote_points=2
correct_definition_points=3
receiving_vote_points=1
llms_temperature=0.9
num_rounds=50

# words_file can be "basic_english_words.csv", or "meta-llama_Meta-Llama-3-8B-Instruct_balderdash_words1.csv"
words_file="meta-llama_Meta-Llama-3-8B-Instruct_balderdash_words1.csv"

# Calculate total number of runs upfront
total_runs=$((${#random_seeds[@]} * 3))

current_run=0

for random_seed in "${random_seeds[@]}"; do
    for i in $(seq 0 2); do
        current_run=$((current_run + 1))
        echo "Running ${current_run}/${total_runs} with random seed ${random_seed}"
        gpu_index=${gpus_indexes[i]}
        echo "GPU index: ${gpu_index}"

        if [ $i -eq 2 ]; then
            words_file="basic_english_words.csv"
            filter_words="all"
        else
            words_file="meta-llama_Meta-Llama-3-8B-Instruct_balderdash_words1.csv"
            filter_words=${filter_words_configs[i]}
        fi
        echo "Words file: ${words_file}"
        echo "Filter words: ${filter_words}"

        # Start the command in the background
        CUDA_VISIBLE_DEVICES=${gpu_index} python3 main.py \
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
            --filter_words ${filter_words} &
        echo "-----------------------------------"

        # Wait for all background jobs to complete after every group of 3 jobs
        if ((current_run % 3 == 0)); then
            wait
        fi

        # sleep here for 4 second to avoid overloading the GPUs
        sleep 4
    done
done

# Wait for any remaining background jobs to complete
wait

# for 0 to len(gpus_indexes) - 1
# for i in $(seq 0 $((${#gpus_indexes[@]} - 1))); do
#     CUDA_VISIBLE_DEVICES=${gpus_indexes[i]} python3 main.py \
#         --player_llm_model ${player_llm_model} \
#         --judge_llm_model ${judge_llm_model} \
#         --history_window_size ${history_window_size} \
#         --random_seed ${random_seed} \
#         --correct_vote_points ${correct_vote_points} \
#         --correct_definition_points ${correct_definition_points} \
#         --receiving_vote_points ${receiving_vote_points} \
#         --llms_temperature ${llms_temperature} \
#         --num_rounds ${num_rounds} \
#         --words_file ${words_file} \
#         --filter_words ${filter_words_configs[i]}
# done
