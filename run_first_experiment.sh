#!/bin/bash

source venv/bin/activate

dry_run=false

# player_llm_models can be a string with multiple models separated by whitestpace
# supported models up to now are: "meta-llama/Meta-Llama-3-8B-Instruct", "microsoft/Phi-3-small-8k-instruct", "google/gemma-1.1-7b-it", "mistralai/Mistral-7B-Instruct-v0.3", "gpt-3.5-turbo-0125"
player_llm_models="meta-llama/Meta-Llama-3-8B-Instruct meta-llama/Meta-Llama-3-8B-Instruct google/gemma-1.1-7b-it google/gemma-1.1-7b-it"
num_players=4
judge_llm_model="meta-llama/Meta-Llama-3-8B-Instruct"
# the first gpu is for the judge, and the rest are for the players
llm_gpu_mapping=(0 0 0 2 2)
# first make a unique set of indexes from llm_gpu_mapping array
llm_gpu_mapping_set=($(echo "${llm_gpu_mapping[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))
# join the elements of llm_gpu_mapping_set array with comma and store it in gpu_indexes
gpu_indexes=$(
    IFS=,
    echo "${llm_gpu_mapping_set[*]}"
)
# join the elements of llm_gpu_mapping array with whitespace and store it in llm_gpu_mapping
llm_gpu_mapping=$(
    IFS=' '
    echo "${llm_gpu_mapping[*]}"
)
# history type can be "full", "mini", or "none"
history_type="full"
history_window_size=20
receiving_vote_points=1
correct_vote_points=2
correct_definition_points=3
llms_temperature=0.9

# words_file can be "basic_english_words.csv", or "meta-llama_Meta-Llama-3-8B-Instruct_balderdash_words1.csv",  "gpt-3.5-turbo-0125_balderdash_words1.csv"
words_file="meta-llama_Meta-Llama-3-8B-Instruct_balderdash_words1.csv"

# "game_rules.txt" or "game_rules_no_history.txt"
game_rules_prompt_file="prompts/game_rules.txt"
system_judge_prompt_file="prompts/system_judge.txt"
user_judge_prompt_file="prompts/user_judge.txt"
# "full_history.txt" or "mini_history.txt" or "none"
history_prompt_file="prompts/full_history.txt"
# "user_generate_definition_prompt_file.txt" or "user_generate_definition_no_history.txt"
user_generate_definition_prompt_file="prompts/user_generate_definition.txt"
# "user_vote_definition_prompt_file.txt" or "user_vote_definition_no_history.txt"
vote_definition_prompt_file="prompts/vote_definition.txt"

# if dry_run is true, then num_rounds has to be 1, and random_seeds has to be of length 1
if [ $dry_run = true ]; then
    num_rounds=1
    random_seed=5
    filter_words="all"
    game_description="Dry run."
    # echo the full command to be run
    echo "CUDA_VISIBLE_DEVICES=${gpu_indexes} python3 main.py \
        --game_description \"${game_description}\" \
        --player_llm_models ${player_llm_models} \
        --num_players ${num_players} \
        --judge_llm_model ${judge_llm_model} \
        --llm_gpu_mapping ${llm_gpu_mapping} \
        --history_type ${history_type} \
        --history_window_size ${history_window_size} \
        --random_seed ${random_seed} \
        --correct_vote_points ${correct_vote_points} \
        --correct_definition_points ${correct_definition_points} \
        --receiving_vote_points ${receiving_vote_points} \
        --llms_temperature ${llms_temperature} \
        --num_rounds ${num_rounds} \
        --words_file ${words_file} \
        --filter_words ${filter_words} \
        --game_rules_prompt_file ${game_rules_prompt_file} \
        --system_judge_prompt_file ${system_judge_prompt_file} \
        --user_judge_prompt_file ${user_judge_prompt_file} \
        --history_prompt_file ${history_prompt_file} \
        --user_generate_definition_prompt_file ${user_generate_definition_prompt_file} \
        --vote_definition_prompt_file ${vote_definition_prompt_file} \
        --dry_run"
    CUDA_VISIBLE_DEVICES="${gpu_indexes}" python3 main.py \
        --game_description "${game_description}" \
        --player_llm_models ${player_llm_models} \
        --num_players ${num_players} \
        --judge_llm_model ${judge_llm_model} \
        --llm_gpu_mapping ${llm_gpu_mapping} \
        --history_type ${history_type} \
        --history_window_size ${history_window_size} \
        --random_seed ${random_seed} \
        --correct_vote_points ${correct_vote_points} \
        --correct_definition_points ${correct_definition_points} \
        --receiving_vote_points ${receiving_vote_points} \
        --llms_temperature ${llms_temperature} \
        --num_rounds ${num_rounds} \
        --words_file ${words_file} \
        --filter_words ${filter_words} \
        --game_rules_prompt_file ${game_rules_prompt_file} \
        --system_judge_prompt_file ${system_judge_prompt_file} \
        --user_judge_prompt_file ${user_judge_prompt_file} \
        --history_prompt_file ${history_prompt_file} \
        --user_generate_definition_prompt_file ${user_generate_definition_prompt_file} \
        --vote_definition_prompt_file ${vote_definition_prompt_file} \
        --dry_run &
    echo "-----------------------------------"
    wait
else
    num_rounds=20
    random_seeds=(5 10 15 20 25)
    # filter words can be "all" or "known"
    filter_words="all"
    # Format the game_description string using the variables
    game_description="Convergence experiment with default scoring rules (${receiving_vote_points}, ${correct_vote_points}, ${correct_definition_points}) and ${#random_seeds[@]} different seeds, no communication, no seed stories."
fi

# Calculate total number of runs upfront
total_runs=$((${#random_seeds[@]}))

current_run=0

for random_seed in "${random_seeds[@]}"; do
    current_run=$((current_run + 1))
    echo "Running ${current_run}/${total_runs} with random seed ${random_seed}"

    # if words_file is "basic_english_words.csv", then filter_words has to be "all"
    # but if words_file is "meta-llama_Meta-Llama-3-8B-Instruct_balderdash_words1.csv", then filter_words can be "all" or "known"
    if [ $words_file = "basic_english_words.csv" ]; then
        filter_words="all"
    fi

    # echo the full command to be run
    echo "CUDA_VISIBLE_DEVICES=${gpu_indexes} python3 main.py \
        --game_description \"${game_description}\" \
        --player_llm_models ${player_llm_models} \
        --num_players ${num_players} \
        --judge_llm_model ${judge_llm_model} \
        --llm_gpu_mapping ${llm_gpu_mapping} \
        --history_type ${history_type} \
        --history_window_size ${history_window_size} \
        --random_seed ${random_seed} \
        --correct_vote_points ${correct_vote_points} \
        --correct_definition_points ${correct_definition_points} \
        --receiving_vote_points ${receiving_vote_points} \
        --llms_temperature ${llms_temperature} \
        --num_rounds ${num_rounds} \
        --words_file ${words_file} \
        --filter_words ${filter_words} \
        --game_rules_prompt_file ${game_rules_prompt_file} \
        --system_judge_prompt_file ${system_judge_prompt_file} \
        --user_judge_prompt_file ${user_judge_prompt_file} \
        --history_prompt_file ${history_prompt_file} \
        --user_generate_definition_prompt_file ${user_generate_definition_prompt_file} \
        --vote_definition_prompt_file ${vote_definition_prompt_file}"

    # Start the command in the background
    CUDA_VISIBLE_DEVICES=${gpu_indexes} python3 main.py \
        --game_description "${game_description}" \
        --player_llm_models ${player_llm_models} \
        --num_players ${num_players} \
        --judge_llm_model ${judge_llm_model} \
        --llm_gpu_mapping ${llm_gpu_mapping} \
        --history_type ${history_type} \
        --history_window_size ${history_window_size} \
        --random_seed ${random_seed} \
        --correct_vote_points ${correct_vote_points} \
        --correct_definition_points ${correct_definition_points} \
        --receiving_vote_points ${receiving_vote_points} \
        --llms_temperature ${llms_temperature} \
        --num_rounds ${num_rounds} \
        --words_file ${words_file} \
        --filter_words ${filter_words} \
        --game_rules_prompt_file ${game_rules_prompt_file} \
        --system_judge_prompt_file ${system_judge_prompt_file} \
        --user_judge_prompt_file ${user_judge_prompt_file} \
        --history_prompt_file ${history_prompt_file} \
        --user_generate_definition_prompt_file ${user_generate_definition_prompt_file} \
        --vote_definition_prompt_file ${vote_definition_prompt_file} &
    echo "-----------------------------------"
    # Wait for the background job to complete
    wait
done

# Wait for any remaining background jobs to complete
wait

# TODO: call the evaluation script here with the game_id ranges related to this experiment
# TODO: Email notification when the experiment is done
