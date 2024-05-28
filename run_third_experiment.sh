#!/bin/bash

source venv/bin/activate

dry_run=false

# player_llm_models can be a string with multiple models separated by whitestpace
# supported models up to now are: "meta-llama/Meta-Llama-3-8B-Instruct", "microsoft/Phi-3-small-8k-instruct", "google/gemma-1.1-7b-it", "mistralai/Mistral-7B-Instruct-v0.3", "gpt-3.5-turbo-0125"
player_llm_models_list="meta-llama/Meta-Llama-3-8B-Instruct microsoft/Phi-3-small-8k-instruct google/gemma-1.1-7b-it mistralai/Mistral-7B-Instruct-v0.3 gpt-3.5-turbo-0125"
# correct_definition_points should be two numbers 0 and 50
correct_definition_points_list="0 50"

current_run=0

# for each player_llm_models in player_llm_models_list
for player_llm_models in $player_llm_models_list; do
    # for each correct_definition_points in correct_definition_points_list
    for correct_definition_points in $correct_definition_points_list; do
        num_players=1
        judge_llm_model="meta-llama/Meta-Llama-3-8B-Instruct"
        # the first gpu is for the judge, and the rest are for the players
        llm_gpu_mapping=(0 1)
        # gpu_indexes in this device are from 0 to 7
        gpu_indexes="0,1,2"
        # join the elements of llm_gpu_mapping array with whitespace and store it in llm_gpu_mapping
        llm_gpu_mapping=$(
            IFS=' '
            echo "${llm_gpu_mapping[*]}"
        )
        history_window_size=20
        history_type="none"

        receiving_vote_points=1
        correct_vote_points=2
        llms_temperature=0.9

        # words_file is $player_llm_models after replacing '/' with '_' and appending "_balderdash_words1.csv"
        words_file=$(echo $player_llm_models | tr '/' '_')"_balderdash_words1.csv"
        # filter words is "known" for this experiment
        filter_words="known"

        system_judge_prompt_file="prompts/system_judge.txt"
        user_judge_prompt_file="prompts/user_judge.txt"
        game_rules_prompt_file="prompts/game_rules_no_history.txt"
        user_generate_definition_prompt_file="prompts/user_generate_definition_no_history.txt"
        vote_definition_prompt_file="prompts/vote_definition_no_history.txt"
        history_prompt_file="none"

        # if dry_run is true, then num_rounds has to be 1, and random_seeds has to be of length 1
        if [ $dry_run = true ]; then
            num_rounds=1
            random_seed=5
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
            # Calculate total number of runs upfront
            total_runs=$((${#random_seeds[@]} * (${#player_llm_models_list[@]} * ${#correct_definition_points_list[@]})))
            # Format the game_description string using the variables
            game_description="Effect of rules without history experiment for player_llm_models: ${player_llm_models}, with modified scoring rules (${receiving_vote_points}, ${correct_vote_points}, ${correct_definition_points}) and ${#random_seeds[@]} different seeds, no communication, no seed stories."
        fi

        for random_seed in "${random_seeds[@]}"; do
            current_run=$((current_run + 1))
            echo "Running ${current_run}/${total_runs} with player LLM model ${player_llm_models}, correct_definition_points ${correct_definition_points}, and random seed ${random_seed}"

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
    done
done

# Wait for all background jobs to complete
wait
