#!/bin/bash

source venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --player_llm_model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --judge_llm_model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --history_window_size 10 \
    --random_seed 10 \
    --correct_vote_points 2 \
    --correct_definition_points 6 \
    --receiving_vote_points 1 \
    --llms_temperature 0.9 \
    --num_rounds 5 \
    --words_file "basic_english_words.csv" \
    --filter_words "all"
    # --words_file "english_frequent_words.csv"
    # --words_file "meta-llama_Meta-Llama-3-8B-Instruct_balderdash_words1.csv"
    # LLM_MODEL = "mlx-community/NeuralBeagle14-7B" # This is regular model for Mac
    # LLM_MODEL = "mlx-community/Mistral-7B-Instruct-v0.2" # This is instruct model for Mac
    # LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
