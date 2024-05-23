# This file is used to evaluate different LLMs as a judge using our human labeled subset of the modified_balderdash_words1.csv file.
import pandas as pd
import os
from ast import literal_eval
from utils.logger import setup_logger
from tqdm import tqdm
from create_subset_of_balderdash_words_llms_know import get_device, get_or_load_llm


llms = {}

if __name__ == "__main__":
    # Set up the logger using the name of this file
    logger = setup_logger(
        os.path.basename(__file__).replace(".py", ""),
        os.path.join("logs", f"{os.path.basename(__file__).replace('.py', '.log')}"),
    )
    judge_llm_model_name = "gpt-3.5-turbo-0125"
    random_seed = None
    device = get_device(judge_llm_model_name, judge_llm_model_name)
    logger.info(f"Device: {device}")
    judge_llm = get_or_load_llm(judge_llm_model_name, device)

    # Load the balderdash_words1_subset_human_labels.csv file
    df_path = os.path.join(os.getcwd(), "data", "balderdash_words1_subset_human_labels.csv")
    df = pd.read_csv(df_path, engine="python")

    df[f"def_deceiving_def_{judge_llm_model_name}_label"] = None
    df[f"def_correct_def_{judge_llm_model_name}_label"] = None

    word_data = {}
    for _, row in df.iterrows():
        word = row["words"].strip().lower()
        definition = literal_eval(row["def"])[0].strip()
        gpt_correct_def = row["gpt-3.5-turbo_correct_definition"].strip()
        gpt_deceiving_def = row["gpt-3.5-turbo_deceiving_definition"].strip()

        word_data[len(word_data) + 1] = {
            "word": word,
            "def": definition,
            "gpt_deceiving_def": gpt_deceiving_def,
            "gpt_correct_def": gpt_correct_def,
        }

    second_tqdm_progress_bar = tqdm(word_data.items(), total=len(word_data), leave=False)
    for _, data in second_tqdm_progress_bar:
        second_tqdm_progress_bar.set_description(f"Word: {data['word']}")
        word = data["word"]
        definition = data["def"]
        gpt_deceiving_def = data["gpt_deceiving_def"]
        gpt_correct_def = data["gpt_correct_def"]

        with open("prompts/system_judge.txt", "r") as system_judge_prompt_template_file:
            system_judge_prompt_template = system_judge_prompt_template_file.read()
        messages = [
            {
                "role": "system",
                "content": system_judge_prompt_template,
            }
        ]
        with open("prompts/user_judge.txt", "r") as user_judge_prompt_template_file:
            user_judge_prompt_template = user_judge_prompt_template_file.read()
            user_judge_prompt = user_judge_prompt_template.format(
                word=word, correct_definition=definition, definition=gpt_deceiving_def
            )
        messages.append(
            {
                "role": "user",
                "content": user_judge_prompt,
            }
        )
        judgement = judge_llm.judge_decision(word, messages)
        df.loc[df["words"] == word, f"def_deceiving_def_{judge_llm_model_name}_label"] = judgement

        with open("prompts/system_judge.txt", "r") as system_judge_prompt_template_file:
            system_judge_prompt_template = system_judge_prompt_template_file.read()
        messages = [
            {
                "role": "system",
                "content": system_judge_prompt_template,
            }
        ]
        with open("prompts/user_judge.txt", "r") as user_judge_prompt_template_file:
            user_judge_prompt_template = user_judge_prompt_template_file.read()
            user_judge_prompt = user_judge_prompt_template.format(
                word=word, correct_definition=definition, definition=gpt_correct_def
            )
        messages.append(
            {
                "role": "user",
                "content": user_judge_prompt,
            }
        )
        judgement = judge_llm.judge_decision(word, messages)
        df.loc[df["words"] == word, f"def_correct_def_{judge_llm_model_name}_label"] = judgement

    df.to_csv(df_path, index=False)
