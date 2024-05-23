# This file is used to create a subset of the modified_balderdash_words1.csv file that only contains words that the LLMs (or our particular LLM: Llama 3) know.
from utils.llm import LLM, is_api_model
import pandas as pd
import os
import torch
from ast import literal_eval
from utils.logger import setup_logger
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional

load_dotenv()


llms = {}


def get_device(llm_model_name: str, judge_llm_model_name: str) -> torch.device:
    if is_api_model(llm_model_name) and is_api_model(judge_llm_model_name):
        return torch.device("cpu")

    if not torch.backends.mps.is_available():
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        return torch.device("mps")


def get_or_load_llm(model_name: str, device: torch.device, random_seed: Optional[int] = None) -> LLM:
    if model_name not in llms:
        llms[model_name] = LLM(device=device, model_name=model_name, temp=0.9, random_seed=random_seed)
    return llms[model_name]


if __name__ == "__main__":
    # Set up the logger using the name of this file
    logger = setup_logger(
        os.path.basename(__file__).replace(".py", ""),
        os.path.join("logs", f"{os.path.basename(__file__).replace('.py', '.log')}"),
    )
    llm_model_name = "gpt-3.5-turbo-0125"
    judge_llm_model_name = "gpt-3.5-turbo-0125"
    random_seed = None
    device = get_device(llm_model_name, judge_llm_model_name)
    logger.info(f"Device: {device}")

    llm = get_or_load_llm(llm_model_name, device, random_seed=random_seed)
    judge_llm = get_or_load_llm(judge_llm_model_name, device, random_seed=random_seed)

    if llm.is_api_model or judge_llm.is_api_model:
        if llm.model_name.startswith("gpt-") or judge_llm.model_name.startswith("gpt-"):
            client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                project=os.getenv("OPENAI_PROJECT_ID"),
            )

    # Load the modified_balderdash_words1.csv file
    df = pd.read_csv(os.path.join(os.getcwd(), "data", "modified_balderdash_words1.csv"), engine="python")

    word_data = {}
    for _, row in df.iterrows():
        word = row["words"].strip().lower()
        definition = literal_eval(row["def"])[0].strip()
        pos = literal_eval(row["POS"])[0]
        if pos:
            pos = pos.strip()
        word_data[len(word_data) + 1] = {"word": word, "def": definition, "POS": pos}

    # Create the following columns in the dataframe: "llm1_definition", "llm2_definition", "llm3_definition", "llm4_definition", "llm5_definition", all with string data type and  "llm1_judgement", "llm2_judgement", "llm3_judgement", "llm4_judgement", "llm5_judgement", all with boolean data type
    for i in range(1, 6):
        df[f"llm{i}_definition"] = ""
        df[f"llm{i}_judgement"] = False

    first_tqdm_progress_bar = tqdm(range(1, 6))
    # Get definitions for each word from the LLM five times, then pass the correct definition and each of the five generated definitions to the judge LLM
    for i in first_tqdm_progress_bar:
        first_tqdm_progress_bar.set_description(f"Round {i}")
        second_tqdm_progress_bar = tqdm(word_data.items(), total=len(word_data), leave=False)
        for _, data in second_tqdm_progress_bar:
            second_tqdm_progress_bar.set_description(f"Word: {data['word']}")
            word = data["word"]
            definition = data["def"]
            pos = data["POS"]

            messages = [
                {
                    "role": "system",
                    "content": f"You are a universal dictionary. Your task is to provide the definition of a word given to you along with its part of speech. Use at most one sentence.",
                }
            ]
            messages.append({"role": "user", "content": f"{word} ({pos}): "})
            if llm.is_api_model:
                if llm.model_name.startswith("gpt-"):
                    try:
                        logger.info(f"Calling OpenAI chat/completion with messages: {messages}")
                        completion = client.chat.completions.create(
                            model=llm.model_name,
                            messages=messages,
                            max_tokens=llm.max_tokens,
                            seed=llm.random_seed,
                        )
                        if completion.choices[0].finish_reason == "length":
                            raise ValueError(
                                f"Error: OpenAI llm model {llm.model_name} output for messages: {completion.choices[0].message.content} has reached max tokens {llm.max_tokens}"
                            )
                        elif completion.choices[0].finish_reason == "null":
                            raise ValueError(
                                f"Error: OpenAI llm model {llm.model_name} output for messages: API response still in progress or incomplete"
                            )
                        elif completion.choices[0].finish_reason == "stop":
                            df.loc[df["words"] == word, f"llm{i}_definition"] = completion.choices[
                                0
                            ].message.content.strip()
                    except ValueError as e:
                        logger.error(e)
                        df.loc[df["words"] == word, f"llm{i}_definition"] = -1
            else:
                df.loc[df["words"] == word, f"llm{i}_definition"] = llm.generate_answer(messages).strip()

            if df.loc[df["words"] == word, f"llm{i}_definition"].values[0] == -1:
                df.loc[df["words"] == word, f"llm{i}_judgement"] = -1
                continue
            else:
                messages = [
                    {
                        "role": "system",
                        "content": f"You are a semantical equivalence judge. Your task is to determine whether a given definition is semantically equal to the actual definition of a word. Give your answer as a single word, either 'true' or 'false'.",
                    }
                ]
                messages.append(
                    {
                        "role": "user",
                        "content": f"Actual definition: {definition} and generated definition: {df.loc[df['words'] == word, f'llm{i}_definition'].values[0]}. Your judgement: ",
                    }
                )
                if judge_llm.is_api_model:
                    if judge_llm.model_name.startswith("gpt-"):
                        try:
                            logger.info(f"Calling OpenAI chat/completion with messages: {messages}")
                            completion = client.chat.completions.create(
                                model=judge_llm.model_name,
                                messages=messages,
                                max_tokens=judge_llm.max_tokens,
                                seed=judge_llm.random_seed,
                            )
                            if completion.choices[0].finish_reason == "length":
                                raise ValueError(
                                    f"Error: OpenAI judge llm model {judge_llm.model_name} output for messages: {completion.choices[0].message.content} has reached max tokens {judge_llm.max_tokens}"
                                )
                            elif completion.choices[0].finish_reason == "null":
                                raise ValueError(
                                    f"Error: OpenAI judge llm model {judge_llm.model_name} output for messages: API response still in progress or incomplete"
                                )
                            elif completion.choices[0].finish_reason == "stop":
                                judgement = completion.choices[0].message.content
                        except ValueError as e:
                            logger.error(e)
                            df.loc[df["words"] == word, f"llm{i}_judgement"] = -2
                            continue
                else:
                    judgement = judge_llm.generate_answer(messages)
                logger.info(
                    f"word: {word}, POS: {pos}, definition: {definition}\ngenerated definition: {df.loc[df['words'] == word, f'llm{i}_definition'].values[0]}\njudgement: {judgement}"
                )
                df.loc[df["words"] == word, f"llm{i}_judgement"] = judgement.strip().lower() == "true"

    # Create a new column in the dataframe called "llm_knows_word" with boolean data type
    df["llm_knows_word"] = False

    # Now that we have five different judgements for each word, we get the majority vote for each word, and if the majority vote is true, we set the "llm_knows_word" column to true
    # Consider that some of the judgements may be -1 or -2, which means that the LLMs could not generate a definition or a judgement for that word
    for _, row in df.iterrows():
        majority_vote = (
            row[["llm1_judgement", "llm2_judgement", "llm3_judgement", "llm4_judgement", "llm5_judgement"]]
            .mode()
            .values[0]
        )
        if majority_vote == -1 or majority_vote == -2:
            df.loc[df["words"] == row["words"], "llm_knows_word"] = "N/A"
        else:
            df.loc[df["words"] == row["words"], "llm_knows_word"] = majority_vote

    # Save the new dataframe to a new csv file called {llm_model_name}_balderdash_words1.csv
    if not os.path.exists(os.path.join(os.getcwd(), "data")):
        os.makedirs(os.path.join(os.getcwd(), "data"))
    df.to_csv(os.path.join("data", f"{llm_model_name.replace('/', '_')}_balderdash_words1.csv"), index=False)
