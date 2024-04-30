# This file is used to create a subset of the modified_balderdash_words1.csv file that only contains words that the LLMs (or our particular LLM: Llama 3) know.
from utils.llm import LLM
import pandas as pd
import os
import torch
from ast import literal_eval
from utils.logger import setup_logger
from tqdm import tqdm


llms = {}


def get_device() -> torch.device:
    if not torch.backends.mps.is_available():
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        return torch.device("mps")


def get_or_load_llm(model_name: str, device: torch.device) -> LLM:
    if model_name not in llms:
        llms[model_name] = LLM(device=device, model_name=model_name, temp=0.9)
    return llms[model_name]


if __name__ == "__main__":
    # Set up the logger using the name of this file
    logger = setup_logger(
        os.path.basename(__file__).replace(".py", ""),
        os.path.join("logs", f"{os.path.basename(__file__).replace('.py', '.log')}"),
    )
    device = get_device()
    logger.info(f"Device: {device}")
    llm_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    judge_llm_model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    llm = get_or_load_llm(llm_model_name, device)
    judge_llm = get_or_load_llm(judge_llm_model_name, device)

    random_seed = 42

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
            df.loc[df["words"] == word, f"llm{i}_definition"] = llm.generate_answer(messages).strip()

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
            judgement = judge_llm.generate_answer(messages)
            logger.info(
                f"word: {word}, POS: {pos}, definition: {definition}\ngenerated definition: {df.loc[df['words'] == word, f'llm{i}_definition'].values[0]}\njudgement: {judgement}"
            )
            df.loc[df["words"] == word, f"llm{i}_judgement"] = judgement.strip().lower() == "true"

    # Create a new column in the dataframe called "llm_knows_word" with boolean data type
    df["llm_knows_word"] = False

    # Now that we have five different judgements for each word, we get the majority vote for each word, and if the majority vote is true, we set the "llm_knows_word" column to true
    for _, data in word_data.items():
        word = data["word"]
        judgements = [df.loc[df["words"] == word, f"llm{i}_judgement"].values[0] for i in range(1, 6)]
        if judgements.count(True) >= 3:
            df.loc[df["words"] == word, "llm_knows_word"] = True

    # Save the new dataframe to a new csv file called {llm_model_name}_balderdash_words1.csv
    if not os.path.exists(os.path.join(os.getcwd(), "data")):
        os.makedirs(os.path.join(os.getcwd(), "data"))
    df.to_csv(os.path.join("data", f"{llm_model_name.replace('/', '_')}_balderdash_words1.csv"), index=False)
