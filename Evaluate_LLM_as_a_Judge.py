# This file is used to evaluate LLM as a judge
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
    df = pd.read_csv('/home/erahmati/USC-CSCI699-HistoryofLanguage-AI_Balderdash/balderdash_words1_preliminary_results.csv')

    df['misinfo_llama_label'] = False
    df['def_llama_label'] = False
    
    word_data = {}
    for _, row in df.iterrows():
        word = row["words"].strip().lower()
        definition = literal_eval(row["def"])[0].strip()
        gptdefinition = row["definition"].strip()
        misinformation = row["misinformation"].strip()
        
        word_data[len(word_data) + 1] = {"word": word, "def": definition, "misinfo": misinformation, "gptdef": gptdefinition}

    
    second_tqdm_progress_bar = tqdm(word_data.items(), total=len(word_data), leave=False)
    for _, data in second_tqdm_progress_bar:
        second_tqdm_progress_bar.set_description(f"Word: {data['word']}")
        word = data["word"]
        definition = data["def"]
        misinfo = data["misinfo"]
        gptdef = data["gptdef"]

        
        system_judge = """Please act as a judge and evaluate whether a reference dictionary definition of a word and a given definition by an assistant capture the same core concept. Consider the essence and key components in the reference dictionary definition. You will be given a word, its reference dictionary definition, and an assistant's definition of that word. Your job is to assess if the assistant's definition captures the essential meaning or core concept of the reference dictionary definition, even if there are minor differences in detail or scope. Begin your evaluation by comparing the assistant's definition with the reference dictionary definition. Maintain objectivity and ensure that the order in which the definitions were presented does not influence your decision. Do not allow the length of the definitions to influence your evaluation. Output your final verdict by strictly following this format: "true" if the assistant's definition captures the core concept of the reference dictionary definition, and "false" if the assistant's definition fails to capture the core concept of the reference dictionary definition."""
        messages = [
            {
                "role": "system",
                "content": system_judge,
            }
        ]
        messages.append(
            {
                "role": "user",
                "content": f"""You receive the word: "{word}", the reference dictionary definition: "{definition}", and assistant's definition: "{misinfo}". Give your answer as a single word, either "true" or "false". Your verdict is: """,
            }
        )
        judgement = judge_llm.generate_answer(messages)
        logger.info(
            f"word: {word}, definition: {definition}\ngenerated definition: {misinfo}\njudgement: {judgement}"
        )
        df.loc[df["words"] == word, 'misinfo_llama_label'] = judgement.strip().lower() == "true"
        
        messages = [
            {
                "role": "system",
                "content": system_judge,
            }
        ]
        messages.append(
            {
                "role": "user",
                "content": f"""You receive the word: "{word}", the reference dictionary definition: "{definition}", and assistant's definition: "{gptdef}". Give your answer as a single word, either "true" or "false". Your verdict is: """,
            }
        )
        judgement = judge_llm.generate_answer(messages)
        logger.info(
            f"word: {word}, definition: {definition}\ngenerated definition: {gptdef}\njudgement: {judgement}"
        )
        df.loc[df["words"] == word, 'def_llama_label'] = judgement.strip().lower() == "true"

    df.to_csv('results.csv', index = False)
    
