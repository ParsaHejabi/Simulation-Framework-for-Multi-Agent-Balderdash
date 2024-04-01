# import gdown

# url = "1-7furyX1JQfdcnBlTHAqCMlgD9DiM7PK"
# gdown.download(id = url, output='Words1.csv', quiet = True)

# url = "18ANahetu6RynqINS1Ug5Fzmn87jqjXpQ"
# gdown.download(id = url, output='Words2.csv', quiet = True)

import pandas as pd
from openai import OpenAI
from evaluate import load

from dotenv import load_dotenv

load_dotenv()


client = OpenAI(api_key="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")

bertscore = load("bertscore")

df = pd.read_csv("./Words2.csv")
df = df.iloc[:50, :]
df.reset_index(inplace=True, drop=True)


defenition = []
misinformation = []
# defscore = []
# misinfoscore = []
for i in range(len(df)):
    if i % 10 == 0:
        print(i // 10)
        db = pd.DataFrame()
        db["definition"] = defenition
        db["misinformation"] = misinformation
        db.to_csv("./Words2_GPT35.csv", index=False)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "define the word " + df["words"][i] + " in one sentence.",
            }
        ],
        model="gpt-3.5-turbo",
    )
    defenition.append(chat_completion.choices[0].message.content)
    # defscore.append(bertscore.compute(predictions=[chat_completion.choices[0].message.content],
    #                                   references=[df['def'][i]], lang="en"))

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Suppose you are playing Balderdash and you want to deceive your opponants with a deceiving definition. define "
                + df["words"][i]
                + " in one sentence.",
            }
        ],
        model="gpt-3.5-turbo",
    )
    misinformation.append(chat_completion.choices[0].message.content)
    # misinfoscore.append(bertscore.compute(predictions=[chat_completion.choices[0].message.content],
    #                                   references=[df['def'][i]], lang="en"))

df["definition"] = defenition
df["misinformation"] = misinformation
# df['defscore'] = defscore
# df['misinfoscore'] = misinfoscore

df.to_csv("./Words2_GPT35.csv", index=False)
