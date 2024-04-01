from evaluate import load
import pandas as pd
from ast import literal_eval

df = pd.read_csv("./Words2.csv")
df = df.iloc[:40, :]
df2 = pd.read_csv("./Words2_GPT35.csv")
df["definition"] = df2["definition"]
df["misinformation"] = df2["misinformation"]
bertscore = load("bertscore")


defscore, misinfoscore = [], []
for i in range(len(df)):
    temp1, temp2 = [], []

    for item in literal_eval(df["def"][i]):
        temp1.append(bertscore.compute(predictions=[df["definition"][i]], references=[item], lang="en"))
        temp2.append(bertscore.compute(predictions=[df["misinformation"][i]], references=[item], lang="en"))
    defscore.append(temp1)
    misinfoscore.append(temp2)

df["defscore"] = defscore
df["misinfoscore"] = misinfoscore

df.to_csv("./Words2_GPT35.csv", index=False)
