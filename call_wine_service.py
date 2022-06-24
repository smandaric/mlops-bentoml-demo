import requests
import json
import pandas as pd
import numpy as np

CSV_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"


data = pd.read_csv(CSV_URL, sep=";")

df = data.drop(["quality"], axis=1)
df = df.sample(n=1).to_json(orient="values")

print(df)

res = requests.post(
    "http://127.0.0.1:3000/classify",
    headers={"content-type": "application/json"},
    data=df).text

print(res)
