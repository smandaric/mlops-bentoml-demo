import xdrlib
import requests
import json
import pandas as pd
import numpy as np
import bentoml

CSV_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"


data = pd.read_csv(CSV_URL, sep=";")

df = data.drop(["quality"], axis=1)
x = df.sample(n=1).values

iris_clf_runner = bentoml.xgboost.get("xgb_winequality_regressor").to_runner()
iris_clf_runner.init_local()
prediction = iris_clf_runner.predict.run(x)

print(prediction)
