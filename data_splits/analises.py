from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import random


df = pd.read_csv("../dados_preprocessados.csv", header=0)

features = ["rs12610506", "rs1335531", "rs15895", "rs2629396", "rs3911403",
            "rs41265961", "rs4143815", "rs4804801", "rs4963271", "rs7127254", "Class"]

df = df.apply(lambda col: LabelEncoder().fit_transform(col))

for feature in features:
    cor = np.corrcoef(df[feature], df["Class"])
    print("============================")
    print("Correlation ", feature)
    print(cor)
    print("============================ \n")
