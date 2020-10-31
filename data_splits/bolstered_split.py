from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import random


def getSigmas(df):
    return df.apply(lambda col: col.var())


def encodeDF(df):
    # return pd.get_dummies(df)
    return df.apply(lambda col: LabelEncoder().fit_transform(col))


df = pd.read_csv("../dados_preprocessados.csv", header=0)

features = ["rs12610506", "rs1335531", "rs15895", "rs2629396", "rs3911403",
            "rs41265961", "rs4143815", "rs4804801", "rs4963271", "rs7127254", "Class"]

df = df[features]

X_0 = encodeDF(df[df['Class'] == 0])    # Split and Encode Classes
X_1 = encodeDF(df[df['Class'] == 2])    # Split and Encode Classes

X_0.drop(['Class'], axis=1)             # Remove label column
X_1.drop(['Class'], axis=1)             # Remove label column

X_0_sig = getSigmas(X_0)                # Generate features' sigmas
X_1_sig = getSigmas(X_1)                # Generate features' sigmas
