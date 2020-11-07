import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist
from scipy.stats import chi
import random

from sys import exit

BOLSTER_MULTIPLIER = 100

df = pd.read_csv("../dados_preprocessados.csv", header=0)

features = ["rs12610506", "rs1335531", "rs15895", "rs2629396", "rs3911403",
            "rs41265961", "rs4143815", "rs4804801", "rs4963271", "rs7127254", "Class"]


def getSigma(df):
    D = cdist(df, df, 'euclid')
    np.fill_diagonal(D, 99999999)
    meanMinDist = np.mean(D.min(axis=1), axis=0)
    cp = chi.ppf(0.5, df.shape[1])
    return meanMinDist/cp


def encodeDF(df):
    # return pd.get_dummies(df)
    return df.apply(lambda col: LabelEncoder().fit_transform(col))


def bolster(df, sigma, new_points):
    bolstered = pd.DataFrame()
    for sample in df.index:
        for m in range(new_points):
            bolsteredRow = pd.Series()
            for feature in df.columns:
                new_point = random.gauss(
                    mu=df.loc[sample, feature], sigma=sigma)
                bolsteredRow[feature] = new_point
            bolstered = bolstered.append(bolsteredRow, ignore_index=True)
    return bolstered


df = df[features]

X_0 = encodeDF(df[df['Class'] == 0])                # Split and Encode Classes
X_1 = encodeDF(df[df['Class'] == 2])                # Split and Encode Classes

X_0 = X_0.drop(['Class'], axis=1)                   # Remove label column
X_1 = X_1.drop(['Class'], axis=1)                   # Remove label column

X_0_sig = getSigma(X_0)                            # Generate features' sigmas
X_1_sig = getSigma(X_1)                            # Generate features' sigmas

X = pd.concat([bolster(X_0, X_0_sig, BOLSTER_MULTIPLIER),
               bolster(X_1, X_1_sig, BOLSTER_MULTIPLIER)])

y = np.concatenate((np.zeros((len(X_0)*BOLSTER_MULTIPLIER), dtype=int),
                    np.ones((len(X_1)*BOLSTER_MULTIPLIER), dtype=int)))


np.save("x_bolstered.npy", X)
np.save("y_bolstered.npy", y)
