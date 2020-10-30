#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 10:26:52 2020

@author: flavio
"""

import numpy as np
import pandas as pd

df = pd.read_csv("/home/flavio/Dropbox/Documentos/transicao/Pesquisa_Inovacao/Fiocruz/GradienteGenomico/NovoEstudo/dados_preprocessados.csv")

"""
Removes population from asia and not selected RS
"""
df2 = df[["rs12610506","rs1335531","rs15895","rs2629396","rs3911403",
"rs41265961","rs4143815","rs4804801","rs4963271","rs7127254","Class"]]

df2 = df2[df2['Class'] != 1]

"""
Separates input from output data
"""
x_values = df2.loc[:, df2.columns != 'Class'].values
y_values = df2['Class'].values

"""
Performs encoding of input data
"""
from sklearn.preprocessing import LabelEncoder

x = x_values.T
x_values1 = []

for i in range(0,len(x)):
    le = LabelEncoder()
    le.fit(x[i])
    x_values1.append(le.transform(x[i]))

x_values1 = np.array(x_values1).T

"""
Separates training and test files
"""
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
sss.get_n_splits(x_values1, y_values)

for train_index, test_index in sss.split(x_values1,y_values):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = x_values1[train_index],x_values1[test_index]
    y_train, y_test = y_values[train_index], y_values[test_index]
    
"""
Saves data for posterior use in experiments
"""
np.save("x_train.npy",X_train)
np.save("x_test.npy",X_test)
np.save("y_train.npy",y_train)
np.save("y_test.npy",y_test)