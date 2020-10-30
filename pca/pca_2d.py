#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
PCA 2D

Created on Tue Jan 21 15:15:44 2020

@author: flavio
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

from sklearn import decomposition
from sklearn import datasets

np.random.seed(5)

import pandas as pd

#If necessary, removes asians
remove_asians = True

#TODO - change file location
df = pd.read_csv("/home/flavio/Dropbox/Documentos/transicao/Pesquisa_Inovacao/Fiocruz/GradienteGenomico/NovoEstudo/dados_preprocessados.csv")

#Removes unecessary or SNPs with problems
df2 = df.drop(["Patient ID","Population","rs1126535", "rs12846646", "rs1638596", "rs17256081", "rs179008",
"rs179010", "rs179012", "rs179014", "rs179016", "rs179021",
"rs2109134", "rs2159377", "rs2280964", "rs2407992", "rs3092921",
"rs3092946", "rs3764880", "rs3764885", "rs5741883", "rs5744080", "rs5744088"], axis=1)

target_names = ["Africans","Asians","Americans"]

if remove_asians:
    df2 = df2[df2['Class'] != 1]
    target_names = ["Africans","Americans"]

"""
Separates input from output data
"""
x_values = df2.loc[:, df2.columns != 'Class'].values
y = df2['Class'].values

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

X = np.array(x_values1).T



pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()

if remove_asians:
    colors = ['navy','darkorange']
    lw = 2
    
    for color, i, target_name in zip(colors, [0, 2], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)

    plt.title('PCA of Africans and Americans')

else:
    colors = ['navy', 'turquoise', 'darkorange']
    lw = 2

    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    plt.title('PCA of Africans, Asians and Americans')


plt.legend(loc='best', shadow=False, scatterpoints=1)


plt.show()
