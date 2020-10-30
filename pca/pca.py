#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
PCA 3D

Created on Tue Jan 21 15:15:44 2020

@author: flavio
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from sklearn import decomposition
from sklearn import datasets
from sklearn.decomposition import PCA
np.random.seed(5)

import pandas as pd

#TODO: change file location
df = pd.read_csv("/home/flavio/Dropbox/Documentos/transicao/Pesquisa_Inovacao/Fiocruz/GradienteGenomico/NovoEstudo/dados_preprocessados.csv")

#Removes unecessary or SNPs with problems
df2 = df.drop(["Patient ID","Population","rs1126535", "rs12846646", "rs1638596", "rs17256081", "rs179008",
"rs179010", "rs179012", "rs179014", "rs179016", "rs179021",
"rs2109134", "rs2159377", "rs2280964", "rs2407992", "rs3092921",
"rs3092946", "rs3764880", "rs3764885", "rs5741883", "rs5744080", "rs5744088"], axis=1)

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

fig = plt.figure(1, figsize=(8, 8))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=0, azim=200)

plt.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)

for name, label in [('Africans', 0), ('Asians', 1), ('Americans', 2)]:
    ax.text3D(X[y == label, 0].mean(),
              X[y == label, 1].mean() ,
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral,
           edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()
