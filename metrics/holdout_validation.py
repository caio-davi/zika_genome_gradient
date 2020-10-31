#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 13:51:01 2020

@author: flavio
"""
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import tree
import sklearn.neighbors
from sklearn.metrics import accuracy_score
from sklearn.metrics.classification import confusion_matrix
import numpy as np
# from plot_roc_curve import plot_roc_curve

x_train = np.load("../data_splits/x_train.npy")
x_test = np.load("../data_splits/x_test.npy")
y_train = np.load("../data_splits/y_train.npy")
y_test = np.load("../data_splits/y_test.npy")

s = []
for i in y_train:
    if i == 0:
        s.append(0)
    else:
        s.append(1)
y_train = s

s = []
for i in y_test:
    if i == 0:
        s.append(0)
    else:
        s.append(1)
y_test = s


model0 = [LogisticRegression(
    random_state=0, solver='lbfgs', multi_class='multinomial'), "Logistic Regression"]
model1 = [sklearn.neighbors.KNeighborsClassifier(n_neighbors=1), 'KNN']
model2 = [tree.DecisionTreeClassifier(), 'Decision Tree']
model3 = [svm.SVC(kernel='rbf', probability=True), 'SVM-RBF Kernel']
model4 = [svm.SVC(kernel='linear', probability=True), 'SVM-Linear Kernel']
models = [model0, model1, model2, model3, model4]

for m in models:
    model = m[0]
    model.fit(x_train, y_train)
    y_predicted = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_predicted)
    print(m[1])
    print("Accuracy = ", accuracy)
    print(confusion_matrix(y_test, y_predicted))
    # print(classification_report(y_test, y_predicted, target_names=["Afr","Amr"]))
    # plot_roc_curve(m[1],model,x_test,y_test)
