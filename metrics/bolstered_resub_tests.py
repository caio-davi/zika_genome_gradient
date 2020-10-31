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

X = np.load("../data_splits/x_bolstered.npy")
y = np.load("../data_splits/y_bolstered.npy")

model0 = [LogisticRegression(
    random_state=0, solver='lbfgs', multi_class='multinomial'), "Logistic Regression"]
model1 = [sklearn.neighbors.KNeighborsClassifier(n_neighbors=1), 'KNN']
model2 = [tree.DecisionTreeClassifier(), 'Decision Tree']
model3 = [svm.SVC(kernel='rbf', probability=True), 'SVM-RBF Kernel']
model4 = [svm.SVC(kernel='linear', probability=True), 'SVM-Linear Kernel']
models = [model0, model1, model2, model3, model4]

for m in models:
    model = m[0]
    model.fit(X, y)
    y_predicted = model.predict(X)
    accuracy = accuracy_score(y, y_predicted)
    print("\n\n")
    print("====================================")
    print(m[1])
    print("Accuracy = ", accuracy)
    print(confusion_matrix(y, y_predicted))
    print("====================================")
