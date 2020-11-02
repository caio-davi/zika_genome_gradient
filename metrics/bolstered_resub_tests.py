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

from decimal import Decimal


def write_on_file(text, path="../results/bolstered_results.md", end_section=False):
    with open(path, "w") as file_object:
        file_object.write(text)
        file_object.write("\n")


def matrix2string(confMatrix):
    m = " | | 0 | 1 \n"
    m += " -- | -- | -- \n"
    m += "0 |" + str(confMatrix[0][0]) + " | " + str(confMatrix[0][1]) + "\n"
    m += "1 |" + str(confMatrix[1][0]) + " | " + str(confMatrix[1][1]) + "\n"
    m += "\n"
    return m


X = np.load("../data_splits/x_bolstered.npy")
y = np.load("../data_splits/y_bolstered.npy")


models = [
    {"name": "Logistic Regression ", "model": LogisticRegression(
        random_state=0, solver='lbfgs', multi_class='multinomial')},
    {"name": "KNN", "model":
        sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)},
    {"name": "Decision Tree", "model":
        tree.DecisionTreeClassifier()},
    {"name": "SVM-RBF Kernel",
        "model": svm.SVC(kernel='rbf', probability=True)},
    {"name": "SVM-Linear Kernel",
        "model": svm.SVC(kernel='linear', probability=True)},
]


to_file = "## Bolstered Resubstituition  Accuracy\n"

for m in models:
    to_file += " " + m["name"] + " |"

to_file += "\n"
for i in range(len(models)):
    to_file += "--- |"

to_file += "\n"

cf = "\n\n ### Confusion Matrix \n"
for m in models:
    m["model"].fit(X, y)
    y_predicted = m["model"].predict(X)
    accuracy = accuracy_score(y, y_predicted)
    to_file += str(accuracy) + " | "
    cf += "\n\n " + "##### " + m["name"] + "\n"
    cf += matrix2string(confusion_matrix(y, y_predicted))

to_file += cf
write_on_file(to_file)
