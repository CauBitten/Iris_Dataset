from sklearn.datasets import load_iris

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn import metrics

iris = load_iris()

X = iris.data
y = iris.target

# 30% para teste e 70% para aprendizado da máquina
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # random_state=3


# Procurando o melhor valor para K para KNN Model
scores = []
the_biggest_accuracy = 0
the_biggest_knn = 0
k_range = range(1, 26)
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred_knn))
    if scores[k-1] > the_biggest_accuracy:
        the_biggest_accuracy = scores[k-1]
        the_biggest_knn = k


plt.plot(k_range, scores)
plt.xlabel("K Value for KNN")
plt.ylabel("Accuracy of K")
plt.title("The Best K")
plt.show()

# Usando um dos melhores valores para K
print("The best K:", the_biggest_knn)
knn = KNeighborsClassifier(n_neighbors=the_biggest_knn)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

knn_accuracy = metrics.accuracy_score(y_test, y_pred_knn)
print("Near Neighbors Classification:", knn_accuracy)


# Usando o Logistic Regression Model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

logreg_accuracy = metrics.accuracy_score(y_test, y_pred_logreg)
print("Logistic Regression:", logreg_accuracy)


# Usando o Gaussian Process Classifier Model
gpc = GaussianProcessClassifier()
gpc.fit(X_train, y_train)
y_pred_gpc = gpc.predict(X_test)

gpc_accuracy = metrics.accuracy_score(y_test, y_pred_gpc)
print("Gaussian Process: ", gpc_accuracy)


# Vetor das precisões dos modelos
accuracy_vector = [knn_accuracy, logreg_accuracy, gpc_accuracy]


# Testando o predict com valores aleatorios (Você adiciona qualquer dado que quiser)
X_new = [[3, 6, 5, 2],
         [4, 7, 3, 1],
         [5, 8, 9, 4],
         [1, 3, 2, 1],
         [10.9, 5.1, 9.3, 4.6]]

# 0 = Setosa        1 = Versicolor       2 = Virginica


best_model = max(float(number) for number in accuracy_vector)


if best_model == knn_accuracy:
    print("Nova classificação resolvida por KNN: ", knn.predict(X_new))


if best_model == logreg_accuracy:
    print("Nova classificação resolvida por LoR: ", logreg.predict(X_new))


if best_model == gpc_accuracy:
    print("Nova classificação resolvida por GPC: ", gpc.predict(X_new))
