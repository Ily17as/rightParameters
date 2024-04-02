# Библиотеки
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Загрузка набора данных Iris
iris = load_iris()
X = iris.data
y = iris.target

# Гиперпараметры
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'linear']}

# Инициализация svm классификатора
svm = SVC()

# поиск лучших гиперпараметров с помощью поиска по сетке и их вывод
grid_search = GridSearchCV(svm, param_grid, cv=5)
grid_search.fit(X, y)

print("Лучшие параметры:", grid_search.best_params_)
