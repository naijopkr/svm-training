import pandas as pd

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

X = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
y = pd.DataFrame(cancer['target'], columns=['Cancer'])

from sklearn.model_selection import train_test_split as tts

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3, random_state=101)

# Train Support Vector Classifier
from sklearn.svm import SVC

svc = SVC()
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

import metrics

metrics.print_cm(y_test, y_pred)
metrics.print_cr(y_test, y_pred)

# GridSearch
param_grid = dict(
    C = [0.1, 1, 10, 100, 1000],
    gamma = [1, 0.1, 0.01, 0.001, 0.0001]
)

from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(SVC(), param_grid, refit=True,verbose=3)
grid.fit(X_train, y_train)
grid.best_params_
grid.best_estimator_

y_grid_pred = grid.predict(X_test)
metrics.print_cm(y_test, y_grid_pred)
metrics.print_cm(y_test, y_pred)

metrics.print_cr(y_test, y_grid_pred)
metrics.print_cr(y_test, y_pred)
