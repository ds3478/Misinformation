import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
    
def get_log_regression_model(X, y, cross_validation):
    lr = LogisticRegression()

    param_grid = {'C': [0.0001, 0.001, 0.01, 0.1, 1], 'fit_intercept': [False, True]}

    grid = GridSearchCV(lr, param_grid = param_grid, cv = cross_validation, scoring = 'roc_auc')

    grid.fit(X, y)

    #print("Logistic Regression best mean cross-validation score: {:.3f}".format(grid.best_score_))
    #print("Logistic Regression best parameters: {}".format(grid.best_params_))

    lr = grid.best_estimator_
    
    return lr, grid.best_score_, grid.best_params_
