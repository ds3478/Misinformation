import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

def get_random_forest_model(X, y, cross_validation):
    ranForest = RandomForestClassifier()

    param_grid = {'n_estimators': [100, 150], 'max_depth': [None, 4, 7]}

    grid = GridSearchCV(ranForest, param_grid = param_grid, cv = cross_validation, scoring = 'roc_auc')

    grid.fit(X, y)

    #print("Random Forest best mean cross-validation score: {:.3f}".format(grid.best_score_))
    #print("Random Forest best parameters: {}".format(grid.best_params_))

    ranForest = grid.best_estimator_
    
    return ranForest, grid.best_score_, grid.best_params_
