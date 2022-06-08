import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

import xgboost as xgb

def get_xgboost_model(X, y, cross_validation):
    xgbClassifier = xgb.XGBClassifier()

    param_grid = {'n_estimators': [100, 150], 'use_label_encoder': [False], 'verbosity': [0], 'n_jobs': [1]}

    grid = GridSearchCV(xgbClassifier, param_grid = param_grid, cv = cross_validation, scoring = 'roc_auc')

    grid.fit(X, y, eval_metric = 'auc')

    #print("XGBoost best mean cross-validation score: {:.3f}".format(grid.best_score_))
    #print("XGBoost best parameters: {}".format(grid.best_params_))

    xgbClassifier = grid.best_estimator_
    
    return xgbClassifier, grid.best_score_, grid.best_params_
