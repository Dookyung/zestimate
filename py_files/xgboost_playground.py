# -*- coding: utf-8 -*-
'''
@authors: codeWoker & Zibski
'''
import numpy as np
import xgboost as xgb
import sklearn.metrics
import os
from datetime import datetime
'''
Functions
'''

def xgboost_train_and_test(X_train, X_test, y_train, y_test):
    y_mean = np.mean(y_train)
    # xgboost params
    xgb_params = {
        'eta': 0.06,
        'max_depth': 5,
        'subsample': 0.77,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'base_score': y_mean[0],
        'silent': 1
    }
    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test)
    # cross-validation
    cv_result = xgb.cv(xgb_params, 
                       dtrain, 
                       nfold = 5,
                       num_boost_round = 200,
                       early_stopping_rounds = 50,
                       verbose_eval = 10, 
                       show_stdv = False
                      )
    num_boost_rounds = len(cv_result)
    # train model
    model = xgb.train(dict(xgb_params, silent = 1), dtrain, num_boost_round=num_boost_rounds)
    if len(y_test) != 0:
        y_pred = model.predict(dtest)
        # MAE Calculation
        sklearn.metrics.mean_absolute_error(y_test, y_pred), model
    else:
        return 'null', model

def xgboost_validate(X_train, X_validation, y_train):
    y_mean = np.mean(y_train)
    # xgboost params
    xgb_params = {
        'eta': 0.06,
        'max_depth': 5,
        'subsample': 0.77,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'base_score': y_mean[0],
        'silent': 1
    }
    dtrain = xgb.DMatrix(X_train, y_train)
    d_validation = xgb.DMatrix(X_validation)
    # cross-validation
    cv_result = xgb.cv(xgb_params, 
                       dtrain, 
                       nfold = 5,
                       num_boost_round = 200,
                       early_stopping_rounds = 50,
                       verbose_eval = 10, 
                       show_stdv = False
                      )
    num_boost_rounds = len(cv_result)
    # train model
    model = xgb.train(dict(xgb_params, silent = 1), dtrain, num_boost_round=num_boost_rounds)
    # Save model parameters to file
    f = open(os.path.abspath(os.getcwd()+'\\..')+'\\models\\' + 'model_{}_parameters.txt'.format(datetime.now().strftime('%Y%m%d_%H%M%S')),'w')
    f.write(str(xgb_params))
    f.close()
    return model.predict(d_validation)