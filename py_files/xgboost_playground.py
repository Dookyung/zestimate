# -*- coding: utf-8 -*-
'''
@authors: codeWoker & Zibski
'''
import numpy as np
import xgboost as xgb
import sklearn.metrics
import os
import pickle
from datetime import datetime
'''
Functions
'''

def xgboost_train_and_test(X_train, X_test, y_train, y_test):
    y_mean = np.mean(y_train)
    # xgboost params
    params = {
        'eta': 0.04,
        'max_depth': 8,
        'subsample': 0.8,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'base_score': y_mean,
        'silent': 1
    }
    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test)
    # cross-validation
    cv_result = xgb.cv(params, 
                       dtrain, 
                       nfold = 5,
                       num_boost_round = 200,
                       early_stopping_rounds = 50,
                       verbose_eval = 10, 
                       show_stdv = False
                      )
    num_boost_rounds = len(cv_result)
    # train model
    model = xgb.train(dict(params, silent = 1), dtrain, num_boost_round=num_boost_rounds)
    if len(y_test) != 0:
        y_pred = model.predict(dtest)
        # MAE Calculation
        return sklearn.metrics.mean_absolute_error(y_test, y_pred), model
    else:
        return 'null', model


def xgboost_grid_search(X_train, X_test, y_train, y_test):
    y_mean = np.mean(y_train)
    eta_values = [0.04, 0.05]
    max_depth_values = [8, 9]
    subsample_values = [0.8]
    objective_values = ['reg:linear']
    best_mae = 10
    best_params = {'eta': None, 'max_depth': None, 'subsample': None, 'objective': None,
                   'eval_metric': 'mae', 'base_score': y_mean, 'silent': 1}
    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test)
    for eta in eta_values:
        for max_depth in max_depth_values:
            for subsample in subsample_values:
                for objective in objective_values:
                    # xgboost params
                    params = {
                        'eta': eta,
                        'max_depth': max_depth,
                        'subsample': subsample,
                        'objective': objective,
                        'eval_metric': 'mae',
                        'base_score': y_mean,
                        'silent': 1
                    }
                    # train model
                    model = xgb.train(dict(params, silent = 1), dtrain, num_boost_round = 100)
                    y_pred = model.predict(dtest)
                    # MAE Calculation
                    mae = sklearn.metrics.mean_absolute_error(y_test, y_pred)
                    print(mae)
                    if mae < best_mae:
                        best_mae = mae
                        best_params = params
                        best_model = model
                        print(best_params)
    return best_model, best_params
                        

def xgboost_train(X_train, y_train):
    y_mean = np.mean(y_train)
    # xgboost params
    params = {
        'eta': 0.04,
        'max_depth': 8,
        'subsample': 0.8,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'base_score': y_mean,
        'silent': 1
    }
    dtrain = xgb.DMatrix(X_train, y_train)
    # cross-validation
    cv_result = xgb.cv(params, 
                       dtrain, 
                       nfold = 5,
                       num_boost_round = 200,
                       early_stopping_rounds = 50,
                       verbose_eval = 10, 
                       show_stdv = False
                      )
    num_boost_rounds = len(cv_result)
    # train model
    model = xgb.train(dict(params, silent = 1), dtrain, num_boost_round=num_boost_rounds)
    # Save model parameters to file
    f = open(os.path.abspath(os.getcwd()+'\\..')+'\\models\\' + 'model_{}_parameters.txt'.format(datetime.now().strftime('%Y%m%d_%H%M%S')),'w')
    f.write(str(params))
    f.close()
    # Save model to sav file
    pickle.dump(model, open(os.path.abspath(os.getcwd()+'\\..')+'\\models\\' + 'model_{}.sav'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), 'wb'))
    return model


def xgboost_validate(X_validation, model, month = 0):
    if month != 0:
        X_validation['month'] = month
        d_validation = xgb.DMatrix(X_validation)
    else:
        d_validation = xgb.DMatrix(X_validation)
    return model.predict(d_validation)