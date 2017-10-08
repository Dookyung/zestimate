# -*- coding: utf-8 -*-
'''
@authors: codeWoker & Zibski
https://www.kaggle.com/vber852/simple-lgbm-model-lb-0-0643788
'''
import sklearn.metrics
import lightgbm as lgb
import os
import pickle
from datetime import datetime
'''
Functions
'''

def lgbm_train_and_test(X_train, X_test, y_train, y_test):
    print('Training LGBM model...')
    ltrain = lgb.Dataset(X_train, label = y_train)
    params = {}
    params['metric'] = 'mae'
    params['max_depth'] = 100
    params['num_leaves'] = 32
    params['feature_fraction'] = .85
    params['bagging_fraction'] = .95
    params['bagging_freq'] = 8
    params['learning_rate'] = 0.0025
    params['verbosity'] = 0
    model = lgb.train(params, ltrain, valid_sets = [ltrain], verbose_eval=200, num_boost_round=2930)
    if len(y_test) != 0:
        y_pred = model.predict(X_test)
        # MAE Calculation
        return sklearn.metrics.mean_absolute_error(y_test, y_pred), model
    else:
        return 'null', model
    

def lgbm_train(X_train, y_train): 
    print('Training LGBM model...')
    ltrain = lgb.Dataset(X_train, label = y_train)
    params = {}
    params['metric'] = 'mae'
    params['max_depth'] = 100
    params['num_leaves'] = 32
    params['feature_fraction'] = .85
    params['bagging_fraction'] = .95
    params['bagging_freq'] = 8
    params['learning_rate'] = 0.0025
    params['verbosity'] = 0
    model = lgb.train(params, ltrain, valid_sets = [ltrain], verbose_eval=200, num_boost_round=2930)
    # Save model parameters to file
    f = open(os.path.abspath(os.getcwd()+'\\..')+'\\models\\' + 'model_{}_parameters.txt'.format(datetime.now().strftime('%Y%m%d_%H%M%S')),'w')
    f.write(str(params))
    f.close()
    # Save model to sav file
    pickle.dump(model, open(os.path.abspath(os.getcwd()+'\\..')+'\\models\\' + 'model_{}.sav'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), 'wb'))
    return model
    
    
def lgbm_validate(X_validation, model, month = 0):
    print(month)
    if month != 0:
        X_validation['month'] = month
    return model.predict(X_validation)