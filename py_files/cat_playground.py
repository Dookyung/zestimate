# -*- coding: utf-8 -*-
'''
@authors: codeWoker & Zibski
'''
from catboost import CatBoostRegressor
from tqdm import tqdm


def cat_train(X_train, y_train): 
    model = CatBoostRegressor(
        iterations=200, learning_rate=0.03,
        depth=6, l2_leaf_reg=3,
        loss_function='MAE',
        eval_metric='MAE')
    model.fit(X_train, y_train)
    return model
    
    
def cat_validate(X_validation, model, month = 0):
    print(month)
    if month != 0:
        X_validation['month'] = month
    return model.predict(X_validation)