# -*- coding: utf-8 -*-
'''
@authors: codeWoker & Zibski
'''
import numpy as np
import pandas as pd
from datetime import datetime
import gc
import sklearn.metrics

import data_loader
import data_processing
import xgboost_playground
import lgbm_playground
import nn_playground
import cat_playground
'''
DummyRegressor
'''
data_files, properties_files = data_loader.load_all_data_files()
properties = data_loader.load_data(properties_files[len(properties_files)-1])
train_data = data_loader.load_data(data_files)
properties = data_processing.missing_data_dropper(properties)
properties = data_processing.data_cleaning_and_labeling(properties)
X = train_data.merge(properties, how='left', on='parcelid')

y = X['logerror']
# adding new feature
add_months = True
if add_months:
    X = data_processing.feature_month(X)

# drop other features
X = X.drop(['parcelid','transactiondate', 'censustractandblock', 'assessmentyear', 'logerror', 'structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt', 'taxvaluedollarcnt', 'taxamount'], axis=1)
X_validation = properties.drop(['parcelid', 'censustractandblock', 'assessmentyear', 'structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt', 'taxvaluedollarcnt', 'taxamount'], axis=1)

# drop taxes
X = X.drop(X.columns[X.columns.str.startswith('taxdelin')], axis = 1)
X_validation = X_validation.drop(X_validation.columns[X_validation.columns.str.startswith('taxdelin')], axis = 1)

# delete uneccessary data objects and collect garbage
del train_data, data_files, properties_files
gc.collect()

# Model testing
test_models_flag = False
if test_models_flag:
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    # drop outliers
    X_train = X_train[abs(y) < 0.4]
    y_train = y_train[abs(y) < 0.4]
    print('Shape train: {}\nShape test: {}'.format(X_train.shape, X_test.shape))
    # XGBoost
    MAE, model = xgboost_playground.xgboost_train_and_test(X_train, X_test, y_train, y_test)
    print(MAE)
    # LGBT
    MAE, model = lgbm_playground.lgbm_train_and_test(X_train, X_test, y_train, y_test)
    print(MAE)


# Finding optimal log_err bound for outliers   
log_err_bound_tunning = False
if log_err_bound_tunning:
    log_err_bounds = [0.225, 0.25, 0.275, 0.4]
    for idx, bound in enumerate(log_err_bounds):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
        X_train = data_processing.drop_outliers(X_train, ['logerror'], [bound])
        y_train = data_processing.drop_outliers(pd.DataFrame({'logerror' : y_train}), ['logerror'], [bound])
        MAE, model = xgboost_playground.xgboost_train_and_test(X_train, X_test, y_train, y_test)
        print('log error outliers bound: %s - MAE = %s', bound, MAE)        


xgboost_par_opt = True
if xgboost_par_opt == False:
    # Grid search for xgboost hyperparameters
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    # drop outliers
    X_train = X_train[abs(y) < 0.4]
    y_train = y_train[abs(y) < 0.4]
    model, params = xgboost_playground.xgboost_grid_search(X_train, X_test, y_train, y_test)


nn_par_opt = True
if nn_par_opt == False:
    # Grid search for xgboost hyperparameters
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    # drop outliers
    X_train = X_train[abs(y) < 0.4]
    y_train = y_train[abs(y) < 0.4]
    model, params = nn_playground.grid_search(X_train, X_test, y_train, y_test)


# Ensembled Models testing
# Model training
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# drop outliers
X_train = X_train[abs(y) < 0.4]
y_train = y_train[abs(y) < 0.4]
# import standard scaler class from sklearn library - feature scaling for nn
from sklearn.preprocessing import StandardScaler
# we create object sc_X of class StandardScaler
sc_X = StandardScaler()
X_train_nn = sc_X.fit_transform(X_train)
# we dont need any more to fit the test set to object sc_X because it has been already fitted
X_test_nn = sc_X.transform(X_test)

xgboost_model = xgboost_playground.xgboost_train(X_train, y_train)
lgbm_model = lgbm_playground.lgbm_train(X_train, y_train)
nn_model = nn_playground.train(X_train_nn, y_train)
cat_model = cat_playground.cat_train(X_train, y_train)

# Model testing
y_pred_xgboost = xgboost_playground.xgboost_validate(X_test, xgboost_model)
y_pred_lgbm = lgbm_playground.lgbm_validate(X_test, lgbm_model)
y_pred_nn = nn_playground.validate(X_test_nn, nn_model)
y_pred_cat = cat_playground.cat_validate(X_test, cat_model)
y_pred = np.array([y_pred_xgboost, y_pred_lgbm, y_pred_nn, y_pred_cat])
y_pred = y_pred.sum(axis = 0)/4
print(sklearn.metrics.mean_absolute_error(y_test, y_pred_xgboost))
print(sklearn.metrics.mean_absolute_error(y_test, y_pred_lgbm))
print(sklearn.metrics.mean_absolute_error(y_test, y_pred_nn))
print(sklearn.metrics.mean_absolute_error(y_test, y_pred_cat))
print(sklearn.metrics.mean_absolute_error(y_test, y_pred))
# Stacking
X_xgboost = xgboost_playground.xgboost_validate(X, xgboost_model)
X_lgbm = lgbm_playground.lgbm_validate(X, lgbm_model)
X_nn = nn_playground.validate(sc_X.transform(X), nn_model)
X_cat = cat_playground.cat_validate(X, cat_model)
X_stackNet = pd.DataFrame({'xgboost' : X_xgboost, 'lgbm' : X_lgbm, 'nn' : X_nn, 'cat' : X_cat})
#X_stackNet = pd.concat([X, X_stackNet], axis = 1)
y_stackNet = y

X_train_sn, X_test_sn, y_train_sn, y_test_sn = train_test_split(X_stackNet, y_stackNet, test_size = 0.25, random_state = 0)
X_train_sn = X_train_sn[abs(y) < 0.4]
y_train_sn = y_train_sn[abs(y) < 0.4]

import lightgbm as lgb
ltrain = lgb.Dataset(X_train_sn, label = y_train_sn)
params = {}
params['metric'] = 'mae'
params['max_depth'] = 32
params['num_leaves'] = 12
params['feature_fraction'] = .75
params['bagging_fraction'] = .65
params['bagging_freq'] = 8
params['learning_rate'] = 0.01
params['verbosity'] = 0
model = lgb.train(params, ltrain, valid_sets = [ltrain], verbose_eval=200, num_boost_round=1000)
pred = model.predict(X_test_sn)
print(sklearn.metrics.mean_absolute_error(y_test_sn, pred))




from sklearn.model_selection import KFold
from catboost import CatBoostRegressor
kfolds = 4
models = []
kfold = KFold(n_splits = kfolds, shuffle = True)
for i , (train_index, test_index) in enumerate(kfold.split(X)):
    print('Training cat model with fold {}...'.format(i + 1))
    X_train, X_test = X.ix[train_index], X.ix[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = CatBoostRegressor(iterations=200, learning_rate=0.03,
    depth=6, l2_leaf_reg=3,loss_function='MAE',eval_metric='MAE')
    models.append(model.fit(X_train, y_train))
    

months = np.array([10, 11, 12])
y_pred = []

for i in range(0, len(months)):
    pred = 0
    print(months[i])
    if months[i] != 0:
        X_validation['month'] = months[i]
    for model in models:
        print('next model...')
        pred += model.predict(X_validation)/kfolds
    y_pred.append(pred)

output = pd.DataFrame({'ParcelId': properties['parcelid'].astype(np.int32),
        '201610': y_pred[0], '201611': y_pred[1], '201612': y_pred[2],
        '201710': y_pred[0], '201711': y_pred[1], '201712': y_pred[2]})


# Model training
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0, random_state = 0)
# drop outliers
X_train = X_train[abs(y) < 0.4]
y_train = y_train[abs(y) < 0.4]
xgboost_model = xgboost_playground.xgboost_train(X_train, y_train)
lgbm_model = lgbm_playground.lgbm_train(X_train, y_train)
nn_model = nn_playground.train(X_train, y_train)
# Model validation
if add_months:
    months = np.array([10, 11, 12])
    y_pred_xgboost = [None]*len(months)
    y_pred_lgbm = [None]*len(months)
    y_pred_nn = [None]*len(months)
    y_pred = []
    for i in range(0, len(months)):
        #y_pred_xgboost[i] = xgboost_playground.xgboost_validate(X_validation, xgboost_model, months[i])
        y_pred_lgbm[i] = lgbm_playground.lgbm_validate(X_validation, lgbm_model, months[i])
        #y_pred_nn[i] = nn_playground.validate(X_validation, nn_model, months[i])
        #y_pred.append(np.add(y_pred_xgboost[i], y_pred_lgbm[i])*0.5)
        y_pred.append(y_pred_lgbm[i])
    output = pd.DataFrame({'ParcelId': properties['parcelid'].astype(np.int32),
        '201610': y_pred[0], '201611': y_pred[1], '201612': y_pred[2],
        '201710': y_pred[0], '201711': y_pred[1], '201712': y_pred[2]})
else:
    y_pred_xgboost = xgboost_playground.xgboost_validate(X_validation, xgboost_model)
    y_pred_lgbm = lgbm_playground.lgbm_validate(X_validation, lgbm_model)
    y_pred = np.add(y_pred_xgboost[i], y_pred_lgbm[i])*0.5
    output = pd.DataFrame({'ParcelId': properties['parcelid'].astype(np.int32),
        '201610': y_pred, '201611': y_pred, '201612': y_pred,
        '201710': y_pred, '201711': y_pred, '201712': y_pred})

# set col 'ParceID' to first col
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]
data_loader.save_data(output, 'sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')))
