# -*- coding: utf-8 -*-
'''
@authors: codeWoker & Zibski
'''
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime

import data_loader
import data_processing
import xgboost_playground
'''
...
'''
data_files, properties_files = data_loader.load_all_data_files()
properties = data_loader.load_data(properties_files[len(properties_files)-1])
train_data = data_loader.load_data(data_files)
properties = data_processing.data_cleaning_and_labeling(properties)
df = train_data.merge(properties, how='left', on='parcelid')

y = df['logerror'].values.astype(np.float32)
X_validation = properties.drop(['parcelid'], axis=1)

# Splitting data into train and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size = 0, random_state = 0)

X_train = data_processing.drop_outliers(X_train, ['logerror'], [1])
y_train = data_processing.drop_outliers(pd.DataFrame({'logerror' : y_train}), ['logerror'], [1])
X_train = X_train.drop(['parcelid', 'logerror','transactiondate'], axis=1)
X_test = X_test.drop(['parcelid', 'logerror','transactiondate'], axis=1)

print('Shape train: {}\nShape test: {}'.format(X_train.shape, X_test.shape))

MAE, model = xgboost_playground.xgboost_train_and_test(X_train, X_test, y_train, y_test)
print(MAE)

log_err_bound_tunning = False
if log_err_bound_tunning:
    log_err_bounds = [0.225, 0.25, 0.275, 0.4]
    for idx, bound in enumerate(log_err_bounds):
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size = 0.3, random_state = 0)
        X_train = data_processing.drop_outliers(X_train, ['logerror'], [bound])
        y_train = data_processing.drop_outliers(pd.DataFrame({'logerror' : y_train}), ['logerror'], [bound])
        X_train = X_train.drop(['parcelid', 'logerror','transactiondate'], axis=1)
        X_test = X_test.drop(['parcelid', 'logerror','transactiondate'], axis=1)
        MAE, model = xgboost_playground.xgboost_train_and_test(X_train, X_test, y_train, y_test)
        print('log error outliers bound: %s - MAE = %s', bound, MAE)        

# Model validation
y_pred = xgboost_playground.xgboost_validate(X_train, X_validation, y_train)

output = pd.DataFrame({'ParcelId': properties['parcelid'].astype(np.int32),
        '201610': y_pred, '201611': y_pred, '201612': y_pred,
        '201710': y_pred, '201711': y_pred, '201712': y_pred})
# set col 'ParceID' to first col
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]
data_loader.save_data(output, 'sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')))
# Save model to sav file
pickle.dump(model, open(os.path.abspath(os.getcwd()+'\\..')+'\\models\\' + 'model_{}.sav'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), 'wb'))
