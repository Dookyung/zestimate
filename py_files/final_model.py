import numpy as np
import pandas as pd
import gc
import os
from datetime import datetime
from catboost import CatBoostRegressor
from tqdm import tqdm
import datetime as dt

prop_2016 = pd.read_csv('../data/properties_2016.csv', low_memory = False)
prop_2017 = pd.read_csv('../data/properties_2017.csv', low_memory = False)

data_2016 = pd.read_csv('../data/train_2016_v2.csv', parse_dates=['transactiondate'], low_memory=False)
data_2017 = pd.read_csv('../data/train_2017.csv', parse_dates=['transactiondate'], low_memory=False)

def add_date_feature(data):
    data['month'] = (data['transactiondate'].dt.year - 2016)*12 + data['transactiondate'].dt.month
    data['month'][data['month'] > 12] = data['month'] -12
    data.drop(['transactiondate'], inplace=True, axis=1)
    return data

data_2016 = add_date_feature(data_2016)
data_2017 = add_date_feature(data_2017)
data_2016 = pd.merge(data_2016, prop_2016, how = 'left', on = 'parcelid')
data_2017 = pd.merge(data_2017, prop_2017, how = 'left', on = 'parcelid')
data_2017.iloc[:, data_2017.columns.str.startswith('tax')] = np.nan

train_df = pd.concat([data_2016, data_2017], axis = 0)
test_df = prop_2016.rename(columns = {'parcelid': 'ParcelId'})

del prop_2016, prop_2017, data_2016, data_2017
gc.collect();

def data_cleaner(data):
    nan_rem_prc_threshold = 0.995 # it can be changed
    col_to_exclude = []
    row_numb = data.shape[0]
    for c in data.columns:
        num_missing = data[c].isnull().sum()
        if num_missing == 0:
            continue
        missing_frac = num_missing / float(row_numb)
        if missing_frac > nan_rem_prc_threshold:
            col_to_exclude.append(c)
    col_to_exclude_with_unique = []
    for c in data.columns:
        num_uniques = len(data[c].unique())
        if data[c].isnull().sum() != 0:
            num_uniques -= 1
        if num_uniques == 1:
            col_to_exclude_with_unique.append(c)
    other_col_to_exclude = ['parcelid', 'logerror','propertyzoningdesc', 'censustractandblock', 'rawcensustractandblock']
    train_features = []
    for c in data.columns:
        if c not in col_to_exclude \
           and c not in other_col_to_exclude and c not in col_to_exclude_with_unique:
            train_features.append(c)
    return train_features

train_features = data_cleaner(train_df)

def categorical_features(data, train_features):
    categorical_features_idx = []
    categorical_unique_thresh = 1000
    for i, c in enumerate(train_features):
        num_uniques = len(data[c].unique())
        if num_uniques < categorical_unique_thresh \
           and not 'sqft' in c \
           and not 'cnt' in c \
           and not 'nbr' in c \
           and not 'number' in c:
            categorical_features_idx.append(i)
    return categorical_features_idx

categorical_features_idx = categorical_features(train_df, train_features)
train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)

def data_opt(data):
    print('Memory usage reduction...')
    for c in data.columns:
        if data[c].dtype == int:
            data[c] = data[c].astype(np.int32)
        if data[c].dtype == float:
            data[c] = data[c].astype(np.float32)
    return data

train_df = data_opt(train_df)
test_df = data_opt(test_df)

#
X_train = train_df[train_features]
y_train = train_df.logerror

# drop outliers
X_train = X_train[abs(y_train) < 0.4]
y_train = y_train[abs(y_train) < 0.4]

months = np.array([10, 11, 12])
models = []
y_pred = [None]*len(months)
num_ensembles = 5

for i in tqdm(range(num_ensembles)):
    model = CatBoostRegressor(
        iterations = 630, learning_rate = 0.03,
        depth = 6, l2_leaf_reg = 3,
        loss_function='MAE',
        eval_metric='MAE',
        random_seed = i)
    models.append(model.fit(X_train, y_train, cat_features = categorical_features_idx))

for j in range(0, len(months)):
    print(months[j])
    if months[j] != 0:
        test_df['month'] = months[j]
        X_test = test_df[train_features]
    y_pred[j] = 0.0
    for model in models:
        y_pred[j] += model.predict(X_test)
    y_pred[j] /= num_ensembles

output = pd.DataFrame({'ParcelId': test_df['ParcelId'].astype(np.int32),
        '201610': y_pred[0], '201611': y_pred[1], '201612': y_pred[2],
        '201710': y_pred[0], '201711': y_pred[1], '201712': y_pred[2]})

    
def save_data(df, file_name):
    df.to_csv(os.path.abspath(os.getcwd()+'\\..') + '\\results' + '\\' + file_name, index = False, float_format = '%.6f')

    
# set col 'ParceID' to first col
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]
save_data(output, 'sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')))

'''
LOCAL TESTER
'''
from sklearn.cross_validation import train_test_split
import sklearn.metrics

X_train = train_df[train_features]
y_train = train_df.logerror

# drop outliers
#X_train = X_train[abs(y_train) < 0.4]
#y_train = y_train[abs(y_train) < 0.4]

# Model training
X_train_loc, X_test_loc, y_train_loc, y_test_loc = train_test_split(X_train, y_train, test_size = 0.25, random_state = 10)



y_pred = 0
num_ensembles = 5
models = []
for i in tqdm(range(num_ensembles)):
    model = CatBoostRegressor(
        iterations=630, learning_rate=0.03,
        depth=6, l2_leaf_reg=3,
        loss_function='MAE',
        eval_metric='MAE',
        random_seed=i)
    print('training')
    models.append(model.fit(X_train_loc, y_train_loc, cat_features = categorical_features_idx))
    print('testing')
    y_pred += model.predict(X_test_loc)
y_pred /= num_ensembles
print(sklearn.metrics.mean_absolute_error(y_test_loc, y_pred))