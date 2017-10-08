# -*- coding: utf-8 -*-
'''
@authors: codeWoker & Zibski
'''
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import pickle
import os
import datetime as dt

import data_loader
'''
Functions
'''

def data_cleaning_and_labeling(data):
    for c in data.columns:
        data[c]=data[c].fillna(-1)
        if data[c].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(data[c].values))
            data[c] = lbl.transform(list(data[c].values))
    print('Memory usage reduction...')
    for c in data.columns:
        if data[c].dtype == int:
            data[c] = data[c].astype(np.int32)
        if data[c].dtype == float:
            data[c] = data[c].astype(np.float32)
    data[['latitude', 'longitude']] /= 1e6
    return data


def drop_outliers(data, parameters, bounds):
    for idx, parameter in enumerate(parameters):
        data = data[ data[parameter] > - bounds[idx] ]
        data = data[ data[parameter] < bounds[idx] ]
    return data


def feature_month(data):
    data['month'] = (pd.to_datetime(data['transactiondate']).dt.year - 2016)*12 + pd.to_datetime(data['transactiondate']).dt.month
    return data