# -*- coding: utf-8 -*-
'''
@authors: codeWoker & Zibski
'''
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import pickle
import os

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
    return data


def drop_outliers(data, parameters, bounds):
    for idx, parameter in enumerate(parameters):
        data = data[ data[parameter] > - bounds[idx] ]
        data = data[ data[parameter] < bounds[idx] ]
    return data


def save_results_and_model(properties, y_pred, model, model_params):
    output = pd.DataFrame({'ParcelId': properties['parcelid'].astype(np.int32),
        '201610': y_pred, '201611': y_pred, '201612': y_pred,
        '201710': y_pred, '201711': y_pred, '201712': y_pred})
    cols = output.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    output = output[cols]
    from datetime import datetime
    data_loader.save_data(output, 'sub{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')))
    # Save model to sav file
    pickle.dump(model, open(os.path.abspath(os.getcwd()+'\\..')+'\\models\\' + 'model_{}.sav'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), 'wb'))
    # Save model parameters to file
    f = open(os.path.abspath(os.getcwd()+'\\..')+'\\models\\' + 'model_{}_parameters.txt'.format(datetime.now().strftime('%Y%m%d_%H%M%S')),'w')
    f.write(str(model_params))
    f.close()