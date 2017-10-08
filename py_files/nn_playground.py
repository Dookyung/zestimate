# -*- coding: utf-8 -*-
'''
@authors: codeWoker & Zibski
'''
import sklearn.metrics
from sklearn.neural_network import MLPRegressor
import os
import pickle
from datetime import datetime
'''
Functions
'''
def grid_search(X_train, X_test, y_train, y_test):
    alpha_values = [0.003]
    hidden_layer_sizes_values = [(20,20,10,), (25,25,), (20,15,5,)]
    best_mae = 10
    best_params = {'alpha' : None, 'hidden_layer_sizes' : None, 'max_iter' : 5000, 
                 'activation' : 'logistic', 'verbose' : 'True', 'learning_rate' : 'adaptive'}
    for alpha in alpha_values:
        for hidden_layer_size in hidden_layer_sizes_values:
            # nn params
            params = {'alpha' : alpha, 'hidden_layer_sizes' : hidden_layer_size, 'max_iter' : 50000, 
                 'activation' : 'logistic', 'verbose' : 'True', 'learning_rate' : 'adaptive'}
            # train model
            clf = MLPRegressor(alpha = alpha, hidden_layer_sizes = hidden_layer_size, max_iter = 5000, 
                 activation = 'logistic', verbose = 'True', learning_rate = 'adaptive', tol = 1e-4)
            model = clf.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # MAE Calculation
            mae = sklearn.metrics.mean_absolute_error(y_test, y_pred)
            print(mae)
            if mae < best_mae:
                best_mae = mae
                best_params = params
                best_model = model
                print(best_params)
    return best_model, best_params
                        

def train(X_train, y_train):
    params = {'alpha' : 0.003, 'hidden_layer_sizes' : (20,20,10,), 'max_iter' : 50000, 
                 'activation' : 'logistic', 'verbose' : 'True', 'learning_rate' : 'adaptive'}
    clf = MLPRegressor(alpha = 0.003, hidden_layer_sizes = (20,20,10,), max_iter = 5000, 
                 activation = 'logistic', verbose = 'True', learning_rate = 'adaptive')
    model = clf.fit(X_train, y_train)
    # Save model parameters to file
    f = open(os.path.abspath(os.getcwd()+'\\..')+'\\models\\' + 'model_{}_parameters.txt'.format(datetime.now().strftime('%Y%m%d_%H%M%S')),'w')
    f.write(str(params))
    f.close()
    # Save model to sav file
    pickle.dump(model, open(os.path.abspath(os.getcwd()+'\\..')+'\\models\\' + 'model_{}.sav'.format(datetime.now().strftime('%Y%m%d_%H%M%S')), 'wb'))
    return model


def validate(X_validation, model, month = 0):
    if month != 0:
        X_validation['month'] = month
    return model.predict(X_validation)