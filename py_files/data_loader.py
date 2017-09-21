# -*- coding: utf-8 -*-
'''
Created on Sun Sep 17 15:15:17 2017

@author: lukasz
'''

import pandas as pd
import os.path
import os


def load_data(data_file):
    return pd.read_csv(data_file, sep=',')


def load_all_data_files():
    all_data_files = []
    for root, dirs, files in os.walk(os.path.abspath(os.getcwd() + '\\..\\data')):
        for file in files:
            if file.endswith('.csv'):
                all_data_files.append(os.path.join(root, file))
    return all_data_files


def save_data(df, file_name):
    df.to_csv(os.path.abspath(os.getcwd()+'\\..') + '\\results' + '\\' + file_name, index = False)