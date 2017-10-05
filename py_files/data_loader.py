# -*- coding: utf-8 -*-
'''
@authors: codeWoker & Zibski
'''
import pandas as pd
import os.path
import os
'''
Functions
'''

def load_data(files):
    df_all = pd.DataFrame()
    if type(files) != list:
        df_all = pd.read_csv(files, sep=',')
    else:
        for i, file in enumerate(files):
            df = pd.read_csv(file, sep=',')
            df_all = pd.concat([df_all, df], axis = 0)
    return df_all


def load_all_data_files():
    data_files = []
    properties_files = []
    for root, dirs, files in os.walk(os.path.abspath(os.getcwd() + '\\..\\data')):
        for file in files:
            if file.endswith('.csv') and file.startswith('train') :
                data_files.append(os.path.join(root, file))
            if file.endswith('.csv') and file.startswith('prop'):
                properties_files.append(os.path.join(root, file))
    return data_files, properties_files


def save_data(df, file_name):
    df.to_csv(os.path.abspath(os.getcwd()+'\\..') + '\\results' + '\\' + file_name, index = False)