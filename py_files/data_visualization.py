'''
Created on 1 paü 2017

@author: Zibski
'''
import matplotlib.pyplot as plt
import seaborn as sns

import data_loader

os.chdir('C:\kaggle\zillow\zestimate')

data_files = data_loader.load_all_data_files()
properties = data_loader.load_data(data_files[0])
train = data_loader.load_data(data_files[1])

train_df = train.merge(properties, how='left', on='parcelid')


train_df.plot.scatter('longitude', 'latitude',alpha=0.7, s=5, c='logerror', colormap='seismic')