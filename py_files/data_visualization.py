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
properties['nans'] = properties.isnull().sum(axis = 1)
train = data_loader.load_data(data_files[1])

train_df = train.merge(properties, how='left', on='parcelid')


train_df.plot.scatter('longitude', 'latitude',alpha=0.7, s=5, c='logerror', colormap='seismic')



plt.plot(X['nans'], X['logerror'], 'ro')
plt.xlabel('amount of nans in data')
ply.ylabel('log error')
plt.show()

sns.jointplot(x=X['nans'].values, y=X['logerror'].values)
plt.xlabel('amount of nans in data', fontsize = 12)
plt.ylabel('log error', fontsize = 12)
plt.show()


plt.hist(X['nans'], bins = 40, normed = True)
plt.xlabel('amount of nans in data', fontsize = 12)
plt.ylabel('density', fontsize = 12)
plt.show()
