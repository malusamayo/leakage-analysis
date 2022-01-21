#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import random
from sklearn import preprocessing

import gc
from scipy.stats import skew, boxcox

from scipy import sparse
from sklearn.metrics import log_loss
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

seed = 2017


# In[2]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU,LeakyReLU,ELU,ParametricSoftplus,ThresholdedReLU,SReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.optimizers import SGD,Nadam
from keras.regularizers import WeightRegularizer, ActivityRegularizer,l2, activity_l2
from keras.utils.np_utils import to_categorical


# # Load Data

# In[3]:


names = ['low_0','medium_0','high_0',
        'low_1','medium_1','high_1',
        'low_2','medium_2','high_2',
        'low_3','medium_3','high_3',
        'low_4','medium_4','high_4',
        'low_5','medium_5','high_5',
        'low_6','medium_6','high_6',
        'low_7','medium_7','high_7',
        'low_8','medium_8','high_8',
        'low_9','medium_9','high_9']

data_path = "../2nd/"
total_col = 0


# In[4]:


# RFC 1st level 
file_train      = 'train_blend_RFC_gini_BM_MB_add03052240_2017-03-10-22-02' + '.csv'
file_test_mean  = 'test_blend_RFC_gini_mean_BM_MB_add03052240_2017-03-10-22-02' + '.csv'
file_test_gmean = 'test_blend_RFC_gini_gmean_BM_MB_add03052240_2017-03-10-22-02' + '.csv'

train_rfc_gini      = pd.read_csv(data_path + file_train,      header = None)
test_rfc_gini_mean  = pd.read_csv(data_path + file_test_mean,  header = None)
test_rfc_gini_gmean = pd.read_csv(data_path + file_test_gmean, header = None)

n_column = train_rfc_gini.shape[1]
total_col += n_column

train_rfc_gini.columns      = ['rfc_gini_' + x for x in names[:n_column]]
test_rfc_gini_mean.columns  = ['rfc_gini_' + x for x in names[:n_column]]
test_rfc_gini_gmean.columns = ['rfc_gini_' + x for x in names[:n_column]]

file_train      = 'train_blend_RFC_entropy_BM_MB_add03052240_2017-03-10-21-10' + '.csv'
file_test_mean  = 'test_blend_RFC_entropy_mean_BM_MB_add03052240_2017-03-10-21-10' + '.csv'
file_test_gmean = 'test_blend_RFC_entropy_gmean_BM_MB_add03052240_2017-03-10-21-10' + '.csv'

train_rfc_entropy      = pd.read_csv(data_path + file_train,      header = None)
test_rfc_entropy_mean  = pd.read_csv(data_path + file_test_mean,  header = None)
test_rfc_entropy_gmean = pd.read_csv(data_path + file_test_gmean, header = None)

n_column = train_rfc_entropy.shape[1]
total_col += n_column

train_rfc_entropy.columns      = ['rfc_entropy_' + x for x in names[:n_column]]
test_rfc_entropy_mean.columns  = ['rfc_entropy_' + x for x in names[:n_column]]
test_rfc_entropy_gmean.columns = ['rfc_entropy_' + x for x in names[:n_column]]



print('train_rfc_gini: {}\t test_rfc_gini_mean:{}\t test_rfc_gini_gmean:{}'.        format(train_rfc_gini.shape,test_rfc_gini_mean.shape,test_rfc_gini_gmean.shape))
print('\ntrain_rfc_entropy: {}\t test_rfc_entropy_mean:{}\t test_rfc_entropy_gmean:{}'.        format(train_rfc_entropy.shape,test_rfc_entropy_mean.shape,test_rfc_entropy_gmean.shape))

    
print('\ntrain_rfc_gini')
print(train_rfc_gini.iloc[:5,:3])
print('\ntrain_rfc_entropy')
print(train_rfc_entropy.iloc[:5,:3])


# In[5]:


# RFC 1st level 0322
file_train      = 'train_blend_RFC_gini_BM_0322_2017-03-22-17-12' + '.csv'
file_test_mean  = 'test_blend_RFC_gini_mean_BM_0322_2017-03-22-17-12' + '.csv'
file_test_gmean = 'test_blend_RFC_gini_gmean_BM_0322_2017-03-22-17-12' + '.csv'

train_rfc_gini_0322      = pd.read_csv(data_path + file_train,      header = None)
test_rfc_gini_mean_0322  = pd.read_csv(data_path + file_test_mean,  header = None)
test_rfc_gini_gmean_0322 = pd.read_csv(data_path + file_test_gmean, header = None)

n_column = train_rfc_gini_0322.shape[1]
total_col += n_column

train_rfc_gini_0322.columns      = ['rfc_gini_0322_' + x for x in names[:n_column]]
test_rfc_gini_mean_0322.columns  = ['rfc_gini_0322_' + x for x in names[:n_column]]
test_rfc_gini_gmean_0322.columns = ['rfc_gini_0322_' + x for x in names[:n_column]]


file_train      = 'train_blend_RFC_entropy_BM_0322_2017-03-22-16-02' + '.csv'
file_test_mean  = 'test_blend_RFC_entropy_mean_BM_0322_2017-03-22-16-02' + '.csv'
file_test_gmean = 'test_blend_RFC_entropy_gmean_BM_0322_2017-03-22-16-02' + '.csv'

train_rfc_entropy_0322      = pd.read_csv(data_path + file_train,      header = None)
test_rfc_entropy_mean_0322  = pd.read_csv(data_path + file_test_mean,  header = None)
test_rfc_entropy_gmean_0322 = pd.read_csv(data_path + file_test_gmean, header = None)

n_column = train_rfc_entropy_0322.shape[1]
total_col += n_column

train_rfc_entropy_0322.columns      = ['rfc_entropy_0322_' + x for x in names[:n_column]]
test_rfc_entropy_mean_0322.columns  = ['rfc_entropy_0322_' + x for x in names[:n_column]]
test_rfc_entropy_gmean_0322.columns = ['rfc_entropy_0322_' + x for x in names[:n_column]]


print('\ntrain_rfc_entropy: {}\t test_rfc_entropy_mean:{}\t test_rfc_entropy_gmean:{}'.        format(train_rfc_gini_0322.shape,test_rfc_gini_mean_0322.shape,test_rfc_gini_gmean_0322.shape))
print('\ntrain_rfc_entropy: {}\t test_rfc_entropy_mean:{}\t test_rfc_entropy_gmean:{}'.        format(train_rfc_entropy_0322.shape,test_rfc_entropy_mean_0322.shape,test_rfc_entropy_gmean_0322.shape))
    
    
print('\ntrain_rfc_gini_0322')
print(train_rfc_gini_0322.iloc[:5,:3])
print('\ntrain_rfc_entropy_0322')
print(train_rfc_entropy_0322.iloc[:5,:3])


# In[6]:


# LR 1st level
file_train = 'train_blend_LR_BM_2017-03-09-02-38' + '.csv'
file_test_mean = 'test_blend_LR_mean_BM_2017-03-09-02-38' + '.csv'
file_test_gmean = 'test_blend_LR_gmean_BM_2017-03-09-02-38' + '.csv'

train_LR      = pd.read_csv(data_path + file_train, header = None)
test_LR_mean  = pd.read_csv(data_path + file_test_mean, header = None)
test_LR_gmean = pd.read_csv(data_path + file_test_gmean, header = None)

n_column = train_LR.shape[1]
total_col += n_column

train_LR.columns      = ['LR_' + x for x in names[:n_column]]
test_LR_mean.columns  = ['LR_' + x for x in names[:n_column]]
test_LR_gmean.columns = ['LR_' + x for x in names[:n_column]]

print('train_LR: {}\t test_LR_mean:{}\t test_LR_gmean:{}'.        format(train_LR.shape,test_LR_mean.shape,test_LR_gmean.shape))

print('\ntrain_LR')
print(train_LR.iloc[:5,:3])



# In[7]:


# LR 1st level 0322
file_train = 'train_blend_LR_BM_0322_2017-03-22-23-38' + '.csv'
file_test_mean = 'test_blend_LR_mean_BM_0322_2017-03-22-23-38' + '.csv'
file_test_gmean = 'test_blend_LR_gmean_BM_0322_2017-03-22-23-38' + '.csv'

train_LR_0322      = pd.read_csv(data_path + file_train, header = None)
test_LR_mean_0322  = pd.read_csv(data_path + file_test_mean, header = None)
test_LR_gmean_0322 = pd.read_csv(data_path + file_test_gmean, header = None)

n_column = train_LR_0322.shape[1]
total_col += n_column

train_LR_0322.columns      = ['LR_0322_' + x for x in names[:n_column]]
test_LR_mean_0322.columns  = ['LR_0322_' + x for x in names[:n_column]]
test_LR_gmean_0322.columns = ['LR_0322_' + x for x in names[:n_column]]

print('train_LR_0322: {}\t test_LR_mean_0322:{}\t test_LR_gmean_0322:{}'.        format(train_LR_0322.shape,test_LR_mean_0322.shape,test_LR_gmean_0322.shape))

print('\ntrain_LR_0322')
print(train_LR_0322.iloc[:5,:3])


# In[8]:


# ET 1st level
file_train      = 'train_blend_ET_gini_BM_2017-03-10-09-42' + '.csv'
file_test_mean  = 'test_blend_ET_gini_mean_BM_2017-03-10-09-42' + '.csv'
file_test_gmean = 'test_blend_ET_gini_gmean_BM_2017-03-10-09-42' + '.csv'

train_ET_gini      = pd.read_csv(data_path + file_train,      header = None)
test_ET_gini_mean  = pd.read_csv(data_path + file_test_mean,  header = None)
test_ET_gini_gmean = pd.read_csv(data_path + file_test_gmean, header = None)

n_column = train_ET_gini.shape[1]
total_col += n_column

train_ET_gini.columns      = ['ET_gini_' + x for x in names[:n_column]]
test_ET_gini_mean.columns  = ['ET_gini_' + x for x in names[:n_column]]
test_ET_gini_gmean.columns = ['ET_gini_' + x for x in names[:n_column]]

file_train      = 'train_blend_ET_entropy_BM_2017-03-09-20-44' + '.csv'
file_test_mean  = 'test_blend_ET_entropy_mean_BM_2017-03-09-20-44' + '.csv'
file_test_gmean = 'test_blend_ET_entropy_gmean_BM_2017-03-09-20-44' + '.csv'

train_ET_entropy      = pd.read_csv(data_path + file_train,      header = None)
test_ET_entropy_mean  = pd.read_csv(data_path + file_test_mean,  header = None)
test_ET_entropy_gmean = pd.read_csv(data_path + file_test_gmean, header = None)

n_column = train_ET_entropy.shape[1]
total_col += n_column

train_ET_entropy.columns      = ['ET_entropy_' + x for x in names[:n_column]]
test_ET_entropy_mean.columns  = ['ET_entropy_' + x for x in names[:n_column]]
test_ET_entropy_gmean.columns = ['ET_entropy_' + x for x in names[:n_column]]

print('train_ET_gini: {}\t test_ET_gini_mean:{}\t test_ET_gini_gmean:{}'.        format(train_ET_gini.shape,test_ET_gini_mean.shape,test_ET_gini_gmean.shape))
print('\ntrain_ET_entropy: {}\t test_ET_entropy_mean:{}\t test_ET_entropy_gmean:{}'.        format(train_ET_entropy.shape,test_ET_entropy_mean.shape,test_ET_entropy_gmean.shape))
    
    
print('\ntrain_ET_gini')
print(train_ET_gini.iloc[:5,:3])
print('\ntrain_ET_entropy')
print(train_ET_entropy.iloc[:5,:3])





# In[9]:


# ET 1st level 0322
file_train      = 'train_blend_ET_gini_BM_0322_2017-03-23-16-04' + '.csv'
file_test_mean  = 'test_blend_ET_gini_mean_BM_0322_2017-03-23-16-04' + '.csv'
file_test_gmean = 'test_blend_ET_gini_gmean_BM_0322_2017-03-23-16-04' + '.csv'

train_ET_gini_0322      = pd.read_csv(data_path + file_train,      header = None)
test_ET_gini_mean_0322  = pd.read_csv(data_path + file_test_mean,  header = None)
test_ET_gini_gmean_0322 = pd.read_csv(data_path + file_test_gmean, header = None)

n_column = train_ET_gini_0322.shape[1]
total_col += n_column

train_ET_gini_0322.columns      = ['ET_gini_0322_' + x for x in names[:n_column]]
test_ET_gini_mean_0322.columns  = ['ET_gini_0322_' + x for x in names[:n_column]]
test_ET_gini_gmean_0322.columns = ['ET_gini_0322_' + x for x in names[:n_column]]

file_train      = 'train_blend_ET_entropy_BM_0322_2017-03-23-13-40' + '.csv'
file_test_mean  = 'test_blend_ET_entropy_mean_BM_0322_2017-03-23-13-40' + '.csv'
file_test_gmean = 'test_blend_ET_entropy_gmean_BM_0322_2017-03-23-13-40' + '.csv'

train_ET_entropy_0322      = pd.read_csv(data_path + file_train,      header = None)
test_ET_entropy_mean_0322  = pd.read_csv(data_path + file_test_mean,  header = None)
test_ET_entropy_gmean_0322 = pd.read_csv(data_path + file_test_gmean, header = None)

n_column = train_ET_entropy_0322.shape[1]
total_col += n_column

train_ET_entropy_0322.columns      = ['ET_entropy_0322_' + x for x in names[:n_column]]
test_ET_entropy_mean_0322.columns  = ['ET_entropy_0322_' + x for x in names[:n_column]]
test_ET_entropy_gmean_0322.columns = ['ET_entropy_0322_' + x for x in names[:n_column]]

print('train_ET_gini_0322: {}\t test_ET_gini_mean_0322:{}\t test_ET_gini_gmean_0322:{}'.        format(train_ET_gini_0322.shape,test_ET_gini_mean_0322.shape,test_ET_gini_gmean_0322.shape))
print('\ntrain_ET_entropy_0322: {}\t test_ET_entropy_mean_0322:{}\t test_ET_entropy_gmean_0322:{}'.        format(train_ET_entropy_0322.shape,test_ET_entropy_mean_0322.shape,test_ET_entropy_gmean_0322.shape))
    
    
print('\ntrain_ET_gini_0322')
print(train_ET_gini_0322.iloc[:5,:3])
print('\ntrain_ET_entropy_0322')
print(train_ET_entropy_0322.iloc[:5,:3])


# In[10]:


# KNN 1st level
file_train      = 'train_blend_KNN_uniform_BM_MB_add03052240_2017-03-11-18-31' + '.csv'
file_test_mean  = 'test_blend_KNN_uniform_mean_BM_MB_add03052240_2017-03-11-18-31' + '.csv'
file_test_gmean = 'test_blend_KNN_uniform_gmean_BM_MB_add03052240_2017-03-11-18-31' + '.csv'

train_KNN_uniform      = pd.read_csv(data_path + file_train,      header = None)
test_KNN_uniform_mean  = pd.read_csv(data_path + file_test_mean,  header = None)
test_KNN_uniform_gmean = pd.read_csv(data_path + file_test_gmean, header = None)

n_column = train_KNN_uniform.shape[1]
total_col += n_column

train_KNN_uniform.columns      = ['KNN_uniform_' + x for x in names[:n_column]]
test_KNN_uniform_mean.columns  = ['KNN_uniform_' + x for x in names[:n_column]]
test_KNN_uniform_gmean.columns = ['KNN_uniform_' + x for x in names[:n_column]]

file_train      = 'train_blend_KNN_distance_BM_MB_add_2017-03-11-21-51' + '.csv'
file_test_mean  = 'test_blend_KNN_distance_mean_BM_MB_add_2017-03-11-21-51' + '.csv'
file_test_gmean = 'test_blend_KNN_distance_gmean_BM_MB_add_2017-03-11-21-51' + '.csv'

train_KNN_distance      = pd.read_csv(data_path + file_train,      header = None)
test_KNN_distance_mean  = pd.read_csv(data_path + file_test_mean,  header = None)
test_KNN_distance_gmean = pd.read_csv(data_path + file_test_gmean, header = None)

n_column = train_KNN_distance.shape[1]
total_col += n_column

train_KNN_distance.columns      = ['KNN_distance_' + x for x in names[:n_column]]
test_KNN_distance_mean.columns  = ['KNN_distance_' + x for x in names[:n_column]]
test_KNN_distance_gmean.columns = ['KNN_distance_' + x for x in names[:n_column]]

print('train_KNN_uniform: {}\t test_KNN_uniform_mean:{}\t test_KNN_uniform_gmean:{}'.        format(train_KNN_uniform.shape,test_KNN_uniform_mean.shape,test_KNN_uniform_gmean.shape))
print('\ntrain_KNN_distance: {}\t test_KNN_distance_mean:{}\t test_KNN_distance_gmean:{}'.        format(train_KNN_distance.shape,test_KNN_distance_mean.shape,test_KNN_distance_gmean.shape))
    
print('\ntrain_KNN_uniform')
print(train_KNN_uniform.iloc[:5,:3])
print('\ntrain_KNN_distance')
print(train_KNN_distance.iloc[:5,:3])


# In[11]:


# KNN 1st level 0322
file_train      = 'train_blend_KNN_uniform_BM_0322_2017-03-24-07-31' + '.csv'
file_test_mean  = 'test_blend_KNN_uniform_mean_BM_0322_2017-03-24-07-31' + '.csv'
file_test_gmean = 'test_blend_KNN_uniform_gmean_BM_0322_2017-03-24-07-31' + '.csv'

train_KNN_uniform_0322      = pd.read_csv(data_path + file_train,      header = None)
test_KNN_uniform_mean_0322  = pd.read_csv(data_path + file_test_mean,  header = None)
test_KNN_uniform_gmean_0322 = pd.read_csv(data_path + file_test_gmean, header = None)

n_column = train_KNN_uniform_0322.shape[1]
total_col += n_column

train_KNN_uniform_0322.columns      = ['KNN_uniform_0322_' + x for x in names[:n_column]]
test_KNN_uniform_mean_0322.columns  = ['KNN_uniform_0322_' + x for x in names[:n_column]]
test_KNN_uniform_gmean_0322.columns = ['KNN_uniform_0322_' + x for x in names[:n_column]]

file_train      = 'train_blend_KNN_distance_BM_0322_2017-03-25-08-17' + '.csv'
file_test_mean  = 'test_blend_KNN_distance_mean_BM_0322_2017-03-25-08-17' + '.csv'
file_test_gmean = 'test_blend_KNN_distance_gmean_BM_0322_2017-03-25-08-17' + '.csv'

train_KNN_distance_0322      = pd.read_csv(data_path + file_train,      header = None)
test_KNN_distance_mean_0322  = pd.read_csv(data_path + file_test_mean,  header = None)
test_KNN_distance_gmean_0322 = pd.read_csv(data_path + file_test_gmean, header = None)

n_column = train_KNN_distance_0322.shape[1]
total_col += n_column

train_KNN_distance_0322.columns      = ['KNN_distance_0322_' + x for x in names[:n_column]]
test_KNN_distance_mean_0322.columns  = ['KNN_distance_0322_' + x for x in names[:n_column]]
test_KNN_distance_gmean_0322.columns = ['KNN_distance_0322_' + x for x in names[:n_column]]

print('train_KNN_uniform_0322: {}\t test_KNN_uniform_mean_0322:{}\t test_KNN_uniform_gmean_0322:{}'.        format(train_KNN_uniform_0322.shape,test_KNN_uniform_mean_0322.shape,test_KNN_uniform_gmean_0322.shape))
print('\ntrain_KNN_distance: {}\t test_KNN_distance_mean_0322:{}\t test_KNN_distance_gmean_0322:{}'.        format(train_KNN_distance_0322.shape,test_KNN_distance_mean_0322.shape,test_KNN_distance_gmean_0322.shape))
    
print('\ntrain_KNN_uniform_0322')
print(train_KNN_uniform_0322.iloc[:5,:3])
print('\ntrain_KNN_distance_0322')
print(train_KNN_distance_0322.iloc[:5,:3])


# In[12]:


# TFFM 1st level
file_train      = 'train_blend_FM_BM_MB_add_desc_2017-03-16-09-52' + '.csv'
file_test_mean  = 'test_blend_FM_mean_BM_MB_add_desc_2017-03-16-09-52' + '.csv'
file_test_gmean = 'test_blend_FM_gmean_BM_MB_add_desc_2017-03-16-09-52' + '.csv'

train_FM      = pd.read_csv(data_path + file_train,      header = None)
test_FM_mean  = pd.read_csv(data_path + file_test_mean,  header = None)
test_FM_gmean = pd.read_csv(data_path + file_test_gmean, header = None)

n_column = train_FM.shape[1]
total_col += n_column

train_FM.columns      = ['FM_' + x for x in names[:n_column]]
test_FM_mean.columns  = ['FM_' + x for x in names[:n_column]]
test_FM_gmean.columns = ['FM_' + x for x in names[:n_column]]

print('train_FM: {}\t test_FM_mean:{}\t test_FM_gmean:{}'.        format(train_FM.shape,test_FM_mean.shape,test_FM_gmean.shape))

print('\ntrain_FM')
print(train_FM.iloc[:5,:3])


# TFFM 1st level 0322
file_train      = 'train_blend_FM_BM_0322_2017-03-27-04-35' + '.csv'
file_test_mean  = 'test_blend_FM_mean_BM_0322_2017-03-27-04-35' + '.csv'
file_test_gmean = 'test_blend_FM_gmean_BM_0322_2017-03-27-04-35' + '.csv'

train_FM_0322      = pd.read_csv(data_path + file_train,      header = None)
test_FM_mean_0322  = pd.read_csv(data_path + file_test_mean,  header = None)
test_FM_gmean_0322 = pd.read_csv(data_path + file_test_gmean, header = None)

n_column = train_FM_0322.shape[1]
total_col += n_column

train_FM_0322.columns      = ['FM_0322_' + x for x in names[:n_column]]
test_FM_mean_0322.columns  = ['FM_0322_' + x for x in names[:n_column]]
test_FM_gmean_0322.columns = ['FM_0322_' + x for x in names[:n_column]]

print('train_FM_0322: {}\t test_FM_mean_0322:{}\t test_FM_gmean_0322:{}'.        format(train_FM_0322.shape,test_FM_mean_0322.shape,test_FM_gmean_0322.shape))

print('\ntrain_FM_0322')
print(train_FM_0322.iloc[:5,:3])


# In[13]:


# Multinomial Naive Bayes 1st level
file_train      = 'train_blend_MNB_BM_MB_add03052240_2017-03-13-20-51' + '.csv'
file_test_mean  = 'test_blend_MNB_mean_BM_MB_add03052240_2017-03-13-20-51' + '.csv'
file_test_gmean = 'test_blend_MNB_gmean_BM_MB_add03052240_2017-03-13-20-51' + '.csv'

train_MNB      = pd.read_csv(data_path + file_train,      header = None)
test_MNB_mean  = pd.read_csv(data_path + file_test_mean,  header = None)
test_MNB_gmean = pd.read_csv(data_path + file_test_gmean, header = None)

n_column = train_MNB.shape[1]
total_col += n_column

train_MNB.columns      = ['MNB_' + x for x in names[:n_column]]
test_MNB_mean.columns  = ['MNB_' + x for x in names[:n_column]]
test_MNB_gmean.columns = ['MNB_' + x for x in names[:n_column]]

print('train_MNB: {}\t test_MNB_mean:{}\t test_MNB_gmean:{}'.        format(train_MNB.shape,test_MNB_mean.shape,test_MNB_gmean.shape))
    
print('\ntrain_MNB')
print(train_MNB.iloc[:5,:3])


# In[14]:


# TSNE 1st level

file_train = 'X_train_tsne_BM_MB_add_desc_2017-03-18-17-14' + '.csv'
file_test  = 'X_test_tsne_BM_MB_add_desc_2017-03-18-17-14' + '.csv'

train_tsne = pd.read_csv(data_path + file_train, header = None)
test_tsne  = pd.read_csv(data_path + file_test, header = None)


n_column = train_tsne.shape[1]
total_col += n_column

train_tsne.columns = ['tsne_0', 'tsne_1', 'tsne_2']
test_tsne.columns  = ['tsne_0', 'tsne_1', 'tsne_2']

print('train_tsne: {}\t test_tsne:{}'.        format(train_tsne.shape,test_tsne.shape))
    
print('\ntrain_tsne')
print(train_tsne.iloc[:5,:3])


# TSNE 1st level 0322

file_train = 'X_train_tsne_BM_0322_2017-03-26-16-33' + '.csv'
file_test  = 'X_test_tsne_BM_0322_2017-03-26-16-33' + '.csv'

train_tsne_0322 = pd.read_csv(data_path + file_train, header = None)
test_tsne_0322  = pd.read_csv(data_path + file_test, header = None)


n_column = train_tsne_0322.shape[1]
total_col += n_column

train_tsne_0322.columns = ['tsne_0_0322', 'tsne_1_0322', 'tsne_2_0322']
test_tsne_0322.columns  = ['tsne_0_0322', 'tsne_1_0322', 'tsne_2_0322']

print('train_tsne_0322: {}\t test_tsne_0322:{}'.        format(train_tsne_0322.shape,test_tsne_0322.shape))
    
print('\ntrain_tsne_0322')
print(train_tsne_0322.iloc[:5,:3])


# In[15]:


# XGB 1st level

file_train = 'train_blend_xgb_BM_MB_add_desc_2017-03-14-16-54' + '.csv'
file_test_mean = 'test_blend_xgb_mean_BM_MB_add_desc_2017-03-14-16-54' + '.csv'
file_test_gmean = 'test_blend_xgb_gmean_BM_MB_add_desc_2017-03-14-16-54' + '.csv'

train_xgb      = pd.read_csv(data_path + file_train, header = None)
test_xgb_mean  = pd.read_csv(data_path + file_test_mean, header = None)
test_xgb_gmean = pd.read_csv(data_path + file_test_gmean, header = None)

n_column = train_xgb.shape[1]
total_col += n_column

train_xgb.columns = ['xgb_' + x for x in names[:n_column]]
test_xgb_mean.columns = ['xgb_' + x for x in names[:n_column]]
test_xgb_gmean.columns = ['xgb_' + x for x in names[:n_column]]

print('train_xgb: {}\t test_xgb_mean:{}\t test_xgb_gmean:{}'.        format(train_xgb.shape,test_xgb_mean.shape,test_xgb_gmean.shape))
    
print('\ntrain_xgb')
print(train_xgb.iloc[:5,:3])


# In[16]:


# XGB 1st level 0322

file_train      = 'train_blend_xgb_BM_0322_2017-03-25-19-12' + '.csv'
file_test_mean  = 'test_blend_xgb_mean_BM_0322_2017-03-25-19-12' + '.csv'
file_test_gmean = 'test_blend_xgb_gmean_BM_0322_2017-03-25-19-12' + '.csv'

train_xgb_0322      = pd.read_csv(data_path + file_train, header = None)
test_xgb_mean_0322  = pd.read_csv(data_path + file_test_mean, header = None)
test_xgb_gmean_0322 = pd.read_csv(data_path + file_test_gmean, header = None)

n_column = train_xgb_0322.shape[1]
total_col += n_column

train_xgb_0322.columns      = ['xgb_0322_' + x for x in names[:n_column]]
test_xgb_mean_0322.columns  = ['xgb_0322_' + x for x in names[:n_column]]
test_xgb_gmean_0322.columns = ['xgb_0322_' + x for x in names[:n_column]]

print('train_xgb_0322: {}\t test_xgb_mean_0322:{}\t test_xgb_gmean_0322:{}'.        format(train_xgb_0322.shape,test_xgb_mean_0322.shape,test_xgb_gmean_0322.shape))
    
print('\ntrain_xgb_0322')
print(train_xgb_0322.iloc[:5,:3])


# In[17]:


# XGB 1st level 0331

file_train      = 'train_blend_xgb_BM_0331_2017-04-02-17-55' + '.csv'
file_test_mean  = 'test_blend_xgb_mean_BM_0331_2017-04-02-17-55' + '.csv'
file_test_gmean = 'test_blend_xgb_gmean_BM_0331_2017-04-02-17-55' + '.csv'

train_xgb_0331      = pd.read_csv(data_path + file_train, header = None)
test_xgb_mean_0331  = pd.read_csv(data_path + file_test_mean, header = None)
test_xgb_gmean_0331 = pd.read_csv(data_path + file_test_gmean, header = None)

n_column = train_xgb_0331.shape[1]
total_col += n_column

train_xgb_0331.columns      = ['xgb_0331_' + x for x in names[:n_column]]
test_xgb_mean_0331.columns  = ['xgb_0331_' + x for x in names[:n_column]]
test_xgb_gmean_0331.columns = ['xgb_0331_' + x for x in names[:n_column]]

print('train_xgb_0331: {}\t test_xgb_mean_0331:{}\t test_xgb_gmean_0331:{}'.        format(train_xgb_0331.shape,test_xgb_mean_0331.shape,test_xgb_gmean_0331.shape))
    
print('\ntrain_xgb_0331')
print(train_xgb_0331.iloc[:5,:3])


# In[18]:


# XGB 1st level 0331 30fold

file_train      = 'train_blend_xgb_BM_0331_30blend_2017-04-04-09-15' + '.csv'
file_test_mean  = 'test_blend_xgb_mean_BM_0331_30blend_2017-04-04-09-15' + '.csv'
file_test_gmean = 'test_blend_xgb_gmean_BM_0331_30blend_2017-04-04-09-15' + '.csv'

train_xgb_0331_30fold      = pd.read_csv(data_path + file_train, header = None)
test_xgb_mean_0331_30fold  = pd.read_csv(data_path + file_test_mean, header = None)
test_xgb_gmean_0331_30fold = pd.read_csv(data_path + file_test_gmean, header = None)

n_column = train_xgb_0331_30fold.shape[1]
total_col += n_column

train_xgb_0331_30fold.columns      = ['xgb_0331_30fold_' + x for x in names[:n_column]]
test_xgb_mean_0331_30fold.columns  = ['xgb_0331_30fold_' + x for x in names[:n_column]]
test_xgb_gmean_0331_30fold.columns = ['xgb_0331_30fold_' + x for x in names[:n_column]]

print('train_xgb_0331_30fold: {}\t test_xgb_mean_0331_30fold:{}\t test_xgb_gmean_0331_30fold:{}'.        format(train_xgb_0331_30fold.shape,test_xgb_mean_0331_30fold.shape,test_xgb_gmean_0331_30fold.shape))
    
print('\ntrain_xgb_0331_30fold')
print(train_xgb_0331_30fold.iloc[:5,:3])


# In[19]:


# XGB 1st level cv137

file_train      = 'train_blend_xgb_cv137_BM_2017-04-06-11-44' + '.csv'
file_test_mean  = 'test_blend_xgb_mean_cv137_5blend_BM_2017-04-06-11-44' + '.csv'
file_test_gmean = 'test_blend_xgb_gmean_cv137_5blend_BM_2017-04-06-11-44' + '.csv'

train_xgb_cv137      = pd.read_csv(data_path + file_train, header = None)
test_xgb_mean_cv137  = pd.read_csv(data_path + file_test_mean, header = None)
test_xgb_gmean_cv137 = pd.read_csv(data_path + file_test_gmean, header = None)

n_column = train_xgb_cv137.shape[1]
total_col += n_column

train_xgb_cv137.columns      = ['xgb_cv137_' + x for x in names[:n_column]]
test_xgb_mean_cv137.columns  = ['xgb_cv137_' + x for x in names[:n_column]]
test_xgb_gmean_cv137.columns = ['xgb_cv137_' + x for x in names[:n_column]]

print('train_xgb_cv137: {}\t test_xgb_mean_cv137:{}\t test_xgb_gmean_cv137:{}'.        format(train_xgb_cv137.shape,test_xgb_mean_cv137.shape,test_xgb_gmean_cv137.shape))
    
print('\ntrain_xgb_cv137')
print(train_xgb_cv137.iloc[:5,:3])


# In[20]:


# XGB 1st level cv137 2

file_train      = 'train_blend_xgb_cv137_BM_2017-04-06-15-28' + '.csv'
file_test_mean  = 'test_blend_xgb_mean_cv137_5blend_BM_2017-04-06-15-28' + '.csv'
file_test_gmean = 'test_blend_xgb_gmean_cv137_5blend_BM_2017-04-06-15-28' + '.csv'

train_xgb_cv137_1      = pd.read_csv(data_path + file_train, header = None)
test_xgb_mean_cv137_1  = pd.read_csv(data_path + file_test_mean, header = None)
test_xgb_gmean_cv137_1 = pd.read_csv(data_path + file_test_gmean, header = None)

n_column = train_xgb_cv137_1.shape[1]
total_col += n_column

train_xgb_cv137_1.columns      = ['xgb_cv137_1_' + x for x in names[:n_column]]
test_xgb_mean_cv137_1.columns  = ['xgb_cv137_1_' + x for x in names[:n_column]]
test_xgb_gmean_cv137_1.columns = ['xgb_cv137_1_' + x for x in names[:n_column]]

print('train_xgb_cv137_1: {}\t test_xgb_mean_cv137_1:{}\t test_xgb_gmean_cv137_1:{}'.        format(train_xgb_cv137_1.shape,test_xgb_mean_cv137_1.shape,test_xgb_gmean_cv137_1.shape))
    
print('\ntrain_xgb_cv137_1')
print(train_xgb_cv137_1.iloc[:5,:3])


# In[21]:


# XGB 1st level cv_price

file_train      = 'train_blend_xgb_cv_price_BM_2017-04-09-14-06' + '.csv'
file_test_mean  = 'test_blend_xgb_mean_cv_price_BM_2017-04-09-14-06' + '.csv'
file_test_gmean = 'test_blend_xgb_gmean_cv_price_BM_2017-04-09-14-06' + '.csv'

train_xgb_cv_price      = pd.read_csv(data_path + file_train, header = None)
test_xgb_mean_cv_price  = pd.read_csv(data_path + file_test_mean, header = None)
test_xgb_gmean_cv_price = pd.read_csv(data_path + file_test_gmean, header = None)

n_column = train_xgb_cv_price.shape[1]
total_col += n_column

train_xgb_cv_price.columns      = ['xgb_cv_price_' + x for x in names[:n_column]]
test_xgb_mean_cv_price.columns  = ['xgb_cv_price_' + x for x in names[:n_column]]
test_xgb_gmean_cv_price.columns = ['xgb_cv_price_' + x for x in names[:n_column]]

print('train_xgb_cv_price: {}\t test_xgb_mean_cv_price:{}\t test_xgb_gmean_cv_price:{}'.        format(train_xgb_cv_price.shape,test_xgb_mean_cv_price.shape,test_xgb_gmean_cv_price.shape))
    
print('\ntrain_xgb_cv_price')
print(train_xgb_cv_price.iloc[:5,:3])


# In[22]:


# XGB 1st level CV_MS_52571

file_train      = 'train_blend_xgb_CV_MS_BM_2017-04-11-09-18' + '.csv'
file_test_mean  = 'test_blend_xgb_mean_CV_MS_BM_2017-04-11-09-18' + '.csv'
file_test_gmean = 'test_blend_xgb_gmean_CV_MS_BM_2017-04-11-09-18' + '.csv'

train_xgb_cv_MS_52571      = pd.read_csv(data_path + file_train, header = None)
test_xgb_mean_cv_MS_52571  = pd.read_csv(data_path + file_test_mean, header = None)
test_xgb_gmean_cv_MS_52571 = pd.read_csv(data_path + file_test_gmean, header = None)

n_column = train_xgb_cv_MS_52571.shape[1]
total_col += n_column

train_xgb_cv_MS_52571.columns      = ['xgb_cv_MS_52571_' + x for x in names[:n_column]]
test_xgb_mean_cv_MS_52571.columns  = ['xgb_cv_MS_52571_' + x for x in names[:n_column]]
test_xgb_gmean_cv_MS_52571.columns = ['xgb_cv_MS_52571_' + x for x in names[:n_column]]

print('train_xgb_cv_MS_52571: {}\t test_xgb_mean_cv_MS_52571:{}\t test_xgb_gmean_cv_MS_52571:{}'.        format(train_xgb_cv_MS_52571.shape,test_xgb_mean_cv_MS_52571.shape,test_xgb_gmean_cv_MS_52571.shape))
    
print('\ntrain_xgb_cv_MS_52571')
print(train_xgb_cv_MS_52571.iloc[:5,:3])


# In[23]:


# XGB 1st level CV_MS_52571 30fold

file_train      = 'train_blend_xgb_CV_MS_30blend_BM_2017-04-12-08-56' + '.csv'
file_test_mean  = 'test_blend_xgb_mean_CV_MS_30blend_BM_2017-04-12-08-56' + '.csv'
file_test_gmean = 'test_blend_xgb_gmean_CV_MS_30blend_BM_2017-04-12-08-56' + '.csv'

train_xgb_cv_MS_52571_30fold      = pd.read_csv(data_path + file_train, header = None)
test_xgb_mean_cv_MS_52571_30fold  = pd.read_csv(data_path + file_test_mean, header = None)
test_xgb_gmean_cv_MS_52571_30fold = pd.read_csv(data_path + file_test_gmean, header = None)

n_column = train_xgb_cv_MS_52571_30fold.shape[1]
total_col += n_column

train_xgb_cv_MS_52571_30fold.columns      = ['xgb_cv_MS_52571_30fold_' + x for x in names[:n_column]]
test_xgb_mean_cv_MS_52571_30fold.columns  = ['xgb_cv_MS_52571_30fold_' + x for x in names[:n_column]]
test_xgb_gmean_cv_MS_52571_30fold.columns = ['xgb_cv_MS_52571_30fold_' + x for x in names[:n_column]]

print('train_xgb_cv_MS_52571_30fold: {}\t test_xgb_mean_cv_MS_52571_30fold:{}\t test_xgb_gmean_cv_MS_52571_30fold:{}'.        format(train_xgb_cv_MS_52571_30fold.shape,test_xgb_mean_cv_MS_52571_30fold.shape,test_xgb_gmean_cv_MS_52571_30fold.shape))
    
print('\ntrain_xgb_cv_MS_52571_30fold')
print(train_xgb_cv_MS_52571_30fold.iloc[:5,:3])


# In[24]:


# XGB one vs rest 1st level 0322

file_train      = 'train_blend_xgb_ovr_BM_0322_2017-03-27-19-36' + '.csv'
file_test_mean  = 'test_blend_xgb_ovr_mean_BM_0322_2017-03-27-19-36' + '.csv'
file_test_gmean = 'test_blend_xgb_ovr_gmean_BM_0322_2017-03-27-19-36' + '.csv'

train_xgb_ovr_0322      = pd.read_csv(data_path + file_train, header = None)
test_xgb_mean_ovr_0322  = pd.read_csv(data_path + file_test_mean, header = None)
test_xgb_gmean_ovr_0322 = pd.read_csv(data_path + file_test_gmean, header = None)

n_column = train_xgb_ovr_0322.shape[1]
total_col += n_column

train_xgb_ovr_0322.columns      = ['xgb_0322_ovr_' + x for x in names[:n_column]]
test_xgb_mean_ovr_0322.columns  = ['xgb_0322_ovr_' + x for x in names[:n_column]]
test_xgb_gmean_ovr_0322.columns = ['xgb_0322_ovr_' + x for x in names[:n_column]]

print('train_xgb_0322: {}\t test_xgb_mean_0322:{}\t test_xgb_gmean_0322:{}'.        format(train_xgb_ovr_0322.shape,test_xgb_mean_ovr_0322.shape,test_xgb_gmean_ovr_0322.shape))
    
print('\ntrain_xgb_ovr_0322')
print(train_xgb_ovr_0322.iloc[:5,:3])


# In[25]:


# LightGBM 1st level 0322

file_train      = 'train_blend_LightGBM_BM_0322_2017-03-27-08-21' + '.csv'
file_test_mean  = 'test_blend_LightGBM_mean_BM_0322_2017-03-27-08-21' + '.csv'
file_test_gmean = 'test_blend_LightGBM_gmean_BM_0322_2017-03-27-08-21' + '.csv'

train_lgb_0322      = pd.read_csv(data_path + file_train, header = None)
test_lgb_mean_0322  = pd.read_csv(data_path + file_test_mean, header = None)
test_lgb_gmean_0322 = pd.read_csv(data_path + file_test_gmean, header = None)

n_column = train_lgb_0322.shape[1]
total_col += n_column

train_lgb_0322.columns      = ['lgb_0322_' + x for x in names[:n_column]]
test_lgb_mean_0322.columns  = ['lgb_0322_' + x for x in names[:n_column]]
test_lgb_gmean_0322.columns = ['lgb_0322_' + x for x in names[:n_column]]

print('train_lgb_0322: {}\t test_lgb_mean_0322:{}\t test_lgb_gmean_0322:{}'.        format(train_lgb_0322.shape,test_lgb_mean_0322.shape,test_lgb_gmean_0322.shape))
    
print('\ntrain_lgb_0322')
print(train_lgb_0322.iloc[:5,:3])


# LightGBM 1st level dart 0322

file_train      = 'train_blend_LightGBM_dart_BM_0322_2017-03-31-13-03' + '.csv'
file_test_mean  = 'test_blend_LightGBM_dart_mean_BM_0322_2017-03-31-13-03' + '.csv'
file_test_gmean = 'test_blend_LightGBM_dart_gmean_BM_0322_2017-03-31-13-03' + '.csv'

train_lgb_dart_0322      = pd.read_csv(data_path + file_train, header = None)
test_lgb_mean_dart_0322  = pd.read_csv(data_path + file_test_mean, header = None)
test_lgb_gmean_dart_0322 = pd.read_csv(data_path + file_test_gmean, header = None)

n_column = train_lgb_dart_0322.shape[1]
total_col += n_column

train_lgb_dart_0322.columns      = ['lgb_dart_0322_' + x for x in names[:n_column]]
test_lgb_mean_dart_0322.columns  = ['lgb_dart_0322_' + x for x in names[:n_column]]
test_lgb_gmean_dart_0322.columns = ['lgb_dart_0322_' + x for x in names[:n_column]]

print('train_lgb_dart_0322: {}\t test_lgb_mean_dart_0322:{}\t test_lgb_gmean_dart_0322:{}'.        format(train_lgb_dart_0322.shape,test_lgb_mean_dart_0322.shape,test_lgb_gmean_dart_0322.shape))
    
print('\ntrain_lgb_dart_0322')
print(train_lgb_dart_0322.iloc[:5,:3])


# In[26]:


# LightGBM 1st level 0331

file_train      = 'train_blend_LightGBM_BM_0331_2017-04-01-07-33' + '.csv'
file_test_mean  = 'test_blend_LightGBM_mean_BM_0331_2017-04-01-07-33' + '.csv'
file_test_gmean = 'test_blend_LightGBM_gmean_BM_0331_2017-04-01-07-33' + '.csv'

train_lgb_0331      = pd.read_csv(data_path + file_train, header = None)
test_lgb_mean_0331  = pd.read_csv(data_path + file_test_mean, header = None)
test_lgb_gmean_0331 = pd.read_csv(data_path + file_test_gmean, header = None)

n_column = train_lgb_0331.shape[1]
total_col += n_column

train_lgb_0331.columns      = ['lgb_0331_' + x for x in names[:n_column]]
test_lgb_mean_0331.columns  = ['lgb_0331_' + x for x in names[:n_column]]
test_lgb_gmean_0331.columns = ['lgb_0331_' + x for x in names[:n_column]]

print('train_lgb_0331: {}\t test_lgb_mean_0331:{}\t test_lgb_gmean_0331:{}'.        format(train_lgb_0331.shape,test_lgb_mean_0331.shape,test_lgb_gmean_0331.shape))
    
print('\ntrain_lgb_0331')
print(train_lgb_0331.iloc[:5,:3])


# LightGBM 1st level 0401

file_train      = 'train_blend_LightGBM_BM_0401_2017-04-02-12-24' + '.csv'
file_test_mean  = 'test_blend_LightGBM_mean_BM_0401_2017-04-02-12-24' + '.csv'
file_test_gmean = 'test_blend_LightGBM_gmean_BM_0401_2017-04-02-12-24' + '.csv'

train_lgb_0401      = pd.read_csv(data_path + file_train, header = None)
test_lgb_mean_0401  = pd.read_csv(data_path + file_test_mean, header = None)
test_lgb_gmean_0401 = pd.read_csv(data_path + file_test_gmean, header = None)

n_column = train_lgb_0401.shape[1]
total_col += n_column

train_lgb_0401.columns      = ['lgb_0401_' + x for x in names[:n_column]]
test_lgb_mean_0401.columns  = ['lgb_0401_' + x for x in names[:n_column]]
test_lgb_gmean_0401.columns = ['lgb_0401_' + x for x in names[:n_column]]

print('train_lgb_0401: {}\t test_lgb_mean_0401:{}\t test_lgb_gmean_0401:{}'.        format(train_lgb_0401.shape,test_lgb_mean_0401.shape,test_lgb_gmean_0401.shape))
    
print('\ntrain_lgb_0401')
print(train_lgb_0401.iloc[:5,:3])


# In[27]:


# Keras 1st level No.1

file_train      = 'train_blend_Keras_BM_0331_2017-04-04-15-32' + '.csv'
file_test_mean  = 'test_blend_Keras_BM_0331_2017-04-04-15-32' + '.csv'


train_nn_0331      = pd.read_csv(data_path + file_train, header = None)
test_nn_mean_0331  = pd.read_csv(data_path + file_test_mean, header = None)


n_column = train_nn_0331.shape[1]
total_col += n_column

train_nn_0331.columns      = ['nn_0331_' + x for x in names[:n_column]]
test_nn_mean_0331.columns  = ['nn_0331_' + x for x in names[:n_column]]


print('train_nn_0331: {}\t test_nn_mean_0331:{}'.        format(train_nn_0331.shape,test_nn_mean_0331.shape))
    
print('\ntrain_nn_0331')
print(train_nn_0331.iloc[:5,:3])


# Keras 1st level No.2

file_train      = 'train_blend_Keras_BM_0331_2017-04-04-17-23' + '.csv'
file_test_mean  = 'test_blend_Keras_BM_0331_2017-04-04-17-23' + '.csv'


train_nn_0331_1      = pd.read_csv(data_path + file_train, header = None)
test_nn_mean_0331_1  = pd.read_csv(data_path + file_test_mean, header = None)


n_column = train_nn_0331_1.shape[1]
total_col += n_column

train_nn_0331_1.columns      = ['nn_0331_1_' + x for x in names[:n_column]]
test_nn_mean_0331_1.columns  = ['nn_0331_1_' + x for x in names[:n_column]]


print('train_nn_0331_1: {}\t test_nn_mean_0331_1:{}'.        format(train_nn_0331_1.shape,test_nn_mean_0331_1.shape))
    
print('\ntrain_nn_0331_1')
print(train_nn_0331_1.iloc[:5,:3])


# In[28]:


# Keras one vs rest 1st level 0331

file_train      = 'train_blend_Keras_ovr_BM_0331_2017-04-05-03-37' + '.csv'
file_test_mean  = 'test_blend_Keras_ovr_BM_0331_2017-04-05-03-37' + '.csv'


train_nn_ovr_0331      = pd.read_csv(data_path + file_train, header = None)
test_nn_mean_ovr_0331  = pd.read_csv(data_path + file_test_mean, header = None)


n_column = train_nn_ovr_0331.shape[1]
total_col += n_column

train_nn_ovr_0331.columns      = ['nn_0331_ovr_' + x for x in names[:n_column]]
test_nn_mean_ovr_0331.columns  = ['nn_0331_ovr_' + x for x in names[:n_column]]


print('train_nn_ovr_0331: {}\t test_nn_mean_ovr_0331:{}'.        format(train_nn_ovr_0331.shape,test_nn_mean_ovr_0331.shape))
    
print('\ntrain_nn_ovr_0331')
print(train_nn_ovr_0331.iloc[:5,:3])


# In[29]:


# Keras bagging 1st level CV_52571

file_train      = 'train_blend_Keras_CV_52571_BM_2017-04-13-13-59' + '.csv'
file_test_mean  = 'test_blend_Keras_mean_CV_52571_BM_2017-04-13-13-59' + '.csv'


train_nn_bagging      = pd.read_csv(data_path + file_train, header = None)
test_nn_mean_bagging  = pd.read_csv(data_path + file_test_mean, header = None)


n_column = train_nn_bagging.shape[1]
total_col += n_column

train_nn_bagging.columns      = ['nn_bagging_' + x for x in names[:n_column]]
test_nn_mean_bagging.columns  = ['nn_bagging_' + x for x in names[:n_column]]


print('train_nn_bagging: {}\t test_nn_mean_bagging:{}'.        format(train_nn_bagging.shape,test_nn_mean_bagging.shape))
    
print('\ntrain_nn_bagging')
print(train_nn_bagging.iloc[:5,:3])


# In[30]:


# Genetic Programming 1st level

file_train      = 'train_blend_GP_BM_2017-04-09-19-15' + '.csv'
file_test_mean  = 'test_blend_GP_BM_2017-04-09-19-15' + '.csv'


train_gp = pd.read_csv(data_path + file_train, header = None)
test_gp  = pd.read_csv(data_path + file_test_mean, header = None)


n_column = train_gp.shape[1]
total_col += n_column

train_gp.columns = ['gp_' + x for x in names[:n_column]]
test_gp.columns  = ['gp_' + x for x in names[:n_column]]


print('train_gp: {}\t test_gp:{}'.        format(train_gp.shape,test_gp.shape))
    
print('\ntrain_gp')
print(train_gp.iloc[:5,:3])


# In[31]:


# xgb bagging 0

file_train      = 'train_blend_XGB_BM_3bagging_CV_MS_52571_2017-04-13-09-33' + '.csv'
file_test_mean  = 'test_blend_XGB_BM_3bagging_CV_MS_52571_2017-04-13-09-33' + '.csv'


train_bagging_0 = pd.read_csv(data_path + file_train, header = None)
test_bagging_0  = pd.read_csv(data_path + file_test_mean, header = None)


n_column = train_bagging_0.shape[1]
total_col += n_column

train_bagging_0.columns = ['bagging_0_' + x for x in names[:n_column]]
test_bagging_0.columns  = ['bagging_0_' + x for x in names[:n_column]]


print('train_bagging_0: {}\t test_bagging_0:{}'.        format(train_bagging_0.shape,test_bagging_0.shape))
    
print('\ntrain_bagging_0')
print(train_bagging_0.iloc[:5,:3])


# xgb bagging 1
file_train      = 'train_blend_xgb_141bagging_BM_2017-04-13-10-12' + '.csv'
file_test_mean  = 'test_blend_xgb_mean_141bagging_BM_2017-04-13-10-12' + '.csv'


train_bagging_1 = pd.read_csv(data_path + file_train, header = None)
test_bagging_1  = pd.read_csv(data_path + file_test_mean, header = None)


n_column = train_bagging_1.shape[1]
total_col += n_column

train_bagging_1.columns = ['bagging_1_' + x for x in names[:n_column]]
test_bagging_1.columns  = ['bagging_1_' + x for x in names[:n_column]]


print('train_bagging_1: {}\t test_bagging_1:{}'.        format(train_bagging_0.shape,test_bagging_0.shape))
    
print('\ntrain_bagging_1')
print(train_bagging_0.iloc[:5,:3])


# In[32]:


print(total_col)


# In[33]:


train_2nd      = pd.concat([train_rfc_gini, train_rfc_entropy, train_rfc_gini_0322, train_rfc_entropy_0322, 
                            train_LR, train_LR_0322, 
                            train_ET_gini, train_ET_entropy, train_ET_gini_0322, train_ET_entropy_0322,
                            train_KNN_uniform, train_KNN_distance, train_KNN_uniform_0322, train_KNN_distance_0322,
                            train_FM,train_FM_0322,
                            train_MNB,
                            train_tsne, train_tsne_0322,
                            train_xgb, train_xgb_0322, train_xgb_0331,train_xgb_0331_30fold, train_xgb_cv137,
                            train_xgb_cv137_1,train_xgb_cv_price,train_xgb_cv_MS_52571,train_xgb_cv_MS_52571_30fold,
                            train_xgb_ovr_0322,
                            train_lgb_0322, train_lgb_dart_0322, train_lgb_0331, train_lgb_0401,
                            train_nn_0331, train_nn_0331_1,
                            train_nn_ovr_0331,train_nn_bagging,
                            train_gp,
                            train_bagging_0,train_bagging_1
                           ], axis = 1)

test_2nd_mean  = pd.concat([test_rfc_gini_mean,test_rfc_entropy_mean, test_rfc_gini_mean_0322,test_rfc_entropy_mean_0322, 
                            test_LR_mean, test_LR_mean_0322, 
                            test_ET_gini_mean, test_ET_entropy_mean,test_ET_gini_mean_0322, test_ET_entropy_mean_0322,
                            test_KNN_uniform_mean, test_KNN_distance_mean, test_KNN_uniform_mean_0322, test_KNN_distance_mean_0322, 
                            test_FM_mean,test_FM_mean_0322,
                            test_MNB_mean,
                            test_tsne, test_tsne_0322,
                            test_xgb_mean, test_xgb_mean_0322, test_xgb_mean_0331,test_xgb_mean_0331_30fold, test_xgb_mean_cv137,
                            test_xgb_mean_cv137_1,test_xgb_mean_cv_price,test_xgb_mean_cv_MS_52571,test_xgb_mean_cv_MS_52571_30fold,
                            test_xgb_mean_ovr_0322,
                            test_lgb_mean_0322, test_lgb_mean_dart_0322, test_lgb_mean_0331, test_lgb_mean_0401,
                            test_nn_mean_0331, test_nn_mean_0331_1,
                            test_nn_mean_ovr_0331,test_nn_mean_bagging,
                            test_gp,
                            test_bagging_0,test_bagging_1
                           ], axis = 1)

test_2nd_gmean = pd.concat([test_rfc_gini_gmean,test_rfc_entropy_gmean, test_rfc_gini_gmean_0322,test_rfc_entropy_gmean_0322, 
                            test_LR_gmean, test_LR_gmean_0322, 
                            test_ET_gini_gmean, test_ET_entropy_gmean,test_ET_gini_gmean_0322, test_ET_entropy_gmean_0322,
                            test_KNN_uniform_gmean, test_KNN_distance_gmean,test_KNN_uniform_gmean_0322, test_KNN_distance_gmean_0322,
                            test_FM_gmean,test_FM_gmean_0322,
                            test_MNB_gmean,
                            test_tsne, test_tsne_0322,
                            test_xgb_gmean, test_xgb_gmean_0322, test_xgb_gmean_0331,test_xgb_gmean_0331_30fold,test_xgb_gmean_cv137,
                            test_xgb_gmean_cv137_1,test_xgb_gmean_cv_price,test_xgb_gmean_cv_MS_52571,test_xgb_gmean_cv_MS_52571_30fold,
                            test_xgb_gmean_ovr_0322,
                            test_lgb_gmean_0322, test_lgb_gmean_dart_0322, test_lgb_gmean_0331, test_lgb_gmean_0401,
                            test_nn_mean_0331, test_nn_mean_0331_1,
                            test_nn_mean_ovr_0331,test_nn_mean_bagging,
                            test_gp,
                            test_bagging_0,test_bagging_1
                           ], axis = 1)



print('train_2nd: {}\t test_2nd_mean:{}\t test_2nd_gmean:{}'.            format(train_2nd.shape,test_2nd_mean.shape,test_2nd_gmean.shape))


# In[34]:


data_path = "../input/"
train_X = pd.read_csv(data_path + 'train_BM_0331.csv')
test_X = pd.read_csv(data_path + 'test_BM_0331.csv')

train_y = np.ravel(pd.read_csv('../input/' + 'labels_BrandenMurray.csv'))
train_y = to_categorical(train_y)

ntrain = train_X.shape[0]
sub_id = test_X.listing_id.astype('int32').values
# all_features = features_to_use + desc_sparse_cols + feat_sparse_cols
print(train_X.shape, test_X.shape, train_y.shape)


# In[35]:


null_ind = test_X.num_loc_price_diff.isnull()
test_X['num_loc_price_diff'] = test_X['num_price'] - test_X['num_loc_median_price']


# In[36]:


train_X = pd.concat([train_X,train_2nd],axis=1)
test_X = pd.concat([test_X,test_2nd_mean],axis=1)

print(train_X.shape)
print(test_X.shape)


# In[37]:


full_data = pd.concat([train_X,test_X])
print(full_data.shape)


# In[38]:


full_data.isnull().values.any()


# In[39]:


full_data.columns.values


# In[40]:


feat_to_use = ['building_id_mean_med', 'building_id_mean_high',
       'manager_id_mean_med', 'manager_id_mean_high', 'median_price_bed',
       'ratio_bed', 'compound', 'neg', 'neu', 'pos', 
#                'street', 'avenue',
#        'east', 'west', 'north', 'south', 'other_address', 'top_10_manager',
#        'top_25_manager', 'top_5_manager', 'top_50_manager',
#        'top_1_manager', 'top_2_manager', 'top_15_manager',
#        'top_20_manager', 'top_30_manager', 'Zero_building_id',
#        'top_10_building', 'top_25_building', 'top_5_building',
#        'top_50_building', 'top_1_building', 'top_2_building',
#        'top_15_building', 'top_20_building', 'top_30_building',
       'listing_id', 'num_latitude', 'num_longitude',
       'num_dist_from_center', 'num_OutlierAggregated', 'num_pos_density',
       'num_building_null', 'num_fbuilding', 'num_fmanager',
       'num_created_weekday', 'num_created_weekofyear', 'num_created_day',
       'num_created_month', 'num_created_hour', 'num_bathrooms',
       'num_bedrooms', 'num_price', 'num_price_q', 'num_priceXroom',
       'num_even_bathrooms', 'num_features', 'num_photos',
       'num_desc_length', 'num_desc_length_null',
#                'num_location_6_3',
#        'num_location_6_1', 'num_location_6_0', 'num_location_6_5',
#        'num_location_6_4', 'num_location_6_2', 'num_location_40_18',
#        'num_location_40_31', 'num_location_40_11', 'num_location_40_24',
#        'num_location_40_14', 'num_location_40_36', 'num_location_40_3',
#        'num_location_40_7', 'num_location_40_33', 'num_location_40_5',
#        'num_location_40_37', 'num_location_40_12', 'num_location_40_16',
#        'num_location_40_2', 'num_location_40_20', 'num_location_40_34',
#        'num_location_40_9', 'num_location_40_0', 'num_location_40_21',
#        'num_location_40_26', 'num_location_40_13', 'num_location_40_25',
#        'num_location_40_32', 'num_location_40_19', 'num_location_40_17',
#        'num_location_40_4', 'num_location_40_15', 'num_location_40_35',
#        'num_location_40_22', 'num_location_40_30', 'num_location_40_1',
#        'num_location_40_23', 'num_location_40_10', 'num_location_40_38',
#        'num_location_40_28', 'num_location_40_6', 'num_location_40_29',
#        'num_location_40_27', 'num_location_40_39', 'num_location_40_8',
#        'num_room_type_0', 'num_room_type_1', 'num_room_type_2',
#        'num_room_type_3', 'num_room_type_4', 'num_room_type_5',
#        'num_room_type_6', 'num_room_type_7', 'num_room_type_8',
#        'num_room_type_9', 'num_room_type_10', 'num_room_type_11',
#        'num_room_type_12', 'num_room_type_13', 'num_room_type_14',
#        'num_room_type_15', 'num_room_type_16', 'num_room_type_17',
#        'num_room_type_18', 'num_room_type_19', 
               'num_6_median_price',
       'num_6_price_ratio', 'num_6_price_diff', 'num_loc_median_price',
       'num_loc_price_ratio', 'num_loc_price_diff', 'num_loc_ratio',
       'num_loc_diff', 'hcc_pos_pred_1', 'hcc_pos_pred_2', 'building_id',
       'display_address', 'manager_id', 'street_address',
       'num_pricePerBed', 'num_pricePerBath', 'num_pricePerRoom',
       'num_bedPerBath', 'num_bedBathDiff', 'num_bedBathSum',
       'num_bedsPerc', 
#                'hcc_building_id_pred_1', 'hcc_building_id_pred_2',
#        'hcc_manager_id_pred_1', 'hcc_manager_id_pred_2',
               
#        'feature_1_month_free', 'feature_24/7_concierge',
#        'feature_24/7_doorman', 'feature_24/7_doorman_concierge',
#        'feature_actual_apt._photos', 'feature_air_conditioning',
#        'feature_all_pets_ok', 'feature_all_utilities_included',
#        'feature_assigned-parking-space', 'feature_attended_lobby',
#        'feature_backyard', 'feature_balcony', 'feature_basement_storage',
#        'feature_basketball_court', 'feature_bike_room',
#        'feature_bike_storage', 'feature_billiards_room',
#        'feature_billiards_table_and_wet_bar', 'feature_brand_new',
#        'feature_breakfast_bar', 'feature_bright', 'feature_brownstone',
#        'feature_building-common-outdoor-space', 'feature_business_center',
#        'feature_cable/satellite_tv', 'feature_cable_ready',
#        'feature_call/text_abraham_caro_@_917-373-0862',
#        'feature_cats_allowed', 'feature_central_a/c', 'feature_central_ac',
#        'feature_central_air', 'feature_chefs_kitchen',
#        "feature_children's_playroom", 'feature_childrens_playroom',
#        'feature_cinema_room', 'feature_city_view',
#        'feature_close_to_subway', 'feature_closets_galore!',
#        'feature_club_sun_deck_has_spectacular_city_and_river_views',
#        'feature_cold_storage', 'feature_common_backyard',
#        'feature_common_garden', 'feature_common_outdoor_space',
#        'feature_common_parking/garage', 'feature_common_roof_deck',
#        'feature_common_storage', 'feature_common_terrace',
#        'feature_community_recreation_facilities',
#        'feature_complimentary_sunday_brunch', 'feature_concierge',
#        'feature_concierge_service', 'feature_condo_finishes',
#        'feature_courtyard', 'feature_crown_moldings', 'feature_deck',
#        'feature_deco_brick_wall', 'feature_decorative_fireplace',
#        'feature_dining_room', 'feature_dishwasher', 'feature_dogs_allowed',
#        'feature_doorman', 'feature_dry_cleaning_service',
#        'feature_dryer_in_unit', 'feature_duplex', 'feature_duplex_lounge',
#        'feature_eat-in_kitchen', 'feature_eat_in_kitchen',
#        'feature_elegant_glass-enclosed_private_lounge_with_magnificent_river_views',
#        'feature_elevator', 'feature_exclusive',
#        'feature_exercise/yoga_studio', 'feature_exposed_brick',
#        'feature_extra_room', 'feature_fireplace', 'feature_fireplaces',
#        'feature_fitness_center', 'feature_fitness_room', 'feature_flex-2',
#        'feature_flex-3', 'feature_free_wifi_in_club_lounge',
#        'feature_ft_doorman', 'feature_full-time_doorman',
#        'feature_full_service_garage',
#        'feature_fully-equipped_club_fitness_center',
#        'feature_fully__equipped', 'feature_furnished', 'feature_game_room',
#        'feature_garage', 'feature_garbage_disposal', 'feature_garden',
#        'feature_garden/patio', 'feature_granite_countertops',
#        'feature_granite_kitchen', 'feature_green_building',
#        'feature_guarantors_accepted', 'feature_gut_renovated',
#        'feature_gym', 'feature_gym/fitness', 'feature_gym_in_building',
#        'feature_hardwood', 'feature_hardwood_floors',
#        'feature_health_club', 'feature_hi_rise',
#        'feature_high-speed_internet', 'feature_high_ceiling',
#        'feature_high_ceilings', 'feature_high_speed_internet',
#        'feature_highrise', 'feature_housekeeping_service',
#        'feature_in-unit_washer/dryer', 'feature_indoor_pool',
#        'feature_intercom', 'feature_jacuzzi', 'feature_large_living_room',
#        'feature_laundry', 'feature_laundry_&_housekeeping',
#        'feature_laundry_in_building', 'feature_laundry_in_unit',
#        'feature_laundry_on_every_floor', 'feature_laundry_on_floor',
#        'feature_laundry_room', 'feature_light', 'feature_live-in_super',
#        'feature_live-in_superintendent', 'feature_live/work',
#        'feature_live_in_super', 'feature_loft', 'feature_lounge',
#        'feature_lounge_room', 'feature_lowrise', 'feature_luxury_building',
#        'feature_magnificent_venetian-style', 'feature_mail_room',
#        'feature_marble_bath', 'feature_marble_bathroom',
#        'feature_media_room', 'feature_media_screening_room',
#        'feature_microwave', 'feature_midrise', 'feature_multi-level',
#        'feature_new_construction', 'feature_newly_renovated',
#        'feature_no_fee', 'feature_no_pets', 'feature_on-site_atm_machine',
#        'feature_on-site_attended_garage', 'feature_on-site_garage',
#        'feature_on-site_laundry', 'feature_on-site_parking',
#        'feature_on-site_parking_available', 'feature_on-site_parking_lot',
#        'feature_on-site_super', 'feature_one_month_free',
#        'feature_outdoor_areas', 'feature_outdoor_entertainment_space',
#        'feature_outdoor_pool',
#        'feature_outdoor_roof_deck_overlooking_new_york_harbor_and_battery_park',
#        'feature_outdoor_space', 'feature_package_room', 'feature_parking',
#        'feature_parking_available', 'feature_parking_space',
#        'feature_part-time_doorman', 'feature_party_room', 'feature_patio',
#        'feature_penthouse', 'feature_pet_friendly', 'feature_pets',
#        'feature_pets_allowed', 'feature_pets_on_approval',
#        'feature_playroom', 'feature_playroom/nursery', 'feature_pool',
#        'feature_post-war', 'feature_post_war', 'feature_pre-war',
#        'feature_pre_war', 'feature_prewar', 'feature_private-balcony',
#        'feature_private-outdoor-space', 'feature_private_backyard',
#        'feature_private_balcony', 'feature_private_deck',
#        'feature_private_garden',
#        'feature_private_laundry_room_on_every_floor',
#        'feature_private_outdoor_space', 'feature_private_parking',
#        'feature_private_roof_deck', 'feature_private_roofdeck',
#        'feature_private_terrace', 'feature_publicoutdoor',
#        'feature_queen_size_bedrooms', 'feature_queen_sized_rooms',
#        'feature_reduced_fee', 'feature_renovated',
#        'feature_renovated_kitchen', 'feature_residents_garden',
#        'feature_residents_lounge', 'feature_roof-deck',
#        'feature_roof_access', 'feature_roof_deck',
#        'feature_roof_deck_with_grills', 'feature_roofdeck',
#        'feature_rooftop_deck', 'feature_rooftop_terrace',
#        'feature_s/s_appliances', 'feature_sauna', 'feature_screening_room',
#        'feature_separate_kitchen', 'feature_shared_backyard',
#        'feature_shared_garden', 'feature_shares_ok',
#        'feature_short_term_allowed', 'feature_simplex', 'feature_skylight',
#        'feature_skylight_atrium', 'feature_southern_exposure',
#        'feature_spa_services', 'feature_ss_appliances',
#        'feature_stainless_steel', 'feature_stainless_steel_appliances',
#        'feature_state-of-the-art_fitness_center', 'feature_storage',
#        'feature_storage_available', 'feature_storage_facilities_available',
#        'feature_storage_room', 'feature_sublet', 'feature_subway',
#        'feature_sundeck', 'feature_swimming_pool', 'feature_tenant_lounge',
#        'feature_terrace', 'feature_terraces_/_balconies',
#        'feature_tons_of_natural_light', 'feature_valet',
#        'feature_valet_parking', 'feature_valet_service',
#        'feature_valet_services',
#        'feature_valet_services_including_dry_cleaning',
#        'feature_video_intercom', 'feature_view', 'feature_virtual_doorman',
#        'feature_virtual_tour', 'feature_walk-in_closet', 'feature_walk-up',
#        'feature_walk_in_closet', 'feature_walk_in_closet(s)',
#        'feature_washer/dryer', 'feature_washer/dryer_hookup',
#        'feature_washer/dryer_in-unit', 'feature_washer/dryer_in_building',
#        'feature_washer/dryer_in_unit', 'feature_washer_&_dryer',
#        'feature_washer_in_unit', 'feature_wheelchair_access',
#        'feature_wheelchair_ramp', 'feature_wifi', 'feature_wifi_access',
#        'feature_wood-burning_fireplace', 'feature_yard',
#        'feature_yoga_classes',
                'rfc_gini_low_0', 'rfc_gini_medium_0',
       'rfc_gini_high_0', 'rfc_entropy_low_0', 'rfc_entropy_medium_0',
       'rfc_entropy_high_0', 'rfc_gini_0322_low_0',
       'rfc_gini_0322_medium_0', 'rfc_gini_0322_high_0',
       'rfc_entropy_0322_low_0', 'rfc_entropy_0322_medium_0',
       'rfc_entropy_0322_high_0', 'LR_low_0', 'LR_medium_0', 'LR_high_0',
       'LR_low_1', 'LR_medium_1', 'LR_high_1', 'LR_low_2', 'LR_medium_2',
       'LR_high_2', 'LR_low_3', 'LR_medium_3', 'LR_high_3', 'LR_low_4',
       'LR_medium_4', 'LR_high_4', 'LR_low_5', 'LR_medium_5', 'LR_high_5',
       'LR_low_6', 'LR_medium_6', 'LR_high_6', 'LR_0322_low_0',
       'LR_0322_medium_0', 'LR_0322_high_0', 'LR_0322_low_1',
       'LR_0322_medium_1', 'LR_0322_high_1', 'LR_0322_low_2',
       'LR_0322_medium_2', 'LR_0322_high_2', 'LR_0322_low_3',
       'LR_0322_medium_3', 'LR_0322_high_3', 'LR_0322_low_4',
       'LR_0322_medium_4', 'LR_0322_high_4', 'LR_0322_low_5',
       'LR_0322_medium_5', 'LR_0322_high_5', 'LR_0322_low_6',
       'LR_0322_medium_6', 'LR_0322_high_6', 'ET_gini_low_0',
       'ET_gini_medium_0', 'ET_gini_high_0', 'ET_entropy_low_0',
       'ET_entropy_medium_0', 'ET_entropy_high_0', 'ET_gini_0322_low_0',
       'ET_gini_0322_medium_0', 'ET_gini_0322_high_0',
       'ET_entropy_0322_low_0', 'ET_entropy_0322_medium_0',
       'ET_entropy_0322_high_0', 'KNN_uniform_low_0',
       'KNN_uniform_medium_0', 'KNN_uniform_high_0', 'KNN_distance_low_0',
       'KNN_distance_medium_0', 'KNN_distance_high_0',
       'KNN_uniform_0322_low_0', 'KNN_uniform_0322_medium_0',
       'KNN_uniform_0322_high_0', 'KNN_uniform_0322_low_1',
       'KNN_uniform_0322_medium_1', 'KNN_uniform_0322_high_1',
       'KNN_uniform_0322_low_2', 'KNN_uniform_0322_medium_2',
       'KNN_uniform_0322_high_2', 'KNN_uniform_0322_low_3',
       'KNN_uniform_0322_medium_3', 'KNN_uniform_0322_high_3',
       'KNN_uniform_0322_low_4', 'KNN_uniform_0322_medium_4',
       'KNN_uniform_0322_high_4', 'KNN_distance_0322_low_0',
       'KNN_distance_0322_medium_0', 'KNN_distance_0322_high_0',
       'KNN_distance_0322_low_1', 'KNN_distance_0322_medium_1',
       'KNN_distance_0322_high_1', 'KNN_distance_0322_low_2',
       'KNN_distance_0322_medium_2', 'KNN_distance_0322_high_2',
       'KNN_distance_0322_low_3', 'KNN_distance_0322_medium_3',
       'KNN_distance_0322_high_3', 'KNN_distance_0322_low_4',
       'KNN_distance_0322_medium_4', 'KNN_distance_0322_high_4',
       'FM_low_0', 'FM_medium_0', 'FM_high_0', 'FM_0322_low_0',
       'FM_0322_medium_0', 'FM_0322_high_0', 'MNB_low_0', 'MNB_medium_0',
       'MNB_high_0', 'MNB_low_1', 'MNB_medium_1', 'MNB_high_1',
       'MNB_low_2', 'MNB_medium_2', 'MNB_high_2', 'tsne_0', 'tsne_1',
       'tsne_2', 'tsne_0_0322', 'tsne_1_0322', 'tsne_2_0322', 'xgb_low_0',
       'xgb_medium_0', 'xgb_high_0', 'xgb_low_1', 'xgb_medium_1',
       'xgb_high_1', 'xgb_low_2', 'xgb_medium_2', 'xgb_high_2',
       'xgb_low_3', 'xgb_medium_3', 'xgb_high_3', 'xgb_low_4',
       'xgb_medium_4', 'xgb_high_4', 'xgb_0322_low_0', 'xgb_0322_medium_0',
       'xgb_0322_high_0', 'xgb_0322_low_1', 'xgb_0322_medium_1',
       'xgb_0322_high_1', 'xgb_0322_low_2', 'xgb_0322_medium_2',
       'xgb_0322_high_2', 'xgb_0322_low_3', 'xgb_0322_medium_3',
       'xgb_0322_high_3', 'xgb_0322_low_4', 'xgb_0322_medium_4',
       'xgb_0322_high_4', 'xgb_0331_low_0', 'xgb_0331_medium_0',
       'xgb_0331_high_0', 'xgb_0331_low_1', 'xgb_0331_medium_1',
       'xgb_0331_high_1', 'xgb_0331_low_2', 'xgb_0331_medium_2',
       'xgb_0331_high_2', 'xgb_0331_low_3', 'xgb_0331_medium_3',
       'xgb_0331_high_3', 'xgb_0331_low_4', 'xgb_0331_medium_4',
       'xgb_0331_high_4', 'xgb_0331_30fold_low_0',
       'xgb_0331_30fold_medium_0', 'xgb_0331_30fold_high_0',
       'xgb_cv137_low_0', 'xgb_cv137_medium_0', 'xgb_cv137_high_0',
       'xgb_cv137_1_low_0', 'xgb_cv137_1_medium_0', 'xgb_cv137_1_high_0',
       'xgb_cv_price_low_0', 'xgb_cv_price_medium_0',
       'xgb_cv_price_high_0', 'xgb_cv_MS_52571_low_0',
       'xgb_cv_MS_52571_medium_0', 'xgb_cv_MS_52571_high_0',
       'xgb_cv_MS_52571_low_1', 'xgb_cv_MS_52571_medium_1',
       'xgb_cv_MS_52571_high_1', 'xgb_cv_MS_52571_low_2',
       'xgb_cv_MS_52571_medium_2', 'xgb_cv_MS_52571_high_2',
       'xgb_cv_MS_52571_low_3', 'xgb_cv_MS_52571_medium_3',
       'xgb_cv_MS_52571_high_3', 'xgb_cv_MS_52571_low_4',
       'xgb_cv_MS_52571_medium_4', 'xgb_cv_MS_52571_high_4',
       'xgb_cv_MS_52571_30fold_low_0', 'xgb_cv_MS_52571_30fold_medium_0',
       'xgb_cv_MS_52571_30fold_high_0', 'xgb_0322_ovr_low_0',
       'xgb_0322_ovr_medium_0', 'xgb_0322_ovr_high_0', 'lgb_0322_low_0',
       'lgb_0322_medium_0', 'lgb_0322_high_0', 'lgb_0322_low_1',
       'lgb_0322_medium_1', 'lgb_0322_high_1', 'lgb_0322_low_2',
       'lgb_0322_medium_2', 'lgb_0322_high_2', 'lgb_0322_low_3',
       'lgb_0322_medium_3', 'lgb_0322_high_3', 'lgb_0322_low_4',
       'lgb_0322_medium_4', 'lgb_0322_high_4', 'lgb_dart_0322_low_0',
       'lgb_dart_0322_medium_0', 'lgb_dart_0322_high_0',
       'lgb_dart_0322_low_1', 'lgb_dart_0322_medium_1',
       'lgb_dart_0322_high_1', 'lgb_dart_0322_low_2',
       'lgb_dart_0322_medium_2', 'lgb_dart_0322_high_2',
       'lgb_dart_0322_low_3', 'lgb_dart_0322_medium_3',
       'lgb_dart_0322_high_3', 'lgb_dart_0322_low_4',
       'lgb_dart_0322_medium_4', 'lgb_dart_0322_high_4', 'lgb_0331_low_0',
       'lgb_0331_medium_0', 'lgb_0331_high_0', 'lgb_0331_low_1',
       'lgb_0331_medium_1', 'lgb_0331_high_1', 'lgb_0331_low_2',
       'lgb_0331_medium_2', 'lgb_0331_high_2', 'lgb_0331_low_3',
       'lgb_0331_medium_3', 'lgb_0331_high_3', 'lgb_0331_low_4',
       'lgb_0331_medium_4', 'lgb_0331_high_4', 'lgb_0401_low_0',
       'lgb_0401_medium_0', 'lgb_0401_high_0', 'lgb_0401_low_1',
       'lgb_0401_medium_1', 'lgb_0401_high_1', 'lgb_0401_low_2',
       'lgb_0401_medium_2', 'lgb_0401_high_2', 'lgb_0401_low_3',
       'lgb_0401_medium_3', 'lgb_0401_high_3', 'lgb_0401_low_4',
       'lgb_0401_medium_4', 'lgb_0401_high_4', 'nn_0331_low_0',
       'nn_0331_medium_0', 'nn_0331_high_0', 'nn_0331_low_1',
       'nn_0331_medium_1', 'nn_0331_high_1', 'nn_0331_1_low_0',
       'nn_0331_1_medium_0', 'nn_0331_1_high_0', 'nn_0331_1_low_1',
       'nn_0331_1_medium_1', 'nn_0331_1_high_1', 'nn_0331_ovr_low_0',
       'nn_0331_ovr_medium_0', 'nn_0331_ovr_high_0', 'nn_0331_ovr_low_1',
       'nn_0331_ovr_medium_1', 'nn_0331_ovr_high_1', 'nn_bagging_low_0',
       'nn_bagging_medium_0', 'nn_bagging_high_0', 'nn_bagging_low_1',
       'nn_bagging_medium_1', 'nn_bagging_high_1', 'gp_low_0',
       'gp_medium_0', 'gp_high_0', 'bagging_0_low_0', 'bagging_0_medium_0',
       'bagging_0_high_0', 'bagging_1_low_0', 'bagging_1_medium_0',
       'bagging_1_high_0'
               ]


# In[41]:


for col in feat_to_use:
    full_data.loc[:,col] = preprocessing.StandardScaler().fit_transform(full_data[col].values.reshape(-1,1))
train_df_nn = full_data[:ntrain]
test_df_nn = full_data[ntrain:]

train_df_nn = sparse.csr_matrix(train_df_nn)
test_df_nn = sparse.csr_matrix(test_df_nn)


print(train_df_nn.shape)
print(test_df_nn.shape)


# In[ ]:





# In[33]:





# In[38]:





# In[45]:


X_train, X_val, y_train, y_val = train_test_split(train_df_nn, train_y, train_size=.80, random_state=3)


# In[42]:


def batch_generator(X, y, batch_size, shuffle):
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0


# In[48]:


early_stop = EarlyStopping(monitor='val_loss', # custom metric
                           patience=5, #early stopping for epoch
                           verbose=0)
checkpointer = ModelCheckpoint(filepath="weights.hdf5", 
                               monitor='val_loss', 
                               verbose=0, save_best_only=True)

def create_model(input_dim):
    model = Sequential()
    init = 'glorot_uniform'
    
    
    model.add(Dense(200, # number of input units: needs to be tuned
                    input_dim = input_dim, # fixed length: number of columns of X
                    init=init,
                   ))
    model.add(Activation('sigmoid'))
    model.add(PReLU()) # activation function
    model.add(BatchNormalization()) # normalization
    model.add(Dropout(0.4)) #dropout rate. needs to be tuned
        
    model.add(Dense(20,init=init)) # number of hidden1 units. needs to be tuned.
    model.add(Activation('sigmoid'))
    model.add(PReLU())
    model.add(BatchNormalization())    
    model.add(Dropout(0.4)) #dropout rate. needs to be tuned
    
#     model.add(Dense(20,init=init)) # number of hidden2 units. needs to be tuned.
#     model.add(Activation('sigmoid'))
#     model.add(PReLU())
#     model.add(BatchNormalization())    
#     model.add(Dropout(0.4)) #dropout rate. needs to be tuned
    
    model.add(Dense(3,
                   init = init,
                   activation = 'softmax')) # 1 for regression 
    model.compile(loss = 'categorical_crossentropy',
#                   metrics=[mae_log],
                  optimizer = 'Adamax' # optimizer. you may want to try different ones
                 )
    return(model)



model = create_model(X_train.shape[1])
fit= model.fit_generator(generator=batch_generator(X_train, y_train, 256, True),
                         nb_epoch=1000,
                         samples_per_epoch=ntrain,
                         validation_data=(X_val.todense(), y_val),
                         callbacks=[early_stop,checkpointer]
                         )

print(min(fit.history['val_loss']))


# In[ ]:


# 200 0.4 20 0.4 0.497530702701   random = 1234
# 200 0.4 20 0.4 0.4926247189   random = 2017
# 200 0.4 20 0.4 0.490076879584   random = 3


# In[ ]:


# 200 0.4 20 0.4 0.497530702701
# 200 0.4 20 0.5 0.497915365548
# 200 0.4 20 0.3 0.497819799823
# 200 0.4 20 0.2 0.500961469331


# In[ ]:


# 200 0.4 100 0.4 0.499785665459
# 200 0.4 80 0.4 0.498103796427
# 200 0.4 60 0.4 0.49918418799
# 200 0.4 40 0.4 0.497752921884
# 200 0.4 30 0.4 0.497920578542
# 200 0.4 20 0.4 0.497530702701
# 200 0.4 15 0.4 0.499171547144
# 


# In[ ]:


# 200 0.5 0.498075507469
# 200 0.3 0.500961016786
# 200 0.2 0.49741881171
# 200 0.1 0.500332858386


# 100 0.4 0.49807900107
# 200 0.4 0.497540953517
# 500 0.4 0.498385540367
# 400 0.4 0.498565992195
# 300 0.4 0.499884162814


# In[ ]:


# 200 0.4 60 0.4 'glorot_uniform' 'Adamax' PReLU 0.498146744116
# 200 0.4 60 0.4 'glorot_uniform' 'Adamax' LeakyReLU 0.49935725494
# 200 0.4 60 0.4 'glorot_normal' 'Adamax' LeakyReLU 0.499323913487
# 200 0.4 60 0.4 'glorot_uniform' 'Adamax' ELU 0.50005558814


# In[ ]:


# 200 0.4 60 0.4 'glorot_uniform' 'Adamax' 0.498146744116
# 200 0.4 60 0.4 'glorot_uniform' 'Adam' 0.499969745217
# 200 0.4 60 0.4 'glorot_uniform' 'RMSprop' 0.501694146019
# 200 0.4 60 0.4 'glorot_uniform' 'Nadam' 0.5026388399


# In[ ]:


# 200 0.4 60 0.4 'glorot_uniform' 'Adamax' 0.498146744116
# 200 0.4 60 0.4 'glorot_normal' 'Adamax' 0.498454758799
# 200 0.4 60 0.4 'lecun_uniform' 'Adamax' 0.498964471961
# 200 0.4 60 0.4 'glorot_normal' 'Adam' 0.499057393308
# 200 0.4 60 0.4 'he_normal' 'Adamax' 0.49938832341
# 200 0.4 60 0.4 'glorot_uniform' 'Adam' 0.499969745217
# 200 0.4 60 0.4 'he_uniform' 'Adamax' 0.502686485616


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[45]:





# In[46]:


model.load_weights("weights.hdf5")

model.compile(loss = 'categorical_crossentropy',optimizer = 'adam' )


# In[55]:


pred_y = model.predict_proba(x=test_df_nn.toarray(),batch_size = 128,verbose=0)


# In[56]:


pred_y


# In[57]:


now = datetime.now()
sub_name = '../output/sub_Keras_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'

out_df = pd.DataFrame(pred_y)
out_df.columns = ["low", "medium","high"]
out_df["listing_id"] = sub_id
out_df.to_csv(sub_name, index=False)


# In[ ]:





# In[ ]:





# In[53]:


def nn_model(params):
    model = Sequential()
    init = 'glorot_uniform'
    
    model.add(Dense(params['input_size'], # number of input units: needs to be tuned
                    input_dim = params['input_dim'], # fixed length: number of columns of X
                    init=init,
                   ))
    model.add(Activation('sigmoid'))
    model.add(PReLU()) # activation function
    model.add(BatchNormalization()) # normalization
    model.add(Dropout(params['input_drop_out'])) #dropout rate. needs to be tuned
        
    model.add(Dense(params['hidden_size'],
                    init=init)) # number of hidden1 units. needs to be tuned.
    model.add(Activation('sigmoid'))
    model.add(PReLU())
    model.add(BatchNormalization())    
    model.add(Dropout(params['hidden_drop_out'])) #dropout rate. needs to be tuned
    
#     model.add(Dense(20,init=init)) # number of hidden2 units. needs to be tuned.
#     model.add(Activation('sigmoid'))
#     model.add(PReLU())
#     model.add(BatchNormalization())    
#     model.add(Dropout(0.5)) #dropout rate. needs to be tuned
    
    model.add(Dense(3,
                    init = init,
                    activation = 'softmax')) # 1 for regression 
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = 'Adamax' # optimizer. you may want to try different ones
                 )
    return(model)



def nn_blend_data(parameters, train_x, train_y, test_x, fold, early_stopping_rounds=0, batch_size=128,randomseed = 1234):
    N_params = len(parameters)
#     print ("Blend %d estimators for %d folds" % (len(parameters), fold))
    skf = KFold(n_splits=fold, shuffle=True, random_state=randomseed)
    N_class = train_y.shape[1]
    
    train_blend_x = np.zeros((train_x.shape[0], N_class*N_params))
    test_blend_x = np.zeros((test_x.shape[0], N_class*N_params))
    scores = np.zeros ((fold,N_params))
    best_rounds = np.zeros ((fold, N_params))
    fold_start = time.time() 

    
    for j, nn_params in enumerate(parameters):
#         print ("Model %d: %s" %(j+1, nn_params))
        test_blend_x_j = np.zeros((test_x.shape[0], N_class*fold))
        
        for i, (train_index, val_index) in enumerate(skf.split(train_x)):
#             print ("Model %d fold %d" %(j+1,i+1))
            train_x_fold = train_x[train_index]
            train_y_fold = train_y[train_index]
            val_x_fold = train_x[val_index]
            val_y_fold = train_y[val_index]
            

            model = nn_model(nn_params)
#             print (model)
            fit= model.fit_generator(generator=batch_generator(train_x_fold, train_y_fold, batch_size, True),
                                     nb_epoch=60,
                                     samples_per_epoch=train_x_fold.shape[0],
                                     validation_data=(val_x_fold.todense(), val_y_fold),
                                     verbose = 0,
                                     callbacks=[ModelCheckpoint(filepath="weights.hdf5", 
                                                                monitor='val_loss', 
                                                                verbose=0, save_best_only=True)]
                                    )

            best_round=len(fit.epoch)-early_stopping_rounds-1
            best_rounds[i,j]=best_round
#             print ("best round %d" % (best_round))
            
            model.load_weights("weights.hdf5")
            # Compile model (required to make predictions)
            model.compile(loss = 'categorical_crossentropy',optimizer = 'Adamax' )
            
            # print (mean_absolute_error(np.exp(y_val)-200, pred_y))
            val_y_predict_fold = model.predict_proba(x=val_x_fold.toarray(),verbose=0)
            score = log_loss(val_y_fold, val_y_predict_fold)
#             print ("Score: ", score)
            scores[i,j]=score   
            train_blend_x[val_index, (j*N_class):(j+1)*N_class] = val_y_predict_fold
            
            model.load_weights("weights.hdf5")
            # Compile model (required to make predictions)
            model.compile(loss = 'categorical_crossentropy',optimizer = 'Adamax' )            
            test_blend_x_j[:,(i*N_class):(i+1)*N_class] = model.predict_proba(x=test_x.toarray(),verbose=0)
#             print ("Model %d fold %d fitting finished in %0.3fs" % (j+1,i+1, time.time() - fold_start))            
            
        test_blend_x[:,(j*N_class):(j+1)*N_class] =                 np.stack([test_blend_x_j[:,list(range(0,N_class*fold,N_class))].mean(1),
                          test_blend_x_j[:,list(range(1,N_class*fold,N_class))].mean(1),
                          test_blend_x_j[:,list(range(2,N_class*fold,N_class))].mean(1)]).T
            
#         print ("Score for model %d is %f" % (j+1,np.mean(scores[:,j])))
    print("Score for blended models is %f in %0.3fm" % (np.mean(scores), (time.time() - fold_start)/60))
    return (train_blend_x, test_blend_x, scores,best_rounds)


# In[54]:


train_total = np.zeros((train_df_nn.shape[0],3))
test_total = np.zeros((test_df_nn.shape[0],3))
score_total = 0
count = 100
print('Starting.........')
for n in range(count):
#     print n
    nn_parameters = [
        { 'input_size' :200 ,
         'input_dim' : train_X.shape[1],
         'input_drop_out' : 0.4 ,
         'hidden_size' : 20 ,
         'hidden_drop_out' :0.4},

    ]

    (train_blend_x, test_blend_x, blend_scores,best_round) = nn_blend_data(nn_parameters, train_df_nn, train_y, test_df_nn,
                                                             10,
                                                             5,256,n)
    train_total += train_blend_x
    test_total += test_blend_x
    score_total += np.mean(blend_scores)
    
    name_train_blend = '../tmp/train.csv'
    name_test_blend = '../tmp/test.csv'

    np.savetxt(name_train_blend,train_total, delimiter=",")
    np.savetxt(name_test_blend,test_total, delimiter=",")
    
train_total = train_total / count
test_total = test_total / count
score_total = score_total / count


# In[102]:


test_total


# In[103]:


now = datetime.now()
sub_name = '../output/sub_2ndKeras_100bagging_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'

out_df = pd.DataFrame(test_total)
out_df.columns = ["low", "medium", "high"]
out_df["listing_id"] = sub_id
out_df.to_csv(sub_name, index=False)


# In[104]:



# now = datetime.now()

name_train_blend = '../output/train_blend_2ndKeras_100bagging_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
name_test_blend = '../output/test_blend_2ndKeras_100bagging_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'



print((np.mean(blend_scores,axis=0)))
print((np.mean(best_round,axis=0)))
np.savetxt(name_train_blend,train_total, delimiter=",")
np.savetxt(name_test_blend,test_total, delimiter=",")


# In[ ]:




