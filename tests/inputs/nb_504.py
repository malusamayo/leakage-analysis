#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/emadphysics/Amsterdam_Airbnb_predictive_models/blob/main/airbnb_pytorch.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from sklearn.feature_selection import *
from sklearn.linear_model import *
from sklearn.neighbors import *
from sklearn.svm import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.preprocessing import *
from xgboost import *
from sklearn.metrics import *
from geopy.distance import great_circle
# Geographical analysis
import json # library to handle JSON files
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe
from statsmodels.tsa.seasonal import seasonal_decompose
import requests
import descartes
import math


print('Libraries imported.')


# In[ ]:


from google.colab import drive
drive.mount("/content/gdrive")


# In[ ]:


df=pd.read_csv('/content/gdrive/My Drive/listingss.csv')


# In[ ]:


print(f'the numer of observations are {len(df)}')


# In[ ]:


categoricals = [var for var in df.columns if df[var].dtype=='object']
numerics = [var for var in df.columns if (df[var].dtype=='int64')|(df[var].dtype=='float64')]
dates=[var for var in df.columns if df[var].dtype=='datetime64[ns]']


# In[ ]:


#pandas data types: numeric(float,integer),object(string),category,Boolean,date
one_hot_col_names = ['host_id',  'host_location', 'host_response_time','host_is_superhost','host_neighbourhood','host_has_profile_pic','host_identity_verified',
           'neighbourhood','neighbourhood_cleansed','neighbourhood_group_cleansed', 'zipcode', 'is_location_exact', 'property_type', 'room_type', 'bed_type', 'has_availability', 'requires_license', 'instant_bookable', 
           'is_business_travel_ready', 'cancellation_policy', 'cancellation_policy','require_guest_profile_picture', 'require_guest_phone_verification', 'calendar_updated']

text_cols = ['name', 'summary', 'space', 'description', 'neighborhood_overview', 'notes', 'transit', 'access', 'interaction', 'house_rules', 'host_name', 'host_about']

features = ['host_listings_count', 'host_total_listings_count', 'latitude', 'longitude', 
      'accommodates', 'bathrooms', 'bedrooms', 'beds', 'square_feet',     
      'guests_included', 'minimum_nights', 'maximum_nights', 'availability_30', 'availability_60', 
      'availability_90', 'availability_365', 'number_of_reviews', 'review_scores_rating', 'review_scores_accuracy', 
      'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 
      'review_scores_value', 'calculated_host_listings_count', 'reviews_per_month']
 
price_features = ['security_deposit', 'cleaning_fee', 'extra_people','price'] 


date_cols = ['host_since', 'first_review', 'last_review']


# In[ ]:


def host_verification(cols):
    possible_words = {}
    i = 0
    for col in cols:
        words = col.split()
        for w in words:
            wr = re.sub(r'\W+', '', w)
            if wr != '' and wr not in possible_words:
                possible_words[wr] = i
                i += 1
    l = len(possible_words)

    new_cols = np.zeros((cols.shape[0], l))
    for i, col in enumerate(cols):
        words = col.split()
        arr = np.zeros(l)
        for w in words:
            wr = re.sub(r'\W+', '', w)
            if wr != '':
                arr[possible_words[wr]] = 1
        new_cols[i] = arr
    return new_cols

def amenities(cols):
    dic = {}
    i = 0
    for col in cols:
        arr = col.split(',')
        for a in arr:
            ar = re.sub(r'\W+', '', a)
            if len(ar) > 0:
                if ar not in dic:
                    dic[ar] = i
                    i += 1
    
    l = len(dic)
    new_cols = np.zeros((cols.shape[0], l))
    for i, col in enumerate(cols):
        words = col.split(',')
        arr = np.zeros(l)
        for w in words:
            wr = re.sub(r'\W+', '', w)
            if wr != '':
                arr[dic[wr]] = 1
        new_cols[i] = arr
    return new_cols


def one_hot(arr):
    
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(arr)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

one_hot_col_names = ['host_response_time','host_is_superhost','host_has_profile_pic','host_identity_verified',
           'neighbourhood_cleansed','neighbourhood_group_cleansed', 'zipcode', 'is_location_exact', 'property_type', 'room_type', 'bed_type', 'has_availability', 'requires_license', 'instant_bookable', 
           'is_business_travel_ready', 'cancellation_policy','require_guest_profile_picture', 'require_guest_phone_verification','calendar_updated']
one_hot_dict = {}
for i in one_hot_col_names:
    one_hot_dict[i] = one_hot(np.array(df[i].fillna(""), dtype=str))
one_hot_dict['host_verifications'] = host_verification(df['host_verifications'])
one_hot_dict['amenities'] = amenities(df['amenities'])
ont_hot_list = []

for i in list(one_hot_dict.keys()):
    if 1<one_hot_dict[i].shape[1]<400:
        
        ont_hot_list.append(one_hot_dict[i])
#        print(i,one_hot_dict[i].shape[1])
onehot_variables = np.concatenate(ont_hot_list, axis=1)


# In[ ]:


hot_cat_variables=pd.DataFrame(onehot_variables)
hot_cat_variables.isnull().sum().sum()
hot_cat_variables.shape


# In[ ]:


def check_nan(cols):
    for col in cols:
        if np.isnan(col):
            return True
    return False

def clean_host_response_rate(host_response_rate, num_data):
    total = 0
    count = 0
    for col in host_response_rate:
        if not isinstance(col, float):
            total += float(col.strip('%'))
            count += 1

    arr = np.zeros(num_data)
    mean = total / count
    for i, col in enumerate(host_response_rate):
        if not isinstance(col, float):
            arr[i] += float(col.strip('%'))
        else:
            assert(math.isnan(col))
            arr[i] = mean
    return arr

def clean_price(price, num_data):
    arr = np.zeros(num_data)
    for i, col in enumerate(price):
        if not isinstance(col, float):
            arr[i] += float(col.strip('$').replace(',', ''))
        else:
            assert(math.isnan(col))
            arr[i] = 0
    return arr

def to_np_array_fill_NA_mean(cols):
    return np.array(cols.fillna(np.nanmean(np.array(cols))))


num_data = df.shape[0]
arr = np.zeros((len(features) + len(price_features) + 1, num_data))

host_response_rate = clean_host_response_rate(df['host_response_rate'], num_data)
arr[0] = host_response_rate
i = 0
for feature in features:
    i += 1
    if check_nan(df[feature]):
        arr[i] = to_np_array_fill_NA_mean(df[feature])
    else:
        arr[i] = np.array(df[feature])
    
for feature in price_features:
    i += 1
    arr[i] = clean_price(df[feature], num_data)

target = arr[-1]
numeric_variables = arr[:-1].T


# In[ ]:


numeric_variables=pd.DataFrame(numeric_variables)
numeric_variables.isnull().sum()                          .sum()


# In[ ]:


inde_variables=np.concatenate((numeric_variables,hot_cat_variables),axis=1)
inde_variables=pd.DataFrame(inde_variables)


# In[ ]:


inde_variables.isnull().sum().sum()


# In[ ]:


mean = np.mean(inde_variables, axis = 0)
std = np.std(inde_variables, axis = 0)
inde_variables=(inde_variables-mean)/std


# In[ ]:


inde_variables.shape


# In[ ]:


import torch
from torch import nn
import torch.optim as optim
import numpy as np
import random
import copy
import torch.utils.data as data
import os


# In[ ]:


class NN229(nn.Module):
    def __init__(self, input_size=355, hidden_size1=128, hidden_size2=512, hidden_size3=64, output_size=1, drop_prob=0.05):
        super(NN229, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
        self.W1 = nn.Linear(input_size, hidden_size1)
        self.W2 = nn.Linear(hidden_size1, hidden_size2)
        self.W3 = nn.Linear(hidden_size2, hidden_size3)
        self.W4 = nn.Linear(hidden_size3, output_size)
        
    def forward(self, x):
        hidden1 = self.dropout(self.relu(self.W1(x)))
        hidden2 = self.dropout(self.relu(self.W2(hidden1)))
        hidden3 = self.dropout(self.relu(self.W3(hidden2)))
        out = self.W4(hidden3)
        return out


# In[ ]:


class AirBnb(data.Dataset):
    def __init__(self, train_path, label_path):
        super(AirBnb, self).__init__()

        self.x = torch.from_numpy(train_path).float()
        self.y = torch.from_numpy(label_path).float()

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        
        return x, y

    def __len__(self):
        return self.x.shape[0]


# In[ ]:



class CSVDataset(data.Dataset):

  def __init__(self, train_path, label_path):

        super(CSVDataset, self).__init__()
        self.x = torch.from_numpy(train_path).float()
        self.y = torch.from_numpy(label_path).float()
        self.y = self.y.reshape((len(self.y), 1))
  def __len__(self):
    return len(self.x)
  def __getitem__(self, idx):
    return [self.x[idx], self.y[idx]]
  def get_splits(self, n_test=0.33):
    test_size = round(n_test * len(self.x))
    train_size = len(self.x) - test_size
    return data.random_split(self, [train_size, test_size])        


# In[ ]:


def load_model(model, optimizer, checkpoint_path, model_only = False):
    ckpt_dict = torch.load(checkpoint_path, map_location="cuda:0")

    model.load_state_dict(ckpt_dict['state_dict'])
    if not model_only:
        optimizer.load_state_dict(ckpt_dict['optimizer'])
        epoch = ckpt_dict['epoch']
        val_loss = ckpt_dict['val_loss']
    else:
        epoch = None
        val_loss = None
    return model, optimizer, epoch, val_loss


# In[ ]:


np.log(target)


# In[ ]:


def train(model:NN229, optimizer, loss_fn, epoch = 0):
    train_dataset = CSVDataset(inde_variables.to_numpy(), target)
    train, test = train_dataset.get_splits()
    train_loader = data.DataLoader(train,
                                  batch_size=batch_size,
                                  shuffle=True)
    dev_loader = data.DataLoader(test,
                                batch_size=batch_size,
                                shuffle=True)
    model.train()
    
    step = 0
    best_model = NN229()
    best_epoch = 0
    best_val_loss = None
    while epoch < max_epoch:
        epoch += 1
        stats = []
        with torch.enable_grad():
            for x, y in train_loader:
                step += 1
                # print (x)
                # print (y)
                # break
                x = x.cuda()
                y = y.cuda()
                optimizer.zero_grad()
                pred = model(x).reshape(-1)
                loss = loss_fn(pred, y)
                loss_val = loss.item()
                loss.backward()
                optimizer.step()
                stats.append(loss_val)
                # stats.append((epoch, step, loss_val))
                # print ("Epoch: ", epoch, " Step: ", step, " Loss: ", loss_val)
        print(("Train loss: ", sum(stats) / len(stats)))
        val_loss = evaluate(dev_loader, model)
        if best_val_loss is None or best_val_loss > val_loss:
            best_val_loss = val_loss
            model.cpu()
            best_model = copy.deepcopy(model)
            model.cuda()
            best_epoch = epoch
        # print (evaluate(dev_loader, model))
        
    return best_model, best_epoch, best_val_loss


# In[ ]:


def evaluate(dev_loader, model:NN229):
    model.eval()
    stats = []
    with torch.no_grad():
        for x, y in dev_loader:
            x = x.cuda()
            y = y.cuda()
            pred = model(x).reshape(-1)
            loss_val = loss_fn(pred, y).item()
            stats.append(loss_val)
            # print ("Loss: ", loss_val)
    print(("Val loss: ", sum(stats) / len(stats)))
    return sum(stats) / len(stats)


# In[ ]:


lr = 1e-4
weight_decay = 1e-5
beta = (0.9, 0.999)
max_epoch = 100
batch_size = 64

model = NN229().cuda()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=beta)
loss_fn = nn.MSELoss()


# In[ ]:


best_model, best_epoch, best_val_loss = train(model, optimizer, loss_fn, epoch = 0)


# In[ ]:


train_dataset = CSVDataset(inde_variables.to_numpy(), target)
train, test = train_dataset.get_splits()
dev_loader = data.DataLoader(test,
                                shuffle=True)


# In[ ]:


y_truth_list = []

for _, y_truth in dev_loader:
  y_truth_list.append(y_truth[0][0].cpu().numpy())
y_pred_list = [a.squeeze().tolist() for a in y_truth_list]
y_t=np.array(y_truth_list)
y_t


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
y_pred_list = []
with torch.no_grad():
    model.eval()
    for X_batch, _ in dev_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_pred_list.append(y_test_pred.cpu().numpy())
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
y_p=np.array(y_pred_list)


# In[ ]:


y_p


# In[ ]:


import sklearn.metrics
sklearn.metrics.r2_score(y_t, y_p)

