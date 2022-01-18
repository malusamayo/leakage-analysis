#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV,StratifiedKFold, KFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import random
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
import gc
from scipy.stats import skew, boxcox
from bayes_opt import BayesianOptimization
from scipy import sparse
from sklearn.metrics import log_loss
from datetime import datetime
from scipy.stats.mstats import gmean
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

seed = 2017


# # Load Data

# In[48]:


data_path = "../input/"
train_X = pd.read_csv(data_path + 'train_BM_MB_add03052240.csv')
test_X = pd.read_csv(data_path + 'test_BM_MB_add03052240.csv')
train_y = np.ravel(pd.read_csv(data_path + 'labels_BrandenMurray.csv'))
ntrain = train_X.shape[0]
sub_list = test_X.listing_id.values.copy()
# all_features = features_to_use + desc_sparse_cols + feat_sparse_cols
print(train_X.shape, test_X.shape, train_y.shape)


# In[49]:


full_data=pd.concat([train_X,test_X])
features_to_use = train_X.columns.values

skewed_cols = full_data[features_to_use].apply(lambda x: skew(x.dropna()))

SSL = preprocessing.StandardScaler()
skewed_cols = skewed_cols[skewed_cols > 0.25].index.values
for skewed_col in skewed_cols:
    full_data[skewed_col], lam = boxcox(full_data[skewed_col] - full_data[skewed_col].min() + 1)
#     print skewed_col, '\t', lam
for col in features_to_use:
    full_data[col] = SSL.fit_transform(full_data[col].values.reshape(-1,1))
    full_data[col] = full_data[col] - full_data[col].min() + 1
    train_X[col] = full_data.iloc[:ntrain][col]
    test_X[col] = full_data.iloc[ntrain:][col]

    
del full_data


# In[50]:


X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.20, random_state = seed)


# In[51]:


def MNB_cv(alpha=1.0):
    scores=[]
    est=MultinomialNB(alpha=alpha)
    est.fit(X_train, y_train)
    y_val_pred = est.predict_proba(X_val)
    return -1*log_loss(y_val, y_val_pred)


# In[52]:


cv_score = -1
for x in range(600,1000,50):
    score = MNB_cv(alpha = x)
    if score > cv_score:
        alpha = x
        cv_score = score
    print('alpha = {0}\t {1:.12}'.format(x,score))


# In[53]:


def MNB_blend(est, train_x, train_y, test_x, fold):
    N_params = len(est)
    print("Blend %d estimators for %d folds" % (N_params, fold))
    skf = KFold(n_splits=fold,random_state=seed)
    N_class = len(set(train_y))
    
    train_blend_x = np.zeros((train_x.shape[0], N_class*N_params))
    test_blend_x_mean = np.zeros((test_x.shape[0], N_class*N_params))
    test_blend_x_gmean = np.zeros((test_x.shape[0], N_class*N_params))
    scores = np.zeros((fold,N_params))
    best_rounds = np.zeros((fold, N_params))    
    
    for j, ester in enumerate(est):
        print("Model %d:" %(j+1))
        test_blend_x_j = np.zeros((test_x.shape[0], N_class*fold))

            
        for i, (train_index, val_index) in enumerate(skf.split(train_x)):
            print("Model %d fold %d" %(j+1,i+1))
            fold_start = time.time() 
            train_x_fold = train_x.iloc[train_index]
            train_y_fold = train_y[train_index]
            val_x_fold = train_x.iloc[val_index]
            val_y_fold = train_y[val_index]            
            

            ester.fit(train_x_fold,train_y_fold)
            
            val_y_predict_fold = ester.predict_proba(val_x_fold)
            score = log_loss(val_y_fold, val_y_predict_fold)
            print("Score: ", score)
            scores[i,j]=score            
            
            train_blend_x[val_index, (j*N_class):(j+1)*N_class] = val_y_predict_fold
            test_blend_x_j[:,(i*N_class):(i+1)*N_class] = ester.predict_proba(test_x)
            
            print("Model %d fold %d fitting finished in %0.3fs" % (j+1,i+1, time.time() - fold_start))            

        test_blend_x_mean[:,(j*N_class):(j+1)*N_class] =                 np.stack([test_blend_x_j[:,list(range(0,N_class*fold,N_class))].mean(1),
                          test_blend_x_j[:,list(range(1,N_class*fold,N_class))].mean(1),
                          test_blend_x_j[:,list(range(2,N_class*fold,N_class))].mean(1)]).T
        
        test_blend_x_gmean[:,(j*N_class):(j+1)*N_class] =                 np.stack([gmean(test_blend_x_j[:,list(range(0,N_class*fold,N_class))], axis=1),
                          gmean(test_blend_x_j[:,list(range(1,N_class*fold,N_class))], axis=1),
                          gmean(test_blend_x_j[:,list(range(2,N_class*fold,N_class))], axis=1)]).T
            
        print("Score for model %d is %f" % (j+1,np.mean(scores[:,j])))
    print("Score for blended models is %f" % (np.mean(scores)))
    return (train_blend_x, test_blend_x_mean, test_blend_x_gmean, scores,best_rounds)


# In[54]:


est = [MultinomialNB(alpha = 800),
      MultinomialNB(alpha = 850),
      MultinomialNB(alpha = 900),]

(train_blend_x_MNB,
 test_blend_x_MNB_mean,
 test_blend_x_MNB_gmean,
 blend_scores_MNB,
 best_rounds_MNB) = MNB_blend(est,
                             train_X,train_y,
                             test_X,
                             10)


# In[57]:


now = datetime.now()

name_train_blend = '../blend/train_blend_MNB_BM_MB_add03052240_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
name_test_blend_mean = '../blend/test_blend_MNB_mean_BM_MB_add03052240_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
name_test_blend_gmean = '../blend/test_blend_MNB_gmean_BM_MB_add03052240_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'


print((np.mean(blend_scores_MNB,axis=0)))
# print (np.mean(best_rounds_RFC,axis=0))
np.savetxt(name_train_blend,train_blend_x_MNB, delimiter=",")
np.savetxt(name_test_blend_mean,test_blend_x_MNB_mean, delimiter=",")
np.savetxt(name_test_blend_gmean,test_blend_x_MNB_gmean, delimiter=",")


# In[55]:


sub_name = '../output/sub_MNB_mean_BM_MB_add03052240_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'

out_df = pd.DataFrame(test_blend_x_MNB_mean[:,:3])
out_df.columns = ["low", "medium", "high"]
out_df["listing_id"] = sub_list
out_df.to_csv(sub_name, index=False)


# In[ ]:




