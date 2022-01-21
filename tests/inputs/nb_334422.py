#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


normal_data = pd.read_csv("../data/ReviewData100k.csv")
data_stop_words_removed = pd.read_csv("../data/data_stop_word_removed.csv")
data_pos_tags = pd.read_csv("../data/data_POS_tags.csv")


# In[3]:


X_train_norm = normal_data[["text","stars"]]
X_train_sw = data_stop_words_removed[["text","stars"]]
X_train_pos = data_pos_tags[["text","stars"]]


# In[4]:


X_train_norm = X_train_norm[X_train_norm.text.isnull() != True]
X_train_sw = X_train_sw[X_train_sw.text.isnull() != True]
X_train_pos = X_train_pos[X_train_pos.text.isnull() != True]


# In[5]:


from sklearn.feature_extraction.text import TfidfVectorizer

transformer = TfidfVectorizer()


# In[6]:


y_norm = X_train_norm["stars"]
X_train_norm = transformer.fit_transform(X_train_norm["text"])


# In[7]:


y_sw = X_train_sw["stars"]
X_train_sw = transformer.fit_transform(X_train_sw["text"])


# In[8]:


y_pos = X_train_pos["stars"]
X_train_pos = transformer.fit_transform(X_train_pos["text"])


# ## Evaluation metric - rmse

# In[9]:


def rmse(pred,labels):
    return np.sqrt(np.mean((pred - labels) ** 2))


# ## Split train and test data

# In[10]:


RANDOM_STATE = 2016


# In[11]:


from sklearn.model_selection import train_test_split


# Choose data set and train and test split accordingly

# In[12]:


#X_train, X_test, y_train, y_test = train_test_split(X_train_norm, y_norm, test_size=0.33, random_state=RANDOM_STATE)


# In[13]:


#X_train, X_test, y_train, y_test = train_test_split(X_train_sw, y_sw, test_size=0.33, random_state=RANDOM_STATE)


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X_train_pos, y_pos, test_size=0.33, random_state=RANDOM_STATE)


# ## Baseline model

# In[15]:


y_pred_base = [np.mean(y_train)] * len(y_test)


# In[16]:


print("Baseline model accuracy: ", rmse(y_pred_base,y_test))


# ## Linear regression

# In[17]:


from sklearn.linear_model import LinearRegression


# In[18]:


lr_model = LinearRegression()
lr_model.fit(X_train,y_train)


# In[19]:


y_pred_lr = lr_model.predict(X_test)


# In[20]:


print("Ridge model accuracy: ", rmse(y_pred_lr,y_test))


# ## Ridge and Lasso regression

# In[21]:


from sklearn.linear_model import RidgeCV,LassoCV


# In[22]:


ridge_model = RidgeCV(alphas=[0.01,0.05,0.10,0.20,0.50,1])
lasso_model = LassoCV(alphas=[0.01,0.05,0.10,0.20,0.50,1])


# In[23]:


ridge_model.fit(X_train,y_train)
lasso_model.fit(X_train,y_train)


# In[24]:


y_pred_rm = ridge_model.predict(X_test)
y_pred_lm = lasso_model.predict(X_test)


# In[25]:


print("Ridge model accuracy: ", rmse(y_pred_rm,y_test))
print("Lasso model accuracy: ",rmse(y_pred_lm,y_test))


# ## K nearest neighbour

# In[26]:


from sklearn.neighbors import KNeighborsRegressor


# In[27]:


knn_model5 = KNeighborsRegressor(n_neighbors=5)
knn_model10 = KNeighborsRegressor(n_neighbors=10)
knn_model50 = KNeighborsRegressor(n_neighbors=50)
#knn_model100 = KNeighborsRegressor(n_neighbors=100)


# In[28]:


knn_model5.fit(X_train,y_train)
knn_model10.fit(X_train,y_train)
knn_model50.fit(X_train,y_train)
#knn_model100.fit(X_train,y_train)


# In[29]:


y_pred_knn5 = knn_model5.predict(X_test)
y_pred_knn10 = knn_model10.predict(X_test)
y_pred_knn50 = knn_model50.predict(X_test)
#y_pred_knn100 = knn_model100.predict(X_test)


# In[30]:


print("knn model with 5 neighbours accuracy: ", rmse(y_pred_knn5,y_test))
print("knn model with 10 neighbours accuracy: ",rmse(y_pred_knn10,y_test))
print("knn model with 50 neighbours accuracy: ",rmse(y_pred_knn50,y_test))


# ### Decision trees regressor

# In[31]:


from sklearn.tree import DecisionTreeRegressor


# In[32]:


dt_model = DecisionTreeRegressor(random_state=RANDOM_STATE)


# In[33]:


dt_model.fit(X_train,y_train)


# In[34]:


y_pred_dt = dt_model.predict(X_test)


# In[35]:


print("Decision tree accuracy: ", rmse(y_pred_dt,y_test))


# ### Random forest regressor

# In[36]:


from sklearn.ensemble import RandomForestRegressor


# In[37]:


rf_model = RandomForestRegressor(max_depth=4,n_estimators=100,max_features='sqrt',verbose=1,random_state=RANDOM_STATE)


# In[38]:


rf_model.fit(X_train,y_train)


# In[39]:


y_pred_rf = rf_model.predict(X_test)


# In[40]:


print("Randomforest with 100 estimators accuracy: ", rmse(y_pred_rf,y_test))


# ## Adaboost

# In[41]:


from sklearn.ensemble import AdaBoostRegressor


# In[42]:


adb_model = AdaBoostRegressor(n_estimators=100,learning_rate=0.01,random_state=RANDOM_STATE,loss="square")


# In[43]:


adb_model.fit(X_train,y_train)


# In[44]:


y_pred_adb = adb_model.predict(X_test)


# In[45]:


print("Adaboost with 100 estimators accuracy: ", rmse(y_pred_adb,y_test))


# ### Gradient boosting

# In[46]:


from sklearn.ensemble import GradientBoostingRegressor


# In[47]:


gbm_model = GradientBoostingRegressor(n_estimators=100,learning_rate=0.01,
                                      random_state=RANDOM_STATE,max_depth=4,max_features="sqrt")


# In[48]:


gbm_model.fit(X_train,y_train)


# In[49]:


y_pred_gbm = gbm_model.predict(X_test.todense())


# In[50]:


print("GBM with 100 estimators accuracy: ", rmse(y_pred_gbm,y_test))


# In[ ]:




