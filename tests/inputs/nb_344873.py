#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'png' #set 'png' here when working on notebook")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:



train = pd.read_csv("~/Learning/Data/kaggle/house_prices/train.csv")
test = pd.read_csv("~/Learning/Data/kaggle/house_prices/test.csv")


# In[3]:


all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))


# In[4]:


matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})
prices.hist()


# In[5]:


#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
numeric_feats


# In[6]:


skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])


# In[7]:


all_data = pd.get_dummies(all_data)


# In[8]:


#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())


# In[41]:


#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice


# In[10]:


from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

def rmse_cv(model, X_train, y):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)


# In[11]:


model_ridge = Ridge()


# In[12]:


alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha), X_train, y).mean() 
            for alpha in alphas]


# In[13]:


cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")


# In[14]:


cv_ridge.min()


# In[15]:


model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)


# In[16]:


rmse_cv(model_lasso, X_train, y).mean()


# In[17]:


coef = pd.Series(model_lasso.coef_, index = X_train.columns)


# In[18]:


print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


# In[19]:


imp_coef = pd.concat([coef.sort_values().head(5),
                     coef.sort_values().tail(5)])


# In[20]:


matplotlib.rcParams['figure.figsize'] = (8.0, 4.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")


# In[24]:


#let's look at the residuals as well:
matplotlib.rcParams['figure.figsize'] = (6.0, 4.0)

preds = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":y})


# In[22]:


preds["residuals"] = preds["true"] - preds["preds"]
preds.plot(x = "preds", y = "residuals",kind = "scatter")


# In[23]:


import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label = y)
dtest = xgb.DMatrix(X_test)

params = {"max_depth":10, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)

model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()


# In[ ]:



model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(X_train, y)


# In[ ]:


xgb_preds = np.expm1(model_xgb.predict(X_test))
lasso_preds = np.expm1(model_lasso.predict(X_test))


# In[ ]:


predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})
predictions.plot(x = "xgb", y = "lasso", kind = "scatter")


# In[ ]:


preds = 0.7*lasso_preds + 0.3*xgb_preds
solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})
solution.to_csv("ridge_sol.csv", index = False)


# In[25]:



from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l1
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[45]:


X_train = StandardScaler().fit_transform(X_train)


# In[46]:


X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, random_state = 3)
X_tr.shape


# In[43]:


model = Sequential()
#model.add(Dense(256, activation="relu", input_dim = X_train.shape[1]))
model.add(Dense(1, input_dim = X_train.shape[1], W_regularizer=l1(0.001)))

model.compile(loss = "mse", optimizer = "adam")


# In[29]:


model.summary()


# In[47]:


hist = model.fit(X_tr, y_tr, validation_data = (X_val, y_val))


# In[37]:


pd.Series(model.predict(X_val)[:,0]).hist()


# In[40]:


model.predict(X_val)[:,0]


# In[ ]:




