#!/usr/bin/env python
# coding: utf-8

# In[16]:


#-*-encoding:utf-8 -*-
#!/usr/bin/python
import pandas as pd
import numpy as np
from io import StringIO
csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
0.0,11.0,12.0,'''
csv_data = unicode(csv_data)
df = pd.read_csv(StringIO(csv_data))


# In[17]:


#Eliminating samples or features with missing values
df.dropna()
df.dropna(axis = 1)


# In[18]:


#Replacing missing data with mean interpolation
from sklearn.impute import SimpleImputer as Imputer
# from sklearn.preprocessing import Imputer
imr = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
imputed_data


# In[19]:


'''
Handling categoical data
'''
df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                   ['red', 'L', 13.5, 'class2'],
                   ['blue', 'XL', 15.3, 'class1']])
df.columns = ['color', 'size', 'price', 'classlabel']

#Mapping ordinal features
size_mapping = {'XL':3, 'L':2, 'M':1}
df['size'] = df['size'].map(size_mapping)

#Encoding class labels
class_mapping = {label : idx for idx,label in enumerate(np.unique(df['classlabel']))}
df['classlabel'] = df['classlabel'].map(class_mapping)


# In[20]:


#Partitioning a dataset in training and test sets
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash','Alcalinity of ash', 'Magnesium','Total phenols', 'Flavanoids',
                   'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[21]:


'''
Features prepocessing includes two main methods
One is normalization, another is standardization
'''
#N0rmalization
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

#Standardardization
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X_train_std = std.fit_transform(X_train)
X_test_std = std.transform(X_test)


# In[22]:


'''Selecting meaningful features
In the overfitting case, there are 4 commmon ways
1. More training data
2. penalty
3. simpler model
4. reduce dimensionality for data
'''
#L1 regularization
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty = 'l1', C = 0.1)
lr.fit(X_train_std, y_train)
Training_accuracy = lr.score(X_train_std, y_train)
Testing_accuracy = lr.score(X_test_std, y_test)

#Regularization path
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan','magenta', 'yellow', 'black','pink', 'lightgreen', 'lightblue','gray', 'indigo', 'orange']
weights, params = [], []
for c in np.arange(-4,6):
    lr = LogisticRegression(penalty='l1', C=10**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column],label=df_wine.columns[column+1], color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center',bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
plt.show()


# In[51]:


'''
@Author: Darcy
@Date: May, 17, 2017
@Topic: SBS
A classic sequential feature selection algorithm is Sequential Backward Selection (SBS)
which aims to reduce the dimensionality of the initial feature subspace 
with a minimum decay in performance of the classifier 
to improve upon computational efficiency
'''
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

'''
@Author: Darcy
@Date: May, 17, 2017
@Topic: SBS
A classic sequential feature selection algorithm is Sequential Backward Selection (SBS)
which aims to reduce the dimensionality of the initial feature subspace 
with a minimum decay in performance of the classifier 
to improve upon computational efficiency
'''
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SBS:
	def __init__(self, estimator, k_features,scoring=accuracy_score,
		         test_size=0.25, random_state=1):
		self.scoring = scoring
		self.estimator = clone(estimator)
		self.k_features = k_features
		self.test_size = test_size
		self.random_state = random_state

	def fit(self, X, Y):
		x_train, y_train, x_test, y_test = 				train_test_split(X, Y, test_size = self.test_size, 
									random_state = self.random_state) 
		dim = np.shape(x_train)[1]
		self.indices = tuple(range(dim))
		self.subSets_ = [self.indices]
		score = self.calScore(self.x_train, y_train, x_test, y_test, self.indices)
		self.scores_ = [score]
		while dim > self.k_features:
			scores = []
			subSet = []
			for p in combinations(self.indices, dim - 1):
				score = self.calScore(self.x_train, y_train, x_test, y_test, p)
				scores.append(score)
				subSet.append(p)
			best = np.argmax(score)
			self.indices = subSet[best]
			self.subSets_.append(self.indices)
			dim -= 1
			self.scores_.append(scores[best])
		self.k_score = self.scores_[-1]
		return self


	def calScore(self, x_train, y_train, x_test, y_test, indices):
		self.estimator.fit(x_train, y_train)
		y_pred = self.estimator.predict(x_test)
		score = self.scoring(y_test, y_pred)
		return score

class SBS():
	def __init__(self, estimator, k_features,
            scoring=accuracy_score,
            test_size=0.25, random_state=1):
		self.scoring = scoring
		self.estimator = clone(estimator)
		self.k_features = k_features
		self.test_size = test_size
		self.random_state = random_state
        
	def fit(self, X, y):
		X_train, X_test, y_train, y_test = 		train_test_split(X, y, test_size=self.test_size,
		random_state=self.random_state)
		dim = X_train.shape[1]
		self.indices_ = tuple(range(dim))
		self.subsets_ = [self.indices_]
		score = self._calc_score(X_train, y_train,
		X_test, y_test, self.indices_)
		self.scores_ = [score]
		while dim > self.k_features:
			scores = []
			subsets = []
			for p in combinations(self.indices_, r=dim-1):
				score = self._calc_score(X_train, y_train,
				X_test, y_test, p)
				scores.append(score)
				subsets.append(p)
				best = np.argmax(scores)
				self.indices_ = subsets[best]
				self.subsets_.append(self.indices_)
			dim -= 1
			self.scores_.append(scores[best])
		self.k_score_ = self.scores_[-1]
		return self

	def _calc_score(self, x_train, y_train, x_test, y_test, indices):
		self.estimator.fit(x_train, y_train)
		y_pred = self.estimator.predict(x_test)
		score = self.scoring(y_test, y_pred)
		return score



# In[52]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)
import matplotlib.pyplot as plt
fig = plt.figure()
k_fea = [len(k) for k in sbs.subsets_]
plt.plot(k_fea, sbs.scores_, marker = 'o')
plt.show()


# In[ ]:





# In[ ]:




