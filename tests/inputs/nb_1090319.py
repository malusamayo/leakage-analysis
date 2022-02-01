#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import metrics
import seaborn as sns
plt.style.use('fivethirtyeight')
import patsy
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


get_ipython().run_line_magic('load_ext', 'sql')


# ## Pre-Task: Describe the goals of your study

# Wealthier Women and children had a higher chance of surviving

#   

# ## Part 1: Aquire the Data

# psql -h dsi.c20gkj5cvu3l.us-east-1.rds.amazonaws.com -p 5432 -U dsi_student titanic
# password: gastudents
# CONNECTED IN TERMINAL

# #### 1. Connect to the remote database

# In[3]:


#import remote data into python notebook
from sqlalchemy import create_engine
engine = create_engine('postgresql://dsi_student:gastudents@dsi.c20gkj5cvu3l.us-east-1.rds.amazonaws.com/titanic')


# In[4]:


#create dataframe in pandas from titanic data
df = pd.read_sql('SELECT * FROM train', engine)


# #### 2. Query the database and aggregate the data

# In[5]:


df.head()


# #### 5. What are the risks and assumptions of our data? 

# Assume that the data without age can be dropped, assume that this is representative of all data. Assume that survived is accurate. 

# ## Part 2: Exploratory Data Analysis

# #### 1. Describe the Data

# In[6]:


df.describe()


# In[7]:


df.info()


# #### 2. Visualize the Data

# In[9]:


sns.heatmap(df.corr(), annot=True)


# In[10]:


sns.set(font_scale=1)
sns.clustermap(df.corr(), annot=True)


# In[12]:


sns.set(font_scale=1.5)
sns.pairplot(df, x_vars=["Survived"], y_vars=["Age", "Parch", "Pclass"], size=6)


# In[13]:


sns.pairplot(df.dropna())


# In[14]:


df['AgeRounded']=[round(x) for x in df['Age']]


# In[15]:


df['AgeRounded'].unique()


# In[16]:


survived_age = df[["AgeRounded", "Survived"]].groupby(['AgeRounded'],as_index=False).mean()
survived_age


# In[17]:


sns.set(font_scale=2.2)
plt.figure(figsize=(45,30))

sns.barplot(x='AgeRounded', y='Survived', data=survived_age)


# ## Part 3: Data Wrangling

# #### 1. Create Dummy Variables for *Sex* 

# In[18]:


df.head()


# In[19]:


df['female']=[1 if x=='female' else 0 for x in df['Sex']]


# In[20]:


df.head()


# ## Part 4: Logistic Regression and Model Validation

# #### 1. Define the variables that we will use in our classification analysis

# In[21]:


df=df.dropna(subset=['Age'])


# In[22]:


'''according to National Statistical Standards the age distribution of the population for demographic purposes should be
given in five-year age groups extending to 85 years and over.'''
#May try another one with smaller groups

bins = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84]
#highest age is 80
group_names =['0-4 years', '5-9 years','10-14 years', '15-19 years', '20-24 years', '25-29 years', '30-34 years','35-39 years', '40-44 years', '45-49 years', '50-54 years', '55-59 years', '60-64 years', '65-69 years','70-74 years', '75-79 years', '80-84 years']


# In[23]:


#create new column with age labels
df['agebin'] = pd.cut(df['Age'], bins, labels=group_names)


# In[28]:


#define features
features=df[['agebin','Sex','Pclass', 'Parch', 'Fare' ]]


# In[29]:


features.head()


# In[30]:


farelist=df['Fare'].unique()


# In[31]:


sorted(farelist)
#decide to note include fare as a predictor


# In[33]:


#define Predictors in categorical contexts
X= patsy.dmatrix('~ C(agebin) + C(Sex) + C(Pclass)+ C(Parch)', df)


# In[34]:


X


# #### 2. Transform "Y" into a 1-Dimensional Array for SciKit-Learn

# In[35]:


#Define category that will be predicted
y=df['Survived'].values


# #### 3. Conduct the logistic regression

# In[36]:


#DECIDED NOT TO SCALE BECUASE THERE ARE NO CONTINUOUS VARIABLES


# In[37]:


#split data into test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=.33, random_state=66)


# In[38]:


#create 'vanilla' linear model
lr = LogisticRegression(solver='liblinear') 
lr_model = lr.fit(X_train, Y_train)


# #### 4. Examine the coefficients to see our correlations

# In[39]:


lr_model.coef_[0]


# In[40]:


featurenames=X.design_info.column_names


# In[43]:


coeff_logreg= pd.DataFrame(list(zip(featurenames, lr_model.coef_[0])))
coeff_logreg.columns= ['Feature_Name', 'Coefficient']
coeff_logreg


# #### 6. Test the Model by introducing a *Test* or *Validaton* set 

# In[44]:


#Done ABOVE


# #### 7. Predict the class labels for the *Test* set

# In[45]:


#predict the survival for X_test 
y_pred= lr_model.predict(X_test)


# #### 8. Predict the class probabilities for the *Test* set

# In[46]:


lr_model.predict_proba(X_test)


# #### 9. Evaluate the *Test* set

# In[47]:


lr_model.score(X_test, Y_test)


# #### 10. Cross validate the test set

# In[48]:


cross_val_score(lr_model, X_test, Y_test, cv=3, scoring='f1_weighted').mean()


# #### 11. Check the Classification Report

# In[49]:


print(classification_report(Y_test, y_pred, labels=lr_model.classes_))


# #### 12. What do the classification metrics tell us?

# #F1 score is weighted average of the precision and recall. This is where 1 is best and 0 is worst. 

# #### 13. Check the Confusion Matrix

# In[50]:


cm = confusion_matrix(Y_test, y_pred, labels=lr_model.classes_)
cm = pd.DataFrame(cm, columns=lr_model.classes_, index=lr_model.classes_)
cm


# #### 14. What does the Confusion Matrix tell us? 

# Errors of predicted survival tend to occur 18% of the time. Specifically, there were 15 occurances of a false positive and 28 occurances of false negatives

# #### 15. Plot the ROC curve

# In[51]:


#get score for the y prediction
y_score = lr_model.decision_function(X_test)


# In[52]:


FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(Y_test, y_score)
ROC_AUC[1] = auc(FPR[1], TPR[1])

# Plot of a ROC curve for class 1 (has_cancer)
plt.figure(figsize=[11,9])
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('Receiver operating characteristic for high/low income', fontsize=18)
plt.legend(loc="lower right")
plt.show()


# #### 16. What does the ROC curve tell us?

# "This means that the top left corner of the plot is the “ideal” point - a false positive rate of zero, and a true positive rate of one."-http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

# ## Part 5: Gridsearch

# #### 1. Use GridSearchCV with logistic regression to search for optimal parameters 
# 
# - Use the provided parameter grid. Feel free to add if you like (such as n_jobs).
# - Use 5-fold cross-validation.

# In[53]:


logreg_parameters = {
    'penalty':['l1','l2'],
    'C':np.logspace(-5,1,50),
    'solver':['liblinear']
}


# In[54]:


#run gridsearch on the dict
gs = GridSearchCV(lr_model, {'penalty': logreg_parameters['penalty'], 'C': logreg_parameters['C']}, verbose=False, cv=15)
gs.fit(X_train, Y_train)


# In[55]:


#return best model parameters
gs.best_params_


# In[56]:


#return best score
gs.best_score_


# In[57]:


#fit model with gridsearch info
logreg = LogisticRegression(C=gs.best_params_['C'], penalty=gs.best_params_['penalty'])
gs_model = logreg.fit(X_train, Y_train)

 
#predict y
gs_pred = gs_model.predict(X_test)


# In[58]:


#create confusion matrix
cm1 = confusion_matrix(Y_test, gs_pred, labels=logreg.classes_)
cm1 = pd.DataFrame(cm1, columns=logreg.classes_, index=logreg.classes_)
cm1


# #### 2. Print out the best parameters and best score. Are they better than the vanilla logistic regression?

# Seen above... It seems as though the false positive rate grew in the gridsearch model and the false negative fell. The model now errors on the side of saying someone survived. 

# In[59]:


y_score2 = gs_model.decision_function(X_test)


# In[60]:


FPR_GS = dict()
TPR_GS = dict()
ROC_AUC_GS = dict()

# For class 1, find the area under the curve
FPR_GS[1], TPR_GS[1], _ = roc_curve(Y_test, y_score2)
ROC_AUC_GS[1] = auc(FPR_GS[1], TPR_GS[1])

# Plot of a ROC curve for class 1 (has_cancer)
plt.figure(figsize=[11,9])
plt.plot(FPR_GS[1], TPR_GS[1], label='ROC curve (area = %0.2f)' % ROC_AUC_GS[1], linewidth=4)
plt.plot(FPR[1], TPR[1], label='ROC curve (area = %0.2f)' % ROC_AUC[1], linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('Receiver operating characteristic for high/low income', fontsize=18)
plt.legend(loc="lower right")
plt.show()


# #### 3. Explain the difference between the difference between the L1 (Lasso) and L2 (Ridge) penalties on the model coefficients.

# L1 tends to drop one variable. In a way the coefficient becomes 0. L2(Ridge) tends to change the coefficients in a smaller way.

# #### 4. What hypothetical situations are the Ridge and Lasso penalties useful?

# Use Lasso when there are too many variables and you would like to remove one/some. Ridge is used when you would like to find a middle ground between variables. 

# #### 5. [BONUS] Explain how the regularization strength (C) modifies the regression loss function. Why do the Ridge and Lasso penalties have their respective effects on the coefficients?

# In[ ]:





# #### 6.a. [BONUS] You decide that you want to minimize false positives. Use the predicted probabilities from the model to set your threshold for labeling the positive class to need at least 90% confidence. How and why does this affect your confusion matrix?

# In[ ]:





# ## Part 6: Gridsearch and kNN

# #### 1. Perform Gridsearch for the same classification problem as above, but use KNeighborsClassifier as your estimator
# 
# At least have number of neighbors and weights in your parameters dictionary.

# In[61]:


knn = KNeighborsClassifier()


# In[62]:


#create dictionary for gridsearch
param_dict = dict(n_neighbors=list(range(1, 51)),                  weights=['uniform', 'distance'])


# #### 2. Print the best parameters and score for the gridsearched kNN model. How does it compare to the logistic regression model?

# In[63]:


#Perform gridsearch
gscv = GridSearchCV(knn, param_dict, scoring='accuracy')


# In[64]:


#fit model based upon gridsearch
gscv_model = gscv.fit(X_train, Y_train)


# In[65]:


#return estimator info
gscv_model.best_estimator_.get_params()


# In[66]:


#return best parameters
gscv.best_params_


# In[67]:


#return score for these parameters
gscv.best_score_


# #### 3. How does the number of neighbors affect the bias-variance tradeoff of your model?
# 
# #### [BONUS] Why?

# The lower neighbors the possibility of a neighbor skewing the result is higher. Too high of neighbors could yeild just a majority of the data set

# #### 4. In what hypothetical scenario(s) might you prefer logistic regression over kNN, aside from model performance metrics?

# the data doesnt seem to have a clear split between classification

# #### 5. Fit a new kNN model with the optimal parameters found in gridsearch. 

# In[69]:


#model fit above with parameters


# In[68]:


gscv_ypred = gscv_model.predict(X_test)


# In[70]:


print(classification_report(Y_test, gscv_ypred))


# #### 6. Construct the confusion matrix for the optimal kNN model. Is it different from the logistic regression model? If so, how?

# In[71]:


cm2 = confusion_matrix(Y_test, gscv_ypred)
cm2 = pd.DataFrame(cm2)
cm2


# In[72]:


#looks similar to gridsearch done with logistic regression


# In[ ]:





# #### 7. [BONUS] Plot the ROC curves for the optimized logistic regression model and the optimized kNN model on the same plot.

# In[ ]:





# ## Part 7: [BONUS] Precision-recall

# #### 1. Gridsearch the same parameters for logistic regression but change the scoring function to 'average_precision'
# 
# `'average_precision'` will optimize parameters for area under the precision-recall curve instead of for accuracy.

# In[ ]:





# #### 2. Examine the best parameters and score. Are they different than the logistic regression gridsearch in part 5?

# In[ ]:





# #### 3. Create the confusion matrix. Is it different than when you optimized for the accuracy? If so, why would this be?

# In[ ]:





# #### 4. Plot the precision-recall curve. What does this tell us as opposed to the ROC curve?
# 
# [See the sklearn plotting example here.](http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)

# In[ ]:





# ## Part 8: [VERY BONUS] Decision trees, ensembles, bagging

# #### 1. Gridsearch a decision tree classifier model on the data, searching for optimal depth. Create a new decision tree model with the optimal parameters.

# In[ ]:





# #### 2. Compare the performace of the decision tree model to the logistic regression and kNN models.

# In[ ]:





# #### 3. Plot all three optimized models' ROC curves on the same plot. 

# In[ ]:





# #### 4. Use sklearn's BaggingClassifier with the base estimator your optimized decision tree model. How does the performance compare to the single decision tree classifier?

# In[ ]:





# #### 5. Gridsearch the optimal n_estimators, max_samples, and max_features for the bagging classifier.

# In[ ]:





# #### 6. Create a bagging classifier model with the optimal parameters and compare it's performance to the other two models.

# In[ ]:




