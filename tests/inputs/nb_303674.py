#!/usr/bin/env python
# coding: utf-8

# 
# ### A Classic Naive Bayes Example (80% of Doctors get this wrong):
# 1% of women at age forty who participate in routine screening have breast cancer.  80% of women with breast cancer will get positive mammographies.  9.6% of women without breast cancer will also get positive mammographies.  A woman in this age group had a positive mammography in a routine screening.  
# 
# What is the probability that she actually has breast cancer?
# 
# > .0776
# 
# <!--
# * Prior: 1% of women at age forty have breast cancer.
# * Posterior: Probability woman has breast cancer
# -->

# In[1]:


.8 * .01 / (.8 * .01 + .096 * .99)


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import naive_bayes

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 30)

critics = pd.read_csv('../../DAT18NYC/data/rt_critics.csv')


# In[3]:


from sklearn.feature_extraction.text import CountVectorizer

text = ['Math is great', 'Math is really great', 'Exciting exciting Math']

get_ipython().run_line_magic('pinfo', 'CountVectorizer')

vectorizer = CountVectorizer(ngram_range = (1,2))

vectorizer.fit(text)

print(vectorizer.get_feature_names())

x = vectorizer.transform(text)


# In[4]:


print('Sparse Matrix')
print(x)
print(type(x))
print()
print('Matrix')
x_back = x.toarray()
print(x_back)


# In[5]:


pd.DataFrame(x_back, columns = vectorizer.get_feature_names())


# In[6]:


print(critics.quote[2])


# In[7]:


rotten_vectorizer = vectorizer.fit(critics.quote)
x = vectorizer.fit_transform(critics.quote)


# In[8]:


critics.head()


# In[9]:


y = (critics.fresh == 'fresh').values.astype(int)


# In[10]:


def train_and_measure(classifier, x, y, test_size):
    from sklearn import cross_validation
    
    xtrain, xtest, ytrain, ytest = cross_validation.train_test_split(x, y, test_size = 0.2, random_state = 1234)
    clf = classifier.fit(xtrain, ytrain)
    
    training_accuracy = clf.score(xtrain, ytrain)
    test_accuracy = clf.score(xtest, ytest)
    
    print(classifier)
    print("Accuracy on training data: %0.2f" % training_accuracy)
    print("Accuracy on test data: %0.2f" % test_accuracy)
    
train_and_measure(naive_bayes.MultinomialNB(), x, y, .2)

x_ones = (x > 1) # recall that a bernoulli interpretation will only work with 1s and 0s, or binary data.
train_and_measure(naive_bayes.BernoulliNB(), x_ones, y, .2)
    
from sklearn import linear_model
train_and_measure(linear_model.LogisticRegression(), x, y, .2)


# In[11]:


def kfold_average_sd (classifier, n, x, y, return_plot = False):
    import numpy as np
    from sklearn import cross_validation
    kfold = cross_validation.KFold(n=x.shape[0], n_folds = n, random_state=1234)
    
    train_acc = []
    test_acc = []
    for train_index, test_index in kfold:
        clf = classifier.fit(x[train_index], y[train_index])
        train_acc.append(clf.score(x[train_index], y[train_index]))
        test_acc.append(clf.score(x[test_index], y[test_index]))

    if return_plot:
        plt.figure()
        sns.kdeplot(np.random.normal(loc=np.array(test_acc).mean(), scale=np.array(test_acc).std(), size=10000), shade=True)

    
    return np.array(test_acc).mean(), np.array(test_acc).std()

kfold_average_sd(naive_bayes.MultinomialNB(), 5, x, y, True)


# In[12]:


def find_k(classifier, x, y, max_num_k):
    from sklearn import cross_validation
    import numpy as np

    k_train_acc = []
    k_test_acc = []
    for i in range(2, max_num_k):
        kfold = cross_validation.KFold(n=x.shape[0], n_folds=i, shuffle=True, random_state=1234)
        test_acc, train_acc = [], []
        for train_index, test_index in kfold:
            clf = classifier.fit(x[train_index], y[train_index])
            train_acc.append(clf.score(x[train_index], y[train_index]))
            test_acc.append(clf.score(x[test_index], y[test_index]))
        k_train_acc.append(np.array(train_acc).mean())
        k_test_acc.append(np.array(test_acc).mean())

    plt.figure()
    plt.plot(list(range(2, max_num_k)), k_train_acc)
    plt.plot(list(range(2, max_num_k)), k_test_acc)
    return clf

clf = find_k(naive_bayes.MultinomialNB(), x_ones, y, 20)


# In[13]:


from sklearn.metrics import confusion_matrix


y_true = y
y_pred = clf.predict(x)

'''
Note! the confusion matrix here will be [0 1],
not [1, 0] as in the above image.
'''
conf = confusion_matrix(y_true, y_pred)

print(conf)

print(clf.score(x, y))
print(conf[0, 0] / (conf[0, 0] + conf[0, 1]))
print(conf[1, 1] / (conf[1, 0] + conf[1, 1]))


# In[14]:


prob = clf.predict_proba(x)[:,0]
bad_rotten = np.argsort(prob[y ==0])[:5]
bad_fresh = np.argsort(prob[y ==1])[-5:]

print("Mis-predicted Rotten quotes")
print('---------------------------')
for row in bad_rotten:
    print(critics[y == 0].quote.irow(row))
    print()

print("Mis-predicted Fresh quotes")
print('--------------------------')
for row in bad_fresh:
    print(critics[y == 1].quote.irow(row))
    print()


# f_classif method:
# -----------------
# 
# As noted in the class notes, we haven't dropped a single feature from our dataset in performing this anaylsis. That is to say, we are using the full set of words used in all the reviews to run our model (some 163505!). Intuitively, we could get a model that performs just as well using a much smaller subest of these features. This is desirable because it makes our model much more efficient to run, without any significant loss of accuracy.
# 
# The class notes suggest investigating the 'f_classif' method from sklearn:

# In[15]:


from sklearn.feature_selection import f_classif
get_ipython().run_line_magic('pinfo', 'f_classif')


# In[16]:


print((f_classif(x,y)))
print(len(f_classif(x,y)))
print(len(f_classif(x,y)[0]))
print(len(f_classif(x,y)[1]))


# The documentation tells us that f_classif returns arrays of the Anova F-values and p-values for each feature. A high F-value means a high degree of the variance in the test variable can be explained by variance in the feature. Therefore we should be able to create a good model by only selecting features with high F-values. 

# In[17]:


plt.figure (figsize=(8,6), dpi = 80)
ax1 = plt.plot(np.sort(f_classif(x,y)[0]), color = 'b')
plt.ylabel('F-value')

ax2 = plt.twinx()
ax2.plot(np.sort(f_classif(x,y)[1]), color = 'g')
plt.ylabel('p-value')


# In[18]:


for i in range (90,100,1):
    print(i, np.percentile(f_classif(x,y)[0], i), np.percentile(f_classif(x,y)[1], i))


# The plot and for loop above show that there is only a very small range of features that have a high F-value.

# In[19]:


mask = f_classif(x,y)[0] >= np.percentile(f_classif(x,y)[0],95)
new_features = x[:,mask]
np.shape(new_features)


# In[20]:


train_and_measure(naive_bayes.MultinomialNB(), new_features, y, .2)


# Above, I created a mask that only uses the set of features with an F-value above the 95th percentile, and then applied this to my feature set. The model with all the features had a training accuracy of 0.99, and a test accuracy = 0.76. So this new model is a better predictor for the test data, and there is less overfitting. This is good!

# In[21]:


def train_and_measure(classifier, x, y, test_size):
    '''modifying the function from class to return the training and test scores'''
    
    from sklearn import cross_validation
    
    xtrain, xtest, ytrain, ytest = cross_validation.train_test_split(x, y, test_size = 0.2, random_state = 1234)
    clf = classifier.fit(xtrain, ytrain)
    
    training_accuracy = clf.score(xtrain, ytrain)
    test_accuracy = clf.score(xtest, ytest)
    
    return training_accuracy, test_accuracy


# In[22]:


percentile = []
feature_percentage = [] # the percentage of features used, comapred to 'all features' model
train_acc = []
test_acc = []

for i in range (900,1000,2):
    mask = f_classif(x,y)[0] >= np.percentile(f_classif(x,y)[0],i/10)
    new_features = x[:,mask]
    ratio = np.shape(new_features)[1]/np.shape(x)[1]
    np.shape(new_features)
    
    percentile.append(i/10)
    feature_percentage.append(ratio)
    
    scores = train_and_measure(naive_bayes.MultinomialNB(), new_features, y, .2)
    train_acc.append(scores[0])
    test_acc.append(scores[1])
    


# In[23]:


df = pd.DataFrame({'percentile':np.array(percentile),'feature_percentage':np.array(feature_percentage),
              'train_acc':np.array(train_acc), 'test_acc':np.array(test_acc)})


# In[24]:


plt.figure (figsize=(8,6), dpi =80)
plt.plot(df.percentile, df.train_acc, label = 'Training Accuracy')
plt.plot(df.percentile, df.test_acc, label = 'Test Accuracy')
plt.ylim([.6,1])
plt.xlabel('Percentile for F-statistic')
plt.ylabel('Accuracy')
plt.legend()

ax1 = plt.twinx()
ax1.bar(df.percentile, df.feature_percentage, width =.2, alpha =.2)
plt.xlim([90,100])
plt.ylabel('Ratio of feautures used vs all features model')
plt.show()


# From the plot, we can see that there is a peak in test accuracy when the F-statistic percentile is set at around 94%. The training accuracy is 0.87, which is better than our original model, and we are olny using around 7% of the features, so it is much more efficient.
# 
# *I can't explain the odd behaviour in the ratio of features to the F-stat percentile here. Probably an error in my code. Would be grateful if whoever is marking this could enlighten me. Thanks!*

# Parts of Speech tagger
# ----------------------

# In[25]:


def ad_words(text):
    import nltk
    token = nltk.word_tokenize(text)
    tagger = nltk.pos_tag(token)
    bag_of_words = [j[0] for j in tagger if j[1] in ('JJ','JJR','JJS','RB','RBR','RBS')]
    return ' '.join(list(set(bag_of_words))) if bag_of_words else ''

critics['pos'] = critics.quote.apply(ad_words)


# In[26]:


critics['pos'].head(20)


# In[27]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range = (1,1))
x = vectorizer.fit_transform(critics.pos)
np.shape(x)[1]
# number of features


# In[28]:


train_and_measure(naive_bayes.MultinomialNB(), x, y, test_size = .2)


# By just using adjectives and adverbs from the quotes as features, I get a training accuracy of 0.80 and a test accuracy of 0.689. So using this model is a worse fit on the test data that the all features model, which had a test accuracy of 0.7. However, the number of features used is 6204 comapred to 163505, so the aglorithm is a lot quicker for only a small drop in test accuracy.

# In[29]:


critics[critics.pos == '']


# Looking at the 'pos' column, there are some empty entries. However, when we look at the 'quote' column, there are definitely some adjectives and/or adverbs present. It is clear that the pos-tagger in nltk is not perfect.
