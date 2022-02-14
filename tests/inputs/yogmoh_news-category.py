#!/usr/bin/env python
# coding: utf-8

# # Overview
#   From the beginning, since the first printed newspaper, every news that makes into a page has had a specific section allotted to it. Although pretty much everything changed in newspapers from the ink to the type of paper used, this proper categorization of news was carried over by generations and even to the digital versions of the newspaper. Newspaper articles are not limited to a few topics or subjects, it covers a wide range of interests from politics to sports to movies and so on. For long, this process of sectioning was done manually by people but now technology can do it without much effort. In this hackathon, Data Science and Machine Learning enthusiasts like you will use Natural Language Processing to predict which genre or category a piece of news will fall in to from the story. Size of training set: 7,628 records Size of test set: 2,748 records FEATURES: STORY:  A part of the main content of the article to be published as a piece of news. SECTION: The genre/category the STORY falls in. There are four distinct sections where each story may fall in to. 
#   **The Sections are labelled as follows : **
# *       Politics: 0 
# *       Technology: 1 
# *       Entertainment: 2 
# *       Business: 3

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


get_ipython().system('pip install -U texthero')


# In[3]:


import tensorflow as tf
import pandas as pd
import numpy as np
import texthero as hero

import matplotlib.pyplot as plt
import re
import matplotlib as mpl

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
mpl.rcParams['figure.dpi'] = 300


# In[4]:


train = pd.read_excel('/kaggle/input/new-category/Data_Train.xlsx')
test= pd.read_excel('/kaggle/input/new-category/Data_Test.xlsx')

display(train.sample(5))
display(train.info())
display(test.info())


# In[5]:


train.STORY[0]


# In[6]:


train.SECTION.value_counts(normalize=True)


# In[7]:


combined_df = pd.concat([train.drop('SECTION',axis=1),test])
combined_df.info()


# In[8]:


hero.visualization.wordcloud(combined_df['STORY'], max_words=1000,background_color='BLACK')


# In[9]:


hero.Word2Vec


# In[10]:


combined_df['cleaned_text']=(combined_df['STORY'].pipe(hero.remove_angle_brackets)
                    .pipe(hero.remove_brackets)
                    .pipe(hero.remove_curly_brackets)
                    .pipe(hero.remove_diacritics)
                    .pipe(hero.remove_digits)
                    .pipe(hero.remove_html_tags)
                    .pipe(hero.remove_punctuation)
                    .pipe(hero.remove_round_brackets)
                    .pipe(hero.remove_square_brackets)
                    .pipe(hero.remove_stopwords)
                    .pipe(hero.remove_urls)
                    .pipe(hero.remove_whitespace)
                    .pipe(hero.lowercase))


# In[11]:


lemm = WordNetLemmatizer()

def word_lemma(text):
    words = nltk.word_tokenize(text)
    lemma = [lemm.lemmatize(word) for word in words]
    joined_text = " ".join(lemma)
    return joined_text


# In[12]:


combined_df['lemmatized_text'] = combined_df.cleaned_text.apply(lambda x: word_lemma(x))


# In[13]:


text = []
for i in range(len(combined_df)):
    review = nltk.word_tokenize(combined_df['lemmatized_text'].iloc[i])
    review = ' '.join(review)
    text.append(review)


# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,f1_score,plot_confusion_matrix

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer


# In[15]:


X = combined_df['lemmatized_text'].iloc[:7628]
test_df = combined_df['lemmatized_text'].iloc[7628:]


# In[16]:


cv = CountVectorizer(max_features=9000)
cv.fit(X)
X = cv.transform(X)
test_df = cv.transform(test_df)

y = train.SECTION


# In[17]:


tfid = TfidfVectorizer(max_features=9000)
tfid.fit(X)
X = tfid.transform(X)
test_df = tfid.transform(test_df)

y = train.SECTION


# In[18]:


from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_new,y_new = smote.fit_resample(X,y)


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state=42,stratify=y_new)


# In[20]:


from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
accuracy_score(y_test,y_pred)


# In[21]:


from xgboost import XGBClassifier

xg=XGBClassifier()
xg.fit(X_train,y_train)
y_pred = xg.predict(X_test)
accuracy_score(y_test,y_pred)


# In[22]:


from catboost import CatBoostClassifier
cat=CatBoostClassifier(task_type='GPU')
cat.fit(X_train,y_train)
y_pred = cat.predict(X_test)
accuracy_score(y_test,y_pred)


# In[23]:


predictions=xg.predict(test_df)
submissions = pd.DataFrame({'SECTION':predictions})
submissions.to_csv('./sub8.csv',index=False,header=True)


# In[ ]:




