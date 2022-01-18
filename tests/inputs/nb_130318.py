#!/usr/bin/env python
# coding: utf-8

# All the materials including this notebook are available in Github: [parantapag/IBD4Health2017](https://github.com/parantapag/IBD4Health2017)
# 
# ## Vector Space Model
# We are interested in using this data to build statistical models. So, we now need to **vectorize** this data. The goal is to find a way to represent the data so that the computer can understand it.
# 
# ### Bag of words
# A bag of words represents each document in a corpus as a series of features. Most commonly, the features are the collection of all unique words in the vocabulary of the entire corpus. The values are usually the count of the number of times that word appears in the document (term frequency). A corpus is then represented as a matrix with one row per document and one column per unique word.
# 
# ### Scikit-Learn
# [Scikit-learn](http://scikit-learn.org/stable/) is machine learning library for the Python programming language. It features a wide range of machine learning algorithms for classification, regression and clustering. It also provides various supporting machine learning techniques such as cross validation, text vectorizer. Scikit-learn is designed to interoperate with the Python numerical and scientific libraries [NumPy](http://www.numpy.org/).
# 
# Simple to use: import the required module and call it.

# ## Vectorizer
# To build our initial bag of words count matrix, we will use scikit-learn's CountVectorizer class to transform our corpus into a bag of words representation. CountVectorizer expects as input a list of raw strings containing the documents in the corpus. It takes care of the tokenization, transformation to lowercase, filtering stop words, building the vocabulary etc. It also tabulates occurrance counts per document for each feature.

# In[ ]:


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

raw_docs_sample = ["The dog sat on the mat.", "The cat sat on the mat!", "We have a mat in our house."]

vectorizer = CountVectorizer(stop_words='english')  
X_sample = vectorizer.fit_transform(raw_docs_sample)
X_sample


# ### Sparse Vs Dense Matrices
# Dense matrices store every entry in the matrix. Sparse matrices only store the nonzero entries. Sparse matrices don't have a lot of extra features, and some algorithms may not work for them. You use them when you need to work with matrices that would be too big for the computer to handle them, but they are mostly zero, so they compress easily. Be aware of issues that may arise at:
# - dot product
# - slicing (row, column)
# 
# In python these are taken care almost automatically, by using sparse dot product and implementations of csr and csc matrices (`scipy.sparse.csr_matrix`, `scipy.sparse.csc_matrix`, etc..). 

# In[ ]:


print("Count Matrix:")
print(X_sample.todense())
print("\nWords in vocabulary:")
print(vectorizer.get_feature_names())


# ## TF-IDF Weighting Scheme
# The tf-idf weighting scheme is frequently used text mining applications and has been shown to be effective. It combines local (term frequency or tf) and global(inverse document frequency of idf) term statistics. 
# 
# Scikit-learn has your back, it already provides the module to compute TF-IDF matrix.
# 
# Note: Scikit-learn uses a slightly different formula than that we saw today morning. You can refer to [corresponding documentation](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html) to know more.

# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer  
tfidf = TfidfTransformer()  
X_tfidf_sample = tfidf.fit_transform(X_sample)
print("TF-IDF Matrix:\n")
print(X_tfidf_sample.todense())


# ## A Bigger Collection
# We will use [**Ohsumed** collection](ftp://medir.ohsu.edu/pub/ohsumed). It includes medical abstracts from the MeSH categories of the year 1991. Each abstract is assigned one of [*23 categories*](http://disi.unitn.it/moschitti/corpora/First-Level-Categories-of-Cardiovascular-Disease.txt). Here is an example of an abstract under *Bacterial Infections and Mycoses* catgory:
# >Replacement of an aortic valve cusp after neonatal endocarditis.
# >Septic arthritis developed in a neonate after an infection of her hand.
# >Despite medical and surgical treatment endocarditis of her aortic valve developed and the resultant regurgitation required emergency surgery.
# >At operation a new valve cusp was fashioned from preserved calf pericardium.
# >Nine years later she was well and had full exercise tolerance with minimal aortic regurgitation.
# 
# In this hands-on we will use 1641 documents belonging to two categories, namely *Bacterial Infections and Mycoses* and *Musculoskeletal Diseases*.
# 
# The file **corpus.txt** supplied here, contains 1641 documents. Each line of the file is a document.
# 
# Now we will:
#    1. Load the documents as a list
#    2. Create TF-IDF vectors

# In[ ]:


raw_docs = open("corpus.txt").read().splitlines()
print("Loaded " + str(len(raw_docs)) + " documents.")


# In[ ]:


# Write code to vectorize the raw documents to count matrix


# Write code to create TF-IDF vectors
# Store the TF-IDF matrix in a variable named X_tfidf


# ## Text Classifier
# Machine learning algorithms need a training set. In our text classification scenario, we need category or class labels for all 1641 documents in the collection.
# 
# In our collection we have documents from two categories: "Bacterial Infections and Mycoses" (category C01) and  "Musculoskeletal Diseases" (category C05). For each document we know the labels. The labels are stored in **corpus_labels.txt** file. Each line of **corpus.txt** file corresponds to the label in the same line of file **corpus_labels.txt**.
# 
# Lets load the labels.

# In[ ]:


labels = open("corpus_labels.txt").read().splitlines()
print("Loaded " + str(len(labels)) + " labels.")


# For the sake of simplicity, we will assume numerical labels for the categories. So labels 'C01' is replaces with numeric 1.0 and labels 'C05' are replaced with numeric -1.0

# In[ ]:


# Replace string labels with numerical ones
y = np.array([1.0 if label=="C01" else -1.0 for label in labels])


# ### Training and Testing
# 
# As we wish to first train a model and then to see how well it is. So the norm is to divide the data into two parts:
#  1. **Training set:** Documents along with their class labels are used to train the model.
#  2. **Test set:** Documents are used for predicting the class labels using the trained classifier. However, the class labels of this set are kept *hidden* and are only revealed during evalutaion of the trained model, not before that.

# In[ ]:


# package to split training and testing data
from sklearn.model_selection import train_test_split

# split the data into training and testing
X_train, X_test, y_train, y_test =train_test_split(X_tfidf, y, test_size = 0.2, random_state = 256)


# ### Training the Classifier

# In[ ]:


from sklearn.naive_bayes import MultinomialNB

# train the classifier
classifierNB = MultinomialNB()  
classifierNB.fit(X_train, y_train)


# ### Testing
# Now we will test the trained classifier on out kept hidden test data to see how well it did. 
# 
# Here we will look at the accuracy of the model, the simplest evaluation measure used in machine learning algorithms: $$accuracy = \frac{\text{number of correctly classified examples}}{\text{total number of examples}}$$
# 
# There are more informative and complex evaluaton measures, e.g. precision, recall, f-measure etc.

# In[ ]:


# Package for reporting accuracy
from sklearn.metrics import accuracy_score

# predict labels for the test set
predictionsNB = classifierNB.predict(X_test.toarray())

# report accuracy
accuracyNB = accuracy_score(y_test, predictionsNB)
print("Test accuracy: " + str(accuracyNB))


# ### Other Classifiers
# You have virtually an endless option to choose your classifier. Let's try some more.
# 
# Method is simple: import and call the package
# 
# #### Support Vector Machine (SVM)

# In[ ]:


# import package
from sklearn.svm import SVC

# Write code to create and train a SVM classifier
classifierSVM = ...


# Write code to predict labels for the test set
predictionsSVM = ...

# Write code to report accuracy
accuracySVM = ...
print("Test accuracy: " + str(accuracySVM))


# #### Random Forest

# In[ ]:


# import package
from sklearn.ensemble import RandomForestClassifier

# Write code to create and train a Linear Regression classifier
classifierRF = ...


# Write code to predict labels for the test set
predictionsRF = ...

# Write code to report accuracy
accuracyRF = ...
print("Test accuracy: " + str(accuracyRF))


# In[ ]:




