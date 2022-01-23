#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
from MongoClient import read_mongo
import numpy as np
from textblob import TextBlob
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

#%time nltk.download_shell()
cachedStopWords = stopwords.words("english")

algorithm = 'multinomialnb'

def splitIntoTokens(message):
    return TextBlob(message).words

def splitIntoLemmas(message):
    message = message.lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]

def hasUrl(message):
    r = re.compile(r"(http://+)|(www+)")
    match = r.search(message)
    if match is None:
        return 0
    return 1

def splitIntoWords(text):
    #1 remove html tags
    # Initialize the BeautifulSoup object to strip off html tags     
    textNoHtml = BeautifulSoup(text, "html.parser").get_text()
    #2 remove numbers and punctuation
    # Use regular expressions to do a find-and-replace
    lettersOnly = re.sub("[^a-zA-Z]"," ",textNoHtml)
    # 3. Convert to lower case, split into individual words
    words = lettersOnly.lower().split()
    #3 remove stop words
    # In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(cachedStopWords)
    woStopWords = [word for word in words if not word in stops]
    #4 convert into base form (lemma)
    baseForm = splitIntoLemmas(" " .join(woStopWords))
    #5 Join the words back into one string separated by space, 
    # and return the result.
    return(" ".join(baseForm))

def searchBestModelParameters(algorithm, trainingData):
    if algorithm == 'multinomialnb':
        # model the data using multinomial naive bayes
        # define the parameter values that should be searched
        alpha = [0, 0.2, 0.4, 0.6, 0.8, 1]
        fitPrior = [True, False]
        # specify "parameter distributions" rather than a "parameter grid"
        paramDistribution = dict(alpha = alpha, fit_prior = fitPrior)
        model = MultinomialNB()
            
    bestRun = []
    for _ in range(1):
        rand = RandomizedSearchCV(model, paramDistribution, cv=10, scoring = 'precision', n_iter = 5)
        rand.fit(trainingData, trainingData['isSpam'])
        # examine the best model
        bestRun.append({'score' : round(rand.best_score_,3), 'params' : rand.best_params_})
    print(max(bestRun, key=lambda x:x['score']))
    return max(bestRun, key=lambda x:x['score'])

def predictAndReport(algo, train, test, bestParams = None):
    if algo == 'multinomialnb':
        predictor = MultinomialNB(fit_prior = True, alpha = 0.8)
   
    
    predictor.fit(train, train['isSpam'])
    predicted = predictor.predict(test)
    
    dfWithClass = pd.DataFrame(predicted, columns = ['predictedClass'])
    final = pd.concat([test, dfWithClass], axis=1)
    #take a look at the confusion matrix
    print(pd.crosstab(final.isSpam, final.predictedClass))
    print("0s: %d, 1s: %d" %(np.sum((final.isSpam == 0) & (final.predictedClass == 0)), np.sum((final.isSpam == 1) & (final.predictedClass == 1))))
    print("Accuracy: %.3f" %float(np.sum(final.isSpam == final.predictedClass) / float(len(test))))
    print("Precision: %.3f" %float(np.sum((final.isSpam == 1) & (final.predictedClass == 1)) / np.sum(final.isSpam == 1)))
    

#read journals
rawJournals = read_mongo(db = 'CB', collection = 'journal', host = 'localhost')
journals = pd.DataFrame(list(rawJournals['body']), columns = ['content'])
journals['siteId'] = rawJournals['siteId']
journals['text'] = rawJournals['title'].astype(str) + ' ' + journals['content']
journals.drop(['content'], inplace = True, axis = 1)

#read siteIds
rawSite = read_mongo(db = 'CB', collection = 'site', host = 'localhost', no_id = False)
siteIds = pd.DataFrame(list(rawSite['_id']), columns = ['siteId'])
siteIds['isSpam'] = rawSite['isSpam']
siteIds.isSpam.fillna(0, inplace = True)
siteIds.rename(columns = {'isSpam':'isSiteSpam'}, inplace = True)

#spam data from file
octSiteProfileSpam = pd.read_csv("/Users/dmurali/Documents/spamlist_round25_from_20150809_to_20151015.csv",
                    usecols = ['siteId','isSpam'])
octSiteProfileSpam.rename(columns = {'isSpam':'isOctSpam'}, inplace = True)

#join the frames
journalsFinal = journals.merge(siteIds, how='left', on = ['siteId'], sort = False).merge(octSiteProfileSpam, how='left', on = ['siteId'], sort = False)
journalsFinal['isSpam'] = np.where(journalsFinal['isOctSpam'].isin(journalsFinal['isSiteSpam']), 1, journalsFinal['isSiteSpam'])
journalsFinal.drop(['isOctSpam', 'isSiteSpam'], inplace = True, axis = 1)


#clean up the text
journalsFinal['text'].fillna(' ', inplace = True)
journalsFinal['text'] = journalsFinal['text'].apply(splitIntoWords)
journalsFinal['length'] = journalsFinal['text'].map(lambda text: len(text))


#journalsFinal['hasUrl'] = journalsFinal.content.apply(has_url)
#journalsFinal.groupby('hasUrl').describe(include=['O'])
#journalsFinal.groupby('isSpam').describe(include=['O'])

#journalsFinal.length.plot(bins=50, kind='hist')
#journalsFinal.length.describe()
#print(journalsFinal[journalsFinal.text == 'nan'])
#journalsFinal.hist(column='length', by='isSpam', bins=50)

#tokenize the text
wordsVectorizer = CountVectorizer().fit(journalsFinal['text'])
wordsVector = wordsVectorizer.transform(journalsFinal['text'])

#print('sparse matrix shape:', wordsVector.shape)
#print('number of non-zeros:', wordsVector.nnz)
#print('sparsity: %.2f%%' % (100.0 * wordsVector.nnz / (wordsVector.shape[0] * wordsVector.shape[1])))
#vectorAsArray = wordsVector.toarray()
#wordSum = np.sum(vectorAsArray, axis = 0)
#wordsByCount = zip(wordSum, wordsVectorizer.get_feature_names())
#topByCount = sorted(wordsByCount)[:10]
#counts = pd.DataFrame(topByCount, columns = ['Count', 'Word'])
#print(counts)
#sb.barplot(x = 'Word', y = 'Count', data = counts)
inverseFreqTransformer = TfidfTransformer().fit(wordsVector)
invFreqOfWords = inverseFreqTransformer.transform(wordsVector)
#print(inverseFreqTransformer.idf_[wordsVectorizer.vocabulary_['get']])
#print(inverseFreqTransformer.idf_[wordsVectorizer.vocabulary_['aaaaannnd']])

weightedFreqOfWords = pd.DataFrame(invFreqOfWords.toarray())
weightedFreqOfWords['isSpam'] = journalsFinal['isSpam']
weightedFreqOfWords['isSpam'] = weightedFreqOfWords['isSpam'].astype(int)
train, test, spamLabelTrain, spamLabelTest = train_test_split(weightedFreqOfWords, weightedFreqOfWords['isSpam'], test_size = 0.5)    
predictAndReport(algo = algorithm, train = train, test = test)




# In[ ]:


