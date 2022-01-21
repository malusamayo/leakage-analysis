#!/usr/bin/env python
# coding: utf-8

# In[277]:


from sklearn.model_selection import train_test_split

 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd
import patsy

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GridSearchCV


# MAX_RESULTS_PER_CITY = 1000       ### DO NOT SET MORE THAN 1000
# URL_SEARCH_TERM = 'Data Scientist' ### DO NOT SET MORE THAN SINGLE SEARCH TERM (TITLE)
# CITY_SET = ['Washington_DC',  'Salt_Lake_City', 'Raleigh', 'Portland_OR',   'Houston', 'Denver', 'Los_Angeles']
# 
# 
# ###ALREADY RUN: ['New York', 'Chicago', 'San Francisco', 'Austin', 'Atlanta', 'Boston', 'Seattle']
# 
# ###############
# ################################################################
# 
# 
# import requests
# import bs4
# import pandas as pd
# from bs4 import BeautifulSoup
# import datetime
# import urllib
# 
# def extract_location_from_resultRow(result):
#     try:
#         location = (result.find(class_='location').text.strip())
#     except:
#         location = ''
#     return location
# 
# def extract_company_from_resultRow(result):
#     try:
#         company = (result.find(class_='company').text.strip())
#     except:
#         company = ''
#     return company
# 
# def extract_jkid_from_resultRow(result):
#     try:
#         row = (result.find(class_='jobtitle turnstileLink'))
#         jkid = result['data-jk']
#     except: 
#         jkid = ''
#     return jkid
# 
# def extract_title_from_resultRow(result):
#     try:
#         title = (result.find(class_='turnstileLink'))
#         title_text = title.text
#     except: 
#         title_text = ''
#     return title_text
# 
# def extract_salary_from_resultRow(result):
#     try:
#         salary = (result.find(class_='snip').find('nobr').text)
#     except:
#         salary = ''
#     salary_text = salary
#     return salary_text
# 
# def extract_reviews_from_resultRow(result):
#     try:
#         reviews = (result.find(class_='slNoUnderline').text.strip().strip(' reviews').replace(',',''))
#     except: 
#         reviews = ''
#     return reviews
# 
# def extract_stars_from_resultRow(result):
#     try: 
#         stars = (result.find(class_='rating')['style']).split(';background-position:')[1].split(':')[1].split('px')[0].strip()
#     except: 
#         stars = ''
#     return stars
# 
# def extract_date_from_resultRow(result):
#     try: 
#         date = (result.find(class_='date').text.strip(' ago').strip())
#     except: 
#         date = ''
#     return date
# 
# 
# for city in CITY_SET:
#     job_dict = []
#     now = datetime.datetime.now()
#     for start in range(0, MAX_RESULTS_PER_CITY, 10):
# 
#         URL = "http://www.indeed.com/jobs?q="+urllib.quote(URL_SEARCH_TERM)+"&l="+urllib.quote(city)+"&start="+str(start)
#         r=requests.get(URL)
#         soup = BeautifulSoup(r.content, "lxml")
# 
#         for i in soup.findAll("div", {"data-tn-component" : "organicJob"}):
# 
#             location = extract_location_from_resultRow(i)
#             company = extract_company_from_resultRow(i)
#             jkid = extract_jkid_from_resultRow(i)
#             title = extract_title_from_resultRow(i)
#             salary = extract_salary_from_resultRow(i)
#             reviews = extract_reviews_from_resultRow(i)
#             stars = extract_stars_from_resultRow(i)
#             post_date = extract_date_from_resultRow(i)
# 
#             job_dict.append([location, company, jkid,title, salary, stars, reviews, post_date, now])
#             
#         job_df = pd.DataFrame(job_dict, columns=['location', 'company', 'jkid', 'title', 'salary', 'stars', 'reviews', 'post_date', 'pull_date'])       
# 
#     job_df.to_csv('scrape'+city+'_'+str(MAX_RESULTS_PER_CITY)+'.csv', encoding='utf-8')

# In[205]:


San_Fran= pd.read_csv('/Users/timothyernst/Documents/General Assembly/rewebscraping/scrapeSan Francisco_1000.csv')
Atlanta= pd.read_csv('/Users/timothyernst/Documents/General Assembly/rewebscraping/scrapeAtlanta_1000.csv')
Austin= pd.read_csv('/Users/timothyernst/Documents/General Assembly/rewebscraping/scrapeAustin_1000.csv')
Boston= pd.read_csv('/Users/timothyernst/Documents/General Assembly/rewebscraping/scrapeBoston_1000.csv')
Chicago= pd.read_csv('/Users/timothyernst/Documents/General Assembly/rewebscraping/scrapeChicago_1000.csv')
New_York= pd.read_csv('/Users/timothyernst/Documents/General Assembly/rewebscraping/scrapeNew York_1000.csv')
Seattle= pd.read_csv('/Users/timothyernst/Documents/General Assembly/rewebscraping/scrapeSeattle_1000.csv')
Denver = pd.read_csv('/Users/timothyernst/Documents/General Assembly/newcities/scrapeDenver_1000.csv')
Houston = pd.read_csv('/Users/timothyernst/Documents/General Assembly/newcities/scrapeHouston_1000.csv')
Los_Angeles = pd.read_csv('/Users/timothyernst/Documents/General Assembly/newcities/scrapeLos_Angeles_1000.csv')
Portland = pd.read_csv('/Users/timothyernst/Documents/General Assembly/newcities/scrapePortland_OR_1000.csv')
Raleigh = pd.read_csv('/Users/timothyernst/Documents/General Assembly/newcities/scrapeRaleigh_1000.csv')
Salt_Lake_City = pd.read_csv('/Users/timothyernst/Documents/General Assembly/newcities/scrapeSalt_Lake_City_1000.csv')
Washington_DC = pd.read_csv('/Users/timothyernst/Documents/General Assembly/newcities/scrapeWashington_DC_1000.csv')


# In[249]:


city_data =pd.concat([San_Fran, Atlanta, Austin, Boston, Chicago, New_York, Seattle, Denver, Houston, Los_Angeles, Portland, Raleigh, Salt_Lake_City, Washington_DC])


# In[250]:


#10K data points and only 500 salaries :(
city_data.info()


# In[253]:


#Drop any rows with blank salaries. 
df = city_data.dropna(subset = ['location','company','title','salary'])
df = df.drop_duplicates(subset = ['jkid'])


# In[254]:


df.info()


# In[255]:


#string = "$120,000 - $130,000 a year"

#string.split(' ')[0]

df['min_sal'] = [x.split(' ')[0] for x in df.salary]
df['max_sal'] = [x.split(' ')[-3] for x in df.salary]
df['unit'] = [x.split(' ')[-1] for x in df.salary]


# In[256]:


df['min_sal'] = [x.replace('$', '') for x in df.min_sal]
df['max_sal'] = [x.replace('$', '') for x in df.max_sal]

#Remove the commas and make them 
df['min_sal'] = [float(x.replace(',', '')) for x in df.min_sal]
df['max_sal'] = [float(x.replace(',', '')) for x in df.max_sal]

#For some reason it made the min a float but not the max so let's try again.
#df['max_sal'] = df['max_sal'].astype(float)


# In[257]:


df.info()


# In[258]:


df['max_sal'] = df['max_sal'].astype(float)


# In[259]:


#Average the min and max to get a single number we can work with. 
df['avg_sal'] = (df['min_sal'] + df['max_sal']) /2


# In[260]:


#Anualize the salaries that appear as per month or per hour. We assume 2,000 hrs / year. 
dict_unit={'month':12, 'hour': 2000, 'year': 1, 'week': 50, 'day':250}
df['m'] = df['unit'].map(lambda x: dict_unit[x])
df['ann_sal'] = df['avg_sal']*df['m']


# In[261]:


#Split the location data into two columns.
df['state']=[x.split(', ')[1] for x in df['location']]
df['city']=[x.split(', ')[0] for x in df['location']]
df['state']=[x.split(' ')[0] for x in df['state']]


# In[262]:




def title_category(words):
        if ('scientist' in words.lower()) and ('data' in words.lower()) and ('senior' in words.lower()):
            return 'senior data scientist'
        elif('science' in words.lower()) and ('director' in words.lower()) and ('data' in words.lower()):
            return 'senior data scientist'
        elif ('scientist' in words.lower()) and ('lead' in words.lower()):
            return 'senior data scientist'
        elif ('scientist' in words.lower()) and ('sr' in words.lower()):
            return 'senior data scientist'
        elif ('science' in words.lower()) and ('manager' in words.lower()) and ('data' in words.lower()):
            return 'senior data scientist'
        elif ('scientist' in words.lower()) and ('data' in words.lower()):
            return 'data scientist'
        elif ('learning' in words.lower()) and ('machine' in words.lower()) and ('scientist' in words.lower()):
            return 'data scientist'
        elif ('analyst' in words.lower()) and ('data' in words.lower()):
            return 'data analyst'
        elif ('analyst' in words.lower()) and ('quantitative' in words.lower()):
            return 'data analyst'
        elif ('analytics' in words.lower()) and ('manager' in words.lower()):
            return 'data analyst'
        elif ('analyst' in words.lower()) and ('research' in words.lower()):
            return 'data analyst'
        elif ('research' in words.lower()) and ('associate' in words.lower()):
            return 'data analyst'
        elif ('scientist' in words.lower()) and ('research' in words.lower()):
            return 'data analyst'
        elif ('statistical' in words.lower()) and ('analyst' in words.lower()):
            return 'data analyst'
        elif ('engineer' in words.lower()) and ('data' in words.lower()):
            return 'data engineer'
        elif ('engineer' in words.lower()) and ('learning' in words.lower()) and ('machine' in words.lower()):
            return 'data engineer'
        elif ('statistical' in words.lower()) and ('programmer' in words.lower()):
            return 'data engineer'
        elif ('statistical' in words.lower()) or ('statistician' in words.lower()):
            return 'statistician'
        elif ('python' in words.lower()) or ('data' in words.lower()) or ('SQL' in words.lower()):
            return 'data other'
        elif ('scientist' in words.lower()) or ('science' in words.lower()):
            return 'scientist'
        else:
            return 'misc'


# In[263]:


df['Title_New']=df['title'].map(title_category);

"""
df['Title_New'].value_counts().nunique();

df['Title_New'].value_counts()"""


# In[265]:


#df.loc[df['Title_New']== 'misc', :]
df.Title_New.value_counts()


# In[266]:


#Create a new dataframe with only the relevant columns.
wdf = df[['Title_New', 'city', 'state', 'company', 'ann_sal']]
wdf.reset_index(inplace=True, drop=True)
wdf.head()


# In[267]:


#Generate a new dataframe that shows annaual salary as a binary of above or below the mean.
wdf['high_sal'] = wdf['ann_sal'].apply(lambda x: 1 if x > wdf['ann_sal'].mean() else 0)


# # Patsy

# In[268]:


wdf.head()


# In[271]:


# Make patsy df
X = patsy.dmatrix('~ C(city) + C(state) + C(company)+ C(Title_New)', wdf)
y = wdf['high_sal'].values

#list of column names
X.design_info.column_names


# In[274]:


## create train-test out of the data given

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=.33, random_state=2)


# In[279]:


lr = LogisticRegression(solver='liblinear') 
lr_model = lr.fit(X_train, Y_train)


# In[280]:


y_pred = lr_model.predict(X_test)
y_score = lr_model.decision_function(X_test)


# In[281]:


conmat = np.array(confusion_matrix(Y_test, y_pred, labels=[1,0]))
confusion = pd.DataFrame(conmat, index=['over_mean', 'under_mean'],
                           columns=['predicted_overmean','predicted_undermean'])


# In[282]:


print(confusion)
print(classification_report(Y_test,y_pred))
roc_auc_score(Y_test, y_score)


# In[283]:



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


# In[284]:


#run GRID SEARCH ## FILL IN C VALUES
C_vals = [0.0001, 0.001, 0.01, 0.1, .15, .25, .275, .33, 0.5, .66, 0.75, 1.0, 2.5, 5.0, 10.0, 100.0, 1000.0]
penalties = ['l1','l2']

gs = GridSearchCV(lr, {'penalty': penalties, 'C': C_vals}, verbose=False, cv=15)
gs.fit(X, y)


# In[285]:


#return best model parameters
gs.best_params_


# In[286]:


#fit model with gridsearch info
logreg = LogisticRegression(C=gs.best_params_['C'], penalty=gs.best_params_['penalty'])
cv_model = logreg.fit(X_train, Y_train)

 
#predict x
cv_pred = cv_model.predict(X_test)


# In[288]:


cm = confusion_matrix(Y_test, cv_pred, labels=logreg.classes_)
cm = pd.DataFrame(cm, columns=logreg.classes_, index=logreg.classes_)
cm


# In[289]:


print(classification_report(Y_test, cv_pred, labels=logreg.classes_))


# In[290]:


"""
FPR = dict()
TPR = dict()
ROC_AUC = dict()

# For class 1, find the area under the curve
FPR[1], TPR[1], _ = roc_curve(Y_test, cv_pred)
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
plt.show()"""

