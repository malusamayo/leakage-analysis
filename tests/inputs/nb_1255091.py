#!/usr/bin/env python
# coding: utf-8

# # Informal settlements of Syrian refugees in Lebanon
# 
# ## Overview
# * Data was downloaded from [The Humanitarian Data Exchange](https://data.humdata.org/dataset/informal-settlements-refugees-living-in-informal-settlements)
# * Description from the above link: 
# > In Lebanon Syrian refugees are scattered all over the country, mostly living in the cities and villages with host communities renting or occupying abandoned or unfinished buildings, however over 30% of the refugees are living in tents in Informal Settlements. Informal Settlements are collections of tents (not more than 20 tents).
# 
# ## Aims of this project
# * Learn how to plot maps, and visualize data on maps
# * Practice exploration of variables and feature cleaning 
# * Practice machine learning and more in-depth cross-validation
# * packages: pandas, numpy, matplotlib, mpl_toolkits.basemap, scikit-learn
# 
# ## Summary
# * Explored the variables in the dataset to investigate correlations with the Status of settlements. 
# * Cleaned up data by removing NANs and irrelevant variables.
# * Visualized a heatmap of settlement locations on the country map using basemap.
# * Used Google Geocoding API to find latitude and longitude data for specific Governorates in Lebanon, then plot their locations on the map.
# * Mapped categorical variables into ordinals, and created new ordinal categories out of numerical variables.
# * Used random forest machine learning from scikit-learn to predict the status of settlements.
# * Ran cross-validations, and visualized ROC curves.
# * Explored feature importance.
# ___

# ## Variable definitions
# * **PCode**: Officially assigned PCode. This is unique to every IS. It holds the CAS (Central Administration of Statistics) Number, the IS code and the sequemtial number of the IS. Use this code to refer to the IS.  Medair assigns PCodes. This code should be used as a reference to any documentation citing an informal settlement.
# * **PCode Name**: P-Code based naming of the informal settlement, based on the name of the cadastral and and the site's 3 digit sequential number. This is the official name of the settlement.
# * **Governorate**: Governorate
# * **District**: District
# * **Cadastral Name**: Name of the Cadaster where the settlement is located
# * **Local Name**: Local Name of the Settlement according to NGOs working there or residents
# * **Latitude**: Latitude
# * **Longitude**: Longitude
# * **Shelter Type**: Informal settlement
# * **Status**: Active, Less than 4,Inactive, Not Willing or Erroneous
# * **Number of Tents**: Number of tents in the settlements verified by physical observation
# * **Number of Individuals**: Number of individuals living in Tents, verified by asking each tent's residents how many people sleep there each night.
# * **Date of the current update**: What date the update was taken - accuracy of the data is only for this particular date as settlements change frequently.
# * **Updated By**: Partner who undertook the update, 'Other' if from a secondary data source and not yet verified
# * **Updated On**: "Sweep" in which the site was updated
# * **Discovery Date**: Calculated by running a query on all IAMP data from IAMP 1 to current IAMP to determine the first available date of update
# * **Date the site was created**: Date the settlement was first established
# * **Notes**: Any significant findings or explanation of the data
# 	
# 	
# * **Latrines**: Number of Latrines in the settlement.  This should be all free standing functional latrines, not just those built by NGOs, verified by physical observations.
# * **Water Capacity in L**: The water storage capacity availble on the site in liters. Liters of water tanks are counted and  verified by physical observations.
# * **Type of Water Source**: What is the primary source of water for the settlement? Verified by asking the Shawish.
# * **Waste Disposal**: What is the primary method of waste disposal for the settlement? Verified by asking the Shawish.
# * **Type of Contract**: Type of Contract between the Landlord and the refugees (Verbal, Written or None). Verified by asking the Shawish.
# * **Waste Water Disposal**: What is the primary method of waste water disposal for the settlement? Verified by asking the Shawish.
# * **Type of Internet Connection**: What kind of internet connection is used by the IS residents. Verified by asking the Shawish.
# * **Consultation Fee for PHC (3000/5000)**: If they know they should pay between 3000 and 5000 Lebanese Pounds for each consultation at primary health care centre. Verified by asking the Shawish.
# * **Free Vaccination for Children under 12**: If they know that refugee children < 12 have free access to vaccination at Ministry of Health facilities. Verified by asking the Shawish.
# * **Number of SSBs**: Number of SSBs in the direct confines of the informal settlement. Verified by physical observation
# * **Number of Ind in SSBs**: Number of individuals living in SSBs, verified by asking each SSB's residents how many people sleep there each night.

# ## Question to ask
# Settlements have a **Status** of *Active*, *Inactive* or *Less than 4* meaning less than 4 tents exist in that location. The latter category is on the verge of becoming *Inactive*, i.e., refugees leaving the settlement, moving elsewhere, etc. 
# ### Can we predict if a settlement is on the verge of becoming *Inactive*, i.e., if its **Status** is *Less than 4* or *Active*, based on other information in the dataset?
# ___

# In[1]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.ticker import LogFormatter 
get_ipython().run_line_magic('matplotlib', 'inline')

from datetime import datetime 

# machine learning
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier 

from mpl_toolkits.basemap import Basemap
from matplotlib import cm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# acquire data
df = pd.read_csv('ListofInformalSettlements_29_DEC_2016_All_sites.csv')
df_base = df


# In[5]:


df_base.head()


# In[6]:


df['Governorate'].unique()


# ## Missing governorates
# * Akkar Governorate is missing, and has been lumped with the North Governorate
# * Baalbak-Hermel Governorate is missing, and has been lumped with the Bekaa Governorate
# * The reason may be that the two missing governorates were only added in 2003, see http://www.localiban.org/article5075.html

# ## Heatmap of refugee informal settlement locations
# * Below we can see that the highest concentration of informal settlements is in the North and East of Lebanon. This is possibly because Syrian borders are located at the North and East. 
# * No real concentration in the capital of Beirut. Could be due to the high cost of living there and a more controlled refugee management (if any).
# 

# In[87]:


import urllib.request, urllib.parse, urllib.error, json, time

# initialize dictionary
LocDict = dict()

# get Google Geocoding API
api_table = pd.read_csv('../../PythonLearningSideProjects/APIs/Sarine_APIs.csv')
google_api = api_table['API key'][api_table['Name'].isin(["google_api"])].values[0]

# retrieve latitude and longitude for each of the Governorates and store them in a dictionary
labeltext = df_base['Governorate'].unique()
for ii in range(len(labeltext)):
    address = r"" + labeltext[ii]+" Governorate, Lebanon"
    addP = "address=" + address.replace(" ","+")
    GeoUrl = r"https://maps.googleapis.com/maps/api/geocode/json?" + addP + "&key=" + google_api
    response = urllib.request.urlopen(GeoUrl)
    jsonRaw = response.read()
    jsonData = json.loads(jsonRaw)
    if jsonData['status'] == 'OK':
        res = jsonData['results'][0]
        LocDict[labeltext[ii]] = [res['geometry']['location']['lng'],res['geometry']['location']['lat']]     
    else:
        LocDict = {None,None,None} # this line has not been verified
    
print(LocDict)


# In[88]:


uppercorner = [34.7, 36.8] #lat, long of map upper corner
lowercorner = [33.018977, 34.9] #lat, long of map lower corner

xytexts=[(-50,5),(-120,0),(-50,5),(-50,5),(-50,5),(-50,10)] # text offsets

fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
m = Basemap(projection = 'merc',llcrnrlon=lowercorner[1],llcrnrlat=lowercorner[0],urcrnrlon=uppercorner[1],urcrnrlat=uppercorner[0], lat_ts=lowercorner[0], resolution='i',epsg=22770)
#http://server.arcgisonline.com/arcgis/rest/services ESRI_Imagery_World_2D ESRI_StreetMap_World_2D NatGeo_World_Map

m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 300, verbose= True)
#m.drawcoastlines(linewidth=2)
#m.drawcountries(linewidth=2,zorder=1)
m.drawrivers(color='darkcyan',linewidth=1.5,zorder=1)

plotvar = df['Number of Ind'].values+df['Number of Ind in SSBs'].values
#plotvar = df['Water Capacity in L'].values
x, y = m(df['Longitude'].values, df['Latitude'].values)

for ii in range(len(labeltext)):
    xlab,ylab = m(labelcoord[ii][0], labelcoord[ii][1])
    xlab,ylab = m(LocDict[labeltext[ii]][0], LocDict[labeltext[ii]][1])
    m.plot(xlab, ylab, color = 'lightgray', marker = '*', markersize=15,zorder=2)
    plt.annotate(labeltext[ii],xy=(xlab,ylab),xytext=xytexts[ii],textcoords='offset points',fontsize=16,color='lightgray')

# plot a scatter of the concentration of settlements in log scale
m.hexbin(x, y,  gridsize=100, bins = 'log', mincnt=1, cmap=cm.YlOrRd_r,zorder=2); # bins='log' is log colorscale and yellow is highest (http://matplotlib.org/1.2.1/examples/pylab_examples/hexbin_demo.html)
formatter = LogFormatter(10, labelOnlyBase=False) #,norm=matplotlib.colors.LogNorm()
cb = plt.colorbar() #format=formatter
cb.set_label('Number of  settlements [log10(N)]')
m.readshapefile('/Users/Sarine/Documents/Sarine/Other/PythonLearningSideProjects/lebanon-refugee-data/LBN_adm/LBN_adm1', 'areas',linewidth = 2,zorder = 1)
#shapefile data for Lebanon downloaded from: http://www.diva-gis.org/gdata
#m.areas_info contains information about each of the areas defined in shapefile

# #only plot the 6 governorates in magenta: 
# for info, shape in zip(m.areas_info, m.areas):
#     if info['NAME_1'] in labeltext:
#         x, y = zip(*shape) 
#         m.plot(x, y, marker=None,color='m',zorder = 1)
        


# In[12]:


fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(121)
m = Basemap(projection = 'merc',llcrnrlon=lowercorner[1],llcrnrlat=lowercorner[0],urcrnrlon=uppercorner[1],urcrnrlat=uppercorner[0], lat_ts=lowercorner[0], resolution='i',epsg=22770)
#http://server.arcgisonline.com/arcgis/rest/services

m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 300, verbose= True)
m.drawcoastlines(linewidth=2)
m.drawcountries(linewidth=2,zorder=1)

plotvar = df['Number of Ind'].values+df['Number of Ind in SSBs'].values
#plotvar = df['Number of Latrines'].values
x, y = m(df['Longitude'].values, df['Latitude'].values)
m.hexbin(x, y, C=plotvar, reduce_C_function = np.mean, gridsize=100,  mincnt=1, cmap=cm.YlOrRd_r,zorder=2,norm=matplotlib.colors.LogNorm()); # bins='log' is log colorscale and yellow is highest (http://matplotlib.org/1.2.1/examples/pylab_examples/hexbin_demo.html)
formatter = LogFormatter(10, labelOnlyBase=False) #,norm=matplotlib.colors.LogNorm()
cb = plt.colorbar(format=formatter) #format=formatter
cb.set_ticks([1,10,50,100,500], update_ticks=True)
cb.set_ticklabels(['1','10','50','100','500'], update_ticks=True)
cb.set_label('Mean number of people in settlements')

ax1 = fig.add_subplot(122)
m1 = Basemap(projection = 'merc',llcrnrlon=lowercorner[1],llcrnrlat=lowercorner[0],urcrnrlon=uppercorner[1],urcrnrlat=uppercorner[0], lat_ts=lowercorner[0], resolution='i',epsg=22770)
#http://server.arcgisonline.com/arcgis/rest/services

m1.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 300, verbose= True)
m1.drawcoastlines(linewidth=2)
m1.drawcountries(linewidth=2,zorder=1)

#plotvar = df['Number of Ind'].values+df['Number of Ind in SSBs'].values
plotvar = df['Water Capacity in L'].values
x, y = m1(df['Longitude'].values, df['Latitude'].values)
m1.hexbin(x, y, C=plotvar, reduce_C_function = np.mean, gridsize=100,  mincnt=1, cmap=cm.YlOrRd_r,zorder=2,norm=matplotlib.colors.LogNorm()); # bins='log' is log colorscale and yellow is highest (http://matplotlib.org/1.2.1/examples/pylab_examples/hexbin_demo.html)
formatter = LogFormatter(10, labelOnlyBase=False) #,norm=matplotlib.colors.LogNorm()
cb = plt.colorbar() #format=formatter
cb.set_ticks([1,10,100,1000,10000,50000], update_ticks=True)
cb.set_ticklabels(['1','10','100','1000','10000','50000'], update_ticks=True)
cb.set_label('Water capacity in L')


# In[13]:


df.describe()


# ## Exploratory analysis and data cleaning
# * Can we predict from location data, water capacity, etc. if the **Status** is *Active* or *Less than 4*?
# * Remove *Inactive*, *Erroneous* and *Not Willing*; set *Less than 4* to 0, and *Active* to 1
# * Possible features to explore: **Number of Ind** (range?), **Water Capacity in L** (range), **Waste Disposal**, **Waste Water Disposal**, **Number of Latrines**, **District** or **Governorate**, Length of site creation, **Number of SSBs**, **Type of Water Source**, **Type of Contract**, **Type of Internet Connection**.

# In[14]:


#drop Inactive, Erroneous and Not Willing rows from Status:
df=df[~df['Status'].isin(["Inactive"])].reset_index(drop=True)
df=df[~df['Status'].isin(["Erroneous"])].reset_index(drop=True)
df=df[~df['Status'].isin(["Not Willing"])].reset_index(drop=True)


# In[15]:


# map Status onto 0/1:
df['Status'] = df['Status'].map( {'Active': 1, 'Less than 4': 0} ).astype(int)
df['Status'].value_counts()


# ### Missing values
# * Type of Water Source has 63 missing values
# * Waste Disposal has 66 missing values
# * Waste Water Disposal has 152 missing values
# * Type of Contract has 99 missing values
# * Type of Internet Connection has 450 missing values

# In[16]:


df['Type of Water Source'].value_counts().sum()


# In[17]:


df['Waste Disposal'].value_counts().sum()


# In[18]:


df['Waste Water Disposal'].value_counts().sum()


# In[19]:


df['Type of Contract'].value_counts().sum()


# In[20]:


df['Type of Internet Connection'].value_counts().sum()


# In[21]:


#Drop NAN in all of the above (later: can use the median to replace):
df = df.dropna(axis=0,subset=['Type of Water Source'],how='all')
df = df.dropna(axis=0,subset=['Waste Disposal'],how='all')
df = df.dropna(axis=0,subset=['Waste Water Disposal'],how='all')
df = df.dropna(axis=0,subset=['Type of Contract'],how='all')
df = df.dropna(axis=0,subset=['Type of Internet Connection'],how='all')
df = df.reset_index(drop=True)


# ### Some visualization

# In[22]:


g = sns.FacetGrid(df, col='Status')
g.map(plt.hist, 'Water Capacity in L', bins=50)


# Water capacity close to 0 in Less than 4 category.

# In[23]:


g = sns.FacetGrid(df, col='Status')
g.map(plt.hist, 'Number of Latrines', bins=10)


# In[24]:


g = sns.FacetGrid(df, col='Status')
g.map(plt.hist, 'Number of SSBs', bins=50)


# In[25]:


g = sns.FacetGrid(df, col='Status')
g.map(plt.hist, 'Number of Ind in SSBs', bins=50)


# In[26]:


# map Waste Disposal onto 1, 2, or 3 (lump Bury it with Dump it outside the camp):
df['Waste Disposal'] = df['Waste Disposal'].map( {'Municipality Collection': 1, 'Burn it': 2, 'Dump it outside the camp': 3, 'Burry it': 3} ).astype(int)
df.head()


# In[27]:


df['Waste Water Disposal'].value_counts()


# In[28]:


df['Waste Water Disposal'] = df['Waste Water Disposal'].replace(['Storm water channel', 'Septic tank','Municipality sewer network / treated',                                                                 'Irrigation canal'], 'Rare')
df['Waste Water Disposal'].value_counts()


# In[29]:


# map Waste Water Disposal into 1, 2, 3, 4, or 5:
df['Waste Water Disposal'] = df['Waste Water Disposal'].map( {'Direct discharge to environment': 1, 'Cesspit': 2, 'Open pit': 3, 'Holding tank': 4, 'Municipality sewer network / not treated': 5, 'Rare': 6} ).astype(int)
df['Waste Water Disposal'].value_counts()


# In[30]:


df[['Governorate', 'Status']].groupby(['Governorate'], as_index=False).mean().sort_values(by='Status', ascending=False)
#do not take Governorate as feature?


# In[31]:


# map Waste Water Disposal into 1, 2, 3, 4, or 5:
df['Governorate'] = df['Governorate'].map( {'Bekaa': 1, 'North': 2, 'South': 3, 'Mount Lebanon': 4, 'Nabatiye': 5, 'Beirut': 6} ).astype(int)
df['Governorate'].value_counts()


# In[32]:


df[['District', 'Status']].groupby(['District'], as_index=False).mean().sort_values(by='Status', ascending=False)
#do not take District as feature


# In[33]:


df[['Waste Disposal', 'Status']].groupby(['Waste Disposal'], as_index=False).mean().sort_values(by='Status', ascending=False)
#do not take Waste Disposal as feature


# In[34]:


df[['Waste Water Disposal', 'Status']].groupby(['Waste Water Disposal'], as_index=False).mean().sort_values(by='Status', ascending=False)
#take Waste Water Disposal as feature?


# In[35]:


df['Type of Water Source'].value_counts()


# In[36]:


df['Type of Water Source'] = df['Type of Water Source'].replace(['Spring', 'Well','Others',                                                                 'River'], 'Rare')


# In[37]:


df[['Type of Water Source', 'Status']].groupby(['Type of Water Source'], as_index=False).mean().sort_values(by='Status', ascending=False)
#we *could* take this one as a feature


# In[38]:


# map Type of Water Source into 1, 2, 3, or 4:
df['Type of Water Source'] = df['Type of Water Source'].map( {'Water Trucking': 1, 'Borehole': 2, 'Water Network': 3, 'Rare': 4} ).astype(int)
df['Type of Water Source'].value_counts()


# In[39]:


df['Type of Contract'].value_counts()


# In[40]:


df[['Type of Contract', 'Status']].groupby(['Type of Contract'], as_index=False).mean().sort_values(by='Status', ascending=False)
# this is a good feature


# In[41]:


# map Type of Contract into 1, 2, or 3:
df['Type of Contract'] = df['Type of Contract'].map( {'Verbal': 1, 'None': 2, 'Written': 3} ).astype(int)
df['Type of Contract'].value_counts()


# In[42]:


df['Type of Internet Connection'].value_counts()


# In[43]:


df[['Type of Internet Connection', 'Status']].groupby(['Type of Internet Connection'], as_index=False).mean().sort_values(by='Status', ascending=False)
#possible feature


# In[44]:


# map Type of Internet Connection into 1, 2, or 3:
df['Type of Internet Connection'] = df['Type of Internet Connection'].map( {'Mobile network - 3G / 4G': 1, 'Wifi  / local Internet service provider': 2, 'No Internet access': 3} ).astype(int)
df['Type of Internet Connection'].value_counts()


# In[45]:


df[['Consultation Fee for PHC (3000/5000)', 'Status']].groupby(['Consultation Fee for PHC (3000/5000)'], as_index=False).mean().sort_values(by='Status', ascending=False)
#not a good feature


# In[46]:


df[['Free Vaccination for Children under 12', 'Status']].groupby(['Free Vaccination for Children under 12'], as_index=False).mean().sort_values(by='Status', ascending=False)
#not a good feature


# In[47]:


df['LatrinesBand'] = pd.qcut(df['Number of Latrines'], 4)
df[['LatrinesBand', 'Status']].groupby(['LatrinesBand'], as_index=False).mean().sort_values(by='LatrinesBand', ascending=True)


# In[48]:


df.loc[df['Number of Latrines'] <= 1, 'LatrinesBandNum'] = 0
df.loc[(df['Number of Latrines'] > 1) & (df['Number of Latrines'] <= 2), 'LatrinesBandNum'] = 1
df.loc[(df['Number of Latrines'] > 2) & (df['Number of Latrines'] <= 5), 'LatrinesBandNum']   = 2
df.loc[ df['Number of Latrines'] > 5, 'LatrinesBandNum'] = 3
df['LatrinesBandNum'] = df['LatrinesBandNum'].astype(int)
df = df.drop(['LatrinesBand'], axis=1)
df.head()


# In[49]:


df['WaterBand'] = pd.qcut(df['Water Capacity in L'], 4)
df[['WaterBand', 'Status']].groupby(['WaterBand'], as_index=False).mean().sort_values(by='WaterBand', ascending=True)


# In[50]:


df.loc[df['Water Capacity in L'] <= 1000, 'WaterBandNum'] = 0
df.loc[(df['Water Capacity in L'] > 1000) & (df['Water Capacity in L'] <= 2500), 'WaterBandNum'] = 1
df.loc[(df['Water Capacity in L'] > 2500) & (df['Water Capacity in L'] <= 7000), 'WaterBandNum']   = 2
df.loc[ df['Water Capacity in L'] > 7000, 'WaterBandNum'] = 3
df['WaterBandNum'] = df['WaterBandNum'].astype(int)
df = df.drop(['WaterBand'], axis=1)
df.head()


# ### Features to keep : 
# **Water Capacity in L** (range), **Waste Water Disposal** (maybe), **Number of Latrines**, **Type of Water Source**, **Type of Contract**, **Type of Internet Connection**.

# In[51]:


df1 = df[['Status','Waste Water Disposal','Type of Water Source', 'Type of Contract','Type of Internet Connection','LatrinesBandNum','WaterBandNum','Governorate']]
df1.head()


# ## Machine Learning

# In[52]:


df_x = df1.iloc[:,1:] #all the data except labels
df_y = df1.iloc[:,0] #the labels
x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,test_size=0.2,random_state = 4)


# In[53]:


# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
y_pred = decision_tree.predict(x_test)
acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)
acc_decision_tree


# In[54]:


# k-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
acc_knn = round(knn.score(x_train, y_train) * 100, 2)
acc_knn


# In[55]:


# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
RF_fit = random_forest.fit(x_train, y_train)
#y_pred = random_forest.predict(x_test)
y_pred = RF_fit.predict(x_test)
random_forest.score(x_train, y_train)
acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)
acc_random_forest


# ### Cross-validation

# In[56]:


#CV
scores = cross_val_score(random_forest, x_train, y_train, cv=10) #,scoring='f1_macro'
print(("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)))


# In[57]:


#transforming features
from sklearn import preprocessing
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.4, random_state=0)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
#clf = SVC(C=1).fit(X_train_transformed, y_train)
clf = random_forest.fit(X_train_transformed, y_train)
X_test_transformed = scaler.transform(X_test)
clf.score(X_test_transformed, y_test)  


# In[58]:


#using a pipeline to transform features
from sklearn.pipeline import make_pipeline
clf = make_pipeline(preprocessing.StandardScaler(), SVC(C=1))
cross_val_score(clf, df_x, df_y, cv=10)


# In[59]:


from sklearn.model_selection import cross_val_predict
from sklearn import metrics
predicted = cross_val_predict(clf, df_x, df_y, cv=10)
metrics.accuracy_score(df_y, predicted) 


# ### ROC curves of cross-validation folds

# In[60]:


from scipy import interp
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

fig = plt.figure(figsize=(10,10))

X = df_x
y = df_y

n_samples, n_features = X.shape

# Add noisy features
random_state = np.random.RandomState(0)
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=6)
#classifier = svm.SVC(kernel='linear', probability=True, random_state=random_state) 
classifier = RandomForestClassifier(n_estimators=100, random_state=random_state)
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)

colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
lw = 2

i = 0
for (train, test), color in zip(cv.split(X, y), colors):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area under the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=lw, color=color,
             label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
         label='Luck')

mean_tpr /= cv.get_n_splits(X, y)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic curves')
plt.legend(loc="lower right")
plt.show()


# ### Feature importance
# The plot below suggests that two features, **Number of Latrines** and **Water Capacity in L**, are informative in this classification task.

# In[61]:


coeff_df = pd.DataFrame(df1.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(random_forest.feature_importances_)

print(coeff_df.sort_values(by='Correlation', ascending=False))

importances = random_forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in random_forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# # Print the feature ranking
# print("Feature ranking:")

# for f in range(x_train.shape[1]):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(list(range(x_train.shape[1])), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(list(range(x_train.shape[1])), x_train.columns[indices].values, rotation='vertical')
plt.xlim([-1, x_train.shape[1]])
plt.ylabel('Correlation')
plt.show()


# In[ ]:




