#!/usr/bin/env python
# coding: utf-8

# In[8]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import urllib.request, urllib.parse, urllib.error
import pandas
get_ipython().run_line_magic('matplotlib', 'inline')


# PREPARACION DATOS

# In[9]:


# 1. CARGUE DE DATOS
# URL for the Pima Indians Diabetes dataset (UCI Machine Learning Repository)
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
# download the file
raw_data = urllib.request.urlopen(url)

# load the CSV file as a numpy matrix
dataset = np.loadtxt(raw_data, delimiter=",")
target =[int(x[0]) for x in dataset]
dataset = [x[1:] for x in dataset]
X =dataset
Y =np.int_(target)

#2. ESTANDARIZA LOS DATOS
X = preprocessing.scale(dataset)

#3. CREA LAS COLECCIONES DE ENTRENAMIENTO (70%) y PRUEBA(30%)
traintX, testX, traintY, testY = train_test_split(X,Y,test_size=0.3, train_size=0.7)

print(len(traintX), len(traintY))
print(len(testX), len(testY))


# 2. RANDOM FOREST

# Para implemetar el metodo de clasificacion por random-forest se incluyen todas las variables

# In[14]:



#4. ENTORNO PARA GRAFICAR

from collections import OrderedDict
RandomSqrt=RandomForestClassifier(warm_start=True, oob_score=True, max_features="sqrt")
Randomlog2=RandomForestClassifier(warm_start=True, oob_score=True, max_features='log2')
RandomNone=RandomForestClassifier(warm_start=True, oob_score=True, max_features=None)
                               
arreglo = [
    ("RandomForestClassifier, max_features='sqrt'",RandomSqrt),        
    ("RandomForestClassifier, max_features='log2'",Randomlog2),       
    ("RandomForestClassifier, max_features=None"  ,RandomNone)
]
# Parejas (Etiqueta, tasa de error, estimacion)

tasa_error = OrderedDict((label, []) for label, _ in arreglo)
estimacion = OrderedDict((label, []) for label, _ in arreglo)

#5. EVALUACION DE No ESTIMADORES
min_estimators= 1
max_estimators = 200

for label, rf in arreglo:
    for i in range(min_estimators, max_estimators + 1):
        rf.set_params(n_estimators=i)
        rf.fit(traintX, traintY)
        # Adicionar el error de cada estimador
        error_ob = 1 - rf.oob_score
        tasa_error[label].append((i, error_ob))
        prediction = rf.predict(testX)
        #M = confusion_matrix(testY, prediction, labels=None) 
        valor = rf.score(testX,testY)
        estimacion[label].append((i, valor)) 
   


# In[15]:


# Grafica errores
for label, clf_err in list(tasa_error.items()):
    xs, ys = list(zip(*clf_err))
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("No estimadores")
plt.ylabel("tasa de error_oob")
plt.legend(loc="upper right")
plt.show()


# Analizado la tasa de error vs el numero de estimadores se aprecia que el error no depende ni del numero de variables, ni del numero de estimadores.

# In[16]:


# Grafica exactitud
for label, clf_err in list(estimacion.items()):
    xs, ys = list(zip(*clf_err))
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("No estimadores")
plt.ylabel("Presicion Media")

plt.legend(loc="lower right")
plt.ylim(0.90,1.01)
plt.xlim(0.0,140)
plt.show()
print() 


# Analizando la grafica de la presicion media vs el numero de estimadores,se puede observar que el mejor comportamiento se presenta cuando se incluyen la raiz cuadrada o el logaritmo de las caracteristicas. Igualmente se puede apreciar que con menos estimadores la mejor aproximacion es optima para el logaritmo de las carcateristicas.

# In[17]:


for label, clf_err in list(estimacion.items()):
    xs, ys = list(zip(*clf_err))
    plt.plot(xs, ys, label=label)
plt.xlim(min_estimators, max_estimators)
plt.xlabel("No estimadores")
plt.ylabel("PrecisionMedia")

plt.legend(loc="lower right")
plt.ylim(0.90,1.01)
plt.xlim(0.0,10)
plt.show()


# Ampliando la grafica anterior, se puede observar que incluyendo el logaritmo de las carcateristicas y 4 estimadores o arboles
# se obtiene una aproximacion media de mas del 96% 

# PCA

# In[18]:


n_components=12
K= np.arange(0, n_components, 1)
suma=0.0
SUM = np.zeros(n_components)
pca = PCA(n_components)
pca.fit(traintX, traintY)
#pca.transform(x)
PVE=pca.explained_variance_ratio_
for i in range(0,n_components):
  suma=suma+PVE[i]
  SUM[i]=suma

plt.scatter(K,SUM,color='black')
plt.plot(K,SUM,color='red')
plt.xlabel("components")
plt.ylabel("explained_variance_ratio Acumulada")
print(SUM)


# Como se puede mirar en la grafica  de explained_variance_ratio Acumulada vs componets, 7 componentes me describen el 90% de los datos.

# In[20]:


S=pca.components_
teams_list1=["1"+"C","2"+"C","3"+"C","4"+"C","5"+"C","6"+"C","7"+"C","8"+"C","9"+"C","10"+"C","11"+"C","12"+"C"]
teams_list2=[1,2,3,4,5,6,7,8,9,10,11,12,13]
pandas.DataFrame(S, teams_list1, teams_list2)    


# La tabla que se enecutra arriba indica el valor porcentual de lo que aporta cada variable (vinos) a cada componente.
# De esta manera los elementos 7,6,11,12 y 12 describen en su mayoria la primera componente

# K-MEANS
# 

# In[21]:


n_clusters=4
#d = KMeans.inertia_
#kMeansVar = [KMeans(n_clusters=k).fit(x)for k in range(1, n)]
kmeans = KMeans(n_clusters)
kmeans.fit(traintX, traintY)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

for i in range(n_clusters):
    # select only data observations with cluster label == i
    ds = traintX[np.where(labels==i)]
    # plot the data observations
    plt.plot(ds[:,0],ds[:,1],'o')
    # plot the centroids
    lines = plt.plot(centroids[i,0],centroids[i,1],'kx')
    # make the centroid x's bigger
    plt.setp(lines,ms=15.0)
    plt.setp(lines,mew=2.0)
plt.xlabel("Case1")
plt.ylabel("Clase2")    


# In[22]:


n_clusters=15
scoreR = np.zeros(n_clusters)
inerciaR = np.zeros(n_clusters)
cluster= np.arange(0, n_clusters, 1)

for i in range(1, n_clusters):
    kmeans = KMeans(i)
    kmeans.fit(traintX, traintY)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    scoreR[i] = kmeans.score(traintX, traintY) 
    inerciaR[i] = kmeans.inertia_ 
print(inerciaR, cluster)
plt.xlim(1, n_clusters)
plt.xlabel("No cluster")
plt.ylabel("score")
#plt.legend(loc="upper right")
plt.scatter(cluster,inerciaR)
plt.show()


# La grafica scrore vs Numero de clusters nos indica que desde aproximadamente, con 3 clusters el comportamiento es similar, de esta manera el numero optimo es 3 clusters

# OPTIMIZACION DE METODOS

# In[23]:


#Disminucion dimensionalidad
n_components=7
pca = PCA(n_components)
pca.fit(dataset)
Aux = pca.transform(dataset)
X = preprocessing.scale(Aux)
traintX, testX, traintY, testY = train_test_split(X,Y,test_size=0.3, train_size=0.7)


# PCA sera aplicado para disminuir la dimensionalidad de los datos, luego se implementa k-Means y RandomForest para ver cual de los dos metodos es mas optimo. Teniendo en cuanta los analisis anteriores, para K-Means el que pdoria dar mejor resultado es con 3 cluster y para RandomForest con 4 estimadores y con el logaritmo de las carcateristicas

# In[24]:


from sklearn.metrics import classification_report

#tabala clustering
n_clusters=3
kmeans = KMeans(n_clusters)
kmeans.fit(traintX, traintY)
prediction = kmeans.predict(testX)
print((classification_report(testY, prediction)))


# In[25]:


#tabala random forest
rf1=RandomForestClassifier(warm_start=True, max_features="log2", n_estimators=4)
rf1.fit(traintX, traintY)
prediction = rf1.predict(testX)
print((classification_report(testY, prediction)))


# Las dos tablas anteriores nos arrojan informacion sobre cual fue la precicion para cada clase (en cada metodo), podemos ver que la primera solo es precisa para las clases 1 y 2. Podemos concluir que Random Forest es superior a k-Means en cuanto a que la preciscion de cada clase y  total de la misma es mucho mayor
