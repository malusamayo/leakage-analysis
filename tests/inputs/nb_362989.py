#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn import cross_validation
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from random import randrange, choice
from sklearn.neighbors import NearestNeighbors
from unbalanced_dataset import SMOTE
import seaborn as sns
pd.set_option('display.max_columns', None)
r = pd.read_csv('/Users/Felipe/PycharmProjects/FracasoEscolarChile/DatasetsProcesados/SIMCE/ALU/SIMCE_GEO_2013-2014.csv', sep='|', decimal='.')
r2 = pd.read_csv('/Users/Felipe/PycharmProjects/FracasoEscolarChile/DatasetsProcesados/RBD_GEO_2013_MUESTRA.csv', sep=',', decimal='.')
r = r.drop(r.columns[0:3],1)
#r = r[r.SIT_FIN_R != "Y"]
r = r[(r.GSE_MANZANA_ALU > 0) & (r.CPAD_DISP == 1)]
c = pd.merge(r,r2, how="inner", on=['RBD'])
r.SIT_FIN_R = preprocessing.LabelEncoder().fit_transform(r.SIT_FIN_R)
r.LET_CUR = preprocessing.LabelEncoder().fit_transform(r.LET_CUR)
r.GSE_MANZANA_ALU = preprocessing.LabelEncoder().fit_transform(r.GSE_MANZANA_ALU)
c['IVE_POND'] = c[['IVE_BASICA_RBD','IVE_MEDIA_RBD']].apply(lambda x:x.mean(),axis=1)
c['CONVIVENCIA_POND'] = c.filter(regex="CONVIVENCIA").apply(lambda x:x.mean(),axis=1)
c['AUTOESTIMA_MOTIVACION_POND'] = c.filter(regex="AUTOESTIMA_MOTIVACION").apply(lambda x:x.mean(),axis=1)
c['PARTICIPACION_POND'] = c.filter(regex="PARTICIPACION").apply(lambda x:x.mean(),axis=1)
r = c[['COD_ENSE','COD_GRADO','COD_JOR','GEN_ALU','COD_COM_ALU','REPITENTE_ALU','ABANDONA_ALU','SOBRE_EDAD_ALU','CANT_TRASLADOS_ALU','DIST_ALU_A_RBD_C','DIST_ALU_A_RBD','DESERTA_ALU','EDU_M','EDU_P','ING_HOGAR','TASA_ABANDONO_RBD','TASA_REPITENCIA_RBD','TASA_TRASLADOS_RBD','IAV_MANZANA_RBD','CULT_MANZANA_RBD','DISP_GSE_MANZANA_RBD','DEL_DROG_MANZANA_RBD','CANT_CURSOS_RBD','CANT_DELITOS_MANZANA_RBD','PROF_AULA_H_MAT_RBD','PROF_TAXI_H_MAT_RBD','CONVIVENCIA_POND','AUTOESTIMA_MOTIVACION_POND','PARTICIPACION_POND','PORC_HORAS_LECTIVAS_DOC_RBD','PROM_EDAD_TITULACION_DOC_RBD','PROM_EDAD_DOC_RBD','PROM_ANOS_SERVICIO_DOC_RBD','PROM_ANOS_ESTUDIOS_DOC_RBD','CANT_DOC_RBD','PAGO_MATRICULA_RBD','PAGO_MENSUAL_RBD','IVE_POND']]
r = r.dropna()
def f(x):
    if x.name != "ABANDONA_ALU":
        min_max_scaler = preprocessing.MinMaxScaler((-1,1))
        return min_max_scaler.fit_transform(x)
    return x
r = r.apply(f, axis=0)
X = np.array(r.drop(['DESERTA_ALU','ABANDONA_ALU'],1))
y = np.array(r['ABANDONA_ALU'])


# In[ ]:


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.33)
estimator_log = SGDClassifier(loss="log", penalty="l2", n_jobs=-1,  class_weight='auto')
estimator_linear = SGDClassifier(loss="hinge", penalty="l2", n_jobs=-1,  class_weight='auto')
estimator_per = SGDClassifier(loss="perceptron", penalty="l2", n_jobs=-1,  class_weight='auto')
estimator_mh = SGDClassifier(loss="modified_huber", penalty="l2", n_jobs=-1,  class_weight='auto')
estimator_h = SGDClassifier(loss="squared_hinge", penalty="l2", n_jobs=-1,  class_weight='auto')
estimator_hu = SGDClassifier(loss="huber", penalty="l2", n_jobs=-1,  class_weight='auto')
for estimator in [estimator_log, estimator_linear,estimator_h,estimator_hu,estimator_mh,estimator_per]:
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    m = confusion_matrix(y_test, y_pred)
    print("\n")
    print(("Modelo : " + estimator.loss))
    print("Matriz de Confusion : ")
    print(m)
    print("Precision Total de %f, un %f en la retencion(Clase 0) y %f en la desercion(Clase 1)." % ((m[0][0]+m[1][1])/(m[0][0]+m[0][1]+m[1][1]+m[1][0]),m[0][0]/(m[0][0]+m[0][1]),m[1][1]/(m[1][1]+m[1][0]))*1)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
m = confusion_matrix(y_test, y_pred)
print("\n")
print("Modelo : Naive-Bayes")
print("Matriz de Confusion : ")
print(m)
print("Precision Total de %f, un %f en la retencion(Clase 0) y %f en la desercion(Clase 1)." % ((m[0][0]+m[1][1])/(m[0][0]+m[0][1]+m[1][1]+m[1][0]),m[0][0]/(m[0][0]+m[0][1]),m[1][1]/(m[1][1]+m[1][0]))*1)


# In[15]:


X.shape


# In[16]:


y.shape


# In[ ]:




