#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division
import sys
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
import sklearn.preprocessing as preprocessing
import sklearn.feature_extraction as feature_extraction
from sklearn_pandas import DataFrameMapper, cross_val_score
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import classification_report


# ## Input

# In[2]:


r = pd.read_csv('/Users/Felipe/PycharmProjects/FracasoEscolarChile/DatasetsProcesados/SIMCE/ALU/SIMCE_GEO_2013-2014.csv', header=0, sep='|', decimal='.')
cols = ['MRUN','COD_COM_ALU','NOM_COM_ALU','SIT_FIN_R','EDAD_ALU','CODINE11','LAT_MANZANA_ALU','LON_MANZANA_ALU','RIESGO_DESERCION_RBD','DIR_RBD','LAT_MANZANA_RBD','LON_MANZANA_RBD',
        'CONVIVENCIA_2M_RBD','CONVIVENCIA_4B_RBD','CONVIVENCIA_6B_RBD','AUTOESTIMA_MOTIVACION_2M_RBD','AUTOESTIMA_MOTIVACION_4B_RBD','AUTOESTIMA_MOTIVACION_6B_RBD',
        'AUTOESTIMA_MOTIVACION_8B_RBD','PARTICIPACION_2M_RBD','PARTICIPACION_4B_RBD','PARTICIPACION_6B_RBD','PARTICIPACION_8B_RBD','IVE_MEDIA_RBD','IVE_BASICA_RBD',
        'PSU_PROM_2013_RBD','CPAD_DISP','DGV_RBD','NOM_RBD','NOM_COM_RBD','COD_COM_RBD','RBD','PROM_GRAL','ASISTENCIA','LET_CUR','CLASIFICACION_SEP_RBD']
data = r.drop(cols,1)
data = data.drop(data.columns[[0,21]],1)


# ## Preprocessing

# In[3]:


#data = data.dropna()
#cdata['CLASIFICACION_SEP_RBD'].fillna('NO SEP', inplace=True)
nocat = ['CANT_TRASLADOS_ALU','CANT_DELITOS_COM_ALU','CANT_DELITOS_MANZANA_RBD','CANT_DOC_M_RBD','CANT_DOC_F_RBD','CANT_DOC_RBD','POB_FLOT_RBD','BECAS_DISP_RBD','MAT_TOTAL_RBD','VACANTES_CUR_IN_RBD','PROM_ALU_CUR_RBD','CANT_DELITOS_COM_RBD','EDU_P','EDU_M', 
         'ING_HOGAR','CANT_CURSOS_RBD','PAGO_MATRICULA_RBD','PAGO_MENSUAL_RBD']
output_data = data[['ABANDONA_ALU','DESERTA_ALU','ABANDONA_2014_ALU']]
data = data.drop(['ABANDONA_ALU','DESERTA_ALU','ABANDONA_2014_ALU'],1)
float_data = data.loc[:, data.dtypes == float]
object_data = data.loc[:, data.dtypes == object] # Se convierten en binarias desde texto
categorical_data = data.loc[:, data.dtypes == int].drop(nocat,1) # se convierten en binarias desde numeros sin escala
categorical_data = categorical_data[['GEN_ALU','ORI_RELIGIOSA_RBD']]
integer_data = data.loc[:, data.dtypes == int][nocat]


# In[4]:


float_data_mapper = DataFrameMapper([(float_data.columns.values,[preprocessing.Imputer(missing_values=np.nan),preprocessing.MinMaxScaler((0,1))])])
integer_data_mapper = DataFrameMapper([(integer_data.columns.values,[preprocessing.Imputer(missing_values=np.nan),preprocessing.MinMaxScaler((0,1))])])
categorical_data_mapper = DataFrameMapper([(categorical_data.columns.values,preprocessing.Imputer(missing_values=-999))])
print("Imputando : 25%...")
float_data_arr = float_data_mapper.fit_transform(float_data) #Atributos Floats imputados y escalados
float_data = pd.DataFrame(float_data_arr,columns=float_data.columns)
print("Imputando : 50%...")
integer_data_arr = integer_data_mapper.fit_transform(integer_data) #Atributos Integer imputados y escalados
integer_data = pd.DataFrame(integer_data_arr,columns=integer_data.columns)
print("Imputando : 100%...")

print("Vectorizando : 0%...")
categorical_data_arr = categorical_data_mapper.fit_transform(categorical_data) #Atributos Categoricos numericos imputados
categorical_data = pd.DataFrame(categorical_data_arr,columns=categorical_data.columns)
## Vectorizar las categorias de texto
object_data_vectorizer = feature_extraction.DictVectorizer(sparse=False)
object_data_prep = object_data_vectorizer.fit_transform(object_data.T.to_dict().values())
object_data_prep_df = pd.DataFrame(object_data_prep, columns=object_data_vectorizer.get_feature_names())
print("Vectorizando : 50%...")
## Vectorizar las categorias numericas
categorical_data_vectorizer = feature_extraction.DictVectorizer(sparse=False)
categorical_data_prep = categorical_data_vectorizer.fit_transform(categorical_data.applymap(str).T.to_dict().values())
categorical_data_prep_df = pd.DataFrame(categorical_data_prep,columns=categorical_data_vectorizer.get_feature_names())
print("Vectorizando : 100%")
## Ahora escalamos los vectores binarios
#object_data_mapper_bin = DataFrameMapper([(object_data_prep_df.columns.values,preprocessing.MinMaxScaler((-1,1)))])
#categorical_data_mapper_bin = DataFrameMapper([(categorical_data_prep_df.columns.values,preprocessing.MinMaxScaler((-1,1)))])

#object_data_prep_arr_bin = object_data_mapper_bin.fit_transform(object_data_prep_df)

#categorical_data_prep_arr_bin = categorical_data_mapper_bin.fit_transform(categorical_data_prep_df)

#object_data_prep_df = pd.DataFrame(object_data_prep_arr_bin,columns=object_data_prep_df.columns)

#categorical_data_prep_df = pd.DataFrame(categorical_data_prep_arr_bin,columns=categorical_data_prep_df.columns)


# ## Definicion INPUT y OUTPUT del Modelo

# In[57]:


input_data = pd.concat([float_data, integer_data, object_data_prep_df], axis=1, join='inner')

X = np.array(input_data)
y = np.array(output_data['DESERTA_ALU'])
t_names = ['Alumno Retenido(0)','Alumno Desertor(1)']


# ## Definicion de los Modelos

# In[63]:


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.70, random_state=0)  
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
    print("Modelo : " + estimator.loss)
    print(classification_report(y_test, y_pred, target_names=t_names))
    print("\n")

## RBF, MUY LENTO NO USAR
#clf = svm.SVC(class_weight = 'auto', kernel = 'rbf')
#clf.fit(X_train, y_train)
#y_pred = clf.predict(X_test)
#print("Modelo : SVM Kernel rbf")
#print(classification_report(y_test, y_pred, target_names=t_names))
#print("Matriz de Confusion : ")
#print(m)
#print "Precision Total de %f, un %f en la retencion(Clase 0) y %f en la desercion(Clase 1)." % ((m[0][0]+m[1][1])/(m[0][0]+m[0][1]+m[1][1]+m[1][0]),m[0][0]/(m[0][0]+m[0][1]),m[1][1]/(m[1][1]+m[1][0]))*1
#print("\n")

## NAIVE BAYES, Requiere balanceo de data
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
m = confusion_matrix(y_test, y_pred)
print("Modelo : Naive-Bayes")
print(classification_report(y_test, y_pred, target_names=t_names))
#print("Matriz de Confusion : ")
#print(m)
#print "Precision Total de %f, un %f en la retencion(Clase 0) y %f en la desercion(Clase 1)." % ((m[0][0]+m[1][1])/(m[0][0]+m[0][1]+m[1][1]+m[1][0]),m[0][0]/(m[0][0]+m[0][1]),m[1][1]/(m[1][1]+m[1][0]))*1
print("\n")

