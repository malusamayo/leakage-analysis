import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')
l = len(df) 

y = df['Survived']
df = df.drop('Survived', axis=1)
# df1 = df.copy()
# df1['Fare'] = df['Fare'].fillna(np.mean(df['Fare']), inplace = False)
from sklearn import preprocessing
df1 = preprocessing.scale(df)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.33, random_state=42)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty='l2', solver="sag", random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)