import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')
l = len(df) 

y = df['Survived']
df = df.drop('Survived', axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.33, random_state=42)
X_train['Fare'] = X_train['Fare'].fillna(np.mean(X_train['Fare']), inplace = False)
X_test['Fare'] = X_test['Fare'].fillna(np.mean(X_test['Fare']), inplace = False)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty='l2', solver="sag", random_state=0)
clf.fit(X_train, y_train)
clf.predict(X_train)
y_pred = clf.predict(X_test)