import pandas as pd
import numpy as np
# a = b = 1
# x = A()
# y = B()
# x.f = y.g
# y = t.k
# a, b = b, a
df = pd.read_csv('data.csv')
l = len(df) 
df['Fare'] = df['Fare'].fillna(np.mean(df['Fare']), inplace = False)