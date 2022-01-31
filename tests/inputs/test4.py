import pandas as pd
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import LinearRegression, train_test_split

inputs = pd.read_csv("data.csv")
y = inputs["label"]
data0 = inputs.drop("label")

data = data0
X_train_0, y_train, X_test_0, y_test = train_test_split(data, y)

select = SelectPercentile(chi2, percentile=50)
select.fit(X_train_0)
X_train = select.transform(X_train_0)
X_test = select.transform(X_test_0)

model = LinearRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)