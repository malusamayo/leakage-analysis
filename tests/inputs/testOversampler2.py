from sklearn.datasets import fetch_openml
from imblearn.datasets import make_imbalance
X, y = fetch_openml(
    data_id=1119, as_frame=True, return_X_y=True
)
X = X.select_dtypes(include="number")
X, y = make_imbalance(
    X, y, sampling_strategy={">50K": 300}, random_state=1
)
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_validate

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
sampler = RandomOverSampler(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
model = HistGradientBoostingClassifier(random_state=0)
model.fit(X_resampled, y_resampled)
model.predict(X_test)