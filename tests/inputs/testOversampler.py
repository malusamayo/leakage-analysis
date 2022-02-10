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
X_resampled, y_resampled = sampler.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=0)
model = HistGradientBoostingClassifier(random_state=0)
model.fit(X_train, y_train)
model.predict(X_test)
# cv_results = cross_validate(
#     model, X_resampled, y_resampled, scoring="balanced_accuracy",
#     return_train_score=True, return_estimator=True,
#     n_jobs=-1
# )
# print(
#     f"Balanced accuracy mean +/- std. dev.: "
#     f"{cv_results['test_score'].mean():.3f} +/- "
#     f"{cv_results['test_score'].std():.3f}"
# )
