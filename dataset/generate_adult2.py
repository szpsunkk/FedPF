import shap
X, y_true = shap.datasets.crime()
sex = X["Sex"]
print(sex.value_counts())