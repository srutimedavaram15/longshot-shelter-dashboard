import pandas as pd

df = pd.read_csv("data/cleaned_length_of_stay.csv")
df_baseline = df[df["is_censored"] == False].copy()

print(df_baseline.shape)

features = ["Intake Type", "Intake Condition", "Animal Type", "Sex upon Intake", "Age upon Intake", "Breed"]
target = "length_of_stay"

X = df_baseline[features]
y = df_baseline[target]

print(X.isna().sum())

df_baseline = df_baseline.dropna(subset=["Sex upon Intake"])

X = df_baseline[features]
y = df_baseline[target]

print(X.shape)

breed_counts = X["Breed"].value_counts()
common_breeds = breed_counts[breed_counts >= 100].index

X = X.copy()
X["Breed"] = X["Breed"].where(X["Breed"].isin(common_breeds), "Other")

print(X["Breed"].nunique())

X_encoded = pd.get_dummies(X, columns=features)

print(X_encoded.shape)
print(X_encoded.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

print(X_train.shape, X_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("R2:", r2)

import numpy as np

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

model_log = LinearRegression()
model_log.fit(X_train, y_train_log)

predictions_log = model_log.predict(X_test)

predictions_log_back = np.expm1(predictions_log)

mae_log = mean_absolute_error(y_test, predictions_log_back)
r2_log = r2_score(y_test, predictions_log_back)

print("MAE (log model):", mae_log)
print("R2 (log model):", r2_log)
