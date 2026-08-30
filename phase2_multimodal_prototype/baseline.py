import pandas as pd

df = pd.read_csv("data/train.csv")

df["Type"] = df["Type"].map({1: "Dog", 2: "Cat"})

breed_labels = pd.read_csv("data/BreedLabels.csv")
breed_map = dict(zip(breed_labels["BreedID"], breed_labels["BreedName"]))
df["Breed1_name"] = df["Breed1"].map(breed_map)

features = ["Type", "Breed1_name", "Age", "Gender", "MaturitySize", "FurLength", "Health", "Fee"]
target = "AdoptionSpeed"

df_model = df[[target] + features].dropna()

breed_counts = df_model["Breed1_name"].value_counts()
common_breeds = breed_counts[breed_counts >= 100].index
df_model["Breed1_name"] = df_model["Breed1_name"].where(df_model["Breed1_name"].isin(common_breeds), "Other")

categorical_features = ["Type", "Breed1_name", "Gender"]
df_encoded = pd.get_dummies(df_model, columns=categorical_features, drop_first=True)

print(df_encoded.shape)

from sklearn.model_selection import train_test_split

X = df_encoded.drop(columns=[target])
y = df_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
predictions_rounded = np.clip(np.round(predictions), 0, 4)

mae = mean_absolute_error(y_test, predictions)
accuracy_within_1 = (abs(y_test.values - predictions_rounded) <= 1).mean()

print("MAE (raw):", mae)
print("Accuracy within 1 category:", accuracy_within_1)