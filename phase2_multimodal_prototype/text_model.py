import pandas as pd
import numpy as np

df = pd.read_csv("data/train.csv")

df["Type"] = df["Type"].map({1: "Dog", 2: "Cat"})

breed_labels = pd.read_csv("data/BreedLabels.csv")
breed_map = dict(zip(breed_labels["BreedID"], breed_labels["BreedName"]))
df["Breed1_name"] = df["Breed1"].map(breed_map)

features = ["Type", "Breed1_name", "Age", "Gender", "MaturitySize", "FurLength", "Health", "Fee"]
target = "AdoptionSpeed"

df_model = df[[target, "Description"] + features].dropna()

breed_counts = df_model["Breed1_name"].value_counts()
common_breeds = breed_counts[breed_counts >= 100].index
df_model["Breed1_name"] = df_model["Breed1_name"].where(df_model["Breed1_name"].isin(common_breeds), "Other")

print(df_model.shape)

from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

descriptions = df_model["Description"].tolist()
text_embeddings = embedder.encode(descriptions, show_progress_bar=True)

print(text_embeddings.shape)
np.save("text_embeddings.npy", text_embeddings)


categorical_features = ["Type", "Breed1_name", "Gender"]
df_encoded = pd.get_dummies(df_model, columns=categorical_features, drop_first=True)

structured_features = df_encoded.drop(columns=[target, "Description"]).values
combined_features = np.hstack([structured_features, text_embeddings])

print(combined_features.shape)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

y = df_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(combined_features, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
predictions_rounded = np.clip(np.round(predictions), 0, 4)

mae = mean_absolute_error(y_test, predictions)
accuracy_within_1 = (abs(y_test.values - predictions_rounded) <= 1).mean()

print("MAE (structured + text):", mae)
print("Accuracy within 1 category (structured + text):", accuracy_within_1)