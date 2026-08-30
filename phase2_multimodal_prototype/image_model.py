import os
import numpy as np
import pandas as pd
from PIL import Image
import open_clip
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/train.csv")

df["Type"] = df["Type"].map({1: "Dog", 2: "Cat"})

breed_labels = pd.read_csv("data/BreedLabels.csv")
breed_map = dict(zip(breed_labels["BreedID"], breed_labels["BreedName"]))
df["Breed1_name"] = df["Breed1"].map(breed_map)

features = ["Type", "Breed1_name", "Age", "Gender", "MaturitySize", "FurLength", "Health", "Fee"]
target = "AdoptionSpeed"

# Only keep animals that actually have at least one photo
df_model = df[df["PhotoAmt"] > 0][[target, "PetID"] + features].dropna()

breed_counts = df_model["Breed1_name"].value_counts()
common_breeds = breed_counts[breed_counts >= 100].index
df_model["Breed1_name"] = df_model["Breed1_name"].where(df_model["Breed1_name"].isin(common_breeds), "Other")

print(df_model.shape)

image_dir = "data/train_images"

df_model["image_path"] = df_model["PetID"].apply(lambda pid: f"{image_dir}/{pid}-1.jpg")
df_model["image_exists"] = df_model["image_path"].apply(os.path.exists)

print(df_model["image_exists"].value_counts())

# ----- Generate image embeddings with CLIP (one-time, slow: ~14,648 images) -----
EMBEDDINGS_PATH = os.path.join(os.getcwd(), "image_embeddings.npy")

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
model.eval()

image_embeddings = []

for path in df_model["image_path"]:
    image = Image.open(path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_image(image_tensor)
    image_embeddings.append(embedding.squeeze().numpy())

image_embeddings = np.array(image_embeddings)

np.save(EMBEDDINGS_PATH, image_embeddings)

# Verify the save actually worked before trusting it downstream
assert os.path.exists(EMBEDDINGS_PATH), f"Save failed — file not found at {EMBEDDINGS_PATH}"
print("Saved successfully:", EMBEDDINGS_PATH, image_embeddings.shape)

# Reload from disk to confirm the saved file is valid and readable
image_embeddings = np.load(EMBEDDINGS_PATH)
print("Reloaded from disk:", image_embeddings.shape)

# ----- Build feature sets -----
categorical_features = ["Type", "Breed1_name", "Gender"]
df_encoded = pd.get_dummies(df_model, columns=categorical_features, drop_first=True)

structured_only = df_encoded.drop(columns=[target, "PetID", "image_path", "image_exists"]).values
structured_plus_image = np.hstack([structured_only, image_embeddings])

y = df_encoded[target]

# ----- Fit and evaluate: structured + image -----
X_train, X_test, y_train, y_test = train_test_split(
    structured_plus_image, y, test_size=0.2, random_state=42
)

model_reg = LinearRegression()
model_reg.fit(X_train, y_train)

predictions = model_reg.predict(X_test)
predictions_rounded = np.clip(np.round(predictions), 0, 4)

mae = mean_absolute_error(y_test, predictions)
accuracy_within_1 = (abs(y_test.values - predictions_rounded) <= 1).mean()

print("MAE (structured + image):", mae)
print("Accuracy within 1 category (structured + image):", accuracy_within_1)