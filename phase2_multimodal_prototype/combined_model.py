import os
import numpy as np
import pandas as pd
from PIL import Image
import open_clip
import torch
from sentence_transformers import SentenceTransformer
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


df_model = df[df["PhotoAmt"] > 0][[target, "PetID", "Description"] + features].dropna()

breed_counts = df_model["Breed1_name"].value_counts()
common_breeds = breed_counts[breed_counts >= 100].index
df_model["Breed1_name"] = df_model["Breed1_name"].where(df_model["Breed1_name"].isin(common_breeds), "Other")

print(df_model.shape)

image_dir = "data/train_images"
df_model["image_path"] = df_model["PetID"].apply(lambda pid: f"{image_dir}/{pid}-1.jpg")
df_model["image_exists"] = df_model["image_path"].apply(os.path.exists)
print(df_model["image_exists"].value_counts())


df_model = df_model[df_model["image_exists"]].reset_index(drop=True)
print("Final row count for combined model:", len(df_model))


embedder = SentenceTransformer("all-MiniLM-L6-v2")
descriptions = df_model["Description"].tolist()
text_embeddings = embedder.encode(descriptions, show_progress_bar=True)

TEXT_PATH = os.path.join(os.getcwd(), "combined_text_embeddings.npy")
np.save(TEXT_PATH, text_embeddings)
assert os.path.exists(TEXT_PATH)
print("Text embeddings saved:", text_embeddings.shape)

clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_model.eval()

image_embeddings = []
for path in df_model["image_path"]:
    image = Image.open(path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        embedding = clip_model.encode_image(image_tensor)
    image_embeddings.append(embedding.squeeze().numpy())

image_embeddings = np.array(image_embeddings)

IMAGE_PATH = os.path.join(os.getcwd(), "combined_image_embeddings.npy")
np.save(IMAGE_PATH, image_embeddings)
assert os.path.exists(IMAGE_PATH)
print("Image embeddings saved:", image_embeddings.shape)

print(len(df_model), text_embeddings.shape[0], image_embeddings.shape[0])
assert len(df_model) == text_embeddings.shape[0] == image_embeddings.shape[0], \
    "Row count mismatch — do not proceed until this is fixed."

categorical_features = ["Type", "Breed1_name", "Gender"]
df_encoded = pd.get_dummies(df_model, columns=categorical_features, drop_first=True)

structured_only = df_encoded.drop(
    columns=[target, "PetID", "Description", "image_path", "image_exists"]
).values

combined_features = np.hstack([structured_only, text_embeddings, image_embeddings])

y = df_encoded[target]

X_train, X_test, y_train, y_test = train_test_split(
    combined_features, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
predictions_rounded = np.clip(np.round(predictions), 0, 4)

mae = mean_absolute_error(y_test, predictions)
accuracy_within_1 = (abs(y_test.values - predictions_rounded) <= 1).mean()

print("MAE (structured + text + image):", mae)
print("Accuracy within 1 category (structured + text + image):", accuracy_within_1)