import pandas as pd

df = pd.read_csv("data/train.csv")

print(df.shape)
print(df.columns.tolist())
print(df[["Description", "AdoptionSpeed", "PhotoAmt", "Type", "Breed1"]].head())
print(df["AdoptionSpeed"].value_counts().sort_index())

print(df["Description"].isna().sum())
print(df["PhotoAmt"].value_counts().sort_index())
print(df["Type"].value_counts())

df["Type"] = df["Type"].map({1: "Dog", 2: "Cat"})

breed_labels = pd.read_csv("data/BreedLabels.csv")
breed_map = dict(zip(breed_labels["BreedID"], breed_labels["BreedName"]))
df["Breed1_name"] = df["Breed1"].map(breed_map)

print(df["Type"].value_counts())
print(df["Breed1_name"].value_counts().head(10))
print(df["Breed1_name"].isna().sum())
