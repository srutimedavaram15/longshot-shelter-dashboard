import pandas as pd
from lifelines import CoxPHFitter
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/cleaned_length_of_stay.csv")

print(df.shape)
print(df["is_censored"].sum())

df["event_observed"] = (~df["is_censored"]).astype(int)

print(df["event_observed"].value_counts())

print(df["length_of_stay"].isna().sum())

df["intake_date"] = pd.to_datetime(df["intake_date"], utc=True)

reference_date = df["intake_date"].max()
print("Reference date:", reference_date)

df["length_of_stay"] = df["length_of_stay"].fillna(
    (reference_date - df["intake_date"]).dt.days
)

print(df["length_of_stay"].isna().sum())

def parse_age_to_years(age_str):
    if pd.isna(age_str):
        return None
    parts = age_str.split()
    if len(parts) != 2:
        return None
    number, unit = parts
    number = float(number)
    unit = unit.lower()
    if "year" in unit:
        return number
    elif "month" in unit:
        return number / 12
    elif "week" in unit:
        return number / 52
    elif "day" in unit:
        return number / 365
    else:
        return None

df["age_years"] = df["Age upon Intake"].apply(parse_age_to_years)

print(df["age_years"].describe())
print(df["age_years"].isna().sum())

# Drop rows with impossible negative ages (data entry errors in the source data)
df = df[df["age_years"] >= 0]

print(df.shape)

# Bin age into life-stage categories instead of using it as a raw continuous
# number. The proportional-hazards check showed age_years likely has a
# non-linear relationship with hazard, so binning avoids forcing a straight-line
# effect and lets each life stage have its own hazard ratio.
age_bins = [-0.01, 0.5, 2, 7, 30]
age_labels = ["Puppy/Kitten (0-6mo)", "Young (6mo-2yr)", "Adult (2-7yr)", "Senior (7yr+)"]
df["age_group"] = pd.cut(df["age_years"], bins=age_bins, labels=age_labels)

print(df["age_group"].value_counts())

# Group rare breeds into "Other" so one-hot encoding doesn't create thousands
# of columns with almost no examples each
breed_counts = df["Breed"].value_counts()
common_breeds = breed_counts[breed_counts >= 200].index
df["Breed_grouped"] = df["Breed"].where(df["Breed"].isin(common_breeds), "Other")

# Same idea for intake condition, to avoid the low-variance convergence warning
condition_counts = df["Intake Condition"].value_counts()
common_conditions = condition_counts[condition_counts >= 200].index
df["Condition_grouped"] = df["Intake Condition"].where(df["Intake Condition"].isin(common_conditions), "Other")

# ----- Train/test split -----
# Everything up to this point has fit models on the FULL dataset. To honestly
# evaluate ranking quality (NDCG) we need a held-out test set the model never
# saw during training, same principle as the baseline's train_test_split.
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print(train_df.shape, test_df.shape)

# ----- Model WITH breed (fit on full data - kept as the bias-analysis artifact, not deployed) -----
categorical_features = ["Intake Type", "Condition_grouped", "Animal Type", "Sex upon Intake", "age_group", "Breed_grouped"]

df_model = df[["length_of_stay", "event_observed"] + categorical_features].dropna()

df_model_encoded = pd.get_dummies(df_model, columns=categorical_features, drop_first=True)

print(df_model_encoded.shape)

cph = CoxPHFitter()
cph.fit(df_model_encoded, duration_col="length_of_stay", event_col="event_observed")

cph.print_summary()

# ----- Model WITHOUT breed, fit on TRAIN only (this is the deployable model) -----
categorical_features_no_breed = ["Intake Type", "Condition_grouped", "Animal Type", "Sex upon Intake", "age_group"]

df_model_no_breed = train_df[["length_of_stay", "event_observed"] + categorical_features_no_breed].dropna()

df_model_no_breed_encoded = pd.get_dummies(df_model_no_breed, columns=categorical_features_no_breed, drop_first=True)

print(df_model_no_breed_encoded.shape)

cph_no_breed = CoxPHFitter()
cph_no_breed.fit(df_model_no_breed_encoded, duration_col="length_of_stay", event_col="event_observed")

print("Concordance WITH breed (fit on full data):", cph.concordance_index_)
print("Concordance WITHOUT breed (fit on train only):", cph_no_breed.concordance_index_)

# Re-check proportional hazards assumption now that age is binned instead of continuous
cph_no_breed.check_assumptions(df_model_no_breed_encoded, p_value_threshold=0.05)

from sklearn.metrics import ndcg_score
import numpy as np

# Encode the test set the same way as training, using the same dummy columns
df_test_model = test_df[["length_of_stay", "event_observed"] + categorical_features_no_breed].dropna()
df_test_encoded = pd.get_dummies(df_test_model, columns=categorical_features_no_breed, drop_first=True)

# Align columns: the test set might not contain every category the training set saw
# (or vice versa), so we reindex to match the training columns exactly, filling
# any missing dummy columns with 0
df_test_encoded = df_test_encoded.reindex(columns=df_model_no_breed_encoded.columns, fill_value=0)

print(df_test_encoded.shape)

# For NDCG we need real, non-censored outcomes as ground truth
eval_mask = df_test_encoded.index.isin(
    test_df[test_df["event_observed"] == 1].index
)
eval_set = df_test_encoded[eval_mask]

print(eval_set.shape)

risk_scores = cph_no_breed.predict_partial_hazard(eval_set)

true_length_of_stay = test_df.loc[eval_set.index, "length_of_stay"].values

y_true = true_length_of_stay.reshape(1, -1)
y_score = (-risk_scores.values).reshape(1, -1)

ndcg = ndcg_score(y_true, y_score)

print("NDCG:", ndcg)

import pickle

with open("shelter_model.pkl", "wb") as f:
    pickle.dump({
        "model": cph_no_breed,
        "feature_columns": df_model_no_breed_encoded.columns.tolist(),
        "categorical_features": categorical_features_no_breed
    }, f)

print("Model saved to shelter_model.pkl")