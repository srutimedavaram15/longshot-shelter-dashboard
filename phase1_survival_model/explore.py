import pandas as pd

intakes = pd.read_csv("data/intakes.csv")
outcomes = pd.read_csv("data/outcomes.csv")

intakes = intakes.rename(columns={"DateTime": "intake_date"})
outcomes = outcomes.rename(columns={"DateTime": "outcome_date"})

intakes["intake_date"] = pd.to_datetime(intakes["intake_date"], format="mixed", utc=True)
outcomes["outcome_date"] = pd.to_datetime(outcomes["outcome_date"], format="mixed", utc=True)

intakes_sorted = intakes.sort_values("intake_date")
outcomes_sorted = outcomes.sort_values("outcome_date")

merged = pd.merge_asof(
    intakes_sorted,
    outcomes_sorted[["Animal ID", "outcome_date"]],
    left_on="intake_date",
    right_on="outcome_date",
    by="Animal ID",
    direction="forward"
)

print(merged.shape)
print(merged[["Animal ID", "intake_date", "outcome_date"]].head(10))
print(merged["outcome_date"].isna().sum())

merged["length_of_stay"] = (merged["outcome_date"] - merged["intake_date"]).dt.days
merged["is_censored"] = merged["outcome_date"].isna()

print(merged["length_of_stay"].describe())
print((merged["length_of_stay"] < 0).sum())

merged.to_csv("data/cleaned_length_of_stay.csv", index=False)

import pandas as pd

intakes = pd.read_csv("data/intakes.csv")
outcomes = pd.read_csv("data/outcomes.csv")

print(intakes.columns.tolist())
print(outcomes.columns.tolist())