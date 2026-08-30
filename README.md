# Longshot — Shelter Animal Risk Prioritization

Longshot ranks shelter animals by their statistical risk of an extended stay, so shelter staff can prioritize marketing, foster outreach, or fee adjustments for the animals who need it most.

**[Live demo →](https://longshot-shelter-dashboard-hruynmeyusabpsp56kvboz.streamlit.app)**

## The problem

Prior work on shelter length-of-stay prediction (Austin Animal Center analyses, an SMU paper on Dallas Animal Shelter, a UC Berkeley MIDS capstone) has largely stayed academic — none of it was deployed for a real shelter to use day-to-day. Longshot is built specifically to close that gap: a lightweight, real, usable tool, not just a notebook.

## Phase 1 — Survival model & dashboard (complete)

Built on 173,000+ real intake/outcome records from Austin Animal Center (2013–2025):

- **Survival analysis, not naive regression.** Animals still in the shelter have censored (unknown) length of stay — a plain regression model can't use this data correctly. A Cox proportional hazards model (`lifelines`) handles it properly.
- **Baseline comparison.** A simple linear regression baseline achieved R²=0.06, with a documented, explainable disagreement between MAE and R² driven by the data's heavy right-skew (median stay: 6 days; max: 1,912 days) — motivating the move to survival analysis.
- **Task-appropriate evaluation.** Concordance (0.62) and NDCG (0.73) on a held-out test set — NDCG specifically measures the model's ability to rank the highest-risk animals correctly, which is what the dashboard's actual use case requires.
- **Quantified fairness analysis.** Breed was tested as a feature: it added only ~0.02 concordance, while showing hazard ratios consistent with known adoption bias against certain breeds (e.g., Pit Bull–type breeds at 0.43–0.56x baseline hazard). Breed was excluded from the deployed model based on this cost/benefit analysis, documented rather than silently decided.
- **Interactive dashboard** (Streamlit): a filterable, ranked priority list with an automatically generated plain-English reason per animal, a shelter-wide analytics view, and CSV upload support so a real shelter can score their own currently-housed animals.

📁 See `phase1_survival_model/` for all code.

**Known limitation:** the CSV upload feature expects a specific column schema (matching the Austin dataset's structure). A real shelter's export is unlikely to match this exactly out of the box — adapting the pipeline to a specific shelter's real data format is planned as the next step once a shelter partner is confirmed.

## Phase 2 — Multimodal proof-of-concept (planned)

The Austin dataset has no free-text behavior notes or photos, so multimodal fusion (text + image features) can't be validated on it directly. Phase 2 will be a focused proof-of-concept, built on the public **PetFinder.my Adoption Prediction** dataset (which includes real descriptions, photos, and an adoption-speed outcome label), to validate that text and image embeddings add predictive value — as groundwork for full integration once a real shelter provides richer per-animal data.

📁 See `phase2_multimodal_prototype/` for details (in progress).

## Status

Currently in outreach and pilot discussions with North Texas animal rescues to trial Phase 1 on real, live shelter data.

## Running locally

```
cd phase1_survival_model
pip install -r requirements.txt
streamlit run dashboard.py
```