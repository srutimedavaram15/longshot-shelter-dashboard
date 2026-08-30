# Phase 2 — Multimodal Proof-of-Concept

## Why this exists

Longshot's main dashboard (Phase 1) is built on Austin Animal Center's historical intake/outcome data, which has no free-text behavior notes or photos — so multimodal fusion (text + image features) can't be validated on it directly. This folder is a separate, focused proof-of-concept built on a different public dataset that *does* have text, images, and a real outcome label, to test whether multimodal features add predictive value — as groundwork for full integration once a real shelter provides richer per-animal data (photos, notes) alongside intake/outcome records.

**This is intentionally not wired into the live Longshot dashboard.** The two datasets have incompatible outcome variables (Austin: continuous days with censoring; PetFinder.my: five ordinal adoption-speed buckets, no censoring) and different populations (U.S. municipal shelter intakes vs. Malaysian rescue/foster listings) — merging them would not be statistically valid. See the main project README for the reasoning behind this separation.

## Dataset

[PetFinder.my Adoption Prediction](https://www.kaggle.com/c/petfinder-adoption-prediction) (Kaggle), 14,993 animals with:
- Structured fields (type, breed, age, gender, size, fur length, health, fee, etc.)
- A free-text `Description` written by the rescue/shelter
- Real photos (`PhotoAmt` per animal, primary photo used here)
- `AdoptionSpeed`, an ordinal outcome (0 = adopted same day, 4 = not adopted within 100 days)

## Approach

Since `AdoptionSpeed` is ordinal (0–4) rather than a plain unordered category, it was modeled as a regression target (predicting a continuous value, then rounding to the nearest valid category for evaluation) — the same instinct as Phase 1's baseline, respecting the target's ordering rather than treating all misclassifications as equally bad.

Four models were built and compared, each a simple `LinearRegression` on an 80/20 train/test split (`random_state=42`):

1. **Structured only** — type, breed (grouped, rare breeds → "Other"), age, gender, size, fur length, health, fee.
2. **Structured + Text** — adds a 384-dimensional embedding of each `Description`, generated with a pretrained sentence-transformer (`all-MiniLM-L6-v2`).
3. **Structured + Image** — adds a 512-dimensional embedding of each animal's primary photo, generated with a pretrained CLIP model (`ViT-B-32`, OpenAI weights, via `open_clip`).
4. **Structured + Text + Image** — all three feature types combined.

Evaluation used two metrics: **MAE** (average error in category-units) and **accuracy within 1 category** (a more forgiving, interpretable metric that respects the ordinal nature of the target — was the prediction within one adoption-speed bucket of the truth).

## Results

| Model | MAE | Accuracy within 1 category |
|---|---|---|
| Structured only | 0.991 | 81.0% |
| Structured + Text | 0.950 | 80.8% |
| Structured + Image | 0.910 | 83.2% |
| Structured + Text + Image | 0.901 | 83.0% |

## What this shows

**Image embeddings carried most of the multimodal signal.** Adding photo embeddings to structured features produced the largest single improvement (MAE 0.99 → 0.91; accuracy-within-1 81.0% → 83.2%). Text embeddings alone provided a smaller, more modest improvement in MAE with no real gain in accuracy-within-1. Combining all three feature types together matched — but did not meaningfully exceed — the structured+image model alone, suggesting that once image features are present, the free-text description adds little further predictive value for this dataset and this modeling approach.

This is a genuinely useful finding for prioritizing future work: if a real shelter partner can only reasonably provide one additional data type beyond structured intake fields, this result suggests **photos are likely worth prioritizing over free-text behavior notes** for improving prediction accuracy — though this should be re-validated on real shelter data before drawing firm conclusions, since PetFinder.my's population, photo quality, and description style may differ meaningfully from a specific real shelter's data.

## Known limitations

- Small proof-of-concept models (plain linear regression) — no attempt made to optimize model architecture, hyperparameters, or try alternative embedding models; the goal was a fair, simple comparison across feature types, not a state-of-the-art result.
- `AdoptionSpeed`'s ordinal buckets are a coarser signal than Phase 1's continuous, censored `length_of_stay` — this analysis doesn't use survival analysis or handle censoring, since PetFinder.my's outcome variable doesn't require it.
- Not validated on real shelter data — see main README for status.

## Running this locally

```
cd phase2_multimodal_prototype
pip install -r requirements.txt
python explore.py        # inspect the raw data
python baseline.py       # structured-only baseline
python text_model.py     # + text embeddings
python image_model.py    # + image embeddings
python combined_model.py # + both combined
```

Note: `text_model.py` and `image_model.py` each generate and cache embeddings to local `.npy` files on first run (not committed to the repo — see `.gitignore`); `combined_model.py` regenerates its own embeddings independently, since it uses a different (more restrictive) row filter than the individual stages.