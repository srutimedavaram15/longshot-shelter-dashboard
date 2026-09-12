# Longshot — Shelter Animal Risk Prioritization

Longshot ranks shelter animals by their statistical risk of an extended stay, so shelter staff can prioritize marketing, foster outreach, or fee adjustments for the animals who need it most.

**[Live demo (Streamlit) →](https://longshot-shelter-dashboard-hruynmeyusabpsp56kvboz.streamlit.app)**
**[Live demo (Google Cloud Run) →](https://longshot-dashboard-133354212715.us-west2.run.app)**

## The problem

Prior work on shelter length-of-stay prediction (Austin Animal Center analyses, an SMU paper on Dallas Animal Shelter, a UC Berkeley MIDS capstone) has largely stayed academic — none of it was deployed for a real shelter to use day-to-day. Longshot is built specifically to close that gap: a lightweight, real, usable tool, not just a notebook.

## Phase 1 — Survival model & dashboard (complete)

Built on 173,000+ real intake/outcome records from Austin Animal Center (2013–2025):

- **Survival analysis, not naive regression.** Animals still in the shelter have censored (unknown) length of stay — a plain regression model can't use this data correctly. A Cox proportional hazards model (`lifelines`) handles it properly.
- **Baseline comparison.** A simple linear regression baseline achieved R²=0.06, with a documented, explainable disagreement between MAE and R² driven by the data's heavy right-skew (median stay: 6 days; max: 1,912 days) — motivating the move to survival analysis.
- **Task-appropriate evaluation.** Concordance (0.60) and NDCG (0.73) on a held-out test set — NDCG specifically measures the model's ability to rank the highest-risk animals correctly, which is what the dashboard's actual use case requires.
- **Quantified fairness analysis.** Breed was tested as a feature: it added only ~0.02 concordance, while showing hazard ratios consistent with known adoption bias against certain breeds (e.g., Pit Bull–type breeds at 0.43–0.56x baseline hazard). Breed was excluded from the deployed model based on this cost/benefit analysis, documented rather than silently decided.
- **Diagnosed and fixed a violated model assumption.** A proportional hazards diagnostic flagged a large violation on age when treated as a continuous variable (test statistic of 682) — binning age into four groups (Juvenile, Young, Adult, Senior) resolved it, confirmed by re-running the diagnostic.
- **Interactive dashboard** (Streamlit): a filterable, ranked priority list with an automatically generated plain-English reason per animal, a shelter-wide analytics view, and CSV upload support so a real shelter can score their own currently-housed animals.

📁 See `phase1_survival_model/` for all code.

**Known limitation:** the CSV upload feature expects a specific column schema (matching the Austin dataset's structure). A real shelter's export is unlikely to match this exactly out of the box — adapting the pipeline to a specific shelter's real data format is planned as the next step once a shelter partner is confirmed.

## Infrastructure & Deployment

Longshot is containerized and deployed with a full CI/CD pipeline, and includes automated testing and container orchestration.

- **Testing:** 17 pytest unit tests cover the dashboard's core logic — plain-English reason generation, majority-category detection, and CSV column validation — run automatically on every push, before any build or deploy step.
- **Containerization:** the dashboard is packaged with Docker (see `phase1_survival_model/Dockerfile`), so it runs identically in any environment. Debugged a real cross-platform issue along the way: images built natively on Apple Silicon (arm64) failed on Cloud Run, which requires amd64 — fixed by explicitly targeting the platform at build time.
- **CI/CD:** a GitHub Actions workflow (`.github/workflows/docker-build.yml`) automatically, on every push to `main`:
  1. Runs the full test suite
  2. Builds the Docker image
  3. Pushes it to Google Artifact Registry
  4. Deploys it to Google Cloud Run

  If any test fails, the pipeline stops immediately — nothing broken ever gets built or deployed.
- **Cloud deployment:** the containerized app runs on [Google Cloud Run](https://cloud.google.com/run), a fully managed container-hosting service, giving it a live, public, auto-scaling deployment independent of the Streamlit Community Cloud demo.
- **Container orchestration:** a local Kubernetes deployment (via Minikube) is included (`phase1_survival_model/deployment.yaml`, `service.yaml`), demonstrating orchestration — running the app under Kubernetes' management, which handles restarting and scaling the container automatically.

## Phase 2 — Multimodal proof-of-concept (complete)

The Austin dataset has no free-text behavior notes or photos, so multimodal fusion (text + image features) couldn't be validated on it directly. Phase 2 is a focused proof-of-concept, built on the public PetFinder.my Adoption Prediction dataset (which includes real descriptions, photos, and an adoption-speed outcome label), comparing structured-only, +text, +image, and +text+image models.

**Finding:** image embeddings (CLIP) drove most of the improvement (MAE 0.99 → 0.91; accuracy-within-1 81.0% → 83.2%), with text embeddings adding comparatively little once image features were present — suggesting photos may be worth prioritizing over free-text notes if a future shelter partner can only reasonably provide one additional data type.

📁 See `phase2_multimodal_prototype/README.md` for full methodology, results, and limitations.

## Status

Currently in outreach and pilot discussions with North Texas animal rescues to trial Phase 1 on real, live shelter data.

## Running locally

```
cd phase1_survival_model
pip install -r requirements.txt
streamlit run dashboard.py
```

### Running the test suite

```
cd phase1_survival_model
python -m pytest test_dashboard.py -v
```

### Running with Docker

```
cd phase1_survival_model
docker build -t longshot-dashboard .
docker run -p 8501:8501 longshot-dashboard
```
