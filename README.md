# Longshot — Shelter Animal Risk Prioritization

Longshot ranks shelter animals by their statistical risk of an extended stay, so shelter staff can prioritize marketing, foster outreach, or fee adjustments for the animals who need it most.

**Skills demonstrated:** Python · Survival analysis (`lifelines`, Cox proportional hazards) · pandas / NumPy / scikit-learn · Multimodal ML (Hugging Face Sentence Transformers, CLIP, PyTorch) · Streamlit · pytest · Docker · CI/CD (GitHub Actions) · Google Cloud Run · Google Artifact Registry · Kubernetes

**[Live demo (Streamlit) →](https://longshot-shelter-dashboard-hruynmeyusabpsp56kvboz.streamlit.app)**
**[Live demo (Google Cloud Run) →](https://longshot-dashboard-133354212715.us-west2.run.app)**

## The problem

Prior work on shelter length-of-stay prediction (Austin Animal Center analyses, an SMU paper on Dallas Animal Shelter, a UC Berkeley MIDS capstone) has largely stayed academic — none of it was deployed for a real shelter to use day-to-day. Longshot is built specifically to close that gap: a lightweight, real, usable tool, not just a notebook.

## Architecture

```
Austin Animal Center CSVs (intakes + outcomes, 173,000+ records, 2013–2025)
      │
      ▼
Cleaning & merge (explore.py)
  — mixed date formats, merge_asof intake→outcome matching,
    right-censoring flags for still-housed animals
      │
      ▼
Cox Proportional Hazards model (survival.py, lifelines)
  — baseline linear regression (R²=0.06) shown inadequate
  — age binned after a proportional-hazards diagnostic caught
    a violated model assumption
  — breed tested, quantified, and excluded (fairness tradeoff)
      │
      ▼
Trained model (shelter_model.pkl)
      │
      ▼
Streamlit dashboard (dashboard.py)
  — ranked priority list, plain-English reasons, CSV upload,
    shelter-wide analytics
      │
      ▼
pytest suite (17 tests) → Docker → GitHub Actions CI/CD
      │                                        │
      ▼                                        ▼
Kubernetes (Minikube, local)          Google Artifact Registry
                                                │
                                                ▼
                                       Google Cloud Run (live)
```

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
- **Containerization:** the dashboard is packaged with Docker (see `phase1_survival_model/Dockerfile`), so it runs identically in any environment.
- **CI/CD:** a GitHub Actions workflow (`.github/workflows/docker-build.yml`) automatically, on every push to `main`:
  1. Runs the full test suite
  2. Builds the Docker image
  3. Pushes it to Google Artifact Registry
  4. Deploys it to Google Cloud Run

  If any test fails, the pipeline stops immediately — nothing broken ever gets built or deployed.
- **Cloud deployment:** the containerized app runs on [Google Cloud Run](https://cloud.google.com/run), a fully managed container-hosting service, giving it a live, public, auto-scaling deployment independent of the Streamlit Community Cloud demo.
- **Container orchestration:** a local Kubernetes deployment (via Minikube) is included (`phase1_survival_model/deployment.yaml`, `service.yaml`), demonstrating orchestration — running the app under Kubernetes' management, which handles restarting and scaling the container automatically.

**A few real deployment issues worth mentioning, since they reflect genuine debugging rather than a clean first pass:**

- **Broken relative file paths on Streamlit Cloud.** `dashboard.py` loaded its model and data with relative paths (e.g., `open("shelter_model.pkl")`), which worked locally because the terminal's working directory happened to match, but failed once deployed, since Streamlit Cloud runs the app from a different working directory (the repo root, not `phase1_survival_model/`). Fixed by resolving all file paths relative to the script's own location (`os.path.dirname(os.path.abspath(__file__))`), not the working directory.
- **Docker healthcheck false failures.** The container's `HEALTHCHECK` used `curl` to confirm the app was responding — but the `python:3.11-slim` base image doesn't include `curl`, so the check itself failed even though the app worked correctly, marking the container "unhealthy." Fixed by rewriting the healthcheck in pure Python (`urllib`), removing the dependency on a tool that wasn't actually installed.
- **Cross-platform image incompatibility.** Images built locally on Apple Silicon (arm64) failed on Cloud Run, which requires amd64, with the error `Container manifest type ... must support amd64/linux`. Fixed by explicitly targeting the platform at build time (`docker build --platform linux/amd64`).
- **IAM permission gap in the deploy step.** The GitHub Actions service account had permission to push images and manage Cloud Run, but deployment still failed with `PERMISSION_DENIED: iam.serviceaccounts.actAs`, since Cloud Run also needs permission to act *as* the project's default compute service account at deploy time. Fixed by granting the deploying service account the `Service Account User` role, scoped specifically to that one runtime identity rather than broadly across the project.

## Phase 2 — Multimodal proof-of-concept (complete)

The Austin dataset has no free-text behavior notes or photos, so multimodal fusion (text + image features) couldn't be validated on it directly. Phase 2 is a focused proof-of-concept, built on the public PetFinder.my Adoption Prediction dataset (which includes real descriptions, photos, and an adoption-speed outcome label), comparing structured-only, +text, +image, and +text+image models.

**Finding:** image embeddings (CLIP) drove most of the improvement (MAE 0.99 → 0.91; accuracy-within-1 81.0% → 83.2%), with text embeddings adding comparatively little once image features were present — suggesting photos may be worth prioritizing over free-text notes if a future shelter partner can only reasonably provide one additional data type.

📁 See `phase2_multimodal_prototype/README.md` for full methodology, results, and limitations.

## Tech stack

| Layer | Tools |
|---|---|
| Modeling | Python, `lifelines` (Cox proportional hazards), scikit-learn, pandas, NumPy |
| Multimodal ML (Phase 2) | Hugging Face Sentence Transformers, CLIP (`open_clip`), PyTorch |
| Dashboard | Streamlit, Altair |
| Testing | pytest |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Cloud | Google Cloud Run, Google Artifact Registry |
| Orchestration | Kubernetes (Minikube) |

## Known limitations

- **CSV upload expects Austin's column schema** — a real shelter's export is unlikely to match this out of the box; adapting the pipeline to a specific partner's data format is the planned next step once a shelter partner is confirmed.
- **Kubernetes orchestration currently runs locally via Minikube**, not on a live managed cluster — it demonstrates real orchestration (the app runs and stays healthy under Kubernetes' management), but isn't part of the public-facing deployment; Cloud Run serves that role.
- **Phase 2's multimodal findings are validated on public data only** (PetFinder.my), not yet on a real shelter's own photos or notes — see Phase 2's own README for details.

## Status

Currently in outreach and pilot discussions with North Texas animal rescues to trial Phase 1 on real, live shelter data.

## Running locally

```bash
cd phase1_survival_model
pip install -r requirements.txt
streamlit run dashboard.py
```

### Running the test suite

```bash
cd phase1_survival_model
python -m pytest test_dashboard.py -v
```

### Running with Docker

```bash
cd phase1_survival_model
docker build -t longshot-dashboard .
docker run -p 8501:8501 longshot-dashboard
```
