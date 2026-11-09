# VajraAI

Anomaly flagging for public-spending transactions. Users upload procurement or payment records (CSV/JSON), the backend computes per-vendor features, a PyTorch TabularTransformer scores each transaction, and a rule engine attaches plain-language reasons ("cost overrun 80% above estimate", "single bidder") so an auditor knows where to look first.

Built as a 4-person team project for Hack4Delhi (team: Vineeta Kumari Meena, `nnryn`, `Rahul69-aiiy`, Aadvik Chaturvedi). This repo is Aadvik's copy, with later model work added on top. See [My contributions](#my-contributions).

## Architecture

```mermaid
flowchart LR
    C[React + Vite client<br/>:5173] -->|JWT, REST| S[Express server<br/>:3000]
    S <--> DB[(PostgreSQL<br/>transactions, anomaly_results,<br/>vendors, users)]
    S -->|POST /detect| M[FastAPI ML service<br/>:8000]
    M --> T[TabularTransformer<br/>ieee_transformer.pt]
    S --> R[Rule engine<br/>reasons per transaction]
```

Request flow for `POST /api/analyze`:

1. Load all rows from `transactions`.
2. `featureEngineering.js` computes per-vendor `frequency` and `avg_amount`, and passes through `estimated_cost`, `num_bidders`, department, location, vendor.
3. `mlService.js` posts the batch to the ML service's `/detect`.
4. The ML service builds the same feature pipeline used in training (time features, per-card time deltas, rolling 1h/6h/24h velocity counts and sums, frequency-encoded categoricals, scaled numerics), runs the transformer and returns a sigmoid `anomaly_score` per row. Risk level is relative to the batch: above the 95th percentile is `HIGH`, above the 80th is `MEDIUM`, otherwise `LOW`. Rows with more than half the core fields missing are marked `LOW_CONFIDENCE`.
5. `ruleEngine.js` adds reasons: overspend ratio > 0.5 against `estimated_cost`, a single bidder, amount > 2.5× or 4× the vendor average, amounts ≥ ₹50 lakh, payment clustering (frequency ≥ 4 or ≥ 7), plus department/location context.
6. Results are written to `anomaly_results` and shown on the dashboard, anomaly list/detail, vendor pages and a Leaflet risk heatmap.

## The model

Code: `ml-service/train_ieee.py` (training), `ml-service/ml_transformer_service.py` (serving), `ml-service/dtypes.py` (feature pipeline). Details in [`ml-service/models/model_card.md`](ml-service/models/model_card.md).

It is a **supervised** binary classifier trained on the labelled [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/data) dataset (`isFraud`). Public-sector procurement data with fraud labels was not available, so IEEE-CIS stands in for it.

| | |
|---|---|
| Architecture | Dynamic-width embeddings per categorical (4–64 dims by cardinality) → projected to one token; numerics projected to a second token; a CLS token; 4-layer Transformer encoder (d_model 128, 8 heads, d_ff 512, dropout 0.15) |
| Head | Residual concat `[CLS output, raw categorical embeddings, raw numerics]` → MLP → logit |
| Size | 827,473 parameters |
| Features | 449 numeric (incl. 18 velocity features and 35 identity-table missingness indicators) + 37 categorical |
| Loss / optim | Focal loss (α 0.25, γ 2), AdamW (lr 1e-3, wd 1e-5), cosine annealing, 12 epochs |
| Imbalance handling | Hard-negative mining: after each epoch the top 5% highest-scoring legitimate transactions get 5× sampling weight (`WeightedRandomSampler`) |
| Regularisation | Random masking of 10% of categorical inputs (missing-value augmentation) |
| Validation | Expanding-window CV sorted by `TransactionDT`, 3 folds, so no fold trains on the future |

### Results (documented run: first 20,000 rows, fraud rate 2.81%)

From `model_card.md` and the per-epoch log committed in `1f789182`:

| Fold | Train rows | ROC-AUC | PR-AUC |
|---|---|---|---|
| 1 | 5,000 | 0.7035 | 0.0887 |
| 2 | 10,000 | 0.7611 | 0.1392 |
| 3 | 15,000 | **0.8071** | 0.2159 |
| Mean | | 0.7573 | 0.1479 |

ROC-AUC rises with training size across folds. Hard-negative mining lowered the score of the hardest legitimate transactions in every fold (fold 1: max negative probability 0.521 → 0.111), and fold 3's PR-AUC reached 0.2821 by epoch 12.

The `training_metrics.{txt,json,csv}` files were later replaced with a summary of a full 590,540-row run (fold ROC-AUC 0.8388 / 0.8594 / 0.8587). Those files were not written by `train_ieee.py`, have no per-epoch log, and list the same 2.81% fraud rate as the 20K subset (the model card gives 3.5% for the full dataset). Re-run `./run_train.sh` and keep the console log before quoting them.

## Repository layout

```
VajraAI/
├── client/        React 19 + Vite + Tailwind, Recharts, React-Leaflet
├── server/        Express 5, pg, JWT auth, bcrypt
│   ├── routes/    auth, upload, analyze, dashboard, anomalies, heatmap, vendors
│   └── services/  featureEngineering, mlService, ruleEngine
├── ml-service/    FastAPI + PyTorch
│   ├── train_ieee.py, run_train.sh, dtypes.py
│   ├── ml_transformer_service.py
│   └── models/    weights, artifact.pkl, scaler.pkl, model card, metrics
├── ieee-fraud-detection/   identity CSVs (transaction CSVs are gitignored)
├── dummy.csv, dummy.json   sample upload data
└── render.yaml, DEPLOYMENT.md
```

## API

**Express server** (`server/server.js`)

| Endpoint | Description |
|---|---|
| `POST /api/auth/register`, `POST /api/auth/login` | Accounts, JWT |
| `POST /api/upload` | Insert an array of transactions |
| `POST /api/analyze` | Run ML + rules over all transactions |
| `GET /api/dashboard/overview`, `/risk-distribution`, `/top-vendors`, `/recent-anomalies`, `/department-stats` | Dashboard data |
| `GET /api/anomalies`, `GET /api/anomalies/:id` | Flagged transactions with reasons |
| `GET /api/heatmap/location`, `/department`, `/time` | Aggregations for the heatmap |
| `GET /api/vendors`, `GET /api/vendors/:id` | Vendor profiles |

**ML service** (`ml-service/ml_transformer_service.py`)

| Endpoint | Description |
|---|---|
| `GET /health` | `{status, model_loaded, device}` |
| `POST /detect` | List of transactions → `[{transaction_id, anomaly_score, risk_level}]` |

## Running locally

Needs Node.js 16+, Python 3.9+ and PostgreSQL. Run each service in its own terminal.

**Database + server**

```bash
cd server
npm install
cp .env.sample .env      # fill in your own values
node db/dbSetup.js       # creates tables
npm run dev              # http://localhost:3000
```

| Variable | Example |
|---|---|
| `PORT` | `3000` |
| `DB_CONNECTION_STRING` | `postgresql://username:password@localhost:5432/hack4delhi` |
| `JWT_SECRET` | any long random string |
| `JWT_EXPIRES_IN` | `24h` |
| `ML_SERVICE_URL` | `http://localhost:8000/detect` |

**ML service**

```bash
cd ml-service
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn ml_transformer_service:app --host 0.0.0.0 --port 8000 --reload
```

Trained weights are committed under `ml-service/models/`. If `ieee_transformer.pt` is missing, the service still starts and returns `LOW_CONFIDENCE` for every row.

**Client**

```bash
cd client
npm install
npm run dev              # http://localhost:5173, set VITE_API_URL if the server isn't on :3000
```

Then register, upload `dummy.csv` (include `estimated_cost` and `num_bidders` columns to get the procurement rules), click **Start Analysis** on the Analyze page, and look at Dashboard / Anomalies / Vendors.

**Retraining**

Download `train_transaction.csv` from Kaggle into `ieee-fraud-detection/` (it is ~652 MB and not tracked), then:

```bash
cd ml-service
./run_train.sh --subset 20000   # omit --subset for all 590,540 rows
```

**Deployment.** `render.yaml` defines the Postgres database, the ML service and the Node server; `DEPLOYMENT.md` covers the Vercel client. Add `ML_SERVICE_URL` to the server's Render env vars, because the blueprint does not set it.

## My contributions

From `git log --author=aadvik` (11 of 72 commits).

- **First anomaly model** (`9a62ffc6`, Jan 2026): "GPFT", a Transformer autoencoder over agency/vendor/contract-type embeddings, amount and a text description, trained by reconstruction on 5,000 synthetic public-sector rows from a generator script, with a FastAPI endpoint, Streamlit stress-test UI and Dockerfile. It lived in `Fraud_detection/` and was later dropped from the tree during the frontend restructure (`9aa58cda`).
- **Procurement and bidding logic** (`f72844ce`): added `estimated_cost` and `num_bidders` to the ML service input, plus cost-variance and competition features and a single-bidder override. The server-side overspend and single-bidder rules build on these fields.
- **IEEE-CIS TabularTransformer** (`1f789182`, `a653578c`, `936d4493`, Jul 2026): `train_ieee.py`, the feature pipeline in `dtypes.py` (velocity, time-delta, identity-table merge with missingness indicators, frequency encoder), focal loss, hard-negative mining, expanding-window CV, model card. Also rewrote `ml_transformer_service.py` to serve the trained model with the same pipeline, added `/health`, the missing-field guardrail and schema assertions, and updated `analyze.js` / `featureEngineering.js` to pass the new fields.
- Repo hygiene: dataset link, `.gitignore` for the large CSVs, removed committed `__pycache__`.

Other team members built the React client and its earlier prototypes (Vineeta, Rahul, nnryn), and the Express server, auth, database schema, dashboard/vendor/heatmap routes and Render config (nnryn, Vineeta).

## Limitations / next steps

- **Domain gap.** The model learns card-payment fraud patterns from IEEE-CIS. In the app, government transactions arrive without any IEEE fields (`card1`, `addr1`, `M1`–`M9`, identity columns), and those are filled with defaults. Treat its score on procurement data as a generic outlier signal until it is retrained on labelled public-sector data.
- **Synthetic app fields.** During training, `department`, `location` and `vendor_id` are random values (`add_synthetic_app_fields`), so the model has learned nothing real about them.
- **Relative risk levels.** HIGH/MEDIUM come from percentiles of the current batch, so about 5% of any upload is labelled HIGH even when nothing is wrong.
- **Optimistic metrics.** Each fold reports its best epoch on the validation fold itself, and there is no held-out test split. Fold 1 peaked at epoch 1 and fell to 0.573 by epoch 12.
- **Auth.** `authMiddleware` is imported in `server.js` but not applied to the API routes, so only the client-side `ProtectedRoute` guards them.
- `/api/analyze` re-scores the whole `transactions` table and clears `anomaly_results` on every run.
