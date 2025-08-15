## NFL Home Win Probability (fastai + Gradio)

End‑to‑end open demo: leak‑safe feature engineering from raw game logs → tabular neural net (fastai) → interactive Gradio app for exploration & batch inference.

---
### 1. What this repo does
- Ingests a historical NFL games dataset from CSV (`data/games.csv`) or DuckDB (`data/games.duckdb`).
- Engineers only leak‑safe, pre‑game features (no betting lines, no future info, no targets in inputs).
- Trains a fastai tabular neural network with time‑aware splits (train / validation / future test seasons).
- Exports a lightweight `Learner` (`models/winner_export.pkl`).
- Provides a polished Gradio UI (`app/gradio_app.py`) to upload any compatible CSV, run the identical feature pipeline, and view: 
	- Batch home win probabilities
	- Filtering (season, week, team, text search)
	- Single‑game probability visualization
	- Optional dataset AUC (if targets present in your uploaded set)

> Goal: A concise, reproducible template showing how to keep feature engineering leak‑safe while shipping an interactive model explorer.

---
### 2. Quickstart

```bash
# (recommended) create virtual environment first
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Place your full historical games file at data/games.csv OR data/games.duckdb
# (DuckDB: any table containing all required columns; first matching table is used.)

python train.py --epochs 25 \
	--train_end 2021 \
	--valid_season 2022 \
	--layers 400,200,100

# Launch interactive app (after training exports model):
python app/gradio_app.py
# or (module form)
python -m app.gradio_app
```

If you want a public share link, edit the last line in `gradio_app.py` (`demo.launch(..., share=True)`).

---
### 3. Data Requirements
CSV: file must contain (case‑sensitive) columns.  
DuckDB: at least one table must contain all these columns (a superset is fine):
```
season, date, game_type, week,
away_team, home_team,
away_q1, away_q2, away_q3, away_q4, away_ot, away_final,
home_q1, home_q2, home_q3, home_q4, home_ot, home_final,
away_starting_qb, home_starting_qb,
away_head_coach, home_head_coach,
pfr_game_id
```
Constraints / assumptions:
- `date` parseable to datetime.
- `season`, `week` are integers.
- One row per game (home/away pairing).
- `pfr_game_id` unique per game (acts as stable join key).

Missing any of these → `ValueError` in `load_raw()`. DuckDB loader scans tables returned by `SHOW TABLES` and picks the first with all columns.

---
### 4. Feature Engineering (Leak‑Safe)
Implemented in `src/features.py`:

| Category | Description | Notes |
|----------|-------------|-------|
| Team form | Rolling means of points for / against / margin (windows 1,3,5) | Uses `shift(1)` so current game not included |
| Rest & scheduling | Days since last game | Per team; early season may be NaN (filled) |
| QB form | Rolling team points in games started by that QB (windows 3,5) | Grouped by `(team,qb)` with prior shift |
| Continuity | Binary flags: same starting QB? same head coach? | Prior game comparison only |
| Matchup deltas | Home − Away for symmetric rolling / continuity features | Only if both sides available |
| Calendar context | Month, day of week, Thursday/Sunday flags | Derivable pre‑game |
| Target | `home_team_won` (home_final > away_final) | Not used in inference pipeline |

Constant (all‑same) continuous columns can be pruned during training (`prune_constants=True`). For inference inside the app we keep everything to maintain alignment.

NA handling:
- Categoricals → `'Unknown'`
- Continuous → `0`

No leakage validation: all rolling stats compute on values shifted by one game for each grouping key.

---
### 5. Time‑Aware Splits
`src/splits.py: time_splits(df, train_end, valid_season)`
- Train: seasons `<= train_end`
- Validation: season `== valid_season`
- Test: seasons `>= valid_season + 1`
Fallback: if insufficient seasons, a simple 70/15/15 positional split is used (logged to stdout).

Rationale: emulate forward‑looking evaluation; the model never sees future-season games during validation/test.

---
### 6. Model & Training
`train.py` highlights:
- Framework: fastai tabular (PyTorch backend)
- Architecture: configurable MLP layers (default `400,200,100`) with batchnorm, dropout automatically managed by fastai
- Metrics: `RocAucBinary`, `accuracy` (AUC ordered first; monitored by early stopping & checkpoint)
- Callbacks: `EarlyStoppingCallback(patience=5)`, `SaveModelCallback` (best on validation AUC)
- Learning rate: auto via `lr_find` (valley heuristic) unless `--lr` is provided
- Seed setting for reproducibility (Python, NumPy, Torch, CUDA)
- Exports: `models/winner_export.pkl` (serialized fastai Learner) + `models/best_model.pth` weights

CLI options:
```
--data PATH
--epochs INT
--bs INT (batch size)
--seed INT
--lr FLOAT (override autodetected)
--train_end YEAR
--valid_season YEAR
--layers 400,200,100 (comma list)
--wd FLOAT (weight decay)
```

Test evaluation: If a future test set exists, prints test ROC‑AUC after training.

---
### 7. Inference & Gradio App
`app/gradio_app.py` provides:
- Upload CSV or DuckDB file → immediate feature engineering (keeps constant columns in app)
- Batch prediction producing a table of: game id, season, week, home/away teams, home win probability
- Dynamic filters: multi‑season, multi‑week, team dropdown, free text search (game id or team substring)
- Random game selector
- Detailed single game view with a colored progress bar (blue ≥ 0.5, red < 0.5)
- Optional dataset ROC‑AUC if `home_team_won` is present in uploaded file with both classes

Column alignment: any missing categorical/continuous columns present during training are added with neutral defaults to keep the learner happy.

Launch defaults to local only (`share=False`). Set `share=True` inside `demo.launch()` for a public Gradio link.

---
### 8. Testing
`tests/test_features.py` (light smoke test) ensures feature build doesn’t crash and required outputs exist. Extend with:
- Distribution sanity checks
- Rolling window correctness
- No leakage (e.g., ensure max date used per row < row date)

Run (pytest example):
```bash
pytest -q
```

---
### 9. Project Structure
```
├── app/
│   └── gradio_app.py          # Interactive UI
├── data/
│   └── games.csv              # Your input file (not committed)
├── models/                    # Exported learner + best weights
├── src/
│   ├── features.py            # Feature engineering (long→wide)
│   └── splits.py              # Time‑based splitting
├── tests/
│   └── test_features.py       # Basic feature tests
├── train.py                   # Training CLI script
├── requirements.txt           # Python deps
└── README.md
```

---
### 10. Extending / Next Ideas
- Add calibration (Platt scaling / isotonic on validation set)
- Ensemble (blend logistic regression on engineered features + neural net)
- Feature importance (permutation on validation)
- Hyperparameter sweep (LR, layer widths, weight decay) via `wandb` or `ray[tune]`
- Add richer schedule / travel features (still pre‑game) like days since travel, distance (geocode required)
- Deploy model & app to Hugging Face Space or Docker container

---
### 11. Limitations & Disclaimers
- Intended for educational purposes; not a betting tool.
- Only uses limited historical box score level context; no player injury, weather, or betting line information.
- Performance (ROC‑AUC) will vary with the size & quality of your CSV.

---
### 12. License
This project is released under the MIT License (see `LICENSE`).

---
### 13. Acknowledgements
- [fastai](https://github.com/fastai/fastai)
- [Gradio](https://gradio.app)
- Scikit‑learn for evaluation metrics

---
### 14. Minimal Example (Programmatic)
```python
from src.features import build_features
from fastai.tabular.all import load_learner

df, cats, conts = build_features('data/games.csv')
learn = load_learner('models/winner_export.pkl')
row = df.iloc[0]
pred_class, pred_idx, probs = learn.predict(row)
print('Home win probability:', float(probs[1]))
```

---
### 15. Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: src` | Running from nested dir | Run from repo root or use `python -m app.gradio_app` |
| `CSV is missing required columns` | Schema mismatch | Add missing columns / correct header names |
| `Not enough rows to create splits` | Tiny dataset | Provide more historical games or lower season boundaries |
| AUC not printed | Single class in test set | Ensure test spans multiple seasons / classes |

Open issues / improvements welcome.

---
Enjoy exploring! If this helped you, consider a ⭐ on the repo.

