"""Interactive Gradio application for NFL game win probability.

Features:
  * Upload a games CSV (same schema as data/games.csv).
  * Automatic feature engineering (same pipeline as training).
  * Batch prediction for all uploaded games (home win probability).
  * Rich filtering (season, week, teams, text search).
  * Inline probability bar & sortable table.
  * Single game detail view with pretty gauge.

Run: python app/gradio_app.py
"""

from __future__ import annotations
import gradio as gr
import pandas as pd
import numpy as np
import sys, pathlib, warnings, random
from typing import Any, Dict
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
from fastai.tabular.all import load_learner  # lean import

# Ensure project root is importable
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import build_features  # type: ignore

MODEL_PATH = "models/winner_export.pkl"

# Global caches
STATE: Dict[str, Any] = {
    "features": None,      # engineered features DataFrame
    "preds": None,         # DataFrame with probabilities
    "metrics_md": "",     # summary markdown
    "model_loaded": False,
    "cat_cols": [],
    "cont_cols": []
}


def load_model():
    try:
        learn = load_learner(MODEL_PATH)
        return learn, None
    except Exception as e:  # pragma: no cover
        return None, f"Model load failed: {e}. Train first: python train.py"


def align_columns(df: pd.DataFrame, learn) -> pd.DataFrame:
    """Return copy of df with all learner columns present (missing filled)."""
    df2 = df.copy()
    cat_names = learn.dls.cat_names
    cont_names = learn.dls.cont_names
    for c in cat_names:
        if c not in df2.columns:
            df2[c] = 'Unknown'
        df2[c] = df2[c].fillna('Unknown')
    for c in cont_names:
        if c not in df2.columns:
            df2[c] = 0
        df2[c] = df2[c].fillna(0)
    # Ensure ordering: keep original id/target columns if present first
    ordered = []
    for base in ["season","date","week","game_type","home_team","away_team","home_starting_qb","away_starting_qb","home_head_coach","away_head_coach","pfr_game_id","home_team_won"]:
        if base in df2.columns: ordered.append(base)
    ordered += [c for c in cat_names if c not in ordered]
    ordered += [c for c in cont_names if c not in ordered]
    ordered = [c for c in ordered if c in df2.columns]
    df2 = df2[ordered]
    return df2


def render_prob_bar(prob: float) -> str:
    pct = prob * 100
    color = '#2b8cbe' if prob >= 0.5 else '#b2182b'
    bar = f"<div style='background:#eee;width:100%;border-radius:6px;height:22px;'><div style='background:{color};height:22px;width:{pct:.1f}%;border-radius:6px'></div></div>"
    txt = f"<div style='font-size:14px;margin-top:4px'>Home win probability: <b>{prob:.3f}</b></div>"
    return bar + txt


def process_upload(file):  # outputs summary_md, error_md, table, seasons, weeks, teams
    if file is None:
        STATE.update({"features": None, "preds": None, "metrics_md": ""})
        return "", "No file uploaded.", pd.DataFrame(), [], [], []
    learn, err = load_model()
    if err:
        return "", err, pd.DataFrame(), [], [], []
    # Engineer features (retain constants for alignment) and batch predict
    feats, cat_names, cont_names = build_features(file.name, prune_constants=False)  # type: ignore[arg-type]
    aligned = align_columns(feats, learn)
    test_dl = learn.dls.test_dl(aligned)
    preds, _ = learn.get_preds(dl=test_dl)
    if preds.ndim == 2 and preds.shape[1] == 2:
        p_home = preds[:, 1].numpy()
    else:
        p_home = preds.squeeze().numpy()
    pred_df = feats[["pfr_game_id","season","week","home_team","away_team"]].copy()
    pred_df["home_win_prob"] = p_home
    # Compute ROC-AUC if target present & both classes
    auc_md = ""
    if "home_team_won" in feats.columns and feats["home_team_won"].nunique() == 2:
        try:
            auc = roc_auc_score(feats["home_team_won"], p_home)
            auc_md = f"Model AUC on uploaded games: **{auc:.4f}**"
        except Exception:
            pass
    summary = (
        f"**Rows:** {len(feats)}  |  **Seasons:** {feats['season'].nunique()}  |  "
        f"Range: {feats['season'].min()}–{feats['season'].max()}  |  "
        f"Weeks: {feats['week'].min()}–{feats['week'].max()}" + (f"  |  {auc_md}" if auc_md else "")
    )
    STATE.update({"features": feats, "preds": pred_df, "metrics_md": summary, "cat_cols": cat_names, "cont_cols": cont_names})
    seasons = [str(s) for s in sorted(pred_df['season'].unique())]
    weeks = [str(w) for w in sorted(pred_df['week'].unique())]
    teams = sorted(pd.unique(pred_df[['home_team','away_team']].values.ravel('K')))
    return summary, "", pred_df, gr.update(choices=seasons, value=[]), gr.update(choices=weeks, value=[]), gr.update(choices=teams, value=None)


def filter_table(seasons, weeks, team, search_text):
    df = STATE.get("preds")
    if df is None or df.empty:
        return pd.DataFrame(), gr.update(choices=[], value=None)
    view = df.copy()
    # incoming seasons/weeks are strings from CheckboxGroup
    if seasons:
        seasons_int = {int(s) for s in seasons}
        view = view[view.season.isin(seasons_int)]
    if weeks:
        weeks_int = {int(w) for w in weeks}
        view = view[view.week.isin(weeks_int)]
    if team:
        view = view[(view.home_team == team) | (view.away_team == team)]
    if search_text:
        st = search_text.lower()
        mask = view.pfr_game_id.str.lower().str.contains(st) | view.home_team.str.lower().str.contains(st) | view.away_team.str.lower().str.contains(st)
        view = view[mask]
    view = view.sort_values("home_win_prob", ascending=False)
    game_ids = view.pfr_game_id.tolist()
    return view, gr.update(choices=game_ids, value=(game_ids[0] if game_ids else None))


def pick_random_game(current_df: pd.DataFrame):
    if current_df is None or current_df.empty:
        return gr.update(value=None)
    return random.choice(current_df.pfr_game_id.tolist())


def show_game(game_id: str | None):
    preds = STATE.get("preds")
    if game_id is None or not game_id or preds is None or preds.empty:
        return "Select a game.", ""
    if game_id not in set(preds.pfr_game_id):
        return "Filtered list changed; pick another game.", ""
    row = preds.loc[preds.pfr_game_id == game_id]
    if row.empty:
        return "Game not found.", ""
    r = row.iloc[0]
    prob = r.home_win_prob
    md = f"### {r.home_team} vs {r.away_team} (Season {r.season} Week {r.week})\n**Game ID:** `{r.pfr_game_id}`"
    html = render_prob_bar(float(prob))
    return md, html


theme = gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.slate).set(
    body_background_fill="#f9fafb", block_border_width="1px"
)

with gr.Blocks(theme=theme, title="NFL Win Probability") as demo:
    gr.HTML("""<div style='display:flex;align-items:center;gap:12px'>
    <h1 style='margin:0'>NFL Home Win Probability</h1>
    <span style='font-size:14px;color:#555'>FastAI Tabular Neural Net</span>
    </div>""")
    gr.Markdown("Upload a games CSV to generate features & model probabilities. Filter and inspect individual matchups.")
    with gr.Row():
        with gr.Column(scale=1):
            file_in = gr.File(label="1. Upload games CSV", file_types=[".csv"])
            summary_md = gr.Markdown(label="Dataset Summary")
            error_md = gr.Markdown(visible=False)
            season_filter = gr.CheckboxGroup(label="Season(s)")
            week_filter = gr.CheckboxGroup(label="Week(s)")
            team_filter = gr.Dropdown(label="Team (either side)", choices=[], value=None)
            search_box = gr.Textbox(label="Search (game id / team)")
            apply_filters = gr.Button("Apply Filters")
            random_btn = gr.Button("Random Game")
        with gr.Column(scale=2):
            games_df = gr.Dataframe(headers=["pfr_game_id","season","week","home_team","away_team","home_win_prob"], interactive=False)
            game_select = gr.Dropdown(label="Select Game", choices=[], value=None, allow_custom_value=True)
            game_md = gr.Markdown()
            prob_html = gr.HTML()
    gr.Markdown("---\nBuilt with fastai + gradio. Probability represents model's pre-game home win chance.")

    # Events
    file_in.change(
        process_upload,
        inputs=file_in,
        outputs=[summary_md, error_md, games_df, season_filter, week_filter, team_filter],
    )

    apply_filters.click(
        filter_table,
        inputs=[season_filter, week_filter, team_filter, search_box],
        outputs=[games_df, game_select],
    )
    search_box.submit(
        filter_table,
        inputs=[season_filter, week_filter, team_filter, search_box],
        outputs=[games_df, game_select],
    )
    season_filter.change(filter_table, [season_filter, week_filter, team_filter, search_box], [games_df, game_select])
    week_filter.change(filter_table, [season_filter, week_filter, team_filter, search_box], [games_df, game_select])
    team_filter.change(filter_table, [season_filter, week_filter, team_filter, search_box], [games_df, game_select])
    random_btn.click(lambda: pick_random_game(STATE.get("preds")), outputs=game_select)
    game_select.change(show_game, inputs=game_select, outputs=[game_md, prob_html])

if __name__ == "__main__":  # pragma: no cover
    demo.launch(server_name="0.0.0.0", share=False)