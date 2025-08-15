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
from pathlib import Path
from types import SimpleNamespace

# Ensure project root is importable
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import build_features, load_raw  # type: ignore

ALL_TEAMS_LABEL = "All Teams"

MODEL_PATH = "models/winner_export.pkl"
# Prefer DuckDB snapshot if present, else fallback to CSV
DEFAULT_DUCKDB = Path(ROOT / "data" / "games.duckdb")
DEFAULT_CSV = Path(ROOT / "data" / "games.csv")
DEFAULT_DATA_PATH = DEFAULT_DUCKDB if DEFAULT_DUCKDB.exists() else DEFAULT_CSV

# Global caches
STATE: Dict[str, Any] = {
    "features": None,      # engineered features DataFrame
    "preds": None,         # DataFrame with probabilities
    "metrics_md": "",     # summary markdown
    "model_loaded": False,
    "cat_cols": [],
    "cont_cols": [],
    "last_view": None,     # currently filtered view shown in table
    "raw": None            # original raw games (quarters & finals)
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
    # Higher contrast palette (indigo for >=0.5, deep red for <0.5)
    color = '#1e3a8a' if prob >= 0.5 else '#c62828'
    bar = (
        "<div style='background:#e2e8f0;width:100%;border-radius:6px;height:22px;overflow:hidden;'>"
        f"<div style='background:{color};height:22px;width:{pct:.1f}%;transition:width .4s'></div>"
        "</div>"
    )
    txt = f"<div style='font-size:14px;margin-top:4px'>Home win probability: <b>{prob:.3f}</b></div>"
    return bar + txt


def _overall_predictions_summary() -> str:
    preds = STATE.get("preds")
    if preds is None or preds.empty:
        return "Upload data or use the default dataset to view prediction summaries."
    # Need actual labels
    if "actual_home_win" not in preds.columns:
        return "Predictions loaded. Select a game for details."
    valid = preds.dropna(subset=["actual_home_win"]) if preds["actual_home_win"].isna().any() else preds
    n = len(valid)
    if n == 0:
        return "No completed games with final scores available yet."
    preds_bin = (valid["home_win_prob"] >= 0.5).astype(int)
    correct = int((preds_bin == valid["actual_home_win"]).sum())
    acc = correct / n
    avg_prob = valid["home_win_prob"].mean()
    home_rate = valid["actual_home_win"].mean()
    return (
        f"**Overall Summary**  \nGames with results: **{n}**  |  Correct: **{correct}**  |  Accuracy: **{acc:.2%}**  \n"
        f"Avg home win prob: **{avg_prob:.3f}**  |  Actual home win rate: **{home_rate:.2%}**  \n"
        "Select a row in the table to see per‑game details."
    )


def process_upload(file):  # outputs summary_md, error_md, table, seasons, weeks, teams, detail_md
    if file is None:
        STATE.update({"features": None, "preds": None, "metrics_md": ""})
        return "", "No file uploaded.", pd.DataFrame(), [], [], []
    learn, err = load_model()
    if err:
        return "", err, pd.DataFrame(), [], [], []
    # Engineer features (retain constants for alignment) and batch predict
    feats, cat_names, cont_names = build_features(file.name, prune_constants=False)  # type: ignore[arg-type]
    # Load raw to extract final scores for display (not all kept in feature set)
    try:
        raw_full = load_raw(file.name)
        raw_scores = raw_full[["pfr_game_id","home_final","away_final"]]
    except Exception:
        raw_full = None
        raw_scores = None
    aligned = align_columns(feats, learn)
    test_dl = learn.dls.test_dl(aligned)
    preds, _ = learn.get_preds(dl=test_dl)
    if preds.ndim == 2 and preds.shape[1] == 2:
        p_home = preds[:, 1].numpy()
    else:
        p_home = preds.squeeze().numpy()
    pred_df = feats[["pfr_game_id","season","week","home_team","away_team"]].copy()
    pred_df["home_win_prob"] = p_home
    # attach final scores if available
    if raw_scores is not None:
        pred_df = pred_df.merge(raw_scores, on="pfr_game_id", how="left")
        # derive result label
        if "home_final" in pred_df.columns and "away_final" in pred_df.columns:
            pred_df["final_score"] = pred_df.home_final.fillna(-1).astype(int).astype(str) + "-" + pred_df.away_final.fillna(-1).astype(int).astype(str)
            pred_df["actual_home_win"] = (pred_df.home_final > pred_df.away_final).astype(int)
    pred_df = pred_df.sort_values("home_win_prob", ascending=False)
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
    STATE.update({
        "features": feats,
        "preds": pred_df,
        "metrics_md": summary,
        "cat_cols": cat_names,
        "cont_cols": cont_names,
        "last_view": pred_df,
        "raw": raw_full
    })
    seasons = [str(s) for s in sorted(pred_df['season'].unique())]
    weeks = [str(w) for w in sorted(pred_df['week'].unique())]
    teams = sorted(pd.unique(pred_df[['home_team','away_team']].values.ravel('K')))
    team_choices = [ALL_TEAMS_LABEL] + teams
    # Build display table (hide id & raw finals; add final_score + icon column)
    display_df = pred_df.copy()
    # ensure final_score present (recompute if necessary)
    if "final_score" not in display_df.columns and {"home_final","away_final"}.issubset(display_df.columns):
        mask_scores = display_df.home_final.notna() & display_df.away_final.notna()
        display_df.loc[mask_scores, "final_score"] = (
            display_df.loc[mask_scores, "home_final"].astype(int).astype(str)
            + "-" + display_df.loc[mask_scores, "away_final"].astype(int).astype(str)
        )
    if "final_score" not in display_df.columns:
        display_df["final_score"] = ""
    if "actual_home_win" in display_df.columns:
        display_df["actual"] = display_df["actual_home_win"].map({1:"✅",0:"❌"}).fillna("")
    else:
        display_df["actual"] = ""
    # limit columns for display
    keep_cols = ["season","week","home_team","away_team","home_win_prob","final_score","actual"]
    display_df = display_df[keep_cols]
    # format probability as percentage
    display_df["home_win_prob"] = (display_df["home_win_prob"] * 100).map(lambda x: f"{x:.2f}%")
    detail_summary = _overall_predictions_summary()
    return summary, "", display_df, gr.update(choices=seasons, value=[]), gr.update(choices=weeks, value=[]), gr.update(choices=team_choices, value=ALL_TEAMS_LABEL), detail_summary


def filter_table(seasons, weeks, team, _unused_search=None):
    df = STATE.get("preds")
    if df is None or df.empty:
        return pd.DataFrame()
    view = df.copy()
    # incoming seasons/weeks are strings from CheckboxGroup
    if seasons:
        seasons_int = {int(s) for s in seasons}
        view = view[view.season.isin(seasons_int)]
    if weeks:
        weeks_int = {int(w) for w in weeks}
        view = view[view.week.isin(weeks_int)]
    if team and team not in (ALL_TEAMS_LABEL, ""):
        view = view[(view.home_team == team) | (view.away_team == team)]
    # search removed per UI simplification
    view = view.sort_values("home_win_prob", ascending=False)
    STATE["last_view"] = view  # full view for selection
    # create display version hiding id & finals, mapping actual
    disp = view.copy()
    if "actual_home_win" in disp.columns:
        disp["actual"] = disp["actual_home_win"].map({1:"✅",0:"❌"}).fillna("")
    else:
        disp["actual"] = ""
    if {"home_final","away_final"}.issubset(disp.columns):
        mask_scores = disp.home_final.notna() & disp.away_final.notna()
        disp.loc[mask_scores, "final_score"] = (
            disp.loc[mask_scores, "home_final"].astype(int).astype(str)
            + "-" + disp.loc[mask_scores, "away_final"].astype(int).astype(str)
        )
    if "final_score" not in disp.columns:
        disp["final_score"] = ""
    cols = ["season","week","home_team","away_team","home_win_prob","final_score","actual"]
    out_df = disp[cols].copy()
    out_df["home_win_prob"] = (out_df["home_win_prob"] * 100).map(lambda x: f"{x:.2f}%")
    return out_df


def pick_random_game():
    view = STATE.get("last_view")
    if view is None or getattr(view, 'empty', True):
        return "No games available."
    game_id = random.choice(view.pfr_game_id.tolist())
    return show_game(game_id)


def show_game(game_id: str | None):
    preds = STATE.get("preds")
    feats = STATE.get("features")
    if game_id is None or not game_id or preds is None or preds.empty:
        return "Select a game."
    if game_id not in set(preds.pfr_game_id):
        return "Filtered list changed; pick another game."
    row = preds.loc[preds.pfr_game_id == game_id]
    if row.empty:
        return "Game not found."
    r = row.iloc[0]
    prob = float(r.home_win_prob)
    # enrich with feature row (coaches, qbs, margin) if available
    coach_qb_md = ""
    if feats is not None and not feats.empty and "pfr_game_id" in feats.columns:
        fr = feats.loc[feats.pfr_game_id == game_id]
        if not fr.empty:
            fr0 = fr.iloc[0]
            parts = []
            for label, col in [
                ("Home QB", "home_starting_qb"),
                ("Away QB", "away_starting_qb"),
                ("Home Coach", "home_head_coach"),
                ("Away Coach", "away_head_coach"),
            ]:
                if col in fr0 and not pd.isna(fr0[col]):
                    parts.append(f"**{label}:** {fr0[col]}")
            if "point_margin" in fr0 and not pd.isna(fr0.point_margin):
                parts.append(f"**Point Margin:** {int(fr0.point_margin)}")
            if parts:
                coach_qb_md = "\n" + "  |  ".join(parts)
    raw = STATE.get("raw")
    # Score & correctness
    score_home = score_away = None
    winner_line = ""
    correct_icon = ""
    if raw is not None and isinstance(raw, pd.DataFrame):
        raw_row = raw.loc[raw.pfr_game_id == game_id]
        if not raw_row.empty:
            rr = raw_row.iloc[0]
            if not pd.isna(rr.get("home_final")):
                score_home = int(rr.home_final)
                score_away = int(rr.away_final)
                actual_winner = r.home_team if score_home > score_away else r.away_team
                predicted_winner = r.home_team if prob >= 0.5 else r.away_team
                correct_icon = "✅" if actual_winner == predicted_winner else "❌"
                winner_line = f"Winner: **{actual_winner}** {correct_icon} (pred: {predicted_winner})"
        else:
            raw_row = None
    else:
        raw_row = None
    bar_html = render_prob_bar(prob)
    # Build a markdown detail table
    # Build requested table: team, coach, quarterback, q1..q4, final
    detail_table = ""
    if raw_row is not None and not raw_row.empty:
        rr = raw_row.iloc[0]
        def sv(col):
            return rr[col] if col in rr and not pd.isna(rr[col]) else "—"
        def cumulative(prefix: str):
            vals = []
            running = 0
            for q in ("q1","q2","q3","q4"):
                col = f"{prefix}_{q}"
                raw_v = rr[col] if col in rr and not pd.isna(rr[col]) else None
                add = int(raw_v) if raw_v is not None else 0
                running += add
                vals.append(running if raw_v is not None else "—")
            final_col = f"{prefix}_final"
            final_v = rr[final_col] if final_col in rr and not pd.isna(rr[final_col]) else "—"
            return vals + [final_v if final_v != "—" else (running if any(v != "—" for v in vals) else "—")]
        home_row = [
            f"{r.home_team} <span class='home-pill'>HOME</span>",
            sv("home_head_coach"),
            sv("home_starting_qb"),
            *cumulative("home")
        ]
        away_row = [
            r.away_team,
            sv("away_head_coach"),
            sv("away_starting_qb"),
            *cumulative("away")
        ]
        rows = [home_row, away_row]
        header = "| team | coach | quarterback | q1 | q2 | q3 | q4 | final |"
        sep =    "|------|-------|------------|----|----|----|----|-------|"
        body_lines = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
        detail_table = "\n\n" + "\n".join([header, sep] + body_lines)
    # Short summary sentence above table
    if score_home is not None and score_away is not None:
        correctness_word = "Correct" if correct_icon == "✅" else "Incorrect" if correct_icon else ""
        summary = (
            f"Model home win probability: {prob:.1%}. Final: {r.home_team} {score_home} – {score_away} {r.away_team}. "
            f"{correctness_word} {correct_icon}".strip()
        )
    else:
        summary = f"Model home win probability: {prob:.1%}."
    md = summary + detail_table + "\n\n" + bar_html
    return md


def on_table_select(evt: gr.SelectData):  # type: ignore
    view = STATE.get("last_view")
    if view is None or getattr(view, 'empty', True):
        return "Select a game."
    try:
        row_idx, _ = evt.index  # (row, col)
    except Exception:
        return "Select a game."
    if row_idx is None or row_idx >= len(view):
        return "Select a game."
    game_id = view.iloc[row_idx].pfr_game_id
    return show_game(game_id)


def clear_filters_fn():
    """Reset all filters and return full table view."""
    preds = STATE.get("preds")
    if preds is None or preds.empty:
        return (
            gr.update(value=[]),
            gr.update(value=[]),
            gr.update(value=ALL_TEAMS_LABEL),
            pd.DataFrame(),
            _overall_predictions_summary()
        )
    # reset last_view to full sorted preds
    full = preds.sort_values("home_win_prob", ascending=False)
    STATE["last_view"] = full
    # build display df similar to process_upload / filter_table
    disp = full.copy()
    if "actual_home_win" in disp.columns:
        disp["actual"] = disp["actual_home_win"].map({1:"✅",0:"❌"}).fillna("")
    else:
        disp["actual"] = ""
    if {"home_final","away_final"}.issubset(disp.columns) and "final_score" not in disp.columns:
        mask_scores = disp.home_final.notna() & disp.away_final.notna()
        disp.loc[mask_scores, "final_score"] = (
            disp.loc[mask_scores, "home_final"].astype(int).astype(str) + "-" + disp.loc[mask_scores, "away_final"].astype(int).astype(str)
        )
    if "final_score" not in disp.columns:
        disp["final_score"] = ""
    cols = ["season","week","home_team","away_team","home_win_prob","final_score","actual"]
    disp = disp[cols].copy()
    # format prob as percentage
    disp["home_win_prob"] = (disp["home_win_prob"] * 100).map(lambda x: f"{x:.2f}%")
    return (
        gr.update(value=[]),
        gr.update(value=[]),
        gr.update(value=ALL_TEAMS_LABEL),
        disp,
        _overall_predictions_summary()
    )


theme = gr.themes.Soft(primary_hue=gr.themes.colors.indigo, secondary_hue=gr.themes.colors.amber).set(
    body_background_fill="#f8fafc", 
    block_border_width="1px",
    block_shadow='0 1px 3px rgba(0,0,0,0.08)',
    input_border_color='#475569'
)

CUSTOM_CSS = """
/* Show pointer only when hovering the interactive wrapper, not every nested span */
.clicky:hover { cursor: pointer; }
/* Keep native cursor for inner text but add pointer for native inputs */
.clicky input:hover, .clicky select:hover, .clicky label:hover { cursor: pointer; }
/* Focus outline for accessibility */
.gradio-container .clicky:focus-within { outline: 2px solid #4338ca !important; }
.gradio-container .gr-button:hover { filter: brightness(1.05); }
/* Selected row styling (applied via JS injection later if desired) */
table tbody tr.selected-row { background:#fde68a !important; }
/* Home pill badge */
.home-pill { background:#1e3a8a; color:#fff; padding:2px 6px; border-radius:12px; font-size:11px; font-weight:600; letter-spacing:0.5px; }
"""

with gr.Blocks(theme=theme, title="NFL Win Probability", css=CUSTOM_CSS) as demo:
    gr.HTML("""<div style='display:flex;align-items:center;gap:12px'>
    <h1 style='margin:0'>NFL Home Win Probability</h1>
    <span style='font-size:14px;color:#555'>FastAI Tabular Neural Net</span>
    </div>""")
    gr.Markdown("Upload a games CSV or DuckDB file to generate features & model probabilities. Filter and inspect individual matchups.")
    with gr.Row():
        with gr.Column(scale=1):
            file_in = gr.File(label="1. Upload games file (CSV or DuckDB). Auto loads data/games.duckdb or games.csv if present.", file_types=[".csv", ".duckdb"])
            summary_md = gr.Markdown(label="Dataset Summary")
            error_md = gr.Markdown(visible=False)
            season_filter = gr.CheckboxGroup(label="Season(s)", elem_classes=["clicky"]) 
            week_filter = gr.CheckboxGroup(label="Week(s)", elem_classes=["clicky"]) 
            team_filter = gr.Dropdown(label="Team (either side)", choices=[ALL_TEAMS_LABEL], value=ALL_TEAMS_LABEL, elem_classes=["clicky"]) 
            # search box removed
            clear_filters = gr.Button("Clear Filters", variant="secondary")
        with gr.Column(scale=2):
            games_df = gr.Dataframe(headers=["season","week","home_team","away_team","home_win_prob","final_score","actual"], interactive=False)
            game_md = gr.Markdown()
    gr.Markdown("---\nBuilt with fastai + gradio. Probability represents model's pre-game home win chance.")

    # Events
    file_in.change(process_upload, inputs=file_in, outputs=[summary_md, error_md, games_df, season_filter, week_filter, team_filter, game_md])

    season_filter.change(filter_table, [season_filter, week_filter, team_filter], [games_df])
    week_filter.change(filter_table, [season_filter, week_filter, team_filter], [games_df])
    team_filter.change(filter_table, [season_filter, week_filter, team_filter], [games_df])
    clear_filters.click(clear_filters_fn, outputs=[season_filter, week_filter, team_filter, games_df, game_md])
    games_df.select(on_table_select, outputs=[game_md])

    def bootstrap():  # executed on initial load
        if DEFAULT_DATA_PATH.exists():
            fake = SimpleNamespace(name=str(DEFAULT_DATA_PATH))
            return process_upload(fake)
        # No default data; empty outputs plus placeholder detail text
        return (
            "",  # summary_md
            f"Default data file not found at {DEFAULT_DATA_PATH}",  # error_md
            pd.DataFrame(),  # games_df
            gr.update(choices=[], value=[]),  # seasons
            gr.update(choices=[], value=[]),  # weeks
            gr.update(choices=[], value=ALL_TEAMS_LABEL),  # team dropdown
            "Upload a CSV to view model predictions and summary."  # detail panel
        )

    demo.load(bootstrap, outputs=[summary_md, error_md, games_df, season_filter, week_filter, team_filter, game_md])

if __name__ == "__main__":  # pragma: no cover
    demo.launch(server_name="0.0.0.0", share=False)