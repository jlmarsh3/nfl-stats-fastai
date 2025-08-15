from __future__ import annotations
import pandas as pd
import numpy as np

REQUIRED_COLS = [
    "season","date","game_type","week","away_team","home_team",
    "away_q1","away_q2","away_q3","away_q4","away_ot","away_final",
    "home_q1","home_q2","home_q3","home_q4","home_ot","home_final",
    "away_starting_qb","home_starting_qb","away_head_coach","home_head_coach","pfr_game_id"
]

def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    # ensure dtypes
    df["season"] = df["season"].astype(int)
    df["week"] = df["week"].astype(int)
    return df

def _to_long(df: pd.DataFrame) -> pd.DataFrame:
    # home perspective
    home = df[[
        "date","season","week","game_type","home_team","away_team",
        "home_final","away_final",
        "home_q1","home_q2","home_q3","home_q4","home_ot",
        "away_q1","away_q2","away_q3","away_q4","away_ot",
        "home_starting_qb","home_head_coach","pfr_game_id"
    ]].copy()
    home = home.rename(columns={
        "home_team":"team","away_team":"opp","home_final":"pts_for","away_final":"pts_against",
        "home_q1":"q1","home_q2":"q2","home_q3":"q3","home_q4":"q4","home_ot":"ot",
        "away_q1":"opp_q1","away_q2":"opp_q2","away_q3":"opp_q3","away_q4":"opp_q4","away_ot":"opp_ot",
        "home_starting_qb":"qb","home_head_coach":"coach"
    })
    home["is_home"] = 1

    # away perspective
    away = df[[
        "date","season","week","game_type","away_team","home_team",
        "away_final","home_final",
        "away_q1","away_q2","away_q3","away_q4","away_ot",
        "home_q1","home_q2","home_q3","home_q4","home_ot",
        "away_starting_qb","away_head_coach","pfr_game_id"
    ]].copy()
    away = away.rename(columns={
        "away_team":"team","home_team":"opp","away_final":"pts_for","home_final":"pts_against",
        "away_q1":"q1","away_q2":"q2","away_q3":"q3","away_q4":"q4","away_ot":"ot",
        "home_q1":"opp_q1","home_q2":"opp_q2","home_q3":"opp_q3","home_q4":"opp_q4","home_ot":"opp_ot",
        "away_starting_qb":"qb","away_head_coach":"coach"
    })
    away["is_home"] = 0

    long = pd.concat([home, away], ignore_index=True).sort_values(["team","date"]).reset_index(drop=True)
    return long

def _add_leak_safe_rolls(long: pd.DataFrame) -> pd.DataFrame:
    # rest days
    long["prev_date"] = long.groupby("team")["date"].shift(1)
    long["rest_days"] = (long["date"] - long["prev_date"]).dt.days

    # basic rolling team form (use only prior games via shift)
    for w in (1,3,5):
        s_for = long.groupby("team")["pts_for"].shift(1)
        s_against = long.groupby("team")["pts_against"].shift(1)
        long[f"roll_pts_for_{w}"] = s_for.rolling(w).mean().reset_index(level=0, drop=True)
        long[f"roll_pts_against_{w}"] = s_against.rolling(w).mean().reset_index(level=0, drop=True)
        long[f"roll_margin_{w}"] = (s_for - s_against).rolling(w).mean().reset_index(level=0, drop=True)

    # QB & coach continuity
    long["qb_prev"] = long.groupby("team")["qb"].shift(1)
    long["qb_cont"] = (long["qb"] == long["qb_prev"]).astype("int8").fillna(0)

    long["coach_prev"] = long.groupby("team")["coach"].shift(1)
    long["coach_cont"] = (long["coach"] == long["coach_prev"]).astype("int8").fillna(0)

    # QB rolling points (proxy for QB form)
    long["qb_game_pts"] = long["pts_for"]
    for w in (3,5):
        long[f"qb_roll_pts_{w}"] = (
            long.groupby(["team","qb"])["qb_game_pts"].shift(1).rolling(w).mean().reset_index(drop=True)
        )
    return long

def _to_wide(df_raw: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    home_feats = long[long["is_home"]==1].add_prefix("home_")
    away_feats = long[long["is_home"]==0].add_prefix("away_")

    wide = df_raw.copy()
    wide = wide.merge(home_feats, left_on="pfr_game_id", right_on="home_pfr_game_id", how="left")
    wide = wide.merge(away_feats, left_on="pfr_game_id", right_on="away_pfr_game_id", how="left")

    # clean column suffixes from merge for original teams (retain raw names for modeling convenience)
    rename_cols = {}
    if "home_team_x" in wide.columns: rename_cols["home_team_x"] = "home_team"
    if "away_team_x" in wide.columns: rename_cols["away_team_x"] = "away_team"
    wide = wide.rename(columns=rename_cols)
    # drop duplicate team cols from merge if present
    for dup in ("home_team_y","away_team_y"):
        if dup in wide.columns:
            wide = wide.drop(columns=[dup])

    # target & simple context
    wide["home_team_won"] = (wide["home_final"] > wide["away_final"]).astype("int8")
    wide["point_margin"] = (wide["home_final"] - wide["away_final"]).astype("int16")
    wide["month"] = wide["date"].dt.month.astype("int8")
    wide["day_of_week"] = wide["date"].dt.dayofweek.astype("int8")
    wide["is_thursday"] = (wide["day_of_week"] == 3).astype("int8")
    wide["is_sunday"] = (wide["day_of_week"] == 6).astype("int8")

    # matchup deltas (home - away) for shared features
    def delta(col: str):
        return wide[f"home_{col}"] - wide[f"away_{col}"]

    for base in [
        "rest_days",
        "roll_pts_for_1","roll_pts_for_3","roll_pts_for_5",
        "roll_pts_against_1","roll_pts_against_3","roll_pts_against_5",
        "roll_margin_1","roll_margin_3","roll_margin_5",
        "qb_roll_pts_3","qb_roll_pts_5",
        "qb_cont","coach_cont"
    ]:
        if f"home_{base}" in wide.columns and f"away_{base}" in wide.columns:
            wide[f"delta_{base}"] = delta(base)

    return wide

def build_features(path: str, prune_constants: bool = True) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Returns:
      df_wide: engineered one-row-per-game table
      cat_names: categorical feature names
      cont_names: continuous feature names
    """
    raw = load_raw(path)
    long = _to_long(raw)
    long = _add_leak_safe_rolls(long)
    wide = _to_wide(raw, long)

    cat_names = [
        "home_team","away_team",
        "home_starting_qb","away_starting_qb",
        "home_head_coach","away_head_coach",
        "game_type","week"
    ]
    cont_candidates = [
        # per-side
        "home_rest_days","away_rest_days",
        "home_roll_pts_for_3","away_roll_pts_for_3",
        "home_roll_pts_against_3","away_roll_pts_against_3",
        "home_roll_margin_3","away_roll_margin_3",
        "home_qb_roll_pts_3","away_qb_roll_pts_3",
        "home_qb_cont","away_qb_cont",
        "home_coach_cont","away_coach_cont",
        # deltas
        "delta_rest_days",
        "delta_roll_pts_for_3","delta_roll_pts_against_3","delta_roll_margin_3",
        "delta_qb_roll_pts_3","delta_qb_cont","delta_coach_cont",
        # context
        "month","day_of_week","is_thursday","is_sunday"
    ]
    cont_names = [c for c in cont_candidates if c in wide.columns]
    # keep a tidy set of columns
    keep = [
        "season","date","week","game_type",
        "home_team","away_team",
        "home_starting_qb","away_starting_qb",
        "home_head_coach","away_head_coach",
        "home_team_won","point_margin","pfr_game_id"
    ] + cat_names + cont_names
    keep = list(dict.fromkeys([c for c in keep if c in wide.columns]))  # uniq & existent
    wide = wide[keep].sort_values("date").reset_index(drop=True)
    # basic NA handling and constant column pruning
    existing_cats = [c for c in cat_names if c in wide.columns]
    existing_conts = [c for c in cont_names if c in wide.columns]
    for c in existing_cats:
        wide[c] = wide[c].fillna('Unknown')
    if existing_conts:
        for c in existing_conts:
            wide[c] = wide[c].fillna(0)
    if prune_constants:
        drop_const = [c for c in existing_conts if wide[c].nunique(dropna=False) <= 1]
        if drop_const:
            wide = wide.drop(columns=drop_const)
            cont_names = [c for c in cont_names if c not in drop_const]
    return wide, cat_names, cont_names