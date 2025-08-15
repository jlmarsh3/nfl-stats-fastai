import pandas as pd
from src.features import build_features

def test_build_features_sample():
    df, cats, conts = build_features('data/sample_games.csv')
    assert 'home_team_won' in df.columns
    assert 'home_team' in df.columns and 'away_team' in df.columns
    assert any(c.startswith('home_roll_pts_for_') for c in df.columns)
    assert len(cats) > 0 and len(conts) > 0