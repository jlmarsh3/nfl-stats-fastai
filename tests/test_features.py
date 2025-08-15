import pandas as pd
from src.features import build_features

def test_build_features_sample():
    df, cats, conts = build_features('data/sample_games.csv')
    assert 'home_team_won' in df.columns
    assert 'home_team' in df.columns and 'away_team' in df.columns
    # With tiny sample we at least expect the 3-game rolling window feature scaffolding
    assert 'home_roll_pts_for_3' in df.columns
    assert len(cats) > 0 and len(conts) > 0