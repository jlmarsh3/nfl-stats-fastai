from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, List

def time_splits(df: pd.DataFrame, train_end: int = 2021, valid_season: int = 2022) -> Tuple[tuple, list]:
    trn_idx = list(np.where(df['season'] <= train_end)[0])
    val_idx = list(np.where(df['season'] == valid_season)[0])
    tst_idx = list(np.where(df['season'] >= valid_season + 1)[0])
    return (trn_idx, val_idx), tst_idx