from fastai.tabular.all import *
from pathlib import Path
from src.features import build_features
from src.splits import time_splits
import argparse, random, os, numpy as np, torch

DATA_PATH = "data/games.csv"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_dls(df, cat_names, cont_names, trn, val, bs):
    procs = [Categorify, FillMissing, Normalize]
    to = TabularPandas(df, procs,
                       cat_names=cat_names, cont_names=cont_names,
                       y_names='home_team_won', y_block=CategoryBlock(),
                       splits=(trn, val))
    return to.dataloaders(bs=bs), to

def evaluate_test(learn, dls, df, test_idx):
    if not len(test_idx):
        return None
    test_df = df.iloc[test_idx]
    test_dl = dls.test_dl(test_df)
    preds, targs = learn.get_preds(dl=test_dl)
    if preds.ndim==2 and preds.shape[1]==2:
        pos = preds[:,1]
    else:
        pos = preds.float().squeeze()
    targs_flat = targs.squeeze()
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(targs_flat.cpu().numpy(), pos.cpu().numpy())
    except ValueError:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=DATA_PATH)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--lr', type=float, default=None, help='Override LR (otherwise use lr_find)')
    ap.add_argument('--train_end', type=int, default=2021)
    ap.add_argument('--valid_season', type=int, default=2022)
    ap.add_argument('--layers', type=str, default='400,200,100')
    ap.add_argument('--wd', type=float, default=1e-2)
    args = ap.parse_args()

    set_seed(args.seed)

    df, cat_names, cont_names = build_features(args.data, prune_constants=True)
    (trn,val), test_idx = time_splits(df, train_end=args.train_end, valid_season=args.valid_season)
    if len(trn)==0 or len(val)==0:
        n=len(df)
        if n<5: raise ValueError("Not enough rows to create splits.")
        trn_end=int(n*0.7); val_end=int(n*0.85)
        trn=list(range(trn_end)); val=list(range(trn_end,val_end)); test_idx=list(range(val_end,n))

    print(f"SPLITS train={len(trn)} valid={len(val)} test={len(test_idx)} total={len(df)}")
    print("TARGET train dist:", df.loc[trn,'home_team_won'].value_counts().to_dict())
    print("TARGET valid dist:", df.loc[val,'home_team_won'].value_counts().to_dict())
    print("#cats=", len(cat_names), "#cont=", len(cont_names))

    dls, to = make_dls(df, cat_names, cont_names, trn, val, args.bs)
    if len(dls.train)==0: raise RuntimeError('Empty training DataLoader')

    layer_sizes = [int(x) for x in args.layers.split(',') if x.strip()]
    # Put RocAuc first so monitored metric index is stable
    learn = tabular_learner(dls, layers=layer_sizes,
                            metrics=[RocAucBinary(), accuracy],
                            wd=args.wd,
                            cbs=[EarlyStoppingCallback(monitor='roc_auc_score', patience=5),
                                 SaveModelCallback(monitor='roc_auc_score', fname='best_model')])

    if args.lr is None:
        print("Running lr_find...")
        lr_find_res = learn.lr_find()
        # fastai may return LRFinder object or tuple
        try:
            lr_min, lr_steep = lr_find_res
        except Exception:
            # new API returns an object with suggestion() method
            try:
                lr_min = lr_find_res.valley
                lr_steep = getattr(lr_find_res, 'slide', None) or getattr(lr_find_res, 'steep', None)
            except Exception:
                lr_min = 1e-3
                lr_steep = None
        lr = lr_steep or lr_min or 1e-3
        print(f"lr_find selected lr={lr}")
    else:
        lr = args.lr

    learn.fit_one_cycle(args.epochs, lr)
    # Call validate without training callbacks to avoid SaveModelCallback indexing issues
    print("VALID metrics:", learn.validate(cbs=[]))

    test_roc = evaluate_test(learn, dls, df, test_idx)
    if test_roc is not None:
        print(f"TEST ROC-AUC: {test_roc:.4f}")
    else:
        print("TEST ROC-AUC: unavailable (single class or empty test set)")

    Path('models').mkdir(parents=True, exist_ok=True)
    learn.export('models/winner_export.pkl')
    print("Saved model to models/winner_export.pkl (best_model.pth for weights)")

if __name__ == '__main__':
    main()