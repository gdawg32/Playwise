# File: train.py
# Purpose: Train W-D-L (Home/Draw/Away) model from `features_played_enhanced.csv`
# - conservative, generalization-first pipeline
# - smoothed team target-encoding (train-only)
# - rest-day computation, basic interactions
# - LightGBM + Logistic ensemble, optional Optuna tuning and SHAP
# - Saves artifacts to joblib
#
# Usage examples:
#   python train.py --csv features_played_enhanced.csv --out model_artifacts.joblib
#   python train.py --csv features_played_enhanced.csv --do-optuna --n-trials 20
#
# Notes:
# - This script assumes `result` contains 'H','D','A' and `date_parsed` is present.
# - Optuna and SHAP are optional (install if you want those features).
# - Designed to run locally in your virtualenv (tested with lightgbm 4.x).

import argparse
import json
import warnings
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import joblib

import lightgbm as lgb

# optional libs
try:
    import optuna
except Exception:
    optuna = None

try:
    import shap
    import matplotlib.pyplot as plt
except Exception:
    shap = None

BASE_DIR = Path(__file__).resolve().parent.parent  

DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR.parent / "home" / "artifacts"

DEFAULT_CSV = DATA_DIR / "processed" / "features_played_enhanced.csv"
DEFAULT_OUT = ARTIFACT_DIR / "model_artifacts.joblib"

# ----------------- Utilities / Feature Derivation ----------------- #

def derive_time_features(df, date_col='date_parsed'):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df['matchday'] = df.groupby('season').cumcount() + 1
    df['dow'] = df[date_col].dt.dayofweek
    df['is_weekend'] = df['dow'].isin([5, 6]).astype(int)
    season_start = df.groupby('season')[date_col].transform('min')
    df['days_from_season_start'] = (df[date_col] - season_start).dt.days
    return df


def compute_rest_days(df, date_col='date_parsed'):
    df2 = df.copy().reset_index(drop=False).rename(columns={'index': 'match_idx'})
    home = df2[['match_idx', date_col, 'home_team']].rename(columns={'home_team': 'team'})
    home['role'] = 'home'
    away = df2[['match_idx', date_col, 'away_team']].rename(columns={'away_team': 'team'})
    away['role'] = 'away'
    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(['team', date_col, 'match_idx']).reset_index(drop=True)
    long['prev_date'] = long.groupby('team')[date_col].shift(1)
    long['rest_days'] = (long[date_col] - long['prev_date']).dt.days
    long['rest_days'] = long['rest_days'].fillna(999).astype(int)
    home_rest = long[long['role'] == 'home'].set_index('match_idx')['rest_days']
    away_rest = long[long['role'] == 'away'].set_index('match_idx')['rest_days']
    df_indexed = df2.set_index('match_idx')
    df_indexed['home_rest_days'] = home_rest
    df_indexed['away_rest_days'] = away_rest
    df_indexed['home_rest_days'] = df_indexed['home_rest_days'].fillna(999).astype(int)
    df_indexed['away_rest_days'] = df_indexed['away_rest_days'].fillna(999).astype(int)
    return df_indexed.reset_index(drop=True)


def smooth_target_encoding(df, col, y_col='y', train_mask=None, prior=15):
    """
    Bayesian smoothed target encoding computed using *only* training rows.
    Returns: enc_series (full length) and mapping (train-based)
    """
    if train_mask is None:
        raise ValueError("train_mask must be provided for safe target encoding")
    df_local = df.copy()
    global_mean = df_local.loc[train_mask, y_col].mean()
    train_df = df_local.loc[train_mask]
    grp = train_df.groupby(col)[y_col].agg(['mean', 'count']).rename(columns={'mean': 'team_mean', 'count': 'cnt'})
    grp['enc'] = (grp['team_mean'] * grp['cnt'] + global_mean * prior) / (grp['cnt'] + prior)
    mapping = grp['enc'].to_dict()
    enc = df_local[col].map(mapping).fillna(global_mean).astype(float)
    return enc, mapping


def temporal_split_masks(df, train_seasons=None, val_seasons=None, test_seasons=None):
    # default season splits
    if train_seasons is None:
        train_seasons = ['21/22', '22/23', '23/24']
    if val_seasons is None:
        val_seasons = ['24/25']
    if test_seasons is None:
        test_seasons = ['25/26']
    train_mask = df['season'].isin(train_seasons)
    val_mask = df['season'].isin(val_seasons)
    test_mask = df['season'].isin(test_seasons)
    if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
        warnings.warn("Season-based split empty; falling back to date-based temporal split (70/15/15).")
        df_sorted = df.sort_values('date_parsed').reset_index()
        n = len(df_sorted)
        i1 = int(n * 0.70)
        i2 = int(n * 0.85)
        train_idx = df_sorted.loc[:i1 - 1, 'index'].values
        val_idx = df_sorted.loc[i1:i2 - 1, 'index'].values
        test_idx = df_sorted.loc[i2:, 'index'].values
        train_mask = df.index.isin(train_idx)
        val_mask = df.index.isin(val_idx)
        test_mask = df.index.isin(test_idx)
    return train_mask, val_mask, test_mask


def make_time_series_folds_for_opt(df_train):
    seasons = sorted(df_train['season'].unique())
    folds = []
    if len(seasons) < 3:
        n = len(df_train)
        i1 = int(n * 0.6)
        i2 = int(n * 0.8)
        folds.append((np.arange(0, i1), np.arange(i1, i2)))
        folds.append((np.arange(0, i2), np.arange(i2, n)))
        return folds
    for i in range(1, len(seasons)):
        train_seasons = seasons[:i]
        val_season = seasons[i]
        tr_idx = df_train[df_train['season'].isin(train_seasons)].index.values
        val_idx = df_train[df_train['season'] == val_season].index.values
        if len(tr_idx) and len(val_idx):
            folds.append((tr_idx, val_idx))
    return folds


def optuna_tune_lgb(X_train_df, y_train_df, numeric_cols, n_trials=30, seed=42):
    if optuna is None:
        raise RuntimeError("optuna not installed")
    df_train = X_train_df.copy()
    df_train['y'] = y_train_df.values
    folds = make_time_series_folds_for_opt(df_train)
    if not folds:
        raise RuntimeError("Not enough seasons/data for time-aware folds")
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 3, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 1.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 1.0),
        }
        cv_scores = []
        for tr_idx, val_idx in folds:
            Xtr = X_train_df.loc[tr_idx, numeric_cols]
            ytr = y_train_df.loc[tr_idx]
            Xv = X_train_df.loc[val_idx, numeric_cols]
            yv = y_train_df.loc[val_idx]
            clf = lgb.LGBMClassifier(objective='multiclass', num_class=3, n_estimators=2000, random_state=seed, **params)
            clf.fit(Xtr, ytr, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(stopping_rounds=50)], verbose=-1)
            probs = clf.predict_proba(Xv)
            cv_scores.append(log_loss(yv, probs))
        return float(np.mean(cv_scores))
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study.best_trial.params


# ----------------- Main training flow ----------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default=str(DEFAULT_CSV))
    parser.add_argument('--out', type=str, default=str(DEFAULT_OUT))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--do-optuna', action='store_true', help='Run Optuna time-aware tuning (may be slow)')
    parser.add_argument('--n-trials', type=int, default=20)
    parser.add_argument('--shap', action='store_true', help='Compute SHAP summary (optional)')
    args = parser.parse_args()

    np.random.seed(args.seed)
    print("Python executable:", sys.executable)
    print("LightGBM version:", lgb.__version__)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Load
    df = pd.read_csv(csv_path, parse_dates=['date_parsed'])
    required = {'result', 'home_team', 'away_team', 'season', 'date_parsed'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Derive time & rest features
    df = derive_time_features(df, date_col='date_parsed')
    df = compute_rest_days(df, date_col='date_parsed')

    # Label mapping
    label_map = {'H': 0, 'D': 1, 'A': 2}
    df['y'] = df['result'].map(label_map)
    if df['y'].isnull().any():
        raise ValueError("Unexpected values in 'result' column: " + str(df['result'].unique()))

    # Temporal split masks
    train_mask, val_mask, test_mask = temporal_split_masks(df)
    print("Split sizes -> train:", train_mask.sum(), "val:", val_mask.sum(), "test:", test_mask.sum())

    # Smoothed team encodings (train-only)
    df['home_team_te'], home_map = smooth_target_encoding(df, 'home_team', 'y', train_mask=train_mask, prior=15)
    df['away_team_te'], away_map = smooth_target_encoding(df, 'away_team', 'y', train_mask=train_mask, prior=15)
    df['home_team_te'] = df['home_team_te'].clip(0.01, 0.99)
    df['away_team_te'] = df['away_team_te'].clip(0.01, 0.99)

    # Derived interactions (recommended)
    df['rest_days_diff'] = df['home_rest_days'] - df['away_rest_days']
    df['home_favorite'] = (df['home_elo'] > df['away_elo']).astype(int)
    # numeric interaction - safe (fill if any NaNs)
    df['elo_x_form'] = (df.get('elo_diff', 0).fillna(0) * df.get('last5_points_diff', 0).fillna(0))

    # Conservative feature list (based only on available columns you listed)
    features = [
        'home_elo', 'away_elo', 'elo_diff', 'elo_diff_abs',
        'home_last5_points', 'away_last5_points', 'last5_points_diff',
        'home_last5_goal_diff', 'away_last5_goal_diff', 'last5_goal_diff',
        'home_last5_goals_scored', 'away_last5_goals_scored', 'goals_last5_diff',
        'matchday', 'is_weekend', 'days_from_season_start',
        'home_rest_days', 'away_rest_days', 'rest_days_diff',
        'home_team_te', 'away_team_te', 'home_favorite', 'elo_x_form'
    ]
    # keep only features actually present
    features = [f for f in features if f in df.columns]
    print("Features used:", features)

    # Prepare X/y and basic cleaning
    X_all = df[features].copy()
    y_all = df['y'].astype(int)

    # coerce numeric and drop constant/bad columns
    for c in X_all.columns:
        X_all[c] = pd.to_numeric(X_all[c], errors='coerce')
    const_cols = [c for c in X_all.columns if X_all[c].nunique(dropna=True) <= 1]
    if const_cols:
        print("Dropping constant features:", const_cols)
        X_all = X_all.drop(columns=const_cols)
        features = [f for f in features if f not in const_cols]
    bad_cols = [c for c in X_all.columns if X_all[c].isna().all() or not np.isfinite(X_all[c].fillna(0)).all()]
    if bad_cols:
        print("Dropping bad cols (all-NaN or non-finite):", bad_cols)
        X_all = X_all.drop(columns=bad_cols)
        features = [f for f in features if f not in bad_cols]
    # fill remaining NaNs with median
    for c in X_all.columns:
        if X_all[c].isna().any():
            med = X_all[c].median()
            X_all[c] = X_all[c].fillna(med)
    X_all = X_all.astype('float32')

    # split to train/val/test
    X_train = X_all.loc[train_mask].reset_index(drop=True)
    y_train = y_all.loc[train_mask].reset_index(drop=True)
    X_val = X_all.loc[val_mask].reset_index(drop=True)
    y_val = y_all.loc[val_mask].reset_index(drop=True)
    X_test = X_all.loc[test_mask].reset_index(drop=True)
    y_test = y_all.loc[test_mask].reset_index(drop=True)

    print("Train label dist:", y_train.value_counts(normalize=True).to_dict())
    print("Val label dist:", y_val.value_counts(normalize=True).to_dict())
    print("Test label dist:", y_test.value_counts(normalize=True).to_dict())

    # ensure at least two classes in train
    if y_train.nunique() < 2:
        raise RuntimeError("Training labels have less than 2 classes. Check your split.")

    # scaler (used by logistic)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # class weights
    classes = np.unique(y_train)
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
    print("Class weights:", class_weight)

    # Optional Optuna tuning (time-aware)
    best_opt_params = None
    numeric_cols = X_train.columns.tolist()
    if args.do_optuna:
        if optuna is None:
            raise RuntimeError("optuna is not installed. Install optuna to use --do-optuna")
        print(f"Running Optuna tuning for {args.n_trials} trials (time-aware folds)...")
        try:
            best_opt_params = optuna_tune_lgb(X_train, y_train, numeric_cols, n_trials=args.n_trials, seed=args.seed)
            print("Optuna best params:", best_opt_params)
        except Exception as e:
            print("Optuna tuning failed:", e)
            best_opt_params = None

    # Base LightGBM params (conservative, generalization-first)
    lgb_params = dict(
        objective='multiclass', num_class=3,
        learning_rate=0.01, n_estimators=4000,
        num_leaves=31, max_depth=6, min_child_samples=5,
        min_split_gain=0.0, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=args.seed, n_jobs=-1,
        class_weight=class_weight
    )
    if best_opt_params:
        # merge tuned params but keep some robust defaults
        lgb_params.update(best_opt_params)
        lgb_params.setdefault('n_estimators', 4000)

    # Train LightGBM (train->val early stopping)
    print("Training LightGBM (train -> val early stopping)...")
    lgb_clf = lgb.LGBMClassifier(**lgb_params)
    lgb_clf.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='multi_logloss',
                callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=200)])

    # Train Logistic baseline (on scaled features)
    print("Training LogisticRegression (baseline)...")
    log_clf = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=5000,
                                 class_weight='balanced', random_state=args.seed)
    log_clf.fit(X_train_s, y_train)

    # Ensemble weight search (optimize log-loss on validation)
    print("Searching best ensemble weight on validation (grid 0..1)...")
    lgb_proba_val = lgb_clf.predict_proba(X_val)
    log_proba_val = log_clf.predict_proba(X_val_s)
    best_alpha = 0.5
    best_score = float('inf')
    for alpha in np.linspace(0, 1, 21):
        combo = alpha * lgb_proba_val + (1 - alpha) * log_proba_val
        score = log_loss(y_val, combo)
        if score < best_score:
            best_score = score
            best_alpha = float(alpha)
    print("Best alpha:", best_alpha, "val logloss:", best_score)

    # Retrain final models on train+val with a 20% holdout for early stopping
    X_comb = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_comb = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    scaler = StandardScaler()
    X_comb_s = scaler.fit_transform(X_comb)
    holdout_n = max(1, int(0.20 * len(X_comb)))
    X_final_train = X_comb.iloc[:-holdout_n]
    y_final_train = y_comb.iloc[:-holdout_n]
    X_hold = X_comb.iloc[-holdout_n:]
    y_hold = y_comb.iloc[-holdout_n:]
    X_final_train_s = scaler.transform(X_final_train)
    X_hold_s = scaler.transform(X_hold)

    print("Retraining final LightGBM on train+val with holdout early stopping...")
    final_lgb = lgb.LGBMClassifier(**lgb_params)
    final_lgb.fit(X_final_train, y_final_train,
                  eval_set=[(X_hold, y_hold)],
                  eval_metric='multi_logloss',
                  callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=200)])

    print("Retraining final Logistic on train+val...")
    final_log = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=5000,
                                   class_weight='balanced', random_state=args.seed)
    # Combine train+hold for logistic fit
    final_log.fit(np.vstack([X_final_train_s, X_hold_s]), np.concatenate([y_final_train, y_hold]))

    # Evaluate on test
    print("Evaluating on test set...")
    lgb_proba_test = final_lgb.predict_proba(X_test)
    log_proba_test = final_log.predict_proba(X_test_s)
    combo_test = best_alpha * lgb_proba_test + (1 - best_alpha) * log_proba_test
    preds = np.argmax(combo_test, axis=1)

    ll = log_loss(y_test, combo_test)
    acc = accuracy_score(y_test, preds)
    print("\nTest LogLoss:", ll)
    print("Test Accuracy:", acc)
    print(classification_report(y_test, preds, target_names=['H', 'D', 'A']))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    # Feature importance (gain)
    try:
        fi = list(zip(X_all.columns.tolist(), final_lgb.booster_.feature_importance(importance_type='gain')))
        fi = sorted(fi, key=lambda x: x[1], reverse=True)
    except Exception as e:
        print("Failed to extract feature importances:", e)
        fi = []

    # Optional SHAP (if requested and installed)
    if args.shap:
        if shap is None:
            print("SHAP not installed. Install shap to use --shap")
        else:
            print("Computing SHAP summary (sample <=500)...")
            sample = X_comb.sample(min(500, len(X_comb)), random_state=args.seed)
            explainer = shap.TreeExplainer(final_lgb.booster_)
            shap_vals = explainer.shap_values(sample)
            plt.figure(figsize=(8, 6))
            shap.summary_plot(shap_vals[0], sample.values, feature_names=X_all.columns.tolist(), show=False)
            shp = Path(args.out).with_suffix('.shap_summary.png')
            plt.savefig(shp, bbox_inches='tight')
            plt.close()
            print("Saved SHAP summary to", shp)

    # Save artifacts
    artifacts = {
        'scaler': scaler,
        'lgb_model': final_lgb,
        'log_model': final_log,
        'features': X_all.columns.tolist(),
        'label_map': label_map,
        'team_maps': {'home_map': home_map, 'away_map': away_map},
        'ensemble_alpha': float(best_alpha),
        'feature_importance': fi,
        'optuna_best_params': best_opt_params,
    }
    outp = Path(args.out)
    joblib.dump(artifacts, outp)
    print("Saved artifacts to", outp)

    metrics = {'log_loss': float(ll), 'accuracy': float(acc), 'n_train': int(len(X_train)), 'n_val': int(len(X_val)), 'n_test': int(len(X_test))}
    metrics_path = outp.with_suffix('.metrics.json')
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print("Saved metrics to", metrics_path)

    if fi:
        fi_path = outp.with_suffix('.feature_importance.csv')
        pd.DataFrame(fi, columns=['feature', 'gain']).to_csv(fi_path, index=False)
        print("Saved feature importances to", fi_path)

    print("Done.")


if __name__ == '__main__':
    main()
