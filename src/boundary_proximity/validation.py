from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
    }


def grouped_model_comparison(
    df: pd.DataFrame,
    *,
    target_col: str = "TFe",
    group_col: str = "BHID",
    baseline_features: list[str] | None = None,
    boundary_features: list[str] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare baseline and boundary-feature models under grouped validation."""

    if baseline_features is None:
        baseline_features = ["x", "y", "z", "local_cv", "local_semivariance", "support_imbalance", "lni_structural", "lni_drilling"]
    if boundary_features is None:
        boundary_features = baseline_features + ["boundary_distance", "boundary_zone_code"]

    work = df.dropna(subset=[target_col, group_col] + boundary_features).copy()
    groups = work[group_col].astype(str).to_numpy()
    y = work[target_col].to_numpy(float)
    unique_groups = np.unique(groups)
    folds = min(n_splits, len(unique_groups))
    if folds < 2:
        raise ValueError("at least two groups are required for grouped validation")

    gkf = GroupKFold(n_splits=folds)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(work, y, groups), start=1):
        for model_name, features in [
            ("M4_LNI_baseline", baseline_features),
            ("M8_boundary", boundary_features),
        ]:
            model = RandomForestRegressor(
                n_estimators=120,
                min_samples_leaf=3,
                random_state=random_state,
                n_jobs=1,
            )
            model.fit(work.iloc[train_idx][features], y[train_idx])
            pred = model.predict(work.iloc[test_idx][features])
            row = {"fold": fold, "model": model_name, "n_test": int(len(test_idx))}
            row.update(_metrics(y[test_idx], pred))
            rows.append(row)
    return pd.DataFrame(rows)
