from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def _robust_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo, hi = np.nanpercentile(values, [1, 99])
    clipped = np.clip(values, lo, hi)
    return (clipped - np.nanmean(clipped)) / (np.nanstd(clipped) + 1e-12)


def add_local_nonstationarity(
    assays: pd.DataFrame,
    *,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    grade_col: str = "TFe",
    radius: float = 100.0,
    min_neighbors: int = 3,
) -> pd.DataFrame:
    """Add local nonstationarity descriptors to interval-level assay data.

    The descriptors follow the manuscript logic: local coefficient of variation,
    local semivariance, drilling support imbalance, and two aggregated LNI
    descriptors. The implementation is intentionally transparent so it can be
    audited or adapted to another deposit.
    """

    df = assays.copy()
    xyz = df[[x_col, y_col, z_col]].to_numpy(dtype=float)
    grade = df[grade_col].to_numpy(dtype=float)
    tree = cKDTree(xyz)

    local_cv = np.zeros(len(df), dtype=float)
    local_semivar = np.zeros(len(df), dtype=float)
    neighbor_count = np.zeros(len(df), dtype=float)
    mean_neighbor_distance = np.zeros(len(df), dtype=float)

    for i, point in enumerate(xyz):
        idx = tree.query_ball_point(point, r=radius)
        idx = [j for j in idx if j != i and np.isfinite(grade[j])]
        neighbor_count[i] = len(idx)
        if len(idx) < min_neighbors:
            local_cv[i] = np.nan
            local_semivar[i] = np.nan
            mean_neighbor_distance[i] = radius
            continue
        vals = grade[idx]
        local_cv[i] = np.nanstd(vals) / (abs(np.nanmean(vals)) + 1e-12)
        local_semivar[i] = 0.5 * np.nanmean((grade[i] - vals) ** 2)
        mean_neighbor_distance[i] = np.nanmean(np.linalg.norm(xyz[idx] - point, axis=1))

    local_cv = np.nan_to_num(local_cv, nan=np.nanmedian(local_cv))
    local_semivar = np.nan_to_num(local_semivar, nan=np.nanmedian(local_semivar))
    mean_neighbor_distance = np.nan_to_num(mean_neighbor_distance, nan=radius)

    inverse_support = 1.0 - np.clip(neighbor_count / max(neighbor_count.max(), 1.0), 0, 1)
    distance_support = np.clip(mean_neighbor_distance / max(radius, 1e-12), 0, 1)

    df["local_cv"] = local_cv
    df["local_semivariance"] = local_semivar
    df["support_imbalance"] = 0.5 * inverse_support + 0.5 * distance_support

    cv_z = _robust_scale(local_cv)
    semi_z = _robust_scale(local_semivar)
    support_z = _robust_scale(df["support_imbalance"].to_numpy())

    df["lni_structural"] = 0.5 * cv_z + 0.5 * semi_z
    df["lni_drilling"] = (cv_z + semi_z + support_z) / 3.0
    return df
