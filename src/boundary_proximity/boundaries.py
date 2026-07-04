from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def add_boundary_proximity(
    assays: pd.DataFrame,
    boundary_vertices: pd.DataFrame,
    *,
    section_col: str = "section_id",
    sample_x_col: str = "section_x",
    sample_y_col: str = "section_y",
    boundary_x_col: str = "boundary_x",
    boundary_y_col: str = "boundary_y",
    near_threshold: float = 25.0,
    transition_threshold: float = 75.0,
) -> pd.DataFrame:
    """Add nearest interpreted-boundary distance and boundary-zone class.

    This feature is section relative. It should only be applied to sections
    that have passed the registration evidence gate in the manuscript workflow.
    """

    df = assays.copy()
    df["boundary_distance"] = np.nan
    df["boundary_zone"] = "unassigned"

    for section_id, group in df.groupby(section_col):
        b = boundary_vertices[boundary_vertices[section_col].astype(str) == str(section_id)]
        if b.empty:
            continue
        tree = cKDTree(b[[boundary_x_col, boundary_y_col]].to_numpy(float))
        distances, _ = tree.query(group[[sample_x_col, sample_y_col]].to_numpy(float), k=1)
        df.loc[group.index, "boundary_distance"] = distances

    d = df["boundary_distance"].to_numpy(float)
    zones = np.full(len(df), "far", dtype=object)
    zones[d <= near_threshold] = "near"
    zones[(d > near_threshold) & (d <= transition_threshold)] = "transition"
    zones[~np.isfinite(d)] = "unassigned"
    df["boundary_zone"] = zones
    df["boundary_zone_code"] = pd.Categorical(
        df["boundary_zone"],
        categories=["near", "transition", "far", "unassigned"],
        ordered=False,
    ).codes
    return df
