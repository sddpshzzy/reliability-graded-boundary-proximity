from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SectionRegistration:
    section_id: str
    horizontal_coefficients: tuple[float, float, float]
    vertical_coefficients: tuple[float, float]
    rmse_horizontal: float
    rmse_vertical: float
    n_controls: int


def _rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def fit_section_registration(
    controls: pd.DataFrame,
    *,
    section_col: str = "section_id",
    x_col: str = "collar_x",
    y_col: str = "collar_y",
    elev_col: str = "collar_z",
    dxf_x_col: str = "dxf_x",
    dxf_y_col: str = "dxf_y",
) -> SectionRegistration:
    """Fit the section registration model used by the paper.

    The horizontal drawing coordinate is fitted as a linear function of collar
    X and Y. The vertical drawing coordinate is fitted as a linear function of
    collar elevation. The function returns residual diagnostics used by the
    reliability-grading step.
    """

    if controls.empty:
        raise ValueError("controls must contain at least one retained control point")
    section_ids = controls[section_col].astype(str).unique()
    if len(section_ids) != 1:
        raise ValueError("fit one section at a time")

    c = controls.copy()
    h_design = np.column_stack([c[x_col].to_numpy(float), c[y_col].to_numpy(float), np.ones(len(c))])
    h_coef, *_ = np.linalg.lstsq(h_design, c[dxf_x_col].to_numpy(float), rcond=None)
    h_pred = h_design @ h_coef

    v_design = np.column_stack([c[elev_col].to_numpy(float), np.ones(len(c))])
    v_coef, *_ = np.linalg.lstsq(v_design, c[dxf_y_col].to_numpy(float), rcond=None)
    v_pred = v_design @ v_coef

    return SectionRegistration(
        section_id=str(section_ids[0]),
        horizontal_coefficients=tuple(float(v) for v in h_coef),
        vertical_coefficients=tuple(float(v) for v in v_coef),
        rmse_horizontal=_rmse(c[dxf_x_col].to_numpy(float), h_pred),
        rmse_vertical=_rmse(c[dxf_y_col].to_numpy(float), v_pred),
        n_controls=int(len(c)),
    )


def grade_section_reliability(
    registration: SectionRegistration,
    *,
    min_a_controls: int = 5,
    min_b_controls: int = 3,
    max_a_rmse: float = 25.0,
    max_b_rmse: float = 60.0,
) -> str:
    """Assign A/B/C reliability grade from control support and residuals."""

    residual = max(registration.rmse_horizontal, registration.rmse_vertical)
    if registration.n_controls >= min_a_controls and residual <= max_a_rmse:
        return "A"
    if registration.n_controls >= min_b_controls and residual <= max_b_rmse:
        return "B"
    return "C"
