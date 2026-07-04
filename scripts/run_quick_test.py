#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd

from boundary_proximity import (
    add_boundary_proximity,
    add_local_nonstationarity,
    fit_section_registration,
    grade_section_reliability,
    grouped_model_comparison,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def main() -> None:
    assays = pd.read_csv(EXAMPLES / "synthetic_assays.csv")
    controls = pd.read_csv(EXAMPLES / "synthetic_controls.csv")
    boundaries = pd.read_csv(EXAMPLES / "synthetic_boundary_vertices.csv")

    assays = add_local_nonstationarity(assays, radius=90.0)

    registrations = []
    for _, section_controls in controls.groupby("section_id"):
        reg = fit_section_registration(section_controls)
        registrations.append(
            {
                "section_id": reg.section_id,
                "n_controls": reg.n_controls,
                "rmse_horizontal": reg.rmse_horizontal,
                "rmse_vertical": reg.rmse_vertical,
                "grade": grade_section_reliability(reg),
            }
        )
    print("Section reliability diagnostics")
    print(pd.DataFrame(registrations).to_string(index=False))

    assays = add_boundary_proximity(assays, boundaries)
    metrics = grouped_model_comparison(assays, n_splits=4)
    summary = metrics.groupby("model")[["RMSE", "MAE", "R2", "Bias"]].mean().reset_index()

    print("\nBorehole-grouped validation summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
