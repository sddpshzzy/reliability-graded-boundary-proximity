# Reliability graded boundary proximity for local nonstationary iron ore grade modelling

This repository contains the open-source Python implementation associated with the manuscript:

**Reliability graded boundary proximity for local nonstationary iron ore grade modelling**  
Corresponding author: Dr Zhiyuan Wu

The code implements the reproducible parts of the workflow used in the paper:

1. Local nonstationarity descriptors from borehole assay intervals.
2. Control-point registration of interpreted section coordinates to borehole collar coordinates.
3. Reliability grading of interpreted DXF sections.
4. Boundary proximity feature construction.
5. Borehole-grouped validation for testing boundary-feature effects.

The small example dataset in `examples/` is synthetic and anonymized. It is included only to demonstrate the workflow and quick-test the code. The raw mine dataset and original interpreted CAD drawings are not redistributed here.

## Repository Layout

```text
.
├── LICENSE
├── README.md
├── pyproject.toml
├── examples/
│   ├── synthetic_assays.csv
│   ├── synthetic_boundary_vertices.csv
│   └── synthetic_controls.csv
├── scripts/
│   └── run_quick_test.py
└── src/
    └── boundary_proximity/
        ├── __init__.py
        ├── boundaries.py
        ├── features.py
        ├── registration.py
        └── validation.py
```

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Quick Test

Run the example workflow from the repository root:

```bash
python scripts/run_quick_test.py
```

Expected behavior:

- local nonstationarity columns are created;
- synthetic section registration is fitted and graded;
- boundary proximity columns are created;
- borehole-grouped validation compares a local nonstationarity baseline with a boundary-feature model;
- a short metrics table is printed to the terminal.

The quick test is deliberately small, so the absolute metric values should not be interpreted as the paper results.

## Citation

If you use this code, please cite the manuscript after publication:

Wu, Z., Xu, X., Gu, X., Zheng, X., Zhao, Y., Li, C., Qin, M. Reliability graded boundary proximity for local nonstationary iron ore grade modelling.

## License

This project is released under the MIT License. In practical terms, the author grants permission to use, copy, modify, and redistribute the code only under the license terms included in this repository. Please retain the copyright and license notice and cite the associated manuscript when using or adapting the workflow.
