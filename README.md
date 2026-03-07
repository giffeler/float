# Floating Bodies Monte Carlo Prototype

This repository contains a first reduced simulation prototype for a research question about whether two elongated floating bodies might experience a net inward tendency because wave events affect their outer sides differently from their inner sides.

The implementation is intentionally minimal and falsifiable:

- `notebooks/minimal_event_model.ipynb` is the main research notebook.
- `src/float_sim/event_model.py` contains the typed simulation helpers used by the notebook.
- `tests/test_event_model.py` contains lightweight checks for geometry, symmetry, and reproducibility.

The model is a 2D top-view event-based Monte Carlo approximation. It is not a CFD model and should not be interpreted as one.

## Setup with uv

This project is configured as a `uv`-managed Python package using a `src/` layout.

Create a project-local virtual environment:

```bash
uv venv .venv
source .venv/bin/activate
```

Install runtime, test, and notebook dependencies into the active environment:

```bash
uv sync --active --group dev --extra notebook
```

Run the test suite:

```bash
pytest -q
```

Start JupyterLab for the notebook workflow:

```bash
jupyter lab
```

The notebook is configured to use the project kernel `Python (float-sim)`. If Jupyter opens it with a different kernel, switch it manually in the notebook UI before running cells.

If Matplotlib warns that `~/.matplotlib` is not writable on your machine, set a project-local cache directory before starting tests or notebooks:

```bash
export MPLCONFIGDIR="$PWD/.matplotlib"
```
