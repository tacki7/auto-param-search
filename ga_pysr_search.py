#!/usr/bin/env python3
"""Symbolic feature search backed by PySR.

Drop-in replacement for ga_feature_search.py. Reads a CSV or Parquet file,
treats every column as a variable, and uses PySR (genetic-programming symbolic
regression, Julia-backed) to search for arithmetic/nonlinear combinations of
variables that predict a chosen target column. PySR optimises prediction error;
this script then ranks every equation on its Pareto front by |correlation| with
the target, prints the top ones as formulas, and saves a math-rendered image.

Requires the PySR virtual environment:
    .venv_pysr/bin/python ga_pysr_search.py data.csv --target y --time-limit 60
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Correlation (matches ga_feature_search.py behaviour)
# --------------------------------------------------------------------------- #

def correlation(values: np.ndarray, target: np.ndarray) -> float:
    """Pearson correlation, robust to NaN/inf, constant series, and degenerate
    expressions that are finite on only a few rows."""
    v = np.asarray(values, dtype=float)
    t = np.asarray(target, dtype=float)
    mask = np.isfinite(v) & np.isfinite(t)
    if mask.sum() < max(3, int(0.5 * len(v))):
        return 0.0
    v, t = v[mask], t[mask]
    if np.std(v) < 1e-12 or np.std(t) < 1e-12:
        return 0.0
    c = np.corrcoef(v, t)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #

def load_data(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith(".parquet") or lower.endswith(".pq"):
        return pd.read_parquet(path)
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# Result image (sympy LaTeX via matplotlib mathtext)
# --------------------------------------------------------------------------- #

def save_results_image(entries, target_name, path, math=True):
    """entries: list of (rank, abs_r, signed_r, latex_str, text_str)."""
    import textwrap

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(entries)
    row_h = 0.85 if math else 0.55
    fig_h = max(2.5, row_h * n + 1.0)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Top {n} PySR equations by |correlation| with '{target_name}'",
        fontsize=14, fontweight="bold", loc="left", pad=14,
    )
    for i, (rank, c, signed, latex_str, text_str) in enumerate(entries):
        center = 1.0 - (i + 0.5) / n
        header = f"{rank:2d}.   |r| = {c:.4f}    r = {signed:+.4f}"
        ax.text(0.0, center + 0.32 / n, header, fontsize=11, fontweight="bold",
                family="monospace", va="center", transform=ax.transAxes)
        rendered = False
        if math and latex_str:
            try:
                ax.text(0.04, center - 0.20 / n, f"${latex_str}$", fontsize=14,
                        va="center", transform=ax.transAxes, color="#222222")
                rendered = True
            except Exception:
                rendered = False
        if not rendered:
            wrapped = textwrap.wrap(text_str, width=90) or [""]
            ax.text(0.04, center - 0.20 / n, "\n".join(wrapped), fontsize=9.5,
                    family="monospace", va="center", transform=ax.transAxes,
                    color="#444444")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="Input CSV or Parquet file")
    ap.add_argument("--target", required=True, help="Target (objective) column name")
    ap.add_argument("--columns", nargs="*", default=None,
                    help="Subset of feature columns (default: all numeric except target)")
    ap.add_argument("--time-limit", type=float, default=None,
                    help="Wall-clock search limit in seconds (governs when set)")
    ap.add_argument("--iterations", type=int, default=40,
                    help="PySR iterations when no time limit is given")
    ap.add_argument("--populations", type=int, default=15, help="PySR populations")
    ap.add_argument("--population-size", type=int, default=33, help="PySR population size")
    ap.add_argument("--maxsize", type=int, default=25, help="Max equation complexity")
    ap.add_argument("--top-k", type=int, default=10, help="How many top results to print")
    ap.add_argument("--image-top", type=int, default=20, help="How many results to draw")
    ap.add_argument("--output-image", default="pysr_results.png", help="Output PNG path")
    ap.add_argument("--ascii-image", action="store_true",
                    help="Render image formulas as plain text instead of math")
    ap.add_argument("--seed", type=int, default=None, help="Random seed (forces deterministic run)")
    ap.add_argument("--quiet", action="store_true", help="Suppress PySR progress output")
    args = ap.parse_args()

    df = load_data(args.input)
    if args.target not in df.columns:
        raise SystemExit(f"target column '{args.target}' not found. columns: {list(df.columns)}")

    numeric = df.select_dtypes(include=[np.number])
    if args.columns:
        feature_cols = [c for c in args.columns if c != args.target]
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise SystemExit(f"columns not found: {missing}")
    else:
        feature_cols = [c for c in numeric.columns if c != args.target]
    if not feature_cols:
        raise SystemExit("no feature columns available")

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[args.target], errors="coerce")
    # PySR/Julia cannot ingest NaN; drop rows with any missing value
    keep = X.notna().all(axis=1) & y.notna()
    X, y = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
    y_arr = y.to_numpy(dtype=float)

    # --- baseline: single-variable correlations ---
    print("=" * 70)
    print("Baseline single-variable correlations with target:", args.target)
    print("=" * 70)
    base = [(c, correlation(X[c].to_numpy(float), y_arr)) for c in feature_cols]
    base.sort(key=lambda x: abs(x[1]), reverse=True)
    for c, r in base:
        print(f"  {r:+.4f}   {c}")

    # --- PySR symbolic regression ---
    from pysr import PySRRegressor

    kwargs = dict(
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square", "sqrt", "log", "abs", "inv(x) = 1/x"],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        maxsize=args.maxsize,
        populations=args.populations,
        population_size=args.population_size,
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        model_selection="best",
        verbosity=0 if args.quiet else 1,
        progress=not args.quiet,
    )
    if args.time_limit is not None:
        kwargs["timeout_in_seconds"] = args.time_limit
        kwargs["niterations"] = 10_000_000  # let the clock stop the run
    else:
        kwargs["niterations"] = args.iterations
    if args.seed is not None:
        # deterministic runs require a single process and no multithreading
        kwargs.update(random_state=args.seed, deterministic=True,
                      procs=0, multithreading=False)

    print()
    print("=" * 70)
    print("PySR symbolic regression")
    print("=" * 70)
    model = PySRRegressor(**kwargs)
    model.fit(X, y_arr, variable_names=list(feature_cols))

    # --- rank every equation on the Pareto front by |correlation| ---
    import sympy

    eqs = model.equations_
    scored = []
    for idx in eqs.index:
        try:
            pred = model.predict(X, index=idx)
        except Exception:
            continue
        c = correlation(np.asarray(pred, dtype=float), y_arr)
        expr = eqs.loc[idx, "sympy_format"]
        try:
            latex_str = sympy.latex(expr)
        except Exception:
            latex_str = ""
        scored.append((abs(c), c, str(expr), latex_str))
    scored.sort(key=lambda t: t[0], reverse=True)

    print()
    print("=" * 70)
    print(f"Top {args.top_k} equations by |correlation| with {args.target}")
    print("=" * 70)
    for rank, (ac, sc, text, _latex) in enumerate(scored[:args.top_k], 1):
        print(f"{rank:3d}. |r|={ac:.4f}  (r={sc:+.4f})")
        print(f"      {text}")

    entries = [(rank, ac, sc, latex, text)
               for rank, (ac, sc, text, latex) in enumerate(scored[:args.image_top], 1)]
    save_results_image(entries, args.target, args.output_image, math=not args.ascii_image)
    print()
    print(f"Saved top {args.image_top} results image -> {args.output_image}")


if __name__ == "__main__":
    main()
