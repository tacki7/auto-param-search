#!/usr/bin/env python3
"""Genetic-algorithm feature search.

Reads a CSV or Parquet file, treats every column as a variable, and searches
for arithmetic combinations of variables (using +, -, *, /) whose correlation
with a chosen target column is as strong as possible. Candidate expressions are
represented as binary trees and evolved with a genetic algorithm. Search stops
after a generation limit or a time limit, then prints the top expressions and
their correlations, showing each discovered variable as a formula.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Operators
# --------------------------------------------------------------------------- #

def _protected_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Division that returns NaN where the denominator is (near) zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(a, b)
    out = np.where(np.abs(b) < 1e-12, np.nan, out)
    return out


# name -> (function, infix symbol)
OPERATORS: dict[str, tuple[Callable[[np.ndarray, np.ndarray], np.ndarray], str]] = {
    "add": (np.add, "+"),
    "sub": (np.subtract, "-"),
    "mul": (np.multiply, "*"),
    "div": (_protected_div, "/"),
}
OP_NAMES = list(OPERATORS.keys())


def _protected_log(a: np.ndarray) -> np.ndarray:
    """log(|x|); NaN where x is (near) zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(np.abs(a))
    return np.where(np.abs(a) < 1e-12, np.nan, out)


def _protected_recip(a: np.ndarray) -> np.ndarray:
    """1/x; NaN where x is (near) zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(1.0, a)
    return np.where(np.abs(a) < 1e-12, np.nan, out)


# name -> (function, formatter(operand_string) -> string)
UNARY_OPERATORS: dict[str, tuple[Callable[[np.ndarray], np.ndarray], Callable[[str], str]]] = {
    "log":   (_protected_log,               lambda s: f"log(|{s}|)"),
    "sqrt":  (lambda a: np.sqrt(np.abs(a)), lambda s: f"sqrt(|{s}|)"),
    "sq":    (lambda a: a * a,              lambda s: f"({s})^2"),
    "recip": (_protected_recip,             lambda s: f"(1/{s})"),
    "abs":   (np.abs,                       lambda s: f"|{s}|"),
}
UNARY_NAMES = list(UNARY_OPERATORS.keys())


# --------------------------------------------------------------------------- #
# Expression tree
# --------------------------------------------------------------------------- #

@dataclass
class Node:
    """A binary expression tree node.

    Leaf node: op is None. Holds either a column index (col) or a numeric
    constant (const) produced by algebraic simplification.
    Internal node: op is an operator name, left/right are child Nodes
    (unary nodes use left only).
    """
    op: str | None = None
    col: int | None = None
    const: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None

    def is_leaf(self) -> bool:
        return self.op is None

    def is_const(self) -> bool:
        return self.op is None and self.const is not None

    def is_unary(self) -> bool:
        return self.op in UNARY_OPERATORS

    def children(self) -> "list[Node]":
        out: list[Node] = []
        if self.left is not None:
            out.append(self.left)
        if self.right is not None:
            out.append(self.right)
        return out


def random_leaf(n_cols: int) -> Node:
    return Node(col=random.randrange(n_cols))


def random_tree(n_cols: int, max_depth: int, depth: int = 0) -> Node:
    """Grow a random tree. Deeper trees are more likely to stop at a leaf."""
    # force leaf at max depth; otherwise stop with growing probability
    if depth >= max_depth or (depth > 0 and random.random() < 0.3):
        return random_leaf(n_cols)
    # ~25% unary node, otherwise binary node
    if random.random() < 0.25:
        return Node(
            op=random.choice(UNARY_NAMES),
            left=random_tree(n_cols, max_depth, depth + 1),
        )
    return Node(
        op=random.choice(OP_NAMES),
        left=random_tree(n_cols, max_depth, depth + 1),
        right=random_tree(n_cols, max_depth, depth + 1),
    )


def tree_nodes(node: Node) -> list[Node]:
    """All nodes in the tree (pre-order)."""
    stack = [node]
    out: list[Node] = []
    while stack:
        cur = stack.pop()
        out.append(cur)
        stack.extend(cur.children())
    return out


def tree_depth(node: Node) -> int:
    ch = node.children()
    if not ch:
        return 0
    return 1 + max(tree_depth(c) for c in ch)


def clone(node: Node) -> Node:
    if node.is_leaf():
        return Node(col=node.col, const=node.const)
    return Node(
        op=node.op,
        left=clone(node.left) if node.left is not None else None,
        right=clone(node.right) if node.right is not None else None,
    )


def evaluate(node: Node, data: np.ndarray) -> np.ndarray:
    """Evaluate the tree against a (n_cols, n_rows) column-major array."""
    if node.is_leaf():
        if node.const is not None:
            return np.full(data.shape[1], node.const, dtype=float)
        return data[node.col]
    if node.is_unary():
        return UNARY_OPERATORS[node.op][0](evaluate(node.left, data))
    fn = OPERATORS[node.op][0]
    return fn(evaluate(node.left, data), evaluate(node.right, data))


def _fmt_const(v: float) -> str:
    if v == int(v):
        return str(int(v))
    return f"{v:.4g}"


def formula(node: Node, columns: list[str]) -> str:
    if node.is_leaf():
        if node.const is not None:
            return _fmt_const(node.const)
        return columns[node.col]
    if node.is_unary():
        return UNARY_OPERATORS[node.op][1](formula(node.left, columns))
    sym = OPERATORS[node.op][1]
    return f"({formula(node.left, columns)} {sym} {formula(node.right, columns)})"


# operator precedence for LaTeX parenthesization
_PREC = {"add": 1, "sub": 1, "mul": 2, "div": 3}


def _prec(node: Node) -> int:
    if node.is_leaf() or node.is_unary():
        return 3          # atomic / delimited
    return _PREC[node.op]


def to_latex(node: Node, columns: list[str]) -> str:
    """Render an expression tree as a matplotlib-mathtext string."""
    def wrap(n: Node, minprec: int) -> str:
        s = rec(n)
        if _prec(n) < minprec:
            return r"\left(" + s + r"\right)"
        return s

    def rec(n: Node) -> str:
        if n.is_leaf():
            if n.const is not None:
                return _fmt_const(n.const)
            return r"\mathrm{" + columns[n.col].replace("_", r"\_") + "}"
        if n.is_unary():
            inner = rec(n.left)
            if n.op == "log":
                return r"\log\!\left(\left|" + inner + r"\right|\right)"
            if n.op == "sqrt":
                return r"\sqrt{\left|" + inner + r"\right|}"
            if n.op == "abs":
                return r"\left|" + inner + r"\right|"
            if n.op == "recip":
                return r"\frac{1}{" + inner + "}"
            if n.op == "sq":
                return wrap(n.left, 3) + "^{2}"
        if n.op == "div":
            return r"\frac{" + rec(n.left) + "}{" + rec(n.right) + "}"
        if n.op == "mul":
            return wrap(n.left, 2) + r" \cdot " + wrap(n.right, 2)
        if n.op == "add":
            return rec(n.left) + " + " + rec(n.right)
        # sub
        return rec(n.left) + " - " + wrap(n.right, 2)

    return rec(node)


# --------------------------------------------------------------------------- #
# Algebraic simplification ("評価")
# --------------------------------------------------------------------------- #

def const_node(v: float) -> Node:
    return Node(const=float(v))


def tree_equal(a: Node, b: Node) -> bool:
    """Structural equality of two expression trees."""
    if a.is_leaf() and b.is_leaf():
        return a.col == b.col and a.const == b.const
    if a.op != b.op:
        return False
    if not tree_equal(a.left, b.left):
        return False
    if a.right is None or b.right is None:
        return a.right is None and b.right is None
    return tree_equal(a.right, b.right)


def simplify(node: Node) -> Node:
    """Fold constants and apply algebraic identities.

    Rules: constant folding (e.g. 1 - 1 -> 0), x - x -> 0, x / x -> 1,
    x + 0 -> x, x * 1 -> x, x * 0 -> 0, x / 1 -> x, 0 / x -> 0,
    1/(1/x) -> x, |(|x|)| -> |x|. Applied bottom-up.
    """
    if node.is_leaf():
        return node

    if node.is_unary():
        c = simplify(node.left)
        if c.is_const():
            val = UNARY_OPERATORS[node.op][0](np.array([c.const], dtype=float))[0]
            if np.isfinite(val):
                return const_node(float(val))
            return Node(op=node.op, left=c)  # keep symbolic; fitness prunes it
        if node.op == "recip" and c.is_unary() and c.op == "recip":
            return c.left                     # 1/(1/x) -> x
        if node.op == "abs" and c.is_unary() and c.op == "abs":
            return c                           # ||x|| -> |x|
        return Node(op=node.op, left=c)

    # binary
    l = simplify(node.left)
    r = simplify(node.right)
    op = node.op
    lc, rc = l.is_const(), r.is_const()

    if lc and rc:
        val = OPERATORS[op][0](np.array([l.const], float), np.array([r.const], float))[0]
        if np.isfinite(val):
            return const_node(float(val))

    if op == "add":
        if lc and l.const == 0:
            return r
        if rc and r.const == 0:
            return l
    elif op == "sub":
        if rc and r.const == 0:
            return l
        if tree_equal(l, r):
            return const_node(0.0)            # x - x -> 0
    elif op == "mul":
        if (lc and l.const == 0) or (rc and r.const == 0):
            return const_node(0.0)            # x * 0 -> 0
        if lc and l.const == 1:
            return r
        if rc and r.const == 1:
            return l
    elif op == "div":
        if lc and l.const == 0:
            return const_node(0.0)            # 0 / x -> 0
        if rc and r.const == 1:
            return l
        if tree_equal(l, r):
            return const_node(1.0)            # x / x -> 1

    return Node(op=op, left=l, right=r)


# --------------------------------------------------------------------------- #
# Fitness
# --------------------------------------------------------------------------- #

def correlation(values: np.ndarray, target: np.ndarray) -> float:
    """Pearson correlation, robust to NaN/inf and constant series."""
    v = np.asarray(values, dtype=float)
    mask = np.isfinite(v) & np.isfinite(target)
    # require enough finite rows: guards against degenerate expressions that
    # are NaN/inf on most rows and correlate spuriously on the few survivors
    if mask.sum() < max(3, int(0.5 * len(v))):
        return 0.0
    v = v[mask]
    t = target[mask]
    if np.std(v) < 1e-12 or np.std(t) < 1e-12:
        return 0.0
    c = np.corrcoef(v, t)[0, 1]
    if not np.isfinite(c):
        return 0.0
    return float(c)


def fitness(node: Node, data: np.ndarray, target: np.ndarray, size_penalty: float) -> float:
    """abs(correlation) minus a small penalty for tree size (parsimony)."""
    c = abs(correlation(evaluate(node, data), target))
    penalty = size_penalty * len(tree_nodes(node))
    return c - penalty


# --------------------------------------------------------------------------- #
# Genetic operators
# --------------------------------------------------------------------------- #

def _overwrite(dst: Node, src: Node) -> None:
    """Copy every field of src into dst in place (const included)."""
    dst.op = src.op
    dst.col = src.col
    dst.const = src.const
    dst.left = src.left
    dst.right = src.right


def crossover(a: Node, b: Node) -> Node:
    """Return a child: copy of a with a random subtree replaced by one from b."""
    child = clone(a)
    child_nodes = tree_nodes(child)
    donor = clone(random.choice(tree_nodes(b)))
    target_node = random.choice(child_nodes)
    _overwrite(target_node, donor)
    return child


def mutate(node: Node, n_cols: int, max_depth: int) -> Node:
    """Replace a random subtree with a fresh random subtree."""
    child = clone(node)
    nodes = tree_nodes(child)
    target_node = random.choice(nodes)
    replacement = random_tree(n_cols, max(1, max_depth - tree_depth(child) + 1))
    _overwrite(target_node, replacement)
    return child


def tournament(pop: list[Node], scores: list[float], k: int) -> Node:
    best_i = random.randrange(len(pop))
    for _ in range(k - 1):
        i = random.randrange(len(pop))
        if scores[i] > scores[best_i]:
            best_i = i
    return pop[best_i]


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #

def search(
    data: np.ndarray,
    target: np.ndarray,
    n_cols: int,
    *,
    population_size: int,
    generations: int,
    time_limit: float | None,
    max_depth: int,
    tournament_k: int,
    crossover_rate: float,
    mutation_rate: float,
    elitism: int,
    size_penalty: float,
    columns: list[str],
    verbose: bool,
) -> list[tuple[Node, float]]:
    pop = [simplify(random_tree(n_cols, max_depth)) for _ in range(population_size)]
    start = time.time()

    def score_all(individuals: list[Node]) -> list[float]:
        return [fitness(ind, data, target, size_penalty) for ind in individuals]

    scores = score_all(pop)
    gen = 0
    # When a time limit is given, run until it elapses (generations ignored).
    # Otherwise, run for the fixed number of generations.
    while True:
        elapsed = time.time() - start
        if time_limit is not None:
            if elapsed >= time_limit:
                if verbose:
                    print(f"[stop] time limit {time_limit:.0f}s reached at generation {gen}")
                break
        elif gen >= generations:
            break

        order = sorted(range(len(pop)), key=lambda i: scores[i], reverse=True)
        new_pop = [clone(pop[i]) for i in order[:elitism]]

        while len(new_pop) < population_size:
            p1 = tournament(pop, scores, tournament_k)
            if random.random() < crossover_rate:
                p2 = tournament(pop, scores, tournament_k)
                child = crossover(p1, p2)
            else:
                child = clone(p1)
            if random.random() < mutation_rate:
                child = mutate(child, n_cols, max_depth)
            # algebraic simplification each time an individual is built
            child = simplify(child)
            # guard against runaway growth
            if tree_depth(child) > max_depth + 2:
                child = simplify(random_tree(n_cols, max_depth))
            new_pop.append(child)

        pop = new_pop
        scores = score_all(pop)
        gen += 1

        if verbose:
            best_i = max(range(len(pop)), key=lambda i: scores[i])
            best_corr = abs(correlation(evaluate(pop[best_i], data), target))
            elapsed = time.time() - start
            print(f"gen {gen:4d} | best |r|={best_corr:.4f} | elapsed {elapsed:5.1f}s")

    # rank final population by true |correlation|, dedup by formula
    seen: dict[str, tuple[Node, float]] = {}
    for ind in pop:
        f = formula(ind, columns)
        c = abs(correlation(evaluate(ind, data), target))
        if f not in seen or c > seen[f][1]:
            seen[f] = (ind, c)
    ranked = sorted(seen.values(), key=lambda x: x[1], reverse=True)
    return ranked


# --------------------------------------------------------------------------- #
# Result image
# --------------------------------------------------------------------------- #

def save_results_image(
    ranked: list[tuple[Node, float]],
    feature_cols: list[str],
    data: np.ndarray,
    target: np.ndarray,
    target_name: str,
    path: str,
    top: int,
    math: bool = True,
) -> None:
    """Render the top expressions and correlations to a PNG image.

    math=True renders each formula as real mathematical notation (fractions,
    superscripts, roots) via matplotlib mathtext; on any render failure it
    falls back to the plain-text formula for that row.
    """
    import textwrap

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = ranked[:top]
    n = len(items)
    row_h = 0.85 if math else 0.55   # inches per result
    fig_h = max(2.5, row_h * n + 1.0)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Top {n} derived variables by |correlation| with '{target_name}'",
        fontsize=14, fontweight="bold", loc="left", pad=14,
    )

    for i, (node, c) in enumerate(items):
        rank = i + 1
        signed = correlation(evaluate(node, data), target)
        center = 1.0 - (i + 0.5) / n
        header = f"{rank:2d}.   |r| = {c:.4f}    r = {signed:+.4f}"
        ax.text(0.0, center + 0.32 / n, header, fontsize=11, fontweight="bold",
                family="monospace", va="center", transform=ax.transAxes)
        rendered = False
        if math:
            try:
                tex = "$" + to_latex(node, feature_cols) + "$"
                ax.text(0.04, center - 0.20 / n, tex, fontsize=14,
                        va="center", transform=ax.transAxes, color="#222222")
                rendered = True
            except Exception:
                rendered = False
        if not rendered:
            wrapped = textwrap.wrap(formula(node, feature_cols), width=90) or [""]
            ax.text(0.04, center - 0.20 / n, "\n".join(wrapped), fontsize=9.5,
                    family="monospace", va="center", transform=ax.transAxes,
                    color="#444444")

    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# IO / CLI
# --------------------------------------------------------------------------- #

def load_data(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith(".parquet") or lower.endswith(".pq"):
        return pd.read_parquet(path)
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    # fall back on extension guess
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="Input CSV or Parquet file")
    ap.add_argument("--target", required=True, help="Target (objective) column name")
    ap.add_argument("--columns", nargs="*", default=None,
                    help="Subset of feature columns to use (default: all numeric except target)")
    ap.add_argument("--generations", type=int, default=100, help="Max generations")
    ap.add_argument("--time-limit", type=float, default=None, help="Time limit in seconds")
    ap.add_argument("--population", type=int, default=300, help="Population size")
    ap.add_argument("--max-depth", type=int, default=3, help="Max expression tree depth")
    ap.add_argument("--top-k", type=int, default=10, help="How many top results to print")
    ap.add_argument("--image-top", type=int, default=20, help="How many top results to draw in the image")
    ap.add_argument("--output-image", default="results.png", help="Output PNG path for top results")
    ap.add_argument("--ascii-image", action="store_true",
                    help="Render image formulas as plain text instead of math notation")
    ap.add_argument("--tournament-k", type=int, default=3, help="Tournament selection size")
    ap.add_argument("--crossover-rate", type=float, default=0.7)
    ap.add_argument("--mutation-rate", type=float, default=0.3)
    ap.add_argument("--elitism", type=int, default=2, help="Elite individuals kept each gen")
    ap.add_argument("--size-penalty", type=float, default=0.001,
                    help="Parsimony penalty per tree node during selection")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    ap.add_argument("--quiet", action="store_true", help="Suppress per-generation logs")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    df = load_data(args.input)
    if args.target not in df.columns:
        raise SystemExit(f"target column '{args.target}' not found. columns: {list(df.columns)}")

    # select numeric feature columns
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

    target = pd.to_numeric(df[args.target], errors="coerce").to_numpy(dtype=float)
    # (n_cols, n_rows) column-major for fast leaf lookup
    data = np.vstack([pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
                      for c in feature_cols])
    n_cols = len(feature_cols)

    # --- baseline: single-variable correlations ---
    print("=" * 70)
    print("Baseline single-variable correlations with target:", args.target)
    print("=" * 70)
    base = [(c, correlation(data[i], target)) for i, c in enumerate(feature_cols)]
    base.sort(key=lambda x: abs(x[1]), reverse=True)
    for c, r in base:
        print(f"  {r:+.4f}   {c}")

    # --- GA search over derived features ---
    print()
    print("=" * 70)
    print("Genetic-algorithm search over derived variables")
    print("=" * 70)
    ranked = search(
        data, target, n_cols,
        population_size=args.population,
        generations=args.generations,
        time_limit=args.time_limit,
        max_depth=args.max_depth,
        tournament_k=args.tournament_k,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        elitism=args.elitism,
        size_penalty=args.size_penalty,
        columns=feature_cols,
        verbose=not args.quiet,
    )

    print()
    print("=" * 70)
    print(f"Top {args.top_k} derived variables by |correlation| with {args.target}")
    print("=" * 70)
    for rank, (node, c) in enumerate(ranked[:args.top_k], 1):
        signed = correlation(evaluate(node, data), target)
        print(f"{rank:3d}. |r|={c:.4f}  (r={signed:+.4f})")
        print(f"      {formula(node, feature_cols)}")

    # --- save top results as an image ---
    save_results_image(
        ranked, feature_cols, data, target, args.target,
        args.output_image, args.image_top, math=not args.ascii_image,
    )
    print()
    print(f"Saved top {args.image_top} results image -> {args.output_image}")


if __name__ == "__main__":
    main()
