"""Small deterministic demo helpers for the bundled test data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_demo_bin_assignments(
    truth: pd.DataFrame,
    feature_table: pd.DataFrame | None = None,
    unbin_fraction: float = 0.08,
    misbin_fraction: float = 0.10,
) -> pd.DataFrame:
    """Create deterministic toy bin assignments from ground truth.

    The repository's example data ships with ground truth but no binner output.
    This helper creates a clearly labelled demonstration target so notebooks and
    tests run end-to-end. It should be replaced by real binner outputs for paper
    experiments.
    """

    required = {"contig", "contig_short", "genome"}
    missing = required.difference(truth.columns)
    if missing:
        raise ValueError(f"Truth table is missing required columns: {sorted(missing)}")

    demo = truth[["contig", "contig_short", "genome"]].copy()
    demo["bin"] = "bin_" + demo["genome"].astype(str)

    if feature_table is not None:
        risk = feature_table[["contig_short", "degree", "coverage_neighbor_abs_diff", "lies_in_cycle"]].copy()
        risk["demo_risk"] = (
            risk["degree"].fillna(0).rank(pct=True)
            + risk["coverage_neighbor_abs_diff"].fillna(0).rank(pct=True)
            + risk["lies_in_cycle"].fillna(0)
        )
        demo = demo.merge(risk[["contig_short", "demo_risk"]], on="contig_short", how="left")
        demo["demo_risk"] = demo["demo_risk"].fillna(0)
        demo = demo.sort_values(["demo_risk", "contig_short"], ascending=[False, True]).reset_index(drop=True)
    else:
        demo = demo.sort_values("contig_short").reset_index(drop=True)

    n_rows = len(demo)
    n_unbin = max(1, int(round(n_rows * unbin_fraction)))
    n_misbin = max(1, int(round(n_rows * misbin_fraction)))

    demo.loc[: n_unbin - 1, "bin"] = np.nan

    genomes = sorted(demo["genome"].dropna().unique())
    if len(genomes) > 1:
        genome_to_wrong = {genome: genomes[(idx + 1) % len(genomes)] for idx, genome in enumerate(genomes)}
        start = n_unbin
        stop = min(n_rows, n_unbin + n_misbin)
        for idx in range(start, stop):
            wrong_genome = genome_to_wrong[demo.loc[idx, "genome"]]
            demo.loc[idx, "bin"] = f"bin_{wrong_genome}"

    return demo[["contig", "contig_short", "bin"]].sort_values("contig").reset_index(drop=True)

