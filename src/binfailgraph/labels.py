"""Contig-level binning failure labels."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_contig_labels(
    feature_table: pd.DataFrame,
    contamination_threshold: float = 0.95,
    length_col: str = "length",
) -> pd.DataFrame:
    """Add contig-level correctness and failure labels.

    A binned contig is correct when its ground-truth genome matches the
    length-weighted majority genome of its assigned bin. A contig is a failure
    if it is mis-binned, is unbinned despite having a truth genome, or sits in a
    bin whose majority-genome purity is below ``contamination_threshold``.
    """

    required = {"contig", "genome", "bin"}
    missing = required.difference(feature_table.columns)
    if missing:
        raise ValueError(f"Feature table is missing required columns: {sorted(missing)}")

    df = feature_table.copy()
    df["is_known_truth"] = df["genome"].notna()
    df["is_binned"] = df["bin"].notna()

    binned_truth = df.loc[df["is_known_truth"] & df["is_binned"], ["bin", "genome", length_col]].copy()
    if binned_truth.empty:
        for column in [
            "bin_majority_genome",
            "bin_purity",
            "bin_truth_genome_count",
            "bin_has_mixed_truth",
            "bin_is_contaminated",
            "correctly_binned",
            "mis_binned",
            "unbinned_should_bin",
            "in_contaminated_bin",
            "label_misbin",
            "label_failure",
            "label_success",
        ]:
            df[column] = np.nan
        return df

    weights = binned_truth[length_col].fillna(1).clip(lower=1)
    binned_truth["_weight"] = weights

    weighted = (
        binned_truth.groupby(["bin", "genome"], dropna=False)["_weight"]
        .sum()
        .reset_index(name="genome_weight")
    )
    totals = weighted.groupby("bin")["genome_weight"].sum().rename("bin_truth_weight")
    genome_counts = weighted.groupby("bin")["genome"].nunique(dropna=True).rename("bin_truth_genome_count")
    majority_idx = weighted.groupby("bin")["genome_weight"].idxmax()
    majority = weighted.loc[majority_idx].set_index("bin")
    majority = majority.join(totals).join(genome_counts)
    majority["bin_purity"] = majority["genome_weight"] / majority["bin_truth_weight"]
    majority = majority.rename(columns={"genome": "bin_majority_genome"})

    df = df.merge(
        majority[["bin_majority_genome", "bin_purity", "bin_truth_genome_count"]],
        left_on="bin",
        right_index=True,
        how="left",
    )
    df["bin_has_mixed_truth"] = df["bin_truth_genome_count"].fillna(0).astype(int) > 1
    df["bin_is_contaminated"] = df["is_binned"] & (df["bin_purity"].fillna(0) < contamination_threshold)
    df["correctly_binned"] = (
        df["is_binned"]
        & df["is_known_truth"]
        & (df["genome"].astype(str) == df["bin_majority_genome"].astype(str))
    )
    df["mis_binned"] = df["is_binned"] & df["is_known_truth"] & ~df["correctly_binned"]
    df["unbinned_should_bin"] = df["is_known_truth"] & ~df["is_binned"]
    df["in_contaminated_bin"] = df["is_binned"] & df["bin_is_contaminated"]

    df["label_misbin"] = df["mis_binned"].astype(int)
    df["label_failure"] = (
        df["unbinned_should_bin"] | df["mis_binned"] | df["in_contaminated_bin"]
    ).astype(int)
    df["label_success"] = (
        df["correctly_binned"] & ~df["in_contaminated_bin"]
    ).astype(int)
    return df


def task_frame(
    labelled_table: pd.DataFrame,
    task: str = "failure",
    target_col: str = "target",
) -> pd.DataFrame:
    """Return rows and target column for a paper task.

    ``task="misbin"`` keeps binned contigs with truth labels and predicts
    correct versus incorrect bin membership. The target convention is
    ``1 = correct`` and ``0 = incorrect``. ``task="failure"`` keeps all
    contigs with truth labels and uses ``1 = success`` and ``0 = failure``.
    """

    if task == "misbin":
        subset = labelled_table.loc[
            labelled_table["is_known_truth"] & labelled_table["is_binned"]
        ].copy()
        subset[target_col] = subset["correctly_binned"].astype(int)
        return subset
    if task == "failure":
        subset = labelled_table.loc[labelled_table["is_known_truth"]].copy()
        subset[target_col] = subset["label_success"].astype(int)
        return subset
    raise ValueError("task must be 'misbin' or 'failure'")


def summarize_labels(labelled_table: pd.DataFrame) -> pd.Series:
    """Compact label counts for sanity-checking notebooks and tests."""

    columns = [
        "is_known_truth",
        "is_binned",
        "correctly_binned",
        "mis_binned",
        "unbinned_should_bin",
        "in_contaminated_bin",
        "label_failure",
    ]
    present = [column for column in columns if column in labelled_table]
    return labelled_table[present].sum(numeric_only=True)
