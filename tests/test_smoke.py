from pathlib import Path

import pandas as pd

from binfailgraph.datasets import discover_datasets
from binfailgraph.features import build_feature_table
from binfailgraph.features import _all_kmers, _kmer_frequencies
from binfailgraph.labels import make_contig_labels, task_frame
from binfailgraph.modeling import (
    COMPARISON_FEATURE_SETS,
    combined_dataset_metric_table,
    combined_roc_curve_frame,
    combined_task_frame,
    comparison_feature_columns,
    feature_outcome_significance_table,
    compare_feature_sets,
    make_logistic_regression,
    evaluate_classifier,
    plot_feature_boxplots_by_outcome,
    select_feature_columns,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "tests" / "data"
DATA = DATA_ROOT / "Sim-5G"


def test_dataset_discovery_finds_current_examples():
    datasets = discover_datasets(DATA_ROOT)
    names = {dataset.name for dataset in datasets}

    assert {"Sim-5G", "Sim-10G"}.issubset(names)
    for dataset in datasets:
        assert dataset.graph_file.exists()
        assert dataset.contigs_file.exists()
        assert dataset.contig_paths_file.exists()
        assert dataset.ground_truth_file.exists()
        assert dataset.bin_assignments_file.exists()


def test_all_discovered_datasets_build_feature_tables():
    for dataset in discover_datasets(DATA_ROOT):
        raw_features = build_feature_table(
            graph_file=dataset.graph_file,
            contigs_file=dataset.contigs_file,
            contig_paths_file=dataset.contig_paths_file,
            ground_truth_file=dataset.ground_truth_file,
            bin_assignments_file=dataset.bin_assignments_file,
            include_kmers=False,
        )
        labelled = make_contig_labels(raw_features)
        task = task_frame(labelled, task="misbin")

        assert len(raw_features) > 0
        assert raw_features["bin"].notna().sum() > 0
        assert task["target"].nunique() == 2
        assert (task["target"] == task["correctly_binned"].astype(int)).all()


def test_feature_and_label_pipeline_smoke():
    raw_features = build_feature_table(
        graph_file=DATA / "assembly_graph_with_scaffolds.gfa",
        contigs_file=DATA / "contigs.fasta",
        contig_paths_file=DATA / "contigs.paths",
        ground_truth_file=DATA / "ground_truth.csv",
        bin_assignments_file=DATA / "initial_contig_bins.csv",
        include_kmers=False,
    )
    assert raw_features.shape[0] == 519
    assert {"degree", "coverage_neighbor_abs_diff", "bin_size"}.issubset(raw_features.columns)
    assert not any(column.startswith("kmer4_") for column in raw_features.columns)
    removed_columns = {
        "in_degree",
        "out_degree",
        "weighted_degree",
        "branching_score",
        "bubble_like_cycle_membership",
        "component_size",
        "component_cycle_count",
        "component_predicted_bin_diversity",
    }
    assert removed_columns.isdisjoint(raw_features.columns)
    assert raw_features["bin"].notna().sum() == 209

    labelled = make_contig_labels(raw_features)
    task = task_frame(labelled, task="misbin")

    assert task["target"].nunique() == 2
    assert task["target"].sum() > 0
    assert (task["target"] == 0).sum() > 0
    assert (task["target"] == task["correctly_binned"].astype(int)).all()

    columns = select_feature_columns(task, feature_set="post_binning")
    assert "degree" in columns
    assert "neighbor_same_bin_fraction" in columns
    assert removed_columns.isdisjoint(columns)

    small = task.sample(n=min(120, len(task)), random_state=1)
    result = evaluate_classifier(
        small,
        make_logistic_regression(),
        columns,
        test_size=0.35,
        random_state=1,
    )
    assert pd.notna(result.metrics["f1"])
    assert pd.notna(result.metrics["correct_rate_test"])
    assert "correctness_score" in result.test_predictions.columns
    assert not result.feature_importance.empty


def test_tetranucleotides_are_collapsed_by_reverse_complement():
    kmers = _all_kmers(k=4)

    assert len(kmers) == 136
    assert "AAAA" in kmers
    assert "TTTT" not in kmers

    frequencies = _kmer_frequencies("AAAATTTT", kmers=kmers, k=4)
    assert frequencies["kmer4_AAAA"] == 2 / 5


def test_requested_feature_sets_are_available():
    raw_features = build_feature_table(
        graph_file=DATA / "assembly_graph_with_scaffolds.gfa",
        contigs_file=DATA / "contigs.fasta",
        contig_paths_file=DATA / "contigs.paths",
        ground_truth_file=DATA / "ground_truth.csv",
        bin_assignments_file=DATA / "initial_contig_bins.csv",
        include_kmers=True,
    )
    assert "4mer_composition_distance" in raw_features.columns
    assert "coverage_difference" in raw_features.columns
    assert not any(column.startswith("kmer4_") for column in raw_features.columns)
    binned = raw_features.loc[raw_features["bin"].notna()]
    expected_coverage_difference = (
        binned["coverage"] - binned.groupby("bin")["coverage"].transform("mean")
    ).abs()
    assert (binned["coverage_difference"] - expected_coverage_difference).abs().max() == 0

    task = task_frame(make_contig_labels(raw_features), task="misbin")

    feature_sets = {name: select_feature_columns(task, name) for name in COMPARISON_FEATURE_SETS}
    assert feature_sets["length_only"] == ["length"]
    assert feature_sets["coverage_only"] == ["coverage_difference"]
    assert not any(column.startswith("kmer4_") for columns in feature_sets.values() for column in columns)
    assert "4mer_composition_distance" in feature_sets["composition_coverage"]
    assert "coverage_difference" in feature_sets["composition_coverage"]
    assert "coverage" not in feature_sets["composition_coverage"]
    assert "degree" in feature_sets["graph_only"]
    assert "coverage" not in feature_sets["graph_only"]
    assert "coverage_difference" not in feature_sets["graph_only"]
    assert "gc_content" not in feature_sets["graph_only"]
    assert "4mer_composition_distance" not in feature_sets["graph_only"]
    assert "degree" in feature_sets["composition_coverage_graph"]
    assert "coverage_difference" in feature_sets["composition_coverage_graph"]
    assert "coverage" not in feature_sets["composition_coverage_graph"]
    assert "4mer_composition_distance" in feature_sets["composition_coverage_graph"]

    comparison_table, results = compare_feature_sets(
        task.sample(n=min(120, len(task)), random_state=2),
        model_factory=lambda: make_logistic_regression(),
        feature_sets=COMPARISON_FEATURE_SETS,
        random_state=2,
    )
    assert set(comparison_table["feature_set"]) == set(COMPARISON_FEATURE_SETS)
    assert {"auroc", "auprc"}.issubset(comparison_table.columns)
    assert set(results) == set(COMPARISON_FEATURE_SETS)

    combined_results = {"Sim-5G": results}
    combined_table = combined_dataset_metric_table(
        combined_results,
        feature_sets=COMPARISON_FEATURE_SETS,
    )
    assert set(combined_table["feature_set"]) == set(COMPARISON_FEATURE_SETS)
    assert {"auroc", "auprc", "n_test", "correct_rate"}.issubset(combined_table.columns)

    curve = combined_roc_curve_frame(combined_results, feature_set="length_only")
    assert {"fpr", "tpr", "threshold"}.issubset(curve.columns)
    assert curve["fpr"].between(0, 1).all()
    assert curve["tpr"].between(0, 1).all()


def test_comparison_feature_boxplot_inputs_are_available():
    raw_features = build_feature_table(
        graph_file=DATA / "assembly_graph_with_scaffolds.gfa",
        contigs_file=DATA / "contigs.fasta",
        contig_paths_file=DATA / "contigs.paths",
        ground_truth_file=DATA / "ground_truth.csv",
        bin_assignments_file=DATA / "initial_contig_bins.csv",
        include_kmers=True,
    )
    task = task_frame(make_contig_labels(raw_features), task="misbin")
    pooled = combined_task_frame({"Sim-5G": task})
    columns = comparison_feature_columns(pooled)

    assert set(pooled.loc[pooled["target"] == 1, "binning_outcome"]) == {"Correct"}
    assert set(pooled.loc[pooled["target"] == 0, "binning_outcome"]) == {"Failed"}
    assert "4mer_composition_distance" in columns
    assert "length" in columns
    assert "coverage_difference" in columns
    assert "coverage" not in columns
    assert "degree" in columns
    assert all(not column.startswith("kmer4_") for column in columns)

    significance = feature_outcome_significance_table(
        pooled,
        feature_columns=["length", "coverage_difference", "4mer_composition_distance"],
    )
    assert {"p_value", "p_value_adj", "significance"}.issubset(significance.columns)
    assert set(significance["feature"]) == {
        "length",
        "coverage_difference",
        "4mer_composition_distance",
    }

    figures = plot_feature_boxplots_by_outcome(
        pooled,
        feature_columns=["length", "coverage_difference", "4mer_composition_distance"],
        features_per_figure=3,
        ncols=2,
    )
    assert len(figures) == 1
    assert any(text.get_text().splitlines()[0] in {"ns", "*", "**", "***", "****"} for text in figures[0].axes[0].texts)

    import matplotlib.pyplot as plt

    plt.close("all")
