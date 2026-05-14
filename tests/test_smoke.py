from pathlib import Path

import pandas as pd

from binfailgraph.datasets import discover_datasets
from binfailgraph.features import build_feature_table
from binfailgraph.features import _all_kmers, _kmer_frequencies
from binfailgraph.labels import make_contig_labels, task_frame
from binfailgraph.modeling import (
    COMPARISON_FEATURE_SETS,
    compare_feature_sets,
    make_logistic_regression,
    evaluate_classifier,
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
    task = task_frame(make_contig_labels(raw_features), task="misbin")

    feature_sets = {name: select_feature_columns(task, name) for name in COMPARISON_FEATURE_SETS}
    assert feature_sets["length_only"] == ["length"]
    assert feature_sets["coverage_only"] == ["coverage"]
    assert sum(column.startswith("kmer4_") for column in feature_sets["composition_coverage"]) == 136
    assert "coverage" in feature_sets["composition_coverage"]
    assert "degree" in feature_sets["graph_only"]
    assert "coverage" not in feature_sets["graph_only"]
    assert "gc_content" not in feature_sets["graph_only"]
    assert "degree" in feature_sets["composition_coverage_graph"]
    assert "coverage" in feature_sets["composition_coverage_graph"]

    comparison_table, results = compare_feature_sets(
        task.sample(n=min(120, len(task)), random_state=2),
        model_factory=lambda: make_logistic_regression(),
        feature_sets=COMPARISON_FEATURE_SETS,
        random_state=2,
    )
    assert set(comparison_table["feature_set"]) == set(COMPARISON_FEATURE_SETS)
    assert {"auroc", "auprc"}.issubset(comparison_table.columns)
    assert set(results) == set(COMPARISON_FEATURE_SETS)
