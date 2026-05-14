"""Feature extraction for graph-aware binning failure diagnostics."""

from __future__ import annotations

import itertools
import math
from collections import deque
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from binfailgraph.graph import graph_vertex_names, load_spades_contig_graph
from binfailgraph.io import (
    merge_by_contig_alias,
    parse_spades_contig_name,
    read_bin_assignments,
    read_ground_truth,
)

DNA_ALPHABET = "ACGT"
DEFAULT_K = 4
DNA_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _reverse_complement(kmer: str) -> str:
    return kmer.translate(DNA_COMPLEMENT)[::-1]


def _canonical_kmer(kmer: str) -> str:
    reverse_complement = _reverse_complement(kmer)
    return min(kmer, reverse_complement)


def _all_kmers(k: int = DEFAULT_K) -> list[str]:
    canonical_kmers = {
        _canonical_kmer("".join(chars)) for chars in itertools.product(DNA_ALPHABET, repeat=k)
    }
    return sorted(canonical_kmers)


def _kmer_frequencies(sequence: str, kmers: list[str], k: int = DEFAULT_K) -> dict[str, float]:
    counts = Counter()
    valid = 0
    seq = sequence.upper()
    for idx in range(max(0, len(seq) - k + 1)):
        kmer = seq[idx : idx + k]
        if set(kmer).issubset(DNA_ALPHABET):
            counts[_canonical_kmer(kmer)] += 1
            valid += 1
    if valid == 0:
        return {f"kmer{k}_{kmer}": 0.0 for kmer in kmers}
    return {f"kmer{k}_{kmer}": counts[kmer] / valid for kmer in kmers}


def _safe_entropy(values: pd.Series) -> float:
    clean = values.dropna()
    if clean.empty:
        return 0.0
    proportions = clean.value_counts(normalize=True)
    return float(-(proportions * np.log2(proportions)).sum())


def _n50(lengths: pd.Series) -> float:
    clean = sorted([float(x) for x in lengths.dropna() if x > 0], reverse=True)
    if not clean:
        return 0.0
    half = sum(clean) / 2
    running = 0.0
    for length in clean:
        running += length
        if running >= half:
            return length
    return clean[-1]


def _to_networkx(graph) -> nx.Graph:
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(graph.vcount()))
    nx_graph.add_edges_from(graph.get_edgelist())
    return nx_graph


def _multi_source_shortest_path_lengths(adjacency: list[set[int]], sources: list[int]) -> dict[int, int]:
    distances = {source: 0 for source in sources}
    queue = deque(sources)
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return distances


def _basic_contig_features(contig_graph, include_kmers: bool = True, k: int = DEFAULT_K) -> pd.DataFrame:
    kmers = _all_kmers(k)
    records = []

    for contig in graph_vertex_names(contig_graph):
        parsed = parse_spades_contig_name(contig)
        sequence = str(contig_graph.get_contig_sequence(contig))
        length = len(sequence) if sequence else parsed["length_from_name"]
        gc_content = np.nan
        if sequence:
            gc_content = (sequence.upper().count("G") + sequence.upper().count("C")) / len(sequence)

        record = {
            **parsed,
            "length": length,
            "gc_content": gc_content,
        }
        if include_kmers:
            record.update(_kmer_frequencies(sequence, kmers=kmers, k=k))
        records.append(record)

    return pd.DataFrame.from_records(records)


def _topology_features(contig_graph, basic: pd.DataFrame) -> pd.DataFrame:
    graph = contig_graph.graph
    n_vertices = graph.vcount()
    nx_graph = _to_networkx(graph)
    adjacency = [set(graph.neighbors(idx)) for idx in range(n_vertices)]
    degree = np.asarray(graph.degree(), dtype=float)

    second_hop_counts = []
    for idx, neighbors in enumerate(adjacency):
        second_hop = set()
        for neighbor in neighbors:
            second_hop.update(adjacency[neighbor])
        second_hop.discard(idx)
        second_hop -= neighbors
        second_hop_counts.append(len(second_hop))

    try:
        clustering = graph.transitivity_local_undirected(mode="zero")
    except TypeError:
        clustering = graph.transitivity_local_undirected()
        clustering = [0.0 if value is None or math.isnan(value) else value for value in clustering]

    try:
        closeness = graph.closeness(normalized=True)
    except TypeError:
        closeness = graph.closeness()
    betweenness = graph.betweenness()
    pagerank = graph.pagerank()

    articulation_points = set(graph.articulation_points())
    bridge_nodes: set[int] = set()
    for edge_id in graph.bridges():
        bridge_nodes.update(graph.es[edge_id].tuple)

    cycle_nodes: set[int] = set()
    for cycle in nx.cycle_basis(nx_graph):
        cycle_nodes.update(cycle)

    branch_nodes = [idx for idx, value in enumerate(degree) if value > 2]
    if branch_nodes:
        branch_distances = _multi_source_shortest_path_lengths(adjacency, branch_nodes)
        shortest_path_to_branch = [float(branch_distances.get(idx, np.nan)) for idx in range(n_vertices)]
    else:
        shortest_path_to_branch = [np.nan] * n_vertices

    records = []
    for idx in range(n_vertices):
        records.append(
            {
                "contig": basic.loc[idx, "contig"],
                "degree": degree[idx],
                "neighbor_count": len(adjacency[idx]),
                "second_hop_neighbor_count": second_hop_counts[idx],
                "local_clustering_coefficient": clustering[idx],
                "betweenness_centrality": betweenness[idx],
                "closeness_centrality": closeness[idx],
                "pagerank": pagerank[idx],
                "is_tip": int(degree[idx] <= 1),
                "is_articulation_point": int(idx in articulation_points),
                "is_incident_to_bridge": int(idx in bridge_nodes),
                "lies_in_cycle": int(idx in cycle_nodes),
                "shortest_path_to_branch_node": shortest_path_to_branch[idx],
            }
        )

    return pd.DataFrame.from_records(records)


def _neighbor_numeric_features(
    graph,
    frame: pd.DataFrame,
    include_kmers: bool,
    k: int,
) -> pd.DataFrame:
    adjacency = [graph.neighbors(idx) for idx in range(graph.vcount())]
    coverage = frame["coverage"].astype(float).to_numpy()
    gc = frame["gc_content"].astype(float).to_numpy()

    kmer_columns = [column for column in frame.columns if column.startswith(f"kmer{k}_")]
    kmer_matrix = frame[kmer_columns].to_numpy(dtype=float) if include_kmers and kmer_columns else None

    records = []
    for idx, neighbors in enumerate(adjacency):
        neighbor_cov = coverage[neighbors] if neighbors else np.asarray([])
        neighbor_gc = gc[neighbors] if neighbors else np.asarray([])

        cov_median = float(np.nanmedian(neighbor_cov)) if len(neighbor_cov) else np.nan
        cov_mean = float(np.nanmean(neighbor_cov)) if len(neighbor_cov) else np.nan
        cov_std = float(np.nanstd(neighbor_cov)) if len(neighbor_cov) else np.nan
        gc_median = float(np.nanmedian(neighbor_gc)) if len(neighbor_gc) else np.nan

        coverage_log2_ratio = np.nan
        repeat_likeness = np.nan
        if cov_median and not np.isnan(cov_median) and cov_median > 0 and coverage[idx] > 0:
            coverage_log2_ratio = float(np.log2(coverage[idx] / cov_median))
            repeat_likeness = float(coverage[idx] / cov_median)

        kmer_distance = np.nan
        if kmer_matrix is not None and neighbors:
            node_vector = kmer_matrix[idx]
            neighbor_vector = np.nanmean(kmer_matrix[neighbors], axis=0)
            denominator = np.linalg.norm(node_vector) * np.linalg.norm(neighbor_vector)
            if denominator > 0:
                kmer_distance = float(1 - (np.dot(node_vector, neighbor_vector) / denominator))

        records.append(
            {
                "contig": frame.loc[idx, "contig"],
                "repeat_likeness_coverage_ratio": repeat_likeness,
                "coverage_neighbor_median": cov_median,
                "coverage_neighbor_abs_diff": (
                    abs(coverage[idx] - cov_median) if not np.isnan(cov_median) else np.nan
                ),
                "coverage_neighbor_log2_ratio": coverage_log2_ratio,
                "coverage_neighbor_cv": (
                    cov_std / cov_mean if cov_mean and not np.isnan(cov_mean) else np.nan
                ),
                "gc_neighbor_median": gc_median,
                "gc_neighbor_abs_diff": abs(gc[idx] - gc_median) if not np.isnan(gc_median) else np.nan,
                "kmer_neighbor_cosine_distance": kmer_distance,
            }
        )

    return pd.DataFrame.from_records(records)


def _neighbor_categorical_features(
    graph,
    frame: pd.DataFrame,
    value_column: str,
    prefix: str,
) -> pd.DataFrame:
    adjacency = [graph.neighbors(idx) for idx in range(graph.vcount())]
    values = frame[value_column].astype("object").to_numpy()
    records = []

    for idx, neighbors in enumerate(adjacency):
        current = values[idx]
        neighbor_values = pd.Series(values[neighbors], dtype="object") if neighbors else pd.Series(dtype="object")
        assigned = neighbor_values.dropna()
        same_fraction = np.nan
        different_fraction = np.nan
        if pd.notna(current) and len(neighbor_values):
            same_fraction = float((neighbor_values == current).sum() / len(neighbor_values))
            different_fraction = float((assigned != current).sum() / len(neighbor_values))

        records.append(
            {
                "contig": frame.loc[idx, "contig"],
                f"neighbor_{prefix}_entropy": _safe_entropy(neighbor_values),
                f"neighbor_same_{prefix}_fraction": same_fraction,
                f"neighbor_different_{prefix}_fraction": different_fraction,
                f"neighbor_unassigned_{prefix}_fraction": (
                    float(neighbor_values.isna().sum() / len(neighbor_values)) if len(neighbor_values) else np.nan
                ),
            }
        )

    return pd.DataFrame.from_records(records)


def _bin_output_features(graph, frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"contig": frame["contig"], "was_binned": frame["bin"].notna().astype(int)})
    if "bin" not in frame or frame["bin"].dropna().empty:
        return out

    binned = frame.loc[frame["bin"].notna()].copy()
    grouped = binned.groupby("bin", dropna=True)
    stats = grouped.agg(
        bin_size=("contig", "count"),
        bin_total_length=("length", "sum"),
        bin_coverage_mean=("coverage", "mean"),
        bin_coverage_variance=("coverage", "var"),
        bin_gc_variance=("gc_content", "var"),
    )
    stats["bin_n50"] = grouped["length"].apply(_n50)

    contig_to_idx = {contig: idx for idx, contig in enumerate(frame["contig"])}
    bin_graph_records = []
    for bin_id, subset in binned.groupby("bin"):
        vertices = [contig_to_idx[contig] for contig in subset["contig"]]
        subgraph = graph.induced_subgraph(vertices)
        if subgraph.vcount() == 0:
            component_count = 0
            largest_fraction = 0.0
            density = 0.0
        else:
            components = [len(component) for component in subgraph.components()]
            component_count = len(components)
            largest_fraction = max(components) / subgraph.vcount()
            density = (
                0.0
                if subgraph.vcount() <= 1
                else (2 * subgraph.ecount()) / (subgraph.vcount() * (subgraph.vcount() - 1))
            )
        bin_graph_records.append(
            {
                "bin": bin_id,
                "bin_graph_component_count": component_count,
                "bin_largest_graph_component_fraction": largest_fraction,
                "bin_graph_density": density,
            }
        )

    graph_stats = pd.DataFrame.from_records(bin_graph_records).set_index("bin")
    stats = stats.join(graph_stats)

    out = out.merge(frame[["contig", "bin"]], on="contig", how="left")
    out = out.merge(stats, left_on="bin", right_index=True, how="left").drop(columns=["bin"])
    numeric_cols = [column for column in out.columns if column != "contig"]
    out[numeric_cols] = out[numeric_cols].fillna(0)
    return out


def build_feature_table(
    graph_file: str | Path,
    contigs_file: str | Path,
    contig_paths_file: str | Path,
    ground_truth_file: str | Path | None = None,
    bin_assignments_file: str | Path | None = None,
    include_kmers: bool = True,
    k: int = DEFAULT_K,
) -> pd.DataFrame:
    """Build a contig-level feature table from SPAdes graph/data files."""

    contig_graph = load_spades_contig_graph(graph_file, contigs_file, contig_paths_file)
    graph = contig_graph.graph

    frame = _basic_contig_features(contig_graph, include_kmers=include_kmers, k=k)
    frame = frame.merge(_topology_features(contig_graph, frame), on="contig", how="left")

    if ground_truth_file is not None:
        truth = read_ground_truth(ground_truth_file)
        frame = merge_by_contig_alias(frame, truth, ["genome"])

    if bin_assignments_file is not None:
        bins = read_bin_assignments(bin_assignments_file)
        frame = merge_by_contig_alias(frame, bins, ["bin"])

    frame = frame.merge(
        _neighbor_numeric_features(graph, frame, include_kmers=include_kmers, k=k),
        on="contig",
        how="left",
    )

    if "bin" in frame:
        frame = frame.merge(_neighbor_categorical_features(graph, frame, "bin", "bin"), on="contig", how="left")
    if "genome" in frame:
        frame = frame.merge(
            _neighbor_categorical_features(graph, frame, "genome", "truth_genome"),
            on="contig",
            how="left",
        )

    if "bin" in frame:
        frame = frame.merge(_bin_output_features(graph, frame), on="contig", how="left")

    return frame
