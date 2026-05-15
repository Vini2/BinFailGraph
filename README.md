# Graph-Based Failure Prediction for Metagenomic Binning

BinFailGraph is a diagnostic scaffold for asking:

> Can assembly-graph structure predict which contigs or bins are likely to fail during metagenomic binning?

The code is organized around contig-level failure prediction. It uses agtools to load a SPAdes contig-level graph, extracts non-graph baselines plus graph topology/ambiguity features, builds labels from ground truth and bin assignments, and evaluates simple ML models. The default notebooks use an initial binning result, `initial_contig_bins.csv`, and predict whether each binned contig was assigned correctly.

## Setup

```bash
conda env create -f environment.yml
conda activate binfailgraph
```

The environment installs this repository in editable mode and pulls `agtools` from pip.

## Example Data

The bundled datasets live under `tests/data/Sim-5G/` and `tests/data/Sim-10G/`. Each dataset contains:

- `assembly_graph_with_scaffolds.gfa`: SPAdes assembly graph
- `contigs.paths`: SPAdes contig-to-unitig paths
- `contigs.fasta`: contig sequences
- `ground_truth.csv`: headerless `contig,genome` truth labels
- `initial_contig_bins.csv`: headerless `contig,bin` initial binning result

The bundled initial binning file is the post-binning diagnostic input. For new experiments, replace it with the output from a real binner. A bin assignment input can be either a two-column `contig,bin` table or a directory of one FASTA file per bin.

To add another dataset later, create a new subdirectory under `tests/data/` with the same required filenames. The notebooks discover dataset folders automatically.

## Notebooks

- `notebooks/01_logistic_regression.ipynb`
- `notebooks/02_random_forest.ipynb`
- `notebooks/03_xgboost.ipynb`

Each notebook runs the same feature/label pipeline on every discovered dataset and swaps only the model. The default task is `misbin`: among contigs present in `initial_contig_bins.csv`, predict `target=0` for an incorrect initial bin assignment and `target=1` for a correct assignment.

Each notebook now compares five feature sets with AUROC and AUPRC:

- `length_only`: contig length only
- `coverage_only`: within-bin `coverage_difference` only
- `composition_coverage`: GC content, `4mer_composition_distance`, and `coverage_difference`
- `graph_only`
- `composition_coverage_graph`: composition, coverage-difference, graph-topology, and graph/bin-context features

The ROC section shows one panel per dataset plus a combined ROC panel that pools held-out predictions across all discovered datasets for each feature set.

The feature-distribution section pools all discovered datasets and plots correct-vs-failed boxplots for every comparison feature. Raw 136-dimensional 4-mer vectors are collapsed into one `4mer_composition_distance` feature: the Euclidean distance from each contig's canonical tetranucleotide-frequency vector to the centroid of contigs in the same initial bin. Raw coverage is represented for modeling as `coverage_difference`: the absolute difference between a contig's coverage and the mean coverage of contigs in the same initial bin. Significance asterisks are based on two-sided Mann-Whitney U tests with Benjamini-Hochberg FDR correction.

## Feature Reference

The default notebooks use the features below in the five AUROC/AUPRC comparison sets and in the feature-distribution plots. The assembly graph is treated as an undirected contig graph. A graph neighbour means a contig directly connected to the current contig in the SPAdes graph. The raw SPAdes coverage value is parsed from the contig name and is used to calculate derived features, but raw `coverage` itself is not used as a model feature.

### Basic Contig, Coverage, and Composition Features

| Feature name | What it means and how it is calculated |
| --- | --- |
| `length` | Contig length in bases. This is calculated from the sequence in `contigs.fasta`; if a sequence is unavailable, the SPAdes length encoded in the contig name is used as a fallback. |
| `coverage_difference` | How different a contig's coverage is from the other contigs in its assigned initial bin. For each bin in `initial_contig_bins.csv`, the code calculates the mean raw SPAdes coverage of contigs in that bin, then stores `abs(contig coverage - bin mean coverage)`. Larger values mean the contig is coverage-inconsistent with its bin. |
| `gc_content` | Fraction of bases in the contig that are G or C. It is calculated as `(count(G) + count(C)) / contig length` from the contig sequence. |
| `4mer_composition_distance` | How different a contig's tetranucleotide composition is from the composition of its assigned initial bin. The code counts all valid 4-mers in each contig, combines reverse complements so there are 136 canonical 4-mer frequencies instead of 256 raw 4-mer frequencies, normalizes by the number of valid 4-mer windows, calculates the centroid of those 136-dimensional vectors within each initial bin, then stores the Euclidean distance from the contig vector to its bin centroid. The individual 4-mer frequencies are dropped and are not used as model features. |

### Local Graph Topology Features

| Feature name | What it means and how it is calculated |
| --- | --- |
| `degree` | Number of graph edges incident to the contig node. In plain language, this is how many direct graph connections the contig has. |
| `neighbor_count` | Number of unique directly connected contig neighbours. In these undirected SPAdes contig graphs this is usually the same as `degree`, but it is kept as the explicit neighbour count used by the local-neighbour calculations. |
| `second_hop_neighbor_count` | Number of unique contigs reachable in two graph steps, excluding the contig itself and excluding direct neighbours. Larger values indicate a more locally tangled graph neighbourhood. |
| `local_clustering_coefficient` | How connected the contig's neighbours are to one another. It is the fraction of possible neighbour-neighbour edges that actually exist; isolated and tip-like nodes get 0. |
| `betweenness_centrality` | A graph centrality score measuring how often the contig lies on shortest paths between other contigs. High values suggest the contig may act as a connector through the graph. |
| `closeness_centrality` | A graph centrality score based on the inverse of the average shortest-path distance from this contig to other reachable contigs. Higher values mean the contig is more central within its connected graph region. |
| `pagerank` | PageRank score on the contig graph. A contig receives a higher value when it is connected to other well-connected contigs. |
| `is_tip` | Binary feature: `1` if the contig has degree 0 or 1, otherwise `0`. Tips can be short dead-end graph structures. |
| `is_articulation_point` | Binary feature: `1` if removing the contig would split its graph component into more connected pieces, otherwise `0`. These nodes are graph bottlenecks. |
| `is_incident_to_bridge` | Binary feature: `1` if the contig touches at least one bridge edge, otherwise `0`. A bridge edge is an edge whose removal would disconnect part of the graph. |
| `lies_in_cycle` | Binary feature: `1` if the contig appears in at least one graph cycle found by the cycle-basis calculation, otherwise `0`. Cycles are one form of assembly-graph ambiguity. |
| `shortest_path_to_branch_node` | Shortest graph distance from the contig to any branch node, where a branch node is a contig with degree greater than 2. If the graph has no branch node, the value is missing. Smaller values mean the contig is closer to a branching region. |

### Neighbour Disagreement and Ambiguity Features

| Feature name | What it means and how it is calculated |
| --- | --- |
| `coverage_neighbor_median` | Median raw SPAdes coverage among directly connected graph neighbours. |
| `coverage_neighbor_abs_diff` | Absolute difference between the contig's raw coverage and `coverage_neighbor_median`. Larger values mean the contig's coverage disagrees with its immediate graph context. |
| `coverage_neighbor_log2_ratio` | `log2(contig coverage / neighbour median coverage)`, calculated only when both values are positive. Positive values mean the contig has higher coverage than its neighbours; negative values mean lower coverage. |
| `coverage_neighbor_cv` | Coverage coefficient of variation among directly connected neighbours, calculated as neighbour coverage standard deviation divided by neighbour coverage mean. Larger values mean the local graph neighbourhood has heterogeneous coverage. |
| `repeat_likeness_coverage_ratio` | Ratio of contig raw coverage to median neighbour coverage. Values much greater than 1 can indicate repeat-like sequence, because repeats often assemble with higher apparent coverage than nearby unique sequence. |
| `gc_neighbor_median` | Median GC content among directly connected graph neighbours. |
| `gc_neighbor_abs_diff` | Absolute difference between the contig's GC content and `gc_neighbor_median`. Larger values mean the contig has different base composition from its graph neighbours. |
| `kmer_neighbor_cosine_distance` | Composition disagreement between a contig and its graph neighbours. The code calculates the contig's 136-dimensional canonical 4-mer frequency vector, calculates the mean 4-mer vector of its direct neighbours, then stores `1 - cosine similarity` between the two vectors. Larger values mean more different tetranucleotide composition. |

### Initial-Bin Graph Context Features

These features use `initial_contig_bins.csv`. They are post-binning diagnostic features: they ask whether the initial bin assignment agrees with the assembly graph around the contig.

| Feature name | What it means and how it is calculated |
| --- | --- |
| `neighbor_bin_entropy` | Shannon entropy, in base 2, of the initial bin labels among directly connected neighbours. Missing neighbour bins are ignored for the entropy calculation. Higher entropy means neighbouring contigs are split across more bins. |
| `neighbor_same_bin_fraction` | Fraction of all directly connected neighbours assigned to the same initial bin as the contig. Missing neighbour bin assignments are included in the denominator. |
| `neighbor_different_bin_fraction` | Fraction of all directly connected neighbours that have an assigned bin different from the contig's initial bin. Missing neighbour bin assignments are included in the denominator but are not counted as different. |
| `neighbor_unassigned_bin_fraction` | Fraction of directly connected neighbours that have no initial bin assignment. |
| `bin_graph_component_count` | Number of connected components in the subgraph induced by all contigs assigned to the same initial bin. A good graph-coherent bin often has fewer disconnected pieces. |
| `bin_largest_graph_component_fraction` | Size of the largest connected component inside the bin-induced subgraph divided by the number of contigs in that bin. Values near 1 mean most contigs in the bin form one connected graph region. |
| `bin_graph_density` | Edge density of the bin-induced subgraph, calculated as `2 * edges / (nodes * (nodes - 1))`; bins with one or zero contigs get 0. Higher values mean contigs assigned to the same bin are more densely connected in the assembly graph. |

### Generated Columns Not Used in the Default Comparison Sets

Some extra columns are produced for labels, debugging, or optional API use, but they are not used in the five default notebook comparisons.

| Column name | Purpose |
| --- | --- |
| `coverage` | Raw SPAdes coverage parsed from the contig name. It is used internally to calculate `coverage_difference` and neighbour coverage features, but it is not selected as a model feature. |
| `length_from_name` | SPAdes length parsed from the contig name. It is kept as parsed metadata only; `length` is the model feature. |
| `kmer4_*` | Temporary raw canonical 4-mer frequency columns. They are used to calculate `4mer_composition_distance` and `kmer_neighbor_cosine_distance`, then dropped from the final feature table. |
| `was_binned`, `bin_size`, `bin_total_length`, `bin_n50`, `bin_coverage_mean`, `bin_coverage_variance`, `bin_gc_variance` | Optional bin-summary diagnostics available from the Python API, but not part of the five default feature-set comparisons. |
| `neighbor_truth_genome_entropy`, `neighbor_same_truth_genome_fraction`, `neighbor_different_truth_genome_fraction`, `neighbor_unassigned_truth_genome_fraction` | Ground-truth neighbour features used only for optional oracle-style analysis. They are excluded from default models because they would leak truth labels. |
| label columns such as `correctly_binned`, `mis_binned`, `label_misbin`, `label_success`, and `target` | Outcome labels used for evaluation, not input features. In the default `misbin` task, `target=0` means incorrect and `target=1` means correct. |

## Python API Sketch

```python
from pathlib import Path

from binfailgraph.datasets import discover_datasets
from binfailgraph.features import build_feature_table
from binfailgraph.labels import make_contig_labels, task_frame

for dataset in discover_datasets(Path("tests/data")):
    features = build_feature_table(
        graph_file=dataset.graph_file,
        contigs_file=dataset.contigs_file,
        contig_paths_file=dataset.contig_paths_file,
        ground_truth_file=dataset.ground_truth_file,
        bin_assignments_file=dataset.bin_assignments_file,
    )
    labelled = make_contig_labels(features)
    misbin_task = task_frame(labelled, task="misbin")
```

`misbin_task["target"]` is the contig-level correctness target for the initial binning: `0` means incorrect, `1` means correct.
