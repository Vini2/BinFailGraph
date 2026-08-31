# Graph-Based Failure Prediction for Metagenomic Binning

BinFailGraph is a diagnostic scaffold for asking:

> Can assembly-graph structure predict which contigs or bins are likely to fail during metagenomic binning?

The code is organized around contig-level failure prediction. It uses `agtools` to load a SPAdes contig-level graph, extracts non-graph baselines plus graph topology/ambiguity features, builds labels from ground truth and bin assignments, and evaluates simple ML models. The default notebooks use an initial binning result, `initial_contig_bins.csv`, and predict whether each binned contig was assigned correctly.

## Setup

```bash
conda env create -f environment.yml
conda activate binfailgraph
```

The environment installs this repository in editable mode and pulls `agtools` from pip.

## Example Data

The bundled datasets can be found under `tests/data/Sim-5G/` and `tests/data/Sim-10G/`. Each dataset contains:

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

Each notebook compares six feature sets with AUROC and AUPRC. These sets are constrained to a curated 9-feature universe to reduce repetition while preserving composition, coverage, graph ambiguity, neighbour disagreement, and bin-coherence signals:

- `length_only`: contig length only
- `coverage_only`: within-bin `coverage_difference` only
- `composition_only`: GC content and `4mer_composition_distance`
- `composition_coverage`: GC content, `4mer_composition_distance`, and `coverage_difference`
- `graph_only`: curated graph and graph-context features
- `composition_coverage_graph`: all curated 9 features

The ROC section writes standalone PNG files under `images/`: one ROC image per dataset and one combined ROC image that pools held-out predictions across all discovered datasets for each feature set.
The SHAP section in each model notebook also writes a standalone SHAP summary PNG under `images/`.

The feature-distribution section pools all discovered datasets and writes one standalone correct-vs-failed boxplot PNG per curated feature under `images/`. Raw 136-dimensional 4-mer vectors are collapsed into one `4mer_composition_distance` feature: the Euclidean distance from each contig's normalized canonical tetranucleotide-frequency vector to the centroid of contigs in the same initial bin. Raw coverage is represented for modeling as `coverage_difference`: the absolute difference between a contig's coverage and the mean coverage of contigs in the same initial bin. Significance asterisks are based on two-sided Mann-Whitney U tests with Benjamini-Hochberg FDR correction.

The notebooks compare five bin feature sets:

- `bin_nucleotide_only`: bin size/length, N50, GC summaries, and tetranucleotide-distance summaries
- `bin_coverage_only`: bin coverage summaries and within-bin coverage-difference summaries
- `bin_graph_only`: aggregate graph topology, graph-neighbour disagreement, and bin-induced graph coherence
- `bin_nucleotide_coverage`
- `bin_nucleotide_coverage_graph`

## Feature Reference

The notebooks use the curated 9 features below in the five AUROC/AUPRC comparison sets and in the feature-distribution plots. The assembly graph is treated as an undirected contig graph. A graph neighbour means a contig directly connected to the current contig in the SPAdes graph. The raw SPAdes coverage value is parsed from the contig name and is used to calculate derived features, but raw `coverage` itself is not used as a model feature.

| Feature name | What it means and how it is calculated |
| --- | --- |
| `length` | Contig length in bases, calculated from `contigs.fasta`. |
| `4mer_composition_distance` | Euclidean distance between a contig's normalized 136-dimensional canonical 4-mer frequency vector and the mean 4-mer frequency vector of its assigned initial bin. Reverse complements are collapsed before normalization. |
| `coverage_difference` | Absolute difference between the contig's raw SPAdes coverage and the mean coverage of contigs assigned to the same initial bin. |
| `degree` | Number of graph edges incident to the contig node. This summarizes local graph branching or tangling. |
| `pagerank` | PageRank centrality of the contig in the assembly graph. Higher values indicate connections to more influential or well-connected graph regions. |
| `coverage_neighbor_abs_diff` | Absolute difference between the contig's raw coverage and the median raw coverage of its directly connected graph neighbours. |
| `gc_content` | Fraction of bases in the contig that are G or C. |
| `neighbor_different_bin_fraction` | Fraction of directly connected graph neighbours assigned to a different initial bin. |
| `bin_largest_graph_component_fraction` | Fraction of contigs in the assigned bin that fall in the largest connected component of that bin's induced graph subgraph. |


## Python API Sketch

```python
from pathlib import Path

from binfailgraph.datasets import discover_datasets
from binfailgraph.features import build_bin_feature_table, build_feature_table
from binfailgraph.labels import bin_task_frame, make_contig_labels, task_frame

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
    bin_features = build_bin_feature_table(features, dataset_name=dataset.name)
    bin_task = bin_task_frame(bin_features, labelled, dataset_name=dataset.name)
```

`misbin_task["target"]` is the contig-level correctness target for the initial binning: `0` means incorrect, `1` means correct.
