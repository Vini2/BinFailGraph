# BinFailGraph: Graph-Based Failure Prediction for Metagenomic Binning

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

## Notebooks

- `notebooks/01_logistic_regression.ipynb`
- `notebooks/02_random_forest.ipynb`
- `notebooks/03_xgboost.ipynb`

Each notebook runs the same feature/label pipeline and swaps only the model. The default task is `misbin`: among contigs present in `initial_contig_bins.csv`, predict `target=1` for an incorrect initial bin assignment and `target=0` for a correct assignment.

Each notebook now compares five feature sets with AUROC and AUPRC:

- `length_only`
- `coverage_only`
- `composition_coverage`
- `graph_only`
- `composition_coverage_graph`

## Python API Sketch

```python
from pathlib import Path

from binfailgraph.features import build_feature_table
from binfailgraph.labels import make_contig_labels, task_frame

data = Path("tests/data/Sim-5G")
features = build_feature_table(
    graph_file=data / "assembly_graph_with_scaffolds.gfa",
    contigs_file=data / "contigs.fasta",
    contig_paths_file=data / "contigs.paths",
    ground_truth_file=data / "ground_truth.csv",
    bin_assignments_file=data / "initial_contig_bins.csv",
)

labelled = make_contig_labels(features)
misbin_task = task_frame(labelled, task="misbin")
```

`misbin_task["target"]` is the contig-level correctness target for the initial binning: `1` means incorrect, `0` means correct.
