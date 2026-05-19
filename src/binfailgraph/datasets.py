"""Dataset discovery for BinFailGraph example and benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DATASET_FILE_CANDIDATES = {
    "graph_file": ("assembly_graph_with_scaffolds.gfa",),
    "contigs_file": ("contigs.fasta",),
    "contig_paths_file": ("contigs.paths",),
    "ground_truth_file": ("ground_truth.csv", "ground_truth_mapping.csv"),
    "bin_assignments_file": ("initial_contig_bins.csv",),
}

REQUIRED_DATASET_FILES = {
    field: filenames[0] for field, filenames in DATASET_FILE_CANDIDATES.items()
}


@dataclass(frozen=True)
class DatasetPaths:
    name: str
    root: Path
    graph_file: Path
    contigs_file: Path
    contig_paths_file: Path
    ground_truth_file: Path
    bin_assignments_file: Path


def _dataset_from_dir(path: Path) -> DatasetPaths | None:
    paths = {}
    for field, filenames in DATASET_FILE_CANDIDATES.items():
        for filename in filenames:
            candidate = path / filename
            if candidate.exists():
                paths[field] = candidate
                break
        else:
            return None

    return DatasetPaths(name=path.name, root=path, **paths)


def discover_datasets(data_root: str | Path) -> list[DatasetPaths]:
    """Discover dataset directories containing all required input files.

    If ``data_root`` itself contains the required files it is returned as one
    dataset. Otherwise, immediate child directories are scanned. Add a new
    dataset later by creating another subdirectory with the required filenames.
    """

    root = Path(data_root)
    direct = _dataset_from_dir(root)
    if direct is not None:
        return [direct]

    datasets = []
    for child in sorted(root.iterdir()):
        if child.is_dir():
            dataset = _dataset_from_dir(child)
            if dataset is not None:
                datasets.append(dataset)

    if not datasets:
        required = ", ".join(
            "/".join(filenames) for filenames in DATASET_FILE_CANDIDATES.values()
        )
        raise FileNotFoundError(f"No datasets found under {root}. Required files: {required}")

    return datasets
