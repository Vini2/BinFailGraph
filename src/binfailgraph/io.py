"""Input helpers for contigs, truth labels, and bin assignments."""

from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

SPADES_CONTIG_RE = re.compile(
    r"^(?P<short>NODE_(?P<node_id>\d+))_length_(?P<length>\d+)_cov_(?P<coverage>[0-9.eE+-]+)"
)

FASTA_SUFFIXES = {
    ".fa",
    ".fasta",
    ".fna",
    ".fas",
    ".fa.gz",
    ".fasta.gz",
    ".fna.gz",
    ".fas.gz",
}


def clean_contig_name(name: object) -> str:
    """Return a normalized contig identifier without FASTA adornments."""

    text = str(name).strip()
    if text.startswith(">"):
        text = text[1:]
    text = text.split()[0]
    return text.rstrip("'")


def contig_short_name(name: object) -> str:
    """Return the SPAdes short identifier, e.g. ``NODE_12``."""

    text = clean_contig_name(name)
    match = SPADES_CONTIG_RE.match(text)
    if match:
        return match.group("short")
    short_match = re.match(r"^(NODE_\d+)", text)
    if short_match:
        return short_match.group(1)
    return text


def parse_spades_contig_name(name: object) -> dict[str, float | int | str | None]:
    """Parse length and coverage encoded in a SPAdes contig name."""

    text = clean_contig_name(name)
    match = SPADES_CONTIG_RE.match(text)
    if not match:
        return {
            "contig": text,
            "contig_short": contig_short_name(text),
            "spades_node_id": None,
            "length_from_name": None,
            "coverage": None,
        }

    return {
        "contig": text,
        "contig_short": match.group("short"),
        "spades_node_id": int(match.group("node_id")),
        "length_from_name": int(match.group("length")),
        "coverage": float(match.group("coverage")),
    }


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_csv(path, sep=None, engine="python", comment="#")


def _read_two_column_table(
    path: str | Path,
    preferred_second_names: Iterable[str],
    fallback_second_name: str,
) -> pd.DataFrame:
    """Read a headered or headerless two-column contig metadata table."""

    path = Path(path)
    with path.open() as handle:
        first_line = handle.readline().strip()

    first_parts = re.split(r"[\t,]", first_line)
    has_header = (
        len(first_parts) >= 2
        and "contig" in first_parts[0].strip().lower()
        and first_parts[1].strip().lower() in set(preferred_second_names)
    )

    if has_header:
        df = _read_table(path)
        lower_to_original = {col.lower(): col for col in df.columns}
        contig_col = lower_to_original.get("contig") or lower_to_original.get("contig_id")
        second_col = next(
            (lower_to_original[name] for name in preferred_second_names if name in lower_to_original),
            None,
        )
        if contig_col is None or second_col is None:
            raise ValueError(f"Could not identify contig and value columns in {path}")
        df = df[[contig_col, second_col]].copy()
        df.columns = ["contig", fallback_second_name]
    else:
        df = pd.read_csv(path, sep=None, engine="python", header=None, comment="#")
        if df.shape[1] < 2:
            raise ValueError(f"Expected at least two columns in {path}")
        df = df.iloc[:, :2].copy()
        df.columns = ["contig", fallback_second_name]

    df["contig"] = df["contig"].map(clean_contig_name)
    df["contig_short"] = df["contig"].map(contig_short_name)
    return df.drop_duplicates(subset=["contig", fallback_second_name])


def read_ground_truth(path: str | Path) -> pd.DataFrame:
    """Read a contig-to-genome/species truth table."""

    df = _read_two_column_table(
        path,
        preferred_second_names=("genome", "species", "taxonomy", "truth", "label"),
        fallback_second_name="genome",
    )
    return df.rename(columns={df.columns[1]: "genome"})[["contig", "contig_short", "genome"]]


def _is_fasta_path(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in FASTA_SUFFIXES)


def _open_text(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt")
    return path.open()


def iter_fasta_headers(path: str | Path) -> Iterable[str]:
    """Yield FASTA record identifiers from a plain or gzipped FASTA file."""

    with _open_text(Path(path)) as handle:
        for line in handle:
            if line.startswith(">"):
                yield clean_contig_name(line)


def read_bin_assignments(path: str | Path) -> pd.DataFrame:
    """Read bin assignments from a table or a directory of bin FASTA files.

    Table inputs can be headerless ``contig,bin`` files or headered tables with
    columns named like ``contig`` and ``bin``/``cluster``/``bin_id``. Directory
    inputs are interpreted as one FASTA file per bin, with the file stem used as
    the bin identifier.
    """

    path = Path(path)
    if path.is_dir():
        records: list[dict[str, str]] = []
        for fasta_path in sorted(path.iterdir()):
            if not fasta_path.is_file() or not _is_fasta_path(fasta_path):
                continue
            bin_id = fasta_path.name
            for suffix in FASTA_SUFFIXES:
                if bin_id.lower().endswith(suffix):
                    bin_id = bin_id[: -len(suffix)]
                    break
            for contig in iter_fasta_headers(fasta_path):
                records.append(
                    {
                        "contig": contig,
                        "contig_short": contig_short_name(contig),
                        "bin": bin_id,
                    }
                )
        return pd.DataFrame.from_records(records, columns=["contig", "contig_short", "bin"])

    df = _read_two_column_table(
        path,
        preferred_second_names=("bin", "bin_id", "cluster", "cluster_id", "assignment"),
        fallback_second_name="bin",
    )
    return df.rename(columns={df.columns[1]: "bin"})[["contig", "contig_short", "bin"]]


def merge_by_contig_alias(
    left: pd.DataFrame,
    right: pd.DataFrame,
    value_columns: list[str],
) -> pd.DataFrame:
    """Merge metadata by full contig ID first, then by SPAdes short ID."""

    out = left.copy()
    if right is None or right.empty:
        for column in value_columns:
            out[column] = pd.NA
        return out

    full = right[["contig", *value_columns]].drop_duplicates("contig")
    out = out.merge(full, on="contig", how="left")

    missing = out[value_columns[0]].isna()
    if missing.any():
        short = right[["contig_short", *value_columns]].drop_duplicates("contig_short")
        fill = left.loc[missing, ["contig_short"]].merge(short, on="contig_short", how="left")
        for column in value_columns:
            out.loc[missing, column] = fill[column].to_numpy()

    return out
