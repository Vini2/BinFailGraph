"""Assembly graph loading helpers."""

from __future__ import annotations

import os
from pathlib import Path


def load_spades_contig_graph(
    graph_file: str | Path,
    contigs_file: str | Path,
    contig_paths_file: str | Path,
):
    """Load a SPAdes contig-level graph with agtools.

    agtools 1.1.0 exposes SPAdes support through
    ``agtools.assemblers.spades.get_contig_graph``. The returned object contains
    an ``igraph.Graph`` in ``.graph`` and contig names in ``.contig_names``.
    """

    cache_root = Path.cwd() / ".cache"
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    try:
        from agtools.assemblers.spades import get_contig_graph
    except ImportError as exc:
        raise ImportError(
            "agtools is required to load SPAdes contig graphs. "
            "Install it with the provided conda environment.yml."
        ) from exc

    return get_contig_graph(
        graph_file=str(graph_file),
        contigs_file=str(contigs_file),
        contig_paths_file=str(contig_paths_file),
    )


def graph_vertex_names(contig_graph) -> list[str]:
    """Return graph vertex labels as contig names."""

    if getattr(contig_graph, "contig_names", None) is not None:
        return list(contig_graph.contig_names)
    graph = contig_graph.graph
    if "label" in graph.vs.attributes():
        return list(graph.vs["label"])
    if "name" in graph.vs.attributes():
        return list(graph.vs["name"])
    return [str(idx) for idx in range(graph.vcount())]
