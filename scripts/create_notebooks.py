"""Generate the method notebooks used by the paper scaffold."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


COMMON_IMPORTS = r"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
try:
    import seaborn as sns
except ImportError:
    sns = None

ROOT = Path.cwd()
for candidate in [ROOT, *ROOT.parents]:
    if (candidate / "src").exists() and (candidate / "tests" / "data").exists():
        ROOT = candidate
        break
sys.path.insert(0, str(ROOT / "src"))

from binfailgraph.features import build_feature_table
from binfailgraph.labels import make_contig_labels, summarize_labels, task_frame
from binfailgraph.modeling import (
    COMPARISON_FEATURE_SETS,
    compare_feature_sets,
    plot_feature_set_roc_curves,
    select_feature_columns,
)

if sns is not None:
    sns.set_theme(style="whitegrid", context="notebook")
else:
    plt.style.use("ggplot")
DATASET = "Sim-5G"
DATA = ROOT / "tests" / "data" / DATASET
INITIAL_BINS = DATA / "initial_contig_bins.csv"
INCLUDE_KMERS = True
TASK = "misbin"
"""


LOAD_DATA = r"""
raw_features = build_feature_table(
    graph_file=DATA / "assembly_graph_with_scaffolds.gfa",
    contigs_file=DATA / "contigs.fasta",
    contig_paths_file=DATA / "contigs.paths",
    ground_truth_file=DATA / "ground_truth.csv",
    bin_assignments_file=INITIAL_BINS,
    include_kmers=INCLUDE_KMERS,
)

labelled = make_contig_labels(raw_features)
task = task_frame(labelled, task=TASK)

print(f"Contigs in graph: {len(raw_features):,}")
print(f"Initial binned contigs: {raw_features['bin'].notna().sum():,}")
print(f"Rows in {TASK!r} task: {len(task):,}")
print("Target convention: 1 = incorrect initial bin assignment, 0 = correct initial bin assignment")
display(summarize_labels(labelled).to_frame("count"))
"""


COMPARE_FEATURE_SETS = r"""
comparison_table, comparison_results = compare_feature_sets(
    task,
    model_factory=MODEL_FACTORY,
    feature_sets=COMPARISON_FEATURE_SETS,
    target_col="target",
    test_size=0.30,
    random_state=42,
    top_k_fraction=0.10,
)

display(
    comparison_table[
        ["feature_set_label", "n_features", "auroc", "auprc"]
    ].style.format({"auroc": "{:.3f}", "auprc": "{:.3f}"})
)

result = comparison_results["composition_coverage_graph"]
feature_columns = select_feature_columns(task, feature_set="composition_coverage_graph")
print(f"Using {len(feature_columns):,} composition + coverage + graph features for diagnostics below.")
display(result.metrics.to_frame("value"))
display(result.test_predictions.head(25))
"""


PLOT_IMPORTANCE = r"""
top = result.feature_importance.head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 7))
if sns is not None:
    sns.barplot(data=top, x="abs_importance", y="feature", ax=ax, color="#3b82f6")
else:
    ax.barh(top["feature"], top["abs_importance"], color="#3b82f6")
ax.set_xlabel("Absolute importance")
ax.set_ylabel("")
ax.set_title("Top incorrect-binning risk features")
plt.tight_layout()
plt.show()
"""


PLOT_COMPARISON_ROC = r"""
fig, ax = plt.subplots(figsize=(7, 6))
plot_feature_set_roc_curves(comparison_results, ax=ax)
plt.tight_layout()
plt.show()
"""


def notebook(title: str, model_cell: str, extra_cells: list[tuple[str, str]] | None = None):
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(f"# {title}\n\nPost-binning contig correctness prediction on the bundled SPAdes example data."),
        nbf.v4.new_markdown_cell(
            "This notebook reads `tests/data/initial_contig_bins.csv` as the initial binner output. "
            "The target is `1` for an incorrect initial bin assignment and `0` for a correct assignment among binned contigs."
        ),
        nbf.v4.new_code_cell(COMMON_IMPORTS.strip()),
        nbf.v4.new_markdown_cell("## Data and Labels"),
        nbf.v4.new_code_cell(LOAD_DATA.strip()),
        nbf.v4.new_markdown_cell("## Model"),
        nbf.v4.new_code_cell(model_cell.strip()),
        nbf.v4.new_markdown_cell(
            "## Feature-Set Comparison\n\n"
            "The same model is evaluated with five feature sets: length only, coverage only, "
            "composition + coverage, graph only, and composition + coverage + graph. "
            "Performance is reported with AUROC and AUPRC."
        ),
        nbf.v4.new_code_cell(COMPARE_FEATURE_SETS.strip()),
        nbf.v4.new_markdown_cell("## ROC Curves"),
        nbf.v4.new_code_cell(PLOT_COMPARISON_ROC.strip()),
        nbf.v4.new_markdown_cell("## Full Feature-Set Importance"),
        nbf.v4.new_code_cell(PLOT_IMPORTANCE.strip()),
    ]
    for cell_type, source in extra_cells or []:
        factory = nbf.v4.new_markdown_cell if cell_type == "markdown" else nbf.v4.new_code_cell
        nb["cells"].append(factory(source.strip()))
    return nb


def write_notebook(filename: str, nb) -> None:
    NOTEBOOKS.mkdir(exist_ok=True)
    nbf.write(nb, NOTEBOOKS / filename)


def main() -> None:
    write_notebook(
        "01_logistic_regression.ipynb",
        notebook(
            "Logistic Regression",
            """
from binfailgraph.modeling import make_logistic_regression

MODEL_NAME = "Logistic Regression"
def MODEL_FACTORY():
    return make_logistic_regression(random_state=42)
""",
        ),
    )

    write_notebook(
        "02_random_forest.ipynb",
        notebook(
            "Random Forest",
            """
from binfailgraph.modeling import make_random_forest

MODEL_NAME = "Random Forest"
def MODEL_FACTORY():
    return make_random_forest(random_state=42)
""",
        ),
    )

    write_notebook(
        "03_xgboost.ipynb",
        notebook(
            "XGBoost",
            """
from binfailgraph.modeling import make_xgboost

MODEL_NAME = "XGBoost"
def MODEL_FACTORY():
    return make_xgboost(y_train=task["target"], random_state=42)
""",
            extra_cells=[
                (
                    "markdown",
                    "Optional SHAP summary for XGBoost. This can be slower than built-in feature importance.",
                ),
                (
                    "code",
                    """
try:
    import shap

    transformed = result.model.named_steps["impute"].transform(task[feature_columns])
    explainer = shap.TreeExplainer(result.model.named_steps["model"])
    shap_values = explainer.shap_values(transformed)
    shap.summary_plot(shap_values, transformed, feature_names=feature_columns, max_display=20)
except ImportError:
    print("Install shap from environment.yml to run this cell.")
""",
                ),
            ],
        ),
    )


if __name__ == "__main__":
    main()
