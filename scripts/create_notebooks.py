"""Generate the method notebooks used by the paper scaffold."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


COMMON_IMPORTS = r"""
from pathlib import Path
import importlib
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

for module_name in [
    "binfailgraph.datasets",
    "binfailgraph.features",
    "binfailgraph.labels",
    "binfailgraph.modeling",
]:
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])

from binfailgraph.datasets import discover_datasets
from binfailgraph.features import build_feature_table
from binfailgraph.labels import make_contig_labels, task_frame
from binfailgraph.modeling import (
    COMPARISON_FEATURE_SETS,
    combined_dataset_metric_table,
    combined_task_frame,
    comparison_feature_columns,
    compare_feature_sets,
    plot_combined_dataset_roc_curves,
    plot_feature_boxplots_by_outcome,
    plot_feature_set_roc_curves,
    select_feature_columns,
)

if sns is not None:
    sns.set_theme(style="whitegrid", context="notebook")
else:
    plt.style.use("ggplot")
DATA_ROOT = ROOT / "tests" / "data"
DATASETS = discover_datasets(DATA_ROOT)
INCLUDE_KMERS = True
TASK = "misbin"
"""


LOAD_DATA = r"""
dataset_features = {}
dataset_labelled = {}
dataset_tasks = {}
dataset_summary_rows = []

for dataset in DATASETS:
    raw_features = build_feature_table(
        graph_file=dataset.graph_file,
        contigs_file=dataset.contigs_file,
        contig_paths_file=dataset.contig_paths_file,
        ground_truth_file=dataset.ground_truth_file,
        bin_assignments_file=dataset.bin_assignments_file,
        include_kmers=INCLUDE_KMERS,
    )
    labelled = make_contig_labels(raw_features)
    task = task_frame(labelled, task=TASK)

    dataset_features[dataset.name] = raw_features
    dataset_labelled[dataset.name] = labelled
    dataset_tasks[dataset.name] = task
    dataset_summary_rows.append(
        {
            "dataset": dataset.name,
            "graph_contigs": len(raw_features),
            "initial_binned_contigs": int(raw_features["bin"].notna().sum()),
            "task_rows": len(task),
            "incorrect_assignments": int((task["target"] == 0).sum()),
            "correct_assignments": int(task["target"].sum()),
        }
    )

print("Target convention: 0 = incorrect initial bin assignment, 1 = correct initial bin assignment")
display(pd.DataFrame(dataset_summary_rows))
"""


PLOT_FEATURE_DISTRIBUTIONS = r"""
pooled_task = combined_task_frame(dataset_tasks)
distribution_feature_columns = comparison_feature_columns(
    pooled_task,
    feature_sets=COMPARISON_FEATURE_SETS,
)
print(
    f"Plotting {len(distribution_feature_columns)} comparison features pooled across "
    f"{len(dataset_tasks)} datasets."
)
_feature_distribution_figures = plot_feature_boxplots_by_outcome(
    pooled_task,
    distribution_feature_columns,
    features_per_figure=12,
    ncols=3,
    showfliers=False,
    p_adjust="fdr_bh",
)
plt.show()
"""


COMPARE_FEATURE_SETS = r"""
comparison_tables = []
comparison_results_by_dataset = {}

for dataset_name, current_task in dataset_tasks.items():
    table, results = compare_feature_sets(
        current_task,
        model_factory=lambda current_task=current_task: MODEL_FACTORY(current_task),
        feature_sets=COMPARISON_FEATURE_SETS,
        target_col="target",
        test_size=0.30,
        random_state=42,
        top_k_fraction=0.10,
    )
    table.insert(0, "dataset", dataset_name)
    comparison_tables.append(table)
    comparison_results_by_dataset[dataset_name] = results

comparison_table = pd.concat(comparison_tables, ignore_index=True)
display(
    comparison_table[
        ["dataset", "feature_set_label", "n_features", "auroc", "auprc"]
    ].style.format({"auroc": "{:.3f}", "auprc": "{:.3f}"})
)

primary_dataset = sorted(dataset_tasks)[0]
primary_task = dataset_tasks[primary_dataset]
result = comparison_results_by_dataset[primary_dataset]["composition_coverage_graph"]
feature_columns = select_feature_columns(primary_task, feature_set="composition_coverage_graph")
print(
    f"Using {primary_dataset!r} and {len(feature_columns):,} composition + coverage difference + graph features "
    "for diagnostics below."
)
display(result.metrics.to_frame("value"))
display(result.test_predictions.head(25))
"""


PLOT_IMPORTANCE = r"""
for dataset_name, results in comparison_results_by_dataset.items():
    top = results["composition_coverage_graph"].feature_importance.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    if sns is not None:
        sns.barplot(data=top, x="abs_importance", y="feature", ax=ax, color="#3b82f6")
    else:
        ax.barh(top["feature"], top["abs_importance"], color="#3b82f6")
    ax.set_xlabel("Absolute importance")
    ax.set_ylabel("")
    ax.set_title(f"{dataset_name}: top correct-binning prediction features")
    plt.tight_layout()
    plt.show()
"""


PLOT_COMPARISON_ROC = r"""
fig, axes = plt.subplots(1, len(comparison_results_by_dataset), figsize=(7 * len(comparison_results_by_dataset), 6))
if len(comparison_results_by_dataset) == 1:
    axes = [axes]

for ax, (dataset_name, results) in zip(axes, comparison_results_by_dataset.items()):
    plot_feature_set_roc_curves(results, ax=ax)
    ax.set_title(f"{dataset_name}: held-out ROC curves")

plt.tight_layout()
plt.show()
"""


PLOT_COMBINED_ROC = r"""
combined_table = combined_dataset_metric_table(
    comparison_results_by_dataset,
    feature_sets=COMPARISON_FEATURE_SETS,
)
display(
    combined_table[
        ["feature_set_label", "n_test", "correct_rate", "auroc", "auprc"]
    ].style.format({"correct_rate": "{:.3f}", "auroc": "{:.3f}", "auprc": "{:.3f}"})
)

fig, ax = plt.subplots(figsize=(7, 6))
plot_combined_dataset_roc_curves(
    comparison_results_by_dataset,
    feature_sets=COMPARISON_FEATURE_SETS,
    ax=ax,
)
ax.set_title("All datasets: combined held-out ROC curves")
plt.tight_layout()
plt.show()
"""


SHAP_LOGISTIC = r"""
try:
    import shap

    transformed = result.model.named_steps["impute"].transform(primary_task[feature_columns])
    transformed = result.model.named_steps["scale"].transform(transformed)
    masker = shap.maskers.Independent(transformed, max_samples=min(100, transformed.shape[0]))
    explainer = shap.LinearExplainer(result.model.named_steps["model"], masker=masker)
    shap_values = explainer(transformed)
    shap.summary_plot(shap_values.values, transformed, feature_names=feature_columns, max_display=20)
except ImportError:
    print("Install shap from environment.yml to run this cell.")
"""


SHAP_TREE = r"""
try:
    import shap

    transformed = result.model.named_steps["impute"].transform(primary_task[feature_columns])
    explainer = shap.TreeExplainer(result.model.named_steps["model"])
    shap_values = explainer.shap_values(transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif getattr(shap_values, "ndim", 0) == 3:
        shap_values = shap_values[:, :, 1]
    shap.summary_plot(shap_values, transformed, feature_names=feature_columns, max_display=20)
except ImportError:
    print("Install shap from environment.yml to run this cell.")
"""


def notebook(title: str, model_cell: str, extra_cells: list[tuple[str, str]] | None = None):
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(f"# {title}\n\nPost-binning contig correctness prediction on the bundled SPAdes example data."),
        nbf.v4.new_markdown_cell(
            "This notebook discovers dataset folders under `tests/data/` and reads each folder's "
            "`initial_contig_bins.csv` as the initial binner output. The target is `0` for an "
            "incorrect initial bin assignment and `1` for a correct assignment among binned contigs."
        ),
        nbf.v4.new_code_cell(COMMON_IMPORTS.strip()),
        nbf.v4.new_markdown_cell("## Data and Labels"),
        nbf.v4.new_code_cell(LOAD_DATA.strip()),
        nbf.v4.new_markdown_cell(
            "## Feature Distributions\n\n"
            "Boxplots compare correct and failed contigs after pooling all discovered datasets. "
            "The plotted features exclude raw 136-dimensional 4-mer vectors but include the "
            "collapsed `4mer_composition_distance` feature and the within-bin `coverage_difference` feature. "
            "Asterisks use two-sided Mann-Whitney U tests with Benjamini-Hochberg FDR correction."
        ),
        nbf.v4.new_code_cell(PLOT_FEATURE_DISTRIBUTIONS.strip()),
        nbf.v4.new_markdown_cell("## Model"),
        nbf.v4.new_code_cell(model_cell.strip()),
        nbf.v4.new_markdown_cell(
            "## Feature-Set Comparison\n\n"
            "The same model is evaluated with five feature sets: length only, coverage-difference only, "
            "composition + coverage difference, graph only, and composition + coverage difference + graph. "
            "Performance is reported with AUROC and AUPRC."
        ),
        nbf.v4.new_code_cell(COMPARE_FEATURE_SETS.strip()),
        nbf.v4.new_markdown_cell("## ROC Curves"),
        nbf.v4.new_code_cell(PLOT_COMPARISON_ROC.strip()),
        nbf.v4.new_markdown_cell(
            "## Combined ROC Curves Across Datasets\n\n"
            "These curves pool the held-out predictions from each dataset-specific run "
            "for the same feature set."
        ),
        nbf.v4.new_code_cell(PLOT_COMBINED_ROC.strip()),
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
def MODEL_FACTORY(current_task):
    return make_logistic_regression(random_state=42)
""",
            extra_cells=[
                (
                    "markdown",
                    "## SHAP Summary\n\n"
                    "Optional SHAP summary for the primary dataset using the composition + coverage difference + graph feature set.",
                ),
                ("code", SHAP_LOGISTIC),
            ],
        ),
    )

    write_notebook(
        "02_random_forest.ipynb",
        notebook(
            "Random Forest",
            """
from binfailgraph.modeling import make_random_forest

MODEL_NAME = "Random Forest"
def MODEL_FACTORY(current_task):
    return make_random_forest(random_state=42)
""",
            extra_cells=[
                (
                    "markdown",
                    "## SHAP Summary\n\n"
                    "Optional SHAP summary for the primary dataset using the composition + coverage difference + graph feature set.",
                ),
                ("code", SHAP_TREE),
            ],
        ),
    )

    write_notebook(
        "03_xgboost.ipynb",
        notebook(
            "XGBoost",
            """
from binfailgraph.modeling import make_xgboost

MODEL_NAME = "XGBoost"
def MODEL_FACTORY(current_task):
    return make_xgboost(y_train=current_task["target"], random_state=42)
""",
            extra_cells=[
                (
                    "markdown",
                    "## SHAP Summary\n\n"
                    "Optional SHAP summary for the primary dataset using the composition + coverage difference + graph feature set. "
                    "This can be slower than built-in feature importance.",
                ),
                (
                    "code",
                    SHAP_TREE,
                ),
            ],
        ),
    )


if __name__ == "__main__":
    main()
