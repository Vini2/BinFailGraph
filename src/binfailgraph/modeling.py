"""Model training, metrics, and feature-set utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    auc,
    f1_score,
    precision_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

IDENTIFIER_COLUMNS = {
    "contig",
    "contig_short",
    "spades_node_id",
    "genome",
    "bin",
    "bin_majority_genome",
}

LABEL_COLUMNS = {
    "is_known_truth",
    "is_binned",
    "correctly_binned",
    "mis_binned",
    "unbinned_should_bin",
    "in_contaminated_bin",
    "label_misbin",
    "label_failure",
    "label_success",
    "target",
    "bin_purity",
    "bin_truth_genome_count",
    "bin_has_mixed_truth",
    "bin_is_contaminated",
}

BASELINE_PREFIXES = ("kmer4_",)
BASELINE_COLUMNS = {
    "length",
    "coverage",
    "gc_content",
}

COMPARISON_FEATURE_SETS = [
    "length_only",
    "coverage_only",
    "composition_coverage",
    "graph_only",
    "composition_coverage_graph",
]

FEATURE_SET_LABELS = {
    "length_only": "Length only",
    "coverage_only": "Coverage only",
    "composition_coverage": "Composition + coverage",
    "graph_only": "Graph only",
    "composition_coverage_graph": "Composition + coverage + graph",
}

COMPOSITION_COLUMNS = {
    "gc_content",
}

GRAPH_STRUCTURE_COLUMNS = {
    "degree",
    "neighbor_count",
    "second_hop_neighbor_count",
    "local_clustering_coefficient",
    "betweenness_centrality",
    "closeness_centrality",
    "pagerank",
    "is_tip",
    "is_articulation_point",
    "is_incident_to_bridge",
    "lies_in_cycle",
    "shortest_path_to_branch_node",
}

GRAPH_AMBIGUITY_COLUMNS = {
    "repeat_likeness_coverage_ratio",
    "coverage_neighbor_median",
    "coverage_neighbor_abs_diff",
    "coverage_neighbor_log2_ratio",
    "coverage_neighbor_cv",
    "gc_neighbor_median",
    "gc_neighbor_abs_diff",
    "kmer_neighbor_cosine_distance",
}

GRAPH_BIN_CONTEXT_COLUMNS = {
    "neighbor_same_bin_fraction",
    "neighbor_different_bin_fraction",
    "neighbor_unassigned_bin_fraction",
    "neighbor_bin_entropy",
    "bin_graph_component_count",
    "bin_largest_graph_component_fraction",
    "bin_graph_density",
}

GRAPH_COLUMNS = GRAPH_STRUCTURE_COLUMNS | GRAPH_AMBIGUITY_COLUMNS

POST_BINNING_PREFIXES = ("neighbor_bin_", "bin_")
POST_BINNING_COLUMNS = {
    "was_binned",
    "neighbor_same_bin_fraction",
    "neighbor_different_bin_fraction",
    "neighbor_unassigned_bin_fraction",
    "neighbor_bin_entropy",
    "bin_size",
    "bin_total_length",
    "bin_n50",
    "bin_coverage_mean",
    "bin_coverage_variance",
    "bin_gc_variance",
    "bin_graph_component_count",
    "bin_largest_graph_component_fraction",
    "bin_graph_density",
}

ORACLE_COLUMNS = {
    "neighbor_truth_genome_entropy",
    "neighbor_same_truth_genome_fraction",
    "neighbor_different_truth_genome_fraction",
    "neighbor_unassigned_truth_genome_fraction",
}


@dataclass(frozen=True)
class EvaluationResult:
    model: Pipeline
    metrics: pd.Series
    feature_importance: pd.DataFrame
    test_predictions: pd.DataFrame


def select_feature_columns(
    frame: pd.DataFrame,
    feature_set: str = "graph",
    target_col: str = "target",
) -> list[str]:
    """Select numeric features for a modeling scenario.

    ``baseline`` uses non-graph contig features. ``graph`` adds topology and
    ambiguity features available before binning. ``post_binning`` adds binner
    output diagnostics. ``oracle`` also includes truth-taxonomy features and is
    only useful as an upper-bound sanity check.
    """

    numeric_columns = set(frame.select_dtypes(include=[np.number, "bool"]).columns)
    excluded = IDENTIFIER_COLUMNS | LABEL_COLUMNS | {target_col}
    candidate_columns = sorted(numeric_columns - excluded)

    def is_baseline(column: str) -> bool:
        return column in BASELINE_COLUMNS or column.startswith(BASELINE_PREFIXES)

    def is_composition(column: str) -> bool:
        return column in COMPOSITION_COLUMNS or column.startswith(BASELINE_PREFIXES)

    if feature_set == "length_only":
        allowed = [column for column in candidate_columns if column == "length"]
    elif feature_set == "coverage_only":
        allowed = [column for column in candidate_columns if column == "coverage"]
    elif feature_set == "composition_coverage":
        allowed = [
            column
            for column in candidate_columns
            if column == "coverage" or is_composition(column)
        ]
    elif feature_set == "graph_only":
        graph_only_columns = GRAPH_STRUCTURE_COLUMNS | GRAPH_BIN_CONTEXT_COLUMNS
        allowed = [column for column in candidate_columns if column in graph_only_columns]
    elif feature_set == "composition_coverage_graph":
        full_columns = (
            {"coverage"}
            | COMPOSITION_COLUMNS
            | GRAPH_STRUCTURE_COLUMNS
            | GRAPH_AMBIGUITY_COLUMNS
            | GRAPH_BIN_CONTEXT_COLUMNS
        )
        allowed = [
            column
            for column in candidate_columns
            if column in full_columns or column.startswith(BASELINE_PREFIXES)
        ]
    elif feature_set == "baseline":
        allowed = [column for column in candidate_columns if is_baseline(column)]
    elif feature_set == "graph":
        allowed = [
            column
            for column in candidate_columns
            if is_baseline(column) or column in GRAPH_COLUMNS
        ]
    elif feature_set == "post_binning":
        allowed = [
            column
            for column in candidate_columns
            if is_baseline(column)
            or column in GRAPH_COLUMNS
            or column in POST_BINNING_COLUMNS
            or column.startswith(POST_BINNING_PREFIXES)
        ]
    elif feature_set == "oracle":
        allowed = candidate_columns
    else:
        supported = [
            *COMPARISON_FEATURE_SETS,
            "baseline",
            "graph",
            "post_binning",
            "oracle",
        ]
        raise ValueError(f"feature_set must be one of: {', '.join(supported)}")

    return [
        column
        for column in allowed
        if (column not in ORACLE_COLUMNS or feature_set == "oracle") and frame[column].notna().any()
    ]


def make_logistic_regression(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_random_forest(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_xgboost(y_train: pd.Series | np.ndarray | None = None, random_state: int = 42) -> Pipeline:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost is not installed. Use environment.yml to create the paper environment.") from exc

    scale_pos_weight = 1.0
    if y_train is not None:
        y_array = np.asarray(y_train)
        positives = max(1, int(y_array.sum()))
        negatives = max(1, int((y_array == 0).sum()))
        scale_pos_weight = negatives / positives

    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            (
                "model",
                XGBClassifier(
                    n_estimators=350,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    scale_pos_weight=scale_pos_weight,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def expected_calibration_error(y_true, y_score, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (y_score >= lower) & (y_score < upper)
        if upper == 1.0:
            mask = (y_score >= lower) & (y_score <= upper)
        if not mask.any():
            continue
        ece += (mask.mean()) * abs(y_true[mask].mean() - y_score[mask].mean())
    return float(ece)


def precision_recall_at_top_k(y_true, y_score, top_k_fraction: float = 0.1) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    k = max(1, int(np.ceil(len(y_score) * top_k_fraction)))
    top_idx = np.argsort(y_score)[::-1][:k]
    positives_in_top = y_true[top_idx].sum()
    precision = positives_in_top / k
    recall = positives_in_top / max(1, y_true.sum())
    return float(precision), float(recall)


def _predict_failure_probability(model: Pipeline, x_test: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    decision = model.decision_function(x_test)
    return 1 / (1 + np.exp(-decision))


def _feature_importance(model: Pipeline, feature_columns: list[str]) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.ravel(estimator.coef_)
    else:
        values = np.zeros(len(feature_columns))

    out = pd.DataFrame({"feature": feature_columns, "importance": values})
    out["abs_importance"] = out["importance"].abs()
    return out.sort_values("abs_importance", ascending=False).reset_index(drop=True)


def evaluate_classifier(
    frame: pd.DataFrame,
    model: Pipeline,
    feature_columns: list[str],
    target_col: str = "target",
    test_size: float = 0.3,
    random_state: int = 42,
    top_k_fraction: float = 0.1,
) -> EvaluationResult:
    """Train/test evaluate a binary classifier for failure risk."""

    model_frame = frame.dropna(subset=[target_col]).copy()
    x = model_frame[feature_columns]
    y = model_frame[target_col].astype(int)

    stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        x,
        y,
        model_frame.index,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    model.fit(x_train, y_train)
    y_score = _predict_failure_probability(model, x_test)
    y_pred = (y_score >= 0.5).astype(int)

    metrics = {
        "n_train": len(y_train),
        "n_test": len(y_test),
        "positive_rate_test": float(y_test.mean()),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "ece_10bin": expected_calibration_error(y_test, y_score, n_bins=10),
    }
    if y_test.nunique() == 2:
        metrics["auroc"] = roc_auc_score(y_test, y_score)
        metrics["auprc"] = average_precision_score(y_test, y_score)
    else:
        metrics["auroc"] = np.nan
        metrics["auprc"] = np.nan

    precision_top, recall_top = precision_recall_at_top_k(y_test, y_score, top_k_fraction)
    metrics[f"precision_at_top_{int(top_k_fraction * 100)}pct"] = precision_top
    metrics[f"recall_at_top_{int(top_k_fraction * 100)}pct"] = recall_top

    predictions = model_frame.loc[idx_test, ["contig", "contig_short", target_col]].copy()
    predictions["risk_score"] = y_score
    predictions = predictions.sort_values("risk_score", ascending=False)

    return EvaluationResult(
        model=model,
        metrics=pd.Series(metrics),
        feature_importance=_feature_importance(model, feature_columns),
        test_predictions=predictions,
    )


def compare_feature_sets(
    frame: pd.DataFrame,
    model_factory,
    feature_sets: list[str] | None = None,
    target_col: str = "target",
    test_size: float = 0.3,
    random_state: int = 42,
    top_k_fraction: float = 0.1,
) -> tuple[pd.DataFrame, dict[str, EvaluationResult]]:
    """Evaluate the same model across named feature sets."""

    feature_sets = feature_sets or COMPARISON_FEATURE_SETS
    results: dict[str, EvaluationResult] = {}
    rows = []

    for feature_set in feature_sets:
        feature_columns = select_feature_columns(frame, feature_set=feature_set, target_col=target_col)
        result = evaluate_classifier(
            frame=frame,
            model=model_factory(),
            feature_columns=feature_columns,
            target_col=target_col,
            test_size=test_size,
            random_state=random_state,
            top_k_fraction=top_k_fraction,
        )
        results[feature_set] = result
        rows.append(
            {
                "feature_set": feature_set,
                "feature_set_label": FEATURE_SET_LABELS.get(feature_set, feature_set),
                "n_features": len(feature_columns),
                "auroc": result.metrics["auroc"],
                "auprc": result.metrics["auprc"],
            }
        )

    return pd.DataFrame(rows), results


def roc_curve_frame(
    result: EvaluationResult,
    target_col: str = "target",
) -> pd.DataFrame:
    """Return held-out ROC curve coordinates for an evaluation result."""

    y_true = result.test_predictions[target_col].astype(int)
    y_score = result.test_predictions["risk_score"]
    if y_true.nunique() < 2:
        raise ValueError("ROC curve requires both positive and negative examples in the test set.")

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})


def plot_roc_curve(
    result: EvaluationResult,
    model_name: str = "model",
    ax=None,
    target_col: str = "target",
):
    """Plot the held-out ROC curve for a trained evaluation result."""

    import matplotlib.pyplot as plt

    curve = roc_curve_frame(result, target_col=target_col)
    roc_auc = auc(curve["fpr"], curve["tpr"])

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    ax.step(
        curve["fpr"],
        curve["tpr"],
        where="post",
        linewidth=2,
        label=f"{model_name} (AUROC={roc_auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1, label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Held-out ROC curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    return ax


def plot_feature_set_roc_curves(
    results: dict[str, EvaluationResult],
    ax=None,
    target_col: str = "target",
):
    """Plot held-out ROC curves for several feature-set evaluations."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    for feature_set, result in results.items():
        curve = roc_curve_frame(result, target_col=target_col)
        roc_auc = auc(curve["fpr"], curve["tpr"])
        label = FEATURE_SET_LABELS.get(feature_set, feature_set)
        ax.step(
            curve["fpr"],
            curve["tpr"],
            where="post",
            linewidth=2,
            label=f"{label} (AUROC={roc_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1, label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Held-out ROC curves by feature set")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    return ax
