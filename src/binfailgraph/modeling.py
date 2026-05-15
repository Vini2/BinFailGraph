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

RAW_KMER_PREFIXES = ("kmer4_",)
INTERNAL_FEATURE_COLUMNS = {
    "coverage",
    "length_from_name",
}
BASELINE_PREFIXES: tuple[str, ...] = ()
BASELINE_COLUMNS = {
    "length",
    "coverage_difference",
    "gc_content",
    "4mer_composition_distance",
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
    "coverage_only": "Coverage difference only",
    "composition_coverage": "Composition + coverage difference",
    "graph_only": "Graph only",
    "composition_coverage_graph": "Composition + coverage difference + graph",
}

COMPOSITION_COLUMNS = {
    "gc_content",
    "4mer_composition_distance",
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


def combined_task_frame(
    frames_by_dataset: dict[str, pd.DataFrame],
    target_col: str = "target",
) -> pd.DataFrame:
    """Pool labelled task rows across datasets and add dataset/outcome columns."""

    frames = []
    for dataset_name, frame in sorted(frames_by_dataset.items()):
        current = frame.copy()
        current["dataset"] = dataset_name
        current["binning_outcome"] = np.where(
            current[target_col].astype(int) == 1,
            "Correct",
            "Failed",
        )
        frames.append(current)

    if not frames:
        raise ValueError("No dataset task frames were provided.")

    return pd.concat(frames, ignore_index=True)


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
    excluded = IDENTIFIER_COLUMNS | LABEL_COLUMNS | INTERNAL_FEATURE_COLUMNS | {target_col}
    candidate_columns = sorted(numeric_columns - excluded)

    def is_baseline(column: str) -> bool:
        return column in BASELINE_COLUMNS or column.startswith(BASELINE_PREFIXES)

    def is_composition(column: str) -> bool:
        return column in COMPOSITION_COLUMNS or column.startswith(BASELINE_PREFIXES)

    if feature_set == "length_only":
        allowed = [column for column in candidate_columns if column == "length"]
    elif feature_set == "coverage_only":
        allowed = [column for column in candidate_columns if column == "coverage_difference"]
    elif feature_set == "composition_coverage":
        allowed = [
            column
            for column in candidate_columns
            if column == "coverage_difference" or is_composition(column)
        ]
    elif feature_set == "graph_only":
        graph_only_columns = GRAPH_STRUCTURE_COLUMNS | GRAPH_BIN_CONTEXT_COLUMNS
        allowed = [column for column in candidate_columns if column in graph_only_columns]
    elif feature_set == "composition_coverage_graph":
        full_columns = (
            {"coverage_difference"}
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


def comparison_feature_columns(
    frame: pd.DataFrame,
    feature_sets: list[str] | None = None,
    target_col: str = "target",
    exclude_prefixes: tuple[str, ...] = RAW_KMER_PREFIXES,
) -> list[str]:
    """Return comparison features after dropping raw high-dimensional k-mer columns."""

    feature_sets = feature_sets or COMPARISON_FEATURE_SETS
    columns = []
    seen = set()
    for feature_set in feature_sets:
        for column in select_feature_columns(frame, feature_set=feature_set, target_col=target_col):
            if column in seen or any(column.startswith(prefix) for prefix in exclude_prefixes):
                continue
            seen.add(column)
            columns.append(column)
    return columns


non_kmer_comparison_feature_columns = comparison_feature_columns


def significance_stars(p_value: float) -> str:
    """Map a p-value to the usual significance-star label."""

    if not np.isfinite(p_value):
        return "n/a"
    if p_value <= 1e-4:
        return "****"
    if p_value <= 1e-3:
        return "***"
    if p_value <= 1e-2:
        return "**"
    if p_value <= 5e-2:
        return "*"
    return "ns"


def _benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR adjustment for a p-value series."""

    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return adjusted

    order = np.argsort(valid.to_numpy())
    ordered = valid.to_numpy()[order]
    n_tests = len(ordered)
    ranks = np.arange(1, n_tests + 1)
    adjusted_ordered = ordered * n_tests / ranks
    adjusted_ordered = np.minimum.accumulate(adjusted_ordered[::-1])[::-1]
    adjusted_ordered = np.clip(adjusted_ordered, 0.0, 1.0)
    adjusted.loc[valid.index[order]] = adjusted_ordered
    return adjusted


def feature_outcome_significance_table(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_col: str = "target",
    p_adjust: str | None = "fdr_bh",
) -> pd.DataFrame:
    """Test correct-vs-failed feature shifts with two-sided Mann-Whitney U tests."""

    try:
        from scipy.stats import mannwhitneyu
    except ImportError as exc:
        raise ImportError("scipy is required to compute feature-distribution significance.") from exc

    rows = []
    target = frame[target_col].astype(int)
    for feature in feature_columns:
        if feature not in frame.columns:
            continue

        values = pd.to_numeric(frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        correct = values[target == 1].dropna()
        failed = values[target == 0].dropna()

        p_value = np.nan
        statistic = np.nan
        if not correct.empty and not failed.empty:
            try:
                test = mannwhitneyu(correct, failed, alternative="two-sided")
            except TypeError:
                test = mannwhitneyu(correct, failed)
            statistic = float(test.statistic)
            p_value = float(test.pvalue)

        rows.append(
            {
                "feature": feature,
                "n_correct": len(correct),
                "n_failed": len(failed),
                "median_correct": float(correct.median()) if not correct.empty else np.nan,
                "median_failed": float(failed.median()) if not failed.empty else np.nan,
                "mannwhitney_u": statistic,
                "p_value": p_value,
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    if p_adjust == "fdr_bh":
        table["p_value_adj"] = _benjamini_hochberg(table["p_value"])
        star_values = table["p_value_adj"]
    elif p_adjust in {None, "none"}:
        table["p_value_adj"] = np.nan
        star_values = table["p_value"]
    else:
        raise ValueError("p_adjust must be 'fdr_bh', 'none', or None.")

    table["significance"] = [significance_stars(p_value) for p_value in star_values]
    return table


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


def _predict_correctness_probability(model: Pipeline, x_test: pd.DataFrame) -> np.ndarray:
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
    """Train/test evaluate a binary classifier for correct initial assignments."""

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
    y_score = _predict_correctness_probability(model, x_test)
    y_pred = (y_score >= 0.5).astype(int)

    metrics = {
        "n_train": len(y_train),
        "n_test": len(y_test),
        "correct_rate_test": float(y_test.mean()),
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
    predictions["correctness_score"] = y_score
    predictions = predictions.sort_values("correctness_score", ascending=False)

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
    y_score = result.test_predictions["correctness_score"]
    if y_true.nunique() < 2:
        raise ValueError("ROC curve requires both positive and negative examples in the test set.")

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})


def combined_prediction_frame(
    results_by_dataset: dict[str, dict[str, EvaluationResult]],
    feature_set: str,
    target_col: str = "target",
) -> pd.DataFrame:
    """Concatenate held-out predictions for one feature set across datasets."""

    frames = []
    for dataset_name, results in sorted(results_by_dataset.items()):
        if feature_set not in results:
            continue

        predictions = results[feature_set].test_predictions[[target_col, "correctness_score"]].copy()
        predictions["dataset"] = dataset_name
        predictions["feature_set"] = feature_set
        predictions["feature_set_label"] = FEATURE_SET_LABELS.get(feature_set, feature_set)
        frames.append(predictions)

    if not frames:
        raise ValueError(f"No held-out predictions found for feature_set={feature_set!r}.")

    return pd.concat(frames, ignore_index=True)


def combined_roc_curve_frame(
    results_by_dataset: dict[str, dict[str, EvaluationResult]],
    feature_set: str,
    target_col: str = "target",
) -> pd.DataFrame:
    """Return ROC coordinates after pooling held-out predictions across datasets."""

    combined = combined_prediction_frame(
        results_by_dataset=results_by_dataset,
        feature_set=feature_set,
        target_col=target_col,
    )
    y_true = combined[target_col].astype(int)
    y_score = combined["correctness_score"]
    if y_true.nunique() < 2:
        raise ValueError("Combined ROC curve requires both positive and negative examples.")

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})


def combined_dataset_metric_table(
    results_by_dataset: dict[str, dict[str, EvaluationResult]],
    feature_sets: list[str] | None = None,
    target_col: str = "target",
) -> pd.DataFrame:
    """Compute AUROC and AUPRC after pooling held-out predictions across datasets."""

    feature_sets = feature_sets or COMPARISON_FEATURE_SETS
    rows = []
    for feature_set in feature_sets:
        combined = combined_prediction_frame(
            results_by_dataset=results_by_dataset,
            feature_set=feature_set,
            target_col=target_col,
        )
        y_true = combined[target_col].astype(int)
        y_score = combined["correctness_score"]

        if y_true.nunique() == 2:
            auroc = roc_auc_score(y_true, y_score)
            auprc = average_precision_score(y_true, y_score)
        else:
            auroc = np.nan
            auprc = np.nan

        rows.append(
            {
                "feature_set": feature_set,
                "feature_set_label": FEATURE_SET_LABELS.get(feature_set, feature_set),
                "n_test": len(combined),
                "correct_rate": float(y_true.mean()),
                "auroc": auroc,
                "auprc": auprc,
            }
        )

    return pd.DataFrame(rows)


def transfer_feature_set_table(
    frames_by_dataset: dict[str, pd.DataFrame],
    model_factory,
    feature_set: str = "composition_coverage_graph",
    target_col: str = "target",
    test_size: float = 0.3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Evaluate dataset-to-dataset transfer for one feature set.

    Diagonal cells use a held-out split within the dataset. Off-diagonal cells
    fit on all labelled rows from the source dataset and evaluate on all
    labelled rows from the target dataset.
    """

    rows = []
    for train_dataset, train_frame in sorted(frames_by_dataset.items()):
        train_task = train_frame.dropna(subset=[target_col]).copy()
        if train_task.empty:
            continue

        for test_dataset, test_frame in sorted(frames_by_dataset.items()):
            test_task = test_frame.dropna(subset=[target_col]).copy()
            if test_task.empty:
                continue

            if train_dataset == test_dataset:
                y = train_task[target_col].astype(int)
                stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
                train_rows, test_rows = train_test_split(
                    train_task.index,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=stratify,
                )
                fit_frame = train_task.loc[train_rows]
                eval_frame = train_task.loc[test_rows]
                evaluation_mode = "within_dataset_holdout"
            else:
                fit_frame = train_task
                eval_frame = test_task
                evaluation_mode = "cross_dataset_transfer"

            feature_columns = select_feature_columns(
                fit_frame,
                feature_set=feature_set,
                target_col=target_col,
            )
            if not feature_columns:
                raise ValueError(
                    f"No usable features for {feature_set!r} when training on {train_dataset!r}."
                )

            missing_columns = [
                column for column in feature_columns if column not in eval_frame.columns
            ]
            if missing_columns:
                raise ValueError(
                    f"{test_dataset!r} is missing transfer-test feature columns: {missing_columns}"
                )

            model = model_factory(fit_frame)
            x_train = fit_frame[feature_columns]
            y_train = fit_frame[target_col].astype(int)
            x_test = eval_frame[feature_columns]
            y_test = eval_frame[target_col].astype(int)

            model.fit(x_train, y_train)
            y_score = _predict_correctness_probability(model, x_test)
            y_pred = (y_score >= 0.5).astype(int)
            y_failure = 1 - y_test
            failure_score = 1 - y_score

            if y_test.nunique() == 2:
                auroc = roc_auc_score(y_test, y_score)
                correct_auprc = average_precision_score(y_test, y_score)
                failure_auprc = average_precision_score(y_failure, failure_score)
            else:
                auroc = np.nan
                correct_auprc = np.nan
                failure_auprc = np.nan

            rows.append(
                {
                    "train_dataset": train_dataset,
                    "test_dataset": test_dataset,
                    "evaluation_mode": evaluation_mode,
                    "feature_set": feature_set,
                    "feature_set_label": FEATURE_SET_LABELS.get(feature_set, feature_set),
                    "n_features": len(feature_columns),
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "correct_rate_test": float(y_test.mean()),
                    "failure_rate_test": float(y_failure.mean()),
                    "auroc": auroc,
                    "correct_auprc": correct_auprc,
                    "failure_auprc": failure_auprc,
                    "f1": f1_score(y_test, y_pred, zero_division=0),
                    "precision": precision_score(y_test, y_pred, zero_division=0),
                }
            )

    return pd.DataFrame(rows)


def transfer_metric_matrix(
    transfer_table: pd.DataFrame,
    metric: str = "auroc",
) -> pd.DataFrame:
    """Return a train-dataset by test-dataset matrix from a transfer table."""

    if metric not in transfer_table.columns:
        raise ValueError(f"{metric!r} is not a column in the transfer table.")

    matrix = transfer_table.pivot(
        index="train_dataset",
        columns="test_dataset",
        values=metric,
    )
    return matrix.sort_index(axis=0).sort_index(axis=1)


def plot_transfer_heatmap(
    transfer_table: pd.DataFrame,
    metric: str = "auroc",
    ax=None,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = 1.0,
    cmap: str = "viridis",
    fmt: str = ".3f",
    annotation_fontsize: int = 13,
    title_fontsize: int = 15,
    label_fontsize: int = 13,
    tick_fontsize: int = 12,
):
    """Plot a dataset transfer heatmap for one transfer metric."""

    import matplotlib.pyplot as plt

    matrix = transfer_metric_matrix(transfer_table, metric=metric)
    values = matrix.to_numpy(dtype=float)
    masked_values = np.ma.masked_invalid(values)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    if vmin is None:
        vmin = 0.5 if metric == "auroc" else 0.0

    image = ax.imshow(masked_values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.grid(False)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticklabels(matrix.index)
    ax.tick_params(which="both", length=0)
    ax.tick_params(which="major", labelsize=tick_fontsize)
    ax.set_xlabel("Test dataset", fontsize=label_fontsize)
    ax.set_ylabel("Train dataset", fontsize=label_fontsize)
    ax.set_title(title or f"Dataset transfer: {metric}", fontsize=title_fontsize)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = values[row_idx, col_idx]
            if not np.isfinite(value):
                label = "n/a"
            else:
                label = format(value, fmt)
            ax.text(
                col_idx,
                row_idx,
                label,
                ha="center",
                va="center",
                color="black",
                fontsize=annotation_fontsize,
                fontweight="bold",
            )

    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_feature_boxplots_by_outcome(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_col: str = "target",
    features_per_figure: int = 12,
    ncols: int = 3,
    showfliers: bool = False,
    p_adjust: str | None = "fdr_bh",
):
    """Plot paged boxplots comparing correct and failed contigs for each feature."""

    import matplotlib.pyplot as plt

    if not feature_columns:
        raise ValueError("No feature columns were provided for plotting.")

    plot_frame = frame.dropna(subset=[target_col]).copy()
    if "binning_outcome" not in plot_frame.columns:
        plot_frame["binning_outcome"] = np.where(
            plot_frame[target_col].astype(int) == 1,
            "Correct",
            "Failed",
        )

    available = [
        column
        for column in feature_columns
        if column in plot_frame.columns and plot_frame[column].notna().any()
    ]
    if not available:
        raise ValueError("None of the requested feature columns are available for plotting.")

    features_per_figure = max(1, features_per_figure)
    ncols = max(1, ncols)
    figures = []
    colors = ["#60a5fa", "#f87171"]
    significance = feature_outcome_significance_table(
        plot_frame,
        available,
        target_col=target_col,
        p_adjust=p_adjust,
    ).set_index("feature")

    for page_start in range(0, len(available), features_per_figure):
        chunk = available[page_start : page_start + features_per_figure]
        nrows = int(np.ceil(len(chunk) / ncols))
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(4.2 * ncols, 3.2 * nrows),
            squeeze=False,
        )

        for ax, feature in zip(axes.ravel(), chunk):
            values = pd.to_numeric(plot_frame[feature], errors="coerce").replace(
                [np.inf, -np.inf],
                np.nan,
            )
            correct = values[plot_frame[target_col].astype(int) == 1].dropna()
            failed = values[plot_frame[target_col].astype(int) == 0].dropna()

            if correct.empty or failed.empty:
                ax.text(0.5, 0.5, "insufficient data", ha="center", va="center")
                ax.set_axis_off()
                continue

            box = ax.boxplot(
                [correct, failed],
                patch_artist=True,
                showfliers=showfliers,
                widths=0.6,
            )
            for patch, color in zip(box["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.75)
            for median in box["medians"]:
                median.set_color("#111827")
                median.set_linewidth(1.4)

            row = significance.loc[feature]
            p_value_column = "p_value_adj" if p_adjust == "fdr_bh" else "p_value"
            p_value = row[p_value_column]
            p_label = "q" if p_adjust == "fdr_bh" else "p"
            if np.isfinite(p_value):
                p_text = f"{p_label}={p_value:.2g}"
            else:
                p_text = f"{p_label}=n/a"

            y_min, y_max = ax.get_ylim()
            y_span = y_max - y_min
            if not np.isfinite(y_span) or y_span <= 0:
                y_span = max(abs(y_max), 1.0)
            bracket_y = y_max + 0.06 * y_span
            bracket_h = 0.04 * y_span
            text_y = bracket_y + bracket_h + 0.015 * y_span
            ax.plot(
                [1, 1, 2, 2],
                [bracket_y, bracket_y + bracket_h, bracket_y + bracket_h, bracket_y],
                color="#111827",
                linewidth=1.0,
            )
            ax.text(
                1.5,
                text_y,
                f"{row['significance']}\n{p_text}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                linespacing=0.9,
            )
            ax.set_ylim(y_min, text_y + 0.12 * y_span)
            ax.set_title(feature, fontsize=10)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(
                [f"Correct\nn={len(correct)}", f"Failed\nn={len(failed)}"],
                fontsize=9,
            )
            ax.set_ylabel("value")
            ax.grid(axis="y", alpha=0.25)

        for ax in axes.ravel()[len(chunk) :]:
            ax.set_axis_off()

        page_number = page_start // features_per_figure + 1
        page_count = int(np.ceil(len(available) / features_per_figure))
        fig.suptitle(
            f"Correct vs failed contigs: comparison features ({page_number}/{page_count})",
            fontsize=13,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        figures.append(fig)

    return figures


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


def plot_combined_dataset_roc_curves(
    results_by_dataset: dict[str, dict[str, EvaluationResult]],
    feature_sets: list[str] | None = None,
    ax=None,
    target_col: str = "target",
):
    """Plot empirical ROC curves after pooling held-out predictions across datasets."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    feature_sets = feature_sets or COMPARISON_FEATURE_SETS
    for feature_set in feature_sets:
        curve = combined_roc_curve_frame(
            results_by_dataset=results_by_dataset,
            feature_set=feature_set,
            target_col=target_col,
        )
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
    ax.set_title("Combined held-out ROC curves by feature set")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    return ax
