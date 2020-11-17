from math import sqrt
from typing import Dict, List

import numpy as np
from sklearn import metrics


def auc(y_scores: np.ndarray, y_targets: np.ndarray) -> float:
    """Compute the AUC metric, given predicted and true labels.

    :param y_scores: The predicted label values or model scores.
    :param y_targets: The true label values.
    """
    fpr, tpr, thresholds = metrics.roc_curve(y_targets, y_scores, pos_label=1)
    return metrics.auc(fpr, tpr)


def rmse(y_scores: np.ndarray, y_targets: np.ndarray) -> float:
    """Compute the Root Mean Square Error metric, given predicted and true labels.

    :param y_scores: The predicted label values or model scores.
    :param y_targets: The true label values."""
    return sqrt(metrics.mean_squared_error(y_targets, y_scores))


# Supported metrics
METRICS_FUNCS = {
    "auc": auc,
    "rmse": rmse,
}


def evaluate(metric_list: List[str], y_scores: np.ndarray, y_targets: np.ndarray) -> Dict[str, float]:
    """Compute a set of metrics, given predicted and true labels.

    :param metric_list: A list of metrics to compute.
    :param y_scores: The predicted label values or model scores.
    :param y_targets: The true label values.
    :returns: A dictionary of all requested metrics.
    """
    unsupported_metrics = set(metric_list).difference(METRICS_FUNCS)

    if len(unsupported_metrics) > 0:
        raise ValueError(f"Evaluation failed: these metrics were requested but are not currently implemented: {unsupported_metrics}")

    return {metric_name: METRICS_FUNCS[metric_name](y_scores, y_targets) for metric_name in metric_list}
