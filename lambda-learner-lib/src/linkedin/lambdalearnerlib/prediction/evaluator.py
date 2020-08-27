from math import sqrt
from typing import Dict, List

import numpy as np
from sklearn import metrics


def auc(y_scores: np.ndarray, y_targets: np.ndarray) -> float:
    fpr, tpr, thresholds = metrics.roc_curve(y_targets, y_scores, pos_label=1)
    return metrics.auc(fpr, tpr)


def rmse(y_scores: np.ndarray, y_targets: np.ndarray) -> float:
    return sqrt(metrics.mean_squared_error(y_targets, y_scores))


METRICS_FUNCS = {
    "auc": auc,
    "rmse": rmse,
}


def evaluate(metric_list: List[str], y_scores: np.ndarray, y_targets: np.ndarray) -> Dict[str, float]:
    unsupported_metrics = set(metric_list).difference(METRICS_FUNCS)

    if len(unsupported_metrics) > 0:
        raise ValueError(
            "Evaluation failed because the following metrics were " f"requested but are not currently implemented: {unsupported_metrics}"
        )

    eval_metrics = {metric_name: METRICS_FUNCS[metric_name](y_scores, y_targets) for metric_name in metric_list}

    return eval_metrics
