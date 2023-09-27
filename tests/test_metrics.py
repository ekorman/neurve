import numpy as np
from neurve.metric_learning.metrics import retrieval_metrics


def test_retrieval_metrics_singleton():
    q = np.array([2])
    a = np.array([[1, 0, 2, 5, 2, 6, 7, 2]])
    metrics = retrieval_metrics(q, a, ranks=(1, 5, 10))
    assert metrics == {
        "mAP": (1 / 3 + 2 / 5 + 3 / 8) / 3,
        "mAP_1": 0,
        "mAP_5": (1 / 3 + 2 / 5) / 2,
        "mAP_10": (1 / 3 + 2 / 5 + 3 / 8) / 3,
        "rec_1": 0,
        "rec_5": 1,
        "rec_10": 1,
    }


def test_retrieval_metrics_multiple():
    q = np.array([1, 2])
    a = np.array([[1, 1, 0, 1], [5, 2, 6, 7]])
    metrics = retrieval_metrics(q, a, ranks=(1, 5, 10))
    assert metrics == {
        "mAP": ((1 / 1 + 2 / 2 + 3 / 4) / 3) / 2 + (1 / 2) / 1 / 2,
        "mAP_1": 1 / 2 + 0 / 2,
        "mAP_5": ((1 / 1 + 2 / 2 + 3 / 4) / 3) / 2 + (1 / 2) / 1 / 2,
        "mAP_10": ((1 / 1 + 2 / 2 + 3 / 4) / 3) / 2 + (1 / 2) / 1 / 2,
        "rec_1": 1 / 2,
        "rec_5": 1,
        "rec_10": 1,
    }
