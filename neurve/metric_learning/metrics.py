import numpy as np


def map_score(q, a):
    """Mean average precision for a retrieval task

    Parameters
    ----------
    q : np.ndarray
        array of shape [n]
    a : np.ndarray
        array of shape [n, k] of sorted labels to match against

    Returns
    -------
    float
    """
    indicator = (a == q.reshape(-1, 1)).astype(int)
    tot_hits = indicator.sum(1)
    tot_hits[tot_hits == 0] = 1

    aps = (
        indicator.cumsum(1)
        * indicator
        / np.arange(1, a.shape[1] + 1).reshape(1, -1)
    ).sum(1) / tot_hits
    return aps.mean()


def recall(q, a):
    """Recall average precision for a retrieval task

    Parameters
    ----------
    q : np.ndarray
        array of shape [n]
    a : np.ndarray
        array of shape [n, k] of sorted labels to match against

    Returns
    -------
    float
    """
    return (a == q.reshape(-1, 1)).any(1).mean()


def retrieval_metrics(q, a, ranks=(1, 2, 4, 8)):
    """Average precision for a retrieval task

    Parameters
    ----------
    q : np.ndarray
        array of shape [n]
    a : np.ndarray
        array of shape [n, k] of sorted labels to match against
    ranks : iterable
        rank to compute mAP and recall at. If None then this will be overall AP.

    Returns
    -------
    dict with keys mAP and mAP_r and rec_r for r in ranks
    """
    if q.shape[0] != a.shape[0]:
        raise ValueError("Shapes of q and a must agree at axis 0")
    ret = {"mAP": map_score(q, a)}
    for r in ranks:
        suba = a[:, :r]
        ret[f"mAP_{r}"] = map_score(q, suba)
        ret[f"rec_{r}"] = recall(q, suba)
    return ret
