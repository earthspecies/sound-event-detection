"""
Taken from DCASE 2021 Task 5 evaluation source code
https://github.com/c4dm/dcase-few-shot-bioacoustic

The function _bipartite_match was taken from
https://github.com/mir-evaluation/mir_eval

Both are used under MIT License
"""

from typing import Dict, Hashable, List, Tuple

import numpy as np
import scipy


def fast_intersect(ref: np.ndarray, est: np.ndarray) -> List:
    """Find all intersections between reference events and estimated events (fast).

    Best-case complexity: O(N log N + M log M) where N=length(ref) and M=length(est)

    Parameters
    ----------
    ref : np.ndarray, shape (2, n)
        Array of reference events. Each column is an event.
        The first row denotes onset times and the second row denotes offset times.
    est : np.ndarray, shape (2, m)
        Array of estimated events. Each column is an event.

    Returns
    -------
    list of sets, length n
        ``matches[i]`` contains the set of all indices j such that
        ``(ref[0, i] <= est[1, j]) AND (ref[1, i] >= est[0, j])``.
    """
    ref_on_argsort = np.argsort(ref[0, :])
    ref_off_argsort = np.argsort(ref[1, :])

    est_on_argsort = np.argsort(est[0, :])
    est_off_argsort = np.argsort(est[1, :])

    est_on_maxindex = est.shape[1]
    est_off_minindex = 0
    estref_matches = [set()] * ref.shape[1]
    refest_matches = [set()] * ref.shape[1]
    for ref_id in range(ref.shape[1]):
        ref_onset = ref[0, ref_on_argsort[ref_id]]
        est_off_sorted = est[1, est_off_argsort[est_off_minindex:]]
        search_result = np.searchsorted(est_off_sorted, ref_onset, side="left")
        est_off_minindex += search_result
        refest_match = est_off_argsort[est_off_minindex:]
        refest_matches[ref_on_argsort[ref_id]] = set(refest_match)

        ref_offset = ref[1, ref_off_argsort[-1 - ref_id]]
        est_on_sorted = est[0, est_on_argsort[: (1 + est_on_maxindex)]]
        search_result = np.searchsorted(est_on_sorted, ref_offset, side="right")
        est_on_maxindex = search_result - 1
        estref_match = est_on_argsort[: (1 + est_on_maxindex)]
        estref_matches[ref_off_argsort[-1 - ref_id]] = set(estref_match)

    zip_iterator = zip(refest_matches, estref_matches, strict=False)
    matches = [x.intersection(y) for (x, y) in zip_iterator]
    return matches


def iou(ref: np.ndarray, est: np.ndarray) -> scipy.sparse.dok_matrix:
    """Compute pairwise intersection over union (IOU) between reference and estimated events.

    Parameters
    ----------
    ref : np.ndarray, shape (2, n)
        Array of reference events. Each column is an event.
    est : np.ndarray, shape (2, m)
        Array of estimated events. Each column is an event.

    Returns
    -------
    scipy.sparse.dok_matrix
        Sparse 2-D matrix. S[i,j] contains the IOU between ref[i] and est[j]
        if these events are non-disjoint and zero otherwise.
    """
    n_refs = ref.shape[1]
    n_ests = est.shape[1]
    S = scipy.sparse.dok_matrix((n_refs, n_ests))

    matches = fast_intersect(ref, est)

    for ref_id in range(n_refs):
        matching_ests = matches[ref_id]
        ref_on = ref[0, ref_id]
        ref_off = ref[1, ref_id]

        for matching_est_id in matching_ests:
            est_on = est[0, matching_est_id]
            est_off = est[1, matching_est_id]
            intersection = min(ref_off, est_off) - max(ref_on, est_on)
            union = max(ref_off, est_off) - min(ref_on, est_on)
            intersection_over_union = intersection / union
            S[ref_id, matching_est_id] = intersection_over_union

    return S


def compute_intersection(ref: np.ndarray, est: np.ndarray) -> scipy.sparse.dok_matrix:
    """Compute pairwise intersection between reference and estimated events.

    Parameters
    ----------
    ref : np.ndarray, shape (2, n)
        Array of reference events.
    est : np.ndarray, shape (2, m)
        Array of estimated events.

    Returns
    -------
    scipy.sparse.dok_matrix
        Sparse 2-D matrix. S[i,j] contains the intersection between ref[i] and est[j].
    """
    n_refs = ref.shape[1]
    n_ests = est.shape[1]
    S = scipy.sparse.dok_matrix((n_refs, n_ests))

    matches = fast_intersect(ref, est)

    for ref_id in range(n_refs):
        matching_ests = matches[ref_id]
        ref_on = ref[0, ref_id]
        ref_off = ref[1, ref_id]

        for matching_est_id in matching_ests:
            est_on = est[0, matching_est_id]
            est_off = est[1, matching_est_id]
            intersection = min(ref_off, est_off) - max(ref_on, est_on)
            S[ref_id, matching_est_id] = intersection

    return S


def _bipartite_match(graph: Dict) -> Dict:
    """Find maximum cardinality matching of a bipartite graph (U,V,E).

    The input format is a dictionary mapping members of U to a list
    of their neighbors in V.

    Parameters
    ----------
    graph : dict
        Left-vertex → list of right vertices bipartite graph.

    Returns
    -------
    dict
        Mapping of right vertices to their matched left vertex.
    """
    matching = {}
    for u in graph:
        for v in graph[u]:
            if v not in matching:
                matching[v] = u
                break

    while True:
        preds = {}
        unmatched = []
        pred = {u: unmatched for u in graph}
        for v in matching:
            del pred[matching[v]]
        layer = list(pred)

        while layer and not unmatched:
            new_layer = {}
            for u in layer:
                for v in graph[u]:
                    if v not in preds:
                        new_layer.setdefault(v, []).append(u)
            layer = []
            for v in new_layer:
                preds[v] = new_layer[v]
                if v in matching:
                    layer.append(matching[v])
                    pred[matching[v]] = v
                else:
                    unmatched.append(v)

        if not unmatched:
            unlayered = {}
            for u in graph:
                for v in graph[u]:
                    if v not in preds:
                        unlayered[v] = None
            return matching

        def recurse(
            v: Hashable,
            preds: Dict,
            pred: Dict,
            matching: Dict,
            unmatched: List,
        ) -> bool:
            """Recursively search backward through layers to find alternating paths.

            Returns
            -------
            bool
                True if a path was found, False otherwise.

            Notes
            -----
            This function mutates ``preds``, ``pred``, and ``matching`` in-place.
            """
            if v in preds:
                L = preds[v]
                del preds[v]
                for u in L:
                    if u in pred:
                        pu = pred[u]
                        del pred[u]
                        if pu is unmatched or recurse(pu, preds, pred, matching, unmatched):
                            matching[v] = u
                            return True
            return False

        for v in unmatched:
            recurse(v, preds, pred, matching, unmatched)


def match_events(ref: np.ndarray, est: np.ndarray, min_iou: float = 0.0) -> List[Tuple[int, int]]:
    """Compute a maximum matching between reference and estimated events.

    Parameters
    ----------
    ref : np.ndarray, shape (2, n)
        Array of reference events.
    est : np.ndarray, shape (2, m)
        Array of estimated events.
    min_iou : float
        Minimum IOU to consider a match.

    Returns
    -------
    list of tuple[int, int]
        Every tuple is a matched (ref_idx, est_idx) pair.
    """
    S = iou(ref, est)
    S_bool = scipy.sparse.dok_matrix(S > min_iou)
    hits = S_bool.keys()

    G = {}
    for ref_i, est_i in hits:
        if est_i not in G:
            G[est_i] = []
        G[est_i].append(ref_i)

    matching = sorted(_bipartite_match(G).items())
    return matching
