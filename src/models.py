"""Classifiers + legacy one-class clustering baseline."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def make_logreg(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_random_forest(random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )


def make_lightgbm(random_state: int = 42, n_pos: int = 1, n_neg: int = 1):
    scale = max(n_neg / max(n_pos, 1), 1.0)
    return LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        random_state=random_state,
        verbosity=-1,
    )


def make_xgboost(random_state: int = 42, n_pos: int = 1, n_neg: int = 1):
    scale = max(n_neg / max(n_pos, 1), 1.0)
    return XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        tree_method="hist",
        random_state=random_state,
        eval_metric="aucpr",
    )


# ---------------------------------------------------------------------------
# legacy one-class clustering (for comparison)
# ---------------------------------------------------------------------------

@dataclass
class ClusterBaseline:
    """Train-on-positives nearest-cluster baseline.

    Reproduces the approach from the original notebooks (DBSCAN/KMeans
    trained on positive samples only) but wrapped in a sane API.
    """
    kind: str = "kmeans"  # "dbscan" or "kmeans"
    n_clusters: int = 2
    eps: float = 0.1
    min_samples: int = 5
    random_state: int = 42

    def fit(self, X_pos: np.ndarray):
        self.scaler_ = StandardScaler().fit(X_pos)
        Xs = self.scaler_.transform(X_pos)
        if self.kind == "dbscan":
            self.model_ = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(Xs)
            self.core_ = Xs[self.model_.core_sample_indices_]
        else:
            self.model_ = KMeans(
                n_clusters=self.n_clusters, random_state=self.random_state, n_init="auto"
            ).fit(Xs)
            # per-cluster radius = 95th percentile of distances to centroid
            dists = euclidean_distances(Xs, self.model_.cluster_centers_)
            labels = self.model_.labels_
            self.radii_ = np.array(
                [
                    np.quantile(dists[labels == k, k], 0.95)
                    if (labels == k).any()
                    else 0.0
                    for k in range(self.n_clusters)
                ]
            )
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return a score in [0, 1] – higher means 'closer to a cluster'."""
        Xs = self.scaler_.transform(X)
        if self.kind == "dbscan":
            if self.core_.shape[0] == 0:
                return np.zeros(Xs.shape[0])
            d = euclidean_distances(Xs, self.core_).min(axis=1)
            return np.clip(1.0 - d / (self.eps + 1e-9), 0.0, 1.0)
        d = euclidean_distances(Xs, self.model_.cluster_centers_)
        r = self.radii_[np.newaxis, :]
        in_ball = np.clip(1.0 - d / (r + 1e-9), 0.0, 1.0)
        return in_ball.max(axis=1)
