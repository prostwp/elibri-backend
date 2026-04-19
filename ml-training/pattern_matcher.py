"""
pattern_matcher.py — k-NN on feature vectors for "similar historical situation".

Stores scaled features + outcomes in a BallTree. At inference, query current
feature vector → top-k nearest historical rows with their realized 5/10/20-day
returns. Used for UI wow-factor: "похожая ситуация была 15 марта → +23%".

The index is also serializable to JSON for Go consumption (flat arrays).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler


@dataclass
class SimilarSituation:
    date: str            # ISO-8601
    distance: float      # in scaled feature space
    outcome_5: float     # return 5 bars later
    outcome_10: float    # return 10 bars later
    outcome_20: float    # return 20 bars later
    description: str     # human-readable hint


class PatternIndex:
    """BallTree over z-scored features. Keeps raw outcomes for recall."""

    def __init__(self, feature_cols: List[str]):
        self.feature_cols = feature_cols
        self.scaler = StandardScaler()
        self.tree: BallTree | None = None
        self.outcomes: np.ndarray | None = None  # shape (N, 3) for 5/10/20
        self.timestamps: List[str] = []
        self.closes: np.ndarray | None = None

    def fit(self, features_df: pd.DataFrame, closes: np.ndarray, timestamps: List[str]) -> "PatternIndex":
        """
        features_df: N rows × len(feature_cols) columns.
        closes:      N-length array of close prices, aligned.
        timestamps:  N ISO-8601 strings.
        """
        X = features_df[self.feature_cols].to_numpy()
        X_scaled = self.scaler.fit_transform(X)

        # Compute forward outcomes (NaN for last 20 bars).
        n = len(closes)
        out = np.full((n, 3), np.nan)
        for horizon_idx, h in enumerate((5, 10, 20)):
            shift = np.roll(closes, -h)
            ret = (shift - closes) / (closes + 1e-12)
            ret[-h:] = np.nan
            out[:, horizon_idx] = ret

        # Only keep rows with complete outcomes.
        mask = ~np.isnan(out).any(axis=1)
        X_scaled = X_scaled[mask]
        self.outcomes = out[mask]
        self.closes = closes[mask]
        self.timestamps = [timestamps[i] for i in np.where(mask)[0]]

        self.tree = BallTree(X_scaled, leaf_size=40)
        return self

    def query(self, current_feats: np.ndarray, k: int = 10) -> List[SimilarSituation]:
        if self.tree is None:
            return []
        x = self.scaler.transform(current_feats.reshape(1, -1))
        distances, indices = self.tree.query(x, k=min(k, len(self.timestamps)))
        results: List[SimilarSituation] = []
        for d, i in zip(distances[0], indices[0]):
            out5, out10, out20 = self.outcomes[i]
            results.append(
                SimilarSituation(
                    date=self.timestamps[i],
                    distance=float(d),
                    outcome_5=float(out5),
                    outcome_10=float(out10),
                    outcome_20=float(out20),
                    description=_describe(d, out10),
                )
            )
        return results

    def to_json_payload(self) -> dict:
        """Flat serialization: Go reads scaler + samples (no BallTree — Go uses linear scan for k≤20)."""
        if self.tree is None:
            return {}
        # Recover scaled features for export.
        # BallTree stores data internally; grab from the tree.
        scaled = np.asarray(self.tree.data)
        return {
            "feature_cols": self.feature_cols,
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "samples": scaled.tolist(),  # N × F scaled vectors
            "outcomes": self.outcomes.tolist(),
            "timestamps": self.timestamps,
            "closes": self.closes.tolist() if self.closes is not None else [],
        }

    @classmethod
    def from_json_payload(cls, payload: dict) -> "PatternIndex":
        idx = cls(payload["feature_cols"])
        idx.scaler = StandardScaler()
        idx.scaler.mean_ = np.array(payload["scaler_mean"])
        idx.scaler.scale_ = np.array(payload["scaler_scale"])
        samples = np.array(payload["samples"])
        idx.tree = BallTree(samples, leaf_size=40)
        idx.outcomes = np.array(payload["outcomes"])
        idx.timestamps = payload["timestamps"]
        idx.closes = np.array(payload.get("closes", []))
        return idx


def _describe(distance: float, outcome_10: float) -> str:
    """Generate human-readable hint for UI."""
    closeness = "очень похожая" if distance < 1 else "схожая" if distance < 2 else "отдалённая"
    if outcome_10 > 0.05:
        move = f"+{outcome_10 * 100:.1f}%"
    elif outcome_10 < -0.05:
        move = f"{outcome_10 * 100:.1f}%"
    else:
        move = f"{outcome_10 * 100:+.1f}%"
    return f"{closeness} ситуация → {move} за 10 баров"


def save_index(idx: PatternIndex, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(idx.to_json_payload(), f)


def load_index(path: Path) -> PatternIndex:
    with open(path) as f:
        payload = json.load(f)
    return PatternIndex.from_json_payload(payload)
