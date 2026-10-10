"""
Statistical Z-Score / Dynamic IQR Anomaly Detector (LIGHT Mode)
Optimized for ultra-low computational overhead and millisecond inference on edge UAV microcontrollers.
"""

import numpy as np


class StatisticalDetector:
    def __init__(self, window_size: int = 30, z_threshold: float = 2.5):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.history = []

    def update_history(self, sample: np.ndarray):
        self.history.append(sample)
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def compute_anomaly_score(self, sample: np.ndarray) -> float:
        """
        Calculates normalized outlier score using robust rolling statistics.
        Returns a float in [0.0, 1.0].
        """
        if len(self.history) < 5:
            self.update_history(sample)
            return 0.0

        hist_arr = np.array(self.history)
        mean = np.mean(hist_arr, axis=0)
        std = np.std(hist_arr, axis=0) + 1e-6

        z_scores = np.abs((sample - mean) / std)
        max_z = np.max(z_scores)

        # Update history
        self.update_history(sample)

        # Normalize score via sigmoid scaling around threshold
        normalized_score = float(1.0 / (1.0 + np.exp(-1.2 * (max_z - self.z_threshold))))
        return min(max(normalized_score, 0.0), 1.0)

    def predict(self, sample: np.ndarray, threshold: float = 0.5) -> bool:
        score = self.compute_anomaly_score(sample)
        return score >= threshold
