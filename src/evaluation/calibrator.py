"""
Validation-Based Threshold Calibrator (Leakage Prevention)
Calibrates anomaly decision thresholds strictly on the validation partition
to guarantee zero test-set leakage, as mandated by the 3SVK Season 3 audit standards.
"""

from typing import Callable, List, Tuple
import numpy as np


class ThresholdCalibrator:
    def __init__(self, n_thresholds: int = 100):
        self.n_thresholds = n_thresholds
        self.calibrated_threshold = 0.5
        self.best_val_f1 = 0.0

    def calibrate_f1(self, val_scores: np.ndarray, val_labels: np.ndarray) -> float:
        """
        Maximizes F1 score strictly on validation labels.
        """
        thresholds = np.linspace(0.01, 0.99, self.n_thresholds)
        best_f1 = -1.0
        best_th = 0.5

        for th in thresholds:
            preds = (val_scores >= th).astype(int)
            tp = np.sum((preds == 1) & (val_labels == 1))
            fp = np.sum((preds == 1) & (val_labels == 0))
            fn = np.sum((preds == 0) & (val_labels == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_th = th

        self.calibrated_threshold = float(best_th)
        self.best_val_f1 = float(best_f1)
        return self.calibrated_threshold
