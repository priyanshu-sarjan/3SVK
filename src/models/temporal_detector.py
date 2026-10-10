"""
Temporal Sequence Anomaly Detector (HEAVY Mode)
Vector Autoregressive (VAR) sequence predictor capturing temporal momentum and inter-sensor dynamics.
Trained strictly on nominal training data to avoid any test leakage.
"""

import numpy as np


class TemporalSequenceDetector:
    def __init__(self, input_dim: int = 6, sequence_length: int = 15, lag: int = 3):
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.lag = lag
        self.buffer = []
        self.A = np.zeros((lag * input_dim, input_dim))
        self.mean = np.zeros(input_dim)
        self.baseline_mean = 0.0
        self.baseline_std = 1.0

    def fit_normal_baseline(self, normal_samples: np.ndarray):
        """Fits Vector Autoregressive transition matrix on nominal time-series."""
        self.mean = np.mean(normal_samples, axis=0)
        X_c = normal_samples - self.mean
        n = len(X_c)

        # Build lag design matrix
        Z = []
        Y = []
        for i in range(self.lag, n):
            z_i = X_c[i - self.lag : i].flatten()
            Z.append(z_i)
            Y.append(X_c[i])

        Z = np.array(Z)
        Y = np.array(Y)

        # Ridge regression solution for VAR
        reg = 1e-3
        self.A = np.linalg.solve(np.dot(Z.T, Z) + reg * np.eye(Z.shape[1]), np.dot(Z.T, Y))

        # Baseline prediction residuals
        preds = np.dot(Z, self.A)
        residuals = np.mean((Y - preds) ** 2, axis=1)
        self.baseline_mean = float(np.mean(residuals))
        self.baseline_std = float(np.std(residuals) + 1e-6)

    def compute_anomaly_score(self, sample: np.ndarray) -> float:
        """
        Calculates sequence prediction error over rolling buffer.
        """
        self.buffer.append(sample)
        if len(self.buffer) > self.lag + 1:
            self.buffer.pop(0)

        if len(self.buffer) <= self.lag:
            return 0.0

        x_c_buffer = np.array(self.buffer) - self.mean
        z_t = x_c_buffer[:-1].flatten()
        actual = x_c_buffer[-1]
        predicted = np.dot(z_t, self.A)

        mse = float(np.mean((actual - predicted) ** 2))
        z = (mse - self.baseline_mean) / self.baseline_std
        score = float(1.0 / (1.0 + np.exp(-1.1 * (z - 2.8))))
        return min(max(score, 0.0), 1.0)

    def predict(self, sample: np.ndarray, threshold: float = 0.5) -> bool:
        score = self.compute_anomaly_score(sample)
        return score >= threshold
