"""
Reconstruction Error Detector (STANDARD Mode)
Optimal linear subspace autoencoder (SVD/PCA projection) with learned reconstruction residue.
Trained strictly on nominal training data to avoid any data leakage.
"""

import numpy as np


class AutoencoderDetector:
    def __init__(self, input_dim: int = 6, latent_dim: int = 3):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.mean = np.zeros(input_dim)
        self.components = np.eye(input_dim, latent_dim)
        self.baseline_mean = 0.0
        self.baseline_std = 1.0

    def fit_normal_baseline(self, normal_samples: np.ndarray):
        """Fits optimal subspace autoencoder and computes baseline residue statistics."""
        self.mean = np.mean(normal_samples, axis=0)
        X_centered = normal_samples - self.mean
        # SVD decomposition: optimal low-rank reconstruction
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components = Vt[: self.latent_dim].T  # (input_dim x latent_dim)

        # Baseline reconstruction residue on normal samples
        proj = np.dot(X_centered, self.components)
        recon = np.dot(proj, self.components.T)
        errors = np.mean((X_centered - recon) ** 2, axis=1)

        self.baseline_mean = float(np.mean(errors))
        self.baseline_std = float(np.std(errors) + 1e-6)

    def compute_anomaly_score(self, sample: np.ndarray) -> float:
        """
        Calculates reconstruction residue z-score and squashes to [0.0, 1.0].
        """
        x_centered = (sample - self.mean).reshape(1, -1)
        proj = np.dot(x_centered, self.components)
        recon = np.dot(proj, self.components.T)
        mse = float(np.mean((x_centered - recon) ** 2))

        z = (mse - self.baseline_mean) / self.baseline_std
        # Logistic activation with inflection at 3-sigma
        score = float(1.0 / (1.0 + np.exp(-1.2 * (z - 3.0))))
        return min(max(score, 0.0), 1.0)

    def predict(self, sample: np.ndarray, threshold: float = 0.5) -> bool:
        score = self.compute_anomaly_score(sample)
        return score >= threshold
