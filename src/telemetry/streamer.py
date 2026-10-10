"""
UAV Telemetry Stream Generator (Synthetic Flight Dynamics)
Generates high-fidelity multi-sensor time-series telemetry with physical flight correlations.
Provides cleanly separated Train (70%), Validation (15%), and Test (15%) partitions.
Training set contains nominal flight data; Validation & Test sets contain realistic 5% anomalies.
"""

from typing import Dict, List, Tuple
import numpy as np


class TelemetryStreamer:
    def __init__(self, n_samples: int = 5000, anomaly_rate: float = 0.05, seed: int = 42):
        self.n_samples = n_samples
        self.anomaly_rate = anomaly_rate
        self.seed = seed
        self.feature_names = [
            "accel_z",       # Vertical acceleration (m/s^2)
            "gyro_pitch",     # Angular pitch rate (deg/s)
            "motor_rpm",      # Primary propulsion RPM
            "battery_volt",   # Avionics bus voltage (V)
            "altitude_m",     # Barometric altitude (m)
            "airspeed_mps",   # Pitot airspeed (m/s)
        ]

    def _generate_nominal(self, n: int, start_t: float, rng: np.random.RandomState) -> np.ndarray:
        t = np.linspace(start_t, start_t + (n * 0.05), n)
        accel_z = 9.81 + 0.3 * np.sin(0.4 * t) + rng.normal(0, 0.05, n)
        gyro_pitch = 1.2 * np.cos(0.25 * t) + rng.normal(0, 0.08, n)
        motor_rpm = 5400 + 120 * np.sin(0.15 * t) + rng.normal(0, 15.0, n)
        battery_volt = 16.8 - 0.0002 * np.arange(n) + rng.normal(0, 0.02, n)
        altitude_m = 120.0 + 5.0 * np.sin(0.08 * t) + rng.normal(0, 0.1, n)
        airspeed_mps = 18.5 + 1.5 * np.cos(0.1 * t) + rng.normal(0, 0.15, n)
        return np.column_stack([accel_z, gyro_pitch, motor_rpm, battery_volt, altitude_m, airspeed_mps])

    def _inject_anomalies(self, X: np.ndarray, rate: float, rng: np.random.RandomState) -> np.ndarray:
        n = len(X)
        y = np.zeros(n, dtype=int)
        target_anom = int(n * rate)
        idx = 20

        while np.sum(y) < target_anom and idx < n - 15:
            burst_len = rng.randint(4, 9)
            burst_len = min(burst_len, target_anom - int(np.sum(y)))
            anom_type = rng.choice(["spike", "drift", "motor_drop", "voltage_dip"])

            for b in range(burst_len):
                if idx + b >= n:
                    break
                y[idx + b] = 1
                if anom_type == "spike":
                    X[idx + b, 0] += rng.uniform(4.0, 7.5)   # Heavy G shock
                    X[idx + b, 1] += rng.uniform(10.0, 18.0) # Pitch instability
                elif anom_type == "drift":
                    X[idx + b, 1] += 2.5 * (b + 1)           # Sustained attitude divergence
                elif anom_type == "motor_drop":
                    X[idx + b, 2] -= rng.uniform(1200, 2200) # Sudden motor power loss
                elif anom_type == "voltage_dip":
                    X[idx + b, 3] -= rng.uniform(2.0, 3.5)   # Power sag
            idx += burst_len + rng.randint(15, 30)

        return y

    def generate_dataset(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        rng = np.random.RandomState(self.seed)

        n_train = int(0.70 * self.n_samples)  # 3500
        n_val = int(0.15 * self.n_samples)    # 750
        n_test = self.n_samples - n_train - n_val  # 750

        # Train: Nominal flight
        X_train = self._generate_nominal(n_train, 0.0, rng)
        y_train = np.zeros(n_train, dtype=int)

        # Validation: Nominal + 5% injected anomalies
        X_val = self._generate_nominal(n_val, 200.0, rng)
        y_val = self._inject_anomalies(X_val, self.anomaly_rate, rng)

        # Test: Nominal + 5% injected anomalies
        X_test = self._generate_nominal(n_test, 400.0, rng)
        y_test = self._inject_anomalies(X_test, self.anomaly_rate, rng)

        X = np.vstack([X_train, X_val, X_test])
        y = np.concatenate([y_train, y_val, y_test])

        splits = {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }
        return X, y, splits
