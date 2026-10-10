# AeroEdge-AI: Resource-Aware Adaptive Edge Intelligence Framework

[![3SVK Season 3](https://img.shields.io/badge/3SVK%20Challenge-Season%203-blue.svg)](https://sites.google.com/view/3svk-research-s3/home)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22008262.svg)](https://doi.org/10.5281/zenodo.22008262)

Official research archive, documentation, model implementation, and proceedings for the **National Research & Innovation Challenge Season 3 by 3SVK**.

---

## 📌 Executive Summary

**AeroEdge-AI** is a resource-aware adaptive edge intelligence framework designed for real-time UAV telemetry monitoring and anomaly detection under extreme compute and communication constraints.

By dynamically arbitrating between tiered on-device detection engines (Statistical Z-Score, SVD Subspace Autoencoder, and Vector Autoregressive sequence models) and pairing with field-level delta compression, AeroEdge-AI achieves:
- **99.49% Latency Reduction** compared to Centralized Cloud architectures (from $243.86\text{ ms}$ down to $1.25\text{ ms}$).
- **91.85% Bandwidth Reduction** (from $75,431\text{ B}$ down to $6,150\text{ B}$) via intelligent selective transmission.
- **100.0% Operational Uptime** and zero telemetry loss across $1,000$ simulated link drop events using idempotent queuing.
- **Zero Test-Set Leakage** through strict validation-based F1 threshold calibration.

---

## 🏛️ System Architecture

```
[ UAV Avionics / Multi-Sensor Bus (20 Hz) ]
   (accel_z, gyro_pitch, motor_rpm, battery_volt, altitude, airspeed)
                       │
                       ▼
       ┌───────────────────────────────┐
       │   Adaptive Utility Controller │ ◄── [ Hardware Telemetry ]
       │  max U(mode, transmission)    │     (CPU %, RAM %, Link Latency)
       └───────────────┬───────────────┘
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
┌──────────────┐┌──────────────┐┌──────────────┐
│  LIGHT Mode  ││STANDARD Mode ││  HEAVY Mode  │
│Statistical Z ││SVD Subspace  ││VAR Temporal  │
│  (1.2 ms)    ││Autoencoder   ││Sequence Model│
└──────┬───────┘└──────┬───────┘└──────┬───────┘
       └───────────────┼───────────────┘
                       │ Anomaly Score
                       ▼
       ┌───────────────────────────────┐
       │ Zero-Leakage Threshold Gate   │
       └───────────────┬───────────────┘
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
[ Normal Baseline ]            [ Anomaly Detected! ]
(Periodic Delta Sync)         (Instant Prioritized Alert)
       │                               │
       └───────────────┬───────────────┘
                       ▼
       ┌───────────────────────────────┐
       │ Tamper-Evident SHA-256 Ledger │
       │  & Delta-Compression Engine   │
       └───────────────┬───────────────┘
                       ▼
       [ Idempotent Persistent Queue ]
                       ▼
    [ Ground Station / Cloud Uplink ]
```

---

## 📊 Benchmark Performance Summary

| Metric | Centralized Cloud | Static Edge | AeroEdge-AI (Ours) | Advantage |
|---|---|---|---|---|
| **Mean Latency (ms)** | 243.86 ms | 3.81 ms | **1.25 ms** | **99.49% Reduction** |
| **P95 Latency (ms)** | 277.01 ms | 3.81 ms | **1.30 ms** | Real-time guaranteed |
| **Payload Transferred** | 75,431 B | 0 B | **6,150 B** | **91.85% Reduction** |
| **System Uptime** | 91.4% | 100.0% | **100.0%** | Resilient queuing |
| **Classification Accuracy**| 94.67% | 99.07% | **96.13%** | High fidelity |
| **Precision** | 38.46% | 100.00% | **72.22%** | Controlled FPR |
| **Audit Status** | Baseline | Baseline | **PASS (Zero Leakage)**| Certified |

---

## 🚀 Quickstart & Demo Execution

Clone the repository and run the quickstart suite:

```bash
git clone https://github.com/priyanshu-sarjan/3SVK.git
cd 3SVK

# Using uv (fastest)
uv run run_demo.py

# Or standard Python 3.10+
pip install -r requirements.txt
python run_demo.py
```

To run individual experiments:
```bash
# Run comparative benchmarks
python experiments/benchmark_comparison.py

# Run scientific audit verification
python experiments/verify_benchmarks.py
```

---

## 📂 Project Structure

```
3SVK/
├── PROCEEDINGS.md               # Official 3SVK Season 3 Research Proceedings
├── README.md                    # System documentation and benchmarks
├── SUBMISSION_DELIVERABLES.md   # Unstop submission guide & audit proof
├── LICENSE                      # MIT Open-Source License
├── requirements.txt             # Dependency specification
├── run_demo.py                  # End-to-end executable demo script
├── configs/
│   └── default_config.yaml      # Telemetry & controller settings
├── src/
│   ├── adaptive/
│   │   └── controller.py        # Utility-maximizing adaptive controller
│   ├── evaluation/
│   │   ├── calibrator.py        # Zero-leakage validation threshold calibrator
│   │   └── metrics.py           # Confusion matrix & performance metrics
│   ├── models/
│   │   ├── statistical_detector.py # LIGHT mode (Z-Score filter)
│   │   ├── autoencoder_detector.py # STANDARD mode (SVD autoencoder)
│   │   └── temporal_detector.py    # HEAVY mode (VAR sequence model)
│   └── telemetry/
│       ├── streamer.py          # UAV flight dynamics generator
│       └── delta_encoder.py     # Delta compression & idempotent queue
├── experiments/
│   ├── benchmark_comparison.py  # Comparative evaluation suite
│   └── verify_benchmarks.py     # Automated scientific audit script
└── results/
    └── benchmark_results.json   # Machine-readable verified benchmark data
```

---

## 📚 Official Citation

```bibtex
@misc{3svk_season3_2026,
  author    = {3SVK Official and Priyanshu Sarjan},
  title     = {3SVK Research Series Season 3: Official Archive & Proceedings},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22008262},
  url       = {https://doi.org/10.5281/zenodo.22008262}
}
```

**Intellectual Property & Legal Notice:**  
© 2026 3SVK Official & Priyanshu Sarjan. All rights reserved.
