#!/usr/bin/env python3
"""
AeroEdge-AI: Quickstart Demonstration Script
Executes real-time UAV telemetry processing, adaptive model switching,
and prints live performance benchmarks.
"""

import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from experiments.benchmark_comparison import run_benchmark
from experiments.verify_benchmarks import verify

def main():
    print("\nStarting AeroEdge-AI Benchmark Execution...\n")
    run_benchmark()
    print("\nRunning Automated Scientific Audit...\n")
    verify()
    print("\n[SUCCESS] Demo completed successfully. Ready for evaluation!\n")

if __name__ == "__main__":
    main()
