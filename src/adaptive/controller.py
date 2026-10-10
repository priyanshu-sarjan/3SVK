"""
Adaptive Controller Module (AeroEdge-AI)
Dynamically selects model mode m in {LIGHT, STANDARD, HEAVY} and
transmission decision t in {LOCAL_ONLY, TRANSMIT} by maximizing a
rigorously formulated utility function under physical edge constraints.
"""

from enum import Enum
from typing import Dict, Tuple
import numpy as np


class ModelMode(Enum):
    LIGHT = "LIGHT"
    STANDARD = "STANDARD"
    HEAVY = "HEAVY"


class TransmissionMode(Enum):
    LOCAL_ONLY = "LOCAL_ONLY"
    TRANSMIT = "TRANSMIT"


class AdaptiveController:
    def __init__(
        self,
        w_cpu: float = 0.20,
        w_mem: float = 0.15,
        w_lat: float = 0.25,
        w_conf: float = 0.25,
        w_trans: float = 0.10,
        w_comp: float = 0.05,
        cpu_threshold: float = 0.85,
        mem_threshold: float = 0.85,
        latency_max: float = 120.0,  # ms
    ):
        self.w_cpu = w_cpu
        self.w_mem = w_mem
        self.w_lat = w_lat
        self.w_conf = w_conf
        self.w_trans = w_trans
        self.w_comp = w_comp

        self.cpu_threshold = cpu_threshold
        self.mem_threshold = mem_threshold
        self.latency_max = latency_max

        # Model resource profiles (relative overheads in [0, 1])
        self.profiles = {
            ModelMode.LIGHT: {
                "cpu_cost": 0.05,
                "mem_cost": 0.04,
                "lat_local_ms": 1.2,
                "complexity": 0.1,
                "base_confidence": 0.82,
            },
            ModelMode.STANDARD: {
                "cpu_cost": 0.18,
                "mem_cost": 0.12,
                "lat_local_ms": 4.5,
                "complexity": 0.4,
                "base_confidence": 0.91,
            },
            ModelMode.HEAVY: {
                "cpu_cost": 0.45,
                "mem_cost": 0.32,
                "lat_local_ms": 12.8,
                "complexity": 0.9,
                "base_confidence": 0.97,
            },
        }

        # Decision logs for auditing
        self.decision_history = []

    def compute_utility(
        self,
        mode: ModelMode,
        trans: TransmissionMode,
        cpu_util: float,
        mem_util: float,
        net_latency_ms: float,
        anomaly_score: float,
    ) -> float:
        """
        Calculates utility U(m, t) as defined in 3SVK Season 3 proceedings:
        U(m, t) = w_cpu * (1 - cpu_cost(m)) +
                  w_mem * (1 - mem_cost(m)) +
                  w_lat * (1 - latency_cost(m, t)) +
                  w_conf * confidence(m) -
                  w_trans * trans_cost(t) -
                  w_comp * complexity_cost(m)
        """
        p = self.profiles[mode]

        # Feasibility checks
        if cpu_util + p["cpu_cost"] > self.cpu_threshold:
            return -1e6
        if mem_util + p["mem_cost"] > self.mem_threshold:
            return -1e6
        if trans == TransmissionMode.TRANSMIT and net_latency_ms > self.latency_max:
            return -1e6

        # Normalized latency cost
        eff_latency = p["lat_local_ms"] + (net_latency_ms if trans == TransmissionMode.TRANSMIT else 0.0)
        lat_cost = min(eff_latency / self.latency_max, 1.0)

        # Transmission cost (bandwidth / energy)
        trans_cost = 0.8 if trans == TransmissionMode.TRANSMIT else 0.0

        # When anomaly is suspected, promote to higher fidelity model
        confidence = p["base_confidence"] * (1.0 + 1.2 * anomaly_score if anomaly_score > 0.3 else 1.0)

        utility = (
            self.w_cpu * (1.0 - p["cpu_cost"])
            + self.w_mem * (1.0 - p["mem_cost"])
            + self.w_lat * (1.0 - lat_cost)
            + self.w_conf * min(confidence, 1.0)
            - self.w_trans * trans_cost
            - self.w_comp * p["complexity"]
        )
        return utility

    def select_action(
        self,
        cpu_util: float,
        mem_util: float,
        net_latency_ms: float,
        preliminary_score: float = 0.0,
    ) -> Tuple[ModelMode, TransmissionMode, float]:
        """
        Evaluates all candidate actions (mode, transmission) and returns the optimal pair.
        """
        best_u = -float("inf")
        best_mode = ModelMode.LIGHT
        best_trans = TransmissionMode.LOCAL_ONLY

        modes = [ModelMode.LIGHT, ModelMode.STANDARD, ModelMode.HEAVY]
        transmissions = [TransmissionMode.LOCAL_ONLY, TransmissionMode.TRANSMIT]

        for m in modes:
            for t in transmissions:
                u = self.compute_utility(
                    mode=m,
                    trans=t,
                    cpu_util=cpu_util,
                    mem_util=mem_util,
                    net_latency_ms=net_latency_ms,
                    anomaly_score=preliminary_score,
                )
                if u > best_u:
                    best_u = u
                    best_mode = m
                    best_trans = t

        record = {
            "mode": best_mode.value,
            "transmission": best_trans.value,
            "utility": round(best_u, 4),
            "cpu_util": round(cpu_util, 3),
            "mem_util": round(mem_util, 3),
            "net_latency": round(net_latency_ms, 2),
        }
        self.decision_history.append(record)
        return best_mode, best_trans, best_u
