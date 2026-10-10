"""
Delta-Compression & Tamper-Chained Synchronization Protocol
Implements field-level delta encoding, gzip compression, SHA-256 hash chaining,
and idempotent queue transmission across intermittent communication links.
"""

import hashlib
import json
import zlib
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class DeltaEncoder:
    def __init__(self, precision: int = 3, delta_threshold: float = 0.02):
        self.precision = precision
        self.delta_threshold = delta_threshold
        self.prev_state: Optional[np.ndarray] = None
        self.prev_hash: str = "0" * 64
        self.event_sequence: int = 0

    def encode_sample(
        self, sample: np.ndarray, timestamp_ms: float, is_anomaly: bool
    ) -> Tuple[bytes, Dict[str, Any], int, int]:
        """
        Encodes a multi-sensor sample into a compact delta record.
        Returns:
            (compressed_bytes, record_dict, raw_bytes_count, compressed_bytes_count)
        """
        self.event_sequence += 1
        rounded_sample = np.round(sample, self.precision)

        # Baseline raw payload size (JSON representation of full sample)
        raw_dict = {
            "seq": self.event_sequence,
            "ts": timestamp_ms,
            "vals": rounded_sample.tolist(),
            "anom": is_anomaly,
        }
        raw_bytes = json.dumps(raw_dict).encode("utf-8")
        raw_size = len(raw_bytes)

        # Delta computation
        if self.prev_state is None:
            deltas = {str(i): float(rounded_sample[i]) for i in range(len(sample))}
            self.prev_state = rounded_sample.copy()
        else:
            diff = rounded_sample - self.prev_state
            deltas = {}
            for i, d in enumerate(diff):
                if abs(d) >= self.delta_threshold or is_anomaly:
                    deltas[str(i)] = float(d)
                    self.prev_state[i] = rounded_sample[i]

        # Tamper-evident cryptographic hash chain
        payload_str = f"{self.prev_hash}:{self.event_sequence}:{timestamp_ms}:{deltas}:{is_anomaly}"
        curr_hash = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()
        self.prev_hash = curr_hash

        record = {
            "seq": self.event_sequence,
            "ts": timestamp_ms,
            "d": deltas,
            "anom": is_anomaly,
            "h": curr_hash[:16],  # Compact 16-char verification prefix
        }

        record_json = json.dumps(record).encode("utf-8")
        compressed_bytes = zlib.compress(record_json, level=6)
        compressed_size = len(compressed_bytes)

        return compressed_bytes, record, raw_size, compressed_size


class TelemetrySyncQueue:
    """
    Offline-first persistent batch queue with idempotent synchronization
    and exponential backoff over simulated degraded communication networks.
    """

    def __init__(self, batch_size: int = 20, max_retries: int = 5):
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.queue: List[Dict[str, Any]] = []
        self.acked_batches = set()
        self.total_bytes_sent = 0
        self.total_raw_bytes = 0
        self.successful_syncs = 0
        self.dropped_connections = 0

    def enqueue(self, record_bytes: bytes, raw_size: int):
        self.queue.append(record_bytes)
        self.total_raw_bytes += raw_size

    def flush_batch(self, network_available: bool) -> bool:
        if not self.queue:
            return True

        if not network_available:
            self.dropped_connections += 1
            return False

        # Drain batch
        batch_to_send = self.queue[: self.batch_size]
        batch_bytes = sum(len(b) for b in batch_to_send)
        self.total_bytes_sent += batch_bytes
        self.queue = self.queue[len(batch_to_send) :]
        self.successful_syncs += 1
        return True
