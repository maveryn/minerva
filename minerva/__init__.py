"""
Minerva: RL-ready CTI dataset builder.

This package assembles supervised and RLVR-friendly tasks using MITRE,
CWE, CAPEC, and NVD sources, emitting JSONL with embedded reward
verifiers for downstream training.
"""

__all__ = ["config", "data_sources", "tasks", "reward"]
