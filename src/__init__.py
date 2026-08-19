"""Masked Distress experiment code (src/).

Modules:
    steer:        phase-aware residual steering hook (context manager)
    readout:      residual capture at the readout layer + instrument application
    channels:     logit-based numeric reads (digit expectation, exit read, sentiment)
    conversation: cell conversation builder (scenario response + report battery)
    directions:   steering-direction extraction and storage
    runner:       panel runner (resume, validation, append-only JSONL)

Contract: schema.py at the repo root.
All CUDA use is behind functions; every module imports cleanly on a CPU-only Mac.
"""
