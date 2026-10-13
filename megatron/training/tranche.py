"""Finite-tranche boundaries, independent of model and distributed execution."""

from typing import Optional


def tranche_end_samples(
    total_samples: int,
    consumed_in_tranche: int,
    consumed_total: int,
    global_batch_size: int,
) -> int:
    """Return the absolute boundary reachable without an incomplete global batch.

    A tranche-local counter can restart while the checkpoint's global counter
    and learning-rate schedule remain continuous. Replays of a cached batch do
    not advance either counter.
    """
    if total_samples <= 0 or global_batch_size <= 0:
        raise ValueError("Tranche size and global batch size must be positive")
    if consumed_in_tranche < 0 or consumed_total < consumed_in_tranche:
        raise ValueError("Invalid tranche consumption counters")
    remaining = max(0, total_samples - consumed_in_tranche)
    return consumed_total + remaining // global_batch_size * global_batch_size


def tranche_finished(enabled: bool, boundary: Optional[int], consumed: int) -> bool:
    """Whether training must stop before fetching another batch."""
    return enabled and boundary is not None and consumed >= boundary
