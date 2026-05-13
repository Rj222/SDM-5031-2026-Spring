"""TSP training-instance generation.

This module exposes the legacy uniform sampler used by the POMO baseline plus
an opt-in **Mixed Structured Curriculum (MSC)** that mixes three generators
during training:

* ``uniform``           - i.i.d. uniform points in [0,1]^2 (legacy baseline).
* ``clustered_uniform`` - per-instance random clusters with local Gaussian
                          jitter and a small uniform background fraction.
* ``gaussian_mixture``  - per-instance random Gaussian mixture with random
                          weights and per-component std.

A 3-stage curriculum schedules the per-batch mixing ratio across training
progress.  All public behaviour is **backward compatible**:

* ``get_random_problems(batch_size, problem_size)`` still works (legacy 2-arg
  call) and returns the legacy uniform sample bit-for-bit.
* When ``config is None`` or ``config['enabled'] is False`` the module also
  falls back to the legacy uniform sampler.

The output is always a ``torch.FloatTensor`` of shape
``(batch_size, problem_size, 2)`` clipped to ``[0, 1]``.
"""

from __future__ import annotations

import copy
import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch


logger = logging.getLogger("trainer.msc")


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

#: Names of the three supported generators.
GENERATOR_NAMES = ("uniform", "clustered_uniform", "gaussian_mixture")


#: Default MSC configuration. ``enabled=False`` reverts to the legacy uniform
#: baseline.  All knobs are intentionally exposed so ablations can override
#: them via :func:`copy.deepcopy` + dictionary edits or CLI plumbing.
DEFAULT_MSC_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "use_curriculum": True,
    "curriculum": {
        # Stage transition points expressed as fractions of total training.
        "stage_boundaries": [0.3, 0.7],
        # Per-stage mixing ratio.  Must contain the keys in GENERATOR_NAMES.
        "stage_ratios": [
            {"uniform": 0.7, "clustered_uniform": 0.3, "gaussian_mixture": 0.0},
            {"uniform": 0.4, "clustered_uniform": 0.4, "gaussian_mixture": 0.2},
            {"uniform": 0.2, "clustered_uniform": 0.5, "gaussian_mixture": 0.3},
        ],
        # Used when ``use_curriculum=False`` or when epoch/total_epochs is None.
        "fixed_ratios": {"uniform": 0.5, "clustered_uniform": 0.3, "gaussian_mixture": 0.2},
    },
    "clustered_uniform": {
        "num_clusters_min": 3,
        "num_clusters_max": 6,
        "cluster_std_min": 0.03,
        "cluster_std_max": 0.10,
        "background_uniform_ratio": 0.1,
    },
    "gaussian_mixture": {
        "num_components_min": 3,
        "num_components_max": 6,
        "component_std_min": 0.03,
        "component_std_max": 0.12,
    },
    "postprocess": {
        "clip_to_unit_square": True,
        "jitter_eps": 1e-4,
    },
    # Emit a debug log line every N batches; 0 disables.
    "log_every": 200,
}


def get_default_msc_config() -> Dict[str, Any]:
    """Return a fresh deep copy of :data:`DEFAULT_MSC_CONFIG`."""
    return copy.deepcopy(DEFAULT_MSC_CONFIG)


# ---------------------------------------------------------------------------
# Instance generators
# ---------------------------------------------------------------------------

def generate_uniform(
    batch_size: int,
    problem_size: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Legacy POMO sampler: i.i.d. uniform points in [0,1]^2.

    Behaviour is identical to the original baseline ``torch.rand`` call.
    """
    return torch.rand(size=(batch_size, problem_size, 2), device=device)


def generate_clustered_uniform(
    batch_size: int,
    problem_size: int,
    config: Optional[Mapping[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Per-instance clustered point cloud + small uniform background.

    For each instance:

    1. Sample ``K ~ U[num_clusters_min, num_clusters_max]`` cluster centers in
       [0,1]^2 with per-cluster std ``~ U[cluster_std_min, cluster_std_max]``.
    2. ``main_count = problem_size * (1 - background_uniform_ratio)`` points
       are assigned to a random cluster and locally jittered with that
       cluster's std.
    3. The remaining ``bg_count`` points are sampled uniformly to provide a
       small background.
    4. Points are shuffled along the node axis (cosmetic; node order is
       irrelevant to TSP but easier to inspect).

    Output is **not** clipped here; clipping/jitter is centralised in
    :func:`_postprocess`.
    """
    cfg = config or {}
    K_min = max(1, int(cfg.get("num_clusters_min", 3)))
    K_max = max(K_min, int(cfg.get("num_clusters_max", 6)))
    s_min = float(cfg.get("cluster_std_min", 0.03))
    s_max = float(cfg.get("cluster_std_max", 0.10))
    s_min, s_max = min(s_min, s_max), max(s_min, s_max)
    bg_ratio = float(cfg.get("background_uniform_ratio", 0.1))
    bg_ratio = max(0.0, min(1.0, bg_ratio))

    bg_count = int(round(bg_ratio * problem_size))
    bg_count = min(max(0, bg_count), problem_size)
    main_count = problem_size - bg_count

    # Cluster parameters (B, K_max, ...).
    centers = torch.rand(batch_size, K_max, 2, device=device)
    stds = torch.empty(batch_size, K_max, device=device).uniform_(s_min, s_max)

    # Active cluster count per instance.  We always allocate K_max slots and
    # restrict assignments via a modulo so the tensor shapes stay static.
    if K_min == K_max:
        K_per_inst = torch.full((batch_size,), K_max, dtype=torch.long, device=device)
    else:
        K_per_inst = torch.randint(K_min, K_max + 1, size=(batch_size,), device=device)
    K_per_inst_b = K_per_inst.unsqueeze(1)  # (B, 1)

    # Clustered points.
    if main_count > 0:
        raw_assign = torch.randint(0, K_max, size=(batch_size, main_count), device=device)
        assign = raw_assign % K_per_inst_b
        idx = assign.unsqueeze(-1).expand(-1, -1, 2)
        chosen_centers = torch.gather(centers, 1, idx)
        chosen_stds = torch.gather(stds, 1, assign).unsqueeze(-1)
        noise = torch.randn(batch_size, main_count, 2, device=device)
        main_pts = chosen_centers + chosen_stds * noise
    else:
        main_pts = torch.empty(batch_size, 0, 2, device=device)

    # Background uniform points.
    if bg_count > 0:
        bg_pts = torch.rand(batch_size, bg_count, 2, device=device)
        problems = torch.cat([main_pts, bg_pts], dim=1)
    else:
        problems = main_pts

    # Shuffle node order so background and clusters are interleaved.
    perm = torch.argsort(torch.rand(batch_size, problem_size, device=device), dim=1)
    problems = torch.gather(problems, 1, perm.unsqueeze(-1).expand(-1, -1, 2))

    return problems


def generate_gaussian_mixture(
    batch_size: int,
    problem_size: int,
    config: Optional[Mapping[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Per-instance Gaussian mixture sampler in [0,1]^2.

    For each instance:

    1. Sample ``K ~ U[num_components_min, num_components_max]`` mixture
       components with random centers in [0,1]^2 and per-component std
       ``~ U[component_std_min, component_std_max]``.
    2. Draw mixture weights from an exponential distribution and renormalise
       (Dirichlet-like).  Inactive components beyond ``K`` get weight 0.
    3. Each of the ``problem_size`` points picks a component proportional to
       its weight (vectorised via ``cumsum`` + ``searchsorted``) and is
       sampled from that Gaussian.
    """
    cfg = config or {}
    K_min = max(1, int(cfg.get("num_components_min", 3)))
    K_max = max(K_min, int(cfg.get("num_components_max", 6)))
    s_min = float(cfg.get("component_std_min", 0.03))
    s_max = float(cfg.get("component_std_max", 0.12))
    s_min, s_max = min(s_min, s_max), max(s_min, s_max)

    centers = torch.rand(batch_size, K_max, 2, device=device)
    stds = torch.empty(batch_size, K_max, device=device).uniform_(s_min, s_max)

    if K_min == K_max:
        K_per_inst = torch.full((batch_size,), K_max, dtype=torch.long, device=device)
    else:
        K_per_inst = torch.randint(K_min, K_max + 1, size=(batch_size,), device=device)
    K_per_inst_b = K_per_inst.unsqueeze(1)  # (B, 1)

    # Random weights, masked to active components.
    raw_w = torch.empty(batch_size, K_max, device=device).exponential_()
    arange = torch.arange(K_max, device=device).unsqueeze(0)  # (1, K_max)
    active_mask = (arange < K_per_inst_b).to(raw_w.dtype)
    weights = raw_w * active_mask
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    cum_w = weights.cumsum(dim=1)
    # Last column should be 1.0 ideally; nudge to ensure searchsorted is safe.
    cum_w[:, -1] = 1.0

    u = torch.rand(batch_size, problem_size, device=device)
    # Row-wise binary search; output shape (B, P) with values in [0, K_max-1].
    assign = torch.searchsorted(cum_w, u).clamp(max=K_max - 1)

    idx = assign.unsqueeze(-1).expand(-1, -1, 2)
    chosen_centers = torch.gather(centers, 1, idx)
    chosen_stds = torch.gather(stds, 1, assign).unsqueeze(-1)
    noise = torch.randn(batch_size, problem_size, 2, device=device)
    return chosen_centers + chosen_stds * noise


# ---------------------------------------------------------------------------
# Curriculum scheduling
# ---------------------------------------------------------------------------

def get_curriculum_stage(
    epoch: Optional[int],
    total_epochs: Optional[int],
    stage_boundaries,
) -> Optional[int]:
    """Return the 0-based stage index for ``epoch`` given ``total_epochs``.

    Uses ``progress = (epoch - 1) / total_epochs`` so that ``epoch=1`` lands
    in stage 0 and ``epoch=total_epochs`` lands in the final stage.  Returns
    ``None`` when either argument is missing (caller should fall back to
    fixed ratios).
    """
    if epoch is None or total_epochs is None or total_epochs <= 0:
        return None
    progress = (float(epoch) - 1.0) / float(total_epochs)
    progress = max(0.0, min(1.0, progress))
    for i, b in enumerate(stage_boundaries):
        if progress < float(b):
            return i
    return len(stage_boundaries)


def get_stage_ratios(stage_idx: Optional[int], curriculum_cfg: Mapping[str, Any]) -> Dict[str, float]:
    """Return the ``{generator_name: ratio}`` mapping for the given stage.

    Falls back to ``curriculum_cfg['fixed_ratios']`` when ``stage_idx`` is
    ``None`` (no curriculum) or when ``stage_ratios`` is missing/empty.
    """
    stage_ratios = curriculum_cfg.get("stage_ratios") or []
    if stage_idx is None or not stage_ratios:
        ratios = curriculum_cfg.get("fixed_ratios") or {"uniform": 1.0}
    else:
        idx = max(0, min(int(stage_idx), len(stage_ratios) - 1))
        ratios = stage_ratios[idx]
    # Normalise: ensure all generator names are present (default 0.0).
    return {name: float(ratios.get(name, 0.0)) for name in GENERATOR_NAMES}


def sample_distribution_types(batch_size: int, ratios: Mapping[str, float]) -> Dict[str, int]:
    """Convert per-generator ratios into integer counts that sum to batch_size.

    Robust to:

    * negative / zero ratios (clamped),
    * all-zero ratios (degenerate -> uniform),
    * rounding errors (residual is absorbed by the largest bucket).
    """
    keys = list(ratios.keys()) or list(GENERATOR_NAMES)
    raw = [max(0.0, float(ratios.get(k, 0.0))) for k in keys]
    total = sum(raw)
    if total <= 0.0:
        # Degenerate config: send everything to uniform so the batch is still valid.
        counts = {k: 0 for k in keys}
        counts["uniform"] = batch_size
        return counts

    weights = [w / total for w in raw]
    counts = [int(round(w * batch_size)) for w in weights]

    # Reconcile any rounding drift on the largest-weight buckets.
    diff = batch_size - sum(counts)
    if diff != 0:
        order = sorted(range(len(weights)), key=lambda i: -weights[i])
        i = 0
        while diff != 0 and i < len(order):
            j = order[i]
            step = 1 if diff > 0 else -1
            new_val = counts[j] + step
            if new_val >= 0:
                counts[j] = new_val
                diff = batch_size - sum(counts)
            i += 1
        # Final safety: dump any remainder into the dominant bucket.
        if diff != 0:
            j = order[0]
            counts[j] = max(0, counts[j] + diff)

    return dict(zip(keys, counts))


# ---------------------------------------------------------------------------
# Postprocessing & dispatch
# ---------------------------------------------------------------------------

def _postprocess(problems: torch.Tensor, postprocess_cfg: Optional[Mapping[str, Any]]) -> torch.Tensor:
    """Apply optional jitter, [0,1] clipping, and a NaN/Inf safety net."""
    cfg = postprocess_cfg or {}
    eps = float(cfg.get("jitter_eps", 0.0))
    if eps > 0:
        # Symmetric uniform jitter in [-eps, +eps] avoids exact duplicates.
        problems = problems + eps * (torch.rand_like(problems) * 2.0 - 1.0)
    if bool(cfg.get("clip_to_unit_square", True)):
        problems = problems.clamp(0.0, 1.0)
    # Replace any remaining NaN/Inf with fresh uniform samples to keep training stable.
    if not torch.isfinite(problems).all():
        bad = ~torch.isfinite(problems)
        problems = torch.where(bad, torch.rand_like(problems), problems)
    return problems


# Module-level batch counter used only for periodic logging.
_BATCH_COUNTER = 0


def _build_problems_with_counts(
    problem_size: int,
    counts: Mapping[str, int],
    config: Mapping[str, Any],
    device: Optional[torch.device],
) -> torch.Tensor:
    """Concatenate per-generator chunks honouring the requested counts."""
    cu_cfg = config.get("clustered_uniform", {})
    gm_cfg = config.get("gaussian_mixture", {})

    chunks = []
    n_uniform = int(counts.get("uniform", 0))
    n_clu = int(counts.get("clustered_uniform", 0))
    n_gm = int(counts.get("gaussian_mixture", 0))

    if n_uniform > 0:
        chunks.append(generate_uniform(n_uniform, problem_size, device=device))
    if n_clu > 0:
        chunks.append(generate_clustered_uniform(n_clu, problem_size, cu_cfg, device=device))
    if n_gm > 0:
        chunks.append(generate_gaussian_mixture(n_gm, problem_size, gm_cfg, device=device))

    if not chunks:
        # Should not happen in normal flow, but keeps the contract.
        return generate_uniform(0, problem_size, device=device)
    return torch.cat(chunks, dim=0)


def get_random_problems(
    batch_size: int,
    problem_size: int,
    epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
    config: Optional[Mapping[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate a batch of TSP instances.

    Backward-compatible drop-in for the legacy POMO baseline:

    * Calling ``get_random_problems(batch_size, problem_size)`` (no other
      kwargs) returns the legacy uniform tensor unchanged.
    * Passing ``config=DEFAULT_MSC_CONFIG`` (or a customised dict) enables
      the Mixed Structured Curriculum.
    * ``epoch`` / ``total_epochs`` drive the 3-stage schedule when
      ``config['use_curriculum']`` is true; otherwise ``fixed_ratios`` are
      used.

    Returns a ``torch.FloatTensor`` of shape ``(batch_size, problem_size, 2)``
    with values in ``[0, 1]``.
    """
    # Legacy path / kill switch.
    if config is None or not config.get("enabled", False):
        return generate_uniform(batch_size, problem_size, device=device)

    curriculum_cfg = config.get("curriculum", {}) or {}
    use_curriculum = bool(config.get("use_curriculum", True))

    stage_idx = None
    if use_curriculum:
        stage_idx = get_curriculum_stage(
            epoch, total_epochs, curriculum_cfg.get("stage_boundaries", [])
        )

    ratios = get_stage_ratios(stage_idx, curriculum_cfg)
    counts = sample_distribution_types(batch_size, ratios)

    problems = _build_problems_with_counts(problem_size, counts, config, device=device)

    # Shuffle the batch so adjacent samples don't all share one generator.
    if problems.size(0) > 1:
        perm = torch.randperm(problems.size(0), device=problems.device)
        problems = problems.index_select(0, perm)

    problems = _postprocess(problems, config.get("postprocess"))

    # Lightweight, periodic logging.  Honours an existing ``trainer`` logger
    # if one is configured; otherwise a NullHandler keeps things silent.
    global _BATCH_COUNTER
    _BATCH_COUNTER += 1
    log_every = int(config.get("log_every", 0) or 0)
    if log_every > 0 and (_BATCH_COUNTER == 1 or _BATCH_COUNTER % log_every == 0):
        ratio_str = ", ".join(f"{k}={ratios.get(k, 0.0):.2f}" for k in GENERATOR_NAMES)
        count_str = ", ".join(f"{k}={counts.get(k, 0)}" for k in GENERATOR_NAMES)
        logger.info(
            "[MSC] batch_no=%d epoch=%s/%s stage=%s ratios={%s} counts={%s} bs=%d enabled=%s",
            _BATCH_COUNTER,
            str(epoch),
            str(total_epochs),
            str(stage_idx),
            ratio_str,
            count_str,
            batch_size,
            bool(config.get("enabled", False)),
        )

    return problems


# ---------------------------------------------------------------------------
# Augmentation helper (unchanged; kept here for import compatibility)
# ---------------------------------------------------------------------------

def augment_xy_data_by_8_fold(problems: torch.Tensor) -> torch.Tensor:
    # problems.shape: (batch, problem, 2)

    x = problems[:, :, [0]]
    y = problems[:, :, [1]]
    # x,y shape: (batch, problem, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, problem, 2)

    return aug_problems

