"""Unified 3-phase fine-tune entry for the POMO TSP baseline.

Resumes from the baseline 3000-epoch checkpoint and runs three phases in a
fixed order (bias -> MSC -> leader-focused reward). The per-phase epoch
budget is computed from a single ``--total_finetune_epochs B`` knob using
the spec proportions 15% / 50% / 35%.

This script is **additive**: it does not modify ``train.py`` / ``test.py``
or any module's external interface. It only orchestrates calls into
``TSPTrainer.apply_phase_config`` + ``TSPTrainer.run_phase`` (added in a
small, backward-compatible patch to ``TSPTrainer``).

Quick-start (default 400-epoch recipe)::

    cd TSP/POMO
    python finetune_phased.py \
        --resume_checkpoint ./result/saved_tsp100_model2_longTrain/checkpoint-3000.pt

Ablations are exposed as boolean flags ``--enable_bias / --enable_msc /
--enable_leader``; everything else is a hyperparameter knob with a default
that matches the spec table.

Reproducibility:
* the resulting result/<ts>_<desc>/ directory contains:
    - log.txt                            (full per-epoch trace)
    - checkpoint-latest.pt               (latest model, overwritten each epoch)
    - checkpoint-phase_<P>_best.pt       (best train_score within each phase)
    - checkpoint-<E>.pt                  (periodic + per-phase final)
* ``finetune_phased.json`` (next to this file under configs/) records the
  default recipe in machine-readable form.

Validation hooks:
* in-loop validation is intentionally **not** wired here (kept simple); to
  evaluate avg_aug_gap on the official TSPLIB set, run ``test.py`` against
  the desired phase checkpoint, e.g.::
      python test.py --checkpoint_path .../checkpoint-phase_3_best.pt
"""
from __future__ import annotations

##########################################################################################
# Path / sys.path bootstrap (same convention as train.py / test.py)

import argparse
import json
import logging
import os
import sys
from copy import deepcopy
from datetime import datetime

import pytz
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")   # for TSProblemDef
sys.path.insert(0, "../..")  # for utils

from utils.utils import create_logger, copy_all_src
from TSProblemDef import get_default_msc_config
from TSPTrainer import TSPTrainer as Trainer


# Per-run training peak (observed: phase 3 with leader reward + bias) plus
# a generous safety margin. Used to cap --reserve_vram_gb so the holder can
# never starve the trainer.
_TRAIN_VRAM_BUDGET_GB = 8.0


##########################################################################################
# Defaults (mirror the spec table; CLI flags override).

DEFAULT_BASELINE_CKPT = "./result/saved_tsp100_model2_longTrain/checkpoint-3000.pt"

DEFAULT_PHASE_RECIPE = {
    "phase1": {
        "name": "1_bias",
        "frac": 0.15,
        "msc_ratios": {"uniform": 0.6, "clustered_uniform": 0.3, "gaussian_mixture": 0.1},
    },
    "phase2": {
        "name": "2_msc",
        "frac": 0.50,
        "msc_ratios": {"uniform": 0.4, "clustered_uniform": 0.4, "gaussian_mixture": 0.2},
    },
    "phase3": {
        "name": "3_leader",
        "frac": 0.35,
        "msc_ratios": {"uniform": 0.2, "clustered_uniform": 0.5, "gaussian_mixture": 0.3},
    },
}

DEFAULT_HPARAMS = {
    # ---- Global ----
    "total_finetune_epochs": 400,
    "validate_every": 0,           # 0 = disabled; positive = run a subprocess test.py every N epochs
    # ---- Optimizer ----
    "backbone_lr": 1e-5,
    "new_module_lr": 3e-5,         # informational only: bias module is non-learnable
    "phase3_lr": 5e-5,
    "phase3_final_lr": 5e-6,
    "weight_decay": 1e-6,
    "bias_param_group_lr_multiplier": 2.0,  # informational only
    # ---- Bias (M2) ----
    "enable_distance_bias": True,
    "knn_bias_enabled": True,
    "knn_k": 30,
    "knn_bias_value": 1.0,
    "distance_bias_scale": 1.0,
    "distance_norm_mode": "mean",
    "bias_warmup_epochs": None,    # default = full phase 1; integer overrides
    # ---- MSC ----
    "enable_msc": True,
    "msc_use_curriculum": True,
    # ---- Leader-focused reward (M3) ----
    "enable_leader": True,
    "leader_mode": "bonus_adv",
    "leader_alpha": 20.0,          # mapped to leader_gamma start value
    "leader_alpha_final": 40.0,    # ramp-target gamma
    "leader_aux_weight": 0.1,
    "leader_rampup_portion": 0.2,  # fraction of phase 3 to ramp gamma 0 -> alpha
    "leader_infinite_mode": False, # not implemented; flag preserved for sweeps
    # ---- Misc ----
    "train_episodes": 100_000,
    "train_batch_size": 64,
    "grad_clip_max_norm": 1.0,
    "model_save_interval": 100,
    "img_save_interval": 100,
}


##########################################################################################
# Helpers

def str2bool(value):
    if isinstance(value, bool):
        return value
    s = str(value).lower()
    if s in {"true", "1", "yes", "y"}: return True
    if s in {"false", "0", "no", "n"}: return False
    raise argparse.ArgumentTypeError(f"invalid bool: {value}")


def split_budget(B, phase_recipe=DEFAULT_PHASE_RECIPE):
    """Return (n1, n2, n3) summing to B, with exact spec fractions."""
    n1 = int(round(phase_recipe['phase1']['frac'] * B))
    n2 = int(round(phase_recipe['phase2']['frac'] * B))
    n3 = B - n1 - n2  # absorb rounding drift in the last phase
    if n3 < 0:
        # Pathological tiny budgets: reshuffle to keep all positive.
        n3 = max(0, B - n1)
        n2 = max(0, B - n1 - n3)
    return n1, n2, n3


def map_alpha_to_gamma(alpha):
    """Map the spec's leader_alpha to the leader_reward module's leader_gamma.

    The current ``train_ext.leader_reward`` exposes ``leader_gamma`` (bonus
    multiplier on the leader's advantage). The spec's "alpha" is a bonus
    *scale*. We use a 1:1 mapping with a /20 scale for sanity (alpha=20 ->
    gamma=1.0, alpha=40 -> gamma=2.0). This conversion is recorded in the
    log so sweeps stay reproducible.
    """
    return float(alpha) / 20.0


##########################################################################################
# CLI

def build_parser():
    p = argparse.ArgumentParser(
        description="Phased fine-tune (bias -> MSC -> leader) starting from a baseline checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Global
    p.add_argument("--config", default=None,
                   help="Optional JSON config; CLI flags override its values.")
    p.add_argument("--total_finetune_epochs", type=int, default=DEFAULT_HPARAMS['total_finetune_epochs'],
                   help="Total fine-tune budget B; phases get 15%%/50%%/35%% of B.")
    p.add_argument("--phase_epochs", type=str, default=None,
                   help="Override 15/50/35 split with explicit 'n1,n2,n3'. Use '0,0,140' "
                        "to resume only phase 3 from a phase-2 ckpt (skips bias warmup + MSC). "
                        "When set, --total_finetune_epochs is ignored for split logic.")
    p.add_argument("--phase3_snapshot_epochs", type=str, default=None,
                   help="Comma-separated in-phase epoch numbers (1-indexed) at which to "
                        "snapshot the current phase-3 best to a tagged file "
                        "checkpoint-phase_3_leader_best_at_{N}.pt. Lets one long run produce "
                        "multiple 'best of first N epochs' ckpts. Example: '420' for a 560 ep run.")
    p.add_argument("--resume_checkpoint", default=DEFAULT_BASELINE_CKPT,
                   help="Path to baseline checkpoint to resume from (weights only).")
    p.add_argument("--desc", default="phased_finetune",
                   help="Result directory tag.")
    p.add_argument("--use_cuda", type=str2bool, default=True)
    p.add_argument("--cuda_device_num", type=int, default=0)
    p.add_argument("--train_episodes", type=int, default=DEFAULT_HPARAMS['train_episodes'])
    p.add_argument("--train_batch_size", type=int, default=DEFAULT_HPARAMS['train_batch_size'])
    p.add_argument("--problem_size", type=int, default=100)
    p.add_argument("--pomo_size", type=int, default=100)
    p.add_argument("--grad_clip_max_norm", type=float, default=DEFAULT_HPARAMS['grad_clip_max_norm'])
    p.add_argument("--weight_decay", type=float, default=DEFAULT_HPARAMS['weight_decay'])
    p.add_argument("--continue_epoch_counter", type=str2bool, default=True,
                   help="If True, fine-tune epochs are numbered as ckpt_epoch+1.. (e.g. 3001..).")
    p.add_argument("--model_save_interval", type=int, default=DEFAULT_HPARAMS['model_save_interval'])
    p.add_argument("--img_save_interval", type=int, default=DEFAULT_HPARAMS['img_save_interval'])
    p.add_argument("--reserve_vram_gb", type=float, default=0.0,
                   help=(
                       "If > 0, allocate a placeholder tensor of this many GB on the "
                       "training GPU to prevent other users from squeezing onto the same "
                       "card on a shared cluster. Capped automatically so the trainer "
                       f"always retains at least {_TRAIN_VRAM_BUDGET_GB:.0f} GB. "
                       "Default 0 = no holder."
                   ))

    # Optimizer
    p.add_argument("--backbone_lr", type=float, default=DEFAULT_HPARAMS['backbone_lr'])
    p.add_argument("--new_module_lr", type=float, default=DEFAULT_HPARAMS['new_module_lr'])
    p.add_argument("--phase3_lr", type=float, default=DEFAULT_HPARAMS['phase3_lr'])
    p.add_argument("--phase3_final_lr", type=float, default=DEFAULT_HPARAMS['phase3_final_lr'])

    # Bias
    p.add_argument("--enable_bias", type=str2bool, default=DEFAULT_HPARAMS['enable_distance_bias'])
    p.add_argument("--knn_bias_enabled", type=str2bool, default=DEFAULT_HPARAMS['knn_bias_enabled'])
    p.add_argument("--knn_k", type=int, default=DEFAULT_HPARAMS['knn_k'])
    p.add_argument("--knn_bias_value", type=float, default=DEFAULT_HPARAMS['knn_bias_value'])
    p.add_argument("--distance_bias_scale", type=float, default=DEFAULT_HPARAMS['distance_bias_scale'])
    p.add_argument("--distance_norm_mode", choices=["none", "mean", "max", "std"],
                   default=DEFAULT_HPARAMS['distance_norm_mode'])
    p.add_argument("--bias_warmup_epochs", type=int, default=None,
                   help="Linear ramp of bias_strength over this many epochs at phase-1 start.")

    # MSC
    p.add_argument("--enable_msc", type=str2bool, default=DEFAULT_HPARAMS['enable_msc'])
    p.add_argument("--msc_use_curriculum", type=str2bool, default=DEFAULT_HPARAMS['msc_use_curriculum'])
    for ph in ('phase1', 'phase2', 'phase3'):
        for kind in ('uniform', 'clustered', 'gaussian'):
            arg = f"--{ph}_{kind}_ratio"
            p.add_argument(arg, type=float, default=None,
                           help=f"Override MSC '{kind}_uniform' ratio for {ph}.")

    # Leader
    p.add_argument("--enable_leader", type=str2bool, default=DEFAULT_HPARAMS['enable_leader'])
    p.add_argument("--leader_mode", choices=["bonus_adv", "aux_imitation"],
                   default=DEFAULT_HPARAMS['leader_mode'])
    p.add_argument("--leader_alpha", type=float, default=DEFAULT_HPARAMS['leader_alpha'])
    p.add_argument("--leader_alpha_final", type=float, default=DEFAULT_HPARAMS['leader_alpha_final'])
    p.add_argument("--leader_aux_weight", type=float, default=DEFAULT_HPARAMS['leader_aux_weight'])
    p.add_argument("--leader_rampup_portion", type=float, default=DEFAULT_HPARAMS['leader_rampup_portion'])
    p.add_argument("--leader_infinite_mode", type=str2bool, default=DEFAULT_HPARAMS['leader_infinite_mode'])

    # Ablation convenience flags (mutually-non-exclusive overrides)
    p.add_argument("--ablation", choices=[
        "all_three", "bias_only", "msc_only", "leader_only",
        "bias_msc", "msc_leader", "bias_leader", "control",
    ], default="all_three",
                   help="Quick ablation preset; sets enable_* flags accordingly. "
                        "'control' = all modules off (pure baseline+finetune).")

    # ---- N variation (phase4-like) ----
    p.add_argument("--n_values", default=None,
                   help="Comma-separated N values to sample per batch, e.g. "
                        "'100,150,200,250'. When set, env.problem_size/pomo_size "
                        "are mutated each batch.")
    p.add_argument("--n_weights", default=None,
                   help="Comma-separated weights matching --n_values (default: uniform).")

    return p


def apply_ablation_preset(args):
    """Mutate ``args.enable_*`` according to ``args.ablation``."""
    a = args.ablation
    if a == "all_three":
        return
    if a == "control":
        args.enable_bias = False
        args.enable_msc = False
        args.enable_leader = False
        return
    args.enable_bias = a in {"bias_only", "bias_msc", "bias_leader"}
    args.enable_msc = a in {"msc_only", "bias_msc", "msc_leader"}
    args.enable_leader = a in {"leader_only", "msc_leader", "bias_leader"}


def merge_config_file(args):
    if args.config is None:
        return
    if not os.path.exists(args.config):
        raise FileNotFoundError(args.config)
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    for k, v in cfg.items():
        if hasattr(args, k) and getattr(args, k) == build_parser().get_default(k):
            setattr(args, k, v)


##########################################################################################
# Phase-config builders

def build_phase1_cfg(args, phase_epochs):
    bias_cfg = {
        "distance_bias_enabled": bool(args.enable_bias),
        "distance_bias_scale": float(args.distance_bias_scale),
        "distance_bias_mode": "logit",
        "distance_norm_mode": str(args.distance_norm_mode),
        "knn_bias_enabled": bool(args.enable_bias and args.knn_bias_enabled),
        "knn_k": int(args.knn_k),
        "knn_bias_value": float(args.knn_bias_value),
    }
    msc_ratios = {
        "uniform": _resolve_ratio(args, 'phase1', 'uniform', DEFAULT_PHASE_RECIPE['phase1']['msc_ratios']),
        "clustered_uniform": _resolve_ratio(args, 'phase1', 'clustered',
                                            DEFAULT_PHASE_RECIPE['phase1']['msc_ratios']),
        "gaussian_mixture": _resolve_ratio(args, 'phase1', 'gaussian',
                                            DEFAULT_PHASE_RECIPE['phase1']['msc_ratios']),
    }
    return {
        "phase_name": "1_bias",
        "msc_enabled": bool(args.enable_msc),
        "msc_use_curriculum": False,  # within a phase use fixed ratios
        "msc_fixed_ratios": msc_ratios,
        "distance_bias_cfg": bias_cfg if args.enable_bias else {},
        "leader_cfg": {},  # disabled
        "optimizer_lr": float(args.backbone_lr),
    }


def build_phase2_cfg(args):
    bias_cfg = {
        "distance_bias_enabled": bool(args.enable_bias),
        "distance_bias_scale": float(args.distance_bias_scale),
        "distance_bias_mode": "logit",
        "distance_norm_mode": str(args.distance_norm_mode),
        "knn_bias_enabled": bool(args.enable_bias and args.knn_bias_enabled),
        "knn_k": int(args.knn_k),
        "knn_bias_value": float(args.knn_bias_value),
    }
    msc_ratios = {
        "uniform": _resolve_ratio(args, 'phase2', 'uniform',
                                  DEFAULT_PHASE_RECIPE['phase2']['msc_ratios']),
        "clustered_uniform": _resolve_ratio(args, 'phase2', 'clustered',
                                            DEFAULT_PHASE_RECIPE['phase2']['msc_ratios']),
        "gaussian_mixture": _resolve_ratio(args, 'phase2', 'gaussian',
                                            DEFAULT_PHASE_RECIPE['phase2']['msc_ratios']),
    }
    return {
        "phase_name": "2_msc",
        "msc_enabled": bool(args.enable_msc),
        "msc_use_curriculum": False,
        "msc_fixed_ratios": msc_ratios,
        "distance_bias_cfg": bias_cfg if args.enable_bias else {},
        "leader_cfg": {},
        "optimizer_lr": float(args.backbone_lr),
    }


def build_phase3_cfg(args):
    bias_cfg = {
        "distance_bias_enabled": bool(args.enable_bias),
        "distance_bias_scale": float(args.distance_bias_scale),
        "distance_bias_mode": "logit",
        "distance_norm_mode": str(args.distance_norm_mode),
        "knn_bias_enabled": bool(args.enable_bias and args.knn_bias_enabled),
        "knn_k": int(args.knn_k),
        "knn_bias_value": float(args.knn_bias_value),
    }
    msc_ratios = {
        "uniform": _resolve_ratio(args, 'phase3', 'uniform',
                                  DEFAULT_PHASE_RECIPE['phase3']['msc_ratios']),
        "clustered_uniform": _resolve_ratio(args, 'phase3', 'clustered',
                                            DEFAULT_PHASE_RECIPE['phase3']['msc_ratios']),
        "gaussian_mixture": _resolve_ratio(args, 'phase3', 'gaussian',
                                            DEFAULT_PHASE_RECIPE['phase3']['msc_ratios']),
    }
    leader_cfg = {
        "leader_reward_enabled": bool(args.enable_leader),
        "leader_mode": str(args.leader_mode),
        "leader_gamma": map_alpha_to_gamma(args.leader_alpha),  # initial; ramp callback updates it
        "leader_aux_weight": float(args.leader_aux_weight),
    }
    # The "leader_stage_lr_drop" rule from the spec is coupled with leader
    # being enabled. For ablations that disable leader, keep the previous
    # phases' lr so we don't confound an ablation with an LR change.
    phase3_start_lr = float(args.phase3_lr) if args.enable_leader else float(args.backbone_lr)
    return {
        "phase_name": "3_leader",
        "msc_enabled": bool(args.enable_msc),
        "msc_use_curriculum": False,
        "msc_fixed_ratios": msc_ratios,
        "distance_bias_cfg": bias_cfg if args.enable_bias else {},
        "leader_cfg": leader_cfg if args.enable_leader else {},
        "optimizer_lr": phase3_start_lr,
    }


def _resolve_ratio(args, phase, kind, defaults):
    """``args.phase{1,2,3}_<kind>_ratio`` overrides the phase default."""
    attr = f"{phase}_{kind}_ratio"
    val = getattr(args, attr, None)
    if val is not None:
        return float(val)
    if kind == 'uniform':
        return float(defaults['uniform'])
    if kind == 'clustered':
        return float(defaults['clustered_uniform'])
    if kind == 'gaussian':
        return float(defaults['gaussian_mixture'])
    raise ValueError(kind)


##########################################################################################
# Per-epoch callbacks (bias warmup, leader gamma ramp, phase 3 lr decay)

def make_phase1_callback(args, phase_epochs):
    target_scale = float(args.distance_bias_scale)
    target_knn_v = float(args.knn_bias_value)
    warmup = args.bias_warmup_epochs
    if warmup is None or warmup <= 0:
        warmup = phase_epochs  # default: full phase 1
    warmup = min(int(warmup), int(phase_epochs))

    def cb(trainer, in_phase, abs_epoch):
        if not args.enable_bias or trainer.model.distance_bias_module is None:
            return
        # Linear warmup: 0 -> target over `warmup` epochs.
        frac = min(1.0, in_phase / max(1, warmup))
        mod = trainer.model.distance_bias_module
        mod.cfg["distance_bias_scale"] = frac * target_scale
        mod.cfg["knn_bias_value"] = frac * target_knn_v
        if in_phase == 1 or in_phase == warmup:
            trainer.logger.info(
                "[PHASE 1_bias] bias warmup epoch %d/%d -> scale=%.4f knn_v=%.4f",
                in_phase, warmup, mod.cfg["distance_bias_scale"], mod.cfg["knn_bias_value"],
            )
    return cb


def make_phase3_callback(args, phase_epochs):
    rampup_epochs = max(1, int(round(phase_epochs * float(args.leader_rampup_portion))))
    gamma_start = 0.0
    gamma_target = map_alpha_to_gamma(args.leader_alpha)
    gamma_final = map_alpha_to_gamma(args.leader_alpha_final) if args.leader_alpha_final else gamma_target

    # LR decay only applies when leader is on (otherwise we keep backbone_lr
    # to keep ablations apples-to-apples).
    lr_start = float(args.phase3_lr) if args.enable_leader else float(args.backbone_lr)
    lr_end = float(args.phase3_final_lr) if args.enable_leader else float(args.backbone_lr)

    def cb(trainer, in_phase, abs_epoch):
        # ---- LR linear decay across phase 3 (no-op when start==end) ----
        if phase_epochs > 1:
            t = (in_phase - 1) / (phase_epochs - 1)
        else:
            t = 1.0
        cur_lr = lr_start * (1 - t) + lr_end * t
        for g in trainer.optimizer.param_groups:
            g['lr'] = cur_lr

        # ---- Leader gamma ramp: 0 -> gamma_target during the rampup window,
        #      then linear interpolation to gamma_final by end of phase. ----
        if not args.enable_leader or trainer.leader_cfg is None:
            if in_phase == 1 or in_phase == phase_epochs:
                trainer.logger.info(
                    "[PHASE 3_leader] epoch %d/%d (leader OFF) lr=%.6e",
                    in_phase, phase_epochs, cur_lr,
                )
            return
        if in_phase <= rampup_epochs:
            ramp = in_phase / rampup_epochs
            cur_gamma = gamma_start + (gamma_target - gamma_start) * ramp
        else:
            tail_frac = (in_phase - rampup_epochs) / max(1, (phase_epochs - rampup_epochs))
            tail_frac = min(1.0, max(0.0, tail_frac))
            cur_gamma = gamma_target + (gamma_final - gamma_target) * tail_frac
        trainer.leader_cfg['leader_gamma'] = float(cur_gamma)
        if in_phase == 1 or in_phase == rampup_epochs or in_phase == phase_epochs:
            trainer.logger.info(
                "[PHASE 3_leader] epoch %d/%d ramp=%d gamma=%.4f lr=%.6e",
                in_phase, phase_epochs, rampup_epochs, cur_gamma, cur_lr,
            )
    return cb


##########################################################################################
# Main

def build_logger_params(args):
    """Mirror train.py's logger conventions; only the desc tag changes."""
    process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))
    return {
        "log_file": {
            "desc": f"phased_{args.desc}_B{args.total_finetune_epochs}_{args.ablation}",
            "filename": "log.txt",
            "filepath": "./result/" + process_start_time.strftime("%Y%m%d_%H%M%S") + "{desc}",
        }
    }


def build_trainer(args):
    env_params = {"problem_size": args.problem_size, "pomo_size": args.pomo_size}
    model_params = {
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128 ** (1/2),
        'encoder_layer_num': 6,
        'qkv_dim': 16,
        'head_num': 8,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'eval_type': 'argmax',
    }
    optimizer_params = {
        'optimizer': {
            'lr': float(args.backbone_lr),
            'weight_decay': float(args.weight_decay),
        },
        # MultiStepLR with milestone past total epochs => no decay; keeps API.
        'scheduler': {
            'milestones': [10**9],
            'gamma': 0.1,
        },
    }
    trainer_params = {
        'use_cuda': bool(args.use_cuda),
        'cuda_device_num': int(args.cuda_device_num),
        # Upper bound; run_phase advances it as needed.
        'epochs': args.total_finetune_epochs,
        'train_episodes': int(args.train_episodes),
        'train_batch_size': int(args.train_batch_size),
        'logging': {
            'model_save_interval': int(args.model_save_interval),
            'img_save_interval': int(args.img_save_interval),
            'log_image_params_1': {
                'json_foldername': 'log_image_style',
                'filename': 'style_tsp_100.json',
            },
            'log_image_params_2': {
                'json_foldername': 'log_image_style',
                'filename': 'style_loss_1.json',
            },
        },
        'model_load': {'enable': False},
        'finetune_from': os.path.abspath(args.resume_checkpoint),
        'finetune_continue_epoch_counter': bool(args.continue_epoch_counter),
        'grad_clip_max_norm': float(args.grad_clip_max_norm),
        'msc_cfg': get_default_msc_config(),  # will be overridden per-phase
    }

    # ---- N variation: per-batch random N from --n_values ----
    if args.n_values:
        n_vals = [int(x.strip()) for x in args.n_values.split(',') if x.strip()]
        n_wts = None
        if args.n_weights:
            n_wts = [float(x.strip()) for x in args.n_weights.split(',') if x.strip()]
            if len(n_wts) != len(n_vals):
                raise ValueError(
                    f"--n_weights count ({len(n_wts)}) must match --n_values count ({len(n_vals)})"
                )
        trainer_params['n_sampler_cfg'] = {
            'enabled': True,
            'n_values': n_vals,
            'weights': n_wts,
        }

    trainer = Trainer(
        env_params=env_params,
        model_params=model_params,
        optimizer_params=optimizer_params,
        trainer_params=trainer_params,
    )
    return trainer


def reserve_vram(reserve_gb, log):
    """Allocate a placeholder tensor on the current CUDA device (defensive
    "占座" for shared GPUs).

    Returns the holder tensor — caller MUST keep a reference until the run
    is done, otherwise PyTorch may release the memory back to the driver.

    Caps the request to ``free_vram - _TRAIN_VRAM_BUDGET_GB`` so the trainer
    can never be starved by an over-aggressive holder.
    """
    if reserve_gb is None or reserve_gb <= 0:
        return None
    if not torch.cuda.is_available():
        log.warning("[reserve_vram] CUDA unavailable; --reserve_vram_gb ignored.")
        return None

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)
    cap_gb = max(0.0, free_gb - _TRAIN_VRAM_BUDGET_GB)
    actual_gb = min(float(reserve_gb), cap_gb)

    if actual_gb <= 0.0:
        log.warning(
            "[reserve_vram] free=%.1f GB too small to reserve %.1f GB while keeping "
            "a %.1f GB training budget; skipping holder.",
            free_gb, reserve_gb, _TRAIN_VRAM_BUDGET_GB,
        )
        return None

    n_floats = int(actual_gb * (1024 ** 3) // 4)
    holder = torch.empty(n_floats, dtype=torch.float32)
    log.info(
        "[reserve_vram] holder=%.1f GB | free_before=%.1f GB total=%.1f GB device=cuda:%s",
        actual_gb, free_gb, total_gb, torch.cuda.current_device(),
    )
    if actual_gb < float(reserve_gb):
        log.warning(
            "[reserve_vram] requested %.1f GB but capped to %.1f GB to leave "
            "%.1f GB safety margin for the trainer.",
            reserve_gb, actual_gb, _TRAIN_VRAM_BUDGET_GB,
        )
    return holder


def main():
    args = build_parser().parse_args()
    merge_config_file(args)
    apply_ablation_preset(args)

    if args.total_finetune_epochs <= 0:
        raise ValueError("--total_finetune_epochs must be positive.")

    if args.phase_epochs is not None:
        parts = [p.strip() for p in args.phase_epochs.split(',')]
        if len(parts) != 3:
            raise ValueError("--phase_epochs must be 'n1,n2,n3' (3 ints)")
        n1, n2, n3 = (int(parts[0]), int(parts[1]), int(parts[2]))
        if n1 < 0 or n2 < 0 or n3 < 0:
            raise ValueError("--phase_epochs values must be non-negative")
        if (n1 + n2 + n3) == 0:
            raise ValueError("--phase_epochs must have at least one phase > 0")
    else:
        n1, n2, n3 = split_budget(args.total_finetune_epochs)

    logger_params = build_logger_params(args)
    create_logger(**logger_params)
    log = logging.getLogger('root')

    log.info("==[FINETUNE-PHASED]== ablation=%s B=%d -> phases=(%d,%d,%d)",
             args.ablation, args.total_finetune_epochs, n1, n2, n3)
    log.info("Resume checkpoint: %s", os.path.abspath(args.resume_checkpoint))
    log.info("HParams: bias=%s knn_k=%d knn_v=%.3f scale=%.3f norm=%s",
             args.enable_bias, args.knn_k, args.knn_bias_value,
             args.distance_bias_scale, args.distance_norm_mode)
    log.info("HParams: msc=%s leader=%s leader_mode=%s alpha=%.2f->%.2f rampup=%.2f",
             args.enable_msc, args.enable_leader, args.leader_mode,
             args.leader_alpha, args.leader_alpha_final, args.leader_rampup_portion)
    log.info("HParams: backbone_lr=%.2e new_module_lr=%.2e (info-only) phase3_lr=%.2e -> %.2e",
             args.backbone_lr, args.new_module_lr, args.phase3_lr, args.phase3_final_lr)

    trainer = build_trainer(args)
    copy_all_src(trainer.result_folder)

    # Optional VRAM "占座" — kept alive in main()'s local scope so it lives
    # until training completes (do NOT del or rebind this variable).
    _vram_holder = reserve_vram(args.reserve_vram_gb, log)  # noqa: F841

    # Persist the resolved config alongside the run for easy diffing.
    config_dump_path = os.path.join(trainer.result_folder, 'finetune_phased_config.json')
    with open(config_dump_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2, default=str)

    summaries = []

    # -------------------- Phase 1: bias adapter --------------------
    if n1 > 0:
        cfg1 = build_phase1_cfg(args, n1)
        trainer.apply_phase_config(cfg1)
        cb1 = make_phase1_callback(args, n1) if args.enable_bias else None
        summaries.append(trainer.run_phase(n1, phase_name='1_bias', epoch_callback=cb1))

    # -------------------- Phase 2: MSC main adaptation -------------
    if n2 > 0:
        cfg2 = build_phase2_cfg(args)
        trainer.apply_phase_config(cfg2)
        # No special callback; bias scale already at target after phase 1.
        summaries.append(trainer.run_phase(n2, phase_name='2_msc'))

    # -------------------- Phase 3: leader-focused reward -----------
    if n3 > 0:
        cfg3 = build_phase3_cfg(args)
        trainer.apply_phase_config(cfg3)
        # Always pass the phase-3 callback when leader is on (gamma ramp +
        # lr decay). When leader is off, we still want a no-op callback for
        # the visibility log line at start/end of phase 3.
        cb3 = make_phase3_callback(args, n3)
        snapshot_eps = []
        if args.phase3_snapshot_epochs:
            snapshot_eps = [int(x.strip()) for x in args.phase3_snapshot_epochs.split(',')]
            log.info("Phase 3 will snapshot phase-best at in-phase epochs: %s", snapshot_eps)
        summaries.append(trainer.run_phase(
            n3, phase_name='3_leader', epoch_callback=cb3,
            snapshot_at_phase_epochs=snapshot_eps,
        ))

    # -------------------- Final summary ---------------------------
    log.info("==[FINETUNE-PHASED DONE]==")
    for s in summaries:
        log.info("  %s: epochs=%s phase_best_score=%.4f -> %s",
                 s['phase_name'], s['epoch_range'], s['phase_best_score'],
                 s['phase_best_path'])
    log.info("Result folder: %s", trainer.result_folder)
    log.info("To evaluate avg_aug_gap, run e.g.:")
    log.info("    python test.py --checkpoint_path %s/checkpoint-phase_3_leader_best.pt",
             trainer.result_folder)


if __name__ == "__main__":
    main()
