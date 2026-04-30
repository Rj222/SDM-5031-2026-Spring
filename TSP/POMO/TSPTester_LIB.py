import os
import time
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model

from TSProblemDef import augment_xy_data_by_8_fold

from tsplib_utils import TSPLIBReader, tsplib_cost

# Optional extensions (M1 SGBS-lite, M2 distance bias, M4 2-opt)
from search.sgbs_lite import rerank_sgbs_lite, pool_and_select_best
from search.two_opt import two_opt_refine
try:
    from model_ext.distance_bias import DistanceBiasModule  # M2
except Exception:  # pragma: no cover - module may be stubbed during bootstrap
    DistanceBiasModule = None  # type: ignore


def _normalize_to_unit_square(node_xy: torch.Tensor) -> torch.Tensor:
    """Normalize to [0,1] with uniform scaling (same style as ICAM script)."""
    xy_max = torch.max(node_xy, dim=1, keepdim=True).values
    xy_min = torch.min(node_xy, dim=1, keepdim=True).values
    ratio = torch.max((xy_max - xy_min), dim=-1, keepdim=True).values
    ratio[ratio == 0] = 1
    return (node_xy - xy_min) / ratio.expand(-1, 1, 2)


@dataclass
class LibResult:
    instances: List[str]
    optimal: List[Optional[float]]
    problem_size: List[int]
    no_aug_score: List[float]
    aug_score: List[float]
    no_aug_gap: List[Optional[float]]
    aug_gap: List[Optional[float]]
    total_instance_num: int = 0
    solved_instance_num: int = 0
    per_instance: List[Dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def _mean_valid(values: List[Optional[float]]) -> Optional[float]:
        valid_values = [value for value in values if value is not None]
        if not valid_values:
            return None
        return float(np.mean(valid_values))

    @property
    def avg_no_aug_gap(self) -> Optional[float]:
        return self._mean_valid(self.no_aug_gap)

    @property
    def avg_aug_gap(self) -> Optional[float]:
        return self._mean_valid(self.aug_gap)

    def to_dict(self) -> Dict[str, object]:
        return {
            "instances": self.instances,
            "optimal": self.optimal,
            "problem_size": self.problem_size,
            "no_aug_score": self.no_aug_score,
            "aug_score": self.aug_score,
            "no_aug_gap": self.no_aug_gap,
            "aug_gap": self.aug_gap,
            "total_instance_num": self.total_instance_num,
            "solved_instance_num": self.solved_instance_num,
            "avg_no_aug_gap": self.avg_no_aug_gap,
            "avg_aug_gap": self.avg_aug_gap,
            "per_instance": self.per_instance,
        }


class TSPTester_LIB:
    def __init__(self, model_params, tester_params):
        self.model_params = model_params
        self.tester_params = tester_params

        self.logger = getLogger('root')

        use_cuda = self.tester_params['use_cuda']
        if use_cuda:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        self.model = Model(**self.model_params)

        checkpoint_fullname = tester_params.get('checkpoint_path')
        if checkpoint_fullname is None:
            model_load = tester_params['model_load']
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        total = sum([param.nelement() for param in self.model.parameters()])
        self.logger.info("Model loaded from: {}".format(checkpoint_fullname))
        self.logger.info("Number of parameters: %.2fM" % (total / 1e6))

        # Optional: attach M2 distance bias module to the model.
        # Priority order:
        #   1. CLI override: if user explicitly passed --distance_bias_enabled
        #      or --knn_bias_enabled, build bias from tester_params.
        #   2. Ckpt-embedded auto-attach: if the checkpoint was produced by
        #      finetune_phased.py with a bias module active, ``checkpoint['bias_cfg']``
        #      records the live cfg (knn_k, scales, etc.). Re-attach it so that the
        #      ckpt evaluates with the same bias it was trained under -- without
        #      requiring any extra CLI flag from the course grader.
        #   3. Else: no bias.
        cli_bias_on = (
            self.tester_params.get('distance_bias_enabled', False)
            or self.tester_params.get('knn_bias_enabled', False)
        )
        ckpt_bias_cfg = checkpoint.get('bias_cfg') if isinstance(checkpoint, dict) else None
        ckpt_bias_on = bool(ckpt_bias_cfg) and (
            bool(ckpt_bias_cfg.get('distance_bias_enabled'))
            or bool(ckpt_bias_cfg.get('knn_bias_enabled'))
        )

        bias_cfg_to_attach = None
        bias_source = None
        if cli_bias_on:
            bias_cfg_to_attach = {
                'distance_bias_enabled': self.tester_params.get('distance_bias_enabled', False),
                'distance_bias_scale': float(self.tester_params.get('distance_bias_scale', 1.0)),
                'distance_bias_mode': self.tester_params.get('distance_bias_mode', 'logit'),
                'distance_norm_mode': self.tester_params.get('distance_norm_mode', 'mean'),
                'knn_bias_enabled': self.tester_params.get('knn_bias_enabled', False),
                'knn_k': int(self.tester_params.get('knn_k', 10)),
                'knn_bias_value': float(self.tester_params.get('knn_bias_value', 0.5)),
            }
            bias_source = 'CLI/tester_params'
        elif ckpt_bias_on:
            bias_cfg_to_attach = dict(ckpt_bias_cfg)
            bias_source = "ckpt['bias_cfg']"

        if bias_cfg_to_attach is not None:
            if DistanceBiasModule is None:
                raise RuntimeError("DistanceBiasModule unavailable; install model_ext.distance_bias.")
            self.model.attach_distance_bias(DistanceBiasModule(bias_cfg_to_attach))
            self.logger.info("[bias] Attached from %s: %s", bias_source, bias_cfg_to_attach)
        else:
            self.logger.info("[bias] No distance/kNN bias attached (ckpt has no bias_cfg, no CLI override).")

    def run_lib(self) -> LibResult:
        filename = self.tester_params['filename']
        scale_range_all = self.tester_params.get('scale_range_all', [[0, 1000]])
        detailed_log = self.tester_params.get('detailed_log', False)

        start_time_all = time.time()
        all_instance_num = 0
        solved_instance_num = 0

        result = LibResult(
            instances=[],
            optimal=[],
            problem_size=[],
            no_aug_score=[],
            aug_score=[],
            no_aug_gap=[],
            aug_gap=[],
        )

        for scale_range in scale_range_all:
            self.logger.info("#################  Test scale range: {}  #################".format(scale_range))

            for root, _, files in os.walk(filename):
                for file in files:
                    if not file.endswith('.tsp'):
                        continue

                    full_path = os.path.join(root, file)
                    name, dimension, locs, ew_type = TSPLIBReader(full_path)

                    all_instance_num += 1

                    if name is None:
                        self.logger.info(f"Skip (unsupported or invalid TSPLIB): {full_path}")
                        continue

                    if not (scale_range[0] <= dimension < scale_range[1]):
                        continue

                    optimal = tsplib_cost.get(name, None)
                    if optimal is None:
                        self.logger.info(
                            f"Optimal not found for {name}. "
                            "Will report scores but leave gap fields empty."
                        )

                    self.logger.info("===============================================================")
                    self.logger.info("Instance name: {}, problem_size: {}, EDGE_WEIGHT_TYPE: {}".format(name, dimension, ew_type))

                    coords_orig_np = np.array(locs, dtype=np.float32)
                    coords_orig = torch.from_numpy(coords_orig_np).to(self.device)
                    node_coord = coords_orig[None, :, :]

                    nodes_xy_normalized = _normalize_to_unit_square(node_coord)

                    try:
                        no_aug_score, aug_score, per_info = self._test_one_instance(
                            nodes_xy_normalized=nodes_xy_normalized,
                            coords_orig=coords_orig,
                            ew_type=ew_type,
                        )
                    except Exception as e:
                        self.logger.exception(f"Failed on instance {name}: {e}")
                        continue

                    solved_instance_num += 1

                    if optimal is None:
                        no_aug_gap = None
                        aug_gap = None
                    else:
                        no_aug_gap = (no_aug_score - optimal) / optimal * 100
                        aug_gap = (aug_score - optimal) / optimal * 100

                    result.instances.append(name)
                    result.optimal.append(float(optimal) if optimal is not None else None)
                    result.problem_size.append(int(dimension))
                    result.no_aug_score.append(float(no_aug_score))
                    result.aug_score.append(float(aug_score))
                    result.no_aug_gap.append(float(no_aug_gap) if no_aug_gap is not None else None)
                    result.aug_gap.append(float(aug_gap) if aug_gap is not None else None)
                    per_info['instance'] = name
                    per_info['problem_size'] = int(dimension)
                    per_info['optimal'] = float(optimal) if optimal is not None else None
                    per_info['no_aug_score'] = float(no_aug_score)
                    per_info['aug_score'] = float(aug_score)
                    per_info['no_aug_gap'] = float(no_aug_gap) if no_aug_gap is not None else None
                    per_info['aug_gap'] = float(aug_gap) if aug_gap is not None else None
                    result.per_instance.append(per_info)

                    if optimal is None:
                        self.logger.info(
                            "no public optimum. no_aug: {:.3f}, aug: {:.3f}".format(
                                no_aug_score, aug_score
                            )
                        )
                    else:
                        self.logger.info(
                            "optimal: {:.3f}, no_aug: {:.3f} (gap {:.3f}%), aug: {:.3f} (gap {:.3f}%)".format(
                                optimal, no_aug_score, no_aug_gap, aug_score, aug_gap
                            )
                        )

        end_time_all = time.time()
        result.total_instance_num = all_instance_num
        result.solved_instance_num = solved_instance_num

        self.logger.info("=========================== Summary ===========================")
        self.logger.info(
            "All done, solved instance number: {}/{}, total time: {:.2f}s".format(
                solved_instance_num, all_instance_num, end_time_all - start_time_all
            )
        )

        if solved_instance_num > 0 and result.avg_aug_gap is not None:
            self.logger.info(
                "Avg gap(no aug): {:.3f}%, Avg gap(aug): {:.3f}%".format(
                    result.avg_no_aug_gap,
                    result.avg_aug_gap,
                )
            )
        elif solved_instance_num > 0:
            self.logger.info(
                "Avg gap unavailable because public optimal tour lengths were not provided "
                "for the evaluated instances."
            )

        if detailed_log:
            self.logger.info("===============================================================")
            self.logger.info("instance: {}".format(result.instances))
            self.logger.info("optimal: {}".format(result.optimal))
            self.logger.info("problem_size: {}".format(result.problem_size))
            self.logger.info("no_aug_score: {}".format(result.no_aug_score))
            self.logger.info("aug_score: {}".format(result.aug_score))
            self.logger.info("no_aug_gap: {}".format(result.no_aug_gap))
            self.logger.info("aug_gap: {}".format(result.aug_gap))

        return result

    def _test_one_instance(self, nodes_xy_normalized: torch.Tensor, coords_orig: torch.Tensor, ew_type: str) -> Tuple[float, float, Dict[str, Any]]:
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
            if aug_factor != 8:
                raise NotImplementedError('Only aug_factor=8 is supported.')
        else:
            aug_factor = 1

        problems = nodes_xy_normalized
        if aug_factor > 1:
            problems = augment_xy_data_by_8_fold(problems)

        effective_batch = problems.size(0)
        problem_size = problems.size(1)

        env = Env(problem_size=problem_size, pomo_size=problem_size)

        env.batch_size = effective_batch
        env.problems = problems.to(self.device)
        env.BATCH_IDX = torch.arange(effective_batch, device=self.device)[:, None].expand(effective_batch, env.pomo_size)
        env.POMO_IDX = torch.arange(env.pomo_size, device=self.device)[None, :].expand(effective_batch, env.pomo_size)

        # Unify TSPLIB scoring: let Env compute integer tour length.
        # - original coords are used for TSPLIB cost (not normalized)
        # - edge_weight_type controls EUC_2D vs CEIL_2D discretization
        env.original_node_xy_lib = coords_orig[None, :, :]
        env.edge_weight_type = ew_type

        baseline_t0 = time.time()
        self.model.eval()
        with torch.no_grad():
            reset_state, _, _ = env.reset()
            self.model.pre_forward(reset_state)

            state, reward, done = env.pre_step()
            while not done:
                selected, _ = self.model(state)
                state, reward, done = env.step(selected, lib_mode=True)

        baseline_ms = (time.time() - baseline_t0) * 1000.0

        # reward is negative tour length at the final step (lib-mode integer distances).
        baseline_tour_lens = (-reward).detach()  # (aug, pomo) float tensor
        baseline_tours = env.selected_node_list.detach().clone()  # (aug, pomo, N) int64

        baseline_no_aug_score = float(baseline_tour_lens[0].min().item())
        baseline_aug_score = float(baseline_tour_lens.min().item())

        rerank_enabled = bool(self.tester_params.get('rerank_enabled', False))
        two_opt_enabled = bool(self.tester_params.get('two_opt_enabled', False))

        per_info: Dict[str, Any] = {
            'baseline_no_aug_score': baseline_no_aug_score,
            'baseline_aug_score': baseline_aug_score,
            'baseline_ms': float(baseline_ms),
            'rerank_enabled': rerank_enabled,
            'two_opt_enabled': two_opt_enabled,
        }

        # Fast path: all off — keep behavior bit-exact with the original tester.
        if not rerank_enabled and not two_opt_enabled:
            per_info['final_no_aug_score'] = baseline_no_aug_score
            per_info['final_aug_score'] = baseline_aug_score
            return baseline_no_aug_score, baseline_aug_score, per_info

        rerank_tours_np: Optional[np.ndarray] = None
        rerank_lens_np: Optional[np.ndarray] = None
        if rerank_enabled:
            rerank_cfg = self._collect_rerank_cfg()
            with torch.no_grad():
                rerank_tours_np, rerank_lens_np, rerank_info = rerank_sgbs_lite(
                    self.model, Env, problems, coords_orig, ew_type, rerank_cfg, self.device,
                )
            per_info['rerank_info'] = rerank_info

        baseline_tours_np = baseline_tours.cpu().numpy()
        baseline_lens_np = baseline_tour_lens.cpu().numpy()
        coords_orig_np = coords_orig.detach().cpu().numpy()

        # Build aug-slice views for no_aug scoring (aug index 0 only).
        aug0_baseline_tours = baseline_tours_np[:1]
        aug0_baseline_lens = baseline_lens_np[:1]
        if rerank_tours_np is not None:
            aug0_rerank_tours = rerank_tours_np[:1]
            aug0_rerank_lens = rerank_lens_np[:1]
        else:
            aug0_rerank_tours = None
            aug0_rerank_lens = None

        two_opt_cfg = self._collect_two_opt_cfg() if two_opt_enabled else None
        pool_across_augs = bool(self.tester_params.get('rerank_pool_across_augs', True))
        deduplicate = bool(self.tester_params.get('rerank_deduplicate', True))

        no_aug_tour, no_aug_len, no_aug_dbg = self._select_and_refine(
            aug0_baseline_tours, aug0_baseline_lens,
            aug0_rerank_tours, aug0_rerank_lens,
            coords_orig_np, ew_type,
            rerank_enabled, two_opt_enabled, two_opt_cfg,
            pool_across_augs=True, deduplicate=deduplicate,
        )
        aug_tour, aug_len, aug_dbg = self._select_and_refine(
            baseline_tours_np, baseline_lens_np,
            rerank_tours_np, rerank_lens_np,
            coords_orig_np, ew_type,
            rerank_enabled, two_opt_enabled, two_opt_cfg,
            pool_across_augs=pool_across_augs, deduplicate=deduplicate,
        )

        per_info['no_aug_pool'] = no_aug_dbg
        per_info['aug_pool'] = aug_dbg
        per_info['final_no_aug_score'] = float(no_aug_len)
        per_info['final_aug_score'] = float(aug_len)
        per_info['improved_no_aug'] = bool(no_aug_len < baseline_no_aug_score - 1e-9)
        per_info['improved_aug'] = bool(aug_len < baseline_aug_score - 1e-9)

        return float(no_aug_len), float(aug_len), per_info

    # ------------------------------------------------------------------
    # Helpers

    def _collect_rerank_cfg(self) -> Dict[str, Any]:
        keys = [
            'rerank_enabled', 'rerank_beam_width', 'rerank_depth', 'rerank_topk_per_step',
            'rerank_use_entropy_gate', 'rerank_entropy_threshold',
            'rerank_pool_across_augs', 'rerank_deduplicate',
        ]
        cfg: Dict[str, Any] = {}
        for k in keys:
            if k in self.tester_params:
                cfg[k] = self.tester_params[k]
        return cfg

    def _collect_two_opt_cfg(self) -> Dict[str, Any]:
        keys = [
            'two_opt_enabled', 'two_opt_target', 'two_opt_topk',
            'two_opt_max_iters', 'two_opt_first_improvement', 'two_opt_time_budget_ms',
        ]
        cfg: Dict[str, Any] = {}
        for k in keys:
            if k in self.tester_params:
                cfg[k] = self.tester_params[k]
        return cfg

    def _select_and_refine(
        self,
        baseline_tours: np.ndarray,
        baseline_lens: np.ndarray,
        rerank_tours: Optional[np.ndarray],
        rerank_lens: Optional[np.ndarray],
        coords_orig_np: np.ndarray,
        ew_type: str,
        rerank_enabled: bool,
        two_opt_enabled: bool,
        two_opt_cfg: Optional[Dict[str, Any]],
        pool_across_augs: bool,
        deduplicate: bool,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Pool candidates (M1), then optionally refine with 2-opt (M4)."""
        r_tours = rerank_tours if rerank_enabled else None
        r_lens = rerank_lens if rerank_enabled else None
        best_source, best_tour, best_len, pool_info = pool_and_select_best(
            baseline_tours, baseline_lens, r_tours, r_lens,
            pool_across_augs=pool_across_augs, deduplicate=deduplicate,
        )

        dbg: Dict[str, Any] = {
            'pool_info': pool_info,
            'best_source_pool': int(best_source),
        }

        if not two_opt_enabled:
            dbg['two_opt'] = None
            return best_tour, best_len, dbg

        assert two_opt_cfg is not None
        target = str(self.tester_params.get('two_opt_target', 'final_best'))
        topk_n = int(self.tester_params.get('two_opt_topk', 3))

        two_opt_records: List[Dict[str, Any]] = []
        if target == 'topk_candidates':
            # Collect global candidate pool, pick top-k shortest, 2-opt each,
            # then compare to the currently-selected best.
            aug = baseline_tours.shape[0]
            n = baseline_tours.shape[-1]
            pool_tours = baseline_tours.reshape(-1, n)
            pool_lens = baseline_lens.reshape(-1)
            if rerank_enabled and rerank_tours is not None:
                pool_tours = np.concatenate(
                    [pool_tours, rerank_tours.reshape(-1, n)], axis=0
                )
                pool_lens = np.concatenate(
                    [pool_lens, rerank_lens.reshape(-1)], axis=0
                )
            k_eff = max(1, min(topk_n, pool_lens.shape[0]))
            top_idx = np.argsort(pool_lens)[:k_eff]
            for idx in top_idx:
                refined_tour, refined_len, info = two_opt_refine(
                    pool_tours[idx], coords_orig_np, ew_type, two_opt_cfg,
                )
                two_opt_records.append({'orig_len': float(pool_lens[idx]), **info})
                if refined_len < best_len - 1e-9:
                    best_len = float(refined_len)
                    best_tour = refined_tour
        else:  # final_best
            refined_tour, refined_len, info = two_opt_refine(
                best_tour, coords_orig_np, ew_type, two_opt_cfg,
            )
            two_opt_records.append({'orig_len': float(best_len), **info})
            if refined_len < best_len - 1e-9:
                best_len = float(refined_len)
                best_tour = refined_tour

        dbg['two_opt'] = two_opt_records
        return best_tour, best_len, dbg
