
import torch
from logging import getLogger

from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *

# M2 / M3 extensions (all default OFF via configs)
try:
    from model_ext.distance_bias import DistanceBiasModule
except Exception:  # pragma: no cover
    DistanceBiasModule = None  # type: ignore
from train_ext.leader_reward import compute_leader_loss


class TSPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load.get('enable', False):
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')
        elif trainer_params.get('finetune_from') is not None:
            # Finetune path: load weights only, keep optimizer/scheduler fresh.
            ckpt = trainer_params['finetune_from']
            checkpoint = torch.load(ckpt, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.logger.info('Finetune from checkpoint: {}'.format(ckpt))
            # Optional: continue the *epoch counter* from the checkpoint so
            # log lines and saved checkpoint filenames reflect the absolute
            # epoch number (e.g. 3001..). Default off for backward compat;
            # turned on by the phased fine-tune driver.
            if trainer_params.get('finetune_continue_epoch_counter', False):
                ck_epoch = int(checkpoint.get('epoch', 0))
                if ck_epoch > 0:
                    self.start_epoch = ck_epoch + 1
                    # Keep scheduler aligned with continued counter (does not
                    # change LR because the only milestone is at 3001 with
                    # gamma=0.1 in the baseline recipe).
                    self.scheduler.last_epoch = ck_epoch - 1
                    self.logger.info(
                        'finetune_continue_epoch_counter=True: start_epoch=%d',
                        self.start_epoch,
                    )

        # Optional M2 distance bias.
        bias_cfg = trainer_params.get('distance_bias_cfg')
        if bias_cfg is not None and (bias_cfg.get('distance_bias_enabled') or bias_cfg.get('knn_bias_enabled')):
            if DistanceBiasModule is None:
                raise RuntimeError("DistanceBiasModule unavailable; check model_ext/distance_bias.py.")
            module = DistanceBiasModule(bias_cfg)
            self.model.attach_distance_bias(module)
            self.logger.info('Attached distance-bias module: {}'.format(bias_cfg))

        # Optional M3 leader reward.
        leader_cfg = trainer_params.get('leader_cfg')
        self.leader_cfg = leader_cfg if (leader_cfg and leader_cfg.get('leader_reward_enabled')) else None
        if self.leader_cfg is not None:
            self.logger.info('Leader reward enabled: {}'.format(self.leader_cfg))

        # Optional gradient clipping (turned on automatically when any new
        # training feature is active).
        self.grad_clip_max_norm = trainer_params.get('grad_clip_max_norm')
        if self.grad_clip_max_norm is None and (self.leader_cfg is not None or bias_cfg is not None):
            self.grad_clip_max_norm = 1.0
            self.logger.info('Auto grad-clipping enabled (max_norm=1.0) for stability.')

        # Optional Mixed Structured Curriculum for training data generation.
        # ``msc_cfg`` is a plain dict (see TSProblemDef.DEFAULT_MSC_CONFIG).
        # When absent or ``enabled=False`` we fall back to legacy uniform sampling.
        self.msc_cfg = trainer_params.get('msc_cfg')
        if self.msc_cfg is not None and self.msc_cfg.get('enabled', False):
            self.logger.info(
                'MSC enabled: use_curriculum=%s stage_boundaries=%s',
                bool(self.msc_cfg.get('use_curriculum', True)),
                self.msc_cfg.get('curriculum', {}).get('stage_boundaries'),
            )
        else:
            self.logger.info('MSC disabled -> using legacy uniform sampler.')

        # Optional per-batch N sampler. When enabled, _train_one_batch mutates
        # env.problem_size/pomo_size before load_problems.
        # Shape: {'enabled': bool, 'n_values': [int,...], 'weights': [float,...]?}
        self.n_sampler_cfg = trainer_params.get('n_sampler_cfg')
        if self.n_sampler_cfg and self.n_sampler_cfg.get('enabled'):
            self.logger.info(
                'n_sampler enabled: n_values=%s weights=%s',
                self.n_sampler_cfg.get('n_values'),
                self.n_sampler_cfg.get('weights'),
            )

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            # LR Decay
            self.scheduler.step()

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            if epoch > 1:  # save latest images, every epoch
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data(),
                    'bias_cfg': self._snapshot_bias_cfg(),
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    # ------------------------------------------------------------------
    # Phased fine-tune support (used by ``finetune_phased.py``).
    # Existing ``run()`` is left untouched so every existing script /
    # ablation continues to work unchanged.
    # ------------------------------------------------------------------

    def apply_phase_config(self, phase_cfg):
        """Apply per-phase configuration (MSC ratios / bias / leader / lr).
        ``phase_cfg`` is a flat dict with optional keys (all of them
        optional; any missing key keeps the trainer's current state):

            phase_name              : str   (for logging)
            msc_enabled             : bool
            msc_use_curriculum      : bool  (False => use fixed_ratios)
            msc_fixed_ratios        : dict  (uniform/clustered_uniform/gaussian_mixture)
            distance_bias_cfg       : dict  (None => keep, {} => detach)
            leader_cfg              : dict  (None => keep, {} => disable)
            optimizer_lr            : float (sets group 0 lr)
            optimizer_lr_groups     : list[float] (per-group override)
        """
        log = self.logger
        name = phase_cfg.get('phase_name', 'phase')

        # ---- MSC ratios ------------------------------------------------
        msc_cfg = self.trainer_params.get('msc_cfg')
        if msc_cfg is None:
            from TSProblemDef import get_default_msc_config
            msc_cfg = get_default_msc_config()
            self.trainer_params['msc_cfg'] = msc_cfg
        if 'msc_enabled' in phase_cfg:
            msc_cfg['enabled'] = bool(phase_cfg['msc_enabled'])
        if 'msc_use_curriculum' in phase_cfg:
            msc_cfg['use_curriculum'] = bool(phase_cfg['msc_use_curriculum'])
        if 'msc_fixed_ratios' in phase_cfg and phase_cfg['msc_fixed_ratios'] is not None:
            ratios = phase_cfg['msc_fixed_ratios']
            msc_cfg.setdefault('curriculum', {})['fixed_ratios'] = {
                'uniform': float(ratios.get('uniform', 0.0)),
                'clustered_uniform': float(ratios.get('clustered_uniform', 0.0)),
                'gaussian_mixture': float(ratios.get('gaussian_mixture', 0.0)),
            }
        self.msc_cfg = msc_cfg

        # ---- Distance / kNN bias --------------------------------------
        bias_cfg = phase_cfg.get('distance_bias_cfg', None)
        if bias_cfg is not None:
            if not bias_cfg or not (
                bias_cfg.get('distance_bias_enabled') or bias_cfg.get('knn_bias_enabled')
            ):
                # Detach.
                self.model.attach_distance_bias(None)
                self.trainer_params.pop('distance_bias_cfg', None)
            else:
                if DistanceBiasModule is None:
                    raise RuntimeError("DistanceBiasModule unavailable; check model_ext/distance_bias.py.")
                module = DistanceBiasModule(bias_cfg)
                self.model.attach_distance_bias(module)
                self.trainer_params['distance_bias_cfg'] = bias_cfg
                if self.grad_clip_max_norm is None:
                    self.grad_clip_max_norm = 1.0

        # ---- Leader-focused reward ------------------------------------
        if 'leader_cfg' in phase_cfg:
            lc = phase_cfg['leader_cfg']
            if not lc or not lc.get('leader_reward_enabled'):
                self.leader_cfg = None
                self.trainer_params.pop('leader_cfg', None)
            else:
                self.leader_cfg = lc
                self.trainer_params['leader_cfg'] = lc
                if self.grad_clip_max_norm is None:
                    self.grad_clip_max_norm = 1.0

        # ---- Optimizer LR ---------------------------------------------
        if 'optimizer_lr_groups' in phase_cfg and phase_cfg['optimizer_lr_groups']:
            lrs = phase_cfg['optimizer_lr_groups']
            for g, lr in zip(self.optimizer.param_groups, lrs):
                g['lr'] = float(lr)
        elif 'optimizer_lr' in phase_cfg and phase_cfg['optimizer_lr'] is not None:
            for g in self.optimizer.param_groups:
                g['lr'] = float(phase_cfg['optimizer_lr'])

        # ---- Log a one-line summary of the active configuration -------
        cur_lr = [round(g['lr'], 8) for g in self.optimizer.param_groups]
        msc_ratio = (msc_cfg.get('curriculum', {}) or {}).get('fixed_ratios') if msc_cfg else None
        log.info(
            "==[PHASE %s]== msc_enabled=%s msc_fixed_ratios=%s "
            "bias=%s leader=%s lr=%s grad_clip=%s",
            name,
            bool(msc_cfg.get('enabled', False)) if msc_cfg else False,
            msc_ratio,
            self.trainer_params.get('distance_bias_cfg'),
            self.trainer_params.get('leader_cfg'),
            cur_lr,
            self.grad_clip_max_norm,
        )

    def run_phase(self, num_epochs, phase_name='phase', save_phase_best=True,
                  epoch_callback=None, snapshot_at_phase_epochs=None):
        """Run ``num_epochs`` training epochs starting from ``self.start_epoch``.

        Mirrors :meth:`run` but is bounded by ``num_epochs``. Updates
        ``self.start_epoch`` so a follow-up call resumes seamlessly. The
        optional ``epoch_callback(trainer, epoch_in_phase, abs_epoch)`` is
        invoked *before* each epoch (after scheduler.step) so callers can
        ramp up bias strength / leader gamma per-epoch.

        ``snapshot_at_phase_epochs`` is a list/set of in-phase epoch indices
        (1-indexed). After processing those epochs, the current phase-best
        checkpoint is copied to a tagged file
        ``checkpoint-phase_{phase_name}_best_at_{N}.pt``. Useful when a single
        long run should produce ckpts equivalent to multiple shorter runs
        (e.g., run 560 ep but also keep "best of first 420 ep").
        """
        log = self.logger
        end_epoch = self.start_epoch + int(num_epochs) - 1
        log.info("[PHASE %s] running epochs %d..%d (count=%d)",
                 phase_name, self.start_epoch, end_epoch, num_epochs)

        # Use absolute end_epoch as ``total_epochs`` so MSC's optional
        # use_curriculum=True path remains well defined; the typical phased
        # recipe uses use_curriculum=False (per-phase fixed ratios) anyway.
        self.trainer_params['epochs'] = max(self.trainer_params.get('epochs', 0), end_epoch)

        self.time_estimator.reset(self.start_epoch)
        phase_best_score = float('inf')
        phase_best_path = None
        snapshot_set = set(int(e) for e in (snapshot_at_phase_epochs or []))
        model_save_interval = self.trainer_params['logging']['model_save_interval']
        img_save_interval = self.trainer_params['logging']['img_save_interval']
        for i, epoch in enumerate(range(self.start_epoch, end_epoch + 1), start=1):
            log.info('=================================================================')

            self.scheduler.step()

            if epoch_callback is not None:
                try:
                    epoch_callback(self, i, epoch)
                except Exception as exc:  # callback errors must not kill the run
                    log.warning("epoch_callback raised %r; ignored.", exc)

            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(i, num_epochs)
            log.info("[PHASE %s] Epoch %3d (in-phase %d/%d) score=%.4f loss=%.4f Elapsed[%s] Remain[%s]",
                     phase_name, epoch, i, num_epochs, train_score, train_loss,
                     elapsed_time_str, remain_time_str)

            phase_done = (epoch == end_epoch)
            if phase_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            # Save: latest every epoch, periodic checkpoint, and a phase_best
            # tracker (lowest train_score within the phase).
            ckpt_dict = {
                'epoch': epoch,
                'phase': phase_name,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'result_log': self.result_log.get_raw_data(),
                'bias_cfg': self._snapshot_bias_cfg(),
            }

            latest_path = '{}/checkpoint-latest.pt'.format(self.result_folder)
            torch.save(ckpt_dict, latest_path)

            if phase_done or (epoch % model_save_interval) == 0:
                periodic_path = '{}/checkpoint-{}.pt'.format(self.result_folder, epoch)
                torch.save(ckpt_dict, periodic_path)

            if save_phase_best and train_score < phase_best_score:
                phase_best_score = train_score
                phase_best_path = '{}/checkpoint-phase_{}_best.pt'.format(self.result_folder, phase_name)
                torch.save(ckpt_dict, phase_best_path)

            # Snapshot the current phase best at user-specified in-phase epochs.
            # Captures "best of first N epochs" for multi-budget reporting.
            if i in snapshot_set:
                if phase_best_path is not None and os.path.exists(phase_best_path):
                    import shutil
                    snap_path = '{}/checkpoint-phase_{}_best_at_{}.pt'.format(
                        self.result_folder, phase_name, i)
                    shutil.copy(phase_best_path, snap_path)
                    log.info("[snapshot] in-phase %d: copied phase best (score=%.4f) -> %s",
                             i, phase_best_score, os.path.basename(snap_path))
                else:
                    log.warning("[snapshot] in-phase %d: no phase best yet, skipping snapshot.", i)

        if phase_best_path is not None:
            log.info("[PHASE %s] phase_best train_score=%.4f saved to %s",
                     phase_name, phase_best_score, phase_best_path)

        # Advance start_epoch so the next phase resumes seamlessly.
        self.start_epoch = end_epoch + 1
        return {
            'phase_name': phase_name,
            'epoch_range': (end_epoch - num_epochs + 1, end_epoch),
            'phase_best_score': phase_best_score,
            'phase_best_path': phase_best_path,
        }

    def _snapshot_bias_cfg(self):
        """Return a JSON-safe snapshot of the currently-attached distance-bias
        module's live cfg, or ``None`` if no module is attached / enabled.

        Embedding this into the checkpoint lets the tester (TSPTester_LIB)
        auto-reattach the bias at evaluation time without requiring the
        course-grader to pass any extra CLI flag to ``test.py``.
        """
        mod = getattr(self.model, 'distance_bias_module', None)
        if mod is None or not getattr(mod, 'enabled', False):
            return None
        # ``mod.cfg`` may contain non-JSON-safe types in principle; cast the
        # numeric / bool fields explicitly to keep ckpt portable.
        cfg = mod.cfg
        return {
            'distance_bias_enabled': bool(cfg.get('distance_bias_enabled', False)),
            'distance_bias_scale':   float(cfg.get('distance_bias_scale', 1.0)),
            'distance_bias_mode':    str(cfg.get('distance_bias_mode', 'logit')),
            'distance_norm_mode':    str(cfg.get('distance_norm_mode', 'mean')),
            'knn_bias_enabled':      bool(cfg.get('knn_bias_enabled', False)),
            'knn_k':                 int(cfg.get('knn_k', 10)),
            'knn_bias_value':        float(cfg.get('knn_bias_value', 0.5)),
        }

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, avg_loss = self._train_one_batch(batch_size, epoch=epoch)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size, epoch=None):

        # Prep
        ###############################################
        self.model.train()

        # Variable-N synthetic batch (when n_sampler is enabled) or fixed N.
        # The env's problem_size/pomo_size attributes are mutated *before*
        # load_problems so the regenerated BATCH_IDX/POMO_IDX/ninf_mask pick
        # up the new N.
        if self.n_sampler_cfg and self.n_sampler_cfg.get('enabled'):
            import random as _rand
            n_values = self.n_sampler_cfg['n_values']
            n_weights = self.n_sampler_cfg.get('weights')
            n = _rand.choices(n_values, weights=n_weights, k=1)[0]
            self.env.problem_size = int(n)
            self.env.pomo_size = int(n)
        self.env.load_problems(
            batch_size,
            epoch=epoch,
            total_epochs=self.trainer_params.get('epochs'),
            msc_config=self.msc_cfg,
        )
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        ###############################################
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)

        if self.leader_cfg is not None:
            loss_mean, leader_stats = compute_leader_loss(reward, log_prob, self.leader_cfg)
            if leader_stats.get('nan_detected', False):
                self.logger.warning('NaN detected in leader loss, skipping step.')
                self.optimizer.zero_grad(set_to_none=True)
                max_pomo_reward, _ = reward.max(dim=1)
                score_mean = -max_pomo_reward.float().mean()
                return score_mean.item(), 0.0
            self._last_leader_stats = leader_stats
        else:
            advantage = reward - reward.float().mean(dim=1, keepdims=True)
            # shape: (batch, pomo)
            loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
            # shape: (batch, pomo)
            loss_mean = loss.mean()

        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss_mean.backward()
        if self.grad_clip_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.grad_clip_max_norm))
        self.optimizer.step()
        return score_mean.item(), loss_mean.item()
