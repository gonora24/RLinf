# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from rlinf.config import SupportedModel
from rlinf.data.embodied_buffer_dataset import (
    PreloadReplayBufferDataset,
    ReplayBufferDataset,
    replay_buffer_collate_fn,
)
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.modules.entropy_tunning import EntropyTemperature
from rlinf.scheduler import Channel, Worker
from rlinf.utils import color_jitter, drq
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_split_num,
)
from rlinf.utils.nested_dict_process import (
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.utils import clear_memory
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


def _main_image_for_save(img: torch.Tensor) -> torch.Tensor:
    """Convert one main image to [C, H, W] float32 in [0, 1] for save_image."""
    x = img.detach().cpu()
    if x.ndim != 3:
        raise ValueError(f"Expected 3D image tensor, got {tuple(x.shape)}")
    if x.shape[-1] == 3:
        x = x.permute(2, 0, 1)
    elif x.shape[0] != 3:
        raise ValueError(f"Unrecognized image layout: {tuple(x.shape)}")
    if x.dtype == torch.uint8:
        x = x.float().div(255.0)
    else:
        x = x.float()
        if x.max() > 1.0 + 1e-3:
            x = x.div(255.0)
    return x.clamp(0.0, 1.0)


class EmbodiedSACFSDPPolicy(EmbodiedFSDPActor):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # SAC-specific initialization
        self.replay_buffer = None
        self.target_model = None
        self.entropy_temp = None
        self.demo_buffer = None
        self.alpha_optimizer = None
        self.update_step = 0
        self.enable_drq = bool(getattr(self.cfg.actor, "enable_drq", False))
        self.color_jitter = bool(getattr(self.cfg.actor, "color_jitter", False))

    def init_worker(self):
        self.setup_model_and_optimizer(initialize_target=True)
        self.setup_sac_components()
        self.soft_update_target_model(tau=1.0)
        if self.use_dsrl:
            self._init_target_shadow()
        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()
        if self.cfg.actor.get("compile_model", False):
            self.model = torch.compile(
                self.model, mode="default"
            )  # max-autotune-no-cudagraphs
            self.target_model = torch.compile(self.target_model, mode="default")
        elif self.use_dsrl:
            # Compile DSRL sub-module forward passes (not the modules themselves).
            # Replacing sub-modules with torch.compile(module) inserts _orig_mod
            # into FSDP parameter names and breaks sync_model_to_rollout; compiling
            # forward keeps the module tree intact while JIT-ing the hot paths.
            def _unwrap_fsdp(m):
                while hasattr(m, "_fsdp_wrapped_module"):
                    m = m._fsdp_wrapped_module
                return m

            def _compile_forward(module):
                module.forward = torch.compile(module.forward, fullgraph=False)

            _dsrl_actor_attrs = [
                "actor_image_encoder",
                "critic_image_encoder",
                "dsrl_action_noise_net",
                "q_head",
            ]
            _dsrl_target_attrs = ["critic_image_encoder", "q_head"]
            try:
                compiled_ids = set()
                inner = _unwrap_fsdp(self.model)
                for attr in _dsrl_actor_attrs:
                    if hasattr(inner, attr):
                        mod = getattr(inner, attr)
                        if id(mod) not in compiled_ids:
                            _compile_forward(mod)
                            compiled_ids.add(id(mod))
                target_inner = _unwrap_fsdp(self.target_model)
                for attr in _dsrl_target_attrs:
                    if hasattr(target_inner, attr):
                        mod = getattr(target_inner, attr)
                        if id(mod) not in compiled_ids:
                            _compile_forward(mod)
                            compiled_ids.add(id(mod))
                self.log_on_first_rank(
                    "[DSRL] Compiled DSRL sub-module forward passes with "
                    f"torch.compile ({_dsrl_actor_attrs})"
                )
            except Exception as e:
                self.log_on_first_rank(
                    f"[DSRL] torch.compile for DSRL sub-modules failed (continuing "
                    f"without compile): {e}"
                )

    def setup_model_and_optimizer(self, initialize_target=False) -> None:
        """Setup model, lr_scheduler, optimizer and grad_scaler."""
        """Add initializing target model logic."""
        module = self.model_provider_func()
        if initialize_target:
            target_module = self.model_provider_func()

        # Enable gradient checkpointing if configured
        if self.cfg.actor.model.get("gradient_checkpointing", False):
            self.logger.info("[FSDP] Enabling gradient checkpointing")
            module.gradient_checkpointing_enable()
            if initialize_target:
                target_module.gradient_checkpointing_enable()
        else:
            self.logger.info("[FSDP] Gradient checkpointing is disabled")

        # build model, optimizer, lr_scheduler, grad_scaler
        self.model = self._strategy.wrap_model(
            model=module, device_mesh=self._device_mesh
        )
        # When precision is null (e.g. Pi0), detect actual dtype from wrapped model
        if self.torch_dtype is None:
            self.torch_dtype = next(self.model.parameters()).dtype
        if initialize_target:
            self.target_model = self._strategy.wrap_model(
                model=target_module, device_mesh=self._device_mesh
            )
            self.target_model.requires_grad_(False)
            self.target_model_initialized = True
            self._trainable_param_names = {
                name for name, p in self.model.named_parameters() if p.requires_grad
            }

        self.use_dsrl = self.cfg.actor.model.get("openpi", {}).get("use_dsrl", False)
        use_dsrl = self.use_dsrl
        if use_dsrl:
            # DSRL: route critic modules to qf_optimizer; actor encoder/noise net to main.
            param_filters = {
                "critic": ["critic_image_encoder", "critic_state_encoder", "q_head", "critic"]
            }
            share_encoder = self.cfg.actor.model.get("openpi", {}).get(
                "dsrl_share_image_encoder", False
            )
            both_optimizers_params = (
                ["critic_image_encoder"] if share_encoder else []
            )
        else:
            param_filters = {"critic": ["encoders", "encoder", "q_head", "state_proj"]}
            both_optimizers_params = []
        filtered_optim_config = {"critic": self.cfg.actor.critic_optim}
        optimizers = self.build_optimizers(
            model=self.model,
            main_optim_config=self.cfg.actor.optim,
            param_filters=param_filters,
            filtered_optim_config=filtered_optim_config,
            both_optimizers_params=both_optimizers_params,
        )
        self.optimizer = optimizers[0]
        self.qf_optimizer = optimizers[1]

        # SAC alpha
        # Initialize temperature parameter for automatic entropy tuning
        alpha_type = self.cfg.algorithm.entropy_tuning.get(
            "alpha_type", "softplus"
        )  # supported type: ["softplus","exp","fixed_alpha"]
        self.entropy_temp = EntropyTemperature(
            initial_alpha=self.cfg.algorithm.entropy_tuning.get("initial_alpha", 0.01),
            alpha_type=alpha_type,
            device=self.device,
            dtype=self.torch_dtype,
        )
        if alpha_type != "fixed_alpha":
            self.target_entropy = self.cfg.algorithm.entropy_tuning.get(
                "target_entropy",
                -self.cfg.actor.model.action_dim,
            )

            self.alpha_optimizer = torch.optim.Adam(
                self.entropy_temp.parameters(),
                lr=self.cfg.algorithm.entropy_tuning.optim.lr,
            )

        self.build_lr_schedulers()

        self.grad_scaler = self.build_grad_scaler(
            self.cfg.actor.fsdp_config.grad_scaler
        )

    def build_lr_schedulers(self):
        self.lr_scheduler = self.build_lr_scheduler(
            self.optimizer, self.cfg.actor.optim
        )
        self.qf_lr_scheduler = self.build_lr_scheduler(
            self.qf_optimizer, self.cfg.actor.critic_optim
        )
        if self.alpha_optimizer is not None:
            self.alpha_lr_scheduler = self.build_lr_scheduler(
                self.alpha_optimizer, self.cfg.algorithm.entropy_tuning.optim
            )

    def setup_sac_components(self):
        """Initialize SAC-specific components"""
        # Initialize replay buffer
        seed = self.cfg.actor.get("seed", 1234)
        auto_save_path = self.cfg.algorithm.replay_buffer.get("auto_save_path", None)
        if auto_save_path is None:
            auto_save_path = os.path.join(
                self.cfg.runner.logger.log_path, f"replay_buffer/rank_{self._rank}"
            )
        else:
            auto_save_path = os.path.join(auto_save_path, f"rank_{self._rank}")
        self.replay_buffer = TrajectoryReplayBuffer(
            seed=seed,
            enable_cache=self.cfg.algorithm.replay_buffer.enable_cache,
            cache_size=self.cfg.algorithm.replay_buffer.cache_size,
            sample_window_size=self.cfg.algorithm.replay_buffer.sample_window_size,
            auto_save=self.cfg.algorithm.replay_buffer.get("auto_save", False),
            auto_save_path=auto_save_path,
            trajectory_format=self.cfg.algorithm.replay_buffer.get(
                "trajectory_format", "pt"
            ),
        )
        self._last_rollout_num_transitions = 0

        min_demo_buffer_size = 0
        if self.cfg.algorithm.get("demo_buffer", None) is not None:
            auto_save_path = self.cfg.algorithm.demo_buffer.get("auto_save_path", None)
            if auto_save_path is None:
                auto_save_path = os.path.join(
                    self.cfg.runner.logger.log_path, f"demo_buffer/rank_{self._rank}"
                )
            else:
                auto_save_path = os.path.join(auto_save_path, f"rank_{self._rank}")
            self.demo_buffer = TrajectoryReplayBuffer(
                seed=seed,
                enable_cache=self.cfg.algorithm.demo_buffer.enable_cache,
                cache_size=self.cfg.algorithm.demo_buffer.cache_size,
                sample_window_size=self.cfg.algorithm.demo_buffer.sample_window_size,
                auto_save=self.cfg.algorithm.demo_buffer.get("auto_save", False),
                auto_save_path=auto_save_path,
                trajectory_format="pt",
            )
            min_demo_buffer_size = self.cfg.algorithm.demo_buffer.min_buffer_size
            if self.cfg.algorithm.demo_buffer.get("load_path", None) is not None:
                self.demo_buffer.load_checkpoint(
                    self.cfg.algorithm.demo_buffer.load_path,
                    is_distributed=True,
                    local_rank=self._rank,
                    world_size=self._world_size,
                )

        if self.cfg.algorithm.replay_buffer.get("enable_preload", False):
            buffer_dataset_cls = PreloadReplayBufferDataset
        else:
            buffer_dataset_cls = ReplayBufferDataset
        self.buffer_dataset = buffer_dataset_cls(
            replay_buffer=self.replay_buffer,
            demo_buffer=self.demo_buffer,
            batch_size=self.cfg.actor.global_batch_size // self._world_size,
            min_replay_buffer_size=self.cfg.algorithm.replay_buffer.min_buffer_size,
            min_demo_buffer_size=min_demo_buffer_size,
            prefetch_size=self.cfg.algorithm.replay_buffer.get("prefetch_size", 10),
        )
        self.buffer_dataloader = DataLoader(
            self.buffer_dataset,
            batch_size=1,
            num_workers=0,
            drop_last=True,
            collate_fn=replay_buffer_collate_fn,
        )
        self.buffer_dataloader_iter = iter(self.buffer_dataloader)

        self.critic_actor_ratio = self.cfg.algorithm.get("critic_actor_ratio", 1)
        self.critic_subsample_size = self.cfg.algorithm.get("critic_subsample_size", -1)
        self.critic_sample_generator = torch.Generator(self.device)
        self.critic_sample_generator.manual_seed(seed)

        self.target_update_type = self.cfg.algorithm.get("target_update_type", "all")
        assert self.target_update_type in ["all", "q_head_only"], (
            f"{self.target_update_type=} is not suppported!"
        )

    def _init_target_shadow(self):
        """Create persistent float32 shadow of target model parameters.

        bfloat16 has only 7 mantissa bits (ULP ~0.002 at magnitude 0.3).
        With tau=0.005, per-step EMA delta can be smaller than ULP/2, so
        storing back to bf16 each step rounds away the update. The shadow
        keeps the accumulated EMA state in float32 (ULP ~3.6e-8) across
        steps, preventing precision loss.
        """
        self._target_shadow_f32 = {}
        for name, param in self.target_model.named_parameters():
            self._target_shadow_f32[name] = param.data.float().clone()

    def soft_update_target_model(self, tau: Optional[float] = None):
        """Soft update target model parameters.

        For DSRL (bfloat16 models), uses a persistent float32 shadow buffer
        to prevent EMA precision loss. For non-DSRL SAC, uses direct EMA
        on model parameters.
        """
        if tau is None:
            tau = self.cfg.algorithm.tau

        assert self.target_model_initialized

        with torch.no_grad():
            if not hasattr(self, "_target_shadow_f32"):
                # Non-DSRL path (or before shadow init): direct EMA update
                for (name1, online_param), (name2, target_param) in zip(
                    self.model.named_parameters(),
                    self.target_model.named_parameters(),
                ):
                    assert name1 == name2
                    if "q_head" not in name1:
                        if self.target_update_type == "all":
                            target_param.data.mul_(1.0 - tau)
                            target_param.data.add_(online_param.data * tau)
                        else:
                            target_param.data.mul_(0.0)
                            target_param.data.add_(online_param.data)
                    else:
                        target_param.data.mul_(1.0 - tau)
                        target_param.data.add_(online_param.data * tau)
            else:
                # DSRL path: float32 shadow buffer for bf16 precision
                for (name1, online_param), (name2, target_param) in zip(
                    self.model.named_parameters(),
                    self.target_model.named_parameters(),
                ):
                    if name1 not in self._trainable_param_names:
                        continue # skip non-trainable parameters
                    assert name1 == name2
                    if "q_head" not in name1 and self.target_update_type != "all":
                        shadow = self._target_shadow_f32[name1]
                        shadow.copy_(online_param.data.float())
                        target_param.data.copy_(shadow.to(target_param.data.dtype))
                    else:
                        shadow = self._target_shadow_f32[name1]
                        shadow.mul_(1.0 - tau).add_(
                            online_param.data.float(), alpha=tau
                        )
                        target_param.data.copy_(shadow.to(target_param.data.dtype))

    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        """
        Receive rollout trajectories from rollout workers.

        Args:
            input_channel: The input channel to read from.
        """
        clear_memory(sync=False)

        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []

        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        if self.use_dsrl and recv_list:
            # Pre-resize trajectory images from env resolution (e.g. 256×256) to
            # the 64×64 DSRL encoder input resolution.  This avoids running the
            # same bilinear resize 2×400=800 times per rollout step inside
            # update_one_epoch; resize_main_images_for_dsrl becomes a no-op when
            # images are already at the target size.
            for traj in recv_list:
                for obs_dict in (traj.curr_obs, traj.next_obs):
                    if obs_dict and "main_images" in obs_dict:
                        img = obs_dict["main_images"]
                        if isinstance(img, torch.Tensor) and img.ndim >= 3:
                            orig_shape = img.shape
                            # Flatten leading dims so resize_main_images_for_dsrl
                            # sees a 4-D [B, H, W, C] or [B, C, H, W] tensor.
                            flat = img.reshape(-1, *orig_shape[-3:])
                            resized_dict = drq.resize_main_images_for_dsrl(
                                {"main_images": flat}, size=64
                            )
                            obs_dict["main_images"] = resized_dict["main_images"].reshape(
                                *orig_shape[:-3], *resized_dict["main_images"].shape[-3:]
                            )

        self.replay_buffer.add_trajectories(recv_list)

        if recv_list:
            self._last_rollout_num_transitions = max(
                self._trajectory_num_sac_transitions(traj) for traj in recv_list
            )

        if self.demo_buffer is not None:
            intervene_traj_list = []
            for traj in recv_list:
                assert isinstance(traj, Trajectory)
                intervene_trajs = traj.extract_intervene_traj()
                if intervene_trajs is not None:
                    intervene_traj_list.extend(intervene_trajs)

            if len(intervene_traj_list) > 0:
                self.demo_buffer.add_trajectories(intervene_traj_list)

    @staticmethod
    def _trajectory_num_sac_transitions(trajectory: Trajectory) -> int:
        if getattr(trajectory, "num_sac_transitions", 0) > 0:
            return int(trajectory.num_sac_transitions)
        if trajectory.rewards is not None:
            return int(trajectory.rewards.shape[0])
        if trajectory.curr_obs:
            first = next(iter(trajectory.curr_obs.values()))
            if isinstance(first, torch.Tensor) and first.dim() >= 1:
                return int(first.shape[0])
        return 0

    def _resolve_update_epoch(self) -> int:
        if self.cfg.algorithm.get("dynamic_update_epoch", False):
            multi_grad_step = int(self.cfg.algorithm.get("multi_grad_step", 20))
            num_transitions = int(getattr(self, "_last_rollout_num_transitions", 0))
            if num_transitions > 0:
                return num_transitions * multi_grad_step
        return int(self.cfg.algorithm.get("update_epoch", 1))

    def _dsrl_bootstrap_horizon(self, batch, use_action_chunking: bool) -> int:
        """Steps between Pi0 re-queries used for Bellman discounting."""
        if use_action_chunking:
            return int(batch["actions"].shape[1])
        return int(self.cfg.actor.model.get("num_action_chunks", 1))

    @Worker.timer("forward_critic")
    def forward_critic(self, batch):
        use_intra_chunking = self.cfg.algorithm.get("use_intra_chunking", False)
        if use_intra_chunking:
            return self._forward_critic_intra_chunking(batch)

        use_true_bellmann_reward = self.cfg.algorithm.get("use_true_bellmann_reward", False)
        if use_true_bellmann_reward:
            return self._forward_critic_true_bellmann_reward(batch)

        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")
        agg_q = self.cfg.algorithm.get("agg_q", "min")
        use_dsrl = self.cfg.actor.model.get("openpi", {}).get("use_dsrl", False)
        chunk_reward = self.cfg.algorithm.get("chunk_reward", False)
        use_action_chunking = self.cfg.actor.model.get("openpi", {}).get("use_action_chunking", False)
        if use_dsrl:
            self.action_horizon = self._dsrl_bootstrap_horizon(batch, use_action_chunking)
            discount = self.cfg.algorithm.gamma**self.action_horizon
            batch_dim = batch["actions"].shape[0]
            if chunk_reward:
                # self.action_horizon = batch["actions"].shape[1]
                batch_dim = batch["actions"].shape[0]
                # # Create mask: True until first termination, then False
                # mask = (~batch["terminations"]).cumprod(dim=1).bool()
                # # Apply mask
                # batch["rewards"] = batch["rewards"] * mask
                exponents = torch.arange(0, self.action_horizon).float() # [0, 1, 2, ..., H-1]
                gamma_powers = torch.pow(self.cfg.algorithm.gamma, exponents) #[gamma^0, gamma^1, ..., gamma^H-1], Shape: (H,)
                gamma_powers = gamma_powers.unsqueeze(0).expand(batch_dim, -1).to(self.device) # Shape: (B, H)
                rewards_for_bootstrap = (batch["rewards"] * gamma_powers).sum(dim=-1, keepdim=True).to(self.torch_dtype)
                # rewards_for_bootstrap = rewards_for_bootstrap / self.action_horizon # normalize
            else:
                # After loading batch["rewards"] and batch["terminations"]
                r = batch["rewards"][:, 0:1].clone()
                term = batch["terminations"].any(dim=-1, keepdim=True)
                # succ = term & (batch["terminations"].any(dim=-1, keepdim=True))  # refine if needed
                # Last success query: if terminated and reward at any step in chunk is 0:
                has_zero = (batch["rewards"] == 0).any(dim=-1, keepdim=True)
                r = torch.where(term & has_zero, torch.zeros_like(r), r)
                rewards_for_bootstrap = r
                # rewards_for_bootstrap = batch["rewards"][:, 0:1].to(self.torch_dtype)
        else:
            discount = self.cfg.algorithm.gamma
            rewards_for_bootstrap = (
                batch["rewards"].sum(dim=-1, keepdim=True).to(self.torch_dtype)
            )
        terminations = batch["terminations"].to(self.torch_dtype)

        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]

        with torch.no_grad():
            kwargs = {}
            if SupportedModel(self.cfg.actor.model.model_type) in [
                SupportedModel.OPENVLA,
                SupportedModel.OPENVLA_OFT,
            ]:
                kwargs["temperature"] = (
                    self.cfg.algorithm.sampling_params.temperature_train
                )
            if use_dsrl:
                kwargs["train"] = True
            next_state_actions, next_state_log_pi, shared_feature = self.model(
                forward_type=ForwardType.SAC, obs=next_obs, **kwargs
            )
            if next_state_log_pi.ndim == 1:
                next_state_log_pi = next_state_log_pi.unsqueeze(-1)
            next_state_log_pi = next_state_log_pi.sum(dim=-1, keepdim=True)
            if not use_crossq:
                dsrl_kwargs = {"train": True} if use_dsrl else {}
                all_qf_next_target = self.target_model(
                    forward_type=ForwardType.SAC_Q,
                    obs=next_obs,
                    actions=next_state_actions,
                    shared_feature=None,
                    **dsrl_kwargs,
                )
                if self.critic_subsample_size > 0:
                    sample_idx = torch.randint(
                        0,
                        all_qf_next_target.shape[-1],
                        (self.critic_subsample_size,),
                        generator=self.critic_sample_generator,
                        device=self.device,
                    )
                    all_qf_next_target = all_qf_next_target.index_select(
                        dim=-1, index=sample_idx
                    )

                if agg_q == "min":
                    qf_next_target, _ = torch.min(
                        all_qf_next_target, dim=1, keepdim=True
                    )
                elif agg_q == "mean":
                    qf_next_target = torch.mean(all_qf_next_target, dim=1, keepdim=True)

                if self.cfg.algorithm.get("backup_entropy", True):
                    qf_next_target = (
                        qf_next_target - self.entropy_temp.alpha * next_state_log_pi
                    )
                    qf_next_target = qf_next_target.to(dtype=self.torch_dtype)
                if bootstrap_type == "always":
                    target_q_values = (
                        rewards_for_bootstrap + discount * qf_next_target
                    )  # [bsz, 1]
                elif bootstrap_type == "standard":
                    target_q_values = (
                        rewards_for_bootstrap
                        + (~(terminations.any(dim=-1, keepdim=True)))
                        * discount
                        * qf_next_target
                    )  # [bsz, 1]
                else:
                    raise NotImplementedError(f"{bootstrap_type=} is not supported!")

        if not use_crossq:
            dsrl_kwargs = {"train": True} if use_dsrl else {}
            all_data_q_values = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=actions,
                **dsrl_kwargs,
            )
        else:
            all_data_q_values, all_qf_next = self.model(
                forward_type=ForwardType.CROSSQ_Q,
                obs=curr_obs,
                actions=actions,
                next_obs=next_obs,
                next_actions=next_state_actions,
            )

            all_qf_next = all_qf_next.detach()
            if agg_q == "min":
                qf_next, _ = torch.min(all_qf_next, dim=1, keepdim=True)
            elif agg_q == "mean":
                qf_next = torch.mean(all_qf_next, dim=1, keepdim=True)
            if self.cfg.algorithm.get("backup_entropy", True):
                qf_next = qf_next - self.entropy_temp.alpha * next_state_log_pi
                qf_next = qf_next.to(dtype=self.torch_dtype)

            if bootstrap_type == "always":
                target_q_values = rewards_for_bootstrap + discount * qf_next  # [bsz, 1]
            elif bootstrap_type == "standard":
                target_q_values = (
                    rewards_for_bootstrap
                    + (~(terminations.any(dim=-1, keepdim=True))) * discount * qf_next
                )  # [bsz, 1]
            else:
                raise NotImplementedError(f"{bootstrap_type=} is not supported!")

        # Align dtype: bool ops with Python floats promote to float32,
        # which can mismatch with bfloat16 model outputs.
        target_q_values = target_q_values.to(dtype=all_data_q_values.dtype)
        critic_loss = F.mse_loss(
            all_data_q_values, target_q_values.expand_as(all_data_q_values)
        )
        return critic_loss, {"q_data": all_data_q_values.mean().item(), "target_q_values": target_q_values.mean().item()}

    def _forward_critic_intra_chunking(self, batch):
        """Intra-chunk critic for **CriticGPT (Toperl)** using segment-style n-step returns.

        Batched like ``segments_n_step_return_vf`` (motion-primitive reference),
        adapted to one chunk per row: ``L`` env steps, ``curr_obs`` /
        ``intermediate_obs`` / ``next_obs``.

        - ``future_returns[b, k]``: ``k=0`` → ``agg_h Q_{tar}(s_0, ã)`` (V / MC
          bootstrap); ``k≥1`` → ``V_{tar}(s_k)`` from the context token at ``s_k``.
        - ``discount_seq[k]=γ^k``, masked rewards × discounts, strict lower-triangular
          sum over past columns → add ``future_returns * discount_seq`` →
          ``n_step_targets`` (same algebra as ``tril_discount_rewards.sum(-1) +
          discount_return`` in your snippet).

        Predictions: ``V_φ=[:,0,0]``, ``Q_φ`` prefix ``t`` = ``[:,−1,0]`` for
        ``t=1..L-1``.  Column ``0`` trains ``V`` vs ``n_step_targets[:,0]``; columns
        ``1..L-1`` train prefix Q with alive weights.

        Requires ``intermediate_obs``, DSRL, ``use_toperl_critic``.
        """
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        chunk_reward = self.cfg.algorithm.get("chunk_reward", False)
        if use_crossq:
            raise NotImplementedError(
                "use_intra_chunking with Cross-Q is not implemented."
            )
        if not self.use_dsrl:
            raise NotImplementedError(
                "use_intra_chunking currently requires DSRL (use_dsrl=True)."
            )

        openpi_cfg = self.cfg.actor.model.get("openpi", {})
        if not openpi_cfg.get("use_toperl_critic", False):
            raise NotImplementedError(
                "use_intra_chunking (paper loss) requires use_toperl_critic=True."
            )

        inter = batch.get("intermediate_obs")
        if not inter:
            raise ValueError(
                "use_intra_chunking requires batch['intermediate_obs']. "
                "Set rollout.collect_intermediate_obs: true."
            )

        gamma = float(self.cfg.algorithm.gamma)
        agg_q = self.cfg.algorithm.get("agg_q", "min")
        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")
        backup_entropy = self.cfg.algorithm.get("backup_entropy", True)

        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]
        rewards = batch["rewards"].to(self.torch_dtype)
        terminations = batch["terminations"]

        B, L, _ = actions.shape

        L_raw = actions.shape[1]        # number of actions
        L_total = L_raw + 1            # number of states
        N = L_raw                      # number of targets

        k_inter = None
        for _v in inter.values():
            if isinstance(_v, torch.Tensor) and _v.dim() >= 2:
                k_inter = int(_v.shape[1])
                break
        if k_inter is None:
            raise ValueError("intermediate_obs has no tensor leaves to read K from.")
        if k_inter != L - 1:
            raise ValueError(
                f"Expected intermediate_obs dim-1 K=L-1={L - 1} for L={L} chunk steps, "
                f"got K={k_inter}."
            )


        pi_kw: dict = {"train": True}

        dsrl_q = {"train": True}
        dsrl_prefix = {"train": True, "return_all_prefixes": True}
        dtype = self.torch_dtype
        device = self.device

        # ----- future_returns [B, N]: R_k bootstrap terms (no grad) -----
        future_returns = torch.zeros(B, N, device=device, dtype=dtype)
        with torch.no_grad():
            pi_a, log_pi, _ = self.model(
                forward_type=ForwardType.SAC,
                obs=curr_obs,
                **pi_kw,
            )
            if log_pi.ndim == 1:
                log_pi = log_pi.unsqueeze(-1)
            log_pi = log_pi.sum(dim=-1, keepdim=True)
            q_pi_tar = self.target_model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=pi_a,
                shared_feature=None,
                **dsrl_q,
            )
            if self.critic_subsample_size > 0 and q_pi_tar.shape[-1] > 1:
                idx = torch.randint(
                    0,
                    q_pi_tar.shape[-1],
                    (self.critic_subsample_size,),
                    generator=self.critic_sample_generator,
                    device=device,
                )
                q_pi_tar = q_pi_tar.index_select(dim=-1, index=idx)
            if agg_q == "min":
                q_pi_tar, _ = torch.min(q_pi_tar, dim=-1, keepdim=True)
            elif agg_q == "mean":
                q_pi_tar = torch.mean(q_pi_tar, dim=-1, keepdim=True)
            else:
                raise NotImplementedError(f"{agg_q=} is not supported!")
            if backup_entropy:
                q_pi_col0 = (q_pi_tar - self.entropy_temp.alpha * log_pi).squeeze(-1)
            else:
                q_pi_col0 = q_pi_tar.squeeze(-1)
            q_pi_col0 = q_pi_col0.to(dtype=dtype)

            # for j in range(1, L_total):
            #     q_tar_sj = self.target_model(
            #         forward_type=ForwardType.SAC_Q,
            #         obs=_obs_after_k(j),
            #         actions=actions,
            #         shared_feature=None,
            #         **dsrl_prefix,
            #     )
            #     vj = q_tar_sj[:, 0, 0]
            #     if bootstrap_type == "always":
            #         boot_j = torch.ones(bsz, device=device, dtype=dtype)
            #     elif bootstrap_type == "standard":
            #         boot_j = torch.prod((~terminations[:, :j]).to(dtype=dtype), dim=1)
            #     else:
            #         raise NotImplementedError(f"{bootstrap_type=} is not supported!")
            #     future_returns[:, j-1] = (vj * boot_j).to(dtype=dtype)
            # build obs sequence
            obs_seq = {}
            for k in curr_obs.keys():
                parts = [curr_obs[k].unsqueeze(1)]
                if k in inter:
                    parts.append(inter[k])
                parts.append(next_obs[k].unsqueeze(1))
                obs_seq[k] = torch.cat(parts, dim=1)   # [B, N+1, ...]

            # take s1..sN 
            obs_j = {k: v[:, 1:] for k, v in obs_seq.items()}  # [B, N, ...]

            #  flatten
            obs_flat = {
                k: v.reshape(B * N, *v.shape[2:])
                for k, v in obs_j.items()
            }

            # # repeat actions
            # actions_flat = actions.unsqueeze(1).expand(-1, N, -1, -1)
            # actions_flat = actions_flat.reshape(B * N, N, -1)

            q_tar_all = self.target_model(
                forward_type=ForwardType.SAC_Q,
                obs=obs_flat,
                actions=None,
                shared_feature=None,
                **dsrl_prefix,
            )

            v_tar_flat = q_tar_all[:, 0, :]          # [B*N, num_q_heads]
            if agg_q == "min":
                v_tar_flat = v_tar_flat.min(dim=-1).values
            else:
                v_tar_flat = v_tar_flat.mean(dim=-1)
            v_all = v_tar_flat.view(B, N)             # [B, N]
            if bootstrap_type == "always":
                boot = torch.ones_like(v_all)
            elif bootstrap_type == "standard":
                boot = torch.cumprod((~terminations).to(dtype), dim=1)
                # boot = torch.roll(boot, shifts=1, dims=1)
                # boot[:, 0] = 1.0, sagt Arvind, vllt lügt er auch
            else:
                raise NotImplementedError

            future_returns = (v_all * boot).to(dtype=dtype)

        # ----- N-step targets: tril discounted rewards + γ^k R_k -----
        rw_mask = (~terminations).cumprod(dim=1).bool()
        rw_mask = torch.roll(rw_mask, shifts=1, dims=1)
        rw_mask[:, 0] = True
        seg_r = (rewards * rw_mask.to(dtype=rewards.dtype)).to(dtype=dtype)

        discount_idx = torch.arange(L_total, device=device, dtype=torch.float32)
        discount_seq = torch.pow(
            torch.tensor(gamma, device=device, dtype=torch.float32), discount_idx
        ).to(dtype=dtype)

        seg_discount_r = seg_r * discount_seq[0:N]
        reward_tril_mask = torch.tril(
            torch.ones(N, N, device=device, dtype=dtype), diagonal=0
        )
        tril_discount = (
            seg_discount_r.unsqueeze(1).expand(-1, N, -1) * reward_tril_mask.view(1, N, N)
        ).sum(dim=-1)
        discount_return = future_returns * discount_seq[1:L_total] 
        if chunk_reward:
            n_step_targets = tril_discount + discount_return
        else:
            n_step_targets = rewards[:, 0:1] + discount_return

        # ----- online preds vs targets -----
        with self.worker_timer("forward_critic_sac_q_intra"):
            q_full = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=actions,
                **dsrl_prefix,
            )
        v_online = q_full[:, 0, :]                                         # [B, num_q_heads]
        v_target = q_pi_col0.unsqueeze(-1).expand_as(v_online)             # [B, 1] → [B, num_q_heads]
        v_loss = F.mse_loss(v_online, v_target.to(dtype=v_online.dtype))

        # Per prefix length t: batch-weighted MSE, then mean over t (equal weight per horizon).

        # for t in range(1, L_total):
        #     w_t = torch.prod((~terminations[:, : t]).to(dtype=dtype), dim=1)
        #     # w_t = _prefix_weight(t) # if the episode terminated before t, the weight is 0
        #     if not bool(w_t.any()):
        #         continue
        #     q_raw = self.model(
        #         forward_type=ForwardType.SAC_Q,
        #         obs=curr_obs,
        #         actions=actions[:, :t, :].contiguous(),
        #         **dsrl_prefix,
        #     )
        #     q_pred = q_raw[:, -1, 0].unsqueeze(-1) # last q-value
        #     target_t = n_step_targets[:, t-1].unsqueeze(-1)
        #     target_t = target_t.to(dtype=q_pred.dtype)
        #     td = F.mse_loss(q_pred, target_t, reduction="none")
        #     w = w_t.to(device=td.device, dtype=td.dtype)
        #     denom = w.sum().clamp_min(torch.tensor(1e-8, device=td.device, dtype=td.dtype))
        #     step_mean = (td * w.unsqueeze(-1)).sum() / denom
        #     prefix_step_means.append(step_mean)
        #     n_stats += 1
        #     q_sum += float(
        #         (q_pred.detach().squeeze(-1) * w).sum() / (w.sum() + 1e-8)
        #     )
        #     tgt_sum += float(
        #         (target_t.squeeze(-1) * w).sum() / (w.sum() + 1e-8)
        #     )
        

        # extract all prefix Q(s, a_{1:t})
        q_pred_all = q_full[:, 1:, :]   # [B, N, num_q_heads]

        # targets already aligned as [B, N]
        target_all = n_step_targets.to(dtype=q_pred_all.dtype)

        # w_all[k, i] = 0 once episode k has terminated before prefix step i
        w_all = torch.cumprod((~terminations).to(dtype=dtype), dim=1)  # [B, N]

        # L(ψ) = (1/N) Σ_i L_i,  L_i = (1/Σ_k w_{k,i}) Σ_k w_{k,i}*(Q_ψ(s,a_{1:i})-G^(i))²
        # gradient-level averaging: each horizon i contributes equally
        td_all = F.mse_loss(q_pred_all, target_all.unsqueeze(-1).expand_as(q_pred_all), reduction="none")  # [B, N, num_q_heads]
        if not chunk_reward:
            prefix_loss = td_all[:, -1, :].mean() #then it should be just like before
        # else:
        #     valid_steps = w_all.sum(dim=0) > 0                                       # [N]
        #     denom = w_all.sum(dim=0).clamp_min(1e-8)                                 # [N]
        #     step_means = (td_all * w_all.unsqueeze(-1)).sum(dim=0) / denom.unsqueeze(-1)  # [N, num_q_heads]
        #     step_means = step_means[valid_steps]                                     # [valid_N, num_q_heads]
        #     if step_means.numel() > 0:
        #         prefix_loss = step_means.mean()
        #     else:
        #         prefix_loss = torch.zeros((), device=device, dtype=dtype)
        # flat global-weighted mean (does NOT give equal weight per horizon):
        # denom = w_all.sum().clamp_min(1e-8)
        # prefix_loss = (td_all * w_all).sum() / denom
        # if prefix_step_means:
        #     prefix_loss = torch.stack(prefix_step_means).mean()
        # else:
        #     prefix_loss = torch.zeros((), device=device, dtype=dtype)
        # prefix_loss = td_all
        critic_loss = prefix_loss + v_loss
        # Weighted means over all valid (alive) prefixes for logging.
        total_weight = w_all.sum()
        if total_weight > 0:
            q_data = float((q_pred_all.detach().mean(dim=-1) * w_all).sum() / total_weight)
            target_q_values = float((target_all.detach() * w_all).sum() / total_weight)
        else:
            q_data = float(q_pred_all.detach().mean().item())
            target_q_values = float(target_all.detach().mean().item())
        metrics = {
            "q_data": q_data,
            "target_q_values": target_q_values,
            "value_data": float(v_online.mean().item()),
            "value_target": float(v_target.mean().item()),
            "intrachunk_prefix_steps": float(max(L_total - 1, 1)),
            "prefix_critic_loss": float(prefix_loss.item()),
            "v_align_critic_loss": float(v_loss.item()),
            "critic_loss": float(critic_loss.item()),
        }
        return critic_loss, metrics

    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        if "actor_agg_q" in self.cfg.algorithm:
            agg_q = self.cfg.algorithm["actor_agg_q"]
        else:
            agg_q = self.cfg.algorithm.get("agg_q", "min")

        curr_obs = batch["curr_obs"]
        kwargs = {}
        if self.cfg.actor.model.model_type in ["openvla", "openvla_oft"]:
            kwargs["temperature"] = self.cfg.algorithm.sampling_params.temperature_train
        if self.use_dsrl:
            kwargs["train"] = True
        pi, log_pi, shared_feature = self.model(
            forward_type=ForwardType.SAC, obs=curr_obs, **kwargs
        )
        if log_pi.ndim == 1:
            log_pi = log_pi.unsqueeze(-1)
        log_pi = log_pi.sum(dim=-1, keepdim=True)  # sum over the chunk dimension
        if not use_crossq:
            dsrl_kwargs = {"train": True} if self.use_dsrl else {}
            all_qf_pi = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=pi,
                shared_feature=None,
                detach_encoder=True,
                **dsrl_kwargs,
            )
        else:
            all_qf_pi, _ = self.model(
                forward_type=ForwardType.CROSSQ_Q,
                obs=curr_obs,
                actions=pi,
                next_obs=None,
                next_actions=None,
                shared_feature=None,
                detach_encoder=True,
            )
        metrics = {
            f"q_value_{q_id}": all_qf_pi[..., q_id].mean().item()
            for q_id in range(self.cfg.actor.model.get("num_q_heads", 1))
        }
        if agg_q == "min":
            qf_pi, _ = torch.min(all_qf_pi, dim=1, keepdim=True)
        elif agg_q == "mean":
            qf_pi = torch.mean(all_qf_pi, dim=1, keepdim=True)
        metrics["q_pi"] = qf_pi.mean().item()
        actor_loss = ((self.entropy_temp.alpha * log_pi) - qf_pi).mean()

        entropy = -log_pi.mean()
        return actor_loss, entropy, metrics, log_pi.detach()

    @Worker.timer("forward_alpha")
    def forward_alpha(self, batch, log_pi: Optional[torch.Tensor] = None):
        """Compute alpha loss.

        Args:
            batch: Training batch dict (used only when log_pi is not provided).
            log_pi: Pre-computed detached log-probabilities from the actor
                forward pass.  When supplied the model forward is skipped,
                saving a full SAC forward pass per update epoch.
        """
        if log_pi is None:
            curr_obs = batch["curr_obs"]
            with torch.no_grad():
                kwargs = {}
                if self.cfg.actor.model.model_type in ["openvla", "openvla_oft"]:
                    kwargs["temperature"] = (
                        self.cfg.algorithm.sampling_params.temperature_train
                    )
                if self.use_dsrl:
                    kwargs["train"] = True
                _, log_pi, _ = self.model(
                    forward_type=ForwardType.SAC, obs=curr_obs, **kwargs
                )
                if log_pi.ndim == 1:
                    log_pi = log_pi.unsqueeze(-1)
                log_pi = log_pi.sum(dim=-1, keepdim=True)

        alpha = self.entropy_temp.compute_alpha()
        alpha_loss = -alpha * (log_pi.mean() + self.target_entropy)
        return alpha_loss

    @Worker.timer("update_one_epoch")
    def update_one_epoch(self, train_actor: bool = True):
        global_batch_size_per_rank = (
            self.cfg.actor.global_batch_size // self._world_size
        )

        with self.worker_timer("sample"):
            global_batch = next(self.buffer_dataloader_iter)

        train_micro_batch_list = split_dict_to_chunk(
            global_batch,
            global_batch_size_per_rank // self.cfg.actor.micro_batch_size,
        )

        # debug_image_dir = os.path.join(
        #     self.cfg.runner.logger.log_path, "debug_images"
        # )
        # os.makedirs(debug_image_dir, exist_ok=True)
        # if train_micro_batch_list:
        #     batch = train_micro_batch_list[0]
        #     curr_main_images = batch["curr_obs"]["main_images"]
        #     next_main_images = batch["next_obs"]["main_images"]
        #     n_save = min(5, curr_main_images.shape[0])
        #     for i in range(n_save):
        #         torchvision.utils.save_image(
        #             _main_image_for_save(curr_main_images[i]),
        #             os.path.join(debug_image_dir, f"curr_main_images_{i}.png"),
        #         )
        #         torchvision.utils.save_image(
        #             _main_image_for_save(next_main_images[i]),
        #             os.path.join(debug_image_dir, f"next_main_images_{i}.png"),
        #         )
        with self.worker_timer("apply_augmentations"):
            base_seed = self.cfg.actor.get("seed", 1234) + self.update_step
            for i, batch in enumerate(train_micro_batch_list):
                batch = put_tensor_device(batch, device=self.device)
                if self.use_dsrl and (self.enable_drq or self.color_jitter):
                    batch["curr_obs"] = drq.resize_main_images_for_dsrl(batch["curr_obs"])
                    batch["next_obs"] = drq.resize_main_images_for_dsrl(batch["next_obs"])
                if self.enable_drq:
                    batch["curr_obs"] = drq.apply_drq(batch["curr_obs"], pad=4)
                    batch["next_obs"] = drq.apply_drq(batch["next_obs"], pad=4)
                if self.color_jitter:
                    curr_main_images = batch["curr_obs"]["main_images"]
                    curr_seed = base_seed + i * 2
                    # Keep images as float32 [0, 1] to avoid a uint8 round-trip
                    # conversion in the subsequent _preprocess_dsrl_images call.
                    batch["curr_obs"]["main_images"] = color_jitter.color_transform(
                        curr_main_images.to(torch.float32) / 255.0, seed=curr_seed
                    )
                    next_main_images = batch["next_obs"]["main_images"]
                    batch["next_obs"]["main_images"] = color_jitter.color_transform(
                        next_main_images.to(torch.float32) / 255.0,
                        seed=curr_seed + 1,
                    )
                train_micro_batch_list[i] = batch

        # for batch in train_micro_batch_list:
        #     curr_main_images = batch["curr_obs"]["main_images"]
        #     next_main_images = batch["next_obs"]["main_images"]
        #     n_save = min(5, curr_main_images.shape[0])
        #     for i in range(n_save):
        #         torchvision.utils.save_image(
        #             _main_image_for_save(curr_main_images[i]),
        #             os.path.join(debug_image_dir, f"curr_main_images_{i}_after_aug.png"),
        #         )
        #         torchvision.utils.save_image(
        #             _main_image_for_save(next_main_images[i]),
        #             os.path.join(debug_image_dir, f"next_main_images_{i}_after_aug.png"),
        #         )

        self.qf_optimizer.zero_grad()
        gbs_critic_loss = []
        all_critic_metrics = {}
        for batch in train_micro_batch_list:
            critic_loss, critic_metrics = self.forward_critic(batch)
            critic_loss = critic_loss / self.gradient_accumulation
            critic_loss.backward()
            gbs_critic_loss.append(critic_loss.item() * self.gradient_accumulation)
            append_to_dict(all_critic_metrics, critic_metrics)
        all_critic_metrics = {
            f"critic/{key}": np.mean(value) for key, value in all_critic_metrics.items()
        }
        qf_grad_norm = self.model.clip_grad_norm_(
            max_norm=self.cfg.actor.critic_optim.clip_grad
        )

        self.qf_optimizer.step()
        self.qf_lr_scheduler.step()

        metrics_data = {
            "sac/critic_loss": np.mean(gbs_critic_loss),
            "critic/lr": self.qf_optimizer.param_groups[0]["lr"],
            "critic/grad_norm": qf_grad_norm,
            **all_critic_metrics,
        }

        if self.update_step % self.critic_actor_ratio == 0 and train_actor:
            self.optimizer.zero_grad()
            gbs_actor_loss = []
            gbs_entropy = []
            all_actor_metrics = {}
            gbs_log_pi = []
            for batch in train_micro_batch_list:
                actor_loss, entropy, q_metrics, log_pi_detached = self.forward_actor(batch)
                actor_loss = actor_loss / self.gradient_accumulation
                actor_loss.backward()
                gbs_actor_loss.append(actor_loss.item() * self.gradient_accumulation)
                gbs_entropy.append(entropy.item())
                append_to_dict(all_actor_metrics, q_metrics)
                gbs_log_pi.append(log_pi_detached)
            all_actor_metrics = {
                f"actor/{key}": np.mean(value)
                for key, value in all_actor_metrics.items()
            }
            actor_grad_norm = self.model.clip_grad_norm_(
                max_norm=self.cfg.actor.optim.clip_grad
            )
            self.optimizer.step()
            self.lr_scheduler.step()

            # Update temperature parameter if using automatic entropy tuning
            gbs_alpha_loss = [0]
            alpha_grad_norm = 0
            if self.alpha_optimizer is not None:
                self.alpha_optimizer.zero_grad()
                gbs_alpha_loss = []
                for batch, log_pi_detached in zip(train_micro_batch_list, gbs_log_pi):
                    alpha_loss = self.forward_alpha(batch, log_pi=log_pi_detached) / self.gradient_accumulation
                    alpha_loss.backward()
                    gbs_alpha_loss.append(
                        alpha_loss.item() * self.gradient_accumulation
                    )
                torch.distributed.all_reduce(
                    self.entropy_temp.base_alpha.grad, op=torch.distributed.ReduceOp.AVG
                )
                alpha_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.entropy_temp.base_alpha,
                    self.cfg.algorithm.entropy_tuning.optim.clip_grad,
                )
                self.alpha_optimizer.step()
                self.alpha_lr_scheduler.step()

            # Collect metrics
            metrics_data.update(
                {
                    "sac/actor_loss": np.mean(gbs_actor_loss),
                    "sac/alpha_loss": np.mean(gbs_alpha_loss),
                    "sac/alpha": self.entropy_temp.alpha,
                    "actor/lr": self.optimizer.param_groups[0]["lr"],
                    "actor/grad_norm": actor_grad_norm,
                    "actor/entropy": np.mean(gbs_entropy),
                    "alpha/grad_norm": alpha_grad_norm,
                    **all_actor_metrics,
                }
            )
        # Soft update target network
        if (
            self.target_model_initialized
            and self.update_step % self.cfg.algorithm.get("target_update_freq", 1) == 0
        ):
            self.soft_update_target_model()

        return metrics_data

    def process_train_metrics(self, metrics):
        replay_buffer_stats = self.replay_buffer.get_stats()
        replay_buffer_stats = {
            f"replay_buffer/{key}": value for key, value in replay_buffer_stats.items()
        }
        append_to_dict(metrics, replay_buffer_stats)

        if self.demo_buffer is not None:
            demo_buffer_stats = self.demo_buffer.get_stats()
            demo_buffer_stats = {
                f"demo_buffer/{key}": value for key, value in demo_buffer_stats.items()
            }
            append_to_dict(metrics, demo_buffer_stats)
        # Average metrics across updates
        mean_metric_dict = {}
        for key, value in metrics.items():
            if isinstance(value, list) and len(value) > 0:
                # Convert tensor values to CPU and detach before computing mean
                cpu_values = []
                for v in value:
                    if isinstance(v, torch.Tensor):
                        cpu_values.append(v.detach().cpu().item())
                    else:
                        cpu_values.append(v)
                mean_metric_dict[key] = np.mean(cpu_values)
            else:
                # Handle single values
                if isinstance(value, torch.Tensor):
                    mean_metric_dict[key] = value.detach().cpu().item()
                else:
                    mean_metric_dict[key] = value

        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        return mean_metric_dict

    @Worker.timer("run_training")
    def run_training(self):
        """SAC training using replay buffer"""
        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

        # Check if replay buffer has enough samples
        min_buffer_size = self.cfg.algorithm.replay_buffer.get("min_buffer_size", 100)
        if not self.replay_buffer.is_ready(min_buffer_size):
            self.log_on_first_rank(
                f"Replay buffer size {len(self.replay_buffer)} < {min_buffer_size}, skipping training"
            )
            return {}

        # Delay actor training until buffer has enough samples
        train_actor_steps = self.cfg.algorithm.get("train_actor_steps", 0)
        train_actor_steps = max(min_buffer_size, train_actor_steps)
        train_actor = self.replay_buffer.is_ready(train_actor_steps)

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )
        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        self.model.train()
        metrics = {}

        update_epoch = self._resolve_update_epoch()
        metrics["sac/update_epoch"] = update_epoch
        if self.cfg.algorithm.get("dynamic_update_epoch", False):
            metrics["sac/rollout_num_transitions"] = int(
                getattr(self, "_last_rollout_num_transitions", 0)
            )
        for _ in range(update_epoch):
            metrics_data = self.update_one_epoch(train_actor=train_actor)
            append_to_dict(metrics, metrics_data)
            self.update_step += 1

        mean_metric_dict = self.process_train_metrics(metrics)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        if self.update_step % 20 == 0:   # only periodically
            torch.cuda.empty_cache()
        return mean_metric_dict

    def compute_advantages_and_returns(self):
        """
        SAC doesn't compute advantages/returns like PPO.
        This method is kept for compatibility but returns empty metrics.
        """
        return {}

    def save_checkpoint(self, save_base_path, step):
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
            self.is_weight_offloaded = False
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)
            self.is_optimizer_offloaded = False

        # Save model
        self._strategy.save_checkpoint(
            model=self.model,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            save_path=save_base_path,
            checkpoint_format="local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp",
        )

        # Save sac components
        # save alpha
        if self.alpha_optimizer is not None:
            alpha_save_path = os.path.join(save_base_path, "sac_components/alpha")
            self._strategy.save_checkpoint(
                model=self.entropy_temp,
                optimizers=self.alpha_optimizer,
                lr_schedulers=self.alpha_lr_scheduler,
                save_path=alpha_save_path,
                save_full_model_weights=False,
            )

        # save target model
        target_model_save_path = os.path.join(
            save_base_path, "sac_components/target_model"
        )
        os.makedirs(target_model_save_path, exist_ok=True)
        target_model_state_dict = self._strategy.get_model_state_dict(
            self.target_model, cpu_offload=False, full_state_dict=True
        )
        torch.save(
            target_model_state_dict,
            os.path.join(target_model_save_path, f"checkpoint_rank_{self._rank}.pt"),
        )

        # we don't save replay buffer to avoid OOM
        # # save replay buffer
        # buffer_save_path = os.path.join(
        #     save_base_path, f"sac_components/replay_buffer/rank_{self._rank}"
        # )
        # self.replay_buffer.save_checkpoint(buffer_save_path)

    def load_checkpoint(self, load_base_path):
        # load model
        self._strategy.load_checkpoint(
            model=self.model,
            optimizers=[self.optimizer, self.qf_optimizer],
            lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
            load_path=load_base_path,
            checkpoint_format="local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp",
        )

        # load alpha
        if self.alpha_optimizer is not None:
            alpha_load_path = os.path.join(load_base_path, "sac_components/alpha")
            self._strategy.load_checkpoint(
                model=self.entropy_temp,
                optimizers=self.alpha_optimizer,
                lr_schedulers=self.alpha_lr_scheduler,
                load_path=alpha_load_path,
            )

        # load target model
        target_model_load_path = os.path.join(
            load_base_path, "sac_components/target_model"
        )
        target_model_state_dict = torch.load(
            os.path.join(target_model_load_path, f"checkpoint_rank_{self._rank}.pt")
        )
        self._strategy.load_model_with_state_dict(
            self.target_model,
            target_model_state_dict,
            cpu_offload=False,
            full_state_dict=True,
        )

        # load replay buffer
        buffer_load_path = os.path.join(
            load_base_path, f"sac_components/replay_buffer/rank_{self._rank}"
        )
        self.replay_buffer.load_checkpoint(buffer_load_path)
