import os

import torch

from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class EmbodiedSACFSDPPolicyTest(EmbodiedSACFSDPPolicy):
    """Test variant with lightweight DSRL checkpointing."""

    def _should_use_dsrl_lightweight_checkpoint(self) -> bool:
        """Use lightweight checkpointing when running DSRL."""
        return bool(self.use_dsrl)

    @staticmethod
    def _is_dsrl_component_name(param_name: str) -> bool:
        dsrl_component_names = (
            "dsrl_action_noise_net",
            "actor_image_encoder",
            "actor_state_encoder",
            "critic_image_encoder",
            "critic_state_encoder",
            "q_head",
        )
        return any(component_name in param_name for component_name in dsrl_component_names)

    def _collect_dsrl_state_subset(self, model) -> dict[str, torch.Tensor]:
        """Collect DSRL-specific params/buffers from a wrapped model."""
        state_subset = {}
        for name, param in model.named_parameters():
            if self._is_dsrl_component_name(name):
                state_subset[name] = param.detach().cpu()
        for name, buffer in model.named_buffers():
            if self._is_dsrl_component_name(name):
                state_subset[name] = buffer.detach().cpu()
        return state_subset

    def _load_dsrl_state_subset(self, model, state_subset: dict[str, torch.Tensor]) -> None:
        """Load DSRL-specific params/buffers into a wrapped model."""
        param_dict = dict(model.named_parameters())
        buffer_dict = dict(model.named_buffers())
        loaded_keys = 0

        for name, tensor in state_subset.items():
            if name in param_dict:
                target = param_dict[name]
                target.data.copy_(tensor.to(device=target.device, dtype=target.dtype))
                loaded_keys += 1
            elif name in buffer_dict:
                target = buffer_dict[name]
                target.data.copy_(tensor.to(device=target.device, dtype=target.dtype))
                loaded_keys += 1

        if loaded_keys == 0:
            raise RuntimeError(
                "No DSRL checkpoint keys matched current model parameters/buffers."
            )

    def _save_dsrl_lightweight_checkpoint(self, save_base_path: str) -> None:
        """Save only DSRL trainable modules + SAC states to reduce checkpoint footprint."""
        dsrl_ckpt_root = os.path.join(save_base_path, "sac_components/dsrl_lightweight")
        model_save_path = os.path.join(dsrl_ckpt_root, "model")
        target_model_save_path = os.path.join(dsrl_ckpt_root, "target_model")
        optim_save_path = os.path.join(dsrl_ckpt_root, "optim")
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(target_model_save_path, exist_ok=True)
        os.makedirs(optim_save_path, exist_ok=True)

        model_state_subset = self._collect_dsrl_state_subset(self.model)
        target_model_state_subset = self._collect_dsrl_state_subset(self.target_model)
        torch.save(
            model_state_subset,
            os.path.join(model_save_path, f"checkpoint_rank_{self._rank}.pt"),
        )
        torch.save(
            target_model_state_subset,
            os.path.join(target_model_save_path, f"checkpoint_rank_{self._rank}.pt"),
        )

        optim_state = {
            "optimizer": self.optimizer.state_dict(),
            "qf_optimizer": self.qf_optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "qf_lr_scheduler": self.qf_lr_scheduler.state_dict(),
            "update_step": self.update_step,
        }
        torch.save(
            optim_state,
            os.path.join(optim_save_path, f"checkpoint_rank_{self._rank}.pt"),
        )

        if torch.distributed.get_rank() == 0:
            meta = {"format": "dsrl_lightweight_v1"}
            torch.save(meta, os.path.join(dsrl_ckpt_root, "meta.pt"))
        torch.distributed.barrier()

    def _load_dsrl_lightweight_checkpoint(self, load_base_path: str) -> None:
        """Load DSRL lightweight checkpoint."""
        dsrl_ckpt_root = os.path.join(load_base_path, "sac_components/dsrl_lightweight")
        model_load_path = os.path.join(dsrl_ckpt_root, "model")
        target_model_load_path = os.path.join(dsrl_ckpt_root, "target_model")
        optim_load_path = os.path.join(dsrl_ckpt_root, "optim")

        model_state_subset = torch.load(
            os.path.join(model_load_path, f"checkpoint_rank_{self._rank}.pt"),
            map_location="cpu",
        )
        self._load_dsrl_state_subset(self.model, model_state_subset)

        target_model_state_subset = torch.load(
            os.path.join(target_model_load_path, f"checkpoint_rank_{self._rank}.pt"),
            map_location="cpu",
        )
        self._load_dsrl_state_subset(self.target_model, target_model_state_subset)

        optim_state = torch.load(
            os.path.join(optim_load_path, f"checkpoint_rank_{self._rank}.pt"),
            map_location="cpu",
        )
        self.optimizer.load_state_dict(optim_state["optimizer"])
        self.qf_optimizer.load_state_dict(optim_state["qf_optimizer"])
        self.lr_scheduler.load_state_dict(optim_state["lr_scheduler"])
        self.qf_lr_scheduler.load_state_dict(optim_state["qf_lr_scheduler"])
        self.update_step = optim_state.get("update_step", self.update_step)
        torch.distributed.barrier()

    def save_checkpoint(self, save_base_path, step):
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
            self.is_weight_offloaded = False
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)
            self.is_optimizer_offloaded = False

        if self._should_use_dsrl_lightweight_checkpoint():
            self._save_dsrl_lightweight_checkpoint(save_base_path)
        else:
            self._strategy.save_checkpoint(
                model=self.model,
                optimizers=[self.optimizer, self.qf_optimizer],
                lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
                save_path=save_base_path,
                checkpoint_format="local_shard"
                if self.cfg.actor.fsdp_config.use_orig_params
                else "dcp",
            )

        if self.alpha_optimizer is not None:
            alpha_save_path = os.path.join(save_base_path, "sac_components/alpha")
            self._strategy.save_checkpoint(
                model=self.entropy_temp,
                optimizers=self.alpha_optimizer,
                lr_schedulers=self.alpha_lr_scheduler,
                save_path=alpha_save_path,
                save_full_model_weights=False,
            )

        if not self._should_use_dsrl_lightweight_checkpoint():
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

        buffer_save_path = os.path.join(
            save_base_path, f"sac_components/replay_buffer/rank_{self._rank}"
        )
        self.replay_buffer.save_checkpoint(buffer_save_path)

    def load_checkpoint(self, load_base_path):
        dsrl_meta_path = os.path.join(
            load_base_path, "sac_components/dsrl_lightweight/meta.pt"
        )
        if self._should_use_dsrl_lightweight_checkpoint() and os.path.exists(
            dsrl_meta_path
        ):
            self._load_dsrl_lightweight_checkpoint(load_base_path)
        else:
            self._strategy.load_checkpoint(
                model=self.model,
                optimizers=[self.optimizer, self.qf_optimizer],
                lr_schedulers=[self.lr_scheduler, self.qf_lr_scheduler],
                load_path=load_base_path,
                checkpoint_format="local_shard"
                if self.cfg.actor.fsdp_config.use_orig_params
                else "dcp",
            )

        if self.alpha_optimizer is not None:
            alpha_load_path = os.path.join(load_base_path, "sac_components/alpha")
            self._strategy.load_checkpoint(
                model=self.entropy_temp,
                optimizers=self.alpha_optimizer,
                lr_schedulers=self.alpha_lr_scheduler,
                load_path=alpha_load_path,
            )

        if not (
            self._should_use_dsrl_lightweight_checkpoint() and os.path.exists(dsrl_meta_path)
        ):
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

        buffer_load_path = os.path.join(
            load_base_path, f"sac_components/replay_buffer/rank_{self._rank}"
        )
        self.replay_buffer.load_checkpoint(buffer_load_path)
