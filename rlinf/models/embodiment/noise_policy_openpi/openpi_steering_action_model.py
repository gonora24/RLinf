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

from glob import glob
import math
import os
from collections.abc import Sequence
from dataclasses import  dataclass
from dataclasses import field
from typing import Any, Literal

import jax
from matplotlib import transforms
import numpy as np
from openpi.shared import download
import safetensors
import torch
import torch.nn as nn
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.gaussian_policy import GaussianTanhPolicy
from rlinf.models.embodiment.modules.q_head import QHead
from rlinf.models.embodiment.modules.resnet_utils import ResNetEncoder
from rlinf.models.embodiment.modules.utils import init_mlp_weights, make_mlp
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.openpi_action_model import OpenPi0Config


@dataclass(frozen=False)
class NoisePolicyConfig:
    # TODO: clean
    # config for rl
    # config_name: str = "pi0_libero"  # pi0_libero, pi05_libero, pi0_maniskill, pi05_maniskill, pi0_metaworld, pi05_metaworld
    num_images_in_input: int = 2  # number of images in input
    encoder_config: dict[str, Any] = field(default_factory=dict)
    image_size: list[int] = field(default_factory=list)
    action_dim: int = 4
    # State projector
    state_dim: int = 29
    state_latent_dim: int = 64
    # models
    chunk_critic: bool = False  # critic for action chunk or one action
    action_critic_hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
    noise_critic_hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
    policy_hidden_dims: list[int] = field(default_factory=lambda: [128, 128, 128])
    noise_dim: int = 32 # dimension of one noise action which is denoised

    # hyper-parameters
    action_chunk: int = 5  # action chunk
    action_env_dim: int = 7  # for environment action dim
    # training config
    safe_get_logprob: bool = False
    joint_logprob: bool = False  # designed for flow-noise
    double_layer: bool = False  # designed for flow-sde without acceleration
    ignore_last: bool = False  # ignore the last action for noise injection
    # critic
    detach_critic_input: bool = False  # detach critic input with the action expert
    chunk_critic_input: bool = False  # use only the action chunk for critic estimation
    value_after_vlm: bool = False  # value after vlm, pi05 mode
    value_vlm_mode: str = "mean_token"  # last_token, mean_token, first_token
    openpi: OpenPi0Config = field(default_factory=OpenPi0Config)
    
    def update_from_dict(self, config_dict):
        for key, value in config_dict.items():
            if hasattr(self, key):
                self.__setattr__(key, value)
        self._update_info()

    def _update_info(self):
        assert self.encoder_config['model_path'] is not None, "Please specify the encoder model_path."
        assert "ckpt_name" in self.encoder_config, (
            "Please specify the ckpt_name in encoder_config to load pretrained encoder weights."
        )
        ckpt_path = os.path.join(self.encoder_config['model_path'], self.encoder_config["ckpt_name"])
        assert os.path.exists(ckpt_path), (
            f"Pretrained encoder weights not found at {ckpt_path} with model path {self.encoder_config['model_path']} and encoder ckpt name {self.encoder_config['ckpt_name']}"
        )
        self.encoder_config["ckpt_path"] = ckpt_path


class NoisePolicyForOpenPI(BasePolicy):
    """
    Noise policy model for reinforcement learning action prediction.
    """

    # config: NoisePolicyConfig

    # Tells fsdp what not to split when doing auto-wrapping, do we need it since we're not training the openpi model?
    # TODO
    @property
    def _no_split_modules(self) -> list[str]:
        no_split_modules = [
                "GemmaMLP",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
            ]
        return no_split_modules

    @property
    def _no_split_names(self) -> list[str]:
        return [
            "action_in_proj",
            "action_out_proj",
            "lm_head",
            # --pi0 only--
            "state_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
            # --pi05 only--
            "time_mlp_in",
            "time_mlp_out",
        ]

    def __init__(self, config: NoisePolicyConfig, openpi_model: PI0Pytorch):
        # Override `sample_actions` to prevent parent class polymorphic call
        sample_actions_func = self.sample_actions
        super().__init__()
        self.sample_actions = sample_actions_func
        self.global_step = 0
        self.config = config
        self.openpi = openpi_model
        #TODO: was ist hier der Input fuer den critic, welche encoder werden gebraucht
        # Image encoders (one per camera view)
        self.encoders = nn.ModuleList()
        encoder_out_dim = 0
        sample_x = torch.randn(1, *self.config.image_size)
        for img_id in range(self.config.num_images_in_input):
            self.encoders.append(
                ResNetEncoder(
                    sample_x, out_dim=256, encoder_cfg=self.config.encoder_config
                )
            )
            encoder_out_dim += self.encoders[img_id].out_dim

        # State encoder
        self.state_proj = nn.Sequential(
            *make_mlp(
                in_channels=self.config.state_dim,
                mlp_channels=[
                    self.config.state_latent_dim,
                ],
                act_builder=nn.Tanh,
                last_act=True,
                use_layer_norm=True,
            )
        )
        init_mlp_weights(self.state_proj, nonlinearity="tanh")

        # Action space critic
        if self.config.chunk_critic:
            action_feature_dim = self.config.action_chunk * self.config.action_env_dim
            noise_dim = self.config.noise_dim * self.config.action_chunk
        else:
            action_feature_dim = self.config.action_env_dim
            noise_dim = self.config.noise_dim

        self.action_critic = QHead(hidden_size=encoder_out_dim + self.config.state_latent_dim, 
                                   action_feature_dim=action_feature_dim, 
                                   hidden_dims=self.config.action_critic_hidden_dims,
                                   output_dim=1,
                                   train_action_encoder=False)
        # Noise space critic
        self.noise_critic = QHead(hidden_size=encoder_out_dim + self.config.state_latent_dim, 
                                  action_feature_dim=noise_dim, 
                                  hidden_dims=self.config.noise_critic_hidden_dims, 
                                  output_dim=1,
                                  train_action_encoder=False)
        # Noise policy head
        self.noise_policy = GaussianTanhPolicy(
            obs_dim=encoder_out_dim + self.config.state_latent_dim,
            action_dim=self.config.action_dim,
            hidden_dims=self.config.policy_hidden_dims,
        )

        # setup openpi (otherwise no method access)
        # self.setup_openpi(openpi_model, data_config, norm_stats) 
        # Freeze OpenPI parameters
        # for p in self.openpi.parameters():
        #     p.requires_grad = False
        # self.openpi.eval()

        for name, module in self.named_modules():
            # Set _fsdp_wrap_name to the last part of the path (e.g., "model.action_in_proj" -> "action_in_proj")
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

    def setup_openpi(self, model: PI0Pytorch, data_config, norm_stats):

        # wrappers
        repack_transforms = transforms.Group()
        default_prompt = None

        model.setup_wrappers(
            transforms=[
                *repack_transforms.inputs,
                transforms.InjectDefaultPrompt(default_prompt),
                *data_config.data_transforms.inputs,
                transforms.Normalize(
                    norm_stats, use_quantiles=data_config.use_quantile_norm
                ),
                *data_config.model_transforms.inputs,
            ],
            output_transforms=[
                *data_config.model_transforms.outputs,
                transforms.Unnormalize(
                    norm_stats, use_quantiles=data_config.use_quantile_norm
                ),
                *data_config.data_transforms.outputs,
                *repack_transforms.outputs,
            ],
        )

    def set_global_step(self, global_step):
        self.global_step = global_step

    def setup_wrappers(
        self,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
    ):
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)

    def input_transform(self, obs: dict, transpose=True):
        inputs = jax.tree.map(lambda x: x, obs)
        # process input
        first_process = "prompt" in inputs.keys()
        if first_process:
            inputs.pop("prompt")
        else:
            inputs = {key: inputs[key] for key in inputs.keys() if "/" in key}

        # tensor -> numpy
        inputs = jax.tree.map(
            lambda x: np.asarray(x.detach().cpu()) if torch.is_tensor(x) else x, inputs
        )
        batch_size = next(v.shape[0] for v in inputs.values() if hasattr(v, "shape"))
        # split & transform
        transformed_samples = []
        for i in range(batch_size):
            sample = jax.tree.map(lambda x: x[i], inputs)
            if transpose:
                # convert from [3,256,256] -> [256,256,3]
                sample = jax.tree.map(
                    lambda x: x.transpose(1, 2, 0)
                    if len(x.shape) == 3 and transpose
                    else x,
                    sample,
                )
            else:
                sample = jax.tree.map(lambda x: x if len(x.shape) == 3 else x, sample)
            if first_process:
                sample["prompt"] = obs["prompt"][i]
            else:
                sample["prompt"] = "xxxx"
            transformed_sample = self._input_transform(sample)
            transformed_samples.append(transformed_sample)
        # recombine
        inputs = jax.tree.map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            *transformed_samples,
        )
        # inputs = jax.tree.map(lambda *x: torch.stack(x, axis=0), inputs)
        if not first_process:
            inputs["tokenized_prompt"] = obs["tokenized_prompt"]
            inputs["tokenized_prompt_mask"] = obs["tokenized_prompt_mask"]
        return inputs

    def output_transform(self, outputs):
        # split & transform
        batch_size = outputs["actions"].shape[0]
        transformed_samples = []
        for i in range(batch_size):
            sample = jax.tree.map(lambda x: np.asarray(x[i].detach().cpu()), outputs)
            sample = self._output_transform(sample)
            transformed_samples.append(sample)
        # recombine
        outputs = jax.tree.map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            *transformed_samples,
        )
        outputs["actions"] = outputs["actions"][:, : self.config.action_chunk]
        return outputs
    
    def get_feature(self, obs):
        """Extract features from observations (images + states)"""
        visual_features = []
        # from image_keys to image_num
        for img_id in range(self.cfg.image_num):
            if img_id == 0:
                images = obs["main_images"]
            else:
                images = obs["extra_view_images"][:, img_id - 1]
            if images.shape[3] == 3:
                # [B, H, W, C] -> [B, C, H, W]
                images = images.permute(0, 3, 1, 2)
            visual_features.append(self.encoders[img_id](images))
        visual_feature = torch.cat(visual_features, dim=-1)

        state_feature = self.state_proj(obs["states"])
        full_feature = torch.cat([visual_feature, state_feature], dim=-1)

        return full_feature, visual_feature

    def forward(self, forward_type=ForwardType.DEFAULT, next_obs=None, **kwargs):
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        elif forward_type == ForwardType.SAC:
            return self.sac_forward_action_space(next_obs=next_obs, **kwargs)
        elif forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(next_obs=next_obs, **kwargs)
        elif forward_type == ForwardType.DISTILL_Q:
            return self.sac_q_noise_forward(curr_obs=next_obs, **kwargs)
        elif forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError

    def sft_forward(self, data, **kwargs):
        observation = data["observation"]
        actions = data["actions"]
        return super().forward(observation, actions)
    
    
    def sac_forward_action_space(self, next_obs, **kwargs):
        full_feature, visual_features = self.get_feature(next_obs)
        noise = self.noise_policy(visual_features)
        actions, log_probs = self.openpi.sample_actions(device=next_obs["states"].device, observation=next_obs, noise=noise) 
        return actions, log_probs
    
    def sac_q_action_forward(self, next_obs, **kwargs):
        full_feature, visual_features = self.get_feature(next_obs)
        noise = self.noise_policy(visual_features)
        actions, _ = self.openpi.sample_actions(device=next_obs["states"].device, observation=next_obs, noise=noise) 
        if self.config.chunk_critic_input:
            critic_action_input = actions[:, : self.config.action_chunk * self.config.action_env_dim]
        else:
            critic_action_input = actions
        q_value_action = self.action_critic(full_feature, critic_action_input)
        return q_value_action
    
    def sac_q_noise_forward(self, curr_obs, **kwargs):
        full_feature, visual_features = self.get_feature(curr_obs)
        noise = self.noise_policy(visual_features)
        actions, _ = self.openpi.sample_actions(device=curr_obs["states"].device, observation=curr_obs, noise=noise) 
        current_q_values = self.action_critic(visual_features, actions) # visual or full here?
        q_values_noise = self.noise_critic(visual_features, noise)
        return current_q_values, q_values_noise

    def default_forward(
        self,
        data: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        # get kwargs
        compute_values = kwargs.get("compute_values", False)
        chains = data["chains"]
        denoise_inds = data["denoise_inds"]
        # input transform
        observation = self.input_transform(data, transpose=False)
        observation = _model.Observation.from_dict(observation)
        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )
        # transfer to device
        device = chains.device
        images = [img.to(device) for img in images]
        img_masks = [img_mask.to(device) for img_mask in img_masks]
        state = state.to(device)
        # get log prob
        log_probs, value_t, value_noise_t, entropy = self.get_log_prob_value(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            chains,
            denoise_inds,
            compute_values,
        )
        log_probs = log_probs[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        entropy = entropy[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        # post process
        log_probs = log_probs.mean(dim=1)
        entropy = entropy.mean(dim=[1, 2, 3], keepdim=False)[
            :, None
        ]  # [:,None] to align with loss-mask shape
        value_t = value_t.mean(dim=-1, keepdim=False)
        value_noise_t = value_noise_t.mean(dim=-1, keepdim=False)
        return {
            "logprobs": log_probs,
            "values": value_t,
            "values_noise": value_noise_t,
            "entropy": entropy,
        }

    def obs_processor(self, env_obs):
        # base observation
        processed_obs = {
            "observation/image": env_obs["main_images"],
            "prompt": env_obs["task_descriptions"],
        }
        # state observation
        if "calvin" in self.config.openpi.config_name:
            state = env_obs["states"]
            processed_obs["observation/state_ee_pos"] = state[:, :3]
            processed_obs["observation/state_ee_rot"] = state[:, 3:6]
            processed_obs["observation/state_gripper"] = state[:, 6:7]
        else:
            processed_obs["observation/state"] = env_obs["states"]
        # wrist image observation
        if env_obs["wrist_images"] is not None:
            processed_obs["observation/wrist_image"] = env_obs["wrist_images"]
        # store used keys
        return processed_obs

    def precision_processor(self, processed_obs):
        device = next(self.parameters()).device
        for key, value in processed_obs.items():
            if isinstance(value, list):
                processed_obs[key] = [
                    item.to(device=device).contiguous()
                    if torch.is_tensor(item)
                    else item
                    for item in value
                ]
            elif torch.is_tensor(value):
                processed_obs[key] = value.to(device=device).contiguous()
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    processed_obs[key][sub_key] = sub_value.to(
                        device=device
                    ).contiguous()
        return processed_obs

    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        compute_values=True,
        return_obs=True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        to_process_obs = self.obs_processor(env_obs)  # env obs -> policy input obs
        processed_obs = self.input_transform(
            to_process_obs, transpose=False
        )  # policy input obs -> model input obs
        processed_obs = self.precision_processor(
            processed_obs
        )  # obs precision processor
        observation = _model.Observation.from_dict(processed_obs)
        outputs = self.sample_actions(
            observation, mode=mode, compute_values=compute_values
        )
        actions = self.output_transform(
            {"actions": outputs["actions"], "state": observation.state}
        )["actions"].numpy()

        forward_inputs = {
            "chains": outputs["chains"],
            "denoise_inds": outputs["denoise_inds"],
            "observation/image": env_obs["main_images"],
            "observation/state": env_obs["states"],
            "tokenized_prompt": processed_obs["tokenized_prompt"],
            "tokenized_prompt_mask": processed_obs["tokenized_prompt_mask"],
        }
        if env_obs["wrist_images"] is not None:
            forward_inputs["observation/wrist_image"] = env_obs["wrist_images"]
        forward_inputs.update(to_process_obs)
        forward_inputs.pop("prompt", None)

        result = {
            "prev_logprobs": outputs["prev_logprobs"],
            "prev_values": outputs["prev_values"],
            "prev_values_noise": outputs.get("prev_values_noise"),
            "forward_inputs": forward_inputs,
        }
        return actions, result

    @torch.no_grad()
    def sample_actions(
        self,
        observation: _model.Observation,
        noise=None,
        mode="train",
        compute_values=True,
    ) -> torch.Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        device = observation.state.device
        num_steps = self.config.num_steps
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        # images, img_masks, lang_tokens, lang_masks, state = (
        #     self._preprocess_observation(observation, train=False)
        # )

        # prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
        #     images, img_masks, lang_tokens, lang_masks
        # )
        # prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        # prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # # Compute image and language key value cache
        # prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        # self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        # (prefix_output, _), past_key_values = self.paligemma_with_expert.forward(
        #     attention_mask=prefix_att_2d_masks_4d,
        #     position_ids=prefix_position_ids,
        #     past_key_values=None,
        #     inputs_embeds=[prefix_embs, None],
        #     use_cache=True,
        # )

        # compute policy-predicted noise chunk (flattened)
        full_feature, visual_features = self.get_feature(observation)
        noise_chunk = self.noise_policy(visual_features)
        actions, log_probs = self.openpi.sample_actions(device=observation.device, observation=observation, noise=noise_chunk)

        # Compute values for RL 
        # qs_action = self.action_critic(visual_features, actions)  
        qs_action = torch.zeros((bsize), device=device)

        # Placeholder logprobs (parent's sample_actions doesn't return them by default)
        # log_probs = torch.zeros(
        #     (bsize, self.config.action_chunk, self.config.action_env_dim),
        #     device=device,
        # )

        # chains: [initial noise, denoised actions]
        chains = torch.stack([noise_chunk, actions], dim=1)
        denoise_inds = torch.zeros((bsize, 1), dtype=torch.long, device=device)

        return {
            "actions": actions,
            "chains": chains,
            "prev_logprobs": log_probs,
            "prev_values": qs_action,
            "denoise_inds": denoise_inds,
        }

    def sample_mean_var_val(
        self,
        x_t,
        idx,
        state,
        prefix_pad_masks,
        past_key_values,
        mode,
        denoise_steps,
        compute_values=True,
    ):
        """
        Sample the mean, variance and value of the action at a given timestep.
        Rollout sample (idx is int) and actor get_log_prob_value (idx is tensor) will load this function.
        """
        # expand the shape
        bsize = state.shape[0]
        device = state.device
        if isinstance(idx, int):
            idx = torch.tensor(idx).expand(bsize)
        # build parameters
        if self.config.noise_anneal:
            # noise annealing
            noise_start, noise_end, anneal_steps = self.config.noise_params
            noise_level = (
                noise_start
                + (noise_end - noise_start)
                * min(self.global_step, anneal_steps)
                / anneal_steps
            )
            noise_level = torch.tensor(noise_level).to(device)
        else:
            # fixed noise level
            noise_level = torch.tensor(self.config.noise_level).to(device)
        timesteps = torch.linspace(1, 1 / denoise_steps, denoise_steps, device=device)
        timesteps = torch.cat([timesteps, torch.tensor([0.0], device=device)])
        # input parameters
        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]
        # velocity prediction
        suffix_out = self.get_suffix_out(
            state,
            prefix_pad_masks,
            past_key_values,
            x_t,
            t_input,
        )
        v_t = self.action_out_proj(suffix_out)  # [bs,n_action_steps,max_action_dim]
        # value prediction (action-space)
        if (
            self.config.add_value_head
            and compute_values
            and not self.config.value_after_vlm
        ):
            # use chunk critic input
            if self.config.chunk_critic_input:
                suffix_out_value = torch.mean(
                    suffix_out[:, : self.config.action_chunk], dim=1, keepdim=False
                )
            else:
                suffix_out_value = torch.mean(suffix_out, dim=1, keepdim=False)
            # detach critic input
            if self.config.detach_critic_input:
                suffix_out_value = suffix_out_value.detach()
            value_t = self.value_head(suffix_out_value)[:, 0]
        else:
            value_t = torch.zeros((bsize), device=device)

        # noise-space value prediction: operate on flattened noise chunk
        if self.config.add_value_head and compute_values:
            noise_feat = x_t[:, : self.config.action_chunk, : self.config.action_env_dim]
            noise_feat_flat = noise_feat.reshape(bsize, -1)
            if self.config.detach_critic_input:
                noise_feat_flat = noise_feat_flat.detach()
            noise_value_t = self.noise_value_head(noise_feat_flat)[:, 0]
        else:
            noise_value_t = torch.zeros((bsize), device=device)
        # ode sde mix sampling
        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)
        if mode == "eval":
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = torch.zeros_like(t_input)
        elif mode == "train":
            if self.config.noise_method == "flow_sde":
                sigmas = (
                    noise_level
                    * torch.sqrt(
                        timesteps
                        / (1 - torch.where(timesteps == 1, timesteps[1], timesteps))
                    )[:-1]
                )
                sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)
                x_t_std = torch.sqrt(delta) * sigma_i
            elif self.config.noise_method == "flow_cps":
                pi = torch.pi
                cos_term = torch.cos(pi * noise_level / 2).to(device)
                sin_term = torch.sin(pi * noise_level / 2).to(device)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = (t_input - delta) * cos_term
                x_t_std = (t_input - delta) * sin_term
            elif self.config.noise_method == "flow_noise":
                x0_weight = 1 - (t_input - delta)
                x1_weight = t_input - delta
                x_t_std = self.noise_head(suffix_out)
            else:
                raise ValueError(f"Invalid noise method: {self.config.noise_method}")
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std, value_t, noise_value_t

    def get_suffix_out(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(state, x_t, timestep)
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = (
            "eager"  # noqa: SLF001
        )

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return suffix_out

    # TODO: to check potential nan here
    def get_logprob_norm(self, sample, mu, sigma):
        # logprob = log p(x|mu,sigma) = -log(sigma) - 0.5 * log(2 * pi) - 0.5 * ((x - mu) / sigma) ** 2
        if self.config.safe_get_logprob:
            log_prob = -torch.pow((sample - mu), 2)
        else:
            mask = sigma == 0
            sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
            constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
                2 * torch.pi * torch.ones_like(sample)
            )
            exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
            log_prob = constant_term + exponent_term
            log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
        return log_prob

    def preprocess_for_train(self, data):
        return data

    def get_log_prob_value(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        chains,
        denoise_inds,
        compute_values=False,
    ):
        bsize = state.shape[0]
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        # Compute image and language key value cache
        [prefix_output, _], past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        chains_log_probs = []
        chains_values = []
        chains_entropy = []

        # get log prob
        if self.config.joint_logprob:
            num_steps = self.config.num_steps
            initial_log_prob = self.get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
            )
            initial_entropy = self.gaussian_entropy(torch.ones_like(chains[:, 0]))
            chains_log_probs.append(initial_log_prob)
            chains_entropy.append(initial_entropy)
        else:
            num_steps = 1
        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[torch.arange(bsize), denoise_ind]
            chains_next = chains[torch.arange(bsize), denoise_ind + 1]
            x_t_mean, x_t_std, value_t, noise_value_t = self.sample_mean_var_val(
                chains_pre,
                denoise_ind,
                state,
                prefix_pad_masks,
                past_key_values,
                "train",
                self.config.num_steps,
                compute_values,
            )
            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            entropy = self.gaussian_entropy(x_t_std)
            chains_log_probs.append(log_probs)
            chains_entropy.append(entropy)
            if not self.use_vlm_value:
                # store both action- and noise-space values
                chains_values.append(value_t)
                # create separate list for noise values if not present
                if 'chains_values_noise' not in locals():
                    chains_values_noise = []
                chains_values_noise.append(noise_value_t)
        if self.use_vlm_value:
            chains_values.append(self.get_value_from_vlm(prefix_output))
            # when using VLM-based value, set noise-values to zeros
            chains_values_noise = [torch.zeros_like(chains_values[-1])]
        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        chains_values = torch.stack(chains_values, dim=1)
        chains_values_noise = torch.stack(chains_values_noise, dim=1)

        # entropy is only available for flow-noise method
        if self.config.noise_method == "flow_noise":
            chains_entropy = torch.stack(chains_entropy, dim=1)
        else:
            chains_entropy = torch.zeros_like(chains_log_probs)
        return chains_log_probs, chains_values, chains_values_noise, chains_entropy

    def get_value_from_vlm(self, prefix_output):
        # prefix_output:
        # pi05: [bs, (256 * 3 + 200) = 968, 2048]
        # pi0: [bs, (256 * 3 + 48) = 816, 1024]
        # token length
        if "pi05_" in self.config.config_name:
            lang_token_len = 200
            all_token_length = 968
        elif "pi0_" in self.config.config_name:
            lang_token_len = 48
            all_token_length = 816

        if self.config.value_vlm_mode == "mean_token":
            prefix_mask = (
                [True] * 256 * self.config.num_images_in_input
                + [False] * 256 * (3 - self.config.num_images_in_input)
                + [True] * lang_token_len
            )
        elif self.config.value_vlm_mode == "last_token":
            prefix_mask = [False] * (all_token_length - 1) + [True] * 1
        elif self.config.value_vlm_mode == "first_token":
            prefix_mask = [True] * 1 + [False] * (all_token_length - 1)
        prefix_out_value = prefix_output[:, prefix_mask, :]
        prefix_out_value = prefix_out_value.mean(dim=1, keepdim=False)
        prefix_out_value = prefix_out_value.to(dtype=torch.float32)
        values_vlm = self.value_head(prefix_out_value)[:, 0]
        return values_vlm

    def gaussian_entropy(self, sigma):
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        entropy = 0.5 * torch.log(2 * math.pi * math.e * (sigma_safe**2))
        return entropy

    def freeze_vlm(self):
        if self.config.train_expert_only:
            self.paligemma_with_expert.paligemma.eval()
            for params in self.paligemma_with_expert.paligemma.parameters():
                params.requires_grad = False
