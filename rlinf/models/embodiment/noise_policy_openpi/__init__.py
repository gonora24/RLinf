import glob
import os
from openpi.models.pi0_config import Pi0Config
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.models.embodiment.noise_policy_openpi.openpi_steering_action_model import NoisePolicyConfig, NoisePolicyForOpenPI
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.openpi_action_model import OpenPi0Config
from openpi.shared import download
from openpi.training import checkpoints as _checkpoints
import safetensors
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi import transforms

def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    """Get NoisePolicy openpi_model with openpi openpi_model.

    Args:
        cfg: All configs under [actor.openpi_model] in yaml file.
        torch_dtype: Data type of the openpi_model.

    Returns:
        The openpi_model.
    """

    # Load OpenPI config from registry (like regular OpenPI does)
    config_name = getattr(cfg.openpi, "config_name", None)
    actor_train_config = get_openpi_config(config_name, model_path=cfg.model_path)
    actor_model_config = actor_train_config.model
    # openpi_config = OpenPi0Config(**actor_model_config.__dict__)
    openpi_config = Pi0Config(**actor_model_config.__dict__)
    
    # Override OpenPI config with user values from cfg.openpi
    override_config_kwargs = cfg.openpi
    if override_config_kwargs is not None:
        for key, val in override_config_kwargs.items():
            if hasattr(openpi_config, key):
                openpi_config.__dict__[key] = val

    # load openpi_model
    checkpoint_dir = download.maybe_download(str(cfg.model_path))
    weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if not weight_paths:
        weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]

    openpi_model: PI0Pytorch = PI0Pytorch(
        actor_model_config
    )
    for weight_path in weight_paths:
        safetensors.torch.load_model(openpi_model, weight_path, strict=False)
    openpi_model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    # wrappers
    repack_transforms = transforms.Group()
    default_prompt = None
    # load data stats
    data_config = actor_train_config.data.create(
        actor_train_config.assets_dirs, actor_model_config
    )

    norm_stats = None
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)

    # Create NoisePolicyConfig and set the OpenPI config
    model_config = NoisePolicyConfig()
    model_config.openpi = openpi_config
    
    # Update the rest of NoisePolicyConfig from main cfg (excluding cfg.openpi)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict.pop("openpi", None)  # Remove openpi section since we already handled it
    model_config.update_from_dict(cfg_dict)
    
    noise_model = NoisePolicyForOpenPI(model_config, openpi_model)
    noise_model.setup_wrappers(
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
    return noise_model
