import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.models.embodiment.noise_policy_openpi.openpi_steering_action_model import NoisePolicyConfig, NoisePolicyForOpenPI


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    """Get NoisePolicy model with openpi model.

    Args:
        cfg: All configs under [actor.model] in yaml file.
        torch_dtype: Data type of the model.

    Returns:
        The model.
    """

    model_config = NoisePolicyConfig()
    model_config.update_from_dict(OmegaConf.to_container(cfg.openpi, resolve=True))
    model = NoisePolicyForOpenPI(model_config)


    return model
