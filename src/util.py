"""Utility helpers shared across experiments and training."""

import yaml
import numpy as np
import torch
import torch.nn as nn


def load_config(config_path: str = "configs/experiment.yaml") -> dict:
	"""Load experiment configuration from YAML file."""
	with open(config_path, "r") as f:
		return yaml.safe_load(f)


def set_seed(seed: int):
	"""Set random seed for reproducibility across torch and numpy."""
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def get_prunable_layers(model: nn.Module) -> dict:
	"""Extract Conv2d and Linear weights for pruning."""
	weights = {}
	for name, module in model.named_modules():
		if isinstance(module, (nn.Conv2d, nn.Linear)):
			weights[name] = module.weight.data.cpu().numpy()
	return weights


def apply_masks_to_model(model: nn.Module, masks: dict):
	"""Apply pruning masks to model weights in-place."""
	for name, module in model.named_modules():
		if name in masks:
			mask_tensor = torch.from_numpy(masks[name]).to(module.weight.device)
			module.weight.data *= mask_tensor.float()


def create_mask_apply_fn(model: nn.Module):
	"""Return a closure to reapply masks after each optimizer step."""

	def apply_fn(masks: dict):
		apply_masks_to_model(model, masks)

	return apply_fn
