"""Utility helpers shared across experiments and training."""

import yaml
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional


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


def compute_mask_overlap(
	mask1: Dict[str, np.ndarray],
	mask2: Dict[str, np.ndarray]
) -> float:
	"""Compute the Jaccard overlap between two masks.
	
	Useful for comparing Early-Bird tickets with IMP tickets.
	
	Args:
		mask1: First mask dictionary.
		mask2: Second mask dictionary.
	
	Returns:
		Jaccard similarity (intersection over union) in [0, 1].
	"""
	intersection = 0
	union = 0
	
	for name in mask1.keys():
		if name not in mask2:
			continue
		
		m1 = (mask1[name] > 0.5).astype(bool)
		m2 = (mask2[name] > 0.5).astype(bool)
		
		intersection += np.sum(m1 & m2)
		union += np.sum(m1 | m2)
	
	if union == 0:
		return 1.0  # Both masks are empty
	
	return intersection / union


def random_pruning(
	weights: Dict[str, np.ndarray],
	target_sparsity: float,
	seed: Optional[int] = None,
	pruning_method: str = 'global'
) -> Dict[str, np.ndarray]:
	"""Create random pruning masks (baseline for comparison).
	
	Random pruning serves as a sanity check - lottery tickets from 
	Early-Bird/IMP should significantly outperform random pruning.
	
	Args:
		weights: Dictionary of weight arrays (used for shape information).
		target_sparsity: Fraction of weights to prune.
		seed: Optional random seed for reproducibility.
		pruning_method: 'global' or 'layerwise'.
	
	Returns:
		Dictionary of random binary masks.
	"""
	if seed is not None:
		rng_state = np.random.get_state()
		np.random.seed(seed)
	
	masks = {}
	
	if pruning_method == 'global':
		# Count total weights
		total_size = sum(w.size for w in weights.values())
		num_keep = int(total_size * (1 - target_sparsity))
		
		# Generate random indices to keep
		keep_indices = np.random.choice(total_size, num_keep, replace=False)
		keep_set = set(keep_indices)
		
		# Create masks
		idx = 0
		for name, w in weights.items():
			mask = np.zeros(w.size, dtype=np.float32)
			for i in range(w.size):
				if idx + i in keep_set:
					mask[i] = 1.0
			masks[name] = mask.reshape(w.shape)
			idx += w.size
	else:
		# Per-layer random pruning
		for name, w in weights.items():
			num_keep = int(w.size * (1 - target_sparsity))
			mask = np.zeros(w.size, dtype=np.float32)
			keep_indices = np.random.choice(w.size, num_keep, replace=False)
			mask[keep_indices] = 1.0
			masks[name] = mask.reshape(w.shape)
	
	if seed is not None:
		np.random.set_state(rng_state)
	
	return masks
