"""
Pruning utilities for Lottery Ticket Hypothesis experiments.

This module provides:
1. Iterative Magnitude Pruning (IMP) - the core LTH algorithm
2. Layer-wise and global magnitude pruning
3. Mask management utilities

For Early-Bird pruning (channel pruning via BatchNorm γ), see earlybird.py
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


# =============================================================================
# Iterative Magnitude Pruning (IMP) - Core LTH Algorithm
# =============================================================================

def prune_by_percent(
    percents: Dict[str, float],
    masks: Dict[str, np.ndarray],
    final_weights: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Prune the smallest magnitude weights by a specified percentage for each layer.

    The algorithm:
    1. For each layer, gather all weights that are still active (mask == 1)
    2. Sort these weights by their absolute magnitude
    3. Determine a cutoff threshold based on the pruning percentage
    4. Set mask to 0 for all weights with magnitude <= cutoff
    
    Args:
        percents: Dictionary mapping layer names to pruning percentages.
                 Keys are layer names (e.g., 'layer0', 'layer1').
                 Values are floats between 0 and 1 indicating the fraction
                 of remaining weights to prune (e.g., 0.2 = prune 20%).
        masks: Dictionary of current masks for each layer.
              Keys are layer names, values are numpy arrays with values in {0, 1}.
              0 indicates the weight is pruned, 1 indicates it's active.
        final_weights: Dictionary of trained weights from the last training run.
                      Keys are layer names, values are numpy arrays of weights.
    
    Returns:
        Dictionary of updated masks with the same structure as input masks.
        Additional weights are set to 0 based on magnitude pruning.
    """
    
    def prune_by_percent_once(
        percent: float,
        mask: np.ndarray,
        final_weight: np.ndarray
    ) -> np.ndarray:
        """Prune a single layer by the specified percentage.
        
        Args:
            percent: Fraction of remaining weights to prune (0 to 1).
            mask: Current mask for this layer (0 = pruned, 1 = active).
            final_weight: Trained weights for this layer.
        
        Returns:
            Updated mask with additional weights pruned.
        """
        # Extract only the weights that haven't been pruned yet
        sorted_weights = np.sort(np.abs(final_weight[mask == 1]))
        
        # Handle edge cases
        if sorted_weights.size == 0:
            # All weights already pruned
            return mask
        
        # Determine the cutoff index: prune the bottom 'percent' of remaining weights
        cutoff_index = np.round(percent * sorted_weights.size).astype(int)
        
        # Clamp cutoff_index to valid range [0, size-1]
        cutoff_index = min(cutoff_index, sorted_weights.size - 1)
        
        cutoff = sorted_weights[cutoff_index]
        
        # Create new mask: set to 0 where |weight| < cutoff, keep existing mask otherwise
        # Use strict inequality to avoid pruning exactly at the boundary
        return np.where(np.abs(final_weight) < cutoff, np.zeros(mask.shape), mask)
    
    # Apply pruning to each layer
    new_masks = {}
    for k, percent in percents.items():
        new_masks[k] = prune_by_percent_once(percent, masks[k], final_weights[k])
    
    return new_masks


def prune_by_magnitude_global(
    percent: float,
    masks: Dict[str, np.ndarray],
    final_weights: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Prune weights globally across all layers by magnitude.
    Args:
        percent: Global fraction of remaining weights to prune (0 to 1).
        masks: Dictionary of current masks for each layer.
        final_weights: Dictionary of trained weights for each layer.
    
    Returns:
        Dictionary of updated masks with global magnitude pruning applied.
    """
    # Collect all non-pruned weights from all layers
    all_weights = []
    for k in masks.keys():
        layer_weights = np.abs(final_weights[k][masks[k] == 1])
        all_weights.append(layer_weights)
    
    # Concatenate and sort all weights globally
    all_weights = np.concatenate(all_weights)
    sorted_weights = np.sort(all_weights)
    
    # Determine global cutoff
    cutoff_index = np.round(percent * sorted_weights.size).astype(int)
    cutoff = sorted_weights[cutoff_index]
    
    # Apply cutoff to each layer
    new_masks = {}
    for k in masks.keys():
        new_masks[k] = np.where(
            np.abs(final_weights[k]) <= cutoff,
            np.zeros(masks[k].shape),
            masks[k]
        )
    
    return new_masks


def create_initial_masks(weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Create initial masks with all weights active (no pruning).
    
    Args:
        weights: Dictionary of weight arrays for each layer.
    
    Returns:
        Dictionary of masks (all ones) matching the shape of each weight array.
    """
    masks = {}
    for k, v in weights.items():
        masks[k] = np.ones(v.shape)
    return masks


def get_sparsity(masks: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Calculate the sparsity (fraction of pruned weights) for each layer.
    
    Args:
        masks: Dictionary of masks for each layer.
    
    Returns:
        Dictionary mapping layer names to sparsity values (0 to 1).
        Sparsity of 0 means no pruning, 1 means all weights pruned.
    """
    sparsity = {}
    for k, mask in masks.items():
        total_weights = mask.size
        pruned_weights = np.sum(mask == 0)
        sparsity[k] = pruned_weights / total_weights
    return sparsity


def get_overall_sparsity(masks: Dict[str, np.ndarray]) -> float:
    """Calculate the overall sparsity across all layers.
    
    Args:
        masks: Dictionary of masks for each layer.
    
    Returns:
        Overall sparsity as a float between 0 and 1.
    """
    total_weights = sum(mask.size for mask in masks.values())
    if total_weights == 0:
        return 0.0
    total_pruned = sum(np.sum(mask == 0) for mask in masks.values())
    return total_pruned / total_weights


def iterative_magnitude_pruning(
    initial_weights: Dict[str, np.ndarray],
    trained_weights_per_iteration: list,
    prune_percents: Dict[str, float],
    num_iterations: int
) -> list:
    """Run the complete iterative magnitude pruning process.
    
    1. Start with unpruned network
    2. For each iteration:
       - Apply magnitude-based pruning to create new masks
       - The masks are used to train the network (handled externally)
       - Record the masks for analysis
    
    Args:
        initial_weights: Dictionary of initial (random) weights.
        trained_weights_per_iteration: List of weight dictionaries, one for each
                                      training iteration after pruning.
        prune_percents: Dictionary of pruning percentages per layer.
        num_iterations: Number of pruning iterations to perform.
    
    Returns:
        List of masks, one for each iteration (including initial unpruned mask).
    """
    # Create initial masks (no pruning)
    masks = create_initial_masks(initial_weights)
    masks_history = [masks.copy()]
    
    # Iteratively prune
    for iteration in range(num_iterations):
        if iteration < len(trained_weights_per_iteration):
            # Prune based on trained weights from previous iteration
            masks = prune_by_percent(
                prune_percents,
                masks,
                trained_weights_per_iteration[iteration]
            )
            masks_history.append(masks.copy())
        else:
            break
    
    return masks_history

