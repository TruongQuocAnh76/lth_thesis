import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from collections import deque


def extract_bn_gammas(model: nn.Module) -> Dict[str, np.ndarray]:
    """Extract BatchNorm γ (weight/scaling) values from a model.
    
    These are the learnable scaling factors in BatchNorm that Early-Bird
    uses to determine channel importance.
    
    Args:
        model: PyTorch model with BatchNorm layers.
    
    Returns:
        Dictionary mapping BN layer names to their γ values as numpy arrays.
    """
    bn_gammas = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # BN weight is γ (scaling factor)
            if module.weight is not None:
                bn_gammas[name] = module.weight.detach().cpu().numpy().copy()
    
    return bn_gammas


def compute_channel_mask_from_bn_gamma(
    bn_gammas: Dict[str, np.ndarray],
    target_sparsity: float,
    pruning_method: str = 'global'
) -> Dict[str, np.ndarray]:
    """Compute channel pruning mask based on BatchNorm γ values.
    
    Channels with smaller |γ| are pruned (contribute less to output).
    This is the core of network slimming / Early-Bird.
    
    Args:
        bn_gammas: Dictionary of BN γ values per layer.
        target_sparsity: Fraction of channels to prune (0 to 1).
        pruning_method: 'global' for global threshold, 'layerwise' for per-layer.
    
    Returns:
        Dictionary of binary channel masks (0 = pruned channel, 1 = keep).
    """
    # Preserve module traversal order from extract_bn_gammas (named_modules)
    # Python 3.7+ dicts maintain insertion order - don't sort!
    layer_names = list(bn_gammas.keys())
    
    if pruning_method == 'global':
        # Exact global top-k: flatten with layer offsets, select top-k globally, reconstruct
        # This ensures exact reproducibility and matches the paper's reported sparsities
        
        # Build flattened array with offsets: (gamma_value, layer_idx, channel_idx)
        flattened = []
        layer_offsets = {}
        offset = 0
        
        for layer_idx, name in enumerate(layer_names):
            gamma_abs = np.abs(bn_gammas[name])
            layer_offsets[name] = offset
            
            for ch_idx, val in enumerate(gamma_abs):
                flattened.append((val, layer_idx, ch_idx, name))
            offset += len(gamma_abs)
        
        num_total = len(flattened)
        num_keep = int(num_total * (1 - target_sparsity))
        num_keep = max(1, num_keep)
        
        # Sort by gamma value (descending) and take top-k
        flattened_sorted = sorted(flattened, key=lambda x: x[0], reverse=True)
        top_k_indices = set()
        for i in range(num_keep):
            _, layer_idx, ch_idx, name = flattened_sorted[i]
            top_k_indices.add((layer_idx, ch_idx))
        
        # Reconstruct per-layer masks from top-k global indices
        masks = {}
        for layer_idx, name in enumerate(layer_names):
            gamma_len = len(bn_gammas[name])
            mask = np.zeros(gamma_len, dtype=np.float32)
            
            for ch_idx in range(gamma_len):
                if (layer_idx, ch_idx) in top_k_indices:
                    mask[ch_idx] = 1.0
            
            masks[name] = mask
    else:
        # Per-layer threshold
        masks = {}
        for name in layer_names:
            gamma = np.abs(bn_gammas[name])
            num_channels = len(gamma)
            num_keep = int(num_channels * (1 - target_sparsity))
            num_keep = max(1, num_keep)
            
            if num_keep < num_channels:
                # Exact top-k: sort and take top-k
                sorted_indices = np.argsort(gamma)[::-1]  # descending
                mask = np.zeros(num_channels, dtype=np.float32)
                mask[sorted_indices[:num_keep]] = 1.0
            else:
                mask = np.ones(num_channels, dtype=np.float32)
            
            masks[name] = mask
    
    return masks


def compute_channel_mask_hamming_distance(
    mask1: Dict[str, np.ndarray],
    mask2: Dict[str, np.ndarray]
) -> float:
    """Compute normalized Hamming distance between two channel masks.
    
    Args:
        mask1: First channel mask dictionary.
        mask2: Second channel mask dictionary.
    
    Returns:
        Normalized Hamming distance in [0, 1].
    """
    common_keys = sorted(set(mask1.keys()) & set(mask2.keys()))
    
    if not common_keys:
        return 0.0
    
    total_diff = 0
    total_channels = 0
    
    for name in common_keys:
        m1 = (mask1[name] > 0.5)
        m2 = (mask2[name] > 0.5)
        
        if m1.size != m2.size:
            continue
        
        total_diff += np.sum(m1 != m2)
        total_channels += m1.size
    
    if total_channels == 0:
        return 0.0
    
    return total_diff / total_channels


class EarlyBirdFinder:
    """Early-Bird algorithm for finding lottery tickets via BN γ convergence.
    
    The algorithm:
    1. Train with L1 regularization on BatchNorm γ values
    2. At the end of each epoch, compute channel pruning mask from |γ|
    3. Track mask changes using Hamming distance
    4. Convergence: max(last K distances) < ε → early-bird ticket found
    5. Use this channel mask for efficient structured pruning
    
    This produces **channel/filter masks**, not individual weight masks.
    The convergence criterion uses a sliding window maximum, not a counter.
    """
    
    def __init__(
        self,
        target_sparsity: float,
        patience: int = 5,
        distance_threshold: float = 0.1,
        pruning_method: str = 'global'
    ):
        """Initialize Early-Bird finder.
        
        Args:
            target_sparsity: Fraction of channels to prune (0 to 1).
            patience: Window size K for convergence check (paper uses 5).
            distance_threshold: ε threshold for max(last K distances) < ε.
            pruning_method: 'global' or 'layerwise' channel pruning.
        """
        self.target_sparsity = target_sparsity
        self.patience = patience
        self.distance_threshold = distance_threshold
        self.pruning_method = pruning_method
        
        # Storage for epoch-level mask history
        self.mask_history: List[Dict[str, np.ndarray]] = []
        self.distance_history: List[float] = []
        self.epoch_history: List[int] = []
        
        # Convergence state
        self.converged = False
        self.convergence_epoch: Optional[int] = None
        self.early_bird_mask: Optional[Dict[str, np.ndarray]] = None
    
    def reset(self):
        """Reset finder state for a new search."""
        self.mask_history = []
        self.distance_history = []
        self.epoch_history = []
        self.converged = False
        self.convergence_epoch = None
        self.early_bird_mask = None
    
    def record_epoch(
        self,
        model: nn.Module,
        epoch: int
    ) -> Tuple[bool, Optional[float]]:
        """Record mask at end of epoch and check for convergence.
        
        This should be called ONCE per epoch after training completes.
        
        Args:
            model: The PyTorch model with BatchNorm layers.
            epoch: Current epoch number (0-indexed).
        
        Returns:
            Tuple of (converged, distance):
            - converged: True if early-bird ticket found
            - distance: Hamming distance from previous epoch (None if first epoch)
        """
        # Extract BN γ values
        bn_gammas = extract_bn_gammas(model)
        
        if len(bn_gammas) == 0:
            raise ValueError("Model has no BatchNorm layers. Early-Bird requires BatchNorm.")
        
        # Compute channel mask
        current_mask = compute_channel_mask_from_bn_gamma(
            bn_gammas,
            self.target_sparsity,
            self.pruning_method
        )
        
        distance = None
        
        if len(self.mask_history) > 0:
            # Compute distance from previous mask
            prev_mask = self.mask_history[-1]
            distance = compute_channel_mask_hamming_distance(prev_mask, current_mask)
            self.distance_history.append(distance)
            
            # Early-Bird convergence criterion: max(last K distances) < ε
            # Use deque-like behavior for sliding window
            if len(self.distance_history) >= self.patience:
                recent_distances = self.distance_history[-self.patience:]
                max_recent = max(recent_distances)
                
                if max_recent < self.distance_threshold and not self.converged:
                    self.converged = True
                    self.convergence_epoch = epoch
                    self.early_bird_mask = current_mask
        
        self.mask_history.append(current_mask)
        self.epoch_history.append(epoch)
        
        return self.converged, distance
    
    def record_from_bn_gammas(
        self,
        bn_gammas: Dict[str, np.ndarray],
        epoch: int
    ) -> Tuple[bool, Optional[float]]:
        """Record mask from pre-extracted BN γ values.
        
        Use this when you've already extracted BN γ values and want to
        process them offline.
        
        Args:
            bn_gammas: Dictionary of BN γ values per layer.
            epoch: Current epoch number.
        
        Returns:
            Tuple of (converged, distance).
        """
        if len(bn_gammas) == 0:
            raise ValueError("bn_gammas cannot be empty")
        
        current_mask = compute_channel_mask_from_bn_gamma(
            bn_gammas,
            self.target_sparsity,
            self.pruning_method
        )
        
        distance = None
        
        if len(self.mask_history) > 0:
            prev_mask = self.mask_history[-1]
            distance = compute_channel_mask_hamming_distance(prev_mask, current_mask)
            self.distance_history.append(distance)
            
            # Early-Bird convergence: max(last K distances) < ε
            if len(self.distance_history) >= self.patience:
                recent_distances = self.distance_history[-self.patience:]
                max_recent = max(recent_distances)
                
                if max_recent < self.distance_threshold and not self.converged:
                    self.converged = True
                    self.convergence_epoch = epoch
                    self.early_bird_mask = current_mask
        
        self.mask_history.append(current_mask)
        self.epoch_history.append(epoch)
        
        return self.converged, distance
    
    def get_early_bird_ticket(self) -> Optional[Dict[str, np.ndarray]]:
        """Get the early-bird channel mask if found.
        
        Returns:
            Channel mask dictionary if found, None otherwise.
        """
        if self.converged and self.early_bird_mask is not None:
            return self.early_bird_mask
        
        if len(self.mask_history) > 0:
            return self.mask_history[-1]
        
        return None
    
    def get_statistics(self) -> Dict:
        """Get statistics about the early-bird search.
        
        Returns:
            Dictionary with search statistics.
        """
        stats = {
            'converged': self.converged,
            'convergence_epoch': self.convergence_epoch if self.convergence_epoch is not None else -1,
            'total_epochs_recorded': len(self.mask_history),
            'distance_history': self.distance_history.copy(),
            'epoch_history': self.epoch_history.copy(),
        }
        
        if len(self.distance_history) > 0:
            stats['mean_distance'] = float(np.mean(self.distance_history))
            stats['min_distance'] = float(np.min(self.distance_history))
            stats['max_distance'] = float(np.max(self.distance_history))
            
            # Report last K distances for debugging
            if len(self.distance_history) >= self.patience:
                stats['last_k_distances'] = self.distance_history[-self.patience:]
                stats['max_last_k'] = max(stats['last_k_distances'])
        
        return stats


def early_bird_search_offline(
    bn_gammas_per_epoch: List[Dict[str, np.ndarray]],
    target_sparsity: float,
    patience: int = 5,
    distance_threshold: float = 0.1,
    pruning_method: str = 'global'
) -> Tuple[Dict[str, np.ndarray], int, List[float]]:
    """Run Early-Bird search on recorded BN γ snapshots (offline).
    
    This is for when you've recorded BN γ values during training and
    want to find the early-bird ticket afterwards.
    
    Args:
        bn_gammas_per_epoch: List of BN γ dictionaries, one per epoch.
        target_sparsity: Fraction of channels to prune.
        patience: Window size K for convergence.
        distance_threshold: ε threshold for convergence.
        pruning_method: 'global' or 'layerwise'.
    
    Returns:
        Tuple of (channel_mask, convergence_epoch, distance_history).
    """
    if len(bn_gammas_per_epoch) == 0:
        raise ValueError("bn_gammas_per_epoch cannot be empty")
    
    finder = EarlyBirdFinder(
        target_sparsity=target_sparsity,
        patience=patience,
        distance_threshold=distance_threshold,
        pruning_method=pruning_method
    )
    
    for epoch, bn_gammas in enumerate(bn_gammas_per_epoch):
        converged, _ = finder.record_from_bn_gammas(bn_gammas, epoch)
        if converged:
            break
    
    ticket = finder.get_early_bird_ticket()
    stats = finder.get_statistics()
    
    return ticket, stats['convergence_epoch'], stats['distance_history']


def expand_channel_mask_to_conv_weights(
    channel_mask: Dict[str, np.ndarray],
    model: nn.Module
) -> Dict[str, np.ndarray]:
    """Expand channel masks to full weight masks for Conv2d layers.
    
    Maps BN layers to their nearest preceding Conv2d by module order and out_channels match.
    Robust to complex architectures (skip connections, bottlenecks, etc).
    
    Args:
        channel_mask: Dictionary of channel masks from Early-Bird.
        model: The PyTorch model to get Conv2d shapes from.
    
    Returns:
        Dictionary of weight masks for Conv2d layers.
    """
    weight_masks = {}
    
    # Build ordered list of all modules (preserve model traversal order)
    module_list = list(model.named_modules())
    
    # For each BN layer with a mask, find the nearest PRECEDING Conv2d with matching out_channels
    for bn_idx, (bn_name, bn_module) in enumerate(module_list):
        if not isinstance(bn_module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            continue
        
        if bn_name not in channel_mask:
            continue
        
        num_channels = len(channel_mask[bn_name])
        
        # Scan backwards from current position to find nearest preceding Conv2d
        best_conv_name = None
        best_conv_idx = -1
        
        for search_idx in range(bn_idx - 1, -1, -1):
            search_name, search_module = module_list[search_idx]
            
            if isinstance(search_module, nn.Conv2d):
                if search_module.weight.shape[0] == num_channels:
                    # Found matching Conv2d before this BN
                    best_conv_name = search_name
                    best_conv_idx = search_idx
                    break
        
        if best_conv_name is None:
            # No matching preceding Conv2d found
            continue
        
        # Get the Conv2d module
        _, conv_module = module_list[best_conv_idx]
        weight_shape = conv_module.weight.shape
        
        # Expand channel mask [out_channels] → [out_channels, in_channels, H, W]
        mask = channel_mask[bn_name]
        expanded_mask = mask.reshape(-1, 1, 1, 1)
        expanded_mask = np.broadcast_to(
            expanded_mask, weight_shape
        ).astype(np.float32)
        
        weight_masks[best_conv_name] = expanded_mask
    
    return weight_masks


def add_l1_regularization_to_bn(
    model: nn.Module,
    l1_coef: float = 1e-4
) -> torch.Tensor:
    """Compute L1 regularization loss on BatchNorm γ values.
    
    Add this to your training loss to encourage BN γ sparsity:
        loss = criterion(output, target) + add_l1_regularization_to_bn(model, l1_coef)
    
    Properly initializes loss as a tensor on the model's device to avoid
    CPU/GPU mismatches and type promotion issues.
    
    Args:
        model: PyTorch model with BatchNorm layers.
        l1_coef: L1 regularization coefficient.
    
    Returns:
        L1 regularization loss term (scalar tensor).
    """
    # Get device from first parameter of the model
    device = next(model.parameters()).device
    
    # Initialize loss as a tensor on the correct device
    l1_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if module.weight is not None:
                l1_loss = l1_loss + module.weight.abs().sum()
    
    return l1_loss * l1_coef


def get_bn_layer_count(model: nn.Module) -> int:
    """Count the number of BatchNorm layers in a model.
    
    Args:
        model: PyTorch model.
    
    Returns:
        Number of BatchNorm layers.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            count += 1
    return count


def get_channel_sparsity(channel_mask: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Calculate channel sparsity per layer.
    
    Args:
        channel_mask: Dictionary of channel masks.
    
    Returns:
        Dictionary of sparsity values per layer.
    """
    sparsity = {}
    for name, mask in channel_mask.items():
        total = mask.size
        pruned = np.sum(mask == 0)
        sparsity[name] = pruned / total if total > 0 else 0.0
    return sparsity


def get_overall_channel_sparsity(channel_mask: Dict[str, np.ndarray]) -> float:
    """Calculate overall channel sparsity.
    
    Args:
        channel_mask: Dictionary of channel masks.
    
    Returns:
        Overall channel sparsity as float.
    """
    total_channels = sum(mask.size for mask in channel_mask.values())
    if total_channels == 0:
        return 0.0
    total_pruned = sum(np.sum(mask == 0) for mask in channel_mask.values())
    return total_pruned / total_channels


# =============================================================================
# Alternative: Slimming Coefficient Approach (as in EarlyBERT)
# For models without BatchNorm, you can use learnable slimming coefficients
# =============================================================================

class SlimmingModule(nn.Module):
    """A module that adds learnable slimming coefficients.
    
    This can be used to add Early-Bird capability to models without BatchNorm,
    following the EarlyBERT approach for attention heads and FFN neurons.
    
    Usage:
        # Wrap a linear layer
        slimmed_linear = SlimmingModule(nn.Linear(768, 768), num_units=12)
    """
    
    def __init__(self, module: nn.Module, num_units: int, init_value: float = 1.0):
        """Initialize slimming module.
        
        Args:
            module: The module to wrap.
            num_units: Number of slimming coefficients (e.g., attention heads).
            init_value: Initial value for coefficients.
        """
        super().__init__()
        self.module = module
        self.slimming_coef = nn.Parameter(torch.ones(num_units) * init_value)
    
    def forward(self, x):
        output = self.module(x)
        # Apply slimming coefficients (broadcasting depends on use case)
        return output * self.slimming_coef
    
    def get_slimming_coef(self) -> np.ndarray:
        """Get slimming coefficients as numpy array."""
        return self.slimming_coef.detach().cpu().numpy()


def extract_slimming_coefficients(model: nn.Module) -> Dict[str, np.ndarray]:
    """Extract slimming coefficients from all SlimmingModules in model.
    
    Args:
        model: PyTorch model potentially containing SlimmingModule layers.
    
    Returns:
        Dictionary mapping module names to slimming coefficient arrays.
    """
    coefficients = {}
    
    for name, module in model.named_modules():
        if isinstance(module, SlimmingModule):
            coefficients[name] = module.get_slimming_coef()
    
    return coefficients


def add_l1_regularization_to_slimming(
    model: nn.Module,
    l1_coef: float = 1e-4
) -> torch.Tensor:
    """Compute L1 regularization on slimming coefficients.
    
    Properly initializes loss as a tensor on the model's device to avoid
    CPU/GPU mismatches and type promotion issues.
    
    Args:
        model: Model with SlimmingModule layers.
        l1_coef: L1 regularization coefficient.
    
    Returns:
        L1 loss term.
    """
    # Get device from first parameter of the model
    device = next(model.parameters()).device
    
    # Initialize loss as a tensor on the correct device
    l1_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    
    for module in model.modules():
        if isinstance(module, SlimmingModule):
            l1_loss = l1_loss + module.slimming_coef.abs().sum()
    
    return l1_loss * l1_coef
