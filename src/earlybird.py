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
        # Build flattened list of (value, layer_idx, ch_idx)
        flattened = []
        layer_sizes = []
        for layer_idx, name in enumerate(layer_names):
            gamma_abs = np.abs(bn_gammas[name]).astype(float)
            layer_sizes.append(len(gamma_abs))
            for ch_idx, val in enumerate(gamma_abs):
                flattened.append((val, layer_idx, ch_idx))

        total_channels = len(flattened)
        # Compute how many channels to keep globally (use round for consistency)
        num_keep = max(1, int(round(total_channels * (1.0 - target_sparsity))))

        num_layers = len(layer_names)
        # Ensure we can reserve at least one per layer
        if num_keep < num_layers:
            num_keep = num_layers

        # Step 1: reserve best channel per layer
        reserved = set()  # (layer_idx, ch_idx)
        for layer_idx, name in enumerate(layer_names):
            gamma_abs = np.abs(bn_gammas[name]).astype(float)
            if gamma_abs.size == 0:
                continue
            best_ch = int(np.argmax(gamma_abs))
            reserved.add((layer_idx, best_ch))

        # Step 2: pick remaining channels globally, excluding reserved
        remaining_to_pick = num_keep - len(reserved)
        if remaining_to_pick > 0:
            # Build candidate list excluding reserved
            candidates = []
            for val, layer_idx, ch_idx in flattened:
                if (layer_idx, ch_idx) in reserved:
                    continue
                candidates.append((val, layer_idx, ch_idx))
            # sort candidates descending and take top remaining_to_pick
            candidates.sort(key=lambda x: x[0], reverse=True)
            for i in range(min(remaining_to_pick, len(candidates))):
                _, li, ci = candidates[i]
                reserved.add((li, ci))

        # Reconstruct per-layer masks
        masks: Dict[str, np.ndarray] = {}
        for layer_idx, name in enumerate(layer_names):
            size = layer_sizes[layer_idx]
            mask = np.zeros(size, dtype=np.float32)
            for ch in range(size):
                if (layer_idx, ch) in reserved:
                    mask[ch] = 1.0
            masks[name] = mask

        return masks

    else:
        # Per-layer top-k with global correction to match target sparsity.
        # Step 1: Compute per-layer keep counts via rounding
        total_channels = sum(len(bn_gammas[n]) for n in layer_names)
        desired_keep = max(len(layer_names), int(round(total_channels * (1.0 - target_sparsity))))

        per_layer_keep = {}
        for name in layer_names:
            num_channels = len(bn_gammas[name])
            k = int(round(num_channels * (1.0 - target_sparsity)))
            k = max(1, min(k, num_channels))
            per_layer_keep[name] = k

        # Step 2: Build initial per-layer masks from rounded keep counts
        # Also collect (|γ|, layer_name, ch_idx, kept_flag) for global adjustment
        masks = {}
        kept_entries = []   # (|γ|, layer_name, ch_idx) — currently kept
        pruned_entries = [] # (|γ|, layer_name, ch_idx) — currently pruned
        for name in layer_names:
            gamma = np.abs(bn_gammas[name]).astype(float)
            num_channels = len(gamma)
            k = per_layer_keep[name]
            sorted_indices = np.argsort(gamma)[::-1]  # descending by |γ|
            mask = np.zeros(num_channels, dtype=np.float32)
            mask[sorted_indices[:k]] = 1.0
            masks[name] = mask
            for ch in range(num_channels):
                if mask[ch] == 1.0:
                    kept_entries.append((gamma[ch], name, ch))
                else:
                    pruned_entries.append((gamma[ch], name, ch))

        total_keep = sum(per_layer_keep.values())

        # Step 3: Global correction — add or remove channels to hit desired_keep
        if total_keep > desired_keep:
            # Remove excess kept channels (lowest |γ| first), preserving ≥1 per layer
            # Count how many are kept per layer for the min-1 guard
            layer_kept_count = {name: int(masks[name].sum()) for name in layer_names}
            # Sort kept entries ascending (weakest first)
            kept_entries.sort(key=lambda x: x[0])
            to_remove = total_keep - desired_keep
            for val, name, ch in kept_entries:
                if to_remove <= 0:
                    break
                if layer_kept_count[name] <= 1:
                    continue  # preserve at least 1 per layer
                masks[name][ch] = 0.0
                layer_kept_count[name] -= 1
                to_remove -= 1

        elif total_keep < desired_keep:
            # Add channels back (highest |γ| first among pruned)
            pruned_entries.sort(key=lambda x: x[0], reverse=True)
            to_add = desired_keep - total_keep
            for val, name, ch in pruned_entries:
                if to_add <= 0:
                    break
                masks[name][ch] = 1.0
                to_add -= 1

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
            'target_sparsity': self.target_sparsity,
            'distance_threshold': self.distance_threshold,
            'patience': self.patience,
            'pruning_method': self.pruning_method,
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
