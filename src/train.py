import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from typing import Optional, Dict, List, Any, Callable, Tuple
import numpy as np


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    masks: Optional[Dict[str, Any]] = None,
    apply_mask_fn: Optional[Callable] = None
) -> tuple:
    """Train model for one epoch.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use for training (cuda/cpu)
        masks: Optional dictionary of pruning masks
        apply_mask_fn: Optional function to apply masks to model
    
    Returns:
        Tuple of (average_loss, accuracy_percentage)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Apply mask after gradient update if pruning is active
        if masks is not None:
            if apply_mask_fn is not None:
                apply_mask_fn(masks)
            elif hasattr(model, 'apply_mask'):
                model.apply_mask(masks)
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    return total_loss / len(train_loader), 100. * correct / total


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """Evaluate model on test/validation set.
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to use for evaluation
    
    Returns:
        Tuple of (average_loss, accuracy_percentage)
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    return test_loss / len(test_loader), 100. * correct / total


def train_iterations(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_iterations: int,
    device: torch.device,
    masks: Optional[Dict[str, Any]] = None,
    eval_freq: int = 100,
    apply_mask_fn: Optional[Callable] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Train for a specified number of iterations.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        criterion: Loss function
        optimizer: Optimizer
        num_iterations: Total number of training iterations
        device: Device to use for training
        masks: Optional dictionary of pruning masks
        eval_freq: Frequency of evaluation (in iterations)
        apply_mask_fn: Optional function to apply masks to model
        verbose: Whether to show progress bar
    
    Returns:
        Dictionary containing training history:
        - 'train_losses': List of training losses per iteration
        - 'train_accs': List of training accuracies per iteration
        - 'test_accs': List of (iteration, test_accuracy) tuples
        - 'final_test_acc': Final test accuracy
    """
    model.train()
    iteration = 0
    epoch = 0
    
    train_losses = []
    train_accs = []
    test_accs = []
    
    pbar = tqdm(total=num_iterations, desc="Training", disable=not verbose)
    
    while iteration < num_iterations:
        for batch_idx, (data, target) in enumerate(train_loader):
            if iteration >= num_iterations:
                break
                
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Apply mask after gradient update if pruning is active
            if masks is not None:
                if apply_mask_fn is not None:
                    apply_mask_fn(masks)
                elif hasattr(model, 'apply_mask'):
                    model.apply_mask(masks)
            
            # Track metrics
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            acc = 100. * correct / target.size(0)
            
            train_losses.append(loss.item())
            train_accs.append(acc)
            
            # Evaluate periodically
            if iteration % eval_freq == 0:
                _, test_acc = evaluate(model, test_loader, criterion, device)
                test_accs.append((iteration, test_acc))
                pbar.set_postfix({'train_acc': f'{acc:.2f}%', 'test_acc': f'{test_acc:.2f}%'})
            
            iteration += 1
            pbar.update(1)
        
        epoch += 1
    
    pbar.close()
    
    # Final evaluation
    _, final_test_acc = evaluate(model, test_loader, criterion, device)
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'final_test_acc': final_test_acc
    }


def train_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    scheduler: Optional[Any] = None,
    masks: Optional[Dict[str, Any]] = None,
    apply_mask_fn: Optional[Callable] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Train for a specified number of epochs.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Total number of training epochs
        device: Device to use for training
        scheduler: Optional learning rate scheduler
        masks: Optional dictionary of pruning masks
        apply_mask_fn: Optional function to apply masks to model
        verbose: Whether to show progress bar
    
    Returns:
        Dictionary containing training history:
        - 'train_losses': List of training losses per epoch
        - 'train_accs': List of training accuracies per epoch
        - 'test_losses': List of test losses per epoch
        - 'test_accs': List of test accuracies per epoch
        - 'final_test_acc': Final test accuracy
    """
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    pbar = tqdm(range(num_epochs), desc="Training", disable=not verbose)
    
    for epoch in pbar:
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, masks, apply_mask_fn
        )
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Step scheduler if provided
        if scheduler is not None:
            scheduler.step()
        
        # Record metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'train_acc': f'{train_acc:.2f}%',
            'test_acc': f'{test_acc:.2f}%'
        })
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'final_test_acc': test_accs[-1] if test_accs else 0.0
    }


# ==============================================================================
# ResNet20 Early-Bird Support Functions
# ==============================================================================

def recalibrate_bn(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_batches: int = 200
):
    """Recalibrate BatchNorm running statistics after pruning.
    
    After pruning, running_mean/var become stale. This function updates
    them by running forward passes without gradient computation.
    
    Args:
        model: Model with BatchNorm layers
        loader: DataLoader for calibration data
        device: Device to use
        num_batches: Number of batches to use for recalibration
    """
    model.train()
    with torch.no_grad():
        i = 0
        for x, _ in loader:
            model(x.to(device))
            i += 1
            if i >= num_batches:
                break
    model.eval()


def get_resnet20_block_masks(
    model: nn.Module,
    channel_masks: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Convert channel masks to weight masks for ResNet20 with block-wise pruning.
    
    For ResNet, we use the BN before the residual add (bn2) as the authority.
    The same mask is applied to:
    - conv1 output channels (first conv in block)
    - conv2 output channels (second conv in block)  
    - shortcut conv output channels (if downsampling exists)
    
    Args:
        model: ResNet20 model
        channel_masks: Channel masks from Early-Bird (keyed by BN layer names)
    
    Returns:
        Weight masks for all Conv2d layers
    """
    weight_masks = {}
    blocks = model.get_block_info()
    
    for block_info in blocks:
        authority_bn_name = block_info['authority_bn']
        
        # Get the authority mask (from bn2)
        if authority_bn_name not in channel_masks:
            continue
            
        mask = channel_masks[authority_bn_name]  # Shape: (out_channels,)
        
        # Get conv layers from model
        conv1_name = block_info['conv1']
        conv2_name = block_info['conv2']
        
        # Parse names to get actual modules
        def get_module(name):
            parts = name.split('.')
            m = model
            for p in parts:
                if p.isdigit():
                    m = m[int(p)]
                else:
                    m = getattr(m, p)
            return m
        
        conv1 = get_module(conv1_name)
        conv2 = get_module(conv2_name)
        
        # Create weight mask for conv1 (output channels)
        # Conv1 weight shape: (out_channels, in_channels, k, k)
        conv1_mask = np.ones(conv1.weight.shape, dtype=np.float32)
        for c in range(len(mask)):
            if mask[c] == 0:
                conv1_mask[c, :, :, :] = 0
        weight_masks[conv1_name] = conv1_mask
        
        # Create weight mask for conv2 (output channels AND input channels)
        # Conv2 takes output of conv1, so prune input channels too
        conv2_mask = np.ones(conv2.weight.shape, dtype=np.float32)
        for c in range(len(mask)):
            if mask[c] == 0:
                conv2_mask[c, :, :, :] = 0  # Output channels
                conv2_mask[:, c, :, :] = 0  # Input channels (from conv1)
        weight_masks[conv2_name] = conv2_mask
        
        # Handle shortcut if exists
        if block_info['has_shortcut']:
            shortcut_conv_name = block_info['shortcut_conv']
            shortcut_conv = get_module(shortcut_conv_name)
            
            # Shortcut conv output channels must match block output
            shortcut_mask = np.ones(shortcut_conv.weight.shape, dtype=np.float32)
            for c in range(len(mask)):
                if mask[c] == 0:
                    shortcut_mask[c, :, :, :] = 0
            weight_masks[shortcut_conv_name] = shortcut_mask
    
    return weight_masks


def apply_resnet20_block_masks(
    model: nn.Module,
    channel_masks: Dict[str, np.ndarray],
    weight_masks: Dict[str, np.ndarray]
):
    """Apply block-wise pruning masks to ResNet20.
    
    Zeros out pruned channels in:
    - Conv2d weights
    - BatchNorm parameters (weight, bias, running_mean, running_var)
    
    Args:
        model: ResNet20 model
        channel_masks: Channel masks from Early-Bird
        weight_masks: Weight masks from get_resnet20_block_masks
    """
    with torch.no_grad():
        # Apply to Conv2d layers
        for name, module in model.named_modules():
            if name in weight_masks and isinstance(module, nn.Conv2d):
                mask_tensor = torch.from_numpy(weight_masks[name]).float().to(module.weight.device)
                module.weight.data *= mask_tensor
        
        # Apply to BatchNorm layers
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d) and name in channel_masks:
                mask = torch.from_numpy(channel_masks[name]).float().to(module.weight.device)
                module.weight.data *= mask
                module.bias.data *= mask
                module.running_mean.data *= mask
                # Avoid division by zero for pruned channels
                module.running_var.data = module.running_var.data * mask + (1 - mask) * 1.0


def train_resnet20_earlybird(
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
    target_sparsity: float = 0.5,
    total_epochs: int = 160,
    initial_lr: float = 0.1,
    lr_milestones: List[int] = [80, 120],
    lr_gamma: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    l1_coef: float = 1e-4,
    distance_threshold: float = 0.1,
    patience: int = 5,
    pruning_method: str = 'global',
    verbose: bool = True
) -> Dict[str, Any]:
    """Train ResNet20 with Early-Bird ticket discovery.
    
    Implements block-wise channel pruning for ResNet:
    - Uses bn2 (before residual add) as authority for each block
    - Applies same mask to conv1, conv2, and shortcut conv
    
    Args:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        device: Device (cuda/cpu)
        num_classes: Number of output classes (default: 10 for CIFAR-10)
        target_sparsity: Fraction of channels to prune (0.3, 0.5, 0.7)
        total_epochs: Maximum training epochs
        initial_lr: Initial learning rate
        lr_milestones: Epochs to reduce LR
        lr_gamma: LR reduction factor
        momentum: SGD momentum
        weight_decay: Weight decay
        l1_coef: L1 regularization coefficient for BN γ
        distance_threshold: ε threshold for mask convergence
        patience: Number of epochs for convergence window
        pruning_method: 'global' or 'layerwise'
        verbose: Print progress
    
    Returns:
        Dictionary with training history and Early-Bird results
    """
    # Import Early-Bird components
    from src.earlybird import (
        EarlyBirdFinder,
        add_l1_regularization_to_bn,
        get_overall_channel_sparsity,
        get_channel_sparsity,
    )
    from src.model import resnet20
    
    # Initialize model
    model = resnet20(num_classes=num_classes).to(device)
    
    # Setup optimizer, scheduler, criterion
    optimizer = optim.SGD(
        model.parameters(),
        lr=initial_lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_gamma)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize Early-Bird finder
    eb_finder = EarlyBirdFinder(
        target_sparsity=target_sparsity,
        patience=patience,
        distance_threshold=distance_threshold,
        pruning_method=pruning_method
    )
    
    # History tracking
    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'mask_distance': [],
        'lr': [],
    }
    
    convergence_epoch = None
    
    # Phase 1: Early-Bird Search
    pbar = tqdm(range(total_epochs), desc="EB Search", disable=not verbose)
    
    for epoch in pbar:
        current_lr = optimizer.param_groups[0]['lr']
        
        # Train one epoch with L1 regularization on BN γ
        model.train()
        total_loss = 0.0
        total_l1_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            ce_loss = criterion(outputs, targets)
            l1_loss = add_l1_regularization_to_bn(model, l1_coef)
            loss = ce_loss + l1_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += ce_loss.item() * inputs.size(0)
            total_l1_loss += l1_loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_loss = total_loss / total
        train_acc = 100.0 * correct / total
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        scheduler.step()
        
        # Record BN γ and check for convergence
        converged, mask_distance = eb_finder.record_epoch(model, epoch)
        
        # Store history
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['mask_distance'].append(mask_distance if mask_distance is not None else -1)
        history['lr'].append(current_lr)
        
        # Update progress bar
        dist_str = f"{mask_distance:.4f}" if mask_distance is not None else "N/A"
        pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'train_acc': f'{train_acc:.2f}%',
            'test_acc': f'{test_acc:.2f}%',
            'mask_dist': dist_str
        })
        
        if converged:
            convergence_epoch = epoch
            pbar.set_description(f"EB Found @ {epoch}")
            pbar.close()
            break
    
    # Get Early-Bird ticket
    eb_ticket = eb_finder.get_early_bird_ticket()
    eb_stats = eb_finder.get_statistics()
    
    if eb_ticket is None:
        return {
            'history': history,
            'eb_stats': eb_stats,
            'model': model,
            'converged': False,
        }
    
    # Phase 2: Apply block-wise pruning
    # Get block-wise weight masks
    weight_masks = get_resnet20_block_masks(model, eb_ticket)
    
    # Apply masks
    apply_resnet20_block_masks(model, eb_ticket, weight_masks)
    
    # Recalibrate BN statistics after pruning
    recalibrate_bn(model, train_loader, device, num_batches=200)
    
    # Verify sparsity
    total_params = 0
    zero_params = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
    actual_sparsity = zero_params / total_params * 100
    
    # Evaluate after pruning
    test_loss_pruned, test_acc_pruned = evaluate(model, test_loader, criterion, device)
    
    # Fine-tuning phase
    finetune_epochs = total_epochs - convergence_epoch
    
    # Reset optimizer with appropriate LR
    finetune_lr = initial_lr
    for milestone in lr_milestones:
        if convergence_epoch >= milestone:
            finetune_lr *= lr_gamma
    
    optimizer_ft = optim.SGD(
        model.parameters(),
        lr=finetune_lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    
    remaining_milestones = [m - convergence_epoch for m in lr_milestones if m > convergence_epoch]
    scheduler_ft = MultiStepLR(optimizer_ft, milestones=remaining_milestones, gamma=lr_gamma)
    
    finetune_history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
    }
    
    best_test_acc = test_acc_pruned
    
    # Fine-tuning with tqdm
    pbar_ft = tqdm(range(finetune_epochs), desc="Fine-tuning", disable=not verbose)
    
    for ft_epoch in pbar_ft:
        global_epoch = convergence_epoch + ft_epoch + 1
        current_lr = optimizer_ft.param_groups[0]['lr']
        
        # Train without L1 regularization
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer_ft, device
        )
        
        # Re-apply mask to maintain sparsity
        apply_resnet20_block_masks(model, eb_ticket, weight_masks)
        
        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        scheduler_ft.step()
        
        finetune_history['epoch'].append(global_epoch)
        finetune_history['train_loss'].append(train_loss)
        finetune_history['train_acc'].append(train_acc)
        finetune_history['test_loss'].append(test_loss)
        finetune_history['test_acc'].append(test_acc)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        
        pbar_ft.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'train_acc': f'{train_acc:.2f}%',
            'test_acc': f'{test_acc:.2f}%',
            'best': f'{best_test_acc:.2f}%'
        })
    
    return {
        'history': history,
        'finetune_history': finetune_history,
        'eb_stats': eb_stats,
        'eb_ticket': eb_ticket,
        'weight_masks': weight_masks,
        'model': model,
        'converged': True,
        'convergence_epoch': convergence_epoch,
        'best_test_acc': best_test_acc,
        'actual_sparsity': actual_sparsity,
        'test_acc_pruned': test_acc_pruned,
    }

