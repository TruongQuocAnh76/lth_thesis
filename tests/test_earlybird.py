"""
Tests for Early-Bird pruning implementation.

Tests cover:
- BN gamma extraction
- Channel mask computation (global and layerwise)
- Hamming distance calculation
- EarlyBirdFinder convergence detection
- Channel mask to Conv weight mask expansion
- L1 regularization on BN layers
- Sparsity metrics
- Offline early-bird search
"""

import torch
import torch.nn as nn
import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.earlybird import (
    extract_bn_gammas,
    compute_channel_mask_from_bn_gamma,
    compute_channel_mask_hamming_distance,
    EarlyBirdFinder,
    expand_channel_mask_to_conv_weights,
    add_l1_regularization_to_bn,
    get_channel_sparsity,
    get_overall_channel_sparsity,
    early_bird_search_offline,
)


# helpers to build simple models
def make_conv_bn_pair(out_channels=4, in_channels=3, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
        nn.BatchNorm2d(out_channels)
    )

def make_model_sequential_conv_bn():
    # explicit names to check mapping behavior
    from collections import OrderedDict
    return nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, 4, 3, padding=1)),
        ('bn1', nn.BatchNorm2d(4)),
        ('conv2', nn.Conv2d(4, 8, 3, padding=1)),
        ('bn2', nn.BatchNorm2d(8)),
    ]))

# ---------- Test 1: extract_bn_gammas returns correct arrays ----------
def test_extract_bn_gammas_values_and_order():
    model = make_model_sequential_conv_bn()
    # set bn weights to known values
    model.bn1.weight.data = torch.tensor([0.1, -0.2, 0.3, -0.4])
    model.bn2.weight.data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, -6.0, 0.0, 0.5])
    gammas = extract_bn_gammas(model)
    # order should be module traversal order (bn1 then bn2)
    keys = list(gammas.keys())
    assert 'bn1' in keys and 'bn2' in keys
    np.testing.assert_allclose(gammas['bn1'], np.array([0.1, -0.2, 0.3, -0.4]))
    np.testing.assert_allclose(gammas['bn2'], np.array([1.0, 2.0, 3.0, 4.0, 5.0, -6.0, 0.0, 0.5]))

# ---------- Test 2: global top-k picks exact number of channels ----------
def test_compute_channel_mask_from_bn_gamma_global_exact_count():
    # construct fake bn_gammas with 3 layers sizes 2,3,1 -> total 6 channels
    bn_gammas = {
        'a': np.array([0.1, 0.9]),     # 2
        'b': np.array([0.2, 0.3, 0.4]),# 3
        'c': np.array([0.8])          # 1
    }
    # target_sparsity 50% => keep 3 channels (int(6*(1-0.5))==3)
    mask = compute_channel_mask_from_bn_gamma(bn_gammas, target_sparsity=0.5, pruning_method='global')
    total_kept = sum(int(m.sum()) for m in mask.values())
    assert total_kept == max(1, int(6 * (1 - 0.5)))  # equals 3

# ---------- Test 3: layerwise top-k per-layer ----------
def test_compute_channel_mask_from_bn_gamma_layerwise_counts():
    bn_gammas = {
        'a': np.array([0.1, 0.9, 0.2]),  # 3 -> keep floor? uses int -> int(3*(1-0.5))=1
        'b': np.array([0.6, 0.5])        # 2 -> keep 1
    }
    mask = compute_channel_mask_from_bn_gamma(bn_gammas, target_sparsity=0.5, pruning_method='layerwise')
    assert int(mask['a'].sum()) == max(1, int(3 * (1 - 0.5)))
    assert int(mask['b'].sum()) == max(1, int(2 * (1 - 0.5)))

# ---------- Test 4: hamming distance correctness ----------
def test_compute_channel_mask_hamming_distance_basic():
    m1 = {'l1': np.array([1, 1, 0], dtype=np.float32)}
    m2 = {'l1': np.array([1, 0, 0], dtype=np.float32)}
    # 1 differing out of 3 => distance 1/3
    d = compute_channel_mask_hamming_distance(m1, m2)
    assert pytest.approx(d, rel=1e-6) == 1.0/3.0

# ---------- Test 5: EarlyBirdFinder convergence on synthetic bn_gammas ----------
def test_earlybirdfinder_converges_when_masks_stabilize():
    # produce bn_gammas per epoch that yield masks:
    # epoch0: [1,1], epoch1: [1,0], epoch2: [1,0], epoch3: [1,0] => should converge with patience=2
    bn_epochs = [
        {'bn': np.array([0.9, 0.8])},   # mask [1,1]
        {'bn': np.array([1.0, 0.01])},  # mask [1,0]
        {'bn': np.array([0.99, 0.0])},  # mask [1,0]
        {'bn': np.array([0.98, 0.0])},  # mask [1,0]
    ]
    finder = EarlyBirdFinder(target_sparsity=0.5, patience=2, distance_threshold=0.01, pruning_method='layerwise')
    # use record_from_bn_gammas to bypass model extraction
    conv_epoch = None
    for epoch, bn in enumerate(bn_epochs):
        converged, dist = finder.record_from_bn_gammas(bn, epoch)
        if converged:
            conv_epoch = epoch
            break
    # converged at epoch 3 in this setup (first two distances after epoch1->2 and epoch2->3 are zero)
    assert conv_epoch is not None
    assert finder.get_early_bird_ticket() is not None
    stats = finder.get_statistics()
    assert stats['converged'] is True

# ---------- Test 6: expand_channel_mask_to_conv_weights basic mapping ----------
def test_expand_channel_mask_to_conv_weights_basic():
    from collections import OrderedDict
    model = nn.Sequential(OrderedDict([
        ('conv1', nn.Conv2d(3, 4, 3, padding=1)),
        ('bn1', nn.BatchNorm2d(4))
    ]))
    # create channel mask for bn1
    ch_mask = {'bn1': np.array([1,0,1,0], dtype=np.float32)}
    wm = expand_channel_mask_to_conv_weights(ch_mask, model)
    # should map to conv1 and have shape (4,3,3,3)
    assert 'conv1' in wm
    assert wm['conv1'].shape == model.conv1.weight.shape
    # collapsed per-channel sums should match ch_mask
    per_out = wm['conv1'].reshape(4, -1).sum(axis=1)
    # kept channels have non-zero sum, pruned have zero
    assert (per_out > 0).astype(int).tolist() == ch_mask['bn1'].tolist()

# ---------- Test 7: expand mapping chooses nearest preceding conv ----------
def test_expand_channel_mask_to_conv_weights_prefers_nearest_conv():
    # build model where two convs have same out_channels; bn follows 2nd conv
    from collections import OrderedDict
    model = nn.Sequential(OrderedDict([
        ('convA', nn.Conv2d(3, 4, 3, padding=1)),
        ('convB', nn.Conv2d(4, 4, 3, padding=1)),
        ('bn_after_b', nn.BatchNorm2d(4)),
    ]))
    ch_mask = {'bn_after_b': np.array([1,1,0,0], dtype=np.float32)}
    wm = expand_channel_mask_to_conv_weights(ch_mask, model)
    # should map to convB (the nearest preceding conv with matching channels)
    assert 'convB' in wm and 'convA' not in wm

# ---------- Test 8: add_l1_regularization_to_bn returns correct scalar on device ----------
def test_add_l1_regularization_to_bn_value_and_device():
    model = make_model_sequential_conv_bn()
    # set bn weights known
    model.bn1.weight.data = torch.tensor([0.5, -0.5, 0.0, 1.0])
    model.bn2.weight.data = torch.tensor([0.1]*8)
    coef = 1e-3
    reg = add_l1_regularization_to_bn(model, l1_coef=coef)
    assert isinstance(reg, torch.Tensor)
    assert reg.device == next(model.parameters()).device
    # expected value = coef * sum(abs(weights))
    expected = coef * (torch.abs(model.bn1.weight).sum() + torch.abs(model.bn2.weight).sum())
    assert torch.allclose(reg, expected)

# ---------- Test 9: get_channel_sparsity and overall sparsity ----------
def test_channel_sparsity_metrics():
    masks = {'a': np.array([1,0,0], dtype=np.float32), 'b': np.array([1,1], dtype=np.float32)}
    per = get_channel_sparsity(masks)
    overall = get_overall_channel_sparsity(masks)
    assert per['a'] == pytest.approx(2/3)
    assert per['b'] == pytest.approx(0/2)  # 0 pruned
    assert overall == pytest.approx((2 + 0) / (3 + 2))

# ---------- Test 10: early_bird_search_offline returns expected convergence epoch ----------
def test_early_bird_search_offline_detects_convergence():
    # craft bn_gammas per epoch that stabilize to same mask at epoch 2 onward
    bn_epochs = [
        {'bn': np.array([0.2, 0.8])},  # mask maybe [0,1] depending sparsity
        {'bn': np.array([0.1, 0.9])},
        {'bn': np.array([0.05, 0.95])},
        {'bn': np.array([0.05, 0.95])},
        {'bn': np.array([0.05, 0.95])},
    ]
    ticket, conv_epoch, dists = early_bird_search_offline(
        bn_gammas_per_epoch=bn_epochs,
        target_sparsity=0.5,
        patience=2,
        distance_threshold=1e-6,
        pruning_method='layerwise'
    )
    # expect convergence (conv_epoch != -1)
    assert conv_epoch != -1
    assert ticket is not None
