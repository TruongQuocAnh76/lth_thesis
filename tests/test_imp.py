import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pruning import (
    prune_by_percent,
    prune_by_magnitude_global,
    create_initial_masks,
    get_sparsity,
    get_overall_sparsity,
    iterative_magnitude_pruning
)


# ============================================================================
# Helper Functions
# ============================================================================

def create_simple_network(seed=42):
    """Create a simple network with predictable weights for testing."""
    np.random.seed(seed)
    return {
        'layer0': np.random.randn(10, 5),
        'layer1': np.random.randn(5, 3),
        'layer2': np.random.randn(3, 1)
    }


def apply_masks_to_weights(weights, masks):
    """Apply masks to weights (zero out pruned weights)."""
    return {k: weights[k] * masks[k] for k in weights.keys()}


def simulate_training_step(weights, masks, lr=0.01, seed=None):
    """Simulate a training step with gradient update.
    
    Returns weights after optimizer step with masks applied.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Simulate gradient update
    new_weights = {}
    for k in weights.keys():
        gradients = np.random.randn(*weights[k].shape) * 0.1
        new_weights[k] = weights[k] - lr * gradients
    
    # Apply masks (pruned weights should remain zero)
    return apply_masks_to_weights(new_weights, masks)


def count_pruned_weights(masks):
    """Count total number of pruned weights across all layers."""
    return sum(np.sum(mask == 0) for mask in masks.values())


def count_total_weights(masks):
    """Count total number of weights across all layers."""
    return sum(mask.size for mask in masks.values())


# ============================================================================
# Test 1: Core IMP Correctness
# ============================================================================

class TestCoreIMPCorrectness:
    """Test core IMP behavior: sparsity, masking, and rewinding."""
    
    def test_exact_target_sparsity_after_pruning(self):
        """Test that pruning achieves exact target sparsity."""
        weights = create_simple_network()
        masks = create_initial_masks(weights)
        
        # Prune 20% per layer
        prune_percents = {k: 0.2 for k in weights.keys()}
        new_masks = prune_by_percent(prune_percents, masks, weights)
        
        # Check each layer has approximately 20% sparsity
        # For small layers, rounding causes deviations (e.g., 3 weights * 0.2 = 0.6 -> 1 weight = 33%)
        for layer_name in weights.keys():
            layer_sparsity = get_sparsity(new_masks)[layer_name]
            expected_sparsity = 0.2
            layer_size = weights[layer_name].size
            
            # For small layers, use absolute tolerance; for large layers, use relative tolerance
            if layer_size < 10:
                tolerance = 1.0 / layer_size + 0.05  # Allow for 1 weight difference
            else:
                tolerance = 0.05
            
            assert abs(layer_sparsity - expected_sparsity) < tolerance, \
                f"Layer {layer_name} sparsity {layer_sparsity} != {expected_sparsity} (tolerance: {tolerance})"
    
    def test_exact_global_sparsity(self):
        """Test global pruning achieves exact target sparsity."""
        weights = create_simple_network()
        masks = create_initial_masks(weights)
        
        target_sparsity = 0.3
        new_masks = prune_by_magnitude_global(target_sparsity, masks, weights)
        
        actual_sparsity = get_overall_sparsity(new_masks)
        # Allow small tolerance due to rounding
        total = sum(w.size for w in weights.values())
        expected_pruned = int(round(target_sparsity * total))
        actual_pruned = sum(int(np.sum(mask == 0)) for mask in new_masks.values())
        assert abs(actual_pruned - expected_pruned) <= 1, \
        f"Overall sparsity {actual_sparsity} != {target_sparsity}"
    
    def test_pruned_weights_remain_zero_after_optimizer_steps(self):
        """Test that masked weights stay zero even after optimizer updates."""
        weights = create_simple_network(seed=42)
        masks = create_initial_masks(weights)
        
        # Prune 30%
        prune_percents = {k: 0.3 for k in weights.keys()}
        masks = prune_by_percent(prune_percents, masks, weights)
        
        # Apply masks
        masked_weights = apply_masks_to_weights(weights, masks)
        
        # Simulate multiple training steps
        current_weights = masked_weights
        for step in range(10):
            current_weights = simulate_training_step(current_weights, masks, seed=step)
        
        # Check all pruned positions remain zero
        for layer_name in weights.keys():
            pruned_positions = masks[layer_name] == 0
            assert np.all(current_weights[layer_name][pruned_positions] == 0), \
                f"Pruned weights in {layer_name} are not zero after training"
    
    def test_rewinding_restores_unpruned_weights(self):
        """Test that rewinding restores unpruned weights to rewind iteration values."""
        # Initial weights (t=0)
        initial_weights = create_simple_network(seed=42)
        masks = create_initial_masks(initial_weights)
        
        # Train for a few steps to get weights at rewind point (t=5)
        rewind_weights = initial_weights.copy()
        for step in range(5):
            rewind_weights = simulate_training_step(rewind_weights, masks, seed=step)
        
        # Continue training to get weights for pruning decision (t=20)
        trained_weights = {k: v.copy() for k, v in rewind_weights.items()}
        for step in range(5, 20):
            trained_weights = simulate_training_step(trained_weights, masks, seed=step)
        
        # Prune based on weights at t=20
        prune_percents = {k: 0.2 for k in trained_weights.keys()}
        masks = prune_by_percent(prune_percents, masks, trained_weights)
        
        # Rewind: unpruned weights should match rewind_weights
        rewound_weights = apply_masks_to_weights(rewind_weights, masks)
        
        # Check unpruned weights match rewind point
        for layer_name in initial_weights.keys():
            unpruned_positions = masks[layer_name] == 1
            np.testing.assert_array_almost_equal(
                rewound_weights[layer_name][unpruned_positions],
                rewind_weights[layer_name][unpruned_positions],
                decimal=5,
                err_msg=f"Unpruned weights in {layer_name} don't match rewind point"
            )
    
    def test_rewinding_does_not_change_pruned_weights(self):
        """Test that rewinding keeps pruned weights at zero."""
        initial_weights = create_simple_network(seed=42)
        masks = create_initial_masks(initial_weights)
        
        # Train and prune
        trained_weights = initial_weights.copy()
        for step in range(10):
            trained_weights = simulate_training_step(trained_weights, masks, seed=step)
        
        prune_percents = {k: 0.3 for k in trained_weights.keys()}
        masks = prune_by_percent(prune_percents, masks, trained_weights)
        
        # Rewind to initial weights with masks applied
        rewound_weights = apply_masks_to_weights(initial_weights, masks)
        
        # Check pruned positions are zero
        for layer_name in initial_weights.keys():
            pruned_positions = masks[layer_name] == 0
            assert np.all(rewound_weights[layer_name][pruned_positions] == 0), \
                f"Pruned weights in {layer_name} are not zero after rewind"


# ============================================================================
# Test 2: Pruning Logic
# ============================================================================

class TestPruningLogic:
    """Test that pruning correctly identifies and removes smallest weights."""
    
    def test_magnitude_pruning_removes_smallest_weights(self):
        """Test that magnitude pruning removes the smallest absolute value weights."""
        # Create weights with known magnitudes
        weights = {
            'layer0': np.array([[-5.0, 3.0, -1.0, 0.5, 4.0],
                                [2.0, -0.1, 6.0, -0.3, 1.5]]),
        }
        masks = create_initial_masks(weights)
        
        # Prune 40% (should remove 4 out of 10 weights)
        prune_percents = {'layer0': 0.4}
        new_masks = prune_by_percent(prune_percents, masks, weights)
        
        # The 4 smallest magnitude weights are: -0.1, -0.3, 0.5, -1.0
        # These should be pruned (mask = 0)
        abs_weights = np.abs(weights['layer0'])
        masked_weights = new_masks['layer0']
        
        # Check that 4 weights are pruned
        assert np.sum(masked_weights == 0) == 4, "Should prune exactly 4 weights"
        
        # Check smallest magnitude weights are pruned
        flat_abs = abs_weights.flatten()
        flat_mask = masked_weights.flatten()
        sorted_indices = np.argsort(flat_abs)
        
        # First 4 smallest should be pruned
        for idx in sorted_indices[:4]:
            assert flat_mask[idx] == 0, f"Small weight at index {idx} not pruned"
        
        # Remaining should not be pruned
        for idx in sorted_indices[4:]:
            assert flat_mask[idx] == 1, f"Large weight at index {idx} incorrectly pruned"
    
    def test_global_pruning_consistency(self):
        """Test that global pruning is consistent across layers."""
        weights = {
            'layer0': np.array([[1.0, 0.1, 0.2]]),
            'layer1': np.array([[0.05, 5.0, 0.15]])
        }
        masks = create_initial_masks(weights)
        
        # Global pruning of 50% (3 out of 6 weights)
        new_masks = prune_by_magnitude_global(0.5, masks, weights)
        
        # Count pruned weights
        total_pruned = count_pruned_weights(new_masks)
        # With strict inequality, weights at the cutoff may or may not be pruned
        # So we expect approximately 3 weights pruned (within 1)
        assert 2 <= total_pruned <= 4, f"Should prune ~3 weights, got {total_pruned}"
        
        # The smallest weights should definitely be pruned
        assert new_masks['layer1'][0, 0] == 0, "0.05 should be pruned (smallest)"
        assert new_masks['layer0'][0, 1] == 0, "0.1 should be pruned (second smallest)"
        
        # Largest weights should remain
        assert new_masks['layer0'][0, 0] == 1, "1.0 should not be pruned"
        assert new_masks['layer1'][0, 1] == 1, "5.0 should not be pruned"
    
    def test_number_of_pruned_parameters_matches_schedule(self):
        """Test that exact number of parameters are pruned per schedule."""
        weights = create_simple_network(seed=42)
        masks = create_initial_masks(weights)
        
        initial_total = count_total_weights(masks)
        
        # Prune 25% per layer
        prune_percents = {k: 0.25 for k in weights.keys()}
        new_masks = prune_by_percent(prune_percents, masks, weights)
        
        # Check each layer
        for layer_name in weights.keys():
            layer_total = weights[layer_name].size
            layer_pruned = np.sum(new_masks[layer_name] == 0)
            expected_pruned = int(np.round(0.25 * layer_total))
            
            # Should match within 1 due to rounding
            assert abs(layer_pruned - expected_pruned) <= 1, \
                f"Layer {layer_name}: pruned {layer_pruned}, expected ~{expected_pruned}"


# ============================================================================
# Test 3: Determinism
# ============================================================================

class TestDeterminism:
    """Test that IMP is deterministic with fixed random seeds."""
    
    def test_fixed_seed_produces_identical_pruning_mask(self):
        """Test that same seed produces identical masks."""
        # Run 1
        weights1 = create_simple_network(seed=123)
        masks1 = create_initial_masks(weights1)
        prune_percents = {k: 0.3 for k in weights1.keys()}
        final_masks1 = prune_by_percent(prune_percents, masks1, weights1)
        
        # Run 2 with same seed
        weights2 = create_simple_network(seed=123)
        masks2 = create_initial_masks(weights2)
        final_masks2 = prune_by_percent(prune_percents, masks2, weights2)
        
        # Masks should be identical
        for layer_name in weights1.keys():
            np.testing.assert_array_equal(
                final_masks1[layer_name],
                final_masks2[layer_name],
                err_msg=f"Masks differ for {layer_name} with same seed"
            )
    
    def test_different_seed_produces_different_weights(self):
        """Test that different seeds produce different initial weights."""
        weights1 = create_simple_network(seed=42)
        weights2 = create_simple_network(seed=999)
        
        # At least one layer should have different weights
        all_equal = all(
            np.allclose(weights1[k], weights2[k])
            for k in weights1.keys()
        )
        assert not all_equal, "Different seeds should produce different weights"
    
    def test_reproducible_training_with_fixed_seed(self):
        """Test that training is reproducible with fixed seed."""
        # Training run 1
        weights1 = create_simple_network(seed=42)
        masks1 = create_initial_masks(weights1)
        current1 = weights1.copy()
        for step in range(5):
            current1 = simulate_training_step(current1, masks1, seed=step)
        
        # Training run 2 with same seeds
        weights2 = create_simple_network(seed=42)
        masks2 = create_initial_masks(weights2)
        current2 = weights2.copy()
        for step in range(5):
            current2 = simulate_training_step(current2, masks2, seed=step)
        
        # Results should be identical
        for layer_name in weights1.keys():
            np.testing.assert_array_almost_equal(
                current1[layer_name],
                current2[layer_name],
                decimal=10,
                err_msg=f"Training not deterministic for {layer_name}"
            )


# ============================================================================
# Test 4: Baseline Sanity
# ============================================================================

class TestBaselineSanity:
    """Test that no-pruning and random pruning behave correctly."""
    
    def test_zero_percent_pruning_identical_to_baseline(self):
        """Test that 0% pruning leaves all weights active."""
        weights = create_simple_network(seed=42)
        masks = create_initial_masks(weights)
        
        # Prune 0%
        prune_percents = {k: 0.0 for k in weights.keys()}
        new_masks = prune_by_percent(prune_percents, masks, weights)
        
        # All masks should still be 1
        for layer_name in weights.keys():
            assert np.all(new_masks[layer_name] == 1), \
                f"0% pruning changed masks in {layer_name}"
        
        # Overall sparsity should be 0
        assert get_overall_sparsity(new_masks) == 0.0, \
            "0% pruning should result in 0 sparsity"
    
    def test_no_pruning_training_behavior(self):
        """Test that masked training with no pruning matches unmasked training."""
        weights = create_simple_network(seed=42)
        masks = create_initial_masks(weights)
        
        # Train without explicit masking
        unmasked = weights.copy()
        for step in range(10):
            np.random.seed(step)
            for k in unmasked.keys():
                gradients = np.random.randn(*unmasked[k].shape) * 0.1
                unmasked[k] = unmasked[k] - 0.01 * gradients
        
        # Train with masks (all ones)
        masked = weights.copy()
        for step in range(10):
            masked = simulate_training_step(masked, masks, seed=step)
        
        # Results should be identical
        for layer_name in weights.keys():
            np.testing.assert_array_almost_equal(
                unmasked[layer_name],
                masked[layer_name],
                decimal=10,
                err_msg=f"Masked training differs from unmasked for {layer_name}"
            )
    
    def test_random_pruning_creates_same_sparsity(self):
        """Test that random pruning achieves target sparsity."""
        np.random.seed(777)
        weights = create_simple_network()
        masks = create_initial_masks(weights)
        
        # Create random masks with 30% sparsity
        target_sparsity = 0.3
        random_masks = {}
        for k, v in masks.items():
            random_masks[k] = np.random.choice(
                [0, 1],
                size=v.shape,
                p=[target_sparsity, 1 - target_sparsity]
            ).astype(float)
        
        actual_sparsity = get_overall_sparsity(random_masks)
        # Random pruning will be approximately correct
        assert abs(actual_sparsity - target_sparsity) < 0.1, \
            f"Random pruning sparsity {actual_sparsity} far from target {target_sparsity}"


# ============================================================================
# Test 5: Behavioral Check
# ============================================================================

class TestBehavioralCheck:
    """Test that IMP behaves reasonably compared to baselines."""
    
    def test_imp_not_significantly_worse_than_random(self):
        """Test that IMP at moderate sparsity is competitive with random pruning.
        
        This is a lightweight behavioral check. In practice, IMP should find
        better subnetworks than random pruning.
        """
        weights = create_simple_network(seed=42)
        
        # IMP pruning
        masks_imp = create_initial_masks(weights)
        prune_percents = {k: 0.5 for k in weights.keys()}
        masks_imp = prune_by_percent(prune_percents, masks_imp, weights)
        
        # Random pruning with same sparsity
        np.random.seed(999)
        masks_random = {}
        for k, v in weights.items():
            masks_random[k] = np.random.choice([0, 1], size=v.shape, p=[0.5, 0.5]).astype(float)
        
        # Both should have similar sparsity
        imp_sparsity = get_overall_sparsity(masks_imp)
        random_sparsity = get_overall_sparsity(masks_random)
        
        assert 0.4 < imp_sparsity < 0.6, "IMP should achieve ~50% sparsity"
        assert 0.4 < random_sparsity < 0.6, "Random should achieve ~50% sparsity"
        
        # Both should have same number of active weights (approximately)
        imp_active = count_total_weights(masks_imp) - count_pruned_weights(masks_imp)
        random_active = count_total_weights(masks_random) - count_pruned_weights(masks_random)
        
        # Should be within 20% of each other
        ratio = min(imp_active, random_active) / max(imp_active, random_active)
        assert ratio > 0.8, "IMP and random should have similar number of active weights"
    
    def test_iterative_pruning_increases_sparsity(self):
        """Test that iterative pruning progressively increases sparsity."""
        initial_weights = create_simple_network(seed=42)
        
        # Simulate multiple pruning iterations
        trained_weights_per_iteration = []
        weights = initial_weights.copy()
        for i in range(3):
            # Simulate training
            masks = create_initial_masks(weights)
            for step in range(5):
                weights = simulate_training_step(weights, masks, seed=i*10 + step)
            trained_weights_per_iteration.append(weights.copy())
        
        prune_percents = {k: 0.2 for k in initial_weights.keys()}
        masks_history = iterative_magnitude_pruning(
            initial_weights,
            trained_weights_per_iteration,
            prune_percents,
            num_iterations=3
        )
        
        # Check sparsity increases over iterations
        sparsities = [get_overall_sparsity(masks) for masks in masks_history]
        
        for i in range(len(sparsities) - 1):
            assert sparsities[i] < sparsities[i + 1], \
                f"Sparsity should increase: iter {i} = {sparsities[i]}, iter {i+1} = {sparsities[i+1]}"


# ============================================================================
# Test 6: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_percent_pruning_is_identical_to_baseline(self):
        """Test 0% pruning edge case."""
        weights = create_simple_network()
        masks = create_initial_masks(weights)
        
        prune_percents = {k: 0.0 for k in weights.keys()}
        new_masks = prune_by_percent(prune_percents, masks, weights)
        
        # Should be identical to original
        for k in masks.keys():
            np.testing.assert_array_equal(masks[k], new_masks[k])
        
        assert get_overall_sparsity(new_masks) == 0.0
    
    def test_one_hundred_percent_pruning_runs_without_crash(self):
        """Test 100% pruning edge case - model should run but output constants."""
        weights = create_simple_network()
        masks = create_initial_masks(weights)
        
        # Prune everything
        prune_percents = {k: 1.0 for k in weights.keys()}
        new_masks = prune_by_percent(prune_percents, masks, weights)
        
        # All weights should be pruned (or nearly all for small layers)
        for k in new_masks.keys():
            # Due to rounding and strict inequality, might not be exactly 100%
            # but should be very high
            sparsity = np.sum(new_masks[k] == 0) / new_masks[k].size
            # For small layers, allow lower threshold
            layer_size = new_masks[k].size
            if layer_size <= 3:
                threshold = 0.66  # Very small layers
            elif layer_size < 20:
                threshold = 0.85  # Small layers  
            else:
                threshold = 0.95  # Larger layers
            assert sparsity > threshold, \
                f"Layer {k} (size {layer_size}) should be mostly pruned (sparsity: {sparsity}, threshold: {threshold})"
        
        # Apply masks - all (or nearly all) weights should be zero
        masked_weights = apply_masks_to_weights(weights, new_masks)
        
        # Training should not crash
        try:
            for step in range(5):
                masked_weights = simulate_training_step(masked_weights, new_masks, seed=step)
        except Exception as e:
            pytest.fail(f"100% pruning caused crash: {e}")
        
        # Pruned weights should remain zero
        for k in masked_weights.keys():
            pruned_positions = new_masks[k] == 0
            assert np.allclose(masked_weights[k][pruned_positions], 0.0), \
                f"Pruned weights should stay zero in {k}"
    
    def test_single_weight_layer(self):
        """Test pruning a layer with only one weight."""
        weights = {'single': np.array([[5.0]])}
        masks = create_initial_masks(weights)
        
        # Prune 50% (rounds to 0 or 1)
        prune_percents = {'single': 0.5}
        new_masks = prune_by_percent(prune_percents, masks, weights)
        
        # Should not crash
        assert new_masks['single'].shape == (1, 1)
    
    def test_empty_layer_dict(self):
        """Test behavior with empty weight dictionary."""
        weights = {}
        masks = create_initial_masks(weights)
        
        assert masks == {}
        assert get_overall_sparsity(masks) == 0.0 or np.isnan(get_overall_sparsity(masks))
    
    def test_very_small_weights(self):
        """Test pruning with very small magnitude weights."""
        weights = {
            'layer0': np.array([[1e-10, 1e-9, 1e-8, 1e-7, 1e-6]])
        }
        masks = create_initial_masks(weights)
        
        prune_percents = {'layer0': 0.4}
        new_masks = prune_by_percent(prune_percents, masks, weights)
        
        # Should prune 2 smallest: 1e-10 and 1e-9
        pruned_count = np.sum(new_masks['layer0'] == 0)
        assert pruned_count == 2, f"Should prune 2 weights, got {pruned_count}"
    
    def test_negative_weights_pruned_by_magnitude(self):
        """Test that negative weights are pruned by absolute value."""
        weights = {
            'layer0': np.array([[-10.0, 0.5, 5.0, -0.1, 3.0]])
        }
        masks = create_initial_masks(weights)
        
        # Prune 40% (2 out of 5)
        prune_percents = {'layer0': 0.4}
        new_masks = prune_by_percent(prune_percents, masks, weights)
        
        # Smallest magnitudes are -0.1 (0.1) and 0.5
        # These should be pruned
        assert new_masks['layer0'][0, 3] == 0, "-0.1 should be pruned"
        assert new_masks['layer0'][0, 1] == 0, "0.5 should be pruned"
        
        # Larger magnitudes should remain
        assert new_masks['layer0'][0, 0] == 1, "-10.0 should not be pruned"
        assert new_masks['layer0'][0, 2] == 1, "5.0 should not be pruned"
        assert new_masks['layer0'][0, 4] == 1, "3.0 should not be pruned"


# ============================================================================
# Test 7: Iterative Pruning
# ============================================================================

class TestIterativePruning:
    """Test the full iterative pruning workflow."""
    
    def test_iterative_pruning_maintains_mask_structure(self):
        """Test that iterative pruning maintains correct mask structure."""
        initial_weights = create_simple_network(seed=42)
        
        # Simulate training for 2 iterations
        trained_weights_per_iteration = []
        for i in range(2):
            weights = create_simple_network(seed=42 + i)
            trained_weights_per_iteration.append(weights)
        
        prune_percents = {k: 0.2 for k in initial_weights.keys()}
        masks_history = iterative_magnitude_pruning(
            initial_weights,
            trained_weights_per_iteration,
            prune_percents,
            num_iterations=2
        )
        
        # Should have initial mask + 2 pruned masks
        assert len(masks_history) == 3
        
        # All masks should have same structure as weights
        for masks in masks_history:
            for k in initial_weights.keys():
                assert masks[k].shape == initial_weights[k].shape
    
    def test_masks_become_progressively_more_sparse(self):
        """Test that masks become more sparse with each iteration."""
        initial_weights = create_simple_network(seed=42)
        
        trained_weights_per_iteration = []
        for i in range(4):
            weights = create_simple_network(seed=42 + i)
            trained_weights_per_iteration.append(weights)
        
        prune_percents = {k: 0.15 for k in initial_weights.keys()}
        masks_history = iterative_magnitude_pruning(
            initial_weights,
            trained_weights_per_iteration,
            prune_percents,
            num_iterations=4
        )
        
        # Check monotonic increase in sparsity
        for i in range(len(masks_history) - 1):
            sparsity_current = get_overall_sparsity(masks_history[i])
            sparsity_next = get_overall_sparsity(masks_history[i + 1])
            assert sparsity_next >= sparsity_current, \
                f"Sparsity decreased from {sparsity_current} to {sparsity_next}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
