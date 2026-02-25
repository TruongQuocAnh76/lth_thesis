"""
Genetic Algorithm for finding Lottery Tickets (Strong Lottery Ticket search).

Evolves binary masks over a fixed, randomly-initialised network to find
sub-networks that perform well *without any weight training*.  The GA
operates directly on the mask space:

    phenotype  w' = b ⊙ w          (element-wise mask × init weights)
    genotype   b  ∈ {0,1}^n        (binary vector over all prunable params)

Fitness is evaluated by forward-passing the masked network on training data
and computing loss (lower is better) or accuracy (higher is better), with
sparsity as a lexicographic secondary objective.

Reference hyper-parameters (from the SLTN-GA paper):
    population size  N = 100
    recombination    rec_rate = 0.3
    mutation         mut_rate = 0.1
    migration        mig_rate = 0.1
    parent pool      par_rate = 0.3
    min generations  100
    max generations  200
    stagnation       50 generations without improvement
"""

from __future__ import annotations

import copy
import hashlib
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


# ------------------------------------------------------------------ #
# Utility: flatten / unflatten masks  ↔  single bit-vector
# ------------------------------------------------------------------ #

def masks_to_bitvector(masks: Dict[str, np.ndarray]) -> np.ndarray:
    """Flatten a dict of per-layer binary masks into a single 1-D array."""
    parts = []
    for name in sorted(masks.keys()):
        parts.append(masks[name].ravel())
    return np.concatenate(parts).astype(np.uint8)


def bitvector_to_masks(
    bitvec: np.ndarray,
    template: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Un-flatten a 1-D bit-vector back into per-layer mask dicts."""
    masks: Dict[str, np.ndarray] = {}
    offset = 0
    for name in sorted(template.keys()):
        size = template[name].size
        shape = template[name].shape
        masks[name] = bitvec[offset:offset + size].reshape(shape).astype(np.float32)
        offset += size
    return masks


# ------------------------------------------------------------------ #
# Fitness evaluation helpers
# ------------------------------------------------------------------ #

class FitnessCache:
    """Hash-based cache to avoid re-evaluating identical genomes."""

    def __init__(self):
        self._cache: Dict[bytes, Tuple[float, float]] = {}
        self.hits = 0
        self.misses = 0

    def _key(self, genome: np.ndarray) -> bytes:
        return hashlib.md5(genome.tobytes()).digest()

    def get(self, genome: np.ndarray) -> Optional[Tuple[float, float]]:
        k = self._key(genome)
        if k in self._cache:
            self.hits += 1
            return self._cache[k]
        self.misses += 1
        return None

    def put(self, genome: np.ndarray, fitness: Tuple[float, float]):
        self._cache[self._key(genome)] = fitness

    def __len__(self):
        return len(self._cache)


def _evaluate_mask_on_data(
    model: nn.Module,
    masks: Dict[str, np.ndarray],
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    """Evaluate a masked sub-network (no training) on *dataloader*.

    Returns (loss, accuracy) averaged over batches (or *max_batches*).
    The model weights are **not** modified permanently; a deep copy is
    used internally.
    """
    # Apply masks to model weights (in-place on a copy)
    model.eval()
    # Save original weights, apply mask, evaluate, then restore
    original_state = {}
    for name, module in model.named_modules():
        if name in masks:
            original_state[name] = module.weight.data.clone()
            mask_t = torch.from_numpy(masks[name]).to(module.weight.device).float()
            module.weight.data.mul_(mask_t)

    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)
            total_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            n_batches += 1
            if max_batches is not None and n_batches >= max_batches:
                break

    # Restore original weights
    for name, module in model.named_modules():
        if name in original_state:
            module.weight.data.copy_(original_state[name])

    avg_loss = total_loss / max(n_batches, 1)
    accuracy = 100.0 * correct / max(total, 1)
    return avg_loss, accuracy


# ------------------------------------------------------------------ #
# Individual
# ------------------------------------------------------------------ #

@dataclass
class Individual:
    """A single candidate solution: a binary mask over network weights."""
    genome: np.ndarray                # 1-D uint8 bit-vector
    fitness: Optional[Tuple[float, float]] = None  # (perf, sparsity)
    origin: str = "init"
    age: int = 0

    @property
    def performance(self) -> float:
        """Primary objective (negative loss = higher is better)."""
        return self.fitness[0] if self.fitness else float('-inf')

    @property
    def sparsity(self) -> float:
        """Secondary objective (higher = sparser)."""
        return self.fitness[1] if self.fitness else 0.0

    def clone(self) -> Individual:
        return Individual(
            genome=self.genome.copy(),
            fitness=self.fitness,
            origin="clone",
            age=self.age,
        )


def _compute_sparsity(genome: np.ndarray) -> float:
    """Fraction of zeros in the genome (higher → sparser)."""
    return 1.0 - genome.sum() / genome.size


# ------------------------------------------------------------------ #
# GA operators
# ------------------------------------------------------------------ #

def random_genome(n: int) -> np.ndarray:
    """Generate a uniformly random binary vector of length *n*."""
    return np.random.randint(0, 2, size=n, dtype=np.uint8)


def single_point_crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Single-point crossover: take p1[:point] + p2[point:]."""
    n = len(p1)
    point = np.random.randint(1, n)
    child = np.empty_like(p1)
    child[:point] = p1[:point]
    child[point:] = p2[point:]
    return child


def mutate_single_bit(genome: np.ndarray) -> np.ndarray:
    """Flip exactly one random bit."""
    child = genome.copy()
    idx = np.random.randint(len(child))
    child[idx] = 1 - child[idx]
    return child


# ------------------------------------------------------------------ #
# Lexicographic fitness comparison
# ------------------------------------------------------------------ #

def _fitness_key(ind: Individual) -> Tuple[float, float]:
    """Sort key: maximise performance first, then sparsity."""
    if ind.fitness is None:
        return (float('-inf'), 0.0)
    return ind.fitness  # (perf, sparsity) — both higher-is-better


# ------------------------------------------------------------------ #
# Core GA
# ------------------------------------------------------------------ #

@dataclass
class GAConfig:
    """All tuneable GA hyper-parameters."""
    population_size: int = 100
    rec_rate: float = 0.3
    mut_rate: float = 0.1
    mig_rate: float = 0.1
    par_rate: float = 0.3
    min_generations: int = 100
    max_generations: int = 200
    stagnation_threshold: int = 50
    use_adaptive_ab: bool = False
    initial_ab_threshold: float = 0.7
    ab_decay_rate: float = 0.95
    use_loss_fitness: bool = True   # True → minimise loss; False → maximise accuracy
    max_eval_batches: Optional[int] = None  # limit batches per evaluation for speed
    post_prune: bool = True         # run post-evolutionary pruning
    # Checkpoint / resume support (Kaggle 12-hour sessions)
    time_limit_seconds: Optional[float] = None  # auto-save before this wall-time
    checkpoint_dir: Optional[str] = None         # where to write checkpoints
    checkpoint_interval: int = 10                # save every N generations


class GeneticAlgorithmPruner:
    """Genetic Algorithm for mask evolution on a fixed-weight network.

    Usage::

        pruner = GeneticAlgorithmPruner(model, train_loader, device, config)
        best_masks, stats = pruner.run()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        device: torch.device,
        config: GAConfig | None = None,
        verbose: bool = True,
    ):
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.cfg = config or GAConfig()
        self.verbose = verbose
        self.criterion = nn.CrossEntropyLoss()
        self.cache = FitnessCache()

        # Build mask template from prunable layers
        from src.util import get_prunable_layers
        self.mask_template: Dict[str, np.ndarray] = {
            name: np.ones_like(w) for name, w in get_prunable_layers(model).items()
        }
        self.genome_length = sum(m.size for m in self.mask_template.values())

        # History tracking
        self.generation_stats: List[Dict[str, Any]] = []

        # Wall-clock start (set when run() begins)
        self._start_time: Optional[float] = None

    # ----- checkpoint / resume --------------------------------------- #

    def save_checkpoint(
        self,
        path: str | Path,
        population: List[Individual],
        best_ever: Individual,
        best_perf_history: List[float],
        generation: int,
        elapsed: float,
        phase: str = "evolve",
    ) -> Path:
        """Persist the full GA state so it can be resumed later.

        Args:
            path: File path for the checkpoint (*.pt).
            population: Current population list.
            best_ever: Best individual found so far.
            best_perf_history: Per-generation best performance list.
            generation: Current generation index.
            elapsed: Elapsed seconds before this checkpoint.
            phase: One of 'evolve' or 'post_prune'.

        Returns:
            The resolved checkpoint path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        ckpt = {
            'phase': phase,
            'generation': generation,
            'elapsed_seconds': elapsed,
            'population': [
                {'genome': ind.genome, 'fitness': ind.fitness,
                 'origin': ind.origin, 'age': ind.age}
                for ind in population
            ],
            'best_ever': {
                'genome': best_ever.genome,
                'fitness': best_ever.fitness,
                'origin': best_ever.origin,
                'age': best_ever.age,
            },
            'best_perf_history': best_perf_history,
            'generation_stats': self.generation_stats,
            'cache': {
                'data': dict(self.cache._cache),
                'hits': self.cache.hits,
                'misses': self.cache.misses,
            },
            'config': {
                'population_size': self.cfg.population_size,
                'rec_rate': self.cfg.rec_rate,
                'mut_rate': self.cfg.mut_rate,
                'mig_rate': self.cfg.mig_rate,
                'par_rate': self.cfg.par_rate,
                'min_generations': self.cfg.min_generations,
                'max_generations': self.cfg.max_generations,
                'stagnation_threshold': self.cfg.stagnation_threshold,
                'use_adaptive_ab': self.cfg.use_adaptive_ab,
                'use_loss_fitness': self.cfg.use_loss_fitness,
                'post_prune': self.cfg.post_prune,
            },
            'mask_template_keys': sorted(self.mask_template.keys()),
            'genome_length': self.genome_length,
            'numpy_rng_state': np.random.get_state(),
        }
        torch.save(ckpt, path)
        if self.verbose:
            print(f"\n  Checkpoint saved → {path}  (gen {generation}, "
                  f"{elapsed:.0f}s elapsed)")
        return path

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        device: torch.device,
        config: GAConfig | None = None,
        verbose: bool = True,
    ) -> Tuple["GeneticAlgorithmPruner", dict]:
        """Restore a GeneticAlgorithmPruner from a saved checkpoint.

        Args:
            path: Checkpoint file path (*.pt).
            model: The *same* model architecture with the *same* initial
                   weights that were used when the GA started.
            train_loader: Training data loader.
            device: Compute device.
            config: Optional config override (defaults read from checkpoint).
            verbose: Whether to print progress.

        Returns:
            (pruner, resume_info) where *resume_info* is a dict with keys
            ``population``, ``best_ever``, ``best_perf_history``,
            ``start_generation``, ``elapsed_seconds``, and ``phase``.
        """
        path = Path(path)
        ckpt = torch.load(path, map_location='cpu', weights_only=False)

        # Rebuild pruner
        pruner = cls(model=model, train_loader=train_loader,
                     device=device, config=config, verbose=verbose)

        # Restore cache
        pruner.cache._cache = ckpt['cache']['data']
        pruner.cache.hits = ckpt['cache']['hits']
        pruner.cache.misses = ckpt['cache']['misses']

        # Restore generation stats
        pruner.generation_stats = ckpt['generation_stats']

        # Restore numpy RNG
        np.random.set_state(ckpt['numpy_rng_state'])

        # Rebuild population
        population = []
        for d in ckpt['population']:
            ind = Individual(
                genome=d['genome'], fitness=d['fitness'],
                origin=d['origin'], age=d['age'],
            )
            population.append(ind)

        best_d = ckpt['best_ever']
        best_ever = Individual(
            genome=best_d['genome'], fitness=best_d['fitness'],
            origin=best_d['origin'], age=best_d['age'],
        )

        resume_info = {
            'population': population,
            'best_ever': best_ever,
            'best_perf_history': ckpt['best_perf_history'],
            'start_generation': ckpt['generation'] + 1,  # resume from next gen
            'elapsed_seconds': ckpt['elapsed_seconds'],
            'phase': ckpt['phase'],
        }

        if verbose:
            print(f"\n  Checkpoint loaded ← {path}")
            print(f"    phase       : {ckpt['phase']}")
            print(f"    generation  : {ckpt['generation']}")
            print(f"    elapsed     : {ckpt['elapsed_seconds']:.0f}s")
            print(f"    cache size  : {len(pruner.cache)}")
            print(f"    best perf   : {best_ever.performance:.4f}")

        return pruner, resume_info

    def _time_remaining(self) -> Optional[float]:
        """Seconds remaining before the configured time limit, or None."""
        if self.cfg.time_limit_seconds is None or self._start_time is None:
            return None
        return self.cfg.time_limit_seconds - (time.time() - self._start_time)

    def _should_stop_for_time(self, margin_seconds: float = 120.0) -> bool:
        """True when we are within *margin_seconds* of the time limit."""
        remaining = self._time_remaining()
        if remaining is None:
            return False
        return remaining < margin_seconds

    def _checkpoint_path(self, tag: str = "ga_checkpoint") -> Path:
        """Build the default checkpoint file path."""
        d = self.cfg.checkpoint_dir or "."
        return Path(d) / f"{tag}.pt"

    # ----- evaluation ------------------------------------------------ #

    def _evaluate(self, ind: Individual) -> Tuple[float, float]:
        """Return (perf, sparsity) for an individual.

        *perf* is ``-loss`` when ``use_loss_fitness`` is True (so higher is
        always better), otherwise it is accuracy.
        """
        cached = self.cache.get(ind.genome)
        if cached is not None:
            return cached

        masks = bitvector_to_masks(ind.genome, self.mask_template)
        loss, acc = _evaluate_mask_on_data(
            self.model, masks, self.train_loader,
            self.criterion, self.device,
            max_batches=self.cfg.max_eval_batches,
        )

        if self.cfg.use_loss_fitness:
            perf = -loss  # higher is better
        else:
            perf = acc

        sparsity = _compute_sparsity(ind.genome)
        fitness = (perf, sparsity)
        self.cache.put(ind.genome, fitness)
        return fitness

    def _evaluate_population(self, population: List[Individual]) -> List[Individual]:
        """Evaluate & sort the population (best first)."""
        for ind in population:
            if ind.fitness is None:
                ind.fitness = self._evaluate(ind)
        population.sort(key=_fitness_key, reverse=True)
        return population

    # ----- initialisation -------------------------------------------- #

    def _init_population(self) -> List[Individual]:
        """Create N random individuals, optionally filtering by adaptive AB."""
        N = self.cfg.population_size
        n = self.genome_length
        population: List[Individual] = []

        if not self.cfg.use_adaptive_ab:
            for _ in range(N):
                population.append(Individual(genome=random_genome(n)))
            return population

        # Adaptive accuracy-bound initialisation
        ab_threshold = self.cfg.initial_ab_threshold
        attempts = 0
        max_attempts_per_decay = N * 5
        while len(population) < N:
            genome = random_genome(n)
            ind = Individual(genome=genome)
            ind.fitness = self._evaluate(ind)
            if ind.performance >= ab_threshold:
                population.append(ind)
                attempts = 0
            else:
                attempts += 1
                if attempts >= max_attempts_per_decay:
                    ab_threshold *= self.cfg.ab_decay_rate
                    attempts = 0
                    if self.verbose:
                        print(f"  AB threshold decayed to {ab_threshold:.4f}")
        return population

    # ----- evolutionary operators ------------------------------------ #

    def _parent_selection(
        self, pop: List[Individual]
    ) -> List[Tuple[Individual, Individual]]:
        """Select parent pairs for recombination."""
        N = len(pop)
        n_parents = max(1, int(self.cfg.rec_rate * N))
        top_k = max(1, int(self.cfg.par_rate * N))

        pairs: List[Tuple[Individual, Individual]] = []
        indices = np.random.choice(N, size=n_parents, replace=False)
        for idx in indices:
            p1 = pop[idx]
            p2_idx = np.random.randint(0, top_k)
            p2 = pop[p2_idx]
            pairs.append((p1, p2))
        return pairs

    def _recombine(self, pop: List[Individual]) -> List[Individual]:
        """Single-point crossover producing offspring."""
        pairs = self._parent_selection(pop)
        offspring = []
        for p1, p2 in pairs:
            child_genome = single_point_crossover(p1.genome, p2.genome)
            offspring.append(Individual(genome=child_genome, origin="crossover"))
        return offspring

    def _mutate(self, pop: List[Individual]) -> List[Individual]:
        """Flip a single random bit for selected individuals."""
        mutants = []
        for ind in pop:
            if np.random.random() < self.cfg.mut_rate:
                child_genome = mutate_single_bit(ind.genome)
                mutants.append(Individual(genome=child_genome, origin="mutate"))
        return mutants

    def _migrate(self) -> List[Individual]:
        """Inject fresh random individuals for diversity."""
        n_migrants = max(1, int(self.cfg.mig_rate * self.cfg.population_size))
        return [
            Individual(genome=random_genome(self.genome_length), origin="migrate")
            for _ in range(n_migrants)
        ]

    def _survive(self, extended: List[Individual]) -> List[Individual]:
        """Elitist survivor selection: keep top N."""
        extended = self._evaluate_population(extended)
        return extended[:self.cfg.population_size]

    # ----- stagnation check ------------------------------------------ #

    @staticmethod
    def _is_stagnant(history: List[float], threshold: int) -> bool:
        if len(history) < threshold:
            return False
        recent = history[-threshold:]
        return max(recent) <= recent[0]  # no improvement over window

    # ----- post-evolutionary pruning --------------------------------- #

    def _post_evolutionary_pruning(
        self, best: Individual
    ) -> Individual:
        """Sequentially try to zero each active bit; keep if no perf drop."""
        if self.verbose:
            print("\nPost-evolutionary pruning …")
        genome = best.genome.copy()
        masks = bitvector_to_masks(genome, self.mask_template)
        base_loss, _ = _evaluate_mask_on_data(
            self.model, masks, self.train_loader,
            self.criterion, self.device,
            max_batches=self.cfg.max_eval_batches,
        )
        base_perf = -base_loss if self.cfg.use_loss_fitness else None
        if not self.cfg.use_loss_fitness:
            _, base_acc = _evaluate_mask_on_data(
                self.model, masks, self.train_loader,
                self.criterion, self.device,
                max_batches=self.cfg.max_eval_batches,
            )
            base_perf = base_acc

        active_indices = np.where(genome == 1)[0]
        n_pruned = 0

        # Shuffle to avoid positional bias
        np.random.shuffle(active_indices)
        for idx in tqdm(active_indices, desc="Post-pruning", disable=not self.verbose):
            genome[idx] = 0
            trial_masks = bitvector_to_masks(genome, self.mask_template)
            loss, acc = _evaluate_mask_on_data(
                self.model, trial_masks, self.train_loader,
                self.criterion, self.device,
                max_batches=self.cfg.max_eval_batches,
            )
            trial_perf = -loss if self.cfg.use_loss_fitness else acc
            if trial_perf >= base_perf:
                base_perf = trial_perf
                n_pruned += 1
            else:
                genome[idx] = 1  # revert

        sparsity = _compute_sparsity(genome)
        if self.verbose:
            print(f"  Post-pruning removed {n_pruned} additional weights "
                  f"→ sparsity {sparsity:.4f}")

        result = Individual(genome=genome, origin="post_pruned")
        result.fitness = (base_perf, sparsity)
        return result

    # ----- main loop ------------------------------------------------- #

    def run(
        self,
        resume_info: Optional[dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Execute the full GA and return (best_masks, stats).

        Args:
            resume_info: If provided, resume from a prior checkpoint.
                Must be the dict returned by ``load_checkpoint()``.

        Returns:
            best_masks: Dict[str, np.ndarray] in the standard mask format.
            stats: dict with per-generation history and summary.
        """
        cfg = self.cfg
        t0 = time.time()
        self._start_time = t0
        prior_elapsed = 0.0
        stopped_early = False

        if self.verbose:
            print(f"\n{'='*60}")
            print("Genetic Algorithm – Lottery Ticket Search")
            print(f"  genome length  : {self.genome_length:,}")
            print(f"  population     : {cfg.population_size}")
            print(f"  generations    : {cfg.min_generations}–{cfg.max_generations}")
            print(f"  rec / mut / mig: {cfg.rec_rate} / {cfg.mut_rate} / {cfg.mig_rate}")
            print(f"  fitness        : {'loss' if cfg.use_loss_fitness else 'accuracy'}")
            print(f"  adaptive AB    : {cfg.use_adaptive_ab}")
            print(f"  post-pruning   : {cfg.post_prune}")
            if cfg.time_limit_seconds:
                print(f"  time limit     : {cfg.time_limit_seconds:.0f}s "
                      f"({cfg.time_limit_seconds/3600:.1f}h)")
            if resume_info:
                print(f"  RESUMING from gen {resume_info['start_generation']}")
            print(f"{'='*60}\n")

        # 1. Initialise or resume
        if resume_info is not None:
            population = resume_info['population']
            best_ever = resume_info['best_ever']
            best_perf_history = resume_info['best_perf_history']
            start_gen = resume_info['start_generation']
            prior_elapsed = resume_info.get('elapsed_seconds', 0.0)
        else:
            population = self._init_population()
            population = self._evaluate_population(population)
            best_perf_history: List[float] = []
            best_ever = population[0].clone()
            start_gen = 0

        # 2. Evolutionary loop
        pbar = tqdm(range(start_gen, cfg.max_generations), desc="GA generation",
                    disable=not self.verbose)
        for gen in pbar:
            # Age everyone
            for ind in population:
                ind.age += 1

            # Genetic operators
            offspring = self._recombine(population)
            mutants = self._mutate(population)
            migrants = self._migrate()

            # Combine and select survivors
            extended = population + offspring + mutants + migrants
            population = self._survive(extended)

            # Track best
            gen_best = population[0]
            best_perf_history.append(gen_best.performance)
            if gen_best.performance > best_ever.performance or (
                gen_best.performance == best_ever.performance
                and gen_best.sparsity > best_ever.sparsity
            ):
                best_ever = gen_best.clone()

            # Stats
            perfs = [ind.performance for ind in population]
            sparsities = [ind.sparsity for ind in population]
            stat = {
                'generation': gen,
                'best_perf': gen_best.performance,
                'best_sparsity': gen_best.sparsity,
                'mean_perf': float(np.mean(perfs)),
                'std_perf': float(np.std(perfs)),
                'mean_sparsity': float(np.mean(sparsities)),
                'pop_size_before_select': len(extended),
                'cache_size': len(self.cache),
                'cache_hit_rate': (
                    self.cache.hits / max(self.cache.hits + self.cache.misses, 1)
                ),
            }
            self.generation_stats.append(stat)

            pbar.set_postfix(
                perf=f"{gen_best.performance:.4f}",
                sp=f"{gen_best.sparsity:.2%}",
                cache=f"{stat['cache_hit_rate']:.0%}",
            )

            # Periodic checkpoint
            if (cfg.checkpoint_dir
                    and cfg.checkpoint_interval > 0
                    and (gen + 1) % cfg.checkpoint_interval == 0):
                elapsed = prior_elapsed + (time.time() - t0)
                self.save_checkpoint(
                    self._checkpoint_path(),
                    population, best_ever, best_perf_history,
                    gen, elapsed, phase="evolve",
                )

            # Time-limit check: save & break if close to deadline
            if self._should_stop_for_time(margin_seconds=120.0):
                elapsed = prior_elapsed + (time.time() - t0)
                ckpt_path = self._checkpoint_path()
                self.save_checkpoint(
                    ckpt_path, population, best_ever, best_perf_history,
                    gen, elapsed, phase="evolve",
                )
                if self.verbose:
                    remaining = self._time_remaining()
                    print(f"\n⏰ Time limit approaching ({remaining:.0f}s left). "
                          f"Checkpoint saved at generation {gen}.")
                stopped_early = True
                break

            # Termination: min generations met AND stagnation
            if gen >= cfg.min_generations - 1:
                if self._is_stagnant(best_perf_history, cfg.stagnation_threshold):
                    if self.verbose:
                        print(f"\nStagnation detected at generation {gen}. Stopping.")
                    break

        evolve_time = prior_elapsed + (time.time() - t0)

        # 3. Post-evolutionary pruning (skip if stopped early)
        if cfg.post_prune and not stopped_early:
            best_ever = self._post_evolutionary_pruning(best_ever)

        total_time = prior_elapsed + (time.time() - t0)

        # Convert back to mask dict
        best_masks = bitvector_to_masks(best_ever.genome, self.mask_template)

        # Final summary
        if self.verbose:
            print(f"\n{'='*60}")
            status = "INTERRUPTED (checkpoint saved)" if stopped_early else "completed"
            print(f"GA {status} in {total_time:.1f}s  ({evolve_time:.1f}s evolution)")
            print(f"  best performance : {best_ever.performance:.4f}")
            print(f"  best sparsity    : {best_ever.sparsity:.4f}")
            print(f"  cache entries    : {len(self.cache)}")
            print(f"{'='*60}\n")

        stats = {
            'generations': self.generation_stats,
            'total_generations': len(self.generation_stats),
            'evolve_time_seconds': evolve_time,
            'total_time_seconds': total_time,
            'best_performance': best_ever.performance,
            'best_sparsity': best_ever.sparsity,
            'cache_entries': len(self.cache),
            'stopped_early': stopped_early,
            'config': {
                'population_size': cfg.population_size,
                'rec_rate': cfg.rec_rate,
                'mut_rate': cfg.mut_rate,
                'mig_rate': cfg.mig_rate,
                'par_rate': cfg.par_rate,
                'min_generations': cfg.min_generations,
                'max_generations': cfg.max_generations,
                'stagnation_threshold': cfg.stagnation_threshold,
                'use_adaptive_ab': cfg.use_adaptive_ab,
                'use_loss_fitness': cfg.use_loss_fitness,
                'post_prune': cfg.post_prune,
            },
        }

        return best_masks, stats
