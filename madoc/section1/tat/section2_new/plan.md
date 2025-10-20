# Evolutionary KAN Implementation Plan: Viability Assessment & Roadmap

## Executive Summary

**Overall Verdict: HIGHLY VIABLE** ✅

All six proposed extensions are implementable with the existing infrastructure. The codebase is well-structured, modular, and already contains many of the foundational pieces needed. Implementation is estimated at 8-12 weeks for a complete system.

**Key Strengths of Current Implementation:**
- Custom KAN implementation with 5 variants (B-spline, Chebyshev, Fourier, Wavelet, RBF)
- Existing pruning infrastructure (`PruningKAN` class with importance metrics)
- Adaptive grid extension capabilities (`AdaptiveGridKAN`)
- Strong optimizer support (Adam, AdamW, L-BFGS, SGD, Levenberg-Marquardt)
- Geophysical forward models already implemented in Section 3
- Modular architecture allowing easy extension
- ~7,200 lines of well-documented utility code

---

## 1. Repository Structure Assessment

### Current Implementation

**KAN Architecture (Section 1):**
- **Core Models** (`section1/models/`):
  - `architectures.py` (488 LOC): MLP, SIREN, standard KAN, PINN, U-Net
  - `kan_variants.py` (753 LOC): 5 KAN variants with different basis functions
  - `kan_modules.py` (528 LOC): Adaptive grid, pruning, symbolic regression
- **Training Infrastructure** (`section1/utils/`):
  - Comprehensive optimizer factory (Adam, L-BFGS, LM, etc.)
  - Parallel training support
  - Evaluation protocols and metrics
  - Baseline comparison framework

**Geophysical Application (Section 3):**
- Forward models for gravity and magnetic anomalies
- Synthetic ore deposit generation
- Ellipsoid-based anomaly computation
- Basic inverse problem setup

**Dependencies:**
- PyTorch ✅
- NumPy ✅
- Matplotlib ✅
- SciPy ✅
- **Missing:** None critical (will need minor additions)

### Code Quality Analysis

**Modularity: 9/10** - Excellent separation of concerns
- Models, experiments, utilities cleanly separated
- Reusable components across sections
- Clear interfaces between modules

**Extensibility: 9/10** - Easy to extend
- Can train multiple networks in parallel (existing infrastructure)
- Architecture modifications supported (grid size, layer changes)
- Training loop hooks available
- Save/load with different architectures works

**Existing Capabilities:**
- ✅ Pruning utilities (`PruningKAN.compute_node_importance()`)
- ✅ Grid extension (`AdaptiveGridKAN.extend_grid()`)
- ✅ Multiple basis functions (5 variants ready)
- ✅ Parallel training support
- ✅ Model comparison framework
- ⚠️ Ensemble utilities (minimal, needs expansion)
- ❌ Evolutionary search (not implemented)
- ❌ Population-based training (not implemented)

---

## 2. Viability Assessment by Extension

### Extension 1: Hierarchical Ensemble of KAN Experts

**Feasibility Score: 5/5** ⭐ (Can implement immediately)

**Rationale:**
- Parallel training already supported
- Pruning infrastructure exists (`PruningKAN.compute_node_importance()`)
- Variable importance via importance scores already computed
- Only need to add clustering and stacking layers

**Implementation Effort:**
- **Lines of Code:** ~400-500
- **Time Estimate:** 1-2 weeks
- **Key Challenges:** None major
  - Clustering logic is straightforward (scikit-learn or custom)
  - Stacking layer is simple PyTorch module
  - Freezing experts well-supported in PyTorch

**Integration Points:**
- **New Files:**
  - `section2_new/ensemble/expert_training.py` (~150 LOC)
  - `section2_new/ensemble/clustering.py` (~100 LOC)
  - `section2_new/ensemble/stacking.py` (~150 LOC)
- **Modify:** None
- **Reuse:**
  - `section1/models/kan_variants.py` (all 5 KAN variants)
  - `section1/models/kan_modules.py` (pruning utilities)
  - Parallel training from existing infrastructure

**Dependencies to Add:**
- `scikit-learn` (for clustering) - optional, can implement custom

---

### Extension 2: Adaptive Densification Based on Node Importance

**Feasibility Score: 4/5** (Minor modifications needed)

**Rationale:**
- `AdaptiveGridKAN` already implements grid extension
- Node importance computation exists in `PruningKAN`
- Need to add **per-node** grid size tracking (currently global)

**Implementation Effort:**
- **Lines of Code:** ~300-400
- **Time Estimate:** 1-2 weeks
- **Key Challenges:**
  - Modify `AdaptiveGridKAN` to support heterogeneous grid sizes per node
  - Track node-specific grid sizes in training loop
  - Efficient selective densification (avoid full model reconstruction)

**Integration Points:**
- **New Files:**
  - `section2_new/adaptive/importance_metrics.py` (~150 LOC)
  - `section2_new/adaptive/selective_densification.py` (~200 LOC)
- **Modify:**
  - `section1/models/kan_modules.py`: Extend `AdaptiveGridKAN` for per-node grids (~50 LOC addition)
- **Reuse:**
  - `AdaptiveGridKAN.extend_grid()` as base
  - `PruningKAN.compute_node_importance()` for importance metrics

**Dependencies:** None

---

### Extension 3: Heterogeneous Basis Functions

**Feasibility Score: 5/5** ⭐ (Can implement immediately)

**Rationale:**
- 5 basis function types already implemented (B-spline, Chebyshev, Fourier, Wavelet, RBF)
- All use same interface pattern
- Just need mixed-basis layer class

**Implementation Effort:**
- **Lines of Code:** ~350-450
- **Time Estimate:** 1-2 weeks
- **Key Challenges:** Minimal
  - Edge-to-basis mapping is dictionary-based
  - Gumbel-softmax for learnable selection is standard PyTorch
  - Heuristic switching needs signal analysis (FFT available in NumPy)

**Integration Points:**
- **New Files:**
  - `section2_new/models/heterogeneous_kan.py` (~300 LOC)
  - `section2_new/models/basis_selection.py` (~150 LOC)
- **Modify:** None
- **Reuse:**
  - All 5 basis classes from `section1/models/kan_variants.py`
  - Existing layer structures

**Dependencies:**
- `scipy.signal` (for signal analysis in heuristic mode) - already installed ✅

---

### Extension 4: Population-Based Training (Multi-Seed Coordination)

**Feasibility Score: 4/5** (Minor modifications needed)

**Rationale:**
- Parallel training supported
- Need to add synchronization mechanisms
- PyTorch parameter averaging is straightforward

**Implementation Effort:**
- **Lines of Code:** ~500-600
- **Time Estimate:** 2-3 weeks
- **Key Challenges:**
  - Gradient/parameter sharing across processes (need shared memory or file-based)
  - Diversity metrics (correlation, structural similarity)
  - Synchronization coordination (every N epochs)

**Integration Points:**
- **New Files:**
  - `section2_new/population/population_trainer.py` (~350 LOC)
  - `section2_new/population/synchronization.py` (~150 LOC)
  - `section2_new/population/diversity_metrics.py` (~150 LOC)
- **Modify:** None (fully modular)
- **Reuse:**
  - Existing training loops
  - All optimizer infrastructure

**Dependencies:**
- PyTorch distributed (optional, for efficient parameter sharing) - already in PyTorch ✅

---

### Extension 5: Evolutionary Architecture Search for KANs

**Feasibility Score: 3/5** (Moderate refactoring required)

**Rationale:**
- Most complex extension
- Need genome representation and evolutionary operators
- Fitness evaluation requires full training runs (computationally expensive)
- All building blocks exist, but integration is non-trivial

**Implementation Effort:**
- **Lines of Code:** ~800-1000
- **Time Estimate:** 3-4 weeks
- **Key Challenges:**
  - Genome-to-architecture instantiation (dynamic model building)
  - Efficient fitness evaluation (parallel, possibly with early stopping)
  - Multi-objective optimization (Pareto frontier)
  - Handling variable-length architectures (different layer counts)

**Integration Points:**
- **New Files:**
  - `section2_new/evolution/genome.py` (~200 LOC)
  - `section2_new/evolution/operators.py` (~250 LOC - mutation, crossover, selection)
  - `section2_new/evolution/evolutionary_search.py` (~350 LOC)
  - `section2_new/evolution/fitness.py` (~200 LOC)
- **Modify:**
  - Minor: Add architecture instantiation from genome dict
- **Reuse:**
  - All KAN variants
  - All optimizers
  - Existing evaluation protocols

**Dependencies:**
- `deap` (evolutionary algorithms library) - optional, can implement custom
- Multiprocessing for parallel fitness evaluation - standard library ✅

---

### Extension 6: Geophysical Application with Physics Constraints

**Feasibility Score: 4/5** (Minor modifications needed)

**Rationale:**
- Section 3 already has forward models (gravity, magnetic)
- Synthetic data generation exists
- Just need to integrate with evolutionary/ensemble approaches
- Physics-informed loss functions are straightforward additions

**Implementation Effort:**
- **Lines of Code:** ~600-700
- **Time Estimate:** 2-3 weeks
- **Key Challenges:**
  - Integrating physics constraints into fitness/loss functions
  - Uncertainty quantification from ensembles (mostly statistics)
  - Realistic noise modeling
  - Validation with known solutions

**Integration Points:**
- **New Files:**
  - `section2_new/geophysics/physics_informed_fitness.py` (~250 LOC)
  - `section2_new/geophysics/uncertainty_quantification.py` (~200 LOC)
  - `section2_new/geophysics/scenario_generator.py` (~200 LOC)
- **Modify:**
  - `section3/data/forward_models.py`: Add more realistic forward models (~100 LOC)
  - `section3/data/synthetic_anomalies.py`: Enhance noise models (~50 LOC)
- **Reuse:**
  - Existing forward models (`section3/data/forward_models.py`)
  - All synthetic data generation
  - Ensemble infrastructure from Extension 1

**Dependencies:** None (all physics already implemented)

---

## 3. Prioritized Implementation Roadmap

### Phase 1: Quick Wins (Weeks 1-3)

**Goal:** Build foundational ensemble and adaptive capabilities

#### Week 1: Extension 1 - Hierarchical Ensemble (5/5 feasibility)
**Deliverables:**
1. Multi-seed expert training framework
2. Variable importance and pruning-based analysis
3. Basic ensemble averaging

**Files to Create:**
- `section2_new/ensemble/expert_training.py`
- `section2_new/ensemble/variable_importance.py`
- `section2_new/utils/parallel_training.py`

**Test Case:**
```python
# Train 10 KAN experts on simple 1D function
experts = train_expert_ensemble(
    n_experts=10,
    task='sin_1d',
    seeds=range(10)
)
# Average predictions
ensemble_pred = average_predictions(experts, X_test)
```

#### Week 2: Extension 3 - Heterogeneous Basis Functions (5/5 feasibility)
**Deliverables:**
1. Mixed-basis KAN layer
2. Fixed basis assignment
3. Validation on multiple task types

**Files to Create:**
- `section2_new/models/heterogeneous_kan.py`
- `section2_new/models/basis_library.py`

**Test Case:**
```python
# Create KAN with different bases per edge
kan = HeterogeneousBasisKAN(
    architecture=[2, 5, 5, 1],
    basis_assignments={
        'layer_0_edge_0': 'fourier',  # Periodic input
        'layer_0_edge_1': 'bspline',  # Smooth input
        'layer_1': 'chebyshev'        # All layer 1 edges
    }
)
```

#### Week 3: Extension 1 Continued - Stacking and Clustering
**Deliverables:**
1. Clustering by variable usage
2. Stacked ensemble architecture
3. Selective ensemble methods

**Files to Create:**
- `section2_new/ensemble/clustering.py`
- `section2_new/ensemble/stacking.py`

---

### Phase 2: Medium Complexity (Weeks 4-7)

**Goal:** Implement adaptive and population-based training

#### Weeks 4-5: Extension 2 - Adaptive Densification (4/5 feasibility)
**Deliverables:**
1. Per-node importance tracking
2. Selective grid densification
3. Comparison: uniform vs. adaptive densification

**Files to Create:**
- `section2_new/adaptive/importance_tracker.py`
- `section2_new/adaptive/selective_densification.py`

**Modifications:**
- Extend `section1/models/kan_modules.py::AdaptiveGridKAN`

**Test Case:**
```python
# Start with coarse grid, adaptively densify important nodes
kan = AdaptiveSelectiveKAN(
    architecture=[2, 5, 1],
    initial_grid=3,
    max_grid=10
)
# Training loop with periodic densification
for epoch in range(1000):
    train_step(kan, X, y)
    if epoch % 100 == 0:
        kan.densify_top_k_nodes(k=3, delta_grid=2)
```

#### Weeks 6-7: Extension 4 - Population-Based Training (4/5 feasibility)
**Deliverables:**
1. Population trainer with synchronization
2. Gradient sharing and parameter averaging
3. Diversity metrics
4. Comparison against independent training

**Files to Create:**
- `section2_new/population/population_trainer.py`
- `section2_new/population/synchronization.py`
- `section2_new/population/diversity_metrics.py`

**Test Case:**
```python
# Train population with periodic synchronization
pop_trainer = PopulationBasedKANTraining(
    population_size=20,
    architecture=[2, 5, 1],
    sync_method='parameter_averaging',
    sync_frequency=50
)
pop_trainer.train(X, y, epochs=500)
final_ensemble = pop_trainer.get_population()
```

---

### Phase 3: Advanced Features (Weeks 8-12)

**Goal:** Full evolutionary system and geophysical application

#### Weeks 8-10: Extension 5 - Evolutionary Architecture Search (3/5 feasibility)
**Deliverables:**
1. Genome representation for KAN architectures
2. Evolutionary operators (mutation, crossover, selection)
3. Parallel fitness evaluation
4. Multi-objective optimization

**Files to Create:**
- `section2_new/evolution/genome.py`
- `section2_new/evolution/operators.py`
- `section2_new/evolution/evolutionary_search.py`
- `section2_new/evolution/fitness.py`
- `section2_new/evolution/pareto.py`

**Test Case:**
```python
# Evolve optimal KAN architecture
evolver = EvolutionaryKANSearch(
    population_size=30,
    n_generations=50,
    objectives=['accuracy', 'complexity', 'speed']
)
best_genomes = evolver.evolve(X_train, y_train, X_val, y_val)
pareto_front = evolver.get_pareto_frontier()
```

#### Weeks 11-12: Extension 6 - Geophysical Application (4/5 feasibility)
**Deliverables:**
1. Physics-informed fitness functions
2. Uncertainty quantification from ensembles
3. Enhanced synthetic scenario generation
4. Validation experiments on iron ore detection

**Files to Create:**
- `section2_new/geophysics/physics_constraints.py`
- `section2_new/geophysics/uncertainty_quantification.py`
- `section2_new/geophysics/enhanced_scenarios.py`

**Integration:**
- Connect ensemble/evolutionary approaches to Section 3

**Test Case:**
```python
# Evolve ensemble for iron ore inversion with physics constraints
geo_evolver = GeophysicsEvolutionaryKAN(
    task='magnetic_inversion',
    physics_constraints=['depth>0', 'susceptibility_range'],
    ensemble_size=15
)
results = geo_evolver.run(
    observations=magnetic_data,
    uncertainty_quantification=True
)
print(f"Depth: {results['mean_depth']} ± {results['std_depth']}")
```

---

## 4. Proposed Directory Structure

```
section2_new/
├── README.md                                  # Overview and quick start
├── IMPLEMENTATION_PLAN.md                    # This file (detailed plan)
│
├── models/                                    # New model architectures
│   ├── __init__.py
│   ├── heterogeneous_kan.py                  # Extension 3: Mixed-basis KAN
│   ├── ensemble_kan.py                       # Extension 1: Ensemble wrapper
│   └── adaptive_selective_kan.py             # Extension 2: Per-node adaptive
│
├── ensemble/                                  # Extension 1: Ensemble methods
│   ├── __init__.py
│   ├── expert_training.py                    # Multi-seed expert training
│   ├── variable_importance.py                # Variable usage analysis
│   ├── clustering.py                          # Cluster experts by specialization
│   └── stacking.py                            # Hierarchical ensemble
│
├── adaptive/                                  # Extension 2: Adaptive densification
│   ├── __init__.py
│   ├── importance_tracker.py                 # Track per-node importance
│   └── selective_densification.py            # Selective grid refinement
│
├── population/                                # Extension 4: Population-based training
│   ├── __init__.py
│   ├── population_trainer.py                 # Main population trainer
│   ├── synchronization.py                    # Gradient/parameter sharing
│   └── diversity_metrics.py                  # Measure population diversity
│
├── evolution/                                 # Extension 5: Evolutionary search
│   ├── __init__.py
│   ├── genome.py                              # KAN genome representation
│   ├── operators.py                           # Mutation, crossover, selection
│   ├── evolutionary_search.py                # Main evolution loop
│   ├── fitness.py                             # Fitness evaluation
│   └── pareto.py                              # Multi-objective optimization
│
├── geophysics/                                # Extension 6: Geophysical application
│   ├── __init__.py
│   ├── physics_constraints.py                # Physics-informed losses
│   ├── uncertainty_quantification.py         # Ensemble uncertainty
│   ├── enhanced_scenarios.py                 # Advanced synthetic data
│   └── validation.py                          # Ground truth validation
│
├── experiments/                               # Experiment scripts
│   ├── __init__.py
│   ├── exp_1_ensemble_basics.py              # Phase 1: Basic ensemble
│   ├── exp_2_heterogeneous_basis.py          # Phase 1: Mixed bases
│   ├── exp_3_adaptive_densification.py       # Phase 2: Adaptive grids
│   ├── exp_4_population_training.py          # Phase 2: Population-based
│   ├── exp_5_evolutionary_search.py          # Phase 3: Evolution
│   ├── exp_6_geophysics_application.py       # Phase 3: Iron ore
│   └── run_all_experiments.py                # Batch runner
│
├── utils/                                     # Utilities
│   ├── __init__.py
│   ├── parallel_training.py                  # Parallel training utilities
│   ├── model_serialization.py                # Save/load variable architectures
│   └── visualization.py                       # Ensemble/evolution plots
│
├── visualization/                             # Visualization
│   ├── __init__.py
│   ├── ensemble_plots.py                     # Ensemble diversity plots
│   ├── evolution_plots.py                    # Evolution progress plots
│   └── uncertainty_plots.py                  # Uncertainty quantification
│
├── tests/                                     # Unit tests
│   ├── test_ensemble.py
│   ├── test_heterogeneous.py
│   ├── test_adaptive.py
│   ├── test_population.py
│   ├── test_evolution.py
│   └── test_geophysics.py
│
└── results/                                   # Experimental results
    ├── phase1_ensemble/
    ├── phase2_population/
    └── phase3_evolution/
```

**Total New Code:** ~4,000-5,000 LOC
**Reused Code:** ~3,000 LOC from Section 1

---

## 5. Key Implementation Recommendations

### Architecture Design

**1. Use Composition Over Inheritance**
```python
# Good: Wrap existing KAN variants
class EnsembleKAN:
    def __init__(self, base_kan_class, n_experts=10):
        self.experts = [base_kan_class(...) for _ in range(n_experts)]

# Avoid: Deep inheritance chains
```

**2. Maintain Backward Compatibility**
- All new modules should be in `section2_new/`
- Do not modify Section 1 core files unless absolutely necessary
- Extensions should be opt-in, not breaking changes

**3. Modular Genome Representation**
```python
@dataclass
class KANGenome:
    architecture: List[int]  # [2, 5, 5, 1]
    basis_types: Dict[str, str]  # 'layer_0': 'fourier'
    grid_sizes: Dict[str, int]  # 'layer_0': 5
    hyperparams: Dict[str, Any]  # optimizer, lr, etc.

    def to_model(self) -> nn.Module:
        """Instantiate KAN from genome"""
        # Use existing KAN classes from section1
        pass
```

---

### Testing Strategy

**Unit Tests (Each Extension):**
1. **Ensemble:** Test that 10 experts train independently, predictions average correctly
2. **Heterogeneous:** Test mixed-basis layers compute correct output shapes
3. **Adaptive:** Test that important nodes get densified, unimportant stay coarse
4. **Population:** Test synchronization (parameters actually update across population)
5. **Evolution:** Test genome → model → genome round-trip, crossover/mutation
6. **Geophysics:** Test physics constraints penalize invalid solutions

**Integration Tests:**
1. Train ensemble on simple function (sin, polynomial) → verify lower variance
2. Evolve architecture on 1D task → verify fitness improves over generations
3. Apply to Section 3 synthetic iron ore → verify uncertainty quantification

**Benchmarks:**
1. Compare ensemble vs. single best KAN on Section 1 tasks
2. Compare adaptive vs. uniform densification (accuracy vs. compute)
3. Compare evolved architectures vs. hand-designed
4. Geophysical inversion: RMSE, uncertainty calibration

---

### Computational Considerations

**Parallelization:**
- **Ensemble training:** Trivially parallel (10-20 independent KANs)
  - Use `torch.multiprocessing` or `joblib` for parallel training
  - Each expert trains on separate CPU/GPU
- **Evolutionary search:** Parallel fitness evaluation
  - Population size 20-50: embarrassingly parallel
  - Use multiprocessing pool for fitness evaluation
- **Population-based training:** Requires synchronization
  - Use shared memory (PyTorch distributed) or file-based checkpointing

**Memory/Speed:**
- **Ensemble:** 10-20 KANs in memory is feasible (each ~100KB-1MB)
- **Evolution:** Only store population (20-50 models), not all generations
- **Bottlenecks:** Fitness evaluation (full training runs)
  - Mitigation: Early stopping, reduced epochs for initial generations
  - Use Section 1's runtime budgeting infrastructure

**Experiment Tracking:**
- Current solution: Manual logging
- **Recommendation:** Add lightweight tracking
  - Option 1: TensorBoard (PyTorch-native, no extra dependency)
  - Option 2: WandB (more features, requires account)
  - Option 3: Custom JSON logging (simplest, already have JSON support)

**Estimated Compute Requirements:**
- Phase 1 (Ensemble): 10-20 hours total (parallelizable to 2-3 hours wall time)
- Phase 2 (Population): 20-30 hours (more sequential due to synchronization)
- Phase 3 (Evolution): 50-100 hours (highly parallel, ~10-15 hours wall time with 10 GPUs)

---

## 6. Missing Dependencies and Blockers

### Dependencies to Add

**Essential:**
- None! All essential dependencies already installed

**Optional (for enhanced features):**
- `scikit-learn` (clustering, PCA): `pip install scikit-learn`
- `deap` (evolutionary algorithms): `pip install deap` (optional, can implement custom)
- `tensorboard` (experiment tracking): `pip install tensorboard`

**Recommended `requirements_section2_new.txt`:**
```
# Already installed (from main requirements.txt)
numpy
torch
matplotlib
scipy

# New optional dependencies for Section 2 New
scikit-learn>=1.0.0  # Clustering, dimensionality reduction
# deap>=1.3.0        # Optional: evolutionary algorithms library
# tensorboard>=2.10  # Optional: experiment tracking
```

### Potential Blockers (and Mitigations)

**1. Computational Cost of Evolutionary Search**
- **Issue:** Training 20-50 KANs × 50 generations = 1000-2500 full training runs
- **Mitigation:**
  - Start with small population (10-15) and fewer generations (20-30)
  - Use reduced epochs for fitness evaluation (100-200 instead of 1000+)
  - Early stopping in fitness evaluation
  - Reuse Section 1's runtime budgeting

**2. Heterogeneous Grid Sizes (Per-Node)**
- **Issue:** Current `AdaptiveGridKAN` uses global grid size
- **Mitigation:**
  - Modify to store grid sizes per layer/node (minor refactor)
  - Use dictionary mapping: `{layer_idx: {node_idx: grid_size}}`
  - Estimated effort: 50-100 LOC modification

**3. Genome-to-Architecture Instantiation**
- **Issue:** Need to dynamically build models from genome dictionaries
- **Mitigation:**
  - Already have 5 KAN variants with consistent interfaces
  - Factory pattern: `create_kan_from_genome(genome)`
  - Estimated effort: 100-150 LOC

**4. No Fundamental Blockers Identified** ✅

---

## 7. Concrete Next Steps

### Immediate Actions (This Week)

**Step 1: Set Up Infrastructure**
```bash
cd /Users/main/Desktop/help/KAN_Repo
mkdir -p section2_new/{ensemble,adaptive,population,evolution,geophysics,models,experiments,utils,visualization,tests,results}
touch section2_new/README.md
cp requirements.txt section2_new/requirements_section2_new.txt
```

**Step 2: Create First Minimal Working Example - Expert Ensemble**

File: `section2_new/ensemble/expert_training.py`

```python
"""Train ensemble of KAN experts with different random seeds."""

import torch
import torch.nn as nn
from typing import List, Dict, Any
import sys
sys.path.append('../section1')
from section1.models.architectures import KAN
from section1.utils.optimizer_factory import create_optimizer

class KANExpertEnsemble:
    """Ensemble of KAN models trained with different random seeds."""

    def __init__(
        self,
        architecture: List[int],
        n_experts: int = 10,
        kan_variant: str = 'bspline'
    ):
        self.architecture = architecture
        self.n_experts = n_experts
        self.experts: List[nn.Module] = []
        self.seeds = list(range(n_experts))

    def train_experts(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        epochs: int = 500,
        lr: float = 0.001
    ) -> Dict[str, Any]:
        """Train all experts independently."""
        from section1.models.kan_variants import ChebyshevKAN, FourierKAN

        results = {'individual_losses': []}

        for i, seed in enumerate(self.seeds):
            torch.manual_seed(seed)

            # Create expert
            expert = ChebyshevKAN(self.architecture)
            optimizer = torch.optim.Adam(expert.parameters(), lr=lr)

            # Train
            for epoch in range(epochs):
                optimizer.zero_grad()
                pred = expert(X_train)
                loss = nn.MSELoss()(pred, y_train)
                loss.backward()
                optimizer.step()

            self.experts.append(expert)
            results['individual_losses'].append(loss.item())
            print(f"Expert {i+1}/{self.n_experts} trained. Loss: {loss.item():.6f}")

        return results

    def predict(self, X: torch.Tensor, method: str = 'mean') -> torch.Tensor:
        """Ensemble prediction."""
        preds = torch.stack([expert(X) for expert in self.experts])

        if method == 'mean':
            return preds.mean(dim=0)
        elif method == 'median':
            return preds.median(dim=0)[0]
        else:
            raise ValueError(f"Unknown method: {method}")

    def predict_with_uncertainty(self, X: torch.Tensor):
        """Predict with epistemic uncertainty."""
        preds = torch.stack([expert(X) for expert in self.experts])
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        return mean, std


if __name__ == '__main__':
    # Test on simple 1D function
    X = torch.linspace(-3, 3, 100).reshape(-1, 1)
    y = torch.sin(X)

    ensemble = KANExpertEnsemble(
        architecture=[1, 5, 1],
        n_experts=10
    )

    results = ensemble.train_experts(X, y, epochs=200)

    # Test ensemble prediction
    X_test = torch.linspace(-3, 3, 50).reshape(-1, 1)
    y_pred, uncertainty = ensemble.predict_with_uncertainty(X_test)

    print(f"\nEnsemble test MSE: {nn.MSELoss()(y_pred, torch.sin(X_test)):.6f}")
    print(f"Mean uncertainty: {uncertainty.mean():.6f}")
```

**Step 3: Validate Minimal Example**
```bash
cd section2_new/ensemble
python expert_training.py
```

**Expected Output:**
```
Expert 1/10 trained. Loss: 0.000234
Expert 2/10 trained. Loss: 0.000198
...
Ensemble test MSE: 0.000156
Mean uncertainty: 0.002341
```

---

### Week 1 Detailed Plan

**Day 1-2:** Ensemble Infrastructure
- Implement `expert_training.py` (above)
- Add variable importance extraction
- Test on Section 1 function approximation tasks

**Day 3-4:** Ensemble Clustering
- Implement Jaccard similarity for variable usage
- K-means clustering on importance vectors
- Visualize clusters

**Day 5:** Stacking Layer
- Simple meta-learner (linear layer on top of expert outputs)
- Test frozen vs. fine-tuned modes

**Deliverable:** Working ensemble that reduces variance on Section 1 tasks

---

## 8. Success Criteria

### Phase 1 (Weeks 1-3)
- ✅ Train 10-20 KAN experts with different seeds
- ✅ Ensemble predictions have lower variance than single model
- ✅ Mixed-basis KAN trains successfully on 1D/2D tasks
- ✅ Stacked ensemble outperforms simple averaging

### Phase 2 (Weeks 4-7)
- ✅ Adaptive densification allocates more grid points to important nodes
- ✅ Adaptive approach achieves equal accuracy with fewer total parameters
- ✅ Population-based training shows diversity metrics evolve over time
- ✅ Synchronized population outperforms independent training

### Phase 3 (Weeks 8-12)
- ✅ Evolutionary search finds architectures competitive with hand-designed
- ✅ Pareto frontier shows accuracy/complexity trade-offs
- ✅ Geophysical application: ensemble provides calibrated uncertainty
- ✅ Physics constraints reduce invalid solutions by >80%
- ✅ Variable importance consensus identifies critical sensors

---

## 9. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Computational cost too high | Medium | High | Start small (10 experts, 20 generations), use runtime budgeting |
| Genome representation too rigid | Low | Medium | Use flexible dict-based representation, test early |
| Physics constraints don't improve results | Low | Low | Validate on known solutions first, tune weights |
| Ensemble doesn't reduce error | Low | High | Ensure diverse initialization (confirmed via seeds) |
| Integration with Section 3 breaks existing code | Very Low | Medium | Keep section2_new fully isolated, use imports only |

---

## 10. Conclusion

**Implementation is HIGHLY VIABLE.** The existing codebase provides an excellent foundation:

**Strengths:**
- ✅ Well-architected, modular code
- ✅ 5 KAN variants ready to use
- ✅ Pruning and adaptive grid infrastructure
- ✅ Geophysical forward models implemented
- ✅ Strong optimizer support
- ✅ Parallel training capabilities

**Minor Gaps (easily addressed):**
- Ensemble utilities (straightforward to add)
- Per-node grid tracking (small modification)
- Evolutionary framework (standard implementation)

**Timeline:** 8-12 weeks for full implementation
**Effort:** ~4,000-5,000 LOC new code + ~100-200 LOC modifications

**Recommendation:** Proceed with phased implementation. Start with Phase 1 (ensemble + heterogeneous basis) to validate approach and build momentum, then expand to full evolutionary system.

The geophysical application is particularly well-positioned, as Section 3 already has the forward models and synthetic data generation. The evolutionary ensemble approach could provide significant value for uncertainty quantification in iron ore exploration.

---

## Appendix: File Modification Summary

### Files to Create (~25 new files)

**Models (3 files):**
- `section2_new/models/heterogeneous_kan.py`
- `section2_new/models/ensemble_kan.py`
- `section2_new/models/adaptive_selective_kan.py`

**Ensemble (4 files):**
- `section2_new/ensemble/expert_training.py`
- `section2_new/ensemble/variable_importance.py`
- `section2_new/ensemble/clustering.py`
- `section2_new/ensemble/stacking.py`

**Adaptive (2 files):**
- `section2_new/adaptive/importance_tracker.py`
- `section2_new/adaptive/selective_densification.py`

**Population (3 files):**
- `section2_new/population/population_trainer.py`
- `section2_new/population/synchronization.py`
- `section2_new/population/diversity_metrics.py`

**Evolution (5 files):**
- `section2_new/evolution/genome.py`
- `section2_new/evolution/operators.py`
- `section2_new/evolution/evolutionary_search.py`
- `section2_new/evolution/fitness.py`
- `section2_new/evolution/pareto.py`

**Geophysics (4 files):**
- `section2_new/geophysics/physics_constraints.py`
- `section2_new/geophysics/uncertainty_quantification.py`
- `section2_new/geophysics/enhanced_scenarios.py`
- `section2_new/geophysics/validation.py`

**Experiments (6 files):**
- 6 experiment scripts for each extension

### Files to Modify (2 files)

**Minor modifications:**
- `section1/models/kan_modules.py`: Extend `AdaptiveGridKAN` for per-node grids (~50 LOC)
- `section3/data/forward_models.py`: Enhanced forward models (~100 LOC)

**Total:** ~150 LOC of modifications to existing code (minimal risk)
