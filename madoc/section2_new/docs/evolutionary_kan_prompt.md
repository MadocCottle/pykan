# Prompt for Claude Code: Evolutionary KAN Implementation Viability Assessment

## Context

I have a repository with three main sections:
1. **Section 1**: Testing MLPs, SIRENs, and KANs (Kolmogorov-Arnold Networks) on simple function approximation
2. **Section 2**: Testing different optimizers for KANs
3. **Section 3**: Applying KANs to iron ore discovery using synthetically generated geophysical data

I want to extend this repo with novel evolutionary and ensemble approaches for KANs. Please analyze the repository structure and assess the viability of implementing the following ideas.

---

## Proposed Extensions: Evolutionary and Ensemble KAN Architectures

### **Core Concept**
KANs trained from different random seeds discover different functional relationships (different decompositions of the underlying function). I want to leverage this through:
1. Ensemble learning with diverse KAN experts
2. Evolutionary architecture search specifically designed for KAN structure
3. Adaptive network modification based on learned importance
4. Application to geophysical inverse problems (iron ore exploration)

---

## Detailed Implementation Ideas

### **1. Hierarchical Ensemble of KAN Experts**

**Goal**: Train multiple KANs with different seeds, identify which variables/relationships each specializes in, then create hierarchical architectures.

**Implementation Requirements**:
- Train K KANs (e.g., K=10-20) with different random seeds in parallel
- For each trained KAN:
  - Apply pruning to identify "active variable set" (which inputs it relies on)
  - Compute variable importance: sum of |weights| on edges connected to each input
  - Store pruning masks and learned function structure
- Cluster KANs by variable usage similarity (Jaccard similarity, cosine similarity, or custom metric)
- Create hierarchical architecture options:
  - **Stacked ensemble**: outputs of base KANs → new hidden layer → final output
  - **Option A**: Freeze base KANs, train only top layer
  - **Option B**: Fine-tune entire stacked architecture with small learning rate
  - **Option C**: Selective ensemble - choose subset based on clustering

**Technical Needs**:
```python
class KANEnsemble:
    def __init__(self, num_experts, base_architecture):
        self.experts = []  # List of trained KANs
        self.variable_importance = []  # Per-expert importance vectors
        self.pruning_masks = []  # Per-expert pruning masks
        self.meta_learner = None  # Top layer for stacking
    
    def train_experts(self, X, y, seeds):
        """Train K KANs with different random seeds"""
        
    def compute_variable_importance(self):
        """For each KAN, compute which inputs are most used"""
        
    def cluster_by_variables(self, method='jaccard'):
        """Cluster KANs by variable usage patterns"""
        
    def build_stacked_model(self, freeze_experts=True):
        """Create hierarchical: [KAN1, ..., KANK] → meta_layer → output"""
        
    def predict_ensemble(self, X, method='average'):
        """Methods: 'average', 'stacked', 'weighted', 'selective'"""
```

---

### **2. Adaptive Densification Based on Node Importance**

**Goal**: Dynamically increase grid resolution (density) for important nodes/edges while keeping less important ones sparse.

**Implementation Requirements**:
- Compute node/edge importance metrics during or after training:
  - **Gradient-based**: `I = ||∂L/∂θ_node||_2` (gradient magnitude)
  - **Activation-based**: `I = Var(activation)` across training samples
  - **Ablation-based**: `I = L(full_model) - L(model_without_node)`
  - **Weight-based**: Sum of |weights| on edges connected to node
- Implement selective densification:
  - Start with coarse grid (e.g., G=3)
  - Every N epochs, identify top-k important nodes (e.g., top 20%)
  - Increase grid size for those nodes: G → G + Δ (e.g., 3→5→7)
- Variants to test:
  - One-shot densification (single increase at epoch T)
  - Iterative densification (periodic increases)
  - Control: uniform densification (increase all equally)

**Technical Needs**:
```python
class AdaptiveKAN:
    def __init__(self, base_architecture, initial_grid_size=3):
        self.kan = KAN(base_architecture, grid_size=initial_grid_size)
        self.node_importance = {}  # Maps node_id → importance score
        self.grid_sizes = {}  # Maps node_id → current grid size
    
    def compute_importance(self, method='gradient'):
        """
        method: 'gradient', 'activation', 'ablation', 'weight'
        Returns: dict of node_id → importance_score
        """
        
    def densify_nodes(self, threshold_percentile=80, delta_G=2):
        """
        Increase grid size for nodes above importance threshold
        threshold_percentile: e.g., 80 means top 20% of nodes
        delta_G: how much to increase grid size
        """
        
    def adaptive_training_loop(self, X, y, densify_every_n_epochs=100):
        """
        Training loop with periodic densification
        """
```

---

### **3. Heterogeneous Basis Functions**

**Goal**: Allow different edges/nodes to use different basis function types (B-splines, Fourier, wavelets, RBFs, polynomials).

**Implementation Requirements**:
- Extend KAN to support multiple basis types per edge
- Basis function library:
  - **B-splines** (current default) - smooth, local support
  - **Fourier** - periodic patterns
  - **Wavelets** (Haar, Daubechies) - multi-resolution, localized
  - **RBFs** (Gaussian, multiquadric) - radial symmetry
  - **Polynomials** (Legendre, Chebyshev) - simple relationships
- Basis selection strategies:
  - **Fixed assignment**: Manually assign or randomly initialize
  - **Learnable selection**: Gumbel-softmax for differentiable discrete choice
  - **Heuristic switching**: Based on signal properties (smoothness, periodicity, locality)

**Technical Needs**:
```python
class HeterogeneousBasisKAN:
    def __init__(self, architecture, basis_library):
        """
        basis_library: dict of available basis functions
            {'bspline': BSplineBasis, 'fourier': FourierBasis, ...}
        """
        self.basis_assignments = {}  # Maps edge_id → basis_type
        
    def assign_basis(self, edge_id, basis_type):
        """Assign specific basis to an edge"""
        
    def learn_basis_selection(self, temperature=1.0):
        """
        Use Gumbel-softmax to make basis selection differentiable
        Each edge has: p(basis_type) distribution
        Sample: basis ~ Gumbel-Softmax(p, τ)
        """
        
    def heuristic_basis_switching(self, X, y):
        """
        Analyze local signal properties and switch basis:
        - High frequency content → Fourier
        - Smooth regions → B-spline
        - Localized features → Wavelet/RBF
        """
```

---

### **4. Population-Based Training (Multi-Seed Coordination)**

**Goal**: Train population of KANs in parallel with periodic information sharing to explore diverse solutions while leveraging collective knowledge.

**Implementation Requirements**:
- Initialize population of K KANs with different random seeds
- Each KAN trains independently for T epochs
- Periodic synchronization (every T epochs):
  - **Gradient sharing**: Update using own gradient + weighted sum of others' gradients
  - **Parameter averaging**: Federated learning style, θ_i ← (1-α)θ_i + α·mean(θ_j)
  - **Knowledge distillation**: Train each to match ensemble predictions
  - **Best practices sharing**: Copy hyperparameters from best performers
- Track diversity metrics: correlation between predictions, structural similarity
- Final evaluation: ensemble vs. best individual

**Technical Needs**:
```python
class PopulationBasedKANTraining:
    def __init__(self, population_size, base_architecture):
        self.population = [KAN(base_architecture) for _ in range(population_size)]
        self.optimizers = [...]  # One per KAN
        self.performance_history = []  # Track each KAN's validation loss
        
    def train_step(self, X_batch, y_batch):
        """Independent training step for each KAN in population"""
        
    def synchronize(self, method='gradient_sharing', alpha=0.1):
        """
        method: 'gradient_sharing', 'parameter_averaging', 
                'knowledge_distillation', 'hyperparameter_sharing'
        alpha: mixing coefficient
        """
        
    def compute_diversity(self):
        """
        Metrics:
        - Prediction correlation across population
        - Structural diversity (pruning mask differences)
        - Functional diversity (which variables each uses)
        """
        
    def population_training_loop(self, X, y, sync_every_n_epochs=50):
        """Full training with periodic synchronization"""
```

---

### **5. Evolutionary Architecture Search for KANs**

**Goal**: Use genetic algorithms to evolve optimal KAN architectures, basis functions, pruning strategies, and hyperparameters.

**Implementation Requirements**:

#### **Genome Representation**:
```python
@dataclass
class KANGenome:
    # Architecture
    layer_widths: List[int]  # e.g., [2, 5, 5, 1]
    
    # Basis functions
    basis_types: Dict[str, str]  # edge_id → 'bspline'|'fourier'|'wavelet'|...
    grid_sizes: Dict[str, int]  # edge_id → G ∈ [3, 10]
    
    # Pruning
    pruning_threshold: float  # ∈ [0.001, 0.1]
    pruning_mask: Optional[Dict]  # Applied after training
    
    # Training hyperparameters
    optimizer_type: str  # 'adam'|'sgd'|'lbfgs'
    learning_rate: float
    weight_decay: float
    
    # Physics constraints (for geophysical applications)
    physics_weight: float  # Weight on physics-informed loss
    
    def mutate(self, mutation_rate=0.2):
        """Apply random mutations"""
        
    def crossover(self, other_genome):
        """Combine with another genome"""
```

#### **Evolutionary Operations**:
```python
class EvolutionaryKANSearch:
    def __init__(self, population_size=20, n_generations=50):
        self.population = []
        self.fitness_history = []
        self.generation = 0
        
    def initialize_population(self, init_strategy='random'):
        """
        Create diverse initial population
        Strategies: 'random', 'latin_hypercube', 'seeded_with_best'
        """
        
    def evaluate_fitness(self, genome, X_train, y_train, X_val, y_val):
        """
        1. Instantiate KAN from genome
        2. Train for E epochs
        3. Compute fitness = f(val_loss, complexity, physics_constraint)
        
        Fitness components:
        - Data fit: MSE on validation set
        - Complexity penalty: number of parameters
        - Physics penalty: violation of known constraints (for geophysics)
        """
        
    def selection(self, fitness_scores, method='tournament'):
        """
        Methods: 'tournament', 'roulette', 'rank', 'elitism'
        Keep top performers for next generation
        """
        
    def crossover(self, parent1, parent2):
        """
        Combine architectures:
        - Layer widths: inherit from random parent per layer
        - Basis functions: mix assignments
        - Hyperparameters: average or randomly select
        """
        
    def mutation(self, genome, mutation_rate=0.2):
        """
        Possible mutations:
        - Add/remove neurons from layers (±1 to ±3)
        - Change basis function for random edges
        - Adjust grid size (±1 to ±2)
        - Perturb hyperparameters (learning rate *= uniform(0.5, 2.0))
        - Toggle pruning threshold
        """
        
    def evolve(self, X_train, y_train, X_val, y_val):
        """
        Main evolution loop:
        for generation in range(n_generations):
            1. Evaluate fitness of all genomes
            2. Selection (keep top 50%)
            3. Crossover (pair parents, create offspring)
            4. Mutation (randomly modify offspring)
            5. Repopulate to original size
            6. Log best genome and diversity metrics
        """
```

#### **Advanced Features**:
```python
# Species/niching to maintain diversity
def speciate(self, population, similarity_threshold=0.7):
    """
    Group similar genomes into species
    Prevents premature convergence
    Within each species, apply selection separately
    """
    
# Multi-objective optimization
def pareto_frontier(self, population, objectives=['accuracy', 'complexity', 'speed']):
    """
    Find Pareto-optimal solutions
    No single best, but trade-off between objectives
    """
    
# Transfer learning initialization
def seed_with_pretrained(self, pretrained_genome, population_fraction=0.2):
    """
    Include successful genomes from previous runs
    E.g., best architecture from Pilbara iron ore → seed Carajás search
    """
```

---

### **6. Geophysical Application: Iron Ore Exploration**

**Goal**: Apply evolutionary KAN approaches to synthetic geophysical inverse problems for iron ore discovery.

**Implementation Requirements**:

#### **Forward Models**:
```python
class GeophysicalForwardModel:
    def magnetic_forward(self, ore_body_params):
        """
        Input: [depth, width, length, dip_angle, susceptibility, remanence]
        Output: magnetic_field at sensor locations (nT)
        
        Physics: Magnetic dipole approximation or finite element
        """
        
    def gravity_forward(self, ore_body_params):
        """
        Input: [depth, volume, density_contrast]
        Output: gravity_anomaly at sensor locations (mGal)
        
        Physics: Integration of 1/r² over volume
        """
        
    def em_forward(self, ore_body_params):
        """
        Input: [depth, conductivity, geometry]
        Output: EM_response at multiple frequencies
        
        Physics: Maxwell's equations (simplified)
        """
```

#### **Synthetic Data Generation**:
```python
class IronOreScenarioGenerator:
    def __init__(self):
        self.deposit_types = ['bif', 'magnetite_pipe', 'hematite_lens']
        
    def generate_scenario(self, deposit_type, complexity='medium'):
        """
        Create realistic ore body parameters
        
        BIF (Banded Iron Formation):
        - Depth: 100-800m
        - Dip: 30-70°
        - Shape: Elongated lens
        - Susceptibility: 0.01-0.05 SI
        
        Magnetite Pipe:
        - Depth: 200-600m  
        - Dip: 70-90° (near-vertical)
        - Shape: Cylindrical/irregular
        - Susceptibility: 0.05-0.15 SI
        """
        
    def add_realistic_noise(self, signal, snr_db=20):
        """Add measurement noise typical of field surveys"""
        
    def add_geological_complexity(self, scenario):
        """
        - Regional magnetic trend
        - Multiple interfering bodies
        - Remanent magnetization
        - Anisotropy
        """
```

#### **Physics-Informed Fitness**:
```python
def geophysical_fitness(genome, X, y, scenario_info):
    """
    Fitness for evolutionary search on geophysical inverse problem
    
    Components:
    1. Data misfit: ||predicted - observed||
    2. Physics constraints:
       - Depth > 0
       - Susceptibility ∈ [0.001, 0.2] SI (reasonable for iron ore)
       - Density contrast ∈ [1.5, 3.0] g/cm³
       - Magnetic dipole decay ~ 1/r³
    3. Geological reasonableness:
       - No floating ore bodies (must be connected to surface or known depth)
       - Dip angle consistency with regional structure
       - Volume/tonnage within expected ranges
    4. Complexity penalty: Prefer simpler models (Occam's razor)
    """
    
    # Standard data fit
    kan = build_kan_from_genome(genome)
    predictions = kan(X)
    data_loss = MSE(predictions, y)
    
    # Physics penalties
    physics_loss = 0
    
    # Extract predicted parameters (depth, susceptibility, etc.)
    depth, suscept, density = parse_predictions(predictions)
    
    # Depth constraint
    physics_loss += torch.relu(-depth).sum() * 100  # Heavily penalize negative depth
    
    # Susceptibility bounds
    physics_loss += torch.relu(suscept - 0.2).sum() * 10
    physics_loss += torch.relu(0.001 - suscept).sum() * 10
    
    # Dipole decay check (if enough spatial data)
    if has_spatial_gradients(X):
        predicted_decay_rate = compute_decay_rate(predictions)
        expected_decay = 3.0  # ~1/r³ for dipole
        physics_loss += abs(predicted_decay_rate - expected_decay) * 5
    
    # Geological reasonableness (if scenario info available)
    if scenario_info:
        known_regional_dip = scenario_info['regional_dip']
        predicted_dip = extract_dip(predictions)
        # Prefer dips within ±30° of regional trend
        dip_penalty = max(0, abs(predicted_dip - known_regional_dip) - 30) * 0.1
        physics_loss += dip_penalty
    
    # Complexity
    complexity = count_active_parameters(kan, genome.pruning_mask)
    complexity_loss = complexity / 1000  # Normalize
    
    # Total fitness (lower is better)
    total_loss = data_loss + genome.physics_weight * physics_loss + 0.01 * complexity_loss
    
    return total_loss
```

#### **Uncertainty Quantification**:
```python
class EnsembleUncertainty:
    def __init__(self, evolved_kans):
        self.ensemble = evolved_kans
        
    def predict_with_uncertainty(self, X):
        """
        Returns: mean, std, confidence_interval
        
        For each test point:
        - Get predictions from all ensemble members
        - Mean = average prediction (best estimate)
        - Std = spread (epistemic uncertainty)
        - CI = percentile-based confidence interval
        """
        predictions = [kan(X) for kan in self.ensemble]
        mean = torch.mean(torch.stack(predictions), dim=0)
        std = torch.std(torch.stack(predictions), dim=0)
        
        # 90% confidence interval
        lower = torch.quantile(torch.stack(predictions), 0.05, dim=0)
        upper = torch.quantile(torch.stack(predictions), 0.95, dim=0)
        
        return mean, std, (lower, upper)
    
    def variable_importance_consensus(self):
        """
        Which sensor/input variables are consistently important?
        
        For each evolved KAN:
        - Extract pruning mask or variable importance
        For each input variable:
        - Count how many KANs use it significantly
        - Average importance score across ensemble
        
        Output: consensus variable importance for survey design
        """
```

---

## What I Need From You (Claude Code)

### **1. Repository Structure Assessment**

Please examine the repository and provide:

a) **Current Structure**:
   - What KAN implementation is being used? (Custom, pykan library, other?)
   - How are experiments organized in Sections 1, 2, 3?
   - What data structures/classes exist for:
     - Network definitions
     - Training loops
     - Experiment tracking
     - Visualization
   - What dependencies are installed? (PyTorch, NumPy, matplotlib, etc.)

b) **Code Quality and Modularity**:
   - Is the KAN implementation modular enough to extend?
   - Can we easily:
     - Train multiple networks in parallel?
     - Modify architecture (add/remove layers, change grid sizes)?
     - Hook into training loop for custom callbacks?
     - Save/load models with different architectures?
   - Are there existing utilities for:
     - Pruning?
     - Model comparison/ensembling?
     - Hyperparameter tuning?

### **2. Viability Assessment for Each Extension**

For each of the 6 proposed extensions above, please assess:

**Feasibility Score (1-5)**:
- 5 = Can implement immediately with existing infrastructure
- 4 = Minor modifications needed
- 3 = Moderate refactoring required
- 2 = Significant new code needed
- 1 = Major restructuring or missing dependencies

**Implementation Effort**:
- Estimated lines of code
- Key challenges/blockers
- Dependencies that need to be added
- Existing code that can be reused

**Integration Points**:
- Which files would need modification?
- What new files/modules should be created?
- How would it fit into existing Section 1/2/3 structure?

### **3. Prioritized Implementation Roadmap**

Recommend an implementation order, such as:

**Phase 1 (Quick Wins - 1-2 weeks)**:
- Extensions that can be implemented quickly
- Provide immediate value/learning
- Minimal risk of breaking existing code

**Phase 2 (Medium Complexity - 2-4 weeks)**:
- Extensions requiring moderate development
- Build on Phase 1 foundations
- Core novel contributions

**Phase 3 (Advanced Features - 4-6 weeks)**:
- Most complex implementations
- Require substantial testing
- Full evolutionary system

### **4. Specific Code Recommendations**

Please provide:

a) **Architecture Suggestions**:
```
proposed_structure/
├── experiments/
│   ├── section1_function_approximation/
│   ├── section2_optimizer_comparison/
│   ├── section3_iron_ore/
│   └── section4_evolutionary_kans/  # NEW
│       ├── ensemble/
│       │   ├── expert_training.py
│       │   ├── stacking.py
│       │   └── clustering.py
│       ├── adaptive/
│       │   ├── importance_metrics.py
│       │   └── densification.py
│       ├── evolution/
│       │   ├── genome.py
│       │   ├── operators.py  # mutation, crossover, selection
│       │   └── evolutionary_search.py
│       └── geophysics/
│           ├── forward_models.py
│           ├── synthetic_data.py
│           └── physics_constraints.py
├── models/
│   ├── kan_base.py  # Existing?
│   ├── adaptive_kan.py  # NEW
│   ├── heterogeneous_basis_kan.py  # NEW
│   └── ensemble_kan.py  # NEW
└── utils/
    ├── training_utils.py
    ├── pruning_utils.py  # Existing?
    ├── evolution_utils.py  # NEW
    └── visualization.py
```

b) **Key Modifications Needed**:
- List existing files that need changes
- What interfaces/APIs need to be added
- Backward compatibility concerns

c) **Testing Strategy**:
- What tests should be added?
- How to validate each extension?
- Benchmark comparisons needed

### **5. Geophysical Application Specifics**

For Section 3 (iron ore), please assess:

a) **Current Implementation**:
- What synthetic data generation exists?
- How is the inverse problem currently formulated?
- What metrics are being used?
- Are there any physics constraints already implemented?

b) **Integration Path**:
- Can evolutionary KANs plug into existing Section 3 experiments?
- What modifications needed for:
  - Forward models
  - Loss functions
  - Evaluation metrics
- How to add physics-informed constraints?

c) **Validation Approach**:
- What ground truth data is available?
- How to test uncertainty quantification?
- Can we create "known solution" synthetic tests?

### **6. Computational Considerations**

Please address:

a) **Parallelization**:
- Current support for multi-GPU or multi-process training?
- Can we train population of KANs in parallel?
- Best approach for parallel evolutionary search?

b) **Memory/Speed**:
- Will evolutionary approach with 20-50 population be feasible?
- Bottlenecks in current implementation?
- Suggestions for optimization?

c) **Experiment Tracking**:
- Current solution for tracking experiments? (WandB, TensorBoard, custom?)
- How to log evolutionary progress (generation-by-generation)?
- Visualization of evolved architectures?

### **7. Missing Dependencies or Blockers**

Identify:
- Libraries that need to be installed
- Features missing from current KAN implementation
- Data availability issues for Section 3
- Any fundamental incompatibilities

### **8. Concrete Next Steps**

Please provide:
- A specific file to start with for Phase 1
- A minimal working example of the simplest extension
- A test case to validate the implementation
- Documentation needs

---

## Additional Context

**Priority**: The most important application is **geophysical inverse problems (Section 3)** - specifically using evolutionary ensembles for uncertainty quantification in iron ore exploration. However, we need the infrastructure from earlier phases to build this.

**Constraints**:
- Must maintain compatibility with existing Sections 1-2
- Should be modular enough to apply to other problems beyond geophysics
- Prefer interpretability and debuggability over maximum performance
- Need good experiment tracking for comparing approaches

**Success Criteria**:
1. Can train ensemble of 10-20 KANs and combine their predictions
2. Can evolve KAN architectures over 20+ generations
3. Can apply to synthetic iron ore data with physics constraints
4. Can quantify prediction uncertainty from ensemble
5. Can analyze which input variables (sensors) are most important

Please be thorough and honest about what's feasible. I'd rather know upfront if something requires major refactoring than discover it mid-implementation.