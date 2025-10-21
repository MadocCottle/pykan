# Academic Papers and References

**Repository:** PyKAN - Kolmogorov-Arnold Networks Implementation
**Last Updated:** October 2025

This document contains all academic papers, references, and citations explicitly mentioned or used in this repository.

---

## Table of Contents

1. [Core KAN Papers](#core-kan-papers)
2. [Optimization & Training](#optimization--training)
3. [Physics-Informed Neural Networks](#physics-informed-neural-networks)
4. [Neural Architecture Search](#neural-architecture-search)
5. [Neural Representations](#neural-representations)
6. [Ensemble Methods](#ensemble-methods)
7. [Complete Bibliography](#complete-bibliography)

---

## Core KAN Papers

### KAN: Kolmogorov-Arnold Networks
**Primary Paper**

- **Authors:** Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljačić, Thomas Y. Hou, Max Tegmark
- **Year:** 2024
- **Venue:** arXiv preprint
- **arXiv:** [2404.19756](https://arxiv.org/abs/2404.19756)
- **PDF:** https://arxiv.org/pdf/2404.19756.pdf

**Abstract:** Introduces Kolmogorov-Arnold Networks (KAN) as an alternative to Multi-Layer Perceptrons (MLPs). KANs are based on the Kolmogorov-Arnold representation theorem and place learnable activation functions on edges rather than nodes.

**Key Contributions:**
- Novel architecture with learnable univariate functions on edges
- Demonstrates improved accuracy and interpretability vs MLPs
- Applications to scientific computing and symbolic regression

**Usage in Repo:**
- Foundation for all KAN implementations
- Core architecture in `kan/MultKAN.py`, `kan/LBFGSKAN.py`
- Theoretical basis for all extensions

**BibTeX:**
```bibtex
@article{liu2024kan,
  title={KAN: Kolmogorov-Arnold Networks},
  author={Liu, Ziming and Wang, Yixuan and Vaidya, Sachin and Ruehle, Fabian and Halverson, James and Soljačić, Marin and Hou, Thomas Y and Tegmark, Max},
  journal={arXiv preprint arXiv:2404.19756},
  year={2024}
}
```

---

### KAN 2.0: Kolmogorov-Arnold Networks Meet Science

- **Authors:** Ziming Liu et al.
- **Year:** 2024
- **Venue:** arXiv preprint
- **arXiv:** [2408.10205](https://arxiv.org/abs/2408.10205)
- **PDF:** https://arxiv.org/pdf/2408.10205.pdf

**Abstract:** Extended KAN paper demonstrating applications to scientific computing, including PDE solving, physics-informed learning, and multi-scale problems.

**Key Contributions:**
- Physics-informed KAN training
- Applications to PDEs (heat, wave, Burgers, etc.)
- Multi-scale and adaptive grid methods

**Usage in Repo:**
- Guides Section 2 PDE solver implementations
- Motivates physics-informed training modes
- Adaptive grid algorithms in Section 2 New

---

## Optimization & Training

### Adam: A Method for Stochastic Optimization

- **Authors:** Diederik P. Kingma, Jimmy Ba
- **Year:** 2015
- **Venue:** ICLR (International Conference on Learning Representations)
- **arXiv:** [1412.6980](https://arxiv.org/abs/1412.6980)
- **PDF:** https://arxiv.org/pdf/1412.6980.pdf

**Abstract:** Introduces Adam, an adaptive learning rate optimization algorithm combining ideas from AdaGrad and RMSProp.

**Usage in Repo:**
- Primary optimizer in Section 1 training
- Baseline optimizer in Section 2.1 optimizer comparisons
- Default optimizer for most experiments

**BibTeX:**
```bibtex
@inproceedings{kingma2015adam,
  title={Adam: A method for stochastic optimization},
  author={Kingma, Diederik P and Ba, Jimmy},
  booktitle={ICLR},
  year={2015}
}
```

---

### Decoupled Weight Decay Regularization

- **Authors:** Ilya Loshchilov, Frank Hutter
- **Year:** 2019
- **Venue:** ICLR (International Conference on Learning Representations)
- **arXiv:** [1711.05101](https://arxiv.org/abs/1711.05101)
- **PDF:** https://arxiv.org/pdf/1711.05101.pdf

**Abstract:** Introduces AdamW, which fixes weight decay implementation in Adam by decoupling it from gradient updates.

**Usage in Repo:**
- Optimizer option in Section 2.1
- Compared against Adam and L-BFGS for PDE solving

**BibTeX:**
```bibtex
@inproceedings{loshchilov2019decoupled,
  title={Decoupled weight decay regularization},
  author={Loshchilov, Ilya and Hutter, Frank},
  booktitle={ICLR},
  year={2019}
}
```

---

### On the Limited Memory BFGS Method for Large Scale Optimization

- **Authors:** Dong C. Liu, Jorge Nocedal
- **Year:** 1989
- **Venue:** Mathematical Programming, Volume 45, Issue 1-3, pp. 503-528
- **DOI:** [10.1007/BF01589116](https://doi.org/10.1007/BF01589116)

**Abstract:** Introduces L-BFGS, a memory-efficient quasi-Newton optimization method suitable for large-scale problems.

**Usage in Repo:**
- Implemented in `kan/LBFGSKAN.py`
- Primary optimizer for PDE solving in Section 2
- Compared in Section 2.1 optimizer benchmarks

**BibTeX:**
```bibtex
@article{liu1989limited,
  title={On the limited memory BFGS method for large scale optimization},
  author={Liu, Dong C and Nocedal, Jorge},
  journal={Mathematical Programming},
  volume={45},
  number={1-3},
  pages={503--528},
  year={1989},
  publisher={Springer}
}
```

---

### A Method for the Solution of Certain Non-Linear Problems in Least Squares

- **Authors:** Kenneth Levenberg
- **Year:** 1944
- **Venue:** Quarterly of Applied Mathematics, 2(2), pp. 164-168
- **DOI:** [10.1090/qam/10666](https://doi.org/10.1090/qam/10666)

**Abstract:** Introduces the damping parameter strategy for Newton's method, forming the basis of Levenberg-Marquardt.

**Usage in Repo:**
- Referenced in optimizer comparison analysis
- Motivates future Levenberg-Marquardt implementation

---

### An Algorithm for Least-Squares Estimation of Nonlinear Parameters

- **Authors:** Donald W. Marquardt
- **Year:** 1963
- **Venue:** Journal of the Society for Industrial and Applied Mathematics, 11(2), pp. 431-441
- **DOI:** [10.1137/0111030](https://doi.org/10.1137/0111030)

**Abstract:** Completes the Levenberg-Marquardt algorithm by adding adaptive damping parameter selection.

**Usage in Repo:**
- Referenced in Section 2.1 optimizer analysis
- Cited as optimal optimizer for small PDE problems per Krishnapriyan et al.

---

### Characterizing Possible Failure Modes in Physics-Informed Neural Networks

- **Authors:** Aditi Krishnapriyan, Amir Gholami, Shandian Zhe, Robert Kirby, Michael W. Mahoney
- **Year:** 2021
- **Venue:** NeurIPS (Neural Information Processing Systems)
- **arXiv:** [2205.07430](https://arxiv.org/abs/2205.07430)
- **PDF:** https://arxiv.org/pdf/2205.07430.pdf

**Abstract:** Analyzes why PINNs fail on certain problems, showing Adam/BFGS struggle with low-amplitude components while Levenberg-Marquardt excels.

**Key Findings:**
- Adam requires 26× more parameters than LM to match accuracy
- Second-order methods (L-BFGS, LM) preferred for smooth PDE loss landscapes
- First-order methods struggle with spectral bias

**Usage in Repo:**
- **Critical reference** for Section 2.1 optimizer selection
- Guides implementation of L-BFGS for PDE solving
- Justifies hybrid training strategies

**Location in Repo:**
- `/Users/main/Desktop/my_pykan/pykan/madoc/IMPLEMENTATION_SPEC_2_1.md`
- Section 2 optimizer implementations

**BibTeX:**
```bibtex
@inproceedings{krishnapriyan2021characterizing,
  title={Characterizing possible failure modes in physics-informed neural networks},
  author={Krishnapriyan, Aditi and Gholami, Amir and Zhe, Shandian and Kirby, Robert and Mahoney, Michael W},
  booktitle={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

---

## Physics-Informed Neural Networks

### Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems

- **Authors:** Maziar Raissi, Paris Perdikaris, George Em Karniadakis
- **Year:** 2019
- **Venue:** Journal of Computational Physics, Volume 378, pp. 686-707
- **DOI:** [10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)
- **arXiv:** [1711.10561](https://arxiv.org/abs/1711.10561)

**Abstract:** Foundational PINN paper introducing physics-informed loss functions combining data and PDE residuals for solving forward and inverse problems.

**Key Contributions:**
- Physics-informed loss: L = L_data + λ L_PDE
- Automatic differentiation for computing PDE residuals
- Applications to Navier-Stokes, Schrödinger, etc.

**Usage in Repo:**
- Basis for physics-informed training mode in Section 2
- `PINNTrainer` class in Section 2 implementations
- PDE residual calculations via autograd
- Applications: Poisson, Heat, Burgers, Helmholtz equations

**Location in Repo:**
- `/Users/main/Desktop/my_pykan/pykan/madoc/section2/README.md`
- Section 2 training modules

**BibTeX:**
```bibtex
@article{raissi2019physics,
  title={Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations},
  author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George Em},
  journal={Journal of Computational Physics},
  volume={378},
  pages={686--707},
  year={2019},
  publisher={Elsevier}
}
```

---

## Neural Architecture Search

### Neural Architecture Search with Reinforcement Learning

- **Authors:** Barret Zoph, Quoc V. Le
- **Year:** 2017
- **Venue:** ICLR (International Conference on Learning Representations)
- **arXiv:** [1611.01578](https://arxiv.org/abs/1611.01578)
- **PDF:** https://arxiv.org/pdf/1611.01578.pdf

**Abstract:** Introduces using reinforcement learning to automatically design neural network architectures.

**Usage in Repo:**
- Motivates evolutionary architecture search in Section 2 New (Extension 5)
- Inspires genome representation for KAN architectures

**BibTeX:**
```bibtex
@inproceedings{zoph2017neural,
  title={Neural architecture search with reinforcement learning},
  author={Zoph, Barret and Le, Quoc V},
  booktitle={ICLR},
  year={2017}
}
```

---

### DARTS: Differentiable Architecture Search

- **Authors:** Hanxiao Liu, Karen Simonyan, Yiming Yang
- **Year:** 2019
- **Venue:** ICLR (International Conference on Learning Representations)
- **arXiv:** [1806.09055](https://arxiv.org/abs/1806.09055)
- **PDF:** https://arxiv.org/pdf/1806.09055.pdf

**Abstract:** Introduces differentiable architecture search using continuous relaxation of discrete architecture choices.

**Usage in Repo:**
- Referenced in Section 2 New architecture search design
- Alternative to evolutionary methods

**BibTeX:**
```bibtex
@inproceedings{liu2019darts,
  title={DARTS: Differentiable architecture search},
  author={Liu, Hanxiao and Simonyan, Karen and Yang, Yiming},
  booktitle={ICLR},
  year={2019}
}
```

---

## Neural Representations

### Implicit Neural Representations with Periodic Activation Functions

**Also known as SIREN (Sinusoidal Representation Networks)**

- **Authors:** Vincent Sitzmann, Julien N. P. Martel, Alexander W. Bergman, David B. Lindell, Gordon Wetzstein
- **Year:** 2020
- **Venue:** NeurIPS (Advances in Neural Information Processing Systems)
- **arXiv:** [2006.09661](https://arxiv.org/abs/2006.09661)
- **PDF:** https://arxiv.org/pdf/2006.09661.pdf
- **Project Page:** https://vsitzmann.github.io/siren/

**Abstract:** Introduces SIREN networks using sinusoidal activation functions for implicit neural representations of images, audio, video, and 3D shapes.

**Key Contributions:**
- sin(ω₀·x) activation functions
- Suitable for fitting signals and their derivatives
- Applications to solving PDEs and boundary value problems

**Usage in Repo:**
- **Baseline comparison** for KAN in PDE solving (Section 2)
- Model implementation in `section2/models.py`
- Benchmark for function approximation tasks
- Compared in Poisson, Heat, and Burgers equation experiments

**Location in Repo:**
- `/Users/main/Desktop/my_pykan/pykan/madoc/section2/README.md`
- Multiple comparison experiments

**BibTeX:**
```bibtex
@inproceedings{sitzmann2020implicit,
  title={Implicit neural representations with periodic activation functions},
  author={Sitzmann, Vincent and Martel, Julien NP and Bergman, Alexander W and Lindell, David B and Wetzstein, Gordon},
  booktitle={Advances in Neural Information Processing Systems},
  volume={33},
  pages={7462--7473},
  year={2020}
}
```

---

## Ensemble Methods

### Population-Based Training of Neural Networks

- **Authors:** Max Jaderberg, Valentin Dalibard, Simon Osindero, Wojciech M. Czarnecki, Jeff Donahue, Ali Razavi, Oriol Vinyals, Tim Green, Iain Dunning, Karen Simonyan, Chrisantha Fernando, Koray Kavukcuoglu
- **Year:** 2017
- **Venue:** arXiv preprint
- **arXiv:** [1711.09846](https://arxiv.org/abs/1711.09846)
- **PDF:** https://arxiv.org/pdf/1711.09846.pdf

**Abstract:** Introduces Population-Based Training (PBT), which trains multiple models in parallel with periodic parameter and hyperparameter sharing.

**Key Contributions:**
- Parallel population training with synchronization
- Automatic hyperparameter optimization during training
- Maintains diversity while sharing knowledge

**Usage in Repo:**
- Foundation for **Extension 4: Population-Based Training** in Section 2 New
- Implemented in `section2_new/population/population_trainer.py`
- Synchronization methods: averaging, best-model copying, tournament

**BibTeX:**
```bibtex
@article{jaderberg2017population,
  title={Population based training of neural networks},
  author={Jaderberg, Max and Dalibard, Valentin and Osindero, Simon and Czarnecki, Wojciech M and Donahue, Jeff and Razavi, Ali and Vinyals, Oriol and Green, Tim and Dunning, Iain and Simonyan, Karen and others},
  journal={arXiv preprint arXiv:1711.09846},
  year={2017}
}
```

---

### Ensemble Methods in Machine Learning

- **Authors:** Thomas G. Dietterich
- **Year:** 2000
- **Venue:** Multiple Classifier Systems, Lecture Notes in Computer Science, Volume 1857
- **DOI:** [10.1007/3-540-45014-9_1](https://doi.org/10.1007/3-540-45014-9_1)

**Abstract:** Surveys ensemble methods including bagging, boosting, and error-correcting output codes.

**Usage in Repo:**
- Theoretical foundation for **Extension 1: Hierarchical Ensemble** in Section 2 New
- Guides multi-seed expert training and stacking

**BibTeX:**
```bibtex
@inproceedings{dietterich2000ensemble,
  title={Ensemble methods in machine learning},
  author={Dietterich, Thomas G},
  booktitle={International workshop on multiple classifier systems},
  pages={1--15},
  year={2000},
  organization={Springer}
}
```

---

## Adaptive Grid Methods

### Adaptive Grid Methods for PDEs

- **Authors:** Weizhang Huang, Robert D. Russell
- **Year:** 2011
- **Venue:** Springer Series in Applied Mathematical Sciences, Volume 174
- **ISBN:** 978-1-4419-7915-5
- **DOI:** [10.1007/978-1-4419-7916-2](https://doi.org/10.1007/978-1-4419-7916-2)

**Abstract:** Comprehensive treatment of adaptive grid methods for solving partial differential equations, including moving mesh methods and r-refinement.

**Usage in Repo:**
- Inspires **Extension 2: Adaptive Densification** in Section 2 New
- Motivates importance-based selective grid refinement
- Implemented in `section2_new/models/adaptive_selective_kan.py`

**BibTeX:**
```bibtex
@book{huang2011adaptive,
  title={Adaptive moving mesh methods},
  author={Huang, Weizhang and Russell, Robert D},
  volume={174},
  year={2011},
  publisher={Springer Science \& Business Media}
}
```

---

## Complete Bibliography

### Alphabetical by First Author

1. **Dietterich (2000)** - Ensemble methods foundation
2. **Huang & Russell (2011)** - Adaptive grid methods for PDEs
3. **Jaderberg et al. (2017)** - Population-based training
4. **Kingma & Ba (2015)** - Adam optimizer
5. **Krishnapriyan et al. (2021)** - PINN failure modes and optimizer analysis
6. **Levenberg (1944)** - Levenberg method for nonlinear least squares
7. **Liu & Nocedal (1989)** - L-BFGS optimization
8. **Liu et al. (2019)** - DARTS architecture search
9. **Liu et al. (2024a)** - **KAN: Kolmogorov-Arnold Networks** (primary)
10. **Liu et al. (2024b)** - KAN 2.0: Scientific applications
11. **Loshchilov & Hutter (2019)** - AdamW optimizer
12. **Marquardt (1963)** - Marquardt algorithm for least squares
13. **Raissi et al. (2019)** - Physics-informed neural networks
14. **Sitzmann et al. (2020)** - SIREN implicit representations
15. **Zoph & Le (2017)** - Neural architecture search with RL

---

## Papers by Repository Section

### Section 1 (Core KAN Implementation)
- Liu et al. (2024a) - KAN: Kolmogorov-Arnold Networks
- Kingma & Ba (2015) - Adam optimizer

### Section 2 (PDE Solving & Physics-Informed)
- Liu et al. (2024b) - KAN 2.0
- Raissi et al. (2019) - Physics-informed neural networks
- Sitzmann et al. (2020) - SIREN (baseline comparison)
- Krishnapriyan et al. (2021) - PINN optimizer analysis

### Section 2.1 (Optimizer Comparison)
- Krishnapriyan et al. (2021) - Optimizer failure modes
- Kingma & Ba (2015) - Adam
- Loshchilov & Hutter (2019) - AdamW
- Liu & Nocedal (1989) - L-BFGS
- Levenberg (1944) & Marquardt (1963) - Levenberg-Marquardt

### Section 2 New (Evolutionary & Ensemble KAN)

**Extension 1 (Ensemble):**
- Dietterich (2000) - Ensemble methods
- Jaderberg et al. (2017) - Population-based training

**Extension 2 (Adaptive Densification):**
- Huang & Russell (2011) - Adaptive grid methods

**Extension 3 (Heterogeneous Basis):**
- Liu et al. (2024a) - KAN basis functions

**Extension 4 (Population Training):**
- Jaderberg et al. (2017) - Population-based training

**Extension 5 (Evolutionary Search):**
- Zoph & Le (2017) - Neural architecture search
- Liu et al. (2019) - DARTS

---

## Quick Reference by Topic

### Optimization Theory
- **First-order:** Adam (Kingma & Ba 2015), AdamW (Loshchilov & Hutter 2019)
- **Second-order:** L-BFGS (Liu & Nocedal 1989), LM (Levenberg 1944, Marquardt 1963)
- **Analysis:** Krishnapriyan et al. (2021)

### Physics-Informed Learning
- **Theory:** Raissi et al. (2019)
- **Applications:** Liu et al. (2024b)
- **Failure Analysis:** Krishnapriyan et al. (2021)

### Neural Representations
- **KAN:** Liu et al. (2024a, 2024b)
- **SIREN:** Sitzmann et al. (2020)

### Architecture Search & Ensemble
- **NAS:** Zoph & Le (2017), Liu et al. (2019)
- **Ensemble:** Dietterich (2000)
- **Population:** Jaderberg et al. (2017)

### Adaptive Methods
- **Grid Adaptation:** Huang & Russell (2011)

---

## External Resources

### Official KAN Resources
- **KAN GitHub:** https://github.com/KindXiaoming/pykan
- **KAN Documentation:** https://kindxiaoming.github.io/pykan/
- **KAN Paper (arXiv):** https://arxiv.org/abs/2404.19756

### Related Projects
- **SIREN Project Page:** https://vsitzmann.github.io/siren/
- **PINNs GitHub:** https://github.com/maziarraissi/PINNs

### Datasets & Benchmarks
- Referenced implicitly through PDE test problems (Poisson, Heat, Burgers, Helmholtz)

---

## Citation for This Repository

If you use this repository in your research, please cite:

```bibtex
@software{pykan_implementation,
  title={PyKAN: Kolmogorov-Arnold Networks Implementation},
  author={[Repository Authors]},
  year={2024-2025},
  url={[Repository URL]}
}
```

And the primary KAN paper:

```bibtex
@article{liu2024kan,
  title={KAN: Kolmogorov-Arnold Networks},
  author={Liu, Ziming and Wang, Yixuan and Vaidya, Sachin and Ruehle, Fabian and Halverson, James and Soljačić, Marin and Hou, Thomas Y and Tegmark, Max},
  journal={arXiv preprint arXiv:2404.19756},
  year={2024}
}
```

---

**Last Updated:** October 2025
**Maintained by:** Repository maintainers
**Total Papers:** 15 primary references
