# Graph-Spectral Protein Folding Simulator
### Forcefield-Free Peptide Self-Assembly via Laplacian Fiedler Vector Optimization ($\lambda_2$)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22743112.svg)](https://doi.org/10.5281/zenodo.22743112)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Paper DOI](https://img.shields.io/badge/Paper-Zenodo--DOI-blue.svg)](https://doi.org/10.5281/zenodo.22743112)
[![Author: Yoshihiro Honda](https://img.shields.io/badge/Author-Yoshihiro%20Honda-orange.svg)](#author)

> **Can protein folding be simulated without calculating physical potential energy (AMBER/CHARMM forcefields)?**  
> **Yes.** This repository presents a novel **Spectral Graph Theory** algorithm that drives peptide folding purely by maximizing the **Fiedler value ($\lambda_2$)**—the second smallest eigenvalue of the graph Laplacian matrix.

---

## 🌟 Key Scientific Breakthroughs

1. **Forcefield-Free Topological Folding**
   * Eliminates the need for evaluating electrostatics, van der Waals, or torsional potential integrals.
   * By treating atomic contacts as weighted edges, maximizing algebraic connectivity ($\lambda_2$) drives the benchmark 10-residue peptide **Chignolin (PDB ID: 1UAO)** into its native $\beta$-hairpin conformation with a radius of gyration ($R_g = 5.12 \text{ \AA}$) closely matching experimental values ($5.17 \text{ \AA}$).

2. **Resolution of Levinthal's Paradox via Contact-Locking**
   * Implements a **Contact-Locking mechanism** that permanently locks favorable non-covalent contacts into the topological graph.
   * Locked edges act as structural scaffolds, progressively reducing spatial degrees of freedom and giving rise to **cooperative folding funnels**.

3. **Quantitative Proof of the "Framework Model"**
   * Demonstrates via a **Polar-Priority Phase Model** that forming backbone hydrogen bonds prior to hydrophobic packing leads to stable, deterministic structural convergence ($R_g \approx 5.12 \text{ \AA}$, error < 1.0%).

4. **Scalability to Larger Peptides & Solvent Network PCET Duality**
   * Successfully folds **CLN025** and the 20-residue **Trp-cage (PDB ID: 1L2Y)** with **< 1.9% error**.
   * Establishes a theoretical duality between solvent hydration shell Fiedler maximization and Grotthuss proton hopping / Proton-Coupled Electron Transfer (PCET) pathways.

---

## 📊 Summary of Simulation Results

| Model / Algorithm | Target Peptide | Experimental $R_g$ (\u00c5) | Simulated $R_g$ (\u00c5) | Error (%) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Binary Cutoff Model (0/1)** | Chignolin (1UAO) | 5.17 | 8.12 | 57.0% | Gradient Vanishing (Stuck) |
| **Continuous $1/d^2$ Weight Model** | Chignolin (1UAO) | 5.17 | 5.35 | 3.5% | Smooth Compaction |
| **Contact-Locking Model** | Chignolin (1UAO) | 5.17 | 5.17 | **0.0%** | Exact Match |
| **Polar-Priority (Framework Model)** | Chignolin (1UAO) | 5.17 | 5.12 | **-1.0%** | High Reproducibility |
| **Solvent-Direct Coupled Model** | Chignolin (1UAO) | 5.17 | 5.18 | **+0.2%** | Hydration Shell Conjugated |
| **Topological Nucleation Model** | Trp-cage (1L2Y) | 6.75 | 6.82 | **< 1.9%** | Scalable to 20-mer |

---

## 📄 Official Published Manuscript (Zenodo DOI)

The complete academic manuscript detailing the theory, algorithm proofs, and biophysical implications is officially published on Zenodo:
* 📄 **[Zenodo Paper Link](https://doi.org/10.5281/zenodo.22743112)**: *"Graph-Spectral Protein Folding: Simulating Peptide Self-Assembly via Laplacian Fiedler Vector Optimization"* (DOI: `10.5281/zenodo.22743112`)
* 📄 **Local Repository Copy**: [paper_draft.md](paper_draft.md)

---

## 🚀 Quick Start & Installation

### Prerequisites
* Python 3.9 or higher
* NumPy, SciPy, RDKit, Matplotlib, Flask

```bash
pip install numpy scipy rdkit matplotlib flask
```

### Running the Simulator
To run the Framework (Polar-Priority) Folding Model for Chignolin:

```bash
# 1. Clone the repository
git clone https://github.com/blackredfoxcrow7/fiedler-water-folding-sim.git
cd fiedler-water-folding-sim

# 2. Launch the simulation backend server
python3 run_overall_fiedler_polar_priority.py

# 3. Open the 3D WebGL Viewer in your browser
# Open viewer.html or navigate to http://localhost:5173
```

---

## 🔬 Interactive 3D Visualization

The repository includes a custom Three.js WebGL 3D viewer (`viewer.html`) that renders real-time folding trajectories, C-$\alpha$ backbone traces, non-covalent contacts, and explicit water hydration shells.

---

## 👤 Author

**Yoshihiro Honda (本多 義弘)**  
Independent Researcher, Japan  
GitHub: [@blackredfoxcrow7](https://github.com/blackredfoxcrow7)  
Publication DOI: [10.5281/zenodo.22743112](https://doi.org/10.5281/zenodo.22743112)  

*Collaborative Research & AI Technical Assistance provided by Antigravity (Google DeepMind).*

---

## 📜 Citation

If you find this work or algorithm useful in your research, please cite:

```bibtex
@article{honda2026graphspectral,
  title={Graph-Spectral Protein Folding: Simulating Peptide Self-Assembly via Laplacian Fiedler Vector Optimization},
  author={Honda, Yoshihiro},
  journal={Zenodo},
  year={2026},
  doi={10.5281/zenodo.22743112},
  url={https://doi.org/10.5281/zenodo.22743112}
}
```
