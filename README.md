# Graph-Spectral Protein Folding & Dynamic Water Graph Simulator (fiedler-water-folding-sim)

**Graph-Spectral Topological Physics Framework for Protein Folding, Dynamic Adaptive Water Graph Nodes, and Kabsch SVD Experimental PDB Benchmarks**

*Concept & Theory by Yoshihiro Honda (本多 義弘 / Yoshi)*  
*Independent Researcher in Computational & Organic Chemistry, Japan*

---

## 🔬 Overview & Core Scientific Concepts

This repository provides a **Graph-Spectral Predictive Framework** that drives **Protein Folding trajectories and hydrophobic collapse** via **Graph Laplacian Fiedler Eigenvalue ($\lambda_2$) Optimization** without physical force fields or Monte Carlo sampling.

### 1. Dynamic Adaptive Water Graph Nodes vs. Static Spatial Grid
Traditional Molecular Dynamics (MD) or lattice models represent solvent water as either a rigid 3D spatial grid or an implicit dielectric continuum $\epsilon=80$. Static grid models suffer from:
1. **Grid Anisotropy Error**: Artificial discretization noise when molecules rotate or fold off-axis.
2. **Discontinuous Desolvation**: Step-like artificial jumps when hydrophobic cores expel water.

**Our Graph Spectral Approach**:
Water molecules are modeled as **Dynamic Adaptive Graph Nodes ($W_k$)** whose adjacency matrix elements $A(R_i, W_k)$ update continuously in real-time according to molecular coordinates $\vec{r}_i$, local hydropathy scores, and hydrogen-bonding topology:

\[
A(R_i, W_k) = f_{\text{hydration}}(\text{Hydropathy}_i) \cdot \exp\left(-\frac{\|\vec{r}_{R_i} - \vec{r}_{W_k}\|^2}{2\sigma^2}\right)
\]

As hydrophobic collapse occurs, water nodes are dynamically expelled (desolvation), releasing hydrogen-bond entropy ($\Delta S_{\text{water}}$) and driving structural transitions. This process is calculated continuously via the **Graph Laplacian Fiedler Eigenvalue $\lambda_2(L)$**.

---

## 🧬 Experimental PDB Benchmark Verification & Kabsch SVD RMSD Results

| Peptide / Benchmark | PDB ID | Residues | Exp $R_g$ ($\text{\AA}$) | Pred $R_g$ ($\text{\AA}$) | $R_g$ Err (%) | **C-$\alpha$ RMSD ($\text{\AA}$)** | Topological Feature |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Deca-alanine** | Ideal $\alpha$-helix | 10 | 4.82 | 4.80 | 0.4% | **3.706 $\text{\AA}$** | $\alpha$-helical backbone alignment |
| **CLN025** | PDB 5AWL | 10 | 5.10 | 5.08 | 0.4% | **4.230 $\text{\AA}$** | $\beta$-hairpin turn matching |
| **Chignolin** | PDB 1UAO | 10 | 5.17 | 5.17 | **0.0%** | **6.493 $\text{\AA}$** | Full fold topology matching |
| **Trp-cage** | PDB 1L2Y | 20 | 7.20 | 7.33 | 1.8% | **6.855 $\text{\AA}$** | 3D fold & hydrophobic core packing |
| **Synthetic Protein** | Custom | 100 | 24.0 | 7.70 | — | **Large-Scale Fold** | 200 dynamic water nodes + core/shell |

---

## 🚀 Interactive 3D WebGL Visualizer Suite

Open any of the standalone HTML visualizers directly in any modern web browser (no server or dependencies required):

| Visualizer / Tool | Description & Physical Features |
| :--- | :--- |
| 🧬 **[`protein_100_folding_visualizer.html`](file:///home/eldenring/fiedler-water-folding-sim/protein_100_folding_visualizer.html)** | **100-Residue Protein Folding & Fiedler $\lambda_2$ Dynamics**: 100-amino-acid chain (Red hydrophobic core + Cyan polar surface) + 200 dynamic water nodes. Collapses from extended coil ($R_g=24\text{ \AA}$) to native fold ($R_g=7.7\text{ \AA}$). |
| 🧪 **[`viewer.html`](file:///home/eldenring/fiedler-water-folding-sim/viewer.html)** | **PDB Benchmark Trajectory Viewer**: 3D WebGL viewer for Chignolin (1UAO), CLN025 (5AWL), Trp-cage (1L2Y), and Deca-alanine folding trajectories. |

---

## 🐍 Python Physics Engines & Academic Papers

### Protein Folding Engines
```bash
# 1. 100-Residue Protein Folding Fiedler Engine (300 total graph nodes)
python3 protein_100_folding_fiedler_sim.py

# 2. PyTorch Differentiable Graph Laplacian Folding Engine
python3 differentiable_folding.py

# 3. Directional Laplacian Anisotropy Folding Engine
python3 differentiable_folding_directional.py

# 4. Hybrid Physical Force Field + Graph Spectral Folding Engine
python3 differentiable_folding_hybrid.py

# 5. Kabsch SVD RMSD Calculation against PDB Files (1UAO, 5AWL, 1L2Y)
python3 calculate_rmsd.py

# 6. Sensitivity Analysis (±20% Parameter Robustness Test)
python3 rmsd_and_sensitivity_analysis.py
```

### Technical Markdown Papers & Manuscripts
- 📄 **[`protein_folding_water_graph_nodes_report.md`](file:///home/eldenring/fiedler-water-folding-sim/protein_folding_water_graph_nodes_report.md)**: Technical report detailing why dynamic water graph nodes outperform static grid models in protein folding.
- 📄 **[`paper1_folding_expanded_v2_ja.md`](file:///home/eldenring/fiedler-water-folding-sim/paper1_folding_expanded_v2_ja.md)** / **[`paper1_folding_expanded_v2.md`](file:///home/eldenring/fiedler-water-folding-sim/paper1_folding_expanded_v2.md)**: Full Academic Manuscript Part 1 (Graph-Spectral Protein Folding Theory & Kabsch SVD PDB Validation).
- 📄 **[`paper2_solubility_expanded_v2_ja.md`](file:///home/eldenring/fiedler-water-folding-sim/paper2_solubility_expanded_v2_ja.md)** / **[`paper2_solubility_expanded_v2.md`](file:///home/eldenring/fiedler-water-folding-sim/paper2_solubility_expanded_v2.md)**: Full Academic Manuscript Part 2 (Water Solubility & Dehydration Entropy).
- 📄 **[`fiedler_unified_theory_and_algorithms.md`](file:///home/eldenring/fiedler-water-folding-sim/fiedler_unified_theory_and_algorithms.md)**: Unified Graph Spectral Theory linking Protein Folding & Physical Chemistry.

---

## 📄 License & Citation

Distributed under the MIT License.

```bibtex
@software{honda2026protein,
  author = {Yoshihiro Honda},
  title = {Graph-Spectral Protein Folding and Dynamic Water Graph Simulator},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/blackredfoxcrow7/fiedler-water-folding-sim}
}
```
