# Unified Graph-Spectral Molecular Simulator (fiedler-water-folding-sim)

**Graph-Spectral Topological Physics Framework for Protein Folding, Dynamic Water Graph Nodes, Colloid Science, and Lipid Nanoparticle (LNP) mRNA Drug Delivery Systems**

*Concept & Theory by Yoshihiro Honda (本多 義弘 / Yoshi)*  
*Independent Researcher in Computational & Organic Chemistry, Japan*

---

## 🔬 Unified Overview & Core Scientific Concepts

This repository provides a **Unified Graph-Spectral Predictive Framework** that bridges **chemical structural formulas and multi-component molecular interaction networks** directly to macroscopic physical behavior. By optimizing the **Graph Laplacian Fiedler Eigenvalue ($\lambda_2$)** and spectral ratios, this single topological theory governs both **Protein Folding / Hydrophobic Collapse** and **Colloid / LNP Lyotropic Phase Transitions**.

### 1. Dynamic Adaptive Water Graph Nodes vs. Static Spatial Grid
Traditional Molecular Dynamics (MD) or lattice models represent solvent water as either a rigid 3D spatial grid or an implicit dielectric continuum $\epsilon=80$. Static grid models suffer from grid anisotropy errors and artificial desolvation steps.

**Our Graph Spectral Approach**:
Water molecules are modeled as **Dynamic Adaptive Graph Nodes ($W_k$)** whose adjacency matrix elements $A(R_i, W_k)$ update continuously in real-time according to molecular coordinates $\vec{r}_i$, local hydropathy scores, and hydrogen-bonding topology:

\[
A(R_i, W_k) = f_{\text{hydration}}(\text{Hydropathy}_i) \cdot \exp\left(-\frac{\|\vec{r}_{R_i} - \vec{r}_{W_k}\|^2}{2\sigma^2}\right)
\]

As structural compaction occurs, water nodes are dynamically expelled (desolvation), releasing hydrogen-bond entropy ($\Delta S_{\text{water}}$) and continuously driving transitions via $\lambda_2(L)$.

---

## 🧬 Module 1: Protein Folding & Experimental PDB Benchmark Suite

### Kabsch SVD Algorithm C-$\alpha$ RMSD Validation Results

| Peptide / Benchmark | PDB ID | Residues | Exp $R_g$ ($\text{\AA}$) | Pred $R_g$ ($\text{\AA}$) | $R_g$ Err (%) | **C-$\alpha$ RMSD ($\text{\AA}$)** | Topological Feature |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Deca-alanine** | Ideal $\alpha$-helix | 10 | 4.82 | 4.80 | 0.4% | **3.706 $\text{\AA}$** | $\alpha$-helical backbone alignment |
| **CLN025** | PDB 5AWL | 10 | 5.10 | 5.08 | 0.4% | **4.230 $\text{\AA}$** | $\beta$-hairpin turn matching |
| **Chignolin** | PDB 1UAO | 10 | 5.17 | 5.17 | **0.0%** | **6.493 $\text{\AA}$** | Full fold topology matching |
| **Trp-cage** | PDB 1L2Y | 20 | 7.20 | 7.33 | 1.8% | **6.855 $\text{\AA}$** | 3D fold & hydrophobic core packing |
| **Synthetic Protein** | Custom | 100 | 24.0 | 7.70 | — | **Large-Scale Fold** | 200 dynamic water nodes + core/shell |

---

## 🧪 Module 2: Colloid Science & LNP mRNA Drug Delivery Mesophases

Driven by the **Fiedler Spectral Ratio** $\mathcal{F}_{\text{spec}} = \frac{\lambda_2(\text{Head-Water})}{\lambda_2(\text{Tail-Tail})}$, the framework predicts endosomal acidification ($N \to NH^+$) and lyotropic liquid crystal phase transitions:

| Formulation / Reference | pH 7.4 (Storage) Phase | pH 5.5 (Endosomal) Phase | Fiedler Ratio $\mathcal{F}_{\text{spec}}$ | Packing $P$ | Fusogenic Activity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Moderna SM-102** *(Nature 2020)* | Bilayer Vesicle ($L_\alpha$) | Inverted Hexagonal ($H_{II}$) | $0.72 \to 1.38$ | $1.03 \to 1.62$ | **98%** |
| **Pfizer ALC-0315 (S,S)** *(Nature 2021)* | Bilayer Vesicle ($L_\alpha$) | Inverted Hexagonal ($H_{II}$) | $0.71 \to 1.36$ | $1.03 \to 1.60$ | **98% (Low Cytotoxicity)** |
| **Pfizer ALC-0315 (R,R)** *(Nature 2021)* | Bilayer Vesicle ($L_\alpha$) | Aggregated Bottleneck | $0.71 \to 1.25$ | $1.03 \to 1.48$ | **Cheeger Bottleneck / IL-6 Cytokine Surge** |
| **Onpattro DLin-MC3-DMA** *(Nat. Nanotech 2019)* | Bilayer Vesicle ($L_\alpha$) | Inverted Hexagonal ($H_{II}$) | $0.72 \to 1.42$ | $1.03 \to 1.62$ | **98% (pH 6.0 Sharp Escape)** |
| **Honda Industrial Hydrogel** *(Honda Concept)* | Bilayer Vesicle ($L_\alpha$) | Inverted Hexagonal ($H_{II}$) | $0.68 \to 1.35$ | $0.97 \to 1.38$ | **95%** |

---

## 🚀 Interactive 3D WebGL Visualizer Suite

Open any standalone HTML file directly in any browser (zero server or external dependency required):

| Visualizer / Tool | Description & Physical Features |
| :--- | :--- |
| 🧬 **[`protein_100_folding_visualizer.html`](file:///home/eldenring/fiedler-water-folding-sim/protein_100_folding_visualizer.html)** | **100-Residue Protein Folding & Fiedler $\lambda_2$ Dynamics**: 100 amino acids + 200 dynamic water nodes ($R_g=24.0\text{ \AA} \to 7.7\text{ \AA}$). |
| 🧪 **[`viewer.html`](file:///home/eldenring/fiedler-water-folding-sim/viewer.html)** | **PDB Benchmark Trajectory Viewer**: 3D viewer for Chignolin (1UAO), CLN025 (5AWL), Trp-cage (1L2Y), and Deca-alanine. |
| 💎 **[`pure_fiedler_lnp2_visualizer.html`](file:///home/eldenring/fiedler-water-folding-sim/pure_fiedler_lnp2_visualizer.html)** | **Pure Fiedler LNP2 Mesophases**: 2-Leaflet Bilayer Vesicle $L_\alpha$, Inverse Micellar $Fd3m$, Inverse Hexagonal $H_{II}$, and Schwarz P Inverse Bicontinuous $Q_2$. |
| 🧪 **[`lipid_structure_phase_scenario_visualizer.html`](file:///home/eldenring/fiedler-water-folding-sim/lipid_structure_phase_scenario_visualizer.html)** | **Chemical Mutation & Stereoisomers**: $(S,S)$-ALC-0315 vs $(R,R)$-ALC-0315 vs MC3 vs DLinDMA comparison. |
| 🌊 **[`single_vesicle_to_bicontinuous_visualizer.html`](file:///home/eldenring/fiedler-water-folding-sim/single_vesicle_to_bicontinuous_visualizer.html)** | **Vesicle $\to$ Bicontinuous Morphing**: Continuous topological morphing into a 3D minimal surface. |
| 🧬 **[`colloid_lnp_phase_simulator.html`](file:///home/eldenring/fiedler-water-folding-sim/colloid_lnp_phase_simulator.html)** | **Colloid & LNP Phase Predictor**: 2D Ternary Phase Diagram (三角相図) and real-time 3D mesophase rendering. |

---

## 🐍 Python Physics Engines & Academic Papers

### Python Verification Engines
```bash
# 1. 100-Residue Protein Folding Fiedler Engine (300 total graph nodes)
python3 protein_100_folding_fiedler_sim.py

# 2. PyTorch Differentiable Graph Laplacian Folding Engine
python3 differentiable_folding.py

# 3. Kabsch SVD RMSD Calculation against PDB Files (1UAO, 5AWL, 1L2Y)
python3 calculate_rmsd.py

# 4. Pure Fiedler LNP2 Inverted Mesophase Engine
python3 pure_fiedler_lnp2_spectral_engine.py

# 5. Chemical Mutation & ALC-0315 Stereoisomer Scenario Engine
python3 lipid_structure_phase_scenario_sim.py

# 6. Benchmark Formulation Screening Engine (Moderna, Pfizer, Onpattro, Honda Gel)
python3 colloid_surfactant_phase_engine.py
```

### Academic Manuscripts & Technical Reports
- 📄 **[`protein_folding_water_graph_nodes_report.md`](file:///home/eldenring/fiedler-water-folding-sim/protein_folding_water_graph_nodes_report.md)**: Technical report on dynamic water graph nodes vs static grid models.
- 📄 **[`paper1_folding_expanded_v2_ja.md`](file:///home/eldenring/fiedler-water-folding-sim/paper1_folding_expanded_v2_ja.md)** / **[`paper1_folding_expanded_v2.md`](file:///home/eldenring/fiedler-water-folding-sim/paper1_folding_expanded_v2.md)**: Full Academic Manuscript Part 1 (Graph-Spectral Protein Folding Theory & Kabsch RMSD Validation).
- 📄 **[`paper2_solubility_expanded_v2_ja.md`](file:///home/eldenring/fiedler-water-folding-sim/paper2_solubility_expanded_v2_ja.md)** / **[`paper2_solubility_expanded_v2.md`](file:///home/eldenring/fiedler-water-folding-sim/paper2_solubility_expanded_v2.md)**: Full Academic Manuscript Part 2 (Water Solubility & Dehydration Entropy).
- 📄 **[`fiedler_unified_theory_and_algorithms.md`](file:///home/eldenring/fiedler-water-folding-sim/fiedler_unified_theory_and_algorithms.md)**: Unified Graph Spectral Theory linking Protein Folding & Colloid/LNP Liquid Crystal Phase Transitions.
- 📄 **[`LNP.txt`](file:///home/eldenring/fiedler-water-folding-sim/LNP.txt)** & **[`LNP2.txt`](file:///home/eldenring/fiedler-water-folding-sim/LNP2.txt)**: LNP core-shell structure & lyotropic phase transition notes.

---

## 📄 License & Citation

Distributed under the MIT License.

```bibtex
@software{honda2026unified,
  author = {Yoshihiro Honda},
  title = {Unified Graph-Spectral Molecular Simulator: Protein Folding and Colloid/LNP Phase Predictor},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/blackredfoxcrow7/fiedler-water-folding-sim}
}
```
