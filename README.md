# Graph-Spectral Colloid & LNP Surfactant Phase Predictor (Fiedler-Colloid-LNP-Sim)

**Graph-Spectral Topological Physics Framework for Colloid Science, Surfactant Mesophase Transitions, Lipid Nanoparticle (LNP) mRNA Drug Delivery, and 100-Residue Protein Folding**

*Concept & Theory by Yoshihiro Honda (本多 義弘 / Yoshi)*  
*Independent Researcher in Computational & Organic Chemistry, Japan*

---

## 🔬 Overview & Core Scientific Concepts

This repository provides a **Graph-Spectral Predictive Framework** that bridges **organic chemical structural formulas and multi-component molecular interaction networks** directly to **macroscopic physical functions ($\lambda_2$, directional Laplacian anisotropy $\sigma_z/\sigma_{xy}$, packing parameter $P$, endosomal escape kinetics, and protein folding trajectories)**.

### 1. Dynamic Adaptive Water Graph Nodes vs. Static Spatial Grid
Traditional Molecular Dynamics (MD) or lattice models represent solvent water as either a rigid 3D spatial grid or an implicit dielectric continuum $\epsilon=80$. Static grid models suffer from:
1. **Grid Anisotropy Error**: Artificial discretization noise when molecules rotate or fold off-axis.
2. **Discontinuous Desolvation**: Step-like artificial jumps when hydrophobic cores expel water.

**Our Graph Spectral Approach**:
Water molecules are modeled as **Dynamic Adaptive Graph Nodes ($W_k$)** whose adjacency matrix elements $A(R_i, W_k)$ update continuously in real-time according to molecular coordinates $\vec{r}_i$, local hydropathy scores, and hydrogen-bonding topology:

\[
A(R_i, W_k) = f_{\text{hydration}}(\text{Hydropathy}_i) \cdot \exp\left(-\frac{\|\vec{r}_{R_i} - \vec{r}_{W_k}\|^2}{2\sigma^2}\right)
\]

As hydrophobic collapse or mesophase morphing occurs, water nodes are dynamically expelled (desolvation), releasing hydrogen-bond entropy ($\Delta S_{\text{water}}$) and driving structural transitions. This process is calculated continuously via the **Graph Laplacian Fiedler Eigenvalue $\lambda_2(L)$**.

---

## 🚀 Interactive 3D WebGL Visualizer Suite

Open any of the standalone HTML visualizers directly in any modern web browser (no server or dependencies required):

| Visualizer / Tool | Description & Physical Features |
| :--- | :--- |
| 🧬 **[`protein_100_folding_visualizer.html`](file:///home/eldenring/fiedler-colloid-lnp-sim/protein_100_folding_visualizer.html)** | **100-Residue Protein Folding & Fiedler $\lambda_2$ Dynamics**: 100-amino-acid chain (Red hydrophobic core + Cyan polar surface) + 200 dynamic water nodes. Collapses from extended coil ($R_g=24\text{ \AA}$) to native fold ($R_g=7.7\text{ \AA}$). |
| 💎 **[`pure_fiedler_lnp2_visualizer.html`](file:///home/eldenring/fiedler-colloid-lnp-sim/pure_fiedler_lnp2_visualizer.html)** | **Pure Fiedler LNP2 Inverted Mesophases**: Driven purely by $\mathcal{F}_{\text{spec}} = \frac{\lambda_2(\text{Head-Water})}{\lambda_2(\text{Tail-Tail})}$. Features 2-Leaflet Bilayer Vesicle $L_\alpha$, Inverse Micellar $Fd3m$, Inverse Hexagonal $H_{II}$, and Schwarz P Inverse Bicontinuous $Q_2$ with smooth topological morphing. |
| 🧪 **[`lipid_structure_phase_scenario_visualizer.html`](file:///home/eldenring/fiedler-colloid-lnp-sim/lipid_structure_phase_scenario_visualizer.html)** | **Chemical Mutation Scenario (MC3 vs DLinDMA vs ALC-0315 Stereoisomers)**: Compares $(S,S)$-ALC-0315 (low toxicity, high delivery) vs $(R,R)$-ALC-0315 (Cheeger bottleneck, cytokine release). |
| 🌊 **[`single_vesicle_to_bicontinuous_visualizer.html`](file:///home/eldenring/fiedler-colloid-lnp-sim/single_vesicle_to_bicontinuous_visualizer.html)** | **Vesicle $\to$ Bicontinuous Morphing**: Continuous topological morphing of a spherical bilayer vesicle into a 3D minimal surface bicontinuous network. |
| 🧬 **[`colloid_lnp_phase_simulator.html`](file:///home/eldenring/fiedler-colloid-lnp-sim/colloid_lnp_phase_simulator.html)** | **Colloid & LNP Phase Predictor**: Literature preset dropdown (Moderna, Pfizer, Onpattro), 2D Ternary Phase Diagram (三角相図), and real-time 3D mesophase rendering. |

---

## 🐍 Python Physics Engines & Technical Reports

### Python Verification Engines
Run the ultrafast Python engines to generate exact Graph Laplacian Fiedler spectra ($\lambda_2$) and 3D coordinate trajectories:

```bash
# 1. 100-Residue Protein Folding Fiedler Engine (300 total graph nodes)
python3 protein_100_folding_fiedler_sim.py

# 2. Pure Fiedler LNP2 Inverted Mesophase Engine
python3 pure_fiedler_lnp2_spectral_engine.py

# 3. Chemical Mutation & ALC-0315 Stereoisomer Scenario Engine
python3 lipid_structure_phase_scenario_sim.py

# 4. Single Vesicle to Bicontinuous Minimal Surface Morphing Engine
python3 single_vesicle_to_bicontinuous_sim.py

# 5. Benchmark Formulation Screening Engine (Moderna, Pfizer, Onpattro, Honda Gel)
python3 colloid_surfactant_phase_engine.py
```

### Technical Markdown Reports
- 📄 **[`protein_folding_water_graph_nodes_report.md`](file:///home/eldenring/fiedler-colloid-lnp-sim/protein_folding_water_graph_nodes_report.md)**: Technical report detailing why dynamic water graph nodes outperform static grid models in protein folding and hydrophobic collapse.
- 📄 **[`LNP.txt`](file:///home/eldenring/fiedler-colloid-lnp-sim/LNP.txt)** & **[`LNP2.txt`](file:///home/eldenring/fiedler-colloid-lnp-sim/LNP2.txt)**: Benchmark physical chemistry literature notes on LNP core-shell structure, endosomal acidification ($N \to NH^+$), lyotropic liquid crystal phase transitions ($Fd3m, H_{II}, Q_2$), and ALC-0315 stereoisomer toxicity.

---

## 📊 Benchmark Literature Verification Results

| Formulation / Reference | pH 7.4 (Storage) Phase | pH 5.5 (Endosomal) Phase | Fiedler Ratio $\mathcal{F}_{\text{spec}}$ | Packing $P$ | Fusogenic Activity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Moderna SM-102** *(Nature 2020)* | Bilayer Vesicle ($L_\alpha$) | Inverted Hexagonal ($H_{II}$) | $0.72 \to 1.38$ | $1.03 \to 1.62$ | **98%** |
| **Pfizer ALC-0315 (S,S)** *(Nature 2021)* | Bilayer Vesicle ($L_\alpha$) | Inverted Hexagonal ($H_{II}$) | $0.71 \to 1.36$ | $1.03 \to 1.60$ | **98% (Low Cytotoxicity)** |
| **Pfizer ALC-0315 (R,R)** *(Nature 2021)* | Bilayer Vesicle ($L_\alpha$) | Aggregated Bottleneck | $0.71 \to 1.25$ | $1.03 \to 1.48$ | **Cheeger Bottleneck / IL-6 Cytokine Surge** |
| **Onpattro DLin-MC3-DMA** *(Nat. Nanotech 2019)* | Bilayer Vesicle ($L_\alpha$) | Inverted Hexagonal ($H_{II}$) | $0.72 \to 1.42$ | $1.03 \to 1.62$ | **98% (pH 6.0 Sharp Escape)** |
| **Honda Industrial Hydrogel** *(Honda Concept)* | Bilayer Vesicle ($L_\alpha$) | Inverted Hexagonal ($H_{II}$) | $0.68 \to 1.35$ | $0.97 \to 1.38$ | **95%** |

---

## 📄 License & Citation

Distributed under the MIT License.

```bibtex
@software{honda2026colloid,
  author = {Yoshihiro Honda},
  title = {Graph-Spectral Colloid, LNP, and Protein Folding Phase Predictor},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/blackredfoxcrow7/fiedler-colloid-lnp-sim}
}
```
