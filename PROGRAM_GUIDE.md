# 🧬 Program Architecture & Usage Guide (プログラム全構成ガイド)

This document provides a comprehensive overview of the core engines, peptide folding simulation models, structural evaluation tools, and 3D WebGL viewers included in the `fiedler-water-folding-sim` repository.

*Note: A full Japanese translation is provided in the second half of this document. (後半に日本語訳を併記しています。)*

---

## 🇺🇸 English Guide

### 1. Core Engine & Molecular Interpreter

* **Core Engine Class: `PeptideAgent`** ([`peptide_agent.py`](peptide_agent.py))
  * The unified core engine responsible for sequence/SMILES parsing, 3D conformer initialization, topology extraction, and real-time Laplacian Fiedler value calculation.
  * **Supported Input Formats** (Auto-detected):
    1. **1-Letter Amino Acid Sequences**: e.g., `YYDPETGTWY` (Chignolin), `NLYIQWLKDGGPSSGRPPPS` (Trp-cage)
    2. **3-Letter Amino Acid Sequences**: e.g., `Tyr-Tyr-Asp-Pro-Glu-Thr-Gly-Thr-Trp-Tyr`
    3. **SMILES Strings**: e.g., `CC(C)...` or arbitrary chemical SMILES representations.
  * **Core Capabilities**:
    * Molecule graph parsing and 3D initial conformer generation via RDKit ETKDGv3.
    * Automatic identification of rotatable backbone/sidechain dihedral joints ($\phi, \psi$).
    * Computation of graph Laplacian $L = D - A$ and algebraic connectivity ($\lambda_2$).

---

### 2. Chignolin & Peptide Folding Simulation Models

Primary backend and runner scripts corresponding to the models published in our paper (Zenodo DOI: [`10.5281/zenodo.22743112`](https://doi.org/10.5281/zenodo.22743112)).

* **① [Main Paper Model] Framework Model (Polar-Priority Phase Model)**:
  * Backend: [`server_overall_fiedler_polar_priority.py`](server_overall_fiedler_polar_priority.py)
  * Runner: [`run_overall_fiedler_polar_priority.py`](run_overall_fiedler_polar_priority.py)
  * Input: Chignolin sequence (`YYDPETGTWY`) or corresponding SMILES.
  * Mechanism: Phase 1 (Cycles 0–4) locks backbone hydrogen bonds to form $\beta$-sheet frameworks; Phase 2 (Cycles 5–9) unlocks hydrophobic core packing (Trp9, Tyr1, Tyr3).
  * Output: Folded trajectory JSON ([`1uao_folded_directional.json`](1uao_folded_directional.json)) with radius of gyration $R_g = 5.12 \text{ \AA}$ (Experimental: $5.17 \text{ \AA}$).
* **② Solvent-Direct Coupled Model**:
  * Backend: [`server_overall_fiedler_solvent_direct.py`](server_overall_fiedler_solvent_direct.py)
  * Runner: [`run_overall_fiedler_solvent_direct.py`](run_overall_fiedler_solvent_direct.py)
  * Input: Peptide sequence + explicit water molecules ($O_w$).
  * Mechanism: Directly maximizes the integrated Fiedler value $\lambda_2(G_{total})$ of the combined peptide-solvent hydration network.
* **③ Topological Nucleation-Propagation Model**:
  * Backend: [`server_overall_fiedler_nucleation.py`](server_overall_fiedler_nucleation.py)
  * Runner: [`run_overall_fiedler_nucleation.py`](run_overall_fiedler_nucleation.py)
  * Input: Trp-cage (20-mer: `NLYIQWLKDGGPSSGRPPPS`) or Chignolin.
  * Mechanism: Locks early contacts as nucleation seeds, deterministically restricting degrees of freedom to scale to larger peptides (Trp-cage error < 1.9%).
* **④ Optimized Hydrophobic-Priority Model**:
  * Backend: [`server_overall_fiedler_hydro_priority_optimized.py`](server_overall_fiedler_hydro_priority_optimized.py)
  * Runner: [`run_overall_fiedler_hydro_priority_optimized.py`](run_overall_fiedler_hydro_priority_optimized.py)
  * Mechanism: Batch processes eigenvalue evaluations, achieving a ~7.5x acceleration (3 runs in 1 min 52 sec).

---

### 3. Structural Evaluation & 3D WebGL Viewers

* **RMSD Structural Evaluation**: [`calculate_rmsd.py`](calculate_rmsd.py)
  * Computes Root Mean Square Deviation (RMSD) between simulation output JSON coordinates and experimental PDB files ([`1UAO.pdb`](1UAO.pdb), [`1L2Y.pdb`](1L2Y.pdb)) using the Kabsch algorithm.
* **Interactive 3D WebGL Viewer**: [`viewer.html`](viewer.html)
  * Three.js WebGL viewer for real-time trajectory playback, C-$\alpha$ backbone tracing, hydrogen-bond rendering, and water network visualization.

---

## 🇯🇵 日本語ガイド (Japanese Guide)

### 1. コア・解析エンジン

* **核心クラス：`PeptideAgent`** ([`peptide_agent.py`](peptide_agent.py))
  * 入力解釈（SMILES、1文字アミノ酸 `YYDPETGTWY`、3文字アミノ酸 `Tyr-Tyr-...`）、3D構造初期化（RDKit ETKDGv3）、二面角ジョイント（$\phi, \psi$）抽出、および Fiedler値（$\lambda_2$）算出を行う核心クラス。

### 2. チグノリン（Chignolin）および主要フォールディング・プログラム

* **①【主論文モデル】水素結合優先段階モデル (Framework Model)**:
  * スクリプト: [`run_overall_fiedler_polar_priority.py`](run_overall_fiedler_polar_priority.py)
  * 主鎖水素結合（$\beta$シート骨格）を最優先ロック後、疎水コアをパッキング。慣性半径 $R_g = 5.12 \text{ \AA}$（実験値 $5.17 \text{ \AA}$）。
* **② 水分子直接結合モデル (Solvent-Direct Coupled Model)**:
  * スクリプト: [`run_overall_fiedler_solvent_direct.py`](run_overall_fiedler_solvent_direct.py)
  * ペプチド＋水和水素結合ネットワーク全体の統合 Fiedler値を直接最大化。
* **③ トポロジカル核形成伝播モデル (Nucleation-Propagation Model)**:
  * スクリプト: [`run_overall_fiedler_nucleation.py`](run_overall_fiedler_nucleation.py)
  * 長鎖ペプチド Trp-cage (20残基) に対応し、核形成により誤差 < 1.9% を達成。

### 3. 構造検証および 3D 可視化ツール

* **RMSD構造検証**: [`calculate_rmsd.py`](calculate_rmsd.py) (Kabschアルゴリズムによる実測PDBとの誤差計算)
* **3D WebGL可視化**: [`viewer.html`](viewer.html) (Three.js による3D軌跡再生ツール)

---

## 💻 Execution Example (実行手順)

```bash
# Run Chignolin Framework Folding Model
python3 run_overall_fiedler_polar_priority.py

# Calculate RMSD against Experimental PDB (1UAO)
python3 calculate_rmsd.py
```
