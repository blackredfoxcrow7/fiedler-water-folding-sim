# 🧬 プログラム全構成・使用方法解説ガイド (Program Guide & Architecture)

本ドキュメントは、本リポジトリ（`fiedler-water-folding-sim`）に含まれる各プログラムの役割、入力フォーマット（SMILES、アミノ酸配列、PDB）、動作仕様、および出力ファイルについて詳細に解説したガイドです。

---

## 1. コア・解析エンジン

### 核心クラス：`PeptideAgent` ([`peptide_agent.py`](peptide_agent.py))
ペプチドおよび化学物質の入力解釈、3D構造構築、トポロジー抽出、および Fiedler値算出を司る共通のコアエンジンです。

* **対応入力フォーマット**（自動判定）:
  1. **1文字アミノ酸配列**: 例 `YYDPETGTWY` （チグノリン）、`NLYIQWLKDGGPSSGRPPPS` （Trp-cage）
  2. **3文字アミノ酸配列**: 例 `Tyr-Tyr-Asp-Pro-Glu-Thr-Gly-Thr-Trp-Tyr`
  3. **SMILES 文字列**: 例 `CC(C)...` や高分子・ペプチドの化学構造SMILES
* **主な処理機能**:
  * RDKit による分子グラフの解釈および 3D 初期構造の生成（ETKDGv3 アルゴリズム）
  * 主鎖・側鎖の回転可能な二面角ジョイント（$\phi, \psi$）の自動特定
  * グラフ・ラプラシアン行列 $L = D - A$ および Fiedler値（$\lambda_2$）のリアルタイム算出

---

## 2. チグノリン（Chignolin）および主要フォールディング・プログラム

論文で発表された各モデルを実行するためのバックエンドおよび起動スクリプト一覧です。

### ①【主論文モデル】水素結合優先段階モデル (Framework Model)
* **バックエンド**: [`server_overall_fiedler_polar_priority.py`](server_overall_fiedler_polar_priority.py)
* **起動スクリプト**: [`run_overall_fiedler_polar_priority.py`](run_overall_fiedler_polar_priority.py)
* **入力**: チグノリン配列 (`YYDPETGTWY`) または対応 SMILES
* **処理内容**: 
  * **Phase 1 (0〜4サイクル)**: 主鎖の水素結合（Polar）の Fiedler値貢献のみを評価・ロックし、$\beta$シート骨格を最優先形成。
  * **Phase 2 (5〜9サイクル)**: 骨格固定後、疎水性コア（Trp9, Tyr1, Tyr3等）のパッキングを解禁・固定。
* **出力**: [`1uao_folded_directional.json`](1uao_folded_directional.json) （慣性半径 $R_g = 5.12\text{ \AA}$、実験値 $5.17\text{ \AA}$ と一致）

### ② 水分子直接結合モデル (Solvent-Direct Coupled Model)
* **バックエンド**: [`server_overall_fiedler_solvent_direct.py`](server_overall_fiedler_solvent_direct.py)
* **起動スクリプト**: [`run_overall_fiedler_solvent_direct.py`](run_overall_fiedler_solvent_direct.py)
* **入力**: ペプチド配列 ＋ 周囲の明示的水分子集団 ($O_w$)
* **処理内容**: 「ペプチド＋水和水素結合ネットワーク」全体の統合 Fiedler値を直接最大化。すべての試行で $R_g \approx 5.12\text{--}5.29\text{ \AA}$ に 100% 収束。

### ③ トポロジカル核形成伝播モデル (Nucleation-Propagation Model)
* **バックエンド**: [`server_overall_fiedler_nucleation.py`](server_overall_fiedler_nucleation.py)
* **起動スクリプト**: [`run_overall_fiedler_nucleation.py`](run_overall_fiedler_nucleation.py)
* **入力**: 長鎖ペプチド Trp-cage (20残基: `NLYIQWLKDGGPSSGRPPPS`) または チグノリン
* **処理内容**: 形成されたコンタクトを核（Nucleus）として固定し、自由度を決定論的に絞り込みながら多残基ペプチドへとスケーリング（Trp-cage 誤差 < 1.9%）。

### ④ 高速化最適化モデル (Optimized Hydrophobic Model)
* **バックエンド**: [`server_overall_fiedler_hydro_priority_optimized.py`](server_overall_fiedler_hydro_priority_optimized.py)
* **起動スクリプト**: [`run_overall_fiedler_hydro_priority_optimized.py`](run_overall_fiedler_hydro_priority_optimized.py)
* **処理内容**: 固有値計算の重複処理を一括化し、精度を損なわずに計算速度を約7.5倍高速化（3試行が1分52秒で完了）。

---

## 3. 高分子（PNIPAM / NIPAゲル）シミュレーション

SMILES 形式で入力された高分子・ジェルの温度応答性脱水崩壊シミュレーション群です。

* [`pnipam_differentiable_folding.py`](pnipam_differentiable_folding.py): PNIPAM（ポリN-イソプロピルアクリルアミド）鎖のFiedler最大化
* [`pnipam_hybrid_folding.py`](pnipam_hybrid_folding.py): 物理反発＋Fiedler結合モデル
* [`nipa_gel_simulation.py`](nipa_gel_simulation.py): NIPAゲルの架橋網目トポロジーシミュレーション

---

## 4. 構造検証および 3D 可視化ツール

### 構造検証：`calculate_rmsd.py` ([`calculate_rmsd.py`](calculate_rmsd.py))
Kabsch アルゴリズムを用いて、シミュレーション出力 JSON の原子座標と、実験室 PDB ファイル（[`1UAO.pdb`](1UAO.pdb)、[`1L2Y.pdb`](1L2Y.pdb)）との間の **RMSD（構造重ね合わせ誤差）** を定量計算します。

### 3D WebGL 可視化：`viewer.html` ([`viewer.html`](viewer.html))
Three.js を用いてブラウザ上でリアルタイム 3D 表示を行うビューアです。
* 機能: 折り畳み軌跡アニメーション再生、C-$\alpha$ バックボーン描画、水素結合破線描画、水分子ネットワーク表示。

---

## 💻 実行手順の例

### 例1: チグノリン（SMILES または アミノ酸配列）のフォールディング実行
```bash
python3 run_overall_fiedler_polar_priority.py
```

### 例2: 折り畳み構造と実験PDBとのRMSD計算
```bash
python3 calculate_rmsd.py
```
