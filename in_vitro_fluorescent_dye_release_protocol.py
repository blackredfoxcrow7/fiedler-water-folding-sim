"""
In-Vitro Fluorescent Dye Vesicle Release Assay Simulator & Protocol
Quantitative Assay for pH-Triggered Encapsulation Release (Calcein Self-Quenching Method)
Concept & Experimental Design by Yoshihiro Honda (本多 義弘)

Principle:
1. Encapsulate high-concentration fluorophore (Calcein, 50-100 mM) inside vesicles/LNPs.
   (Fluorescence is self-quenched in concentrated core, F_baseline ≈ 0).
2. Purify encapsulated vesicles using gel filtration column (Sephadex G-50) or fabric adsorption.
3. Lower pH (pH 7.4 -> 5.5): Protonation triggers water channel percolation (λ2 jump).
4. Dye leaks out into diluted medium, restoring bright fluorescence (F_release).
5. Add 0.1% Triton X-100 to determine maximum fluorescence (F_max 100% lysis).
"""

import numpy as np
import scipy.linalg as la
import json

def simulate_fluorescent_dye_release_kinetics(ph_level=5.5, f_ionizable=0.50, f_cholesterol=0.385, time_minutes=60):
    """
    Simulates fluorometric release kinetics F(t) for Calcein-encapsulated vesicles.
    Correlates Graph-Spectral Fiedler Connectivity (λ2) with measured Fluorophore Release (%).
    """
    t_points = np.linspace(0, time_minutes, 61)
    
    is_acidic = (ph_level < 6.2)
    
    # 1. Graph Spectral Calculation (Fiedler Water Channel Connectivity λ2)
    N = 60
    np.random.seed(42)
    pos = np.random.uniform(-8, 8, (N, 3))
    
    # Adjacency matrix for water/dye channel
    adj = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < 4.5:
                w = np.exp(-d**2 / 10.0)
                if is_acidic:
                    w *= (1.8 + 1.2 * f_ionizable) # Enhanced percolation at acidic pH
                adj[i, j] = adj[j, i] = w
                
    deg = np.diag(np.sum(adj, axis=1))
    L = deg - adj
    evals = la.eigvalsh(L)
    nonzero = [e for e in evals if e > 1e-5]
    l2 = float(nonzero[0]) if len(nonzero) > 0 else 0.0
    
    # 2. Release Rate Constant k_release (min^-1) driven by λ2 and pH
    if is_acidic:
        k_rel = 0.08 * (l2 / 0.02) * (1.0 + (6.5 - ph_level) * 0.5)
        max_release_pct = min(98.0, 75.0 + 30.0 * (f_ionizable - 0.20))
    else:
        k_rel = 0.002 # Very slow baseline leakage at pH 7.4
        max_release_pct = 5.0 # Stable encapsulation
        
    # First-order Release Kinetics: % Release = Max * (1 - exp(-k * t))
    release_pct_curve = max_release_pct * (1.0 - np.exp(-k_rel * t_points))
    
    # Baseline fluorescence with self-quenching noise
    fluorescence_intensity = 5.0 + release_pct_curve * 9.5
    
    return {
        "ph_level": ph_level,
        "lambda2_water_channel": l2,
        "k_release_min_1": float(k_rel),
        "final_release_pct": float(release_pct_curve[-1]),
        "time_points_min": t_points.tolist(),
        "release_pct_curve": release_pct_curve.tolist(),
        "fluorescence_intensity_RFU": fluorescence_intensity.tolist()
    }

def generate_experimental_protocol():
    """Generates step-by-step laboratory experimental protocol for Honda's Fluorometric Release Assay."""
    return {
        "title": "簡易型・蛍光色素内包ベシクルによるpH応答性放出評価プロトコル",
        "author": "本多 義弘 (Yoshihiro Honda)",
        "materials": [
            "カルセイン (Calcein) または カルボキシフルオレセイン (50 mM in PBS Buffer, pH 7.4)",
            "イオン化脂質 / カチオン活性剤 (SM-102, ALC-0315, または第四級アンモニウム塩)",
            "Helper脂質 (DSPC / レシチン) & コレステロール",
            "ゲルろ過カラム (Sephadex G-50 または PD-10 脱塩カラム)",
            "トリトン X-100 (Triton X-100, 10% Aqueous Solution - 100%可溶化コントロール)",
            "分光蛍光光度計 (励起波長 495 nm, 蛍光波長 515 nm) または 紫外線ランプ (UV Lamp 365 nm)"
        ],
        "steps": [
            "Step 1 [ベシクル調製]: 脂質混合物と高濃度カルセイン溶液 (50 mM) を混合し、超音波処理 (Sonication) でカルセイン内包ベシクルを形成させる (内包内は自己消光により蛍光オフ)。",
            "Step 2 [カラム精製]: ゲルろ過カラム (Sephadex G-50) に通し、ベシクル外の未内包カルセインを除去して、内包ベシクル画分のみを抽出する (布や膜への吸着固定も可)。",
            "Step 3 [pH変化・蛍光測定]: クエン酸緩衝液を加えて溶液のpHを pH 7.4 (生理的) から pH 5.5 (酸性) へ下げる。",
            "Step 4 [放出検出]: 水チャネル形成に伴いカルセインが希釈放出され、自己消光が解除されて蛍光強度が急上昇する様子を分光蛍光光度計でリアルタイム測定する。",
            "Step 5 [100%対照比算]: 最後に 0.1% Triton X-100 を加えて膜を完全破壊し、最大蛍光強度 F_max を得て、放出率 (%) = (F - F_initial) / (F_max - F_initial) * 100 を算出する。"
        ]
    }

def main():
    print("==========================================================================")
    print("  In-Vitro Fluorescent Dye Vesicle Release Assay Simulator (Honda Assay)  ")
    print("==========================================================================")
    
    # Simulate pH 7.4 (Physiological Storage) vs pH 5.5 (Acidic Endosomal Escape)
    res_74 = simulate_fluorescent_dye_release_kinetics(ph_level=7.4, time_minutes=30)
    res_55 = simulate_fluorescent_dye_release_kinetics(ph_level=5.5, time_minutes=30)
    
    print("\n--- 1. Neutral Storage Control (pH 7.4) ---")
    print(f"Graph Fiedler λ2: {res_74['lambda2_water_channel']:.4f}")
    print(f"Release Rate k: {res_74['k_release_min_1']:.4f} min^-1")
    print(f"Final 30-min Dye Release: {res_74['final_release_pct']:.1f}% (Stable Encapsulation)")
    
    print("\n--- 2. Acidic Activation Assay (pH 5.5) ---")
    print(f"Graph Fiedler λ2: {res_55['lambda2_water_channel']:.4f}")
    print(f"Release Rate k: {res_55['k_release_min_1']:.4f} min^-1")
    print(f"Final 30-min Dye Release: {res_55['final_release_pct']:.1f}% (Rapid Fluorophore Burst!)")

    protocol = generate_experimental_protocol()
    
    output = {
        "protocol": protocol,
        "simulations": {
            "ph_7_4": res_74,
            "ph_5_5": res_55
        }
    }
    
    with open("in_vitro_dye_release_assay_results.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
        
    print("\nProtocol and simulation data exported to in_vitro_dye_release_assay_results.json.")

if __name__ == "__main__":
    main()

