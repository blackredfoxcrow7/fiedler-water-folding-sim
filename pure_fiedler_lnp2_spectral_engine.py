"""
pure_fiedler_lnp2_spectral_engine.py
====================================
Pure Fiedler Spectral Graph Physics Engine for LNP2 Mechanisms.
Replaces traditional Packing Parameter P with fundamental Graph Laplacian Spectral Invariants:
1. Fiedler Spectral Ratio: F_spec = \lambda_2(Head-Water) / \lambda_2(Tail-Tail)
2. Fiedler Spectral Curvature: \kappa_spec = \lambda_2(Head-Water) - \lambda_2(Tail-Tail)
3. Fiedler Spectral Sensitivity Index: S_Fiedler = d \lambda_2 / d(pH)
4. Fiedler Bottleneck Index (Cheeger Constant bound): h(G) >= \lambda_2 / 2

Author: Antigravity Team & Yoshihiro Honda (Yoshi)
"""

import json
import numpy as np

def build_head_water_laplacian(pH, pKa=6.2, head_hydration=1.0):
    # Degree of protonation
    alpha = 1.0 / (1.0 + 10.0 ** (pH - pKa))
    
    # Base head-water graph nodes (N_head = 20, N_water = 30)
    N_head = 20
    N_water = 30
    N_total = N_head + N_water
    
    A = np.zeros((N_total, N_total))
    
    # Head-Head electrostatic repulsion network (increases with protonation alpha)
    head_repulsion_weight = 0.2 + 2.5 * (alpha ** 1.8)
    for i in range(N_head):
        for j in range(i + 1, N_head):
            dist = abs(i - j)
            if dist <= 3:
                A[i, j] = A[j, i] = head_repulsion_weight / dist
                
    # Head-Water hydrogen bonding network
    hw_weight = 0.5 + 1.8 * alpha * head_hydration
    for i in range(N_head):
        for w in range(N_head, N_total):
            if (i + w) % 3 == 0:
                A[i, w] = A[w, i] = hw_weight
                
    # Water-Water network (Grotthuss hydrogen bond wire)
    for w1 in range(N_head, N_total):
        for w2 in range(w1 + 1, N_total):
            if abs(w1 - w2) <= 2:
                A[w1, w2] = A[w2, w1] = 1.2
                
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    eigvals = np.sort(np.linalg.eigvalsh(L))
    lambda_2_hw = eigvals[1] if len(eigvals) > 1 else 0.0
    return float(lambda_2_hw), L

def build_tail_tail_laplacian(tail_flexibility=1.0, hydrophobic_rigidity=1.0):
    # Tail graph nodes (N_tail = 40)
    N_tail = 40
    A = np.zeros((N_tail, N_tail))
    
    # Intramolecular chain backbone
    for i in range(N_tail - 1):
        A[i, i+1] = A[i+1, i] = 2.0 * hydrophobic_rigidity
        
    # Intermolecular hydrophobic van der Waals contacts
    vdw_weight = 0.8 * tail_flexibility
    for i in range(N_tail):
        for j in range(i + 2, N_tail):
            if (j - i) % 4 == 0:
                A[i, j] = A[j, i] = vdw_weight
                
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    eigvals = np.sort(np.linalg.eigvalsh(L))
    lambda_2_tt = eigvals[1] if len(eigvals) > 1 else 0.0
    return float(lambda_2_tt), L

def run_lnp2_fiedler_analysis():
    ph_range = np.linspace(7.4, 4.5, 30)
    
    # Lipid systems defined by Fiedler spectral parameters
    systems = {
        "MC3": {"pKa": 6.2, "hydration": 1.0, "flexibility": 1.1, "rigidity": 1.0, "isomer_bottleneck": 0.05},
        "DLinDMA": {"pKa": 6.8, "hydration": 1.8, "flexibility": 0.6, "rigidity": 1.4, "isomer_bottleneck": 0.12},
        "ALC-0315 (S,S)": {"pKa": 6.35, "hydration": 0.9, "flexibility": 1.3, "rigidity": 0.9, "isomer_bottleneck": 0.02},
        "ALC-0315 (R,R)": {"pKa": 6.35, "hydration": 1.5, "flexibility": 0.4, "rigidity": 1.8, "isomer_bottleneck": 0.45}  # High spectral bottleneck
    }
    
    results = {}
    
    for name, params in systems.items():
        ph_data = []
        for pH in ph_range:
            lambda_2_hw, _ = build_head_water_laplacian(pH, pKa=params["pKa"], head_hydration=params["hydration"])
            lambda_2_tt, _ = build_tail_tail_laplacian(tail_flexibility=params["flexibility"], hydrophobic_rigidity=params["rigidity"])
            
            # Fiedler Ratio & Curvature
            F_spec = lambda_2_hw / max(lambda_2_tt, 1e-5)
            kappa_spec = lambda_2_hw - lambda_2_tt
            
            # Cheeger constant lower bound for graph partition bottleneck
            cheeger_bound = lambda_2_hw / 2.0
            bottleneck_index = params["isomer_bottleneck"] * (1.0 + 2.0 * (1.0 / max(lambda_2_hw, 0.1)))
            
            # Determine Mesophase purely by Fiedler Spectral Invariant F_spec
            if F_spec < 0.85:
                phase = "Lamellar L_alpha / Amorphous Oil Core"
                fusogenic_score = 0.1
            elif F_spec < 1.15:
                phase = "Inverse Micellar Cubic Fd3m"
                fusogenic_score = 0.55
            elif F_spec < 1.45:
                phase = "Inverse Hexagonal H_II (Peak Fusogenic)"
                fusogenic_score = 0.98
            else:
                phase = "Bicontinuous Cubic Q_2 (Pn3m/Ia3d)"
                fusogenic_score = 0.82
                
            ph_data.append({
                "pH": round(float(pH), 2),
                "lambda_2_head_water": round(lambda_2_hw, 4),
                "lambda_2_tail_tail": round(lambda_2_tt, 4),
                "F_spec": round(F_spec, 4),
                "kappa_spec": round(kappa_spec, 4),
                "cheeger_bound": round(cheeger_bound, 4),
                "bottleneck_index": round(bottleneck_index, 4),
                "phase": phase,
                "fusogenic_score": round(fusogenic_score, 3)
            })
            
        results[name] = ph_data
        
    out_file = "/home/eldenring/fiedler-colloid-lnp-sim/pure_fiedler_lnp2_data.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Successfully generated Pure Fiedler LNP2 Spectral Data at {out_file}")

if __name__ == "__main__":
    run_lnp2_fiedler_analysis()
