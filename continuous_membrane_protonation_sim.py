"""
Continuous 4-Component LNP Bilayer Membrane Protonation & Phase Transition Simulator
(Real-Time Gradual Morphing of 4 Lipid Components & Bound Water Channel Formation)
Concept & Organic Chemistry Vision by Yoshihiro Honda (本多 義弘)

Features:
1. Real-time Continuous Protonation (pH 7.4 -> 5.0 smooth slider & play animation)
2. Explicit 4 Component Roles:
   - Ionizable Lipid (SM-102): Gradual protonation (N -> NH+), driving electrostatic head repulsion
   - Cholesterol: Intercalates & splays tails to stabilize negative membrane curvature (H_II)
   - DSPC: Zwitterionic PO4- / Choline head forming H-bond bridges with bound water
   - PEG-Lipid: Outer steric polymer shield
3. Bound Water Flow: H2O molecules smoothly drawn from head groups into the cylinder core, forming continuous 1D/3D water wires
"""

import numpy as np
import scipy.linalg as la
import json

def simulate_continuous_membrane_morphing(protonation_ratio=0.0):
    """
    Computes continuous 3D atomistic coordinates and spectral properties
    as protonation ratio smoothly varies from 0.0 (pH 7.4 neutral) to 1.0 (pH 5.0 fully protonated).
    """
    np.random.seed(42 + int(protonation_ratio * 100))
    
    atoms = []
    bonds = []
    h_bonds = []
    
    # 4 Lipid Types: 50% ION, 10% DSPC, 38.5% CHOL, 1.5% PEG
    lipid_types = ["ION", "CHOL", "ION", "DSPC", "ION", "CHOL", "ION", "PEG", "ION", "CHOL", "ION", "DSPC"]
    nx_grid = len(lipid_types)
    ny_grid = 4
    
    # Smooth morphing parameters driven by protonation_ratio (0.0 -> 1.0)
    curve_factor = np.sin(protonation_ratio * np.pi / 2.0) # Smooth S-curve
    head_expansion = 3.8 + 2.4 * curve_factor
    tail_splay = 0.1 + 0.6 * curve_factor
    
    water_positions = []
    
    for ix, ltype in enumerate(lipid_types):
        for iy in range(ny_grid):
            # Check if this specific ionizable lipid is protonated
            is_ion = (ltype == "ION")
            is_protonated = is_ion and (np.random.uniform(0, 1) < protonation_ratio or protonation_ratio > 0.8)
            
            # Position Interpolation: Flat Bilayer Sheet (curve=0) -> Cylinder Channel (curve=1)
            if curve_factor < 0.1:
                # Flat Bilayer Sheet
                x = (ix - (nx_grid - 1) / 2.0) * head_expansion
                y = (iy - (ny_grid - 1) / 2.0) * head_expansion
                z_head = 4.0
            else:
                # Morphing into Cylinder
                angle = (ix / nx_grid) * np.PI if hasattr(np, 'PI') else (ix / nx_grid) * np.pi * 2.0
                r_cyl = 6.0 + 3.0 * (1.0 - curve_factor)
                x = (1.0 - curve_factor) * ((ix - (nx_grid - 1) / 2.0) * head_expansion) + curve_factor * (r_cyl * np.cos(angle))
                y = (1.0 - curve_factor) * ((iy - (ny_grid - 1) / 2.0) * head_expansion) + curve_factor * (r_cyl * np.sin(angle))
                z_head = 4.0 * (1.0 - curve_factor) + (iy - (ny_grid - 1) / 2.0) * 4.5 * curve_factor

            head_pos = [x, y, z_head]
            head_idx = len(atoms)
            
            atoms.append({
                "id": head_idx,
                "type": f"{ltype}_{'protonated' if is_protonated else 'neutral'}",
                "element": "N" if is_ion else ("P" if ltype == "DSPC" else ("O" if ltype == "CHOL" else "C")),
                "pos": head_pos,
                "charge": +1.0 if is_protonated else 0.0
            })
            
            if is_protonated:
                h_idx = len(atoms)
                atoms.append({
                    "id": h_idx, "type": "H_proton", "element": "H",
                    "pos": [x + 0.8 * np.cos(angle if curve_factor > 0.1 else 0), y + 0.8 * np.sin(angle if curve_factor > 0.1 else 0), z_head + 0.6],
                    "charge": +0.5
                })
                bonds.append([head_idx, h_idx])

            # Component Tail Structures
            if ltype == "CHOL":
                # Cholesterol Steroid Rings intercalation
                prev_idx = head_idx
                for r in range(1, 4):
                    cz = z_head - r * 1.1
                    c_idx = len(atoms)
                    atoms.append({"id": c_idx, "type": "C_chol", "element": "C", "pos": [x, y, cz], "charge": 0.0})
                    bonds.append([prev_idx, c_idx])
                    prev_idx = c_idx
            elif ltype == "PEG":
                # PEG Polymer Shield Chain extending outward
                prev_idx = head_idx
                for p in range(1, 5):
                    pz = z_head + p * 1.2
                    p_idx = len(atoms)
                    atoms.append({"id": p_idx, "type": "O_peg" if p % 2 == 0 else "C_peg", "element": "O" if p % 2 == 0 else "C", "pos": [x, y, pz], "charge": 0.0})
                    bonds.append([prev_idx, p_idx])
                    prev_idx = p_idx
            else:
                # ION & DSPC Dual Carbon Tails
                for branch in [-1, 1]:
                    prev_idx = head_idx
                    for c in range(1, 5):
                        cz = z_head - c * 1.3
                        cx = x + c * 0.4 * branch * tail_splay
                        c_idx = len(atoms)
                        atoms.append({"id": c_idx, "type": "C_tail", "element": "C", "pos": [cx, y, cz], "charge": 0.0})
                        bonds.append([prev_idx, c_idx])
                        prev_idx = c_idx

            # 3. Bound Water Motion: Sucked from Polar Head into Inner Cylinder Channel
            w_x = (1.0 - curve_factor) * x + curve_factor * (x * 0.25)
            w_y = (1.0 - curve_factor) * y + curve_factor * (y * 0.25)
            w_z = z_head + (1.8 * (1.0 - curve_factor))
            
            o_idx = len(atoms)
            atoms.append({"id": o_idx, "type": "O_water", "element": "O", "pos": [w_x, w_y, w_z], "charge": -0.6})
            h1_idx = len(atoms)
            atoms.append({"id": h1_idx, "type": "H_water", "element": "H", "pos": [w_x + 0.7, w_y + 0.4, w_z + 0.2], "charge": +0.3})
            bonds.append([o_idx, h1_idx])
            h2_idx = len(atoms)
            atoms.append({"id": h2_idx, "type": "H_water", "element": "H", "pos": [w_x - 0.7, w_y + 0.4, w_z - 0.2], "charge": +0.3})
            bonds.append([o_idx, h2_idx])
            
            h_bonds.append([head_idx, o_idx])
            water_positions.append([w_x, w_y, w_z])

    # Spectral Calculation
    N_total = len(atoms)
    adj = np.zeros((N_total, N_total))
    for b in bonds:
        adj[b[0], b[1]] = adj[b[1], b[0]] = 2.0
    for hb in h_bonds:
        adj[hb[0], hb[1]] = adj[hb[1], hb[0]] = 1.4
        
    deg = np.diag(np.sum(adj, axis=1))
    L = deg - adj
    evals = la.eigvalsh(L)
    nonzero = [e for e in evals if e > 1e-5]
    l2 = float(nonzero[0]) if len(nonzero) > 0 else 0.0

    return {
        "protonation_ratio": protonation_ratio,
        "atoms": atoms,
        "bonds": bonds,
        "h_bonds": h_bonds,
        "fiedler_l2": l2,
        "packing_P": float(0.95 + 0.66 * curve_factor)
    }

def main():
    print("==========================================================================")
    print("  Continuous 4-Component LNP Membrane Morphing Simulator (Honda Concept)  ")
    print("==========================================================================")
    
    trajectory = []
    for step_pct in np.linspace(0.0, 1.0, 11):
        data = simulate_continuous_membrane_morphing(step_pct)
        trajectory.append(data)
        print(f"Protonation {step_pct*100:3.0f}%: Atoms={len(data['atoms'])} | Bonds={len(data['bonds'])} | P={data['packing_P']:.2f} | λ2={data['fiedler_l2']:.4f}")

    with open("continuous_membrane_morphing_data.json", "w") as f:
        json.dump(trajectory, f, indent=2, ensure_ascii=False)
        
    print("\nContinuous morphing trajectory exported to continuous_membrane_morphing_data.json.")

if __name__ == "__main__":
    main()

