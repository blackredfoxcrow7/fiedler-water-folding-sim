"""
Real 3D Lipid Bilayer Membrane & Protonation Phase Transition Simulator
(Bound Water at Polar Head Groups & Membrane Topological Transition)
Concept & Organic Chemistry Vision by Yoshihiro Honda (本多 義弘)

Step 1: Planar Lipid Bilayer Sheet with Water Hydrogen-Bonded to Polar Head Groups (pH 7.4)
Step 2: Surface Protonation (N + H+ -> NH+)
Step 3: Repulsion & Membrane Curvature Induction (Head expansion & water influx)
Step 4: Topological Rearrangement into Inverted Hexagonal Channel (H_II Phase)
"""

import numpy as np
import scipy.linalg as la
import json

def build_real_membrane_protonation_step(step=1):
    """
    Builds explicit 3D Planar Bilayer Membrane Sheet + Bound Water at Polar Heads.
    Transitioning into Inverted Hexagonal Cylindrical Channels upon protonation.
    """
    np.random.seed(101 + step)
    
    atoms = []
    bonds = []
    h_bonds = []
    
    is_protonated = (step >= 2)
    progress = (step - 1) / 3.0
    
    # Grid dimensions for Planar Bilayer Sheet (Step 1-2) -> Cylinder (Step 3-4)
    nx_grid = 6
    ny_grid = 4
    
    lipid_types = ["ION", "CHOL", "ION", "DSPC", "ION", "PEG"]
    
    # 1. Lipid Bilayer Assembly (Upper & Lower Leaflets)
    for ix in range(nx_grid):
        for iy in range(ny_grid):
            ltype = lipid_types[(ix + iy) % len(lipid_types)]
            
            # Position morphing from Planar Bilayer Sheet to Cylinder
            if step <= 2:
                # Step 1 & 2: Planar Membrane Sheet in X-Y plane
                x = (ix - (nx_grid - 1) / 2.0) * 3.8
                y = (iy - (ny_grid - 1) / 2.0) * 3.8
                z_head_upper = 3.5
                z_head_lower = -3.5
            else:
                # Step 3 & 4: Curving into Cylindrical Channel
                angle = (ix / nx_grid) * 2.0 * np.pi
                r_cyl = 5.0 + 2.0 * progress
                x = r_cyl * np.cos(angle)
                y = r_cyl * np.sin(angle)
                z_head_upper = (iy - (ny_grid - 1) / 2.0) * 4.5
                z_head_lower = z_head_upper
                
            # Upper Leaflet Polar Head
            head_idx = len(atoms)
            charge_n = +1.0 if (ltype == "ION" and is_protonated) else 0.0
            
            atoms.append({
                "id": head_idx,
                "type": f"{ltype}_head_{'protonated' if charge_n > 0 else 'neutral'}",
                "element": "N" if ltype == "ION" else ("P" if ltype == "DSPC" else ("O" if ltype == "CHOL" else "C")),
                "pos": [x, y, z_head_upper],
                "charge": charge_n
            })
            
            # If protonated ION, add attached proton H+
            if ltype == "ION" and is_protonated:
                h_idx = len(atoms)
                atoms.append({
                    "id": h_idx, "type": "H_proton", "element": "H",
                    "pos": [x + 0.8, y + 0.5, z_head_upper + 0.6], "charge": +0.5
                })
                bonds.append([head_idx, h_idx])
                
            # Hydrophobic Tails extending inward
            tail_dir = -1.0 if step <= 2 else -0.8
            prev_idx = head_idx
            for c in range(1, 5):
                cz = z_head_upper + tail_dir * c * 1.3
                c_idx = len(atoms)
                atoms.append({
                    "id": c_idx, "type": "C_tail", "element": "C",
                    "pos": [x, y, cz], "charge": 0.0
                })
                bonds.append([prev_idx, c_idx])
                prev_idx = c_idx
                
            # 2. Bound Water Molecules Hydrogen-Bonded to Polar Head Groups!
            # In Step 1: Water is bound directly on top of head groups (N...H-O-H, PO4...H-O-H)
            # In Step 3-4: Water is pulled inside the cylinder channel
            if step <= 2:
                w_pos = [x + np.random.uniform(-0.5, 0.5), y + np.random.uniform(-0.5, 0.5), z_head_upper + 1.8]
            else:
                w_pos = [x * 0.3, y * 0.3, z_head_upper]
                
            o_idx = len(atoms)
            atoms.append({"id": o_idx, "type": "O_bound_water", "element": "O", "pos": w_pos, "charge": -0.6})
            h1_idx = len(atoms)
            atoms.append({"id": h1_idx, "type": "H_water", "element": "H", "pos": [w_pos[0] + 0.7, w_pos[1] + 0.4, w_pos[2] + 0.2], "charge": +0.3})
            bonds.append([o_idx, h1_idx])
            h2_idx = len(atoms)
            atoms.append({"id": h2_idx, "type": "H_water", "element": "H", "pos": [w_pos[0] - 0.7, w_pos[1] + 0.4, w_pos[2] - 0.2], "charge": +0.3})
            bonds.append([o_idx, h2_idx])
            
            # H-bond from polar head group to bound water!
            h_bonds.append([head_idx, o_idx])

    # Spectral Laplacian Calculation
    N_total = len(atoms)
    adj = np.zeros((N_total, N_total))
    for b in bonds:
        adj[b[0], b[1]] = adj[b[1], b[0]] = 2.0
    for hb in h_bonds:
        adj[hb[0], hb[1]] = adj[hb[1], hb[0]] = 1.4 # Strong bound-water H-bond
        
    deg = np.diag(np.sum(adj, axis=1))
    L = deg - adj
    evals = la.eigvalsh(L)
    nonzero = [e for e in evals if e > 1e-5]
    l2 = float(nonzero[0]) if len(nonzero) > 0 else 0.0

    return {
        "step": step,
        "atoms": atoms,
        "bonds": bonds,
        "h_bonds": h_bonds,
        "fiedler_l2": l2,
        "packing_P": float(0.95 + 0.22 * (step - 1))
    }

def main():
    print("==========================================================================")
    print("  Real 3D Lipid Bilayer Membrane & Protonation Simulator (Honda Concept)  ")
    print("==========================================================================")
    
    all_steps = []
    for s in range(1, 5):
        data = build_real_membrane_protonation_step(step=s)
        all_steps.append(data)
        print(f"Step {s}: Atoms={len(data['atoms'])} | Bonds={len(data['bonds'])} | H-Bonds={len(data['h_bonds'])} | P={data['packing_P']:.2f} | λ2={data['fiedler_l2']:.4f}")

    with open("real_membrane_protonation_data.json", "w") as f:
        json.dump(all_steps, f, indent=2, ensure_ascii=False)
        
    print("\nData exported to real_membrane_protonation_data.json successfully.")

if __name__ == "__main__":
    main()
