"""
4-Component LNP Explicit Organic Chemist Molecular Model
(Ionizable Lipid : DSPC : Cholesterol : PEG-Lipid + Explicit H2O)
Concept & Organic Chemistry Vision by Yoshihiro Honda (本多 義弘)

Explicit Structural Representation of:
1. Ionizable Lipid (SM-102/ALC-0315 type with tertiary N amine)
2. DSPC Helper Phospholipid (Zwitterionic Phosphocholine PO4- and Choline N+)
3. Cholesterol (Fused 4-ring Steroid Core + Hydroxyl OH head)
4. PEG-Lipid (Polyethylene Glycol (CH2-CH2-O)n steric shield chain)
5. Explicit Water Molecules (H2O V-shape + Hydrogen Bonds)
"""

import numpy as np
import scipy.linalg as la
import json

def build_4component_lnp_explicit_step(step=1):
    """
    Builds explicit atomic coordinates for 4-component LNP lipid assembly + explicit water
    at 4 protonation stages:
    Step 1: Neutral state (pH 7.4)
    Step 2: Protonation N -> NH+ (pH 5.5)
    Step 3: Electrostatic Repulsion & Water Influx
    Step 4: Cone-shape H_II Inverted Hexagonal Water Channel
    """
    np.random.seed(42 + step)
    
    atoms = []
    bonds = []
    h_bonds = []
    
    progress = (step - 1) / 3.0
    r_head = 3.0 + 3.0 * progress
    splay = 0.1 + 0.55 * progress
    is_protonated = (step >= 2)
    
    # 4-Component Mixture Ratios (50% Ionizable, 10% DSPC, 38.5% Cholesterol, 1.5% PEG)
    # Total 12 lipids: 6 Ionizable, 1 DSPC, 4 Cholesterol, 1 PEG-Lipid
    lipid_types = ["ION", "ION", "CHOL", "ION", "DSPC", "CHOL", "ION", "CHOL", "ION", "PEG", "ION", "CHOL"]
    
    n_lipids = len(lipid_types)
    
    for i, ltype in enumerate(lipid_types):
        angle = (2 * np.pi / n_lipids) * i
        nx = r_head * np.cos(angle)
        ny = r_head * np.sin(angle)
        nz = 0.0
        
        if ltype == "ION":
            # 1. Ionizable Lipid (Tertiary Amine N)
            n_idx = len(atoms)
            atoms.append({
                "id": n_idx, "type": "NH+" if is_protonated else "N_ion",
                "element": "N", "pos": [nx, ny, nz], "charge": +1.0 if is_protonated else 0.0
            })
            
            if is_protonated:
                h_idx = len(atoms)
                atoms.append({"id": h_idx, "type": "H_proton", "element": "H", "pos": [nx + 0.8 * np.cos(angle), ny + 0.8 * np.sin(angle), 0.6], "charge": +0.5})
                bonds.append([n_idx, h_idx])
                
            # Dual hydrocarbon tails
            for branch in [-1, 1]:
                prev_idx = n_idx
                for c in range(1, 6):
                    dist = c * 1.4
                    cx = nx + dist * np.cos(angle + branch * splay)
                    cy = ny + dist * np.sin(angle + branch * splay)
                    cz = -c * 1.3
                    c_idx = len(atoms)
                    atoms.append({"id": c_idx, "type": "C_tail", "element": "C", "pos": [cx, cy, cz], "charge": 0.0})
                    bonds.append([prev_idx, c_idx])
                    prev_idx = c_idx

        elif ltype == "DSPC":
            # 2. Helper Phospholipid DSPC (Phosphate P + Choline N)
            p_idx = len(atoms)
            atoms.append({"id": p_idx, "type": "P_dspc", "element": "P", "pos": [nx, ny, nz], "charge": -0.8})
            
            ch_idx = len(atoms)
            atoms.append({"id": ch_idx, "type": "N_dspc", "element": "N", "pos": [nx + 0.8 * np.cos(angle), ny + 0.8 * np.sin(angle), 1.0], "charge": +0.8})
            bonds.append([p_idx, ch_idx])
            
            # Dual C18 DSPC tails
            for branch in [-0.5, 0.5]:
                prev_idx = p_idx
                for c in range(1, 6):
                    dist = c * 1.4
                    cx = nx + dist * np.cos(angle + branch * splay)
                    cy = ny + dist * np.sin(angle + branch * splay)
                    cz = -c * 1.4
                    c_idx = len(atoms)
                    atoms.append({"id": c_idx, "type": "C_dspc", "element": "C", "pos": [cx, cy, cz], "charge": 0.0})
                    bonds.append([prev_idx, c_idx])
                    prev_idx = c_idx

        elif ltype == "CHOL":
            # 3. Cholesterol (OH Head + Steroid Ring Core)
            oh_idx = len(atoms)
            atoms.append({"id": oh_idx, "type": "O_chol", "element": "O", "pos": [nx, ny, nz], "charge": -0.3})
            
            # Fused Steroid Ring Carbon Nodes (4 fused rings)
            prev_idx = oh_idx
            for r in range(1, 5):
                dist = r * 1.3
                cx = nx + dist * np.cos(angle)
                cy = ny + dist * np.sin(angle)
                cz = -r * 1.1
                c_idx = len(atoms)
                atoms.append({"id": c_idx, "type": "C_chol", "element": "C", "pos": [cx, cy, cz], "charge": 0.0})
                bonds.append([prev_idx, c_idx])
                prev_idx = c_idx

        elif ltype == "PEG":
            # 4. PEG-Lipid (Polyethylene Glycol Chain (CH2-CH2-O)n)
            head_idx = len(atoms)
            atoms.append({"id": head_idx, "type": "C_peg_head", "element": "C", "pos": [nx, ny, nz], "charge": 0.0})
            
            # Extending PEG Polymer Chain Outward (Steric Shield)
            prev_idx = head_idx
            for p in range(1, 7):
                dist = p * 1.4
                px = nx + dist * np.cos(angle)
                py = ny + dist * np.sin(angle)
                pz = p * 1.2 # Polymer extending into water phase
                p_idx = len(atoms)
                atoms.append({"id": p_idx, "type": "O_peg" if p % 3 == 0 else "C_peg", "element": "O" if p % 3 == 0 else "C", "pos": [px, py, pz], "charge": 0.0})
                bonds.append([prev_idx, p_idx])
                prev_idx = p_idx

    # 5. Explicit Water Molecules (H2O)
    n_water = 30
    for w in range(n_water):
        if step == 1:
            z_w = np.random.uniform(-10, 10)
            r_w = r_head + np.random.uniform(5.0, 9.0)
            angle_w = np.random.uniform(0, 2 * np.pi)
        else:
            z_w = (w / (n_water - 1)) * 22.0 - 11.0
            r_w = np.random.uniform(0.1, 2.2)
            angle_w = np.random.uniform(0, 2 * np.pi)
            
        ox = r_w * np.cos(angle_w)
        oy = r_w * np.sin(angle_w)
        oz = z_w
        
        o_idx = len(atoms)
        atoms.append({"id": o_idx, "type": "O_water", "element": "O", "pos": [ox, oy, oz], "charge": -0.6})
        h1_idx = len(atoms)
        atoms.append({"id": h1_idx, "type": "H_water", "element": "H", "pos": [ox + 0.8, oy + 0.5, oz + 0.3], "charge": +0.3})
        bonds.append([o_idx, h1_idx])
        h2_idx = len(atoms)
        atoms.append({"id": h2_idx, "type": "H_water", "element": "H", "pos": [ox - 0.8, oy + 0.5, oz - 0.3], "charge": +0.3})
        bonds.append([o_idx, h2_idx])
        
        if step >= 3 and abs(r_w) < 2.8 and w > 0:
            h_bonds.append([o_idx - 3, o_idx])

    # Spectral Calculation
    N_total = len(atoms)
    adj = np.zeros((N_total, N_total))
    for b in bonds:
        adj[b[0], b[1]] = adj[b[1], b[0]] = 2.0
    for hb in h_bonds:
        adj[hb[0], hb[1]] = adj[hb[1], hb[0]] = 1.2
        
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
    print("  4-Component LNP Explicit Organic Chemist Model (Honda Concept)         ")
    print("==========================================================================")
    
    all_steps = []
    for s in range(1, 5):
        data = build_4component_lnp_explicit_step(step=s)
        all_steps.append(data)
        print(f"Step {s}: Atoms={len(data['atoms'])} | Bonds={len(data['bonds'])} | H-Bonds={len(data['h_bonds'])} | P={data['packing_P']:.2f} | λ2={data['fiedler_l2']:.4f}")

    with open("lnp_4component_explicit_data.json", "w") as f:
        json.dump(all_steps, f, indent=2, ensure_ascii=False)
        
    print("\nData exported to lnp_4component_explicit_data.json successfully.")

if __name__ == "__main__":
    main()

