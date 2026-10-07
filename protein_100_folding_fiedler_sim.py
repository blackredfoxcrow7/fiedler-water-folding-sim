"""
protein_100_folding_fiedler_sim.py
==================================
Graph Spectral Physics Engine for 100-Residue Protein Folding Dynamics.
Computes Graph Laplacian Fiedler Eigenvalue \lambda_2 for a 100-amino-acid chain
undergoing dynamic hydrophobic collapse and tertiary structure folding.

Author: Antigravity Team & Yoshihiro Honda (Yoshi)
"""

import json
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix

def generate_100_residue_sequence():
    # Realistic 100-amino acid protein sequence with hydrophobic core & polar surface
    # H: Hydrophobic (Leu, Val, Ile, Phe, Met)
    # P: Polar/Charged (Lys, Arg, Asp, Glu, Ser, Thr)
    # T: Turn/Flexible (Gly, Pro)
    pattern = "HPPHHPPHHT" * 10
    sequence = []
    amino_types = {'H': 0, 'P': 1, 'T': 2}
    for char in pattern:
        sequence.append(amino_types[char])
    return np.array(sequence)

def compute_folding_frame(s_coordinate, sequence):
    # s_coordinate in [0, 1]: 0 = Extended Denatured Chain, 1 = Compact Native Fold
    N_amino = len(sequence)
    
    # 3D Coordinates generation along reaction coordinate s
    coords_amino = np.zeros((N_amino, 3))
    
    # Radius of gyration shrinks as s increases from extended (R ~ 25) to compact (R ~ 8)
    R_gyration = 24.0 * (1.0 - 0.68 * s_coordinate)
    
    for i in range(N_amino):
        # Extended chain has helical/random coil pitch; Native fold forms globular hydrophobic core
        theta = i * 0.45 * (1.0 + 1.2 * s_coordinate)
        phi = i * 0.18 + s_coordinate * np.pi
        
        # Hydrophobic residues (type 0) pull inwards to center (r_ratio < 1)
        # Polar residues (type 1) remain on surface (r_ratio ~ 1)
        type_factor = 0.55 if sequence[i] == 0 else 1.05
        r_current = R_gyration * (0.4 + 0.6 * np.sin(i * 0.15)) * (1.0 - s_coordinate * (1.0 - type_factor))
        
        coords_amino[i, 0] = r_current * np.sin(phi) * np.cos(theta)
        coords_amino[i, 1] = r_current * np.sin(phi) * np.sin(theta)
        coords_amino[i, 2] = r_current * np.cos(phi)
        
    # Dynamic Water Nodes generation (N_water = 200)
    # Water surrounds polar surface residues and is expelled from hydrophobic core as s -> 1
    N_water = 200
    coords_water = []
    
    for w in range(N_water):
        # Pick random polar residue to attach water node
        polar_indices = np.where(sequence == 1)[0]
        ref_idx = polar_indices[w % len(polar_indices)]
        ref_pos = coords_amino[ref_idx]
        
        # Water node position with dynamic thermal noise
        offset = np.random.normal(0, 1.8, 3)
        water_pos = ref_pos + offset * (1.0 + 0.5 * (1.0 - s_coordinate))
        coords_water.append(water_pos)
        
    coords_water = np.array(coords_water)
    
    # Construct Total Graph Laplacian L_total (Dimension = 100 + 200 = 300)
    N_total = N_amino + N_water
    all_coords = np.vstack([coords_amino, coords_water])
    
    # Distance matrix
    diff = all_coords[:, np.newaxis, :] - all_coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
    
    # Adjacency Matrix A construction
    A = np.zeros((N_total, N_total))
    
    # 1. Amino-Amino Covalent Backbone & Tertiary Contacts
    for i in range(N_amino):
        for j in range(i + 1, N_amino):
            d = dist_matrix[i, j]
            if abs(i - j) == 1:
                A[i, j] = A[j, i] = 3.5  # Backbone Covalent Bond
            elif d < 7.0:
                # Hydrophobic tertiary contact strength
                hydro_weight = 2.5 if (sequence[i] == 0 and sequence[j] == 0) else 1.0
                A[i, j] = A[j, i] = hydro_weight * np.exp(-d / 3.0)
                
    # 2. Amino-Water Hydration Edges
    for i in range(N_amino):
        for w in range(N_amino, N_total):
            d = dist_matrix[i, w]
            if d < 5.5:
                # Polar residues form strong hydration edges; hydrophobic expel water
                hydra_weight = 2.2 if sequence[i] == 1 else 0.3
                A[i, w] = A[w, i] = hydra_weight * np.exp(-d / 2.5)
                
    # 3. Water-Water Grotthuss Hydrogen Bond Wire
    for w1 in range(N_amino, N_total):
        for w2 in range(w1 + 1, N_total):
            d = dist_matrix[w1, w2]
            if d < 4.0:
                A[w1, w2] = A[w2, w1] = 1.4 * np.exp(-d / 2.0)
                
    # Graph Laplacian Matrix L = D - A
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    
    # Compute Fiedler Eigenvalue \lambda_2 (Algebraic Connectivity)
    # Using scipy sparse solver for ultrafast execution
    L_csr = csr_matrix(L)
    try:
        eigvals, _ = eigsh(L_csr, k=2, which='SM', tol=1e-4)
        lambda_2 = float(eigvals[1])
    except:
        eigvals = np.sort(np.linalg.eigvalsh(L))
        lambda_2 = float(eigvals[1]) if len(eigvals) > 1 else 0.0
        
    # Subgraph Fiedler Ratio: \lambda_2(Protein-Water) / \lambda_2(Core-Core)
    L_amino = (np.diag(np.sum(A[:N_amino, :N_amino], axis=1)) - A[:N_amino, :N_amino])
    eig_amino = np.sort(np.linalg.eigvalsh(L_amino))
    lambda_2_amino = float(eig_amino[1]) if len(eig_amino) > 1 else 0.1
    
    fiedler_ratio = lambda_2 / max(lambda_2_amino, 1e-4)
    
    return {
        "s": round(float(s_coordinate), 3),
        "lambda_2_total": round(lambda_2, 4),
        "lambda_2_amino": round(lambda_2_amino, 4),
        "fiedler_ratio": round(fiedler_ratio, 4),
        "rg_gyration": round(float(R_gyration), 2),
        "coords_amino": np.round(coords_amino, 2).tolist(),
        "coords_water": np.round(coords_water, 2).tolist()
    }

def run_100_residue_folding_simulation():
    sequence = generate_100_residue_sequence()
    s_steps = np.linspace(0.0, 1.0, 25)
    
    trajectory_data = []
    print("Starting 100-Residue Protein Folding Fiedler Analysis...")
    
    for idx, s in enumerate(s_steps):
        frame = compute_folding_frame(s, sequence)
        trajectory_data.append(frame)
        print(f"Frame {idx+1}/25 (s={s:.2f}): \lambda_2 = {frame['lambda_2_total']:.4f}, Rg = {frame['rg_gyration']:.2f}")
        
    out_file = "/home/eldenring/fiedler-colloid-lnp-sim/protein_100_folding_data.json"
    with open(out_file, "w") as f:
        json.dump({
            "sequence": sequence.tolist(),
            "trajectory": trajectory_data
        }, f, indent=2)
        
    print(f"Successfully saved 100-Residue Folding Trajectory at {out_file}")

if __name__ == "__main__":
    run_100_residue_folding_simulation()
