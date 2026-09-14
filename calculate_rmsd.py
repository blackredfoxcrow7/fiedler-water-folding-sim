import numpy as np
import json
import os
from differentiable_folding_directional import PeptideAgent

def get_pdb_ca_coords(pdb_path, target_chain=None):
    coords = []
    current_model = 1
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("MODEL"):
                parts = line.split()
                if len(parts) > 1:
                    current_model = int(parts[1])
                if current_model > 1:
                    break
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_name = line[12:16].strip()
                chain = line[21]
                if atom_name == "CA":
                    if target_chain is None or chain == target_chain or chain.strip() == "":
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coords.append([x, y, z])
    return np.array(coords)

def get_json_ca_coords(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    sequence = data["sequence"]
    coords = np.array(data["foldedCoords"])
    
    peptide = PeptideAgent(sequence)
    ca_indices = []
    for idx in range(peptide.n_atoms):
        res_info = peptide.mol.GetAtomWithIdx(idx).GetPDBResidueInfo()
        if res_info is not None and res_info.GetName().strip() == "CA":
            ca_indices.append(idx)
    return coords[ca_indices]

def kabsch_rmsd(P, Q):
    # Centering
    P_centered = P - np.mean(P, axis=0)
    Q_centered = Q - np.mean(Q, axis=0)
    
    # Covariance matrix
    C = np.dot(P_centered.T, Q_centered)
    
    # SVD
    V, S, Wt = np.linalg.svd(C)
    
    # Reflection check
    d = (np.linalg.det(V) * np.linalg.det(Wt)) < 0.0
    if d:
        V[:, -1] = -V[:, -1]
        
    R = np.dot(V, Wt)
    P_rotated = np.dot(P_centered, R)
    
    # RMSD
    diff = P_rotated - Q_centered
    return np.sqrt(np.mean(np.sum(diff**2, axis=-1)))

def generate_ideal_alpha_helix_ca(length):
    # Generates C-alpha coordinates of an ideal alpha helix along Z axis
    # Rise per residue: 1.5 A, Rotation per residue: 100 degrees (1.745 rad), Radius: 2.3 A
    coords = []
    for i in range(length):
        angle = i * 1.74533
        x = 2.3 * np.cos(angle)
        y = 2.3 * np.sin(angle)
        z = i * 1.5
        coords.append([x, y, z])
    return np.array(coords)

def run_validation():
    print("==================================================")
    print("  KABSCH C-ALPHA RMSD VALIDATION REPORT")
    print("  Comparing Folded Structures to PDB References")
    print("==================================================")
    
    # 1. Deca-alanine vs Ideal Alpha Helix
    json_ala = "aaaaaaaaaa_hybrid_folded.json"
    if os.path.exists(json_ala):
        folded_ca = get_json_ca_coords(json_ala)
        ideal_ca = generate_ideal_alpha_helix_ca(10)
        rmsd = kabsch_rmsd(folded_ca, ideal_ca)
        print(f" Deca-alanine (AAAAAAAAAA) vs Ideal Alpha-Helix:")
        print(f"   -> C-alpha RMSD: {rmsd:.3f} A")
    
    # 2. CLN025 vs PDB 5AWL
    json_cln = "yydpetgtwy_hybrid_folded.json"
    pdb_cln = "5awl.pdb"
    if os.path.exists(json_cln) and os.path.exists(pdb_cln):
        folded_ca = get_json_ca_coords(json_cln)
        pdb_ca = get_pdb_ca_coords(pdb_cln, target_chain="A")
        # Verify sizes
        if len(folded_ca) == len(pdb_ca):
            rmsd = kabsch_rmsd(folded_ca, pdb_ca)
            print(f" Chignolin Mutant (CLN025) vs PDB 5AWL (Model 1 / Chain A):")
            print(f"   -> C-alpha RMSD: {rmsd:.3f} A")
        else:
            print(f" Error: size mismatch for CLN025 ({len(folded_ca)} vs {len(pdb_ca)})")
            
    # 3. 1UAO vs PDB 1UAO
    json_1uao = "gydpetgtwg_hybrid_folded.json"
    pdb_1uao = "1uao.pdb"
    if os.path.exists(json_1uao) and os.path.exists(pdb_1uao):
        folded_ca = get_json_ca_coords(json_1uao)
        pdb_ca = get_pdb_ca_coords(pdb_1uao)
        if len(folded_ca) == len(pdb_ca):
            rmsd = kabsch_rmsd(folded_ca, pdb_ca)
            print(f" Wild-type Chignolin (1UAO) vs PDB 1UAO (Model 1):")
            print(f"   -> C-alpha RMSD: {rmsd:.3f} A")
        else:
            print(f" Error: size mismatch for 1UAO ({len(folded_ca)} vs {len(pdb_ca)})")
            
    # 4. Trp-cage vs PDB 1L2Y
    json_trpcage = "nlyiqwlkdggpssgrppps_hybrid_folded.json"
    pdb_trpcage = "1l2y.pdb"
    if os.path.exists(json_trpcage) and os.path.exists(pdb_trpcage):
        folded_ca = get_json_ca_coords(json_trpcage)
        pdb_ca = get_pdb_ca_coords(pdb_trpcage)
        if len(folded_ca) == len(pdb_ca):
            rmsd = kabsch_rmsd(folded_ca, pdb_ca)
            print(f" Trp-cage (1L2Y) vs PDB 1L2Y (Model 1):")
            print(f"   -> C-alpha RMSD: {rmsd:.3f} A")
        else:
            print(f" Error: size mismatch for Trp-cage ({len(folded_ca)} vs {len(pdb_ca)})")
    print("==================================================")

if __name__ == "__main__":
    run_validation()
