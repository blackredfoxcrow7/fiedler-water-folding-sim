import sys
sys.path.append(".")
sys.path.append("scratch")
import numpy as np
from server import PeptideSimulation
from download_pdb import parse_heavy_atoms_pdb
from test_mapping import align_rdkit_to_pdb

def kabsch_alignment(P, Q):
    """
    Aligns P to Q. Returns aligned P and RMSD.
    """
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    
    P_c = P - centroid_P
    Q_c = Q - centroid_Q
    
    H = np.dot(P_c.T, Q_c)
    U, S, Vt = np.linalg.svd(H)
    
    V = Vt.T
    d = np.linalg.det(np.dot(V, U.T))
    R = np.dot(V, np.dot(np.diag([1, 1, np.sign(d)]), U.T))
    
    P_aligned = np.dot(P_c, R.T)
    rmsd = np.sqrt(np.mean(np.sum((P_aligned - Q_c)**2, axis=1)))
    return P_aligned + centroid_Q, rmsd

def main():
    print("Loading reference PDB 1UAO.pdb...")
    pdb_coords, pdb_info = parse_heavy_atoms_pdb("1UAO.pdb")
    
    print("Initializing Peptide Simulation for Chignolin...")
    sim = PeptideSimulation("Gly-Tyr-Asp-Pro-Glu-Thr-Gly-Thr-Trp-Gly")
    
    print("Mapping coordinates...")
    pdb_mapped = align_rdkit_to_pdb(sim, pdb_coords, pdb_info)
    
    # Run simulation with optimized parameters
    sim.params["bulkSolventStrength"] = 0.35
    sim.params["waterBeltContraction"] = 0.50
    sim.params["waterWaterStrength"] = 0.50
    sim.params["waterPeptideStrength"] = 0.60
    sim.params["watersPerAtom"] = 4
    sim.initialize_waters()
    
    # Run 400 annealing steps
    print("Running folding simulation (400 steps)...")
    steps = 400
    dt = 0.02
    for step in range(steps):
        temp = 0.12 - 0.11 * (step / steps)
        sim.params["temperature"] = temp
        sim.step(dt)
        
    # Get atom names from RDKit monomer info
    atom_names = []
    for i in range(sim.n_atoms):
        atom = sim.mol_heavy.GetAtomWithIdx(i)
        info = atom.GetMonomerInfo()
        name = info.GetName().strip() if info else ""
        atom_names.append(name)
        
    # Create mask for different subsets
    all_indices = []
    backbone_c_indices = []
    ca_indices = []
    
    for idx in range(sim.n_atoms):
        # We only compare atoms that are successfully mapped to the PDB (non-zero reference coords)
        if np.linalg.norm(pdb_mapped[idx]) > 0.01:
            all_indices.append(idx)
            name = atom_names[idx]
            if name in ("CA", "C"):
                backbone_c_indices.append(idx)
            if name == "CA":
                ca_indices.append(idx)
                
    # 1. Compare All Heavy Atoms
    P_all = sim.coords[all_indices]
    Q_all = pdb_mapped[all_indices]
    _, rmsd_all = kabsch_alignment(P_all, Q_all)
    
    # 2. Compare Backbone Carbons (CA + C)
    P_bc = sim.coords[backbone_c_indices]
    Q_bc = pdb_mapped[backbone_c_indices]
    _, rmsd_bc = kabsch_alignment(P_bc, Q_bc)
    
    # 3. Compare Alpha Carbons (CA) Only
    P_ca = sim.coords[ca_indices]
    Q_ca = pdb_mapped[ca_indices]
    _, rmsd_ca = kabsch_alignment(P_ca, Q_ca)
    
    print("\n==============================================")
    print("RMSD COMPARISON RESULTS (vs. PDB 1UAO)")
    print("==============================================")
    print(f"All Heavy Atoms ({len(all_indices)} atoms):      {rmsd_all:.4f} Å")
    print(f"Backbone Carbons (CA + C) ({len(backbone_c_indices)} atoms): {rmsd_bc:.4f} Å")
    print(f"Alpha Carbons (CA) only ({len(ca_indices)} atoms):  {rmsd_ca:.4f} Å")
    print("==============================================")

if __name__ == "__main__":
    main()
