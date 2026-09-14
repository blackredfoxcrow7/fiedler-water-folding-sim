import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server_inference import PeptideInferenceSimulation

def test():
    sim = PeptideInferenceSimulation("GYDPETGTWG")
    has_pdb = sim.load_pdb_conformation("1UAO.pdb")
    print(f"PDB Loaded: {has_pdb}")
    
    print(f"Number of joints: {len(sim.peptide.joints)}")
    print("Joint angles (first 10):", sim.peptide.joint_angles[:10])
    print("Sum of absolute joint angles:", np.sum(np.abs(sim.peptide.joint_angles)))
    
    # Check if there are any joints for residue 9
    res9_joints = [j_idx for j_idx, j in enumerate(sim.peptide.joints) 
                   if sim.peptide.atom_residues.get(j["d_idx"]) == 9]
    print(f"Residue 9 joints: {res9_joints}")
    if res9_joints:
        for j_idx in res9_joints:
            print(f"Joint {j_idx}: u_idx={sim.peptide.joints[j_idx]['u_idx']}, d_idx={sim.peptide.joints[j_idx]['d_idx']}, angle={sim.peptide.joint_angles[j_idx]:.4f}")

if __name__ == "__main__":
    test()
