import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server_inference import PeptideInferenceSimulation

def test():
    sim = PeptideInferenceSimulation("GYDPETGTWG")
    sim.load_pdb_conformation("1UAO.pdb")
    sim.learning_mode = True
    sim.reverse = True
    sim.active_residue = sim.n_residues - 1
    sim.step_counter_residue = 0
    
    print("Step-by-step diagnostic of Residue 9 joints:")
    for step_idx in range(20):
        # We will manually perform parts of apply_rotations logic to inspect variables
        dt = 0.02
        sim.step(dt)
        
        # Access joints 34 and 35
        for j_idx in [34, 35]:
            joint = sim.peptide.joints[j_idx]
            u_idx = joint["u_idx"]
            d_idx = joint["d_idx"]
            D = joint["downstream_atoms"]
            axis = sim.peptide.coords[d_idx] - sim.peptide.coords[u_idx]
            axis_len = np.linalg.norm(axis)
            axis /= axis_len
            
            # Let's print the current angle
            angle = sim.peptide.joint_angles[j_idx]
            print(f"Step {step_idx:2d} | Joint {j_idx}: angle={angle:.4f}")

if __name__ == "__main__":
    test()
