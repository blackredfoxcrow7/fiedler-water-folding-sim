import numpy as np
from server import PeptideSimulation

def calculate_angle(v1, v2):
    # Calculate angle between two vectors in radians
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_t = np.dot(v1, v2) / (n1 * n2)
    cos_t = np.clip(cos_t, -1.0, 1.0)
    return np.arccos(cos_t)

def run_conservation_test():
    print("=== Running Torsion-space Dynamics Conservation Test ===")
    
    # 1. Initialize simulation for alanyl-alanyl-alanine (AAA)
    sim = PeptideSimulation("Ala-Ala-Ala")
    print(f"Peptide loaded. Number of heavy atoms: {sim.n_atoms}")
    print(f"Number of rotatable joints: {len(sim.joints)}")
    
    # 2. Record initial bond lengths
    initial_bond_lengths = []
    for b in sim.bonds:
        s, t = b["source"], b["target"]
        length = np.linalg.norm(sim.coords[s] - sim.coords[t])
        initial_bond_lengths.append((s, t, length))
    
    # 3. Record initial bond angles
    # An angle exists between bonds sharing a common atom.
    # We find all triplets (i, j, k) where i-j and j-k are bonds.
    triplets = []
    for j in range(sim.n_atoms):
        neighbors = sim.adj[j]
        if len(neighbors) >= 2:
            for idx1 in range(len(neighbors)):
                for idx2 in range(idx1 + 1, len(neighbors)):
                    i = neighbors[idx1]
                    k = neighbors[idx2]
                    # Compute angle i - j - k
                    v1 = sim.coords[i] - sim.coords[j]
                    v2 = sim.coords[k] - sim.coords[j]
                    angle = calculate_angle(v1, v2)
                    triplets.append((i, j, k, angle))
    
    print(f"Tracking {len(initial_bond_lengths)} bonds and {len(triplets)} bond angles.")
    
    # 4. Enable high solvent forces and temperature, run 100 simulation steps
    sim.params["temperature"] = 0.3
    sim.params["hydrophobicStrength"] = 0.8
    sim.params["hBondStrength"] = 0.8
    
    steps = 100
    dt = 0.02
    for s_idx in range(steps):
        # Run 5 micro-steps per step, just like the server stream
        for _ in range(5):
            sim.step(dt)
            
    # 5. Check bond lengths and angles after simulation
    max_length_error = 0.0
    for s, t, init_len in initial_bond_lengths:
        curr_len = np.linalg.norm(sim.coords[s] - sim.coords[t])
        error = abs(curr_len - init_len)
        if error > max_length_error:
            max_length_error = error
            
    max_angle_error = 0.0
    for i, j, k, init_angle in triplets:
        v1 = sim.coords[i] - sim.coords[j]
        v2 = sim.coords[k] - sim.coords[j]
        curr_angle = calculate_angle(v1, v2)
        error = abs(curr_angle - init_angle)
        if error > max_angle_error:
            max_angle_error = error
            
    print("\n--- Test Results ---")
    print(f"Max Bond Length Change: {max_length_error:.2e} Å")
    print(f"Max Bond Angle Change:  {max_angle_error:.2e} rad")
    
    # Assertions: Errors should be extremely tiny (due to float precision limit, < 1e-10)
    assert max_length_error < 1e-10, f"Bond length not conserved! Max error: {max_length_error}"
    assert max_angle_error < 1e-10, f"Bond angle not conserved! Max error: {max_angle_error}"
    
    print("SUCCESS: Bond lengths and angles are perfectly conserved under torsion-space updates!")
    print("========================================================\n")

if __name__ == '__main__':
    run_conservation_test()
