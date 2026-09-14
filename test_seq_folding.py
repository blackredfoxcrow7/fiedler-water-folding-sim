import numpy as np
from server import PeptideSimulation

def calculate_angle(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_t = np.dot(v1, v2) / (n1 * n2)
    cos_t = np.clip(cos_t, -1.0, 1.0)
    return np.arccos(cos_t)

def run_seq_folding_test():
    print("=== Running Sequential Folding Conservation and Motion Test ===")
    sim = PeptideSimulation("Ala-Ala-Ala-Ala")
    print(f"Peptide loaded. Residues: {sim.residue_labels}")
    
    # Enable sequential folding
    sim.seq_folding["enabled"] = True
    sim.seq_folding["activeResidue"] = 1  # Residue 2
    sim.seq_folding["stepsPerResidue"] = 1000
    sim.params["explicitWaterEnabled"] = False
    sim.params["temperature"] = 1.5
    
    # Record initial bond lengths
    initial_bond_lengths = []
    for b in sim.bonds:
        s, t = b["source"], b["target"]
        length = np.linalg.norm(sim.coords[s] - sim.coords[t])
        initial_bond_lengths.append((s, t, length))
    
    # Record initial bond angles
    triplets = []
    for j in range(sim.n_atoms):
        neighbors = sim.adj[j]
        if len(neighbors) >= 2:
            for idx1 in range(len(neighbors)):
                for idx2 in range(idx1 + 1, len(neighbors)):
                    i = neighbors[idx1]
                    k = neighbors[idx2]
                    v1 = sim.coords[i] - sim.coords[j]
                    v2 = sim.coords[k] - sim.coords[j]
                    angle = calculate_angle(v1, v2)
                    triplets.append((i, j, k, angle))
                    
    # Record initial relative positions of residue 0 (frozen core) and residue 2, 3 (frozen tail)
    initial_coords = sim.coords.copy()
    
    # Run simulation steps
    steps = 300
    dt = 0.02
    for _ in range(steps):
        sim.step(dt)
        
    # Check bond lengths and angles after simulation
    max_length_error = 0.0
    for s, t, init_len in initial_bond_lengths:
        curr_len = np.linalg.norm(sim.coords[s] - sim.coords[t])
        error = abs(curr_len - init_len)
        max_length_error = max(max_length_error, error)
            
    max_angle_error = 0.0
    for i, j, k, init_angle in triplets:
        v1 = sim.coords[i] - sim.coords[j]
        v2 = sim.coords[k] - sim.coords[j]
        curr_angle = calculate_angle(v1, v2)
        error = abs(curr_angle - init_angle)
        max_angle_error = max(max_angle_error, error)
        
    # Check that relative coordinates within residue 0 (frozen core) did not change
    # i.e., distance between any two atoms in residue 0 remains unchanged
    res0_atoms = [i for i in range(sim.n_atoms) if sim.atom_residues[i] == 0]
    max_res0_internal_error = 0.0
    for idx1 in range(len(res0_atoms)):
        for idx2 in range(idx1 + 1, len(res0_atoms)):
            a1, a2 = res0_atoms[idx1], res0_atoms[idx2]
            init_dist = np.linalg.norm(initial_coords[a1] - initial_coords[a2])
            curr_dist = np.linalg.norm(sim.coords[a1] - sim.coords[a2])
            error = abs(curr_dist - init_dist)
            max_res0_internal_error = max(max_res0_internal_error, error)

    # Check that relative coordinates within residue 3 (frozen tail, > R+1) did not change
    tail_atoms = [i for i in range(sim.n_atoms) if sim.atom_residues[i] > 2]
    max_tail_internal_error = 0.0
    for idx1 in range(len(tail_atoms)):
        for idx2 in range(idx1 + 1, len(tail_atoms)):
            a1, a2 = tail_atoms[idx1], tail_atoms[idx2]
            init_dist = np.linalg.norm(initial_coords[a1] - initial_coords[a2])
            curr_dist = np.linalg.norm(sim.coords[a1] - sim.coords[a2])
            error = abs(curr_dist - init_dist)
            max_tail_internal_error = max(max_tail_internal_error, error)
            
    print("\n--- Sequential Folding Test Results ---")
    print(f"Max Bond Length Change:      {max_length_error:.2e} Å")
    print(f"Max Bond Angle Change:       {max_angle_error:.2e} rad")
    print(f"Max Core Internal Change:    {max_res0_internal_error:.2e} Å")
    print(f"Max Tail Internal Change:    {max_tail_internal_error:.2e} Å")
    
    assert max_length_error < 1e-10, f"Bond length not conserved! Max error: {max_length_error}"
    assert max_angle_error < 1e-10, f"Bond angle not conserved! Max error: {max_angle_error}"
    assert max_res0_internal_error < 1e-10, f"Core residue internal conformation changed! Max error: {max_res0_internal_error}"
    assert max_tail_internal_error < 1e-10, f"Tail residues internal conformation changed! Max error: {max_tail_internal_error}"
    
    # Make sure residue 1 (active residue) did actually rotate relative to residue 0
    res1_atoms = [i for i in range(sim.n_atoms) if sim.atom_residues[i] == 1]
    dist_change_detected = False
    for a0 in res0_atoms:
        for a1 in res1_atoms:
            init_dist = np.linalg.norm(initial_coords[a0] - initial_coords[a1])
            curr_dist = np.linalg.norm(sim.coords[a0] - sim.coords[a1])
            if abs(curr_dist - init_dist) > 1e-4:
                dist_change_detected = True
                break
        if dist_change_detected:
            break
            
    print(f"Active residue movement relative to core detected: {dist_change_detected}")
    assert dist_change_detected, "No relative movement detected for active residue! It might be frozen too."
    
    print("SUCCESS: Sequential folding constraints work perfectly!")
    print("========================================================\n")

def test_seq_folding():
    run_seq_folding_test()

if __name__ == '__main__':
    run_seq_folding_test()
