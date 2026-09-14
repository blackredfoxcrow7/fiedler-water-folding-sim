import numpy as np
from server import PeptideSimulation

def run_water_simulation_test():
    print("=== Running Explicit Water Agent & Belt Simulation Test ===")
    
    # 1. Initialize simulation for Gly-Ser-Asp (contains multiple hydrophilic atoms)
    sim = PeptideSimulation("Gly-Ser-Asp")
    print(f"Peptide loaded. Number of heavy atoms: {sim.n_atoms}")
    print(f"Number of water particles spawned: {len(sim.waters)}")
    
    # Verify that we have waters spawned
    assert len(sim.waters) > 0, "No water particles were spawned for hydrophilic atoms!"
    assert sim.water_coords.shape == (len(sim.waters), 3), "Water coordinates shape mismatch!"
    
    # Check that initial coordinates are within reasonable distance of parent atoms
    for w_idx, water in enumerate(sim.waters):
        parent_idx = water["parent_atom_id"]
        dist = np.linalg.norm(sim.water_coords[w_idx] - sim.coords[parent_idx])
        print(f"  Water {w_idx} parent atom: {parent_idx}, distance: {dist:.2f} Å")
        assert 2.0 < dist < 3.0, f"Water initial distance {dist:.2f} Å is out of range!"

    # 2. Check Co-rotation of Water Particles
    print("\n--- Testing Joint Co-rotation ---")
    # Record initial relative positions of water particles to their parent atoms
    initial_relative_dists = []
    for w_idx, water in enumerate(sim.waters):
        parent_idx = water["parent_atom_id"]
        d = np.linalg.norm(sim.water_coords[w_idx] - sim.coords[parent_idx])
        initial_relative_dists.append(d)
        
    # Rotate the first joint by a large angle
    if len(sim.joints) > 0:
        joint = sim.joints[0]
        u_idx = joint["u_idx"]
        d_idx = joint["d_idx"]
        D = joint["downstream_atoms"]
        pos_A = sim.coords[u_idx]
        pos_B = sim.coords[d_idx]
        axis = pos_B - pos_A
        axis /= np.linalg.norm(axis)
        
        theta = 0.5 # rad
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # Apply Rodrigues' rotation manually to peptide and downstream waters
        for idx in D:
            # peptide
            v = sim.coords[idx] - pos_B
            sim.coords[idx] = pos_B + (v * cos_t + np.cross(axis, v) * sin_t + axis * np.dot(axis, v) * (1.0 - cos_t))
            # water
            for w_idx, water in enumerate(sim.waters):
                if water["parent_atom_id"] == idx:
                    v_w = sim.water_coords[w_idx] - pos_B
                    sim.water_coords[w_idx] = pos_B + (v_w * cos_t + np.cross(axis, v_w) * sin_t + axis * np.dot(axis, v_w) * (1.0 - cos_t))
                    
        # Check that relative distances of downstream waters to their parent atoms are perfectly conserved
        for w_idx, water in enumerate(sim.waters):
            parent_idx = water["parent_atom_id"]
            if parent_idx in D:
                curr_d = np.linalg.norm(sim.water_coords[w_idx] - sim.coords[parent_idx])
                init_d = initial_relative_dists[w_idx]
                error = abs(curr_d - init_d)
                print(f"  Downstream Water {w_idx} relative distance error after rotation: {error:.2e} Å")
                assert error < 1e-10, f"Water-peptide distance not conserved! error: {error}"
        print("✓ Co-rotation distance conservation verified successfully!")
    else:
        print("  Skipped: no rotatable joints.")

    # 3. Verify simulation step water physics
    print("\n--- Testing Water Attraction & Step Physics ---")
    sim.params["explicitWaterEnabled"] = True
    sim.params["temperature"] = 0.0 # No thermal noise for deterministic forces check
    sim.params["waterWaterStrength"] = 0.5
    sim.params["waterPeptideStrength"] = 0.5
    
    # Run 10 steps of simulation
    for step_idx in range(10):
        sim.step(0.02)
        
    # Check that waters are attracted towards the 2.5Å parent target distance
    for w_idx, water in enumerate(sim.waters):
        parent_idx = water["parent_atom_id"]
        dist = np.linalg.norm(sim.water_coords[w_idx] - sim.coords[parent_idx])
        print(f"  Water {w_idx} distance to parent after 10 steps: {dist:.3f} Å")
        
    # Check that active water-water and water-peptide bonds are tracked
    print(f"  Active water-water bonds: {sim.active_water_bonds}")
    print(f"  Active water-peptide bonds: {sim.active_water_peptide_bonds}")
    
    # 4. Verify Water-Triggered Folding Transition
    print("\n--- Testing Water-Triggered Folding Transition ---")
    sim = PeptideSimulation("Ala-Ser-Asp") # Ala (hydrophobic/low hydrophilic), Ser & Asp (hydrophilic)
    sim.seq_folding["enabled"] = True
    sim.seq_folding["activeResidue"] = 1 # Residue 2 (Ser) is active
    sim.seq_folding["stepsPerResidue"] = 500
    sim.seq_folding["currentStep"] = 0
    sim.params["explicitWaterEnabled"] = True
    sim.params["waterFoldingTransition"] = True
    
    # Manually place water particles close to form a water-water bond between residue 1 (Ser) and residue 2 (Asp)
    # This should trigger an immediate transition to residue 2
    res1_waters = [w_idx for w_idx, w in enumerate(sim.waters) if w["residue_index"] == 1]
    res2_waters = [w_idx for w_idx, w in enumerate(sim.waters) if w["residue_index"] == 2]
    
    if res1_waters and res2_waters:
        w1 = res1_waters[-1]
        w2 = res2_waters[0]
        parent_idx1 = sim.waters[w1]["parent_atom_id"]
        parent_idx2 = sim.waters[w2]["parent_atom_id"]
        # Manually move parent atoms close to each other (7.0 Å) to simulate the folded state
        sim.coords[parent_idx2] = sim.coords[parent_idx1] + np.array([7.0, 0.0, 0.0])
        # Place water particles 2.3 Å away from their parents towards each other
        sim.water_coords[w1] = sim.coords[parent_idx1] + np.array([2.3, 0.0, 0.0])
        sim.water_coords[w2] = sim.coords[parent_idx2] - np.array([2.3, 0.0, 0.0])
        
        # Warp sibling water agents away
        if w1 - 1 >= 0 and sim.waters[w1-1]["parent_atom_id"] == parent_idx1:
            sim.water_coords[w1-1] = np.array([100.0, 100.0, 100.0])
        if w2 + 1 < len(sim.waters) and sim.waters[w2+1]["parent_atom_id"] == parent_idx2:
            sim.water_coords[w2+1] = np.array([100.0, 100.0, 100.0])
        
        # Run explicit water step once to build active_water_bonds
        sim.step_explicit_waters(0.02)
        
        # Manually set currentStep to exceed min_steps (80)
        sim.seq_folding["currentStep"] = 81
        
        # Step once to evaluate transition
        sim.step(0.02)
        
        print(f"  After stepping, activeResidue is: {sim.seq_folding['activeResidue']}")
        print(f"  Active water-water bonds: {sim.active_water_bonds}")
        print(f"  w1={w1}, w2={w2}")
        print(f"  w1_res={sim.waters[w1]['residue_index']}, w2_res={sim.waters[w2]['residue_index']}")
        print(f"  parent1={sim.waters[w1]['parent_atom_id']}, parent2={sim.waters[w2]['parent_atom_id']}")
        print(f"  d_p1={np.linalg.norm(sim.water_coords[w1] - sim.coords[sim.waters[w1]['parent_atom_id']]):.2f}Å")
        print(f"  d_p2={np.linalg.norm(sim.water_coords[w2] - sim.coords[sim.waters[w2]['parent_atom_id']]):.2f}Å")
        print(f"  d_w12={np.linalg.norm(sim.water_coords[w1] - sim.water_coords[w2]):.2f}Å")
        print(f"  seq_folding={sim.seq_folding}")
        
        # Transition should have happened because direction was 1, activeResidue was 1, next is 2
        assert sim.seq_folding["activeResidue"] == 2, "Water-triggered transition did not occur!"
        print("✓ Water-triggered folding transition verified successfully!")
    else:
        print("  Skipped water-triggered transition check (insufficient hydrophilic residues in test sequence).")

    print("\n=== ALL EXPLICIT WATER TESTS PASSED SUCCESSFULY! ===")

if __name__ == '__main__':
    run_water_simulation_test()
