import sys
sys.path.append(".")
sys.path.append("scratch")
from server_entropy import PeptideSimulation
import numpy as np

def test_entropy_physics():
    print("Initializing Peptide Simulation for Chignolin (Gly-Tyr-Asp-Pro-Glu-Thr-Gly-Thr-Trp-Gly)...")
    # Chignolin has both hydrophobic (Tyr, Pro, Trp) and hydrophilic residues (Asp, Glu, Thr)
    sim = PeptideSimulation("Gly-Tyr-Asp-Pro-Glu-Thr-Gly-Thr-Trp-Gly")
    
    print("Initial waters spawned:", len(sim.waters))
    assert len(sim.waters) > 0, "No waters initialized!"
    
    # Run a few steps to let coordinates settle
    dt = 0.02
    print("Running initial simulation step...")
    sim.step(dt)
    
    # Assertions on entropy values
    entropies = sim.water_entropies
    weights = sim.water_weights
    
    assert len(entropies) == len(sim.waters), "Entropy size mismatch!"
    assert len(weights) == len(sim.waters), "Weight size mismatch!"
    
    # Verify that hydrophilic waters have lower entropy and higher weight
    # and hydrophobic waters have higher entropy and lower weight
    polar_indices = [idx for idx in range(sim.n_atoms) if sim.atoms[idx]["h_bond"] in ("donor", "acceptor", "both")]
    hydrophobic_indices = [idx for idx in range(sim.n_atoms) if sim.atoms[idx]["hydrophobicity"] > 0.15]
    
    low_entropy_count = 0
    high_entropy_count = 0
    
    for i in range(len(sim.waters)):
        pos_o = sim.water_coords[i, 0]
        d_polar = min([np.linalg.norm(pos_o - sim.coords[p]) for p in polar_indices])
        d_hydro = min([np.linalg.norm(pos_o - sim.coords[h]) for h in hydrophobic_indices])
        
        S = entropies[i]
        W = weights[i]
        
        # If very close to polar atom, entropy should be low, weight should be high
        if d_polar < 3.3 and d_hydro > 4.5:
            assert S < 0.6, f"Expected low entropy near polar group, got S={S:.3f}"
            assert W > 1.0, f"Expected high weight near polar group, got W={W:.3f}"
            low_entropy_count += 1
            
        # If very close to hydrophobic atom and far from polar atom, entropy should be high, weight should be low
        if d_hydro < 3.3 and d_polar > 4.5:
            assert S > 0.6, f"Expected high entropy near hydrophobic group, got S={S:.3f}"
            assert W < 1.0, f"Expected low weight near hydrophobic group, got W={W:.3f}"
            high_entropy_count += 1
            
    print(f"Verified hydrophilic waters (low entropy count: {low_entropy_count})")
    print(f"Verified hydrophobic waters (high entropy count: {high_entropy_count})")
    
    # Verify Dijkstra solver and I_capture
    print(f"Current Information Capture Index: {sim.information_capture_index:.4f}")
    assert sim.information_capture_index >= 0.0, "Invalid negative I_capture!"
    print(f"Active Information Paths count: {len(sim.active_information_paths)}")
    
    # Run multiple steps to verify stability
    print("Running 50 stability steps...")
    for _ in range(50):
        sim.step(dt)
        
    print("All entropy & information capture assertions passed successfully!")

if __name__ == "__main__":
    test_entropy_physics()
