import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server_inference import PeptideInferenceSimulation

def test():
    print("Initializing PeptideInferenceSimulation...")
    sim = PeptideInferenceSimulation("GYDPETGTWG")
    
    # Save initial coordinates
    init_coords = sim.coords.copy()
    
    print("Running 50 simulation steps...")
    for step_idx in range(50):
        sim.step(0.02)
        
    final_coords = sim.coords
    
    # Calculate coordinate changes
    diff = np.linalg.norm(final_coords - init_coords, axis=1)
    print("Coordinate changes per atom (min, mean, max):")
    print(f"Min:  {np.min(diff):.6f} A")
    print(f"Mean: {np.mean(diff):.6f} A")
    print(f"Max:  {np.max(diff):.6f} A")
    
    if np.mean(diff) < 1e-4:
        print("WARNING: Coordinates are NOT changing! The peptide is static.")
    else:
        print("SUCCESS: Coordinates are changing successfully.")

if __name__ == "__main__":
    test()
