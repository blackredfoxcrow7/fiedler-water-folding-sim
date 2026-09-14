import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server_inference import PeptideInferenceSimulation
import numpy as np

def main():
    print("Initializing simulation...")
    sim = PeptideInferenceSimulation("GYDPETGTWG")
    
    initial_coords = sim.coords.copy()
    print("First atom initial position:", initial_coords[0])
    
    # Run 200 steps
    dt = 0.02
    for step in range(200):
        sim.step(dt)
        
    final_coords = sim.coords.copy()
    print("First atom final position:  ", final_coords[0])
    
    diff = np.max(np.abs(final_coords - initial_coords))
    print(f"Max absolute change in coordinates: {diff:.6f} Å")
    
    if diff < 1e-5:
        print("❌ Warning: Molecule coordinates did not change!")
    else:
        print("✓ Success: Molecule coordinates are changing.")

if __name__ == "__main__":
    main()
