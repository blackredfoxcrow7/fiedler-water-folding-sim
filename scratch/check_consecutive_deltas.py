import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server_inference import PeptideInferenceSimulation
import numpy as np

def main():
    sim = PeptideInferenceSimulation("GYDPETGTWG")
    dt = 0.02
    
    for step in range(15):
        coords_before = sim.coords.copy()
        sim.step(dt)
        coords_after = sim.coords.copy()
        
        diff = np.max(np.abs(coords_after - coords_before))
        print(f"Step {step+1}: Max delta from previous step: {diff:.6f} Å")

if __name__ == "__main__":
    main()
