import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server_inference import PeptideInferenceSimulation
import numpy as np

def main():
    print("Initializing simulation...")
    sim = PeptideInferenceSimulation("GYDPETGTWG")
    
    initial_coords = sim.coords.copy()
    print("First atom initial position:", initial_coords[0])
    
    # Run 20 steps and measure time per step
    dt = 0.02
    for step in range(20):
        t0 = time.time()
        sim.step(dt)
        t1 = time.time()
        print(f"Step {step+1}/20: {t1 - t0:.4f} seconds. Max coordinate delta: {np.max(np.abs(sim.coords - initial_coords)):.6f} Å")
        
    final_coords = sim.coords.copy()
    diff = np.max(np.abs(final_coords - initial_coords))
    print(f"\nFinal max absolute change in coordinates after 20 steps: {diff:.6f} Å")

if __name__ == "__main__":
    main()
