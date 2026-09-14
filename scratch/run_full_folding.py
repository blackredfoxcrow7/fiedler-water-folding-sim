import sys
import os
import numpy as np

# Ensure server module can be imported
sys.path.append("/home/eldenring/newProject")
from server_entropy import PeptideSimulation

def calculate_radius_of_gyration(sim):
    # Calculate Center of Mass
    com = np.mean(sim.coords, axis=0)
    # Sum of squared distances from COM
    sq_dist_sum = np.sum((sim.coords - com) ** 2)
    return np.sqrt(sq_dist_sum / sim.n_atoms)

def run_folding_simulation(sequence="Gly-Tyr-Asp-Pro-Glu-Thr-Gly-Thr-Trp-Gly", steps_per_residue=15):
    print("=" * 65)
    print(f"  Sequential Folding Simulation: {sequence}")
    print("=" * 65)
    
    sim = PeptideSimulation(sequence)
    sim.seq_folding["enabled"] = True
    sim.seq_folding["activeResidue"] = 0
    sim.seq_folding["stepsPerResidue"] = steps_per_residue
    sim.seq_folding["currentStep"] = 0
    sim.seq_folding["status"] = "folding"
    sim.seq_folding["direction"] = 1  # 1: N->C, -1: C->N
    
    # Enable explicit water belt
    sim.params["explicitWaterEnabled"] = True
    
    # We will log the progress at each residue transition
    history = []
    
    initial_rg = calculate_radius_of_gyration(sim)
    print(f"Initial State:")
    print(f"  Heavy Atoms: {sim.n_atoms}")
    print(f"  Residues: {sim.n_residues} ({', '.join(sim.residue_labels)})")
    print(f"  Initial Radius of Gyration: {initial_rg:.3f} Å")
    print(f"  Initial Hydrophobic Exposure: {sim.hydrophobic_exposure:.3f}")
    print(f"  Initial Information Capture Index: {sim.information_capture_index:.3f}")
    print("-" * 65)
    
    step_count = 0
    max_steps = 20000  # Safety limit
    
    dt = 0.03
    
    # Keep track of active residue transitions
    last_active = -1
    last_direction = 0
    
    while sim.seq_folding["status"] == "folding" and step_count < max_steps:
        # Step the physics
        sim.step(dt)
        step_count += 1
        
        active_res = sim.seq_folding["activeResidue"]
        direction = sim.seq_folding["direction"]
        
        # Calculate stats
        rg = calculate_radius_of_gyration(sim)
        hb_count = len(sim.active_hbonds)
        
        # Log when residue or direction changes
        if active_res != last_active or direction != last_direction:
            dir_str = "N->C" if direction == 1 else "C->N"
            print(f"Step {step_count:04d} | Residue {active_res+1} ({sim.residue_labels[active_res]}) | Direction: {dir_str}")
            print(f"  Radius of Gyration: {rg:.3f} Å")
            print(f"  H-Bonds: {hb_count}")
            print(f"  Info Capture Index (I_capture): {sim.information_capture_index:.3f}")
            print(f"  Hydrophobic Exposure: {sim.hydrophobic_exposure:.3f}")
            print("-" * 45)
            
            history.append({
                "step": step_count,
                "residue_idx": active_res,
                "residue_name": sim.residue_labels[active_res],
                "direction": dir_str,
                "radius_of_gyration": rg,
                "h_bonds": hb_count,
                "info_capture": sim.information_capture_index,
                "hydrophobic_exposure": sim.hydrophobic_exposure
            })
            
            last_active = active_res
            last_direction = direction

    # One final step to complete the completion state
    sim.step(dt)
    final_rg = calculate_radius_of_gyration(sim)
    final_hb = len(sim.active_hbonds)
    
    print("\n" + "=" * 65)
    print("  SIMULATION COMPLETED SUCCESSFULLY!")
    print("=" * 65)
    print(f"Total steps run: {step_count}")
    print(f"Final Radius of Gyration: {final_rg:.3f} Å (Compaction: {(initial_rg - final_rg)/initial_rg*100:.1f}%)")
    print(f"Final H-Bonds Formed: {final_hb}")
    print(f"Final Information Capture Index: {sim.information_capture_index:.3f}")
    print(f"Final Hydrophobic Exposure: {sim.hydrophobic_exposure:.3f}")
    print("-" * 65)
    
    print("\nFinal Coordinates of Peptide Atoms (first 5):")
    for i in range(min(5, sim.n_atoms)):
        atom = sim.atoms[i]
        coords = sim.coords[i]
        print(f"  Atom {i} ({atom['element']} in {atom['res_name']}{atom['res_num']}): "
              f"[{coords[0]:.4f}, {coords[1]:.4f}, {coords[2]:.4f}]")
              
    return history, sim

if __name__ == "__main__":
    run_folding_simulation()
