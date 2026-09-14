import sys
sys.path.append(".")
import numpy as np
from server import PeptideSimulation
from download_pdb import parse_heavy_atoms_pdb
from test_mapping import align_rdkit_to_pdb

def kabsch_alignment(P, Q):
    """
    Aligns P to Q. Returns aligned P and RMSD.
    P, Q are shape (N, 3)
    """
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    
    P_c = P - centroid_P
    Q_c = Q - centroid_Q
    
    H = np.dot(P_c.T, Q_c)
    U, S, Vt = np.linalg.svd(H)
    
    V = Vt.T
    d = np.linalg.det(np.dot(V, U.T))
    R = np.dot(V, np.dot(np.diag([1, 1, np.sign(d)]), U.T))
    
    P_aligned = np.dot(P_c, R.T)
    rmsd = np.sqrt(np.mean(np.sum((P_aligned - Q_c)**2, axis=1)))
    return P_aligned + centroid_Q, rmsd

def run_optimization():
    # Load reference structure (Chignolin)
    print("Loading reference structure...")
    pdb_coords, pdb_info = parse_heavy_atoms_pdb("1UAO.pdb")
    
    # Initialize simulation
    print("Initializing simulation...")
    sim_init = PeptideSimulation("Gly-Tyr-Asp-Pro-Glu-Thr-Gly-Thr-Trp-Gly")
    
    # Align and map coordinates
    pdb_mapped = align_rdkit_to_pdb(sim_init, pdb_coords, pdb_info)
    
    # Grid search parameters
    # Let's search K_solv, K_contract, K_ww, K_wp
    k_solv_list = [0.0, 0.15, 0.35]
    k_contract_list = [0.0, 0.25, 0.50]
    k_ww_list = [0.25, 0.50]
    k_wp_list = [0.30, 0.60]
    
    best_rmsd = float('inf')
    best_params = None
    
    # Run grid search
    total_runs = len(k_solv_list) * len(k_contract_list) * len(k_ww_list) * len(k_wp_list)
    print(f"Starting Grid Search (Total {total_runs} runs)...")
    
    run_idx = 0
    for k_solv in k_solv_list:
        for k_contract in k_contract_list:
            for k_ww in k_ww_list:
                for k_wp in k_wp_list:
                    run_idx += 1
                    
                    # Create simulation instance
                    sim = PeptideSimulation("Gly-Tyr-Asp-Pro-Glu-Thr-Gly-Thr-Trp-Gly")
                    sim.params["bulkSolventStrength"] = k_solv
                    sim.params["waterBeltContraction"] = k_contract
                    sim.params["waterWaterStrength"] = k_ww
                    sim.params["waterPeptideStrength"] = k_wp
                    sim.params["watersPerAtom"] = 3 # Fixed at 3 waters per hydrophilic atom
                    sim.initialize_waters()
                    
                    # Run simulation with annealing temperature schedule
                    # 400 steps of folding simulation
                    steps = 400
                    dt = 0.02
                    for step in range(steps):
                        # Temperature annealing: start high (0.12) to cross barriers, cool to 0.01
                        temp = 0.12 - 0.11 * (step / steps)
                        sim.params["temperature"] = temp
                        sim.step(dt)
                        
                    # Calculate aligned RMSD (only for heavy atoms of the peptide)
                    _, rmsd = kabsch_alignment(sim.coords, pdb_mapped)
                    
                    if rmsd < best_rmsd:
                        best_rmsd = rmsd
                        best_params = {
                            "bulkSolventStrength": k_solv,
                            "waterBeltContraction": k_contract,
                            "waterWaterStrength": k_ww,
                            "waterPeptideStrength": k_wp,
                            "watersPerAtom": 3
                        }
                        print(f"  [Run {run_idx}/{total_runs}] New Best RMSD: {best_rmsd:.3f} Å (K_solv={k_solv}, K_contract={k_contract}, K_ww={k_ww}, K_wp={k_wp})")
                    
    print("\n=======================================================")
    print("OPTIMIZATION COMPLETED!")
    print(f"Best Parameters: {best_params}")
    print(f"Minimum RMSD reached: {best_rmsd:.3f} Å")
    print("=======================================================")
    
    # Save the best parameters to a JSON file
    import json
    with open("optimized_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

if __name__ == "__main__":
    run_optimization()
