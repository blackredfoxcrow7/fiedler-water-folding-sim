import sys
sys.path.append(".")
import numpy as np
import json
from server import PeptideSimulation
from download_pdb import parse_heavy_atoms_pdb
from test_mapping import align_rdkit_to_pdb

def kabsch_alignment(P, Q):
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

def main():
    print("Loading PDB structure 1UAO.pdb...")
    pdb_coords, pdb_info = parse_heavy_atoms_pdb("1UAO.pdb")
    
    print("Initializing mapping...")
    sim_init = PeptideSimulation("Gly-Tyr-Asp-Pro-Glu-Thr-Gly-Thr-Trp-Gly")
    pdb_mapped = align_rdkit_to_pdb(sim_init, pdb_coords, pdb_info)
    
    candidate_sets = [
        {
            "name": "Default Unoptimized",
            "bulkSolventStrength": 0.0,
            "waterBeltContraction": 0.0,
            "waterWaterStrength": 0.35,
            "waterPeptideStrength": 0.40,
            "watersPerAtom": 2
        },
        {
            "name": "Optimized (3 Waters)",
            "bulkSolventStrength": 0.35,
            "waterBeltContraction": 0.50,
            "waterWaterStrength": 0.50,
            "waterPeptideStrength": 0.60,
            "watersPerAtom": 3
        },
        {
            "name": "Optimized (4 Waters - Dense)",
            "bulkSolventStrength": 0.35,
            "waterBeltContraction": 0.50,
            "waterWaterStrength": 0.50,
            "waterPeptideStrength": 0.60,
            "watersPerAtom": 4
        },
        {
            "name": "High Contraction (3 Waters)",
            "bulkSolventStrength": 0.45,
            "waterBeltContraction": 0.60,
            "waterWaterStrength": 0.60,
            "waterPeptideStrength": 0.70,
            "watersPerAtom": 3
        }
    ]
    
    results = []
    
    for idx, params in enumerate(candidate_sets):
        print(f"\n--- Running evaluation for: {params['name']} ---")
        sim = PeptideSimulation("Gly-Tyr-Asp-Pro-Glu-Thr-Gly-Thr-Trp-Gly")
        sim.params["bulkSolventStrength"] = params["bulkSolventStrength"]
        sim.params["waterBeltContraction"] = params["waterBeltContraction"]
        sim.params["waterWaterStrength"] = params["waterWaterStrength"]
        sim.params["waterPeptideStrength"] = params["waterPeptideStrength"]
        sim.params["watersPerAtom"] = params["watersPerAtom"]
        sim.initialize_waters()
        
        # Run 250 steps with cooling schedule
        steps = 250
        dt = 0.02
        for step in range(steps):
            temp = 0.12 - 0.11 * (step / steps)
            sim.params["temperature"] = temp
            sim.step(dt)
            
        _, rmsd = kabsch_alignment(sim.coords, pdb_mapped)
        print(f"Resulting RMSD: {rmsd:.4f} Å")
        
        results.append({
            "name": params["name"],
            "rmsd": rmsd,
            "params": params
        })
        
    print("\n==============================================")
    print("COMPARATIVE EVALUATION RESULTS:")
    print("==============================================")
    best_set = None
    best_rmsd = float('inf')
    for res in results:
        print(f"{res['name']}: RMSD = {res['rmsd']:.4f} Å")
        if res['rmsd'] < best_rmsd:
            best_rmsd = res['rmsd']
            best_set = res
            
    print("----------------------------------------------")
    print(f"Optimal Parameter Set Found: {best_set['name']}")
    print(f"Optimal Parameters: {best_set['params']}")
    print(f"Minimum RMSD: {best_set['rmsd']:.4f} Å")
    print("==============================================")
    
    # Save the absolute best to optimized_params.json
    output_params = {k: v for k, v in best_set["params"].items() if k != "name"}
    with open("optimized_params.json", "w") as f:
        json.dump(output_params, f, indent=4)
        
if __name__ == "__main__":
    main()
