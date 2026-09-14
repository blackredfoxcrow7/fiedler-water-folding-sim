import sys
import os
import numpy as np
import random
import networkx as nx
from flask import Flask, jsonify, request
from rdkit import Chem

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from peptide_agent import PeptideAgent

app = Flask(__name__)

# CORS headers middleware
@app.after_request
def add_cors_headers(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

def center_coordinates(coords):
    """Centers coordinates around the center of mass."""
    com = np.mean(coords, axis=0)
    return (coords - com).tolist()

def center_coordinates_np(coords):
    """Centers coordinates around the center of mass (returns numpy array)."""
    com = np.mean(coords, axis=0)
    return coords - com

def generate_extended_coords(peptide):
    """
    Generates a truly extended straight-chain conformation.
    Aligns each residue's C-alpha atom along the x-axis spaced by 3.8 Å,
    preserving the local internal coordinates (geometry) of each residue.
    """
    n_atoms = peptide.n_atoms
    initial_coords = peptide.initial_coords
    new_coords = np.zeros_like(initial_coords)
    
    # Find the C-alpha (CA) atom index for each residue
    ca_indices = {}
    for i in range(n_atoms):
        res_idx = peptide.atom_residues[i]
        atom_name = peptide.atoms[i]["name"]
        if atom_name == "CA" or (peptide.atoms[i]["element"] == "C" and "CA" in atom_name):
            ca_indices[res_idx] = i
            
    # Fallback if CA is not found for a residue
    for r in range(peptide.n_residues):
        if r not in ca_indices:
            res_atoms = [i for i in range(n_atoms) if peptide.atom_residues[i] == r]
            ca_indices[r] = res_atoms[0] if res_atoms else 0
            
    # Place residues sequentially along the x-axis
    for i in range(n_atoms):
        res_idx = peptide.atom_residues[i]
        ca_idx = ca_indices[res_idx]
        # Calculate local offset relative to its residue's C-alpha
        offset = initial_coords[i] - initial_coords[ca_idx]
        # Place C-alpha at x = 3.8 * res_idx, y=0, z=0
        new_coords[i] = [3.8 * res_idx + offset[0], offset[1], offset[2]]
        
    return new_coords

def generate_sphere_points(n):
    """Generates n points on a sphere using the Fibonacci spiral algorithm."""
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle in radians
    for i in range(n):
        y = 1.0 - (i / float(n - 1)) * 2.0  # y goes from 1 to -1
        radius = np.sqrt(1.0 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)

def solvate_peptide(peptide_coords, atoms, num_waters=60):
    """
    Highly optimized vector-based solvent coordinate placer.
    - Polar atoms (N, O, S) coordinate waters at a hydrogen-bonding distance of 2.8 Å.
    - Non-polar carbons (C) coordinate waters at a van der Waals distance of 3.5 Å.
    - Enforces a steric clearance of 2.3 Å from the peptide and 2.6 Å between waters.
    """
    peptide_arr = np.array(peptide_coords)
    n_atoms = len(atoms)
    
    # Pre-calculate radii for polar (2.8) and non-polar (3.5)
    radii = np.array([2.8 if a["element"] in ("N", "O", "S") else 3.5 for a in atoms])
    
    # Generate 8 sphere points (enough to cover spatial diversity and very fast)
    sphere_points = np.array([
        [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.577, 0.577, 0.577], [-0.577, -0.577, -0.577]
    ])
    
    # Calculate all candidate points: shape (n_atoms, 8, 3)
    cands = peptide_arr[:, None, :] + radii[:, None, None] * sphere_points[None, :, :]
    cands = cands.reshape(-1, 3)  # shape (M, 3)
    
    # Compute distances between all candidates and all peptide atoms: shape (M, n_atoms)
    dists = np.linalg.norm(cands[:, None, :] - peptide_arr[None, :, :], axis=-1)
    min_dists = np.min(dists, axis=1)
    
    # Filter candidates with clearance >= 2.3
    valid_mask = min_dists >= 2.3
    valid_cands = cands[valid_mask]
    valid_min_dists = min_dists[valid_mask]
    
    # Sort closest to surface first
    sort_idx = np.argsort(valid_min_dists)
    sorted_cands = valid_cands[sort_idx]
    
    # Greedy selection of num_waters with 2.6 clearance
    selected = []
    for cand in sorted_cands:
        if len(selected) >= num_waters:
            break
        if len(selected) > 0:
            selected_arr = np.array(selected)
            dists_to_selected = np.linalg.norm(selected_arr - cand, axis=-1)
            if np.any(dists_to_selected < 2.6):
                continue
        selected.append(cand)
        
    # Fallback padding if necessary
    while len(selected) < num_waters:
        com = np.mean(peptide_coords, axis=0)
        random_dir = np.random.normal(size=3)
        random_dir_norm = np.linalg.norm(random_dir)
        if random_dir_norm > 1e-5:
            random_dir /= random_dir_norm
        pos = com + (10.0 + len(selected) * 0.2) * random_dir
        selected.append(pos.tolist())
        
    return selected

def rotate_joint(coords, joint, d_theta):
    """Rotates all downstream atoms of a joint around its bond axis by d_theta."""
    new_coords = coords.copy()
    u_idx = joint["u_idx"]
    d_idx = joint["d_idx"]
    D = joint["downstream_atoms"]
    
    pos_A = coords[u_idx]
    pos_B = coords[d_idx]
    
    axis = pos_B - pos_A
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-5:
        return new_coords
    axis /= axis_len
    
    cos_t = np.cos(d_theta)
    sin_t = np.sin(d_theta)
    
    r_vec = coords[D] - pos_B
    cross_axis_r = np.cross(axis, r_vec)
    dot_axis_r = r_vec @ axis
    
    rotated = r_vec * cos_t + cross_axis_r * sin_t + axis[np.newaxis, :] * (dot_axis_r * (1.0 - cos_t))[:, np.newaxis]
    new_coords[D] = pos_B + rotated
    return new_coords

def randomize_dihedrals(coords, joints):
    """Rotates all joints by moderate random angles to explore diverse pathways without completely dispersing the chain."""
    curr_coords = coords.copy()
    for joint in joints:
        d_theta = random.uniform(-0.5, 0.5)
        curr_coords = rotate_joint(curr_coords, joint, d_theta)
    return curr_coords

def build_peptide_network(coords, bonds, contact_threshold=4.5):
    """Builds the peptide atom network including covalent bonds and spatial contacts."""
    G = nx.Graph()
    n_atoms = len(coords)
    G.add_nodes_from(range(n_atoms))
    
    # Add covalent bonds
    for bond in bonds:
        G.add_edge(bond["source"], bond["target"])
        
    # Add spatial contacts
    coords_arr = np.array(coords)
    dists = np.linalg.norm(coords_arr[:, None, :] - coords_arr[None, :, :], axis=-1)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if dists[i, j] < contact_threshold:
                G.add_edge(i, j)
    return G

def build_water_network(water_coords, contact_threshold=3.5):
    """Builds the pure water network (oxygen-oxygen hydrogen bonding)."""
    G = nx.Graph()
    n_waters = len(water_coords)
    G.add_nodes_from(range(n_waters))
    
    water_arr = np.array(water_coords)
    dists = np.linalg.norm(water_arr[:, None, :] - water_arr[None, :, :], axis=-1)
    for i in range(n_waters):
        for j in range(i + 1, n_waters):
            if dists[i, j] < contact_threshold:
                G.add_edge(i, j)
    return G

def get_network_metrics(G):
    """Calculates Fiedler connectivity and spectral network entropy using fast numpy eigvalsh."""
    try:
        if G.number_of_nodes() <= 1:
            return 0.0, 0.0
        L = nx.laplacian_matrix(G).toarray()
        eigenvals = np.linalg.eigvalsh(L)
        fiedler = float(eigenvals[1])
        
        # Spectral entropy
        nz_eigenvals = eigenvals[eigenvals > 1e-5]
        if len(nz_eigenvals) > 0:
            probs = nz_eigenvals / np.sum(nz_eigenvals)
            spectral_entropy = float(-np.sum(probs * np.log(probs)))
        else:
            spectral_entropy = 0.0
    except Exception:
        fiedler = 0.0
        spectral_entropy = 0.0
        
    return fiedler, spectral_entropy

@app.route('/api/structure', methods=['GET', 'POST'])
def run_simulation_and_get_structure():
    try:
        sequence = "GYDPETGTWG"
        if request.is_json:
            req_data = request.json or {}
            sequence = req_data.get("sequence", "GYDPETGTWG")
        else:
            sequence = request.args.get("sequence", "GYDPETGTWG")
            
        print(f"\n========================================================")
        print(f"  Starting Solvent-Coupled Fiedler Folding Simulation")
        print(f"  Sequence: {sequence}")
        print(f"========================================================")
        
        # 1. Initialize PeptideAgent for unfolded coordinates & topology
        peptide = PeptideAgent(sequence)
        
        # Generate a truly extended conformation for display
        extended_coords = generate_extended_coords(peptide)
        unfolded_coords = center_coordinates_np(extended_coords)
        
        # Maintain the original RDKit conformer coordinates for fallback
        rdkit_start_coords = center_coordinates_np(peptide.initial_coords)
        
        # 2. Get target folded Fiedler connectivity from validation PDB if available
        ideal_fiedler = 0.158524  # default target
        ideal_spectral_entropy = 1.95 # default target
        has_folded_target = False
        
        pdb_map = {
            "GYDPETGTWG": "1UAO.pdb",
            "YYDPETGTWY": "2H2D.pdb"
        }
        seq_upper = sequence.strip().upper()
        
        if seq_upper in pdb_map:
            pdb_path = pdb_map[seq_upper]
            if os.path.exists(pdb_path):
                try:
                    pdb_mol = Chem.MolFromPDBFile(pdb_path, removeHs=True)
                    if pdb_mol:
                        pdb_conf = pdb_mol.GetConformer()
                        raw_folded = np.zeros_like(peptide.initial_coords)
                        for i in range(min(peptide.n_atoms, pdb_mol.GetNumAtoms())):
                            pos = pdb_conf.GetAtomPosition(i)
                            raw_folded[i] = [pos.x, pos.y, pos.z]
                        folded_target = center_coordinates_np(raw_folded)
                        target_G = build_peptide_network(folded_target, peptide.bonds)
                        ideal_fiedler, ideal_spectral_entropy = get_network_metrics(target_G)
                        has_folded_target = True
                except Exception as e:
                    print(f"Error reading PDB file {pdb_path}: {e}")
                    
        acceptance_threshold = 0.70 * ideal_fiedler  # lower threshold to speed up convergence
        print(f"Target Ideal Fiedler: {ideal_fiedler:.6f} | Convergence Threshold: {acceptance_threshold:.6f}")
        
        # 3. Optimization Loop with Multi-Trial Restarts (up to 3 trials)
        successful = False
        trajectory = []
        final_coords = rdkit_start_coords.copy()
        
        best_fiedler = -1.0
        best_traj = []
        best_coords = None
        
        for trial in range(1, 4):
            print(f"--- Trial {trial}/3 starting ---")
            # Generate a fresh randomized starting state for each trial to explore diverse folding pathways
            curr_coords = center_coordinates_np(randomize_dihedrals(peptide.initial_coords, peptide.joints))
            trial_traj = []
            
            # Initial state frame metrics
            G_p_init = build_peptide_network(curr_coords, peptide.bonds)
            p_fiedler_init, p_entropy_init = get_network_metrics(G_p_init)
            
            w_coords_init = solvate_peptide(curr_coords, peptide.atoms, num_waters=60)
            G_w_init = build_water_network(w_coords_init)
            w_fiedler_init, w_entropy_init = get_network_metrics(G_w_init)
            
            trial_traj.append({
                "step": 0,
                "activeResidue": -1,
                "peptideFiedler": p_fiedler_init,
                "peptideEntropy": p_entropy_init,
                "waterFiedler": w_fiedler_init,
                "waterEntropy": w_entropy_init
            })
            
            curr_score = p_fiedler_init + w_fiedler_init
            
            # Run coordinate descent sweeps (12 sweeps)
            step_count = 0
            for sweep in range(12):
                # Swapping residues sequentially (excluding terminal caps 0 and N-1)
                for r in range(1, peptide.n_residues - 1):
                    # Locate joints of residue r
                    res_joints = [j for j in peptide.joints if peptide.atom_residues[j["d_idx"]] == r]
                    if not res_joints:
                        continue
                        
                    for joint in res_joints:
                        step_count += 1
                        
                        # Sample candidate rotations
                        best_cand = None
                        best_f_p = -1.0
                        
                        # We only evaluate the 4 non-zero perturbations to find the best peptide Fiedler
                        for d_theta in [-0.15, -0.075, 0.075, 0.15]:
                            cand_coords = rotate_joint(curr_coords, joint, d_theta)
                            
                            # Check steric clashes in peptide candidate
                            dists = np.linalg.norm(cand_coords[:, None, :] - cand_coords[None, :, :], axis=-1)
                            clash = np.any((dists < 2.0) & (peptide.graph_dist >= 4.0))
                            if clash:
                                continue
                                
                            # Compute peptide network Fiedler
                            G_p = build_peptide_network(cand_coords, peptide.bonds)
                            f_p, e_p = get_network_metrics(G_p)
                            
                            if f_p > best_f_p:
                                best_f_p = f_p
                                best_cand = (cand_coords, f_p, e_p)
                                
                        if best_cand is None:
                            continue
                            
                        # Now, solvate and compute water metrics ONLY for the best peptide candidate!
                        cand_coords, f_p, e_p = best_cand
                        cand_waters = solvate_peptide(cand_coords, peptide.atoms, num_waters=60)
                        G_w = build_water_network(cand_waters)
                        f_w, e_w = get_network_metrics(G_w)
                        
                        score = f_p + f_w
                        
                        # Metropolis Monte Carlo selection criterion to avoid local minima
                        delta = score - curr_score
                        prob = np.exp(delta / 0.003) if delta < 0 else 1.0
                        
                        if prob >= random.random():
                            # Accept rotation step
                            curr_coords = cand_coords
                            curr_score = score
                            trial_traj.append({
                                "step": step_count,
                                "activeResidue": r,
                                "peptideFiedler": f_p,
                                "peptideEntropy": e_p,
                                "waterFiedler": f_w,
                                "waterEntropy": e_w
                            })
                            
            # Check convergence of the trial
            final_p_fiedler = trial_traj[-1]["peptideFiedler"] if trial_traj else p_fiedler_init
            print(f"  Trial {trial} completed. Final Peptide Fiedler: {final_p_fiedler:.6f}")
            
            if final_p_fiedler > best_fiedler:
                best_fiedler = final_p_fiedler
                best_traj = trial_traj
                best_coords = curr_coords.copy()
                
            if final_p_fiedler >= acceptance_threshold:
                print(f"✓ Convergence criteria met!")
                successful = True
                trajectory = trial_traj
                final_coords = curr_coords
                break
            else:
                print(f"✗ Trapped in poor local minimum (under threshold). Resetting...")
                
        # If all 3 trials fail to reach threshold, we fall back to the best trial's coordinates
        if not successful:
            print("⚠ Warning: 3 trials completed without reaching the threshold. Returning best available result.")
            trajectory = best_traj if best_traj else [{
                "step": 0,
                "activeResidue": -1,
                "peptideFiedler": p_fiedler_init,
                "peptideEntropy": p_entropy_init,
                "waterFiedler": w_fiedler_init,
                "waterEntropy": w_entropy_init
            }]
            final_coords = best_coords if best_coords is not None else rdkit_start_coords.copy()

        # Center final outputs
        if has_folded_target and raw_folded is not None:
            unfolded_out = center_coordinates(raw_folded)
        else:
            unfolded_out = center_coordinates(unfolded_coords)
        folded_out = center_coordinates(final_coords)
        
        # Calculate static targets for comparison
        G_p_ideal = build_peptide_network(folded_out, peptide.bonds)
        f_p_ideal, e_p_ideal = get_network_metrics(G_p_ideal)
        
        w_coords_ideal = solvate_peptide(folded_out, peptide.atoms, num_waters=60)
        G_w_ideal = build_water_network(w_coords_ideal)
        f_w_ideal, e_w_ideal = get_network_metrics(G_w_ideal)
        
        print("\n========================================================")
        print(f"  FINAL NETWORK GRAPH SIMULATION RESULTS")
        print(f"--------------------------------------------------------")
        print(f"  UNFOLDED STATE (Step 0):")
        print(f"    Peptide Fiedler: {p_fiedler_init:.6f} | Peptide Spectral Entropy: {p_entropy_init:.4f}")
        print(f"    Water Fiedler:   {w_fiedler_init:.6f} | Water Spectral Entropy:   {w_entropy_init:.4f}")
        print(f"  SIMULATED FOLDED STATE (Final Step):")
        print(f"    Peptide Fiedler: {f_p_ideal:.6f} | Peptide Spectral Entropy: {e_p_ideal:.4f}")
        print(f"    Water Fiedler:   {f_w_ideal:.6f} | Water Spectral Entropy:   {e_w_ideal:.4f}")
        print("========================================================\n")
        
        return jsonify({
            "status": "success",
            "sequence": sequence,
            "atoms": peptide.atoms,
            "bonds": peptide.bonds,
            "unfoldedCoords": unfolded_out,
            "foldedCoords": folded_out,
            "atomResidues": peptide.atom_residues,
            "residueLabels": peptide.residue_labels,
            "trajectory": trajectory,
            "targetFiedler": ideal_fiedler,
            "targetEntropy": ideal_spectral_entropy,
            "hasFoldedTarget": has_folded_target
        })
    except Exception as e:
        print(f"Simulation Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

if __name__ == "__main__":
    # Start server on port 5008
    app.run(host="0.0.0.0", port=5008, debug=False)
