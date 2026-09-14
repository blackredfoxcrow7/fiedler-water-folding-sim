import numpy as np
import networkx as nx
from rdkit import Chem
from server_overall_fiedler_accumulative import PeptideAgent, generate_extended_coords, rotate_joint

def run_differentiable_folding_simulation(sequence="AAAAAAAAAA", steps=100, learning_rate=0.05, temp_noise=0.01):
    print("==================================================")
    print(f"  DIFFERENTIABLE SPECTRAL FOLDING SIMULATOR")
    print(f"  AI-like Forward/Backward Gradient Optimization")
    print("==================================================")
    
    # 1. Initialize peptide agent and use RDKit initial conformer coordinates
    peptide = PeptideAgent(sequence)
    coords = peptide.initial_coords.copy()
    
    n_atoms = peptide.n_atoms
    joints = peptide.joints
    n_joints = len(joints)
    
    print(f" Peptide: {sequence} | Atoms: {n_atoms} | Dihedral Joints: {n_joints}")
    print("==================================================")
    
    # Track joint angles (initialized to 0)
    theta = np.zeros(n_joints)
    
    # Identify polar and hydrophobic atom types
    is_polar = np.zeros(n_atoms, dtype=bool)
    is_hydro = np.zeros(n_atoms, dtype=bool)
    
    for i in range(n_atoms):
        sym = peptide.atoms[i]["symbol"]
        is_sc_atom = peptide.is_sidechain[i]
        
        # Backbone polar atoms (carbonyl oxygen and amide nitrogen)
        if sym in ["O", "N"] and not is_sc_atom:
            is_polar[i] = True
        elif sym == "C" and not is_sc_atom:
            # Carbonyl carbon is neutral/covalent
            pass
        elif sym == "C" and is_sc_atom:
            # Sidechain carbon (hydrophobic for Alanine/Trp/etc)
            is_hydro[i] = True
            
    print(f" Backbone Polar Atoms: {np.sum(is_polar)} | Sidechain Hydrophobic Atoms: {np.sum(is_hydro)}")
    print("--------------------------------------------------")
    print(f" {'Step':<6} | {'Fiedler Value (L2)':<20} | {'Steric Clashes':<15} | {'Rg (A)':<10}")
    print("--------------------------------------------------")
    
    # Main optimization loop
    for step in range(steps):
        # --- FORWARD PASS ---
        # 1. Update 3D coordinates based on joint angles theta starting from initial conformer
        curr_coords = peptide.initial_coords.copy()
        for k, joint in enumerate(joints):
            curr_coords = rotate_joint(curr_coords, joint, theta[k])
            
        # 2. Compute pairwise distances
        dists = np.linalg.norm(curr_coords[:, None, :] - curr_coords[None, :, :], axis=-1)
        
        # 3. Identify active non-covalent bonds based on phase (alternating every 10 steps)
        is_polar_phase = (step // 10) % 2 == 0
        active_bonds = []
        
        if is_polar_phase:
            # Polar Phase: Backbone Hydrogen Bonds (valence constraints: N<=1, O<=2)
            polar_candidates = []
            for i in range(n_atoms):
                if not is_polar[i]: continue
                for p in range(i + 1, n_atoms):
                    if not is_polar[p]: continue
                    if peptide.graph_dist[i, p] < 4.0: continue
                    
                    d = dists[i, p]
                    if d < 4.5:
                        sym_i = peptide.atoms[i]["symbol"]
                        sym_p = peptide.atoms[p]["symbol"]
                        if (sym_i == "O" and sym_p == "N") or (sym_i == "N" and sym_p == "O"):
                            polar_candidates.append((i, p, d))
                            
            polar_candidates = sorted(polar_candidates, key=lambda x: x[2])
            n_deg = {a: 0 for a in range(n_atoms)}
            o_deg = {a: 0 for a in range(n_atoms)}
            
            for i, p, d in polar_candidates:
                sym_i = peptide.atoms[i]["symbol"]
                sym_p = peptide.atoms[p]["symbol"]
                o_idx = i if sym_i == "O" else p
                n_idx = p if sym_i == "O" else i
                
                if o_deg[o_idx] < 2 and n_deg[n_idx] < 1:
                    o_deg[o_idx] += 1
                    n_deg[n_idx] += 1
                    active_bonds.append((i, p, d, 16.0 / (d**2)))
        else:
            # Hydrophobic Phase: Sidechain hydrophobic packing contacts (no valence limits)
            for i in range(n_atoms):
                if not is_hydro[i]: continue
                for p in range(i + 1, n_atoms):
                    if not is_hydro[p]: continue
                    if peptide.graph_dist[i, p] < 4.0: continue
                    
                    d = dists[i, p]
                    if d < 4.5:
                        active_bonds.append((i, p, d, 40.0 / (d**2)))
                        
        # 4. Build the Laplacian matrix for the active bonds
        # Start with covalent backbone template
        G_pep = nx.Graph()
        G_pep.add_nodes_from(range(n_atoms))
        for cb in peptide.bonds:
            G_pep.add_edge(cb["source"], cb["target"], weight=10.0)
            
        # Add active bonds
        for i, p, d, w in active_bonds:
            G_pep.add_edge(i, p, weight=w)
            
        # Compute Fiedler value and eigenvector
        l2 = nx.algebraic_connectivity(G_pep, method='lanczos', tol=1e-5)
        
        # Get Fiedler vector (second smallest eigenvector of Laplacian)
        L_matrix = nx.laplacian_matrix(G_pep).toarray()
        eigvals, eigvecs = np.linalg.eigh(L_matrix)
        v2 = eigvecs[:, 1]
        
        # 5. Calculate steric clashes and steric potential
        clash_count = 0
        steric_potential = 0.0
        clash_pairs = []
        for i in range(n_atoms):
            for p in range(i + 1, n_atoms):
                if peptide.graph_dist[i, p] < 4.0: continue
                d = dists[i, p]
                if d < 2.5:
                    clash_count += 1
                    dist_diff = max(d - 2.4, 0.05)
                    clash_pairs.append((i, p, d))
                    steric_potential += 10.0 / (dist_diff**2) # penalty wall
                    
        # Compute current Radius of Gyration
        center = np.mean(curr_coords, axis=0)
        rg = np.sqrt(np.mean(np.sum((curr_coords - center)**2, axis=-1)))
        
        # Print progress
        if step % 10 == 0 or step == steps - 1:
            print(f" {step:<6} | {l2:<20.6e} | {clash_count:<15} | {rg:<10.3f}")
            
        # --- BACKWARD PASS (GRADIENT CALCULATION) ---
        # 1. Compute Jacobians of atomic coordinates: d(r_i) / d(theta_k)
        # For each joint k, atoms in joint["downstream_atoms"] rotate
        dr_dtheta = np.zeros((n_joints, n_atoms, 3))
        for k, joint in enumerate(joints):
            u_idx = joint["u_idx"]
            d_idx = joint["d_idx"]
            downstream = joint["downstream_atoms"]
            
            pos_A = curr_coords[u_idx]
            pos_B = curr_coords[d_idx]
            
            axis = pos_B - pos_A
            axis_len = np.linalg.norm(axis)
            if axis_len > 1e-5:
                axis /= axis_len
                
            # The derivative of rotated coordinate r_i is: axis x (r_i - pos_B)
            for i in downstream:
                r_vec = curr_coords[i] - pos_B
                dr_dtheta[k, i] = np.cross(axis, r_vec)
                
        # 2. Compute Fiedler gradient: d(l2) / d(theta_k)
        grad_l2 = np.zeros(n_joints)
        for i, p, d, w in active_bonds:
            # Derivative of eigenvalue w.r.t edge weight: (v2[i] - v2[p])^2
            dl2_dw = (v2[i] - v2[p])**2
            
            # Derivative of polar weight w.r.t distance: -32.0 / d^3
            dw_dd = -32.0 / (d**3)
            
            # Derivative of distance w.r.t coordinates
            # d(d_ip) / d(r_i) = (r_i - r_p) / d_ip
            # d(d_ip) / d(r_p) = (r_p - r_i) / d_ip
            for k in range(n_joints):
                dr_i = dr_dtheta[k, i]
                dr_p = dr_dtheta[k, p]
                
                # Chain rule: dd_dtheta_k = (r_i - r_p)/d . (dr_i/dtheta - dr_p/dtheta)
                dd_dtheta = (1.0 / d) * np.dot(curr_coords[i] - curr_coords[p], dr_i - dr_p)
                
                grad_l2[k] += dl2_dw * dw_dd * dd_dtheta
                
        # 3. Compute Steric Gradient: d(E_steric) / d(theta_k)
        grad_steric = np.zeros(n_joints)
        for i, p, d in clash_pairs:
            # Derivative of penalty w.r.t distance: -20.0 / (d - 2.4)^3
            de_dd = -20.0 / ((max(d - 2.4, 0.05))**3)
            
            for k in range(n_joints):
                dr_i = dr_dtheta[k, i]
                dr_p = dr_dtheta[k, p]
                dd_dtheta = (1.0 / max(d, 0.1)) * np.dot(curr_coords[i] - curr_coords[p], dr_i - dr_p)
                
                grad_steric[k] += de_dd * dd_dtheta
                
        # 4. Compute Compaction Gradient (Radius of Gyration squared derivative)
        # d(Rg^2) / d(theta_k) = (2 / N) * sum_i (r_i - r_center) . dr_i/dtheta_k
        grad_rg = np.zeros(n_joints)
        if not is_polar_phase:
            for k in range(n_joints):
                for i in range(n_atoms):
                    grad_rg[k] += (2.0 / n_atoms) * np.dot(curr_coords[i] - center, dr_dtheta[k, i])
                
        # Normalize gradients to ensure stable updates
        grad_l2_norm = np.linalg.norm(grad_l2)
        if grad_l2_norm > 1.0: grad_l2 /= grad_l2_norm
        
        grad_steric_norm = np.linalg.norm(grad_steric)
        if grad_steric_norm > 1.0: grad_steric /= grad_steric_norm
        
        grad_rg_norm = np.linalg.norm(grad_rg)
        if grad_rg_norm > 1.0: grad_rg /= grad_rg_norm
                
        # --- GRADIENT UPDATE (WEIGHT UPDATE) ---
        # We maximize Fiedler (grad_l2), minimize steric potential (grad_steric), and minimize Rg^2 (grad_rg)
        # Gradient Update Step: + grad_l2 - 1.0 * grad_steric - 0.3 * grad_rg
        gradient = grad_l2 - 1.0 * grad_steric - 0.3 * grad_rg
        
        # Apply gradient update to joint angles theta
        theta += learning_rate * gradient
        
        # Add Langevin-like random thermal noise
        theta += np.random.normal(0.0, temp_noise, n_joints)
        
    print("==================================================")
    print("  SIMULATION COMPLETED SUCCESSFULLY!")
    
    # Save final folded coordinates to JSON
    folded_data = {
        "sequence": sequence,
        "atoms": peptide.atoms,
        "atomResidues": peptide.atom_residues,
        "foldedCoords": curr_coords.tolist()
    }
    out_filename = f"{sequence.lower()}_differentiable_folded.json"
    with open(out_filename, "w") as f:
        json.dump(folded_data, f)
    print(f" Saved final structure to {out_filename}")
    print("==================================================")

if __name__ == "__main__":
    import json
    import sys
    seq = "AAAAAAAAAA"
    steps = 300
    lr = 0.15
    if len(sys.argv) > 1:
        seq = sys.argv[1]
    if len(sys.argv) > 2:
        steps = int(sys.argv[2])
    if len(sys.argv) > 3:
        lr = float(sys.argv[3])
    run_differentiable_folding_simulation(sequence=seq, steps=steps, learning_rate=lr, temp_noise=0.005)
