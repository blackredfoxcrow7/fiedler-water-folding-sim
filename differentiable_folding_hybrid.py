import numpy as np
import networkx as nx
import os
import json
import sys
from rdkit import Chem
from rdkit.Chem import AllChem

# --- PEPTIDE AGENT WITH HYBRID SUPPORT ---
class PeptideAgent:
    def __init__(self, sequence):
        self.sequence = sequence
        self.mol = self.generate_mol_from_sequence(sequence)
        self.n_atoms = self.mol.GetNumAtoms()
        
        # Get initial coords from RDKit conformer
        conf = self.mol.GetConformer()
        self.initial_coords = np.array([list(conf.GetAtomPosition(i)) for i in range(self.n_atoms)])
        
        self.atoms = []
        for i in range(self.n_atoms):
            atom = self.mol.GetAtomWithIdx(i)
            self.atoms.append({
                "id": i,
                "symbol": atom.GetSymbol(),
                "element": atom.GetSymbol()
            })
            
        self.bonds = []
        for bond in self.mol.GetBonds():
            self.bonds.append({
                "source": bond.GetBeginAtomIdx(),
                "target": bond.GetEndAtomIdx()
            })
            
        self.is_sidechain = np.zeros(self.n_atoms, dtype=bool)
        self.atom_residues = np.zeros(self.n_atoms, dtype=int)
        
        for atom in self.mol.GetAtoms():
            idx = atom.GetIdx()
            res_info = atom.GetPDBResidueInfo()
            if res_info is not None:
                self.atom_residues[idx] = res_info.GetResidueNumber() - 1
                name = res_info.GetName().strip()
                if name not in ["N", "CA", "C", "O", "OXT", "H"]:
                    self.is_sidechain[idx] = True
                    
        # Compute graph distances
        adj = np.zeros((self.n_atoms, self.n_atoms))
        for bond in self.bonds:
            adj[bond["source"], bond["target"]] = 1
            adj[bond["target"], bond["source"]] = 1
        G = nx.from_numpy_array(adj)
        self.graph_dist = nx.all_pairs_shortest_path_length(G)
        self.graph_dist = dict(self.graph_dist)
        self.graph_dist = np.array([[self.graph_dist[i].get(p, 999) for p in range(self.n_atoms)] for i in range(self.n_atoms)])
        
        self.joints = self.identify_backbone_joints()
        
        # Save unfolded random coil coordinates as base coordinates
        self.extended_coords = self.generate_unfolded_initial_coords()

    def generate_mol_from_sequence(self, sequence):
        mol = Chem.MolFromSequence(sequence)
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.useRandomCoords = True
        AllChem.EmbedMolecule(mol, params)
        AllChem.MMFFOptimizeMolecule(mol)
        return mol

    def generate_unfolded_initial_coords(self):
        # Start from relaxed conformer coordinates
        coords = self.initial_coords.copy()
        # Apply large random rotations to all phi/psi backbone joints to unfold the chain
        for joint in self.joints:
            angle = np.random.uniform(-2.0, 2.0)
            coords = rotate_joint(coords, joint, angle)
        return coords

    def identify_backbone_joints(self):
        joints = []
        residues = {}
        for idx in range(self.n_atoms):
            res_idx = self.atom_residues[idx]
            if res_idx not in residues:
                residues[res_idx] = {}
            res_info = self.mol.GetAtomWithIdx(idx).GetPDBResidueInfo()
            if res_info is not None:
                residues[res_idx][res_info.GetName().strip()] = idx
                
        n_res = len(residues)
        for r in range(n_res):
            res = residues[r]
            if "N" in res and "CA" in res:
                n_idx = res["N"]
                ca_idx = res["CA"]
                downstream = self.get_downstream_atoms(n_idx, ca_idx)
                joints.append({
                    "u_idx": n_idx,
                    "p_idx": ca_idx,
                    "downstream_atoms": downstream,
                    "name": f"phi_{r+1}"
                })
            if "CA" in res and "C" in res:
                ca_idx = res["CA"]
                c_idx = res["C"]
                downstream = self.get_downstream_atoms(ca_idx, c_idx)
                joints.append({
                    "u_idx": ca_idx,
                    "p_idx": c_idx,
                    "downstream_atoms": downstream,
                    "name": f"psi_{r+1}"
                })
        return joints

    def get_downstream_atoms(self, u, p):
        adj = {i: [] for i in range(self.n_atoms)}
        for bond in self.bonds:
            adj[bond["source"]].append(bond["target"])
            adj[bond["target"]].append(bond["source"])
        visited = {u}
        queue = [p]
        downstream = []
        while queue:
            curr = queue.pop(0)
            if curr not in visited:
                visited.add(curr)
                downstream.append(curr)
                for nbr in adj[curr]:
                    if nbr not in visited:
                        queue.append(nbr)
        return downstream

def rotate_joint(coords, joint, angle_rad):
    u = coords[joint["u_idx"]]
    p = coords[joint["p_idx"]]
    axis = p - u
    axis = axis / np.linalg.norm(axis)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    ux, uy, uz = axis
    K = np.array([
        [0, -uz, uy],
        [uz, 0, -ux],
        [-uy, ux, 0]
    ])
    R = np.eye(3) + sin_a * K + (1.0 - cos_a) * np.dot(K, K)
    new_coords = coords.copy()
    downstream = joint["downstream_atoms"]
    for idx in downstream:
        new_coords[idx] = u + np.dot(R, coords[idx] - u)
    return new_coords

# --- DIFFERENTIABLE AMIDE HYDROGEN POSITION ---
def compute_amide_hydrogens(coords, peptide):
    h_coords = {}
    residues = {}
    for idx in range(peptide.n_atoms):
        res_idx = peptide.atom_residues[idx]
        if res_idx not in residues:
            residues[res_idx] = {}
        res_info = peptide.mol.GetAtomWithIdx(idx).GetPDBResidueInfo()
        if res_info is not None:
            residues[res_idx][res_info.GetName().strip()] = idx
            
    n_res = len(residues)
    for r in range(1, n_res):
        prev_res = residues[r-1]
        curr_res = residues[r]
        if "C" in prev_res and "N" in curr_res and "CA" in curr_res:
            c_prev = prev_res["C"]
            n_curr = curr_res["N"]
            ca_curr = curr_res["CA"]
            r_c = coords[c_prev]
            r_n = coords[n_curr]
            r_ca = coords[ca_curr]
            
            # Safe norm
            v_nc = r_c - r_n
            v_nca = r_ca - r_n
            norm_nc = np.linalg.norm(v_nc)
            norm_nca = np.linalg.norm(v_nca)
            if norm_nc < 0.1 or norm_nca < 0.1:
                continue
                
            u_nc = v_nc / norm_nc
            u_nca = v_nca / norm_nca
            v_bisect = -(u_nc + u_nca)
            norm_bisect = np.linalg.norm(v_bisect)
            if norm_bisect < 0.1:
                continue
                
            u_bisect = v_bisect / norm_bisect
            r_h = r_n + 1.01 * u_bisect
            h_coords[n_curr] = {
                "coord": r_h,
                "n_idx": n_curr,
                "c_prev_idx": c_prev,
                "ca_curr_idx": ca_curr,
                "u_bisect": u_bisect
            }
    return h_coords

# --- HYBRID SIMULATOR ENGINE ---
def run_hybrid_simulation(sequence, steps=300, nucleation_steps=30, learning_rate=0.15, temp_noise=0.005):
    print("==================================================")
    print("  HYBRID SPECTRAL FOLDING SIMULATOR")
    print("  Phase 1: Discrete Nucleation & Locking")
    print("  Phase 2: Differentiable Propagation")
    print("==================================================")
    
    peptide = PeptideAgent(sequence)
    n_atoms = peptide.n_atoms
    joints = peptide.joints
    n_joints = len(joints)
    
    print(f" Peptide: {sequence} | Atoms: {n_atoms} | Joints: {n_joints}")
    print("==================================================")
    
    theta = np.zeros(n_joints)
    
    # Identify atom groups
    is_polar_acc = np.zeros(n_atoms, dtype=bool)
    is_polar_don = np.zeros(n_atoms, dtype=bool)
    is_hydro = np.zeros(n_atoms, dtype=bool)
    carbonyl_c_map = {}
    
    residues = {}
    for idx in range(n_atoms):
        res_idx = peptide.atom_residues[idx]
        if res_idx not in residues:
            residues[res_idx] = {}
        res_info = peptide.mol.GetAtomWithIdx(idx).GetPDBResidueInfo()
        if res_info is not None:
            residues[res_idx][res_info.GetName().strip()] = idx
            
    for r in range(len(residues)):
        res = residues[r]
        if "O" in res and "C" in res:
            carbonyl_c_map[res["O"]] = res["C"]
        if "N" in res:
            is_polar_don[res["N"]] = True
        if "O" in res:
            is_polar_acc[res["O"]] = True
            
    for idx in range(n_atoms):
        sym = peptide.atoms[idx]["symbol"]
        is_sc = peptide.is_sidechain[idx]
        if sym == "C" and is_sc:
            is_hydro[idx] = True
            
    # Topological Memory: Locked hydrogen bonds (source, target, weight)
    nucleated_bonds = []
    
    print("--------------------------------------------------")
    print(f" {'Step':<6} | {'Phase':<12} | {'Fiedler (L2)':<15} | {'Clashes':<8} | {'Rg (A)':<8} | {'Active/Locked':<15}")
    print("--------------------------------------------------")
    
    for step in range(steps):
        is_nucleation = step < nucleation_steps
        
        if is_nucleation:
            # --- PHASE 1: DISCRETE NUCLEATION ---
            # Randomly perturb angles and check if new hydrogen bonds form.
            # If they form and meet linear directional criteria, we LOCK them!
            best_theta = theta.copy()
            best_score = -999.0
            best_coords = None
            best_clashes = 999
            
            # Try 10 random perturbations
            for trial in range(15):
                trial_theta = theta + np.random.normal(0.0, 0.25, n_joints)
                
                # Update coords
                trial_coords = peptide.extended_coords.copy()
                for k, joint in enumerate(joints):
                    trial_coords = rotate_joint(trial_coords, joint, trial_theta[k])
                    
                h_map = compute_amide_hydrogens(trial_coords, peptide)
                
                # Check clashes
                clashes = 0
                for i in range(n_atoms):
                    for p in range(i + 1, n_atoms):
                        if peptide.graph_dist[i, p] < 4.0: continue
                        d = np.linalg.norm(trial_coords[i] - trial_coords[p])
                        sym_i = peptide.atoms[i]["symbol"]
                        sym_p = peptide.atoms[p]["symbol"]
                        thresh = 1.8 if (sym_i == "H" or sym_p == "H") else 2.5
                        if d < thresh:
                            clashes += 1
                            
                # Count new valid directional hydrogen bonds
                new_bonds = []
                for n_idx, h_info in h_map.items():
                    res_n = peptide.atom_residues[n_idx]
                    r_h = h_info["coord"]
                    r_n = trial_coords[n_idx]
                    
                    for o_idx in range(n_atoms):
                        if not is_polar_acc[o_idx]: continue
                        res_o = peptide.atom_residues[o_idx]
                        if abs(res_n - res_o) < 3: continue
                        
                        r_o = trial_coords[o_idx]
                        r_c = trial_coords[carbonyl_c_map[o_idx]]
                        
                        d_ho = np.linalg.norm(r_o - r_h)
                        if d_ho < 4.2:
                            u_nh = (r_h - r_n) / np.linalg.norm(r_h - r_n)
                            u_ho = (r_o - r_h) / np.linalg.norm(r_o - r_h)
                            u_oc = (r_c - r_o) / np.linalg.norm(r_c - r_o)
                            
                            cos_theta = np.dot(u_nh, u_ho)
                            cos_phi = np.dot(-u_ho, u_oc)
                            
                            if cos_theta > 0.6 and cos_phi > 0.1:
                                new_bonds.append((n_idx, o_idx, 20.0 / (d_ho**2)))
                                
                # Score is number of new bonds minus clash penalty
                score = len(new_bonds) * 10.0 - clashes * 0.5
                if score > best_score:
                    best_score = score
                    best_theta = trial_theta
                    best_coords = trial_coords
                    best_clashes = clashes
                    # If we found any new bonds, lock them in memory
                    if new_bonds:
                        for b in new_bonds:
                            if b not in nucleated_bonds:
                                nucleated_bonds.append(b)
                                
            theta = best_theta
            curr_coords = best_coords
            clash_count = best_clashes
            
        else:
            # --- PHASE 2: CONTINUOUS DIFF PROPAGATION ---
            curr_coords = peptide.extended_coords.copy()
            for k, joint in enumerate(joints):
                curr_coords = rotate_joint(curr_coords, joint, theta[k])
            h_map = compute_amide_hydrogens(curr_coords, peptide)
            
        # Distances and clashes
        dists = np.linalg.norm(curr_coords[:, None, :] - curr_coords[None, :, :], axis=-1)
        clash_count = 0
        clash_pairs = []
        for i in range(n_atoms):
            for p in range(i + 1, n_atoms):
                if peptide.graph_dist[i, p] < 4.0: continue
                d = dists[i, p]
                sym_i = peptide.atoms[i]["symbol"]
                sym_p = peptide.atoms[p]["symbol"]
                thresh = 1.8 if (sym_i == "H" or sym_p == "H") else 2.5
                if d < thresh:
                    clash_count += 1
                    clash_pairs.append((i, p, d, thresh))
                    
        # Identify active bonds based on phase
        is_polar_phase = (step // 10) % 2 == 0
        active_bonds = []
        
        # Always include locked nucleation bonds in the Laplacian to prevent melting!
        for b_src, b_tgt, b_w in nucleated_bonds:
            active_bonds.append((b_src, b_tgt, b_w, "locked"))
            
        if not is_nucleation:
            if is_polar_phase:
                # Add dynamic polar bonds with directional constraints
                for n_idx, h_info in h_map.items():
                    res_n = peptide.atom_residues[n_idx]
                    r_h = h_info["coord"]
                    r_n = curr_coords[n_idx]
                    
                    for o_idx in range(n_atoms):
                        if not is_polar_acc[o_idx]: continue
                        res_o = peptide.atom_residues[o_idx]
                        if abs(res_n - res_o) < 3: continue
                        
                        r_o = curr_coords[o_idx]
                        r_c = curr_coords[carbonyl_c_map[o_idx]]
                        
                        d_ho = np.linalg.norm(r_o - r_h)
                        if d_ho < 4.5:
                            u_nh = (r_h - r_n) / np.linalg.norm(r_h - r_n)
                            u_ho = (r_o - r_h) / np.linalg.norm(r_o - r_h)
                            u_oc = (r_c - r_o) / np.linalg.norm(r_c - r_o)
                            cos_theta = np.dot(u_nh, u_ho)
                            cos_phi = np.dot(-u_ho, u_oc)
                            
                            dir_factor = max(0, cos_theta)**2 * max(0, cos_phi)**2
                            w = dir_factor * (16.0 / (d_ho**2))
                            if w > 0.05:
                                active_bonds.append((n_idx, o_idx, w, "direct"))
            else:
                # Add dynamic hydrophobic sidechain packing contacts
                for i in range(n_atoms):
                    if not is_hydro[i]: continue
                    for p in range(i + 1, n_atoms):
                        if not is_hydro[p]: continue
                        if peptide.graph_dist[i, p] < 4.0: continue
                        d = dists[i, p]
                        if d < 4.5:
                            active_bonds.append((i, p, 40.0 / (d**2), "hydro"))
                            
        # Build Laplacian & Compute Fiedler Value
        G_pep = nx.Graph()
        G_pep.add_nodes_from(range(n_atoms))
        for cb in peptide.bonds:
            G_pep.add_edge(cb["source"], cb["target"], weight=10.0)
        for idx1, idx2, w, _ in active_bonds:
            G_pep.add_edge(idx1, idx2, weight=w)
            
        l2 = nx.algebraic_connectivity(G_pep, method='lanczos', tol=1e-5)
        
        # Get Fiedler Vector v2
        L_matrix = nx.laplacian_matrix(G_pep).toarray()
        eigvals, eigvecs = np.linalg.eigh(L_matrix)
        v2 = eigvecs[:, 1]
        
        # Compute current Rg
        center = np.mean(curr_coords, axis=0)
        rg = np.sqrt(np.mean(np.sum((curr_coords - center)**2, axis=-1)))
        
        # Print progress
        phase_str = "Nucleation" if is_nucleation else ("Polar" if is_polar_phase else "Hydro")
        if step % 10 == 0 or step == steps - 1:
            print(f" {step:<6} | {phase_str:<12} | {l2:<15.5e} | {clash_count:<8} | {rg:<8.3f} | {len(active_bonds):<15}")
            
        # --- BACKWARD PASS (Only for Phase 2) ---
        if not is_nucleation:
            # Coordinate Jacobians
            dr_dtheta = np.zeros((n_joints, n_atoms, 3))
            for k, joint in enumerate(joints):
                u_idx = joint["u_idx"]
                p_idx = joint["p_idx"]
                axis = curr_coords[p_idx] - curr_coords[u_idx]
                axis = axis / np.linalg.norm(axis)
                downstream = joint["downstream_atoms"]
                for idx in downstream:
                    dr_dtheta[k, idx] = np.cross(axis, curr_coords[idx] - curr_coords[u_idx])
                    
            # 1. Fiedler Gradient
            grad_l2 = np.zeros(n_joints)
            for idx1, idx2, w, b_type in active_bonds:
                dv_sq = (v2[idx1] - v2[idx2])**2
                d = dists[idx1, idx2]
                
                # Weight derivative
                if b_type == "locked":
                    dw_dd = -2.0 * w / d # same derivative
                elif b_type == "direct" or b_type == "hydro":
                    dw_dd = -2.0 * w / d
                else:
                    dw_dd = 0.0 # none
                    
                for k in range(n_joints):
                    dr_1 = dr_dtheta[k, idx1]
                    dr_2 = dr_dtheta[k, idx2]
                    dd_dtheta = (1.0 / max(d, 0.1)) * np.dot(curr_coords[idx1] - curr_coords[idx2], dr_1 - dr_2)
                    grad_l2[k] += dv_sq * dw_dd * dd_dtheta
                    
            # 2. Steric Gradient
            grad_steric = np.zeros(n_joints)
            for i, p, d, thresh in clash_pairs:
                de_dd = -20.0 / ((max(d - (thresh - 0.1), 0.05))**3)
                for k in range(n_joints):
                    dr_i = dr_dtheta[k, i]
                    dr_p = dr_dtheta[k, p]
                    dd_dtheta = (1.0 / max(d, 0.1)) * np.dot(curr_coords[i] - curr_coords[p], dr_i - dr_p)
                    grad_steric[k] += de_dd * dd_dtheta
                    
            # Normalize
            grad_l2_norm = np.linalg.norm(grad_l2)
            if grad_l2_norm > 1.0: grad_l2 /= grad_l2_norm
            
            grad_steric_norm = np.linalg.norm(grad_steric)
            if grad_steric_norm > 1.0: grad_steric /= grad_steric_norm
            
            # Gradient Update (NO COMPACTION Rg FORCE NEEDED! Anchors drive compaction!)
            gradient = grad_l2 - 1.0 * grad_steric
            theta += learning_rate * gradient
            theta += np.random.normal(0.0, temp_noise, n_joints)
            
    print("==================================================")
    print("  HYBRID SIMULATION COMPLETED SUCCESSFULLY!")
    
    # Save final coordinates
    folded_data = {
        "sequence": sequence,
        "atoms": peptide.atoms,
        "atomResidues": peptide.atom_residues.tolist(),
        "foldedCoords": curr_coords.tolist()
    }
    out_filename = f"{sequence.lower()}_hybrid_folded.json"
    with open(out_filename, "w") as f:
        json.dump(folded_data, f)
    print(f" Saved hybrid structure to {out_filename}")
    print("==================================================")

if __name__ == "__main__":
    seq = "AAAAAAAAAA"
    steps = 300
    nuc_steps = 30
    lr = 0.15
    if len(sys.argv) > 1:
        seq = sys.argv[1]
    if len(sys.argv) > 2:
        steps = int(sys.argv[2])
    if len(sys.argv) > 3:
        nuc_steps = int(sys.argv[3])
    if len(sys.argv) > 4:
        lr = float(sys.argv[4])
    run_hybrid_simulation(sequence=seq, steps=steps, nucleation_steps=nuc_steps, learning_rate=lr)
