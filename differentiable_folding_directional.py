import numpy as np
import networkx as nx
import os
import json
import sys
from rdkit import Chem
from rdkit.Chem import AllChem

# --- PEPTIDE AGENT WITH AMIDE HYDROGENS & DIRECTIONAL POTENTIALS ---
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
            
        # Classify sidechain atoms
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
        
        # Identify backbone joints (dihedrals phi/psi)
        self.joints = self.identify_backbone_joints()

    def generate_mol_from_sequence(self, sequence):
        mol = Chem.MolFromSequence(sequence)
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.useRandomCoords = True
        AllChem.EmbedMolecule(mol, params)
        AllChem.MMFFOptimizeMolecule(mol)
        return mol

    def identify_backbone_joints(self):
        # We find N-CA bonds (phi) and CA-C bonds (psi) along the backbone
        joints = []
        
        # Gather backbone indices per residue
        residues = {}
        for idx in range(self.n_atoms):
            res_idx = self.atom_residues[idx]
            if res_idx not in residues:
                residues[res_idx] = {}
            
            # Map by PDB name
            res_info = self.mol.GetAtomWithIdx(idx).GetPDBResidueInfo()
            if res_info is not None:
                name = res_info.GetName().strip()
                residues[res_idx][name] = idx
                
        n_res = len(residues)
        for r in range(n_res):
            res = residues[r]
            
            # Phi joint: N - CA bond
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
                
            # Psi joint: CA - C bond
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
        # Find all atoms downstream of bond u -> p
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
    # Rotate downstream atoms around joint axis u -> p
    u = coords[joint["u_idx"]]
    p = coords[joint["p_idx"]]
    axis = p - u
    axis = axis / np.linalg.norm(axis)
    
    # Rodrigues rotation matrix formula
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

# --- DIFFERENTIABLE AMIDE HYDROGEN POSITION CALCULATOR ---
def compute_amide_hydrogens(coords, peptide):
    # Generates explicit coordinates for the amide hydrogens (H) in a differentiable way.
    # The nitrogen sp2 hydrogen lies in the plane of preceding C, current N, and current CA,
    # trans to the preceding carbonyl oxygen, bisecting C-N-CA.
    
    h_coords = {} # maps nitrogen atom index -> H coordinate vector
    
    # Gather residue mapping
    residues = {}
    for idx in range(peptide.n_atoms):
        res_idx = peptide.atom_residues[idx]
        if res_idx not in residues:
            residues[res_idx] = {}
        res_info = peptide.mol.GetAtomWithIdx(idx).GetPDBResidueInfo()
        if res_info is not None:
            name = res_info.GetName().strip()
            residues[res_idx][name] = idx
            
    n_res = len(residues)
    for r in range(1, n_res): # Residue 0 (N-terminal) usually has NH3+, we skip or handle peptide bonds
        prev_res = residues[r-1]
        curr_res = residues[r]
        
        if "C" in prev_res and "N" in curr_res and "CA" in curr_res:
            c_prev = prev_res["C"]
            n_curr = curr_res["N"]
            ca_curr = curr_res["CA"]
            
            r_c = coords[c_prev]
            r_n = coords[n_curr]
            r_ca = coords[ca_curr]
            
            # Vectors from N
            v_nc = r_c - r_n
            v_nca = r_ca - r_n
            
            # Normalized vectors
            u_nc = v_nc / np.linalg.norm(v_nc)
            u_nca = v_nca / np.linalg.norm(v_nca)
            
            # Bisector vector pointing outward (trans to C=O)
            v_bisect = -(u_nc + u_nca)
            u_bisect = v_bisect / np.linalg.norm(v_bisect)
            
            # N-H bond length is approx 1.01 Angstroms
            r_h = r_n + 1.01 * u_bisect
            h_coords[n_curr] = {
                "coord": r_h,
                "n_idx": n_curr,
                "c_prev_idx": c_prev,
                "ca_curr_idx": ca_curr,
                "u_bisect": u_bisect
            }
    return h_coords

# --- MAIN OPTIMIZATION SIMULATOR ---
def run_differentiable_folding_simulation(sequence, steps=300, learning_rate=0.15, temp_noise=0.005):
    print("==================================================")
    print("  DIRECTIONAL & WATER-BRIDGED SPECTRAL FOLDING")
    print("  Explicit H-Atoms & Water Bridges Simulator")
    print("==================================================")
    
    peptide = PeptideAgent(sequence)
    n_atoms = peptide.n_atoms
    joints = peptide.joints
    n_joints = len(joints)
    
    print(f" Peptide: {sequence} | Atoms: {n_atoms} | Joints: {n_joints}")
    print("==================================================")
    
    theta = np.zeros(n_joints)
    
    # Identify atom groups
    is_polar_acc = np.zeros(n_atoms, dtype=bool) # Acceptors (Oxygen)
    is_polar_don = np.zeros(n_atoms, dtype=bool) # Donors (Nitrogen)
    is_hydro = np.zeros(n_atoms, dtype=bool)     # Hydrophobic carbons
    
    # Map residues to find carbonyl carbon for oxygen
    carbonyl_c_map = {}
    
    residues = {}
    for idx in range(n_atoms):
        res_idx = peptide.atom_residues[idx]
        if res_idx not in residues:
            residues[res_idx] = {}
        res_info = peptide.mol.GetAtomWithIdx(idx).GetPDBResidueInfo()
        if res_info is not None:
            name = res_info.GetName().strip()
            residues[res_idx][name] = idx
            
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
            
    print(f" Polar Donors (N): {np.sum(is_polar_don)} | Polar Acceptors (O): {np.sum(is_polar_acc)} | Hydrophobic: {np.sum(is_hydro)}")
    print("--------------------------------------------------")
    print(f" {'Step':<6} | {'Phase':<8} | {'Fiedler (L2)':<15} | {'Clashes':<8} | {'Rg (A)':<8} | {'Active Bonds':<12}")
    print("--------------------------------------------------")
    
    for step in range(steps):
        # 1. Update 3D Coordinates based on theta starting from initial conformer
        curr_coords = peptide.initial_coords.copy()
        for k, joint in enumerate(joints):
            curr_coords = rotate_joint(curr_coords, joint, theta[k])
            
        # 2. Compute Amide Hydrogens (H)
        h_map = compute_amide_hydrogens(curr_coords, peptide)
        
        # 3. Identify active non-covalent bonds (Alternating Phases)
        is_polar_phase = (step // 10) % 2 == 0
        active_bonds = [] # list of (idx1, idx2, weight, type)
        
        # Distances matrix
        dists = np.linalg.norm(curr_coords[:, None, :] - curr_coords[None, :, :], axis=-1)
        
        if is_polar_phase:
            # POLAR PHASE: Direct Hydrogen Bonds + Virtual Water Bridges with Angle Constraints
            # For each Amide Nitrogen N (and its H) and Carbonyl Oxygen O
            for n_idx, h_info in h_map.items():
                r_n = curr_coords[n_idx]
                r_h = h_info["coord"]
                res_n = peptide.atom_residues[n_idx]
                
                for o_idx in range(n_atoms):
                    if not is_polar_acc[o_idx]: continue
                    res_o = peptide.atom_residues[o_idx]
                    if abs(res_n - res_o) < 3: continue # Skip nearby covalent sequence
                    
                    r_o = curr_coords[o_idx]
                    r_c = curr_coords[carbonyl_c_map[o_idx]]
                    
                    # Distance H ... O
                    d_ho = np.linalg.norm(r_o - r_h)
                    
                    # Directional Vectors
                    d_nh = r_h - r_n
                    d_ho_vec = r_o - r_h
                    d_oc = r_c - r_o
                    
                    u_nh = d_nh / np.linalg.norm(d_nh)
                    u_ho = d_ho_vec / np.linalg.norm(d_ho_vec)
                    u_oc = d_oc / np.linalg.norm(d_oc)
                    
                    # Cosine of donor angle N-H ... O (should be close to 1.0, linear)
                    cos_theta = np.dot(u_nh, u_ho)
                    # Cosine of acceptor angle H ... O=C (should be close to 120 deg, cos(120)=-0.5, i.e. dot product -u_ho . u_oc ~ 0.5)
                    # We want cos(phi) to be positive when aligned correctly
                    cos_phi = np.dot(-u_ho, u_oc)
                    
                    # Directional Factor: maximum at linear NH...O and favorable acceptor angle
                    dir_factor = max(0, cos_theta)**2 * max(0, cos_phi)**2
                    
                    if d_ho < 4.5:
                        # Direct Hydrogen Bond (ideal distance ~1.8 - 2.0 A)
                        w = dir_factor * (16.0 / (d_ho**2))
                        if w > 0.01:
                            # Add connection between N and O in the Laplacian graph
                            active_bonds.append((n_idx, o_idx, w, "direct"))
                    elif 4.5 <= d_ho <= 7.0:
                        # Virtual Water-Mediated Bridge (ideal distance H...O_W...O ~ 4.6 A)
                        # We use a Lorentzian centered at 4.6 A
                        w_water = dir_factor * (8.0 / ((d_ho - 4.6)**2 + 1.0))
                        if w_water > 0.01:
                            active_bonds.append((n_idx, o_idx, w_water, "water"))
        else:
            # HYDROPHOBIC PHASE: Sidechain hydrophobic contacts (no angle constraints)
            for i in range(n_atoms):
                if not is_hydro[i]: continue
                for p in range(i + 1, n_atoms):
                    if not is_hydro[p]: continue
                    if peptide.graph_dist[i, p] < 4.0: continue
                    
                    d = dists[i, p]
                    if d < 4.5:
                        active_bonds.append((i, p, 40.0 / (d**2), "hydro"))
                        
        # 4. Build Laplacian & Compute Fiedler Value
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
        
        # 5. Calculate Clashes
        clash_count = 0
        clash_pairs = []
        for i in range(n_atoms):
            for p in range(i + 1, n_atoms):
                if peptide.graph_dist[i, p] < 4.0: continue
                d = dists[i, p]
                sym_i = peptide.atoms[i]["symbol"]
                sym_p = peptide.atoms[p]["symbol"]
                clash_thresh = 1.8 if (sym_i == "H" or sym_p == "H") else 2.5
                if d < clash_thresh:
                    clash_count += 1
                    clash_pairs.append((i, p, d, clash_thresh))
                    
        # Compute Radius of Gyration
        center = np.mean(curr_coords, axis=0)
        rg = np.sqrt(np.mean(np.sum((curr_coords - center)**2, axis=-1)))
        
        # Print progress
        phase_str = "Polar" if is_polar_phase else "Hydro"
        if step % 10 == 0 or step == steps - 1:
            print(f" {step:<6} | {phase_str:<8} | {l2:<15.5e} | {clash_count:<8} | {rg:<8.3f} | {len(active_bonds):<12}")
            
        # --- BACKWARD PASS ---
        # Compute coordinate Jacobians dr_i / dtheta_k
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
            
            # Derivative of weight w.r.t distance d
            d = dists[idx1, idx2]
            if b_type == "direct" or b_type == "hydro":
                dw_dd = -2.0 * w / d
            else: # water bridge
                dw_dd = -2.0 * w * (d - 4.6) / ((d - 4.6)**2 + 1.0)
                
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
                
        # 3. Compaction Gradient
        grad_rg = np.zeros(n_joints)
        if not is_polar_phase: # Compaction only during hydrophobic phase
            for k in range(n_joints):
                for i in range(n_atoms):
                    grad_rg[k] += (2.0 / n_atoms) * np.dot(curr_coords[i] - center, dr_dtheta[k, i])
                    
        # Normalize
        grad_l2_norm = np.linalg.norm(grad_l2)
        if grad_l2_norm > 1.0: grad_l2 /= grad_l2_norm
        
        grad_steric_norm = np.linalg.norm(grad_steric)
        if grad_steric_norm > 1.0: grad_steric /= grad_steric_norm
        
        grad_rg_norm = np.linalg.norm(grad_rg)
        if grad_rg_norm > 1.0: grad_rg /= grad_rg_norm
        
        # Gradient Update Step
        gradient = grad_l2 - 1.0 * grad_steric - 0.3 * grad_rg
        theta += learning_rate * gradient
        theta += np.random.normal(0.0, temp_noise, n_joints)
        
    print("==================================================")
    print("  SIMULATION COMPLETED SUCCESSFULLY!")
    
    # Save coordinates
    folded_data = {
        "sequence": sequence,
        "atoms": peptide.atoms,
        "atomResidues": peptide.atom_residues.tolist(),
        "foldedCoords": curr_coords.tolist()
    }
    out_filename = f"{sequence.lower()}_directional_folded.json"
    with open(out_filename, "w") as f:
        json.dump(folded_data, f)
    print(f" Saved directional structure to {out_filename}")
    print("==================================================")

if __name__ == "__main__":
    seq = "AAAAAAAAAA"
    steps = 300
    lr = 0.15
    if len(sys.argv) > 1:
        seq = sys.argv[1]
    if len(sys.argv) > 2:
        steps = int(sys.argv[2])
    if len(sys.argv) > 3:
        lr = float(sys.argv[3])
    run_differentiable_folding_simulation(sequence=seq, steps=steps, learning_rate=lr)
