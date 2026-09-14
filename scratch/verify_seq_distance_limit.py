import os
import sys
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem

class PeptideAgent:
    def __init__(self, sequence):
        self.sequence = sequence.strip().upper()
        self.mol = Chem.MolFromSequence(self.sequence)
        self.mol = Chem.AddHs(self.mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        params.useRandomCoords = True
        AllChem.EmbedMolecule(self.mol, params)
        AllChem.MMFFOptimizeMolecule(self.mol)
        self.mol = Chem.RemoveHs(self.mol)
        self.n_atoms = self.mol.GetNumAtoms()
        
        self.atoms = []
        for atom in self.mol.GetAtoms():
            info = atom.GetPDBResidueInfo()
            symbol = info.GetName().strip() if info else atom.GetSymbol()
            self.atoms.append({
                "id": atom.GetIdx(),
                "symbol": symbol,
                "element": atom.GetSymbol()
            })
            
        self.bonds = []
        for bond in self.mol.GetBonds():
            self.bonds.append({
                "source": bond.GetBeginAtomIdx(),
                "target": bond.GetEndAtomIdx()
            })
            
        conf = self.mol.GetConformer()
        self.initial_coords = np.zeros((self.n_atoms, 3))
        for i in range(self.n_atoms):
            pos = conf.GetAtomPosition(i)
            self.initial_coords[i] = [pos.x, pos.y, pos.z]
            
        self.atom_residues = []
        self.residue_labels = []
        for i in range(self.n_atoms):
            info = self.mol.GetAtomWithIdx(i).GetPDBResidueInfo()
            if info:
                res_idx = info.GetResidueNumber() - 1
                self.atom_residues.append(res_idx)
                if len(self.residue_labels) <= res_idx:
                    self.residue_labels.append(info.GetResidueName().strip())
            else:
                self.atom_residues.append(0)
                
        self.n_residues = len(set(self.atom_residues))
        
        adj = nx.adjacency_matrix(self.to_networkx_graph()).toarray()
        G_nx = nx.from_numpy_array(adj)
        self.graph_dist = nx.floyd_warshall_numpy(G_nx)
        self.joints = self.identify_dihedral_joints()

        self.atom_types = []
        self.is_sidechain = []
        hydrophobic_res = {"ALA", "ILE", "LEU", "MET", "PHE", "PRO", "VAL", "TRP", "TYR", "THR"}
        for i in range(self.n_atoms):
            symbol = self.atoms[i]["symbol"]
            res_name = self.residue_labels[self.atom_residues[i]].upper()
            if symbol in ("N", "CA", "C", "O", "OXT"):
                self.is_sidechain.append(False)
            else:
                self.is_sidechain.append(True)
            atom_el = self.mol.GetAtomWithIdx(i).GetSymbol()
            if atom_el in ("N", "O", "S"):
                self.atom_types.append("hydrophilic")
            elif res_name in hydrophobic_res and atom_el == "C":
                self.atom_types.append("hydrophobic")
            else:
                self.atom_types.append("neutral")

    def to_networkx_graph(self):
        G = nx.Graph()
        G.add_nodes_from(range(self.n_atoms))
        for bond in self.bonds:
            G.add_edge(bond["source"], bond["target"])
        return G

    def identify_dihedral_joints(self):
        joints = []
        for bond in self.mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            info1 = a1.GetPDBResidueInfo()
            info2 = a2.GetPDBResidueInfo()
            if not info1 or not info2:
                continue
            name1 = info1.GetName().strip()
            name2 = info2.GetName().strip()
            is_phi = (name1 == "N" and name2 == "CA") or (name1 == "CA" and name2 == "N")
            is_psi = (name1 == "CA" and name2 == "C") or (name1 == "C" and name2 == "CA")
            if is_phi or is_psi:
                u = a1.GetIdx()
                v = a2.GetIdx()
                G = self.to_networkx_graph()
                G.remove_edge(u, v)
                components = list(nx.connected_components(G))
                if len(components) == 2:
                    comp1, comp2 = components
                    downstream = comp2 if v in comp2 else comp1
                    joints.append({
                        "u_idx": u,
                        "d_idx": v,
                        "downstream_atoms": list(downstream),
                        "type": "phi" if is_phi else "psi"
                    })
        return joints

def rotate_joint(coords, joint, d_theta):
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

def get_network_metrics(G):
    try:
        if G.number_of_nodes() <= 1:
            return 0.0
        comp = nx.node_connected_component(G, 0)
        G_sub = G.subgraph(comp)
        if G_sub.number_of_nodes() <= 1:
            return 0.0
        L = nx.laplacian_matrix(G_sub, weight='weight').toarray()
        eigenvals = np.linalg.eigvalsh(L)
        return float(eigenvals[1])
    except Exception:
        return 0.0

def run_folding(peptide, max_seq_dist=None):
    # Initial state
    curr_coords = peptide.initial_coords.copy()
    res_arr = np.array(peptide.atom_residues)
    is_polar = np.array([t == "hydrophilic" for t in peptide.atom_types])
    is_hydro = np.array([t == "hydrophobic" for t in peptide.atom_types])
    is_sc = np.array(peptide.is_sidechain)
    u_idx, v_idx = np.triu_indices(peptide.n_atoms, k=1)
    
    joints_by_res = {r: [] for r in range(peptide.n_residues)}
    for joint in peptide.joints:
        r = peptide.atom_residues[joint["d_idx"]]
        joints_by_res[r].append(joint)
        
    inward_order = list(reversed(range(peptide.n_residues)))
    outward_order = list(range(peptide.n_residues))
    
    n_cycles = 10
    candidate_angles = [-0.5, -0.25, -0.1, -0.05, 0.05, 0.1, 0.25, 0.5]
    accumulated_bonds = set()
    p_f_init = 0.0
    
    for cycle in range(n_cycles):
        is_hydro_only_phase = (cycle < 5)
        
        # Inward Sweep
        for res_idx in inward_order:
            active_atoms = np.where(res_arr == res_idx)[0]
            if len(active_atoms) == 0:
                continue
                
            for joint in joints_by_res[res_idx]:
                best_angle_score = -9999.0
                best_angle_coords = None
                best_angle_f = p_f_init
                
                for d_theta in candidate_angles:
                    cand_coords = rotate_joint(curr_coords, joint, d_theta)
                    
                    # Clash check
                    dists = np.linalg.norm(cand_coords[:, None, :] - cand_coords[None, :, :], axis=-1)
                    clash = np.any((dists < 1.85) & (peptide.graph_dist >= 4.0))
                    if clash:
                        continue
                        
                    dists_pep = np.linalg.norm(cand_coords[:, None, :] - cand_coords[None, :, :], axis=-1)
                    
                    G_pep_cand = nx.Graph()
                    G_pep_cand.add_nodes_from(range(peptide.n_atoms))
                    for covalent_bond in peptide.bonds:
                        G_pep_cand.add_edge(int(covalent_bond["source"]), int(covalent_bond["target"]), weight=10.0)
                        
                    for i, p in zip(u_idx, v_idx):
                        if res_arr[i] == res_arr[p]:
                            continue
                            
                        # Apply sequence distance constraint if specified
                        if max_seq_dist is not None:
                            if abs(int(res_arr[i]) - int(res_arr[p])) > max_seq_dist:
                                continue
                                
                        d = dists_pep[i, p]
                        if not is_hydro_only_phase and is_polar[i] and is_polar[p] and d < 4.0:
                            w = 16.0 / (d**2)
                            if is_sc[i] and is_sc[p]:
                                w *= 0.70
                            elif is_sc[i] or is_sc[p]:
                                w *= 0.85
                            G_pep_cand.add_edge(int(i), int(p), weight=float(w))
                        elif is_hydro[i] and is_hydro[p] and d < 5.0:
                            w = 40.0 / (d**2)
                            if is_sc[i] and is_sc[p]:
                                w *= 0.70
                            elif is_sc[i] or is_sc[p]:
                                w *= 0.85
                            G_pep_cand.add_edge(int(i), int(p), weight=float(w))
                            
                    for r_a, r_b in accumulated_bonds:
                        atoms_a = np.where(res_arr == r_a)[0]
                        atoms_b = np.where(res_arr == r_b)[0]
                        for i in atoms_a:
                            for p in atoms_b:
                                d = dists_pep[i, p]
                                if not is_hydro_only_phase and is_polar[i] and is_polar[p]:
                                    w = 16.0 / (d**2)
                                    if is_sc[i] and is_sc[p]:
                                        w *= 0.70
                                    elif is_sc[i] or is_sc[p]:
                                        w *= 0.85
                                    G_pep_cand.add_edge(int(i), int(p), weight=float(w))
                                elif is_hydro[i] and is_hydro[p]:
                                    w = 40.0 / (d**2)
                                    if is_sc[i] and is_sc[p]:
                                        w *= 0.70
                                    elif is_sc[i] or is_sc[p]:
                                        w *= 0.85
                                    G_pep_cand.add_edge(int(i), int(p), weight=float(w))
                                    
                    cand_f = get_network_metrics(G_pep_cand)
                    if cand_f > best_angle_score:
                        best_angle_score = cand_f
                        best_angle_coords = cand_coords
                        best_angle_f = cand_f
                        
                if best_angle_coords is not None:
                    curr_coords = best_angle_coords
                    p_f_init = best_angle_f
                    
                    # Accumulate closest contact (also subject to sequence distance constraint)
                    dists_pep = np.linalg.norm(curr_coords[:, None, :] - curr_coords[None, :, :], axis=-1)
                    best_target_res_idx = None
                    min_d = 999.0
                    for target_res_idx in range(peptide.n_residues):
                        if target_res_idx == res_idx:
                            continue
                        if max_seq_dist is not None:
                            if abs(res_idx - target_res_idx) > max_seq_dist:
                                continue
                        target_atoms = np.where(res_arr == target_res_idx)[0]
                        for i in active_atoms:
                            for p in target_atoms:
                                d = dists_pep[i, p]
                                if is_hydro_only_phase:
                                    if not (is_hydro[i] and is_hydro[p]):
                                        continue
                                if d < min_d:
                                    min_d = d
                                    best_target_res_idx = target_res_idx
                                    
                    if best_target_res_idx is not None and min_d < (5.0 if is_hydro_only_phase else 4.5):
                        r_min = min(res_idx, best_target_res_idx)
                        r_max = max(res_idx, best_target_res_idx)
                        accumulated_bonds.add((r_min, r_max))
                        
        # Outward Sweep
        for res_idx in outward_order:
            active_atoms = np.where(res_arr == res_idx)[0]
            if len(active_atoms) == 0:
                continue
                
            for joint in joints_by_res[res_idx]:
                best_angle_score = -9999.0
                best_angle_coords = None
                best_angle_f = p_f_init
                
                for d_theta in candidate_angles:
                    cand_coords = rotate_joint(curr_coords, joint, d_theta)
                    
                    dists = np.linalg.norm(cand_coords[:, None, :] - cand_coords[None, :, :], axis=-1)
                    clash = np.any((dists < 1.85) & (peptide.graph_dist >= 4.0))
                    if clash:
                        continue
                        
                    dists_pep = np.linalg.norm(cand_coords[:, None, :] - cand_coords[None, :, :], axis=-1)
                    
                    G_pep_cand = nx.Graph()
                    G_pep_cand.add_nodes_from(range(peptide.n_atoms))
                    for covalent_bond in peptide.bonds:
                        G_pep_cand.add_edge(int(covalent_bond["source"]), int(covalent_bond["target"]), weight=10.0)
                        
                    for i, p in zip(u_idx, v_idx):
                        if res_arr[i] == res_arr[p]:
                            continue
                        if max_seq_dist is not None:
                            if abs(int(res_arr[i]) - int(res_arr[p])) > max_seq_dist:
                                continue
                        d = dists_pep[i, p]
                        if not is_hydro_only_phase and is_polar[i] and is_polar[p] and d < 4.0:
                            w = 16.0 / (d**2)
                            if is_sc[i] and is_sc[p]:
                                w *= 0.70
                            elif is_sc[i] or is_sc[p]:
                                w *= 0.85
                            G_pep_cand.add_edge(int(i), int(p), weight=float(w))
                        elif is_hydro[i] and is_hydro[p] and d < 5.0:
                            w = 40.0 / (d**2)
                            if is_sc[i] and is_sc[p]:
                                w *= 0.70
                            elif is_sc[i] or is_sc[p]:
                                w *= 0.85
                            G_pep_cand.add_edge(int(i), int(p), weight=float(w))
                            
                    for r_a, r_b in accumulated_bonds:
                        atoms_a = np.where(res_arr == r_a)[0]
                        atoms_b = np.where(res_arr == r_b)[0]
                        for i in atoms_a:
                            for p in atoms_b:
                                d = dists_pep[i, p]
                                if not is_hydro_only_phase and is_polar[i] and is_polar[p]:
                                    w = 16.0 / (d**2)
                                    if is_sc[i] and is_sc[p]:
                                        w *= 0.70
                                    elif is_sc[i] or is_sc[p]:
                                        w *= 0.85
                                    G_pep_cand.add_edge(int(i), int(p), weight=float(w))
                                elif is_hydro[i] and is_hydro[p]:
                                    w = 40.0 / (d**2)
                                    if is_sc[i] and is_sc[p]:
                                        w *= 0.70
                                    elif is_sc[i] or is_sc[p]:
                                        w *= 0.85
                                    G_pep_cand.add_edge(int(i), int(p), weight=float(w))
                                    
                    cand_f = get_network_metrics(G_pep_cand)
                    if cand_f > best_angle_score:
                        best_angle_score = cand_f
                        best_angle_coords = cand_coords
                        best_angle_f = cand_f
                        
                if best_angle_coords is not None:
                    curr_coords = best_angle_coords
                    p_f_init = best_angle_f
                    
                    dists_pep = np.linalg.norm(curr_coords[:, None, :] - curr_coords[None, :, :], axis=-1)
                    best_target_res_idx = None
                    min_d = 999.0
                    for target_res_idx in range(peptide.n_residues):
                        if target_res_idx == res_idx:
                            continue
                        if max_seq_dist is not None:
                            if abs(res_idx - target_res_idx) > max_seq_dist:
                                continue
                        target_atoms = np.where(res_arr == target_res_idx)[0]
                        for i in active_atoms:
                            for p in target_atoms:
                                d = dists_pep[i, p]
                                if is_hydro_only_phase:
                                    if not (is_hydro[i] and is_hydro[p]):
                                        continue
                                if d < min_d:
                                    min_d = d
                                    best_target_res_idx = target_res_idx
                                    
                    if best_target_res_idx is not None and min_d < (5.0 if is_hydro_only_phase else 4.5):
                        r_min = min(res_idx, best_target_res_idx)
                        r_max = max(res_idx, best_target_res_idx)
                        accumulated_bonds.add((r_min, r_max))
                        
    final_rg = np.sqrt(np.mean(np.sum((curr_coords - np.mean(curr_coords, axis=0))**2, axis=1)))
    return final_rg, p_f_init, list(accumulated_bonds)

def main():
    print("==================================================================")
    print("      Verification of Sequence Distance Limitation on CLN025")
    print("==================================================================")
    
    peptide = PeptideAgent("YYDPETGTWY")
    print(f"Peptide length: {peptide.n_residues} residues ({peptide.n_atoms} heavy atoms)")
    print("Running simulations under 3 scenarios (3 trials each to check stability)...")
    
    scenarios = {
        "A. No sequence distance limit (Control)": None,
        "B. Sequence distance <= 1 (Adjacent only)": 1,
        "C. Sequence distance <= 2 (Adjacent + Next-adjacent)": 2
    }
    
    for name, limit in scenarios.items():
        print(f"\nScenario: {name}")
        rgs = []
        fiedlers = []
        bonds_list = []
        for trial in range(1, 4):
            rg, fiedler, bonds = run_folding(peptide, max_seq_dist=limit)
            rgs.append(rg)
            fiedlers.append(fiedler)
            bonds_list.append(bonds)
            print(f"  Trial {trial}: Rg = {rg:.3f} A | Peptide Fiedler = {fiedler:.6f} | Accumulated Bonds = {bonds}")
            
        print(f"  Average Rg: {np.mean(rgs):.3f} A (min: {np.min(rgs):.3f} A)")
        print(f"  Average Peptide Fiedler: {np.mean(fiedlers):.6f}")

if __name__ == '__main__':
    main()
