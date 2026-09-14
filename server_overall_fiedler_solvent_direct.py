import os
import random
import numpy as np
import networkx as nx
from flask import Flask, request, jsonify
from flask_cors import CORS
from rdkit import Chem
from rdkit.Chem import AllChem

app = Flask(__name__)
CORS(app)

class PeptideAgent:
    def __init__(self, sequence):
        self.sequence = sequence.strip().upper()
        self.mol = Chem.MolFromSequence(self.sequence)
        if self.mol is None:
            raise ValueError(f"Invalid peptide sequence: {self.sequence}")
        
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

def generate_extended_coords(peptide):
    coords = peptide.initial_coords.copy()
    for joint in peptide.joints:
        u_idx = joint["u_idx"]
        d_idx = joint["d_idx"]
        coords = rotate_joint(coords, joint, np.pi)
    return coords

def center_coordinates(coords):
    if coords is None or len(coords) == 0:
        return []
    coords_np = np.array(coords)
    centered = coords_np - np.mean(coords_np, axis=0)
    return centered.tolist()

def center_coordinates_np(coords):
    return coords - np.mean(coords, axis=0)

def generate_h2o_coords(pos_O):
    pos_O = np.array(pos_O)
    v1 = np.random.normal(size=3)
    v1_len = np.linalg.norm(v1)
    v1 = v1 / v1_len if v1_len > 1e-5 else np.array([1.0, 0.0, 0.0])
    v2 = np.random.normal(size=3)
    v2 = v2 - np.dot(v2, v1) * v1
    v2_len = np.linalg.norm(v2)
    v2 = v2 / v2_len if v2_len > 1e-5 else np.cross(v1, [0.0, 1.0, 0.0])
    
    angle_rad = np.radians(52.25)
    h1_dir = np.cos(angle_rad) * v1 + np.sin(angle_rad) * v2
    h2_dir = np.cos(angle_rad) * v1 - np.sin(angle_rad) * v2
    return [pos_O.tolist(), (pos_O + 0.96 * h1_dir).tolist(), (pos_O + 0.96 * h2_dir).tolist()]

def update_solvent_physics(peptide_coords, water_coords, atom_types):
    n_pep = len(peptide_coords)
    n_wat = len(water_coords)
    if n_wat == 0:
        return water_coords
    pep_arr = np.array(peptide_coords)
    wat_arr = np.array([w[0] for w in water_coords])
    
    d_pw = wat_arr[:, None, :] - pep_arr[None, :, :]
    dist_pw = np.maximum(np.linalg.norm(d_pw, axis=-1), 0.5)
    
    F_polar = -1.5 * d_pw / (dist_pw**3)[:, :, None]
    F_hydro = 4.5 * d_pw / (dist_pw**5)[:, :, None]
    
    is_polar = np.array([t == "hydrophilic" for t in atom_types])
    is_hydro = np.array([t == "hydrophobic" for t in atom_types])
    
    F_pw_polar = np.sum(F_polar[:, is_polar, :], axis=1) if np.any(is_polar) else np.zeros((n_wat, 3))
    hydro_mask = (dist_pw < 4.5) & is_hydro[None, :]
    F_hydro_masked = np.zeros_like(F_hydro)
    F_hydro_masked[hydro_mask] = F_hydro[hydro_mask]
    F_pw_hydro = np.sum(F_hydro_masked, axis=1)
    F_pw = F_pw_polar + F_pw_hydro
    
    d_ww = wat_arr[None, :, :] - wat_arr[:, None, :]
    dist_ww = np.maximum(np.linalg.norm(d_ww, axis=-1), 0.5)
    F_ww_terms = 1.0 * d_ww / (dist_ww**3)[:, :, None]
    F_ww_terms[np.arange(n_wat), np.arange(n_wat), :] = 0.0
    F_ww_terms[dist_ww >= 5.0] = 0.0
    F_ww = np.sum(F_ww_terms, axis=1)
    
    F_total = F_pw + F_ww
    F_len = np.linalg.norm(F_total, axis=-1, keepdims=True)
    F_total = np.where(F_len > 4.0, F_total * (4.0 / np.maximum(F_len, 1e-5)), F_total)
    F_len_capped = np.linalg.norm(F_total, axis=-1, keepdims=True)
    
    new_pos_O = wat_arr + 0.15 * F_total
    
    dir_u = np.zeros_like(F_total)
    valid_force = F_len_capped[:, 0] > 1e-3
    dir_u[valid_force] = F_total[valid_force] / F_len_capped[valid_force]
    dir_u[~valid_force] = np.array([1.0, 0.0, 0.0])
    
    v_ref = np.zeros_like(dir_u)
    use_x = np.abs(dir_u[:, 0]) < 0.9
    v_ref[use_x] = [1.0, 0.0, 0.0]
    v_ref[~use_x] = [0.0, 1.0, 0.0]
    
    dir_w = np.cross(dir_u, v_ref)
    dir_w = dir_w / np.maximum(np.linalg.norm(dir_w, axis=-1, keepdims=True), 1e-5)
    
    angle_rad = np.radians(52.25)
    h1_dir = np.cos(angle_rad) * dir_u + np.sin(angle_rad) * dir_w
    h2_dir = np.cos(angle_rad) * dir_u - np.sin(angle_rad) * dir_w
    
    new_waters = []
    for u in range(n_wat):
        new_waters.append([
            new_pos_O[u].tolist(),
            (new_pos_O[u] + 0.96 * h1_dir[u]).tolist(),
            (new_pos_O[u] + 0.96 * h2_dir[u]).tolist()
        ])
    return new_waters

def build_active_bond_graph(peptide_coords, water_coords, bonds, atom_types, is_sidechain, atom_residues, contact_threshold_polar=4.0, contact_threshold_hydrophobic=4.5):
    G = nx.Graph()
    n_pep = len(peptide_coords)
    n_wat = len(water_coords)
    n_total = n_pep + n_wat
    G.add_nodes_from(range(n_total))
    
    for bond in bonds:
        G.add_edge(int(bond["source"]), int(bond["target"]), weight=10.0)
        
    pep_arr = np.array(peptide_coords)
    dists_pep = np.linalg.norm(pep_arr[:, None, :] - pep_arr[None, :, :], axis=-1)
    is_polar = np.array([t == "hydrophilic" for t in atom_types])
    is_hydro = np.array([t == "hydrophobic" for t in atom_types])
    is_sc = np.array(is_sidechain)
    res_arr = np.array(atom_residues)
    
    u_idx, v_idx = np.triu_indices(n_pep, k=1)
    for i, j in zip(u_idx, v_idx):
        if res_arr[i] == res_arr[j]:
            continue
        d = dists_pep[i, j]
        if is_polar[i] and is_polar[j] and d < contact_threshold_polar:
            w = 16.0 / (d**2)
            G.add_edge(int(i), int(j), weight=float(w))
        elif is_hydro[i] and is_hydro[j] and d < contact_threshold_hydrophobic:
            w = 40.0 / (d**2)
            G.add_edge(int(i), int(j), weight=float(w))
        
    wat_arr = np.array([w[0] for w in water_coords])
    water_candidates = {u: [] for u in range(n_wat)}
    
    if n_wat > 0:
        dists_pw = np.linalg.norm(pep_arr[:, None, :] - wat_arr[None, :, :], axis=-1)
        polar_mask = is_polar[:, None] & (dists_pw < 3.0)
        pep_indices, wat_indices = np.where(polar_mask)
        for idx in range(len(pep_indices)):
            i = int(pep_indices[idx])
            u = int(wat_indices[idx])
            water_candidates[u].append((i, float(dists_pw[i, u])))
                    
        dists_ww = np.linalg.norm(wat_arr[:, None, :] - wat_arr[None, :, :], axis=-1)
        ww_mask = (dists_ww < 3.5)
        np.fill_diagonal(ww_mask, False)
        u_indices, v_indices = np.where(ww_mask)
        for idx in range(len(u_indices)):
            u = int(u_indices[idx])
            v = int(v_indices[idx])
            water_candidates[u].append((int(n_pep + v), float(dists_ww[u, v])))
                    
    active_water_bonds = []
    active_water_peptide_bonds = []
    
    for u in range(n_wat):
        candidates = sorted(water_candidates[u], key=lambda x: x[1])
        for neighbor, d in candidates[:4]:
            w = max(0.1, 3.5 - d) if neighbor >= n_pep else max(0.1, 3.0 - d)
            G.add_edge(int(n_pep + u), int(neighbor), weight=float(w))
            if neighbor >= n_pep:
                v = neighbor - n_pep
                if u < v:
                    pos_v_O = np.array(water_coords[v][0])
                    pos_u_H1 = np.array(water_coords[u][1])
                    pos_u_H2 = np.array(water_coords[u][2])
                    h_idx = 1 if np.linalg.norm(pos_u_H1 - pos_v_O) < np.linalg.norm(pos_u_H2 - pos_v_O) else 2
                    active_water_bonds.append([int(u), int(v), int(h_idx)])
            else:
                pos_p = np.array(peptide_coords[neighbor])
                pos_w_O = np.array(water_coords[u][0])
                pos_w_H1 = np.array(water_coords[u][1])
                pos_w_H2 = np.array(water_coords[u][2])
                d_O = np.linalg.norm(pos_w_O - pos_p)
                d_H1 = np.linalg.norm(pos_w_H1 - pos_p)
                d_H2 = np.linalg.norm(pos_w_H2 - pos_p)
                min_d = min(d_O, d_H1, d_H2)
                t = 0 if min_d == d_O else (1 if min_d == d_H1 else 2)
                active_water_peptide_bonds.append([int(u), int(neighbor), int(t)])
                
    active_hbonds = []
    active_sc_bonds = []
    active_bb_sc_bonds = []
    
    for i, j in G.edges():
        if i < n_pep and j < n_pep:
            is_covalent = False
            for bond in bonds:
                if (bond["source"] == i and bond["target"] == j) or (bond["source"] == j and bond["target"] == i):
                    is_covalent = True
                    break
            if not is_covalent:
                d = dists_pep[i, j]
                if is_sidechain[i] and is_sidechain[j]:
                    if d < 7.0:
                        active_sc_bonds.append([int(i), int(j)])
                elif (not is_sidechain[i]) and (not is_sidechain[j]):
                    if d < 6.0:
                        active_hbonds.append([int(i), int(j)])
                else:
                    if d < 6.5:
                        active_bb_sc_bonds.append([int(i), int(j)])
                
    return G, active_hbonds, active_sc_bonds, active_bb_sc_bonds, active_water_bonds, active_water_peptide_bonds

def get_network_metrics(G):
    try:
        if G.number_of_nodes() <= 1:
            return 0.0, 0.0
        comp = nx.node_connected_component(G, 0)
        G_sub = G.subgraph(comp)
        if G_sub.number_of_nodes() <= 1:
            return 0.0, 0.0
        L = nx.laplacian_matrix(G_sub, weight='weight').toarray()
        eigenvals = np.linalg.eigvalsh(L)
        fiedler = float(eigenvals[1])
        nz_eigenvals = eigenvals[eigenvals > 1e-5]
        spectral_entropy = float(-np.sum((nz_eigenvals / np.sum(nz_eigenvals)) * np.log(nz_eigenvals / np.sum(nz_eigenvals)))) if len(nz_eigenvals) > 0 else 0.0
    except Exception:
        fiedler = 0.0
        spectral_entropy = 0.0
    return fiedler, spectral_entropy

def randomize_dihedrals(coords, joints):
    curr_coords = coords.copy()
    for joint in joints:
        curr_coords = rotate_joint(curr_coords, joint, random.uniform(-0.5, 0.5))
    return curr_coords

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

def connect_water_network(peptide_coords, water_coords, threshold=3.5):
    if len(water_coords) <= 1:
        return water_coords
    iteration = 0
    while iteration < 30:
        n_wat = len(water_coords)
        wat_arr = np.array([w[0] for w in water_coords])
        dists_ww = np.linalg.norm(wat_arr[:, None, :] - wat_arr[None, :, :], axis=-1)
        G_w = nx.Graph()
        G_w.add_nodes_from(range(n_wat))
        u_idx, v_idx = np.where(dists_ww < threshold)
        for u, v in zip(u_idx, v_idx):
            if u < v:
                G_w.add_edge(u, v)
        components = list(nx.connected_components(G_w))
        if len(components) <= 1:
            break
        best_d = float('inf')
        best_pair = None
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                nodes_i = list(components[i])
                nodes_j = list(components[j])
                sub_dists = dists_ww[np.ix_(nodes_i, nodes_j)]
                min_idx = np.unravel_index(np.argmin(sub_dists), sub_dists.shape)
                d = sub_dists[min_idx]
                if d < best_d:
                    best_d = d
                    best_pair = (nodes_i[min_idx[0]], nodes_j[min_idx[1]])
        if best_pair is not None:
            u, v = best_pair
            pos_O = 0.5 * (np.array(water_coords[u][0]) + np.array(water_coords[v][0]))
            water_coords.append(generate_h2o_coords(pos_O))
        else:
            break
        iteration += 1
    return water_coords

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
        print(f"  Starting Solvent-Coupled OVERALL DIRECT Fiedler Folding")
        print(f"  Sequence: {sequence}")
        print(f"========================================================")
        
        peptide = PeptideAgent(sequence)
        extended_coords = generate_extended_coords(peptide)
        unfolded_coords = center_coordinates_np(extended_coords)
        
        unfolded_waters = []
        for i in range(peptide.n_atoms):
            pos_atom = unfolded_coords[i]
            for _ in range(2):
                direction = np.random.normal(size=3)
                direction_len = np.linalg.norm(direction)
                direction = direction / direction_len if direction_len > 1e-5 else np.array([1.0, 0.0, 0.0])
                unfolded_waters.append(generate_h2o_coords(pos_atom + 2.8 * direction))
        unfolded_waters = connect_water_network(unfolded_coords, unfolded_waters)
                
        ideal_fiedler = 0.450
        ideal_spectral_entropy = 2.85
        has_folded_target = False
        target_hb, target_sc, target_bb_sc, target_wb, target_wp = [], [], [], [], []
        
        pdb_map = {
            "GYDPETGTWG": "1UAO.pdb",
            "YYDPETGTWY": "2H2D.pdb",
            "NLYIQWLKDGGPSSGRPPPS": "1L2Y.pdb"
        }
        seq_upper = sequence.strip().upper()
        
        raw_folded = None
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
                        
                        ideal_waters = []
                        for i in range(peptide.n_atoms):
                            pos_atom = folded_target[i]
                            for _ in range(2):
                                direction = np.random.normal(size=3)
                                direction_len = np.linalg.norm(direction)
                                direction = direction / direction_len if direction_len > 1e-5 else np.array([1.0, 0.0, 0.0])
                                ideal_waters.append(generate_h2o_coords(pos_atom + 2.8 * direction))
                        for _ in range(10):
                            ideal_waters = update_solvent_physics(folded_target, ideal_waters, peptide.atom_types)
                        ideal_waters = connect_water_network(folded_target, ideal_waters)
                            
                        target_G, target_hb, target_sc, target_bb_sc, target_wb, target_wp = build_active_bond_graph(
                            folded_target, ideal_waters, peptide.bonds, peptide.atom_types, peptide.is_sidechain, peptide.atom_residues
                        )
                        ideal_fiedler, ideal_spectral_entropy = get_network_metrics(target_G)
                        has_folded_target = True
                except Exception as e:
                    print(f"Error reading PDB file {pdb_path}: {e}")
                    
        target_rg = np.sqrt(np.mean(np.sum((folded_target - np.mean(folded_target, axis=0))**2, axis=1))) if has_folded_target else 0.0
        print(f"Target Ideal Overall Fiedler: {ideal_fiedler:.6f} | Target Radius of Gyration: {target_rg:.3f} Å")
        
        trajectory = []
        final_coords = center_coordinates_np(peptide.initial_coords).copy()
        final_waters = []
        
        best_score = -999999.0
        best_fiedler = -1.0
        best_traj = []
        best_coords = None
        best_waters = None
        
        for trial in range(1, 4):
            print(f"--- Trial {trial}/3 starting ---")
            curr_coords = center_coordinates_np(randomize_dihedrals(peptide.initial_coords, peptide.joints))
            
            water_coords = []
            for i in range(peptide.n_atoms):
                pos_atom = curr_coords[i]
                for _ in range(2):
                    direction = np.random.normal(size=3)
                    direction_len = np.linalg.norm(direction)
                    direction = direction / direction_len if direction_len > 1e-5 else np.array([1.0, 0.0, 0.0])
                    water_coords.append(generate_h2o_coords(pos_atom + 2.8 * direction))
            water_coords = connect_water_network(curr_coords, water_coords)
            for _ in range(10):
                water_coords = update_solvent_physics(curr_coords, water_coords, peptide.atom_types)
                
            G_total_init, act_hb, act_sc, act_bb_sc, act_wb, act_wp = build_active_bond_graph(
                curr_coords, water_coords, peptide.bonds, peptide.atom_types, peptide.is_sidechain, peptide.atom_residues
            )
            overall_f_init, overall_e_init = get_network_metrics(G_total_init)
            
            trial_traj = [{
                "step": 0,
                "activeResidue": -1,
                "coords": curr_coords.tolist(),
                "waterCoords": water_coords,
                "activeHBonds": act_hb,
                "activeSidechainBonds": act_sc,
                "activeBBScBonds": act_bb_sc,
                "activeWaterBonds": act_wb,
                "activeWaterPeptideBonds": act_wp,
                "overallFiedler": overall_f_init,
                "overallEntropy": overall_e_init,
                "peptideFiedler": overall_f_init,
                "peptideEntropy": overall_e_init,
                "waterFiedler": 0.0,
                "waterEntropy": 0.0
            }]
            
            res_arr = np.array(peptide.atom_residues)
            is_polar = np.array([t == "hydrophilic" for t in peptide.atom_types])
            is_hydro = np.array([t == "hydrophobic" for t in peptide.atom_types])
            hydro_indices = np.where(is_hydro)[0]
            is_sc = np.array(peptide.is_sidechain)
            
            joints_by_res = {r: [] for r in range(peptide.n_residues)}
            for joint in peptide.joints:
                r = peptide.atom_residues[joint["d_idx"]]
                joints_by_res[r].append(joint)
                
            inward_order = list(reversed(range(peptide.n_residues)))
            outward_order = list(range(peptide.n_residues))
            
            curr_score = overall_f_init
            
            step_counter = 0
            n_cycles = 10
            candidate_angles = [-0.25, -0.05, 0.05, 0.25]
            
            for cycle in range(n_cycles):
                # 1. Inward Sweep
                for res_idx in inward_order:
                    if len(np.where(res_arr == res_idx)[0]) == 0:
                        continue
                    for joint in joints_by_res[res_idx]:
                        step_counter += 1
                        best_angle_score = -9999.0
                        best_angle_coords = None
                        best_angle_waters = None
                        best_angle_f = overall_f_init
                        
                        for d_theta in candidate_angles:
                            cand_coords = rotate_joint(curr_coords, joint, d_theta)
                            dists = np.linalg.norm(cand_coords[:, None, :] - cand_coords[None, :, :], axis=-1)
                            clash = np.any((dists < 1.85) & (peptide.graph_dist >= 4.0))
                            if clash:
                                continue
                                
                            cand_waters = water_coords
                            for _ in range(2):
                                cand_waters = update_solvent_physics(cand_coords, cand_waters, peptide.atom_types)
                                
                            G_cand, _, _, _, _, _ = build_active_bond_graph(
                                cand_coords, cand_waters, peptide.bonds, peptide.atom_types, peptide.is_sidechain, peptide.atom_residues
                            )
                            cand_f, _ = get_network_metrics(G_cand)
                            if cand_f > best_angle_score:
                                best_angle_score = cand_f
                                best_angle_coords = cand_coords
                                best_angle_waters = cand_waters
                                best_angle_f = cand_f
                                
                        if best_angle_coords is not None:
                            curr_coords = best_angle_coords
                            water_coords = best_angle_waters
                            overall_f_init = best_angle_f
                            
                            curr_score = overall_f_init
                            
                        if step_counter % 8 == 0:
                            for _ in range(5):
                                water_coords = update_solvent_physics(curr_coords, water_coords, peptide.atom_types)
                            G_cand, cand_hb, cand_sc, cand_bb_sc, cand_wb, cand_wp = build_active_bond_graph(
                                curr_coords, water_coords, peptide.bonds, peptide.atom_types, peptide.is_sidechain, peptide.atom_residues
                            )
                            overall_f_acc, overall_e_acc = get_network_metrics(G_cand)
                            trial_traj.append({
                                "step": len(trial_traj),
                                "activeResidue": res_idx,
                                "coords": curr_coords.tolist(),
                                "waterCoords": water_coords,
                                "activeHBonds": cand_hb,
                                "activeSidechainBonds": cand_sc,
                                "activeBBScBonds": cand_bb_sc,
                                "activeWaterBonds": cand_wb,
                                "activeWaterPeptideBonds": cand_wp,
                                "overallFiedler": overall_f_acc,
                                "overallEntropy": overall_e_acc,
                                "peptideFiedler": overall_f_acc,
                                "peptideEntropy": overall_e_acc,
                                "waterFiedler": 0.0,
                                "waterEntropy": 0.0
                            })
                            
                # 2. Outward Sweep
                for res_idx in outward_order:
                    if len(np.where(res_arr == res_idx)[0]) == 0:
                        continue
                    for joint in joints_by_res[res_idx]:
                        step_counter += 1
                        best_angle_score = -9999.0
                        best_angle_coords = None
                        best_angle_waters = None
                        best_angle_f = overall_f_init
                        
                        for d_theta in candidate_angles:
                            cand_coords = rotate_joint(curr_coords, joint, d_theta)
                            dists = np.linalg.norm(cand_coords[:, None, :] - cand_coords[None, :, :], axis=-1)
                            clash = np.any((dists < 1.85) & (peptide.graph_dist >= 4.0))
                            if clash:
                                continue
                                
                            cand_waters = water_coords
                            for _ in range(2):
                                cand_waters = update_solvent_physics(cand_coords, cand_waters, peptide.atom_types)
                                
                            G_cand, _, _, _, _, _ = build_active_bond_graph(
                                cand_coords, cand_waters, peptide.bonds, peptide.atom_types, peptide.is_sidechain, peptide.atom_residues
                            )
                            cand_f, _ = get_network_metrics(G_cand)
                            if cand_f > best_angle_score:
                                best_angle_score = cand_f
                                best_angle_coords = cand_coords
                                best_angle_waters = cand_waters
                                best_angle_f = cand_f
                                
                        if best_angle_coords is not None:
                            curr_coords = best_angle_coords
                            water_coords = best_angle_waters
                            overall_f_init = best_angle_f
                            
                            curr_score = overall_f_init
                            
                        if step_counter % 8 == 0:
                            for _ in range(5):
                                water_coords = update_solvent_physics(curr_coords, water_coords, peptide.atom_types)
                            G_cand, cand_hb, cand_sc, cand_bb_sc, cand_wb, cand_wp = build_active_bond_graph(
                                curr_coords, water_coords, peptide.bonds, peptide.atom_types, peptide.is_sidechain, peptide.atom_residues
                            )
                            overall_f_acc, overall_e_acc = get_network_metrics(G_cand)
                            trial_traj.append({
                                "step": len(trial_traj),
                                "activeResidue": res_idx,
                                "coords": curr_coords.tolist(),
                                "waterCoords": water_coords,
                                "activeHBonds": cand_hb,
                                "activeSidechainBonds": cand_sc,
                                "activeBBScBonds": cand_bb_sc,
                                "activeWaterBonds": cand_wb,
                                "activeWaterPeptideBonds": cand_wp,
                                "overallFiedler": overall_f_acc,
                                "overallEntropy": overall_e_acc,
                                "peptideFiedler": overall_f_acc,
                                "peptideEntropy": overall_e_acc,
                                "waterFiedler": 0.0,
                                "waterEntropy": 0.0
                            })
                            
            final_fiedler = trial_traj[-1]["overallFiedler"] if trial_traj else overall_f_init
            final_rg = np.sqrt(np.mean(np.sum((curr_coords - np.mean(curr_coords, axis=0))**2, axis=1)))
            print(f"  Trial {trial} completed. Final Coupled Fiedler: {final_fiedler:.6f} | Final Rg: {final_rg:.3f} Å | Score: {curr_score:.3f}")
            
            if curr_score > best_score:
                best_score = curr_score
                best_fiedler = final_fiedler
                best_traj = trial_traj
                best_coords = curr_coords.copy()
                best_waters = water_coords
                
        final_rg_sel = np.sqrt(np.mean(np.sum((best_coords - np.mean(best_coords, axis=0))**2, axis=1)))
        print(f"✓ Completed all 3 trials. Selecting the best trial conformation (Coupled Fiedler = {best_fiedler:.6f}, Rg = {final_rg_sel:.3f} Å, Score = {best_score:.3f}).")
        trajectory = best_traj
        final_coords = best_coords
        final_waters = best_waters
            
        final_coords_centered = center_coordinates_np(final_coords)
        displacement = final_coords_centered - final_coords
        centered_final_waters = []
        for w in final_waters:
            centered_final_waters.append([
                (np.array(w[0]) + displacement[0]).tolist(),
                (np.array(w[1]) + displacement[0]).tolist(),
                (np.array(w[2]) + displacement[0]).tolist()
            ])
            
        for frame in trajectory:
            frm_pep = np.array(frame["coords"])
            frm_pep_cen = center_coordinates_np(frm_pep)
            disp = frm_pep_cen - frm_pep
            frame["coords"] = frm_pep_cen.tolist()
            
            frm_wat = frame["waterCoords"]
            cen_wat = []
            for w in frm_wat:
                cen_wat.append([
                    (np.array(w[0]) + disp[0]).tolist(),
                    (np.array(w[1]) + disp[0]).tolist(),
                    (np.array(w[2]) + disp[0]).tolist()
                ])
            frame["waterCoords"] = cen_wat
            
        if has_folded_target and raw_folded is not None:
            unfolded_out = center_coordinates(raw_folded)
            unfolded_waters_out = ideal_waters
        else:
            unfolded_out = center_coordinates(unfolded_coords)
            unfolded_waters_out = unfolded_waters
            
        G_final, _, _, _, _, _ = build_active_bond_graph(
            final_coords_centered, centered_final_waters, peptide.bonds, peptide.atom_types, peptide.is_sidechain, peptide.atom_residues
        )
        f_total_final, e_total_final = get_network_metrics(G_final)
        p_G_f = G_final.subgraph(range(peptide.n_atoms))
        p_f_ref, p_e_ref = get_network_metrics(p_G_f)
        
        print("\n========================================================")
        print(f"  FINAL NETWORK GRAPH SIMULATION RESULTS (SOLVENT-DIRECT)")
        print(f"--------------------------------------------------------")
        print(f"  UNFOLDED STATE (Step 0):")
        print(f"    Overall Fiedler: {overall_f_init:.6f} | Overall Spectral Entropy: {overall_e_init:.4f}")
        print(f"  SIMULATED FOLDED STATE (Final Step):")
        print(f"    Overall Fiedler: {f_total_final:.6f} | Overall Spectral Entropy: {e_total_final:.4f}")
        print(f"    (Peptide Ref):   {p_f_ref:.6f}")
        print("========================================================\n")
        
        return jsonify({
            "status": "success",
            "sequence": sequence,
            "atoms": peptide.atoms,
            "bonds": peptide.bonds,
            "unfoldedCoords": unfolded_out,
            "unfoldedWaters": unfolded_waters_out,
            "foldedCoords": final_coords_centered.tolist(),
            "foldedWaters": centered_final_waters,
            "atomResidues": peptide.atom_residues,
            "residueLabels": peptide.residue_labels,
            "trajectory": trajectory,
            "targetFiedler": ideal_fiedler,
            "targetEntropy": ideal_spectral_entropy,
            "hasFoldedTarget": has_folded_target,
            "targetHBonds": target_hb,
            "targetSidechainBonds": target_sc,
            "targetBBScBonds": target_bb_sc,
            "targetWaterBonds": target_wb,
            "targetWaterPeptideBonds": target_wp
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Simulation Error: {e}")
        return jsonify({"status": "error", "error": str(e)})

@app.route('/api/log', methods=['POST'])
def client_log():
    data = request.json or {}
    level = data.get("level", "INFO")
    message = data.get("message", "")
    print(f"[CLIENT {level}] {message}", flush=True)
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5009)
