import sys
import os
import time
import json
import math
import threading
import numpy as np
from flask import Flask, jsonify, request, Response

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem

app = Flask(__name__)

# CORS middleware for local development
@app.after_request
def add_cors_headers(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

class LipidSimulation:
    def __init__(self, sequence, count=3, initial_layout="dispersed"):
        self.lock = threading.Lock()
        self.sequence = sequence
        self.count = max(1, min(5, int(count))) # Clamp between 1 and 5
        self.initial_layout = initial_layout
        self.step_counter = 0
        self.logs = []
        
        # 1. Parse single molecule from SMILES
        mol_single = self.parse_molecule(sequence)
        
        # 2. Generate 3D coordinates for single copy
        self.initialize_single_molecule(mol_single)
        
        # 3. Extract single copy topology
        self.extract_single_topology()
        
        # 4. Replicate and position copies
        self.replicate_molecules()
        self.n_residues = self.count
        self.residue_labels = [f"Lipid_{i+1}" for i in range(self.count)]
        
        # 5. Extract rotatable status for bonds
        rotatable_info = {}
        for j in self.joints:
            u, d = j["u_idx"], j["d_idx"]
            r = self.atom_residues.get(d, 0)
            rotatable_info[(u, d)] = r
            rotatable_info[(d, u)] = r
            
        for b in self.bonds:
            pair = (b["source"], b["target"])
            if pair in rotatable_info:
                b["is_rotatable"] = True
                b["residue"] = rotatable_info[pair]
            else:
                b["is_rotatable"] = False
                
        # 6. Initialize water belt
        self.initialize_waters()
        
        # 7. Default physics parameters
        self.params = {
            "temperature": 0.20,
            "isPlaying": True,
            "explicitWaterEnabled": True,
            "waterWaterStrength": 0.65,
            "waterPeptideStrength": 0.75,
            "bulkSolventStrength": 0.45,
            "waterBeltContraction": 0.0,
            "watersPerAtom": 3, # Lowered for multi-molecule performance
            "kCapture": 0.70,
            "entropyBias": 1.2,
            "globalWaterBeltContraction": 0.0,
            "autoContractEnabled": True
        }
        
        self.hydrophobic_exposure = 0.0
        self.water_entropies = []
        self.water_weights = []
        self.information_capture_index = 0.0
        self.active_information_paths = []
        self.hydrophobic_contacts = 0
        
        self.logs.append(f"System: Initialized simulation with {self.count} copy(ies) of SMILES: {self.sequence}")
        self.logs.append(f"System: Combined system contains {self.n_atoms} heavy atoms and {len(self.waters)} water molecules.")

    def parse_molecule(self, input_str):
        mol = Chem.MolFromSmiles(input_str.strip())
        if mol:
            return mol
        # Fallback to cap-less sequence if they typed a string of amino acids
        mol = Chem.MolFromSequence(input_str.strip())
        if mol:
            return mol
        raise ValueError("Could not parse structure. Please provide a valid SMILES string.")

    def initialize_single_molecule(self, mol):
        mol_hs = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        embed_status = AllChem.EmbedMolecule(mol_hs, params)
        if embed_status == -1:
            AllChem.EmbedMolecule(mol_hs, useRandomCoords=True)
        try:
            AllChem.MMFFOptimizeMolecule(mol_hs, maxIters=500)
        except Exception:
            pass
        self.mol_heavy_single = Chem.RemoveHs(mol_hs)
        self.n_atoms_single = self.mol_heavy_single.GetNumAtoms()
        
        conf = self.mol_heavy_single.GetConformer()
        self.coords_single = np.zeros((self.n_atoms_single, 3))
        for i in range(self.n_atoms_single):
            pos = conf.GetAtomPosition(i)
            self.coords_single[i] = [pos.x, pos.y, pos.z]
            
        # Center around center of mass
        com = np.mean(self.coords_single, axis=0)
        self.coords_single -= com
        
        # Orient the single molecule so that the hydrophilic headgroup points to +X (outward)
        # and the hydrophobic tail points to -X (inward)
        polar_indices = [idx for idx in range(self.n_atoms_single) if self.mol_heavy_single.GetAtomWithIdx(idx).GetSymbol() in ('O', 'N', 'P', 'S')]
        hydrophobic_indices = [idx for idx in range(self.n_atoms_single) if self.mol_heavy_single.GetAtomWithIdx(idx).GetSymbol() == 'C']
        
        if polar_indices and hydrophobic_indices:
            com_polar = np.mean(self.coords_single[polar_indices], axis=0)
            com_hydrophobic = np.mean(self.coords_single[hydrophobic_indices], axis=0)
            V = com_polar - com_hydrophobic
            # Rotate in XZ plane
            phi = np.arctan2(V[2], V[0])
            cos_phi = np.cos(-phi)
            sin_phi = np.sin(-phi)
            rot_y = np.array([
                [cos_phi, 0.0, sin_phi],
                [0.0, 1.0, 0.0],
                [-sin_phi, 0.0, cos_phi]
            ])
            self.coords_single = np.dot(self.coords_single, rot_y.T)

    def extract_single_topology(self):
        self.atoms_single = []
        self.bonds_single = []
        self.joints_single = []
        
        # Atoms classification
        for i in range(self.n_atoms_single):
            atom = self.mol_heavy_single.GetAtomWithIdx(i)
            symbol = atom.GetSymbol()
            
            # Electronegative or charge-carrying atoms are polar
            is_polar = symbol in ('O', 'N', 'P', 'S')
            h_bond = "none"
            if is_polar:
                h_bond = "acceptor"
                if atom.GetTotalNumHs() > 0:
                    h_bond = "both"
            
            hydrophobicity = 0.0
            if symbol == 'C':
                neighbors_polar = any(n.GetSymbol() in ('O', 'N', 'P', 'S') for n in atom.GetNeighbors())
                if not neighbors_polar:
                    hydrophobicity = 0.8
                    
            radius_map = {'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'P': 1.8, 'S': 1.8}
            radius = radius_map.get(symbol, 1.5)
            
            self.atoms_single.append({
                "id": i,
                "element": symbol,
                "symbol": symbol,
                "radius": radius,
                "h_bond": h_bond,
                "hydrophobicity": hydrophobicity,
            })
            
        # Bonds
        for b in self.mol_heavy_single.GetBonds():
            self.bonds_single.append({
                "source": b.GetBeginAtomIdx(),
                "target": b.GetEndAtomIdx(),
                "type": str(b.GetBondType())
            })
            
        # Single molecule rotatable joints
        rotatable_bonds = list(self.mol_heavy_single.GetSubstructMatches(Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")))
        
        # Connection graph for DFS
        adj_single = {i: [] for i in range(self.n_atoms_single)}
        for b in self.bonds_single:
            adj_single[b["source"]].append(b["target"])
            adj_single[b["target"]].append(b["source"])
            
        for u, d in rotatable_bonds:
            visited = {u}
            downstream = []
            queue = [d]
            while queue:
                curr = queue.pop(0)
                if curr not in visited:
                    visited.add(curr)
                    downstream.append(curr)
                    for n in adj_single[curr]:
                        if n not in visited:
                            queue.append(n)
            self.joints_single.append({
                "u_idx": u,
                "d_idx": d,
                "downstream_atoms": downstream
            })

    def replicate_molecules(self):
        self.n_atoms = self.n_atoms_single * self.count
        self.coords = np.zeros((self.n_atoms, 3))
        self.atoms = []
        self.bonds = []
        self.joints = []
        self.atom_residues = {}
        
        # Place copies in a ring closer together or pre-assembled (solid phase)
        if self.initial_layout == "assembled":
            R = 0.5 + 0.3 * self.count if self.count > 1 else 0.0
        else: # "dispersed"
            R = 3.5 + 1.2 * self.count if self.count > 1 else 0.0
        
        # Calculate maximum X coordinate of polar atoms in the aligned single molecule
        self.max_polar_x = 0.0
        for i in range(self.n_atoms_single):
            # Check polar atoms (O, N, P, S)
            if self.atoms_single[i]["h_bond"] in ("donor", "acceptor", "both"):
                self.max_polar_x = max(self.max_polar_x, self.coords_single[i, 0])
        if self.max_polar_x == 0.0:
            self.max_polar_x = 5.0
            
        for k in range(self.count):
            theta = 2.0 * np.pi * k / self.count if self.count > 1 else 0.0
            offset = np.array([R * np.cos(theta), 0.0, R * np.sin(theta)])
            
            # Rotate by theta around Y-axis so the headgroup (+X) points outward radially!
            cos_a = np.cos(theta)
            sin_a = np.sin(theta)
            rot_matrix = np.array([
                [cos_a, 0.0, sin_a],
                [0.0, 1.0, 0.0],
                [-sin_a, 0.0, cos_a]
            ])
            
            for i in range(self.n_atoms_single):
                idx = k * self.n_atoms_single + i
                self.coords[idx] = np.dot(self.coords_single[i], rot_matrix) + offset
                
                atom_data = dict(self.atoms_single[i])
                atom_data["id"] = idx
                atom_data["res_num"] = k + 1
                atom_data["res_name"] = "LIP"
                atom_data["name"] = f"{atom_data['symbol']}{i}"
                self.atoms.append(atom_data)
                
                self.atom_residues[idx] = k
                
            for b in self.bonds_single:
                self.bonds.append({
                    "source": k * self.n_atoms_single + b["source"],
                    "target": k * self.n_atoms_single + b["target"],
                    "type": b["type"]
                })
                
            for j in self.joints_single:
                self.joints.append({
                    "u_idx": k * self.n_atoms_single + j["u_idx"],
                    "d_idx": k * self.n_atoms_single + j["d_idx"],
                    "downstream_atoms": [k * self.n_atoms_single + x for x in j["downstream_atoms"]]
                })
                
        # Rebuild full system adjacency list
        self.adj = {i: [] for i in range(self.n_atoms)}
        for b in self.bonds:
            self.adj[b["source"]].append(b["target"])
            self.adj[b["target"]].append(b["source"])
            
        # Rebuild full system component-based graph distances
        self.build_graph_distances()

    def build_graph_distances(self):
        self.graph_dist = np.full((self.n_atoms, self.n_atoms), 999)
        for i in range(self.n_atoms):
            self.graph_dist[i, i] = 0
            queue = [i]
            visited = {i}
            dist = {i: 0}
            while queue:
                curr = queue.pop(0)
                curr_d = dist[curr]
                for neighbor in self.adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        dist[neighbor] = curr_d + 1
                        self.graph_dist[i, neighbor] = curr_d + 1
                        queue.append(neighbor)

    def initialize_waters(self):
        self.waters = []
        water_coords_list = []
        
        # Place waters around polar/hydrophobic atoms of the combined system
        # (every 2nd atom is chosen to keep performance optimized)
        for i in range(0, self.n_atoms, 2):
            pos = self.coords[i]
            is_h = self.atoms[i]["hydrophobicity"] > 0.15
            is_p = self.atoms[i]["h_bond"] in ("donor", "acceptor", "both")
            
            if is_h or is_p:
                n_waters = 3
                for w in range(n_waters):
                    w_id = len(self.waters)
                    
                    # Random shell placement (more spacious: 3.8 Å)
                    r_shell = 3.8 + np.random.uniform(-0.3, 0.3)
                    theta = np.random.uniform(0, 2*np.pi)
                    phi = np.random.uniform(0, np.pi)
                    
                    o_pos = pos + np.array([
                        r_shell * np.sin(phi) * np.cos(theta),
                        r_shell * np.cos(phi),
                        r_shell * np.sin(phi) * np.sin(theta)
                    ])
                    
                    # V-shape geometry (H-O-H angle 104.5 degrees, O-H 0.96 Å)
                    # Random rotation around oxygen center
                    u = np.random.normal(0, 1, 3)
                    u /= np.linalg.norm(u)
                    v = np.random.normal(0, 1, 3)
                    v = v - np.dot(v, u)*u
                    v /= np.linalg.norm(v)
                    
                    h1_pos = o_pos + 0.96 * u
                    h2_pos = o_pos + 0.96 * (np.cos(104.5 * np.pi / 180) * u + np.sin(104.5 * np.pi / 180) * v)
                    
                    self.waters.append({
                        "id": w_id,
                        "parent_atom_id": i
                    })
                    water_coords_list.append([o_pos.tolist(), h1_pos.tolist(), h2_pos.tolist()])
        # Place a global outer ring of water molecules wrapping around all lipids
        # It should wrap just outside the headgroups (max_polar_x + 1.5 Å)
        R_lipids = 3.5 + 1.2 * self.count if self.count > 1 else 5.0
        r_outer = R_lipids + self.max_polar_x + 1.5
        n_outer_rings = 3
        n_waters_per_ring = 16
        for ring_y in [-3.0, 0.0, 3.0]:
            for j in range(n_waters_per_ring):
                theta = 2.0 * np.pi * j / n_waters_per_ring
                o_pos = np.array([r_outer * np.cos(theta), ring_y, r_outer * np.sin(theta)])
                
                # Add slight noise
                o_pos += np.random.uniform(-0.3, 0.3, 3)
                
                u = np.array([np.cos(theta), 0.0, np.sin(theta)]) # Outward radial direction
                v = np.array([0.0, 1.0, 0.0]) # Y-axis
                
                h1_pos = o_pos + 0.96 * u
                h2_pos = o_pos + 0.96 * (np.cos(104.5 * np.pi / 180) * u + np.sin(104.5 * np.pi / 180) * v)
                
                w_id = len(self.waters)
                self.waters.append({
                    "id": w_id,
                    "parent_atom_id": 0,
                    "is_outer_belt": True
                })
                water_coords_list.append([o_pos.tolist(), h1_pos.tolist(), h2_pos.tolist()])
                
        self.water_coords = np.array(water_coords_list)

    def calculate_radius_of_gyration(self):
        com = np.mean(self.coords, axis=0)
        sq_dist = np.sum((self.coords - com) ** 2)
        return np.sqrt(sq_dist / self.n_atoms)

    def step_explicit_waters(self, dt):
        num_waters = len(self.waters)
        if num_waters == 0:
            return np.zeros((self.n_atoms, 3))
            
        forces_water = np.zeros((num_waters, 3, 3))
        forces_reaction_on_peptide = np.zeros((self.n_atoms, 3))
        
        k_pw = self.params.get("waterPeptideStrength", 0.75) * 15.0
        k_ww = self.params.get("waterWaterStrength", 0.65) * 25.0
        k_capture = self.params.get("kCapture", 0.70) * 12.0
        
        # 1. Covalent O-H springs for water geometry
        k_int = 250.0
        for i in range(num_waters):
            pos_o = self.water_coords[i, 0]
            pos_h1 = self.water_coords[i, 1]
            pos_h2 = self.water_coords[i, 2]
            
            diff_h1 = pos_h1 - pos_o
            d_h1 = np.linalg.norm(diff_h1)
            if d_h1 > 1e-3:
                f = (d_h1 - 0.96) * k_int * (diff_h1 / d_h1)
                forces_water[i, 1] -= f
                forces_water[i, 0] += f
                
            diff_h2 = pos_h2 - pos_o
            d_h2 = np.linalg.norm(diff_h2)
            if d_h2 > 1e-3:
                f = (d_h2 - 0.96) * k_int * (diff_h2 / d_h2)
                forces_water[i, 2] -= f
                forces_water[i, 0] += f
                
            diff_hh = pos_h2 - pos_h1
            d_hh = np.linalg.norm(diff_hh)
            if d_hh > 1e-3:
                f = (d_hh - 1.52) * k_int * (diff_hh / d_hh)
                forces_water[i, 2] -= f
                forces_water[i, 1] += f

        # 2. Tracking force to parent heavy atom
        for i in range(num_waters):
            if self.waters[i].get("is_outer_belt", False):
                continue # Skip parent tracking force for outer belt waters!
            parent_id = self.waters[i]["parent_atom_id"]
            pos_p = self.coords[parent_id]
            pos_o = self.water_coords[i, 0]
            
            diff = pos_o - pos_p
            d = np.linalg.norm(diff)
            if d > 1e-3:
                f = (d - 3.8) * k_pw * (diff / d)
                forces_water[i, 0] -= f
                forces_reaction_on_peptide[parent_id] += f
                
        # 2b. Global outer water belt contraction force
        k_global_contract = self.params.get("globalWaterBeltContraction", 0.0) * 18.0
        if k_global_contract > 0:
            for idx in range(num_waters):
                w = self.waters[idx]
                if w.get("is_outer_belt", False):
                    pos_o = self.water_coords[idx, 0]
                    # Direct force pointing towards the central Y-axis (XZ projection)
                    dir_center = -np.array([pos_o[0], 0.0, pos_o[2]])
                    d_xz = np.linalg.norm(dir_center)
                    if d_xz > 0.1:
                        dir_center /= d_xz
                        # Force on oxygen pointing inward
                        forces_water[idx, 0] += dir_center * k_global_contract

        # 3. Dynamic Local Entropy
        polar_indices = [idx for idx in range(self.n_atoms) if self.atoms[idx]["h_bond"] in ("donor", "acceptor", "both")]
        hydrophobic_indices = [idx for idx in range(self.n_atoms) if self.atoms[idx]["hydrophobicity"] > 0.15]
        entropy_bias = self.params.get("entropyBias", 1.2)
        
        water_entropies = []
        water_weights = []
        for w_idx in range(num_waters):
            pos_o = self.water_coords[w_idx, 0]
            d_polar = min([np.linalg.norm(pos_o - self.coords[p]) for p in polar_indices]) if polar_indices else 99.0
            d_hydro = min([np.linalg.norm(pos_o - self.coords[h]) for h in hydrophobic_indices]) if hydrophobic_indices else 99.0
            
            f_polar = np.exp(- (d_polar ** 2) / (2 * 2.2 ** 2))
            f_hydro = np.exp(- (d_hydro ** 2) / (2 * 2.2 ** 2))
            
            S = 0.6 + 0.4 * f_hydro * entropy_bias - 0.5 * f_polar * entropy_bias
            S = np.clip(S, 0.05, 1.2)
            water_entropies.append(S)
            
            W = 1.0 + 1.5 * f_polar - 0.85 * f_hydro
            W = np.clip(W, 0.05, 4.0)
            water_weights.append(W)
            
        self.water_entropies = water_entropies
        self.water_weights = water_weights

        # 4. Dijkstra Network Routing
        num_graph_nodes = num_waters + len(polar_indices)
        graph_edges = {u: [] for u in range(num_graph_nodes)}
        
        # Water-Water edges
        for i in range(num_waters):
            pos_o_i = self.water_coords[i, 0]
            for j in range(i + 1, num_waters):
                pos_o_j = self.water_coords[j, 0]
                dist = np.linalg.norm(pos_o_i - pos_o_j)
                if dist < 3.5:
                    w_ij = np.sqrt(water_weights[i] * water_weights[j])
                    cost = 1.0 / (w_ij + 1e-5)
                    graph_edges[i].append((j, cost))
                    graph_edges[j].append((i, cost))
                    
        # Water-Peptide edges
        for i in range(num_waters):
            pos_o = self.water_coords[i, 0]
            for p_idx, p_node in enumerate(polar_indices):
                dist = np.linalg.norm(pos_o - self.coords[p_node])
                if dist < 3.5:
                    w_ip = np.sqrt(water_weights[i] * 2.5)
                    cost = 1.0 / (w_ip + 1e-5)
                    graph_node_p = num_waters + p_idx
                    graph_edges[i].append((graph_node_p, cost))
                    graph_edges[graph_node_p].append((i, cost))
                    
        # Dijkstra search
        polar_graph_nodes = [num_waters + p for p in range(len(polar_indices))]
        paths_by_src = {}
        for src in polar_graph_nodes:
            dists = {u: float('inf') for u in range(num_graph_nodes)}
            parents = {u: None for u in range(num_graph_nodes)}
            dists[src] = 0.0
            
            import heapq
            queue = [(0.0, src)]
            while queue:
                d, u = heapq.heappop(queue)
                if d > dists[u]:
                    continue
                for v, cost in graph_edges[u]:
                    if dists[u] + cost < dists[v]:
                        dists[v] = dists[u] + cost
                        parents[v] = u
                        heapq.heappush(queue, (dists[v], v))
            paths_by_src[src] = (dists, parents)

        # Draw H-Bonds & Routing
        active_path_list = []
        seen_edges = set()
        capture_index = 0.0
        
        for idx_a in range(len(polar_indices)):
            node_a = polar_graph_nodes[idx_a]
            dists, parents = paths_by_src[node_a]
            for idx_b in range(idx_a + 1, len(polar_indices)):
                node_b = polar_graph_nodes[idx_b]
                path_cost = dists[node_b]
                if path_cost < 99.0:
                    # Capturing information transfer capacity
                    capture_index += np.exp(-path_cost / 15.0)
                    
                    # Backtrace the Dijkstra path
                    curr = node_b
                    while parents[curr] is not None:
                        p_node = parents[curr]
                        edge = tuple(sorted([curr, p_node]))
                        if edge not in seen_edges:
                            seen_edges.add(edge)
                            
                            u, v = edge[0], edge[1]
                            if u < num_waters and v < num_waters:
                                # Water-Water path edge
                                active_path_list.append({
                                    "type": 0, # Water-Water
                                    "u": u,
                                    "v": v,
                                    "posU": self.water_coords[u, 0].tolist(),
                                    "posV": self.water_coords[v, 0].tolist()
                                })
                            else:
                                # Water-Peptide path edge
                                water_node = u
                                pep_node_idx = v - num_waters
                                pep_atom_idx = polar_indices[pep_node_idx]
                                active_path_list.append({
                                    "type": 1, # Water-Peptide
                                    "u": water_node,
                                    "v": pep_atom_idx,
                                    "posU": self.water_coords[water_node, 0].tolist(),
                                    "posV": self.coords[pep_atom_idx].tolist()
                                })
                        curr = p_node
                        
        self.information_capture_index = capture_index * 150.0
        self.active_information_paths = active_path_list

        # 5. Apply Dijkstra contraction forces
        for edge in active_path_list:
            posU = np.array(edge["posU"])
            posV = np.array(edge["posV"])
            diff = posV - posU
            d = np.linalg.norm(diff)
            if d > 0.05:
                f_dir = diff / d
                f = k_capture * 0.4
                if edge["type"] == 0:
                    forces_water[edge["u"], 0] += f * f_dir
                    forces_water[edge["v"], 0] -= f * f_dir
                else:
                    forces_water[edge["u"], 0] += f * f_dir
                    forces_reaction_on_peptide[edge["v"]] -= f * f_dir

        # 6. Steric Repulsions (O-O and O-Peptide)
        for i in range(num_waters):
            pos_o_i = self.water_coords[i, 0]
            for j in range(i + 1, num_waters):
                pos_o_j = self.water_coords[j, 0]
                diff = pos_o_j - pos_o_i
                d = np.linalg.norm(diff)
                if d < 2.8:
                    rep = (2.8 - d) * 30.0 * (diff / d)
                    forces_water[i, 0] -= rep
                    forces_water[j, 0] += rep

        for w_idx in range(num_waters):
            pos_o = self.water_coords[w_idx, 0]
            for p_idx in range(self.n_atoms):
                pos_p = self.coords[p_idx]
                diff = pos_p - pos_o
                d = np.linalg.norm(diff)
                min_dist = self.atoms[p_idx]["radius"] + 1.2
                if d < min_dist:
                    rep = (min_dist - d) * 35.0 * (diff / d)
                    forces_water[w_idx, 0] -= rep
                    forces_reaction_on_peptide[p_idx] += rep

        # Euler step for waters
        damping = 0.82
        for i in range(num_waters):
            for atom_type in range(3):
                f = forces_water[i, atom_type]
                noise = np.random.normal(0, 0.05, 3)
                self.water_coords[i, atom_type] += (f + noise) * dt * dt
                
        return forces_reaction_on_peptide

    def step(self, dt):
        with self.lock:
            self.step_counter += 1
            temp = self.params.get("temperature", 0.20)
            
            # Increment globalWaterBeltContraction over time if autoContractEnabled
            if self.params.get("autoContractEnabled", True) and self.params.get("isPlaying", True):
                self.params["globalWaterBeltContraction"] = min(2.5, self.params.get("globalWaterBeltContraction", 0.0) + 0.002)
            
            # Step water solver
            forces_reaction = np.zeros((self.n_atoms, 3))
            if self.params.get("explicitWaterEnabled", True):
                forces_reaction = self.step_explicit_waters(dt)
                
            # Compute distance matrix
            diff = self.coords[:, np.newaxis, :] - self.coords[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            
            forces_other = np.zeros((self.n_atoms, 3))
            
            # Thermal Noise
            if temp > 0:
                noise = np.random.normal(0, temp * 3.5, size=(self.n_atoms, 3))
                forces_other += noise
                
            # Pairwise interactions: Steric repulsion and hydrophobic attraction
            for i in range(self.n_atoms):
                for j in range(i + 1, self.n_atoms):
                    # Only apply forces if they are not bonded closely in the same copy
                    if self.graph_dist[i, j] <= 2:
                        continue
                        
                    d = dists[i, j]
                    if d < 1e-3:
                        continue
                        
                    min_dist = (self.atoms[i]["radius"] + self.atoms[j]["radius"]) * 0.45
                    
                    is_h1 = self.atoms[i]["hydrophobicity"] > 0.15
                    is_h2 = self.atoms[j]["hydrophobicity"] > 0.15
                    if is_h1 and is_h2:
                        # Hydrophobic attraction pulls carbon chains together (extended range)
                        if d > min_dist and d < 12.0:
                            attract_pull = (12.0 - d) * 3.5
                            dir_ij = (self.coords[j] - self.coords[i]) / d
                            forces_other[i] += dir_ij * attract_pull
                            forces_other[j] -= dir_ij * attract_pull
                            
                    if d < min_dist:
                        rep_push = (min_dist - d) * 15.0
                        dir_ij = (self.coords[j] - self.coords[i]) / d
                        forces_other[i] -= dir_ij * rep_push
                        forces_other[j] += dir_ij * rep_push

            # Bulk Solvent Exclusion Force (hydrophobic collapse towards system center of mass)
            k_bulk = self.params.get("bulkSolventStrength", 0.45) * 5.0
            if k_bulk > 0:
                com = np.mean(self.coords, axis=0)
                for i in range(self.n_atoms):
                    h = self.atoms[i]["hydrophobicity"]
                    if h > 0.15:
                        diff_com = com - self.coords[i]
                        d_com = np.linalg.norm(diff_com)
                        if d_com > 0.1:
                            # Pull hydrophobic atoms towards the center of mass
                            forces_other[i] += (diff_com / d_com) * k_bulk * h

            # Calculate inter-molecular hydrophobic contacts (aggregation metric)
            hydrophobic_contacts = 0
            h_atoms = [i for i in range(self.n_atoms) if self.atoms[i]["hydrophobicity"] > 0.15]
            for i in h_atoms:
                res_i = self.atom_residues[i]
                for j in h_atoms:
                    res_j = self.atom_residues[j]
                    if res_i < res_j: # Belong to different molecules
                        d = dists[i, j]
                        if d < 5.0:
                            hydrophobic_contacts += 1
            self.hydrophobic_contacts = hydrophobic_contacts

            # Combine forces
            forces = forces_other + forces_reaction
            
            # Clamp forces
            force_norms = np.linalg.norm(forces, axis=1)
            max_f = 25.0
            mask_large_f = force_norms > max_f
            if np.any(mask_large_f):
                scale_factors = np.ones(self.n_atoms)
                scale_factors[mask_large_f] = max_f / force_norms[mask_large_f]
                for i in range(self.n_atoms):
                    if scale_factors[i] < 1.0:
                        forces_other[i] *= scale_factors[i]
                        forces_reaction[i] *= scale_factors[i]
                forces = forces_other + forces_reaction

            # Rotations based on joint torques
            for joint in self.joints:
                u_idx = joint["u_idx"]
                d_idx = joint["d_idx"]
                D = joint["downstream_atoms"]
                
                pos_A = self.coords[u_idx]
                pos_B = self.coords[d_idx]
                axis = pos_B - pos_A
                axis_len = np.linalg.norm(axis)
                if axis_len < 1e-4:
                    continue
                axis /= axis_len
                
                torque_reaction = 0.0
                torque_other = 0.0
                inertia = 0.0
                
                for idx in D:
                    r = self.coords[idx] - pos_B
                    t_react = np.cross(r, forces_reaction[idx])
                    torque_reaction += np.dot(t_react, axis)
                    
                    t_other = np.cross(r, forces_other[idx])
                    torque_other += np.dot(t_other, axis)
                    
                    cross_r_axis = np.cross(r, axis)
                    inertia += np.dot(cross_r_axis, cross_r_axis)
                    
                torque = torque_reaction + torque_other
                if abs(torque) < 1e-5:
                    continue
                
                # Highly dynamic flexible joints
                damping_const = 0.4
                effective_inertia = max(0.5, inertia * 0.05)
                d_theta = (dt * torque) / (damping_const * effective_inertia)
                
                # Clamp rotation angle
                d_theta = np.clip(d_theta, -0.4, 0.4)
                
                # Rodrigues rotation formula
                cos_t = np.cos(d_theta)
                sin_t = np.sin(d_theta)
                for idx in D:
                    r_vec = self.coords[idx] - pos_B
                    rotated = r_vec * cos_t + np.cross(axis, r_vec) * sin_t + axis * np.dot(axis, r_vec) * (1 - cos_t)
                    self.coords[idx] = pos_B + rotated

            # Apply global rigid-body translation and rotation to each copy
            for k in range(self.count):
                start_idx = k * self.n_atoms_single
                end_idx = (k + 1) * self.n_atoms_single
                coords_copy = self.coords[start_idx:end_idx]
                forces_copy = forces[start_idx:end_idx]
                
                # 1. Translation (F = m * a)
                F_net = np.sum(forces_copy, axis=0)
                d_trans = (F_net / self.n_atoms_single) * 0.20
                d_trans_len = np.linalg.norm(d_trans)
                if d_trans_len > 0.4:
                    d_trans = (d_trans / d_trans_len) * 0.4
                self.coords[start_idx:end_idx] += d_trans
                
                # 2. Rotation around Center of Mass
                com = np.mean(self.coords[start_idx:end_idx], axis=0)
                torque_net = np.zeros(3)
                inertia_net = 1.0
                for i in range(self.n_atoms_single):
                    r = coords_copy[i] - com
                    f = forces_copy[i]
                    torque_net += np.cross(r, f)
                    inertia_net += np.dot(r, r)
                    
                omega = torque_net / (inertia_net * 2.0)
                omega_len = np.linalg.norm(omega)
                if omega_len > 1e-5:
                    if omega_len > 0.15:
                        omega = (omega / omega_len) * 0.15
                    theta_rot = np.linalg.norm(omega)
                    axis_rot = omega / theta_rot
                    cos_tr = np.cos(theta_rot)
                    sin_tr = np.sin(theta_rot)
                    for i in range(self.n_atoms_single):
                        idx = start_idx + i
                        r_vec = self.coords[idx] - com
                        rotated = r_vec * cos_tr + np.cross(axis_rot, r_vec) * sin_tr + axis_rot * np.dot(axis_rot, r_vec) * (1 - cos_tr)
                        self.coords[idx] = com + rotated

            # Calculate Hydrophobic Exposure locally
            hydrophobic_indices = [idx for idx in range(self.n_atoms) if self.atoms[idx]["hydrophobicity"] > 0.15]
            num_waters = len(self.waters)
            if hydrophobic_indices and num_waters > 0:
                h_water_dist = 0.0
                for h in hydrophobic_indices:
                    min_dw = min([np.linalg.norm(self.coords[h] - self.water_coords[w, 0]) for w in range(num_waters)])
                    h_water_dist += min_dw
                self.hydrophobic_exposure = h_water_dist / len(hydrophobic_indices)

# Global simulation instance
sim_instance = None
session_id = None

@app.route('/api/initialize', methods=['POST'])
def initialize():
    global sim_instance, session_id
    try:
        req_data = request.json or {}
        sequence = req_data.get("sequence", "CCCCCCCCCCCCCCCC(=O)OCC(COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCC")
        count = req_data.get("count", 3)
        initial_layout = req_data.get("initialLayout", "dispersed")
        
        sim_instance = LipidSimulation(sequence, count, initial_layout)
        import uuid
        session_id = str(uuid.uuid4())
        
        return jsonify({
            "status": "success",
            "sessionId": session_id,
            "atoms": sim_instance.atoms,
            "bonds": sim_instance.bonds,
            "initialCoords": sim_instance.coords.tolist(),
            "atomResidues": sim_instance.atom_residues,
            "residueLabels": sim_instance.residue_labels,
            "waters": sim_instance.waters,
            "waterCoords": sim_instance.water_coords.tolist() if len(sim_instance.waters) > 0 else []
        })
    except Exception as e:
        print("API Initialize Error:", e)
        return jsonify({"status": "error", "error": str(e)}), 400

@app.route('/api/control', methods=['POST'])
def control():
    global sim_instance
    if not sim_instance:
        return jsonify({"status": "error", "error": "Simulation not initialized"}), 400
    try:
        req_data = request.json or {}
        with sim_instance.lock:
            if "temperature" in req_data:
                sim_instance.params["temperature"] = float(req_data["temperature"])
            if "isPlaying" in req_data:
                sim_instance.params["isPlaying"] = bool(req_data["isPlaying"])
            if "explicitWaterEnabled" in req_data:
                sim_instance.params["explicitWaterEnabled"] = bool(req_data["explicitWaterEnabled"])
            if "waterWaterStrength" in req_data:
                sim_instance.params["waterWaterStrength"] = float(req_data["waterWaterStrength"])
            if "waterPeptideStrength" in req_data:
                sim_instance.params["waterPeptideStrength"] = float(req_data["waterPeptideStrength"])
            if "bulkSolventStrength" in req_data:
                sim_instance.params["bulkSolventStrength"] = float(req_data["bulkSolventStrength"])
            if "kCapture" in req_data:
                sim_instance.params["kCapture"] = float(req_data["kCapture"])
            if "entropyBias" in req_data:
                sim_instance.params["entropyBias"] = float(req_data["entropyBias"])
            if "globalWaterBeltContraction" in req_data:
                sim_instance.params["globalWaterBeltContraction"] = float(req_data["globalWaterBeltContraction"])
            if "autoContractEnabled" in req_data:
                sim_instance.params["autoContractEnabled"] = bool(req_data["autoContractEnabled"])
                
        return jsonify({"status": "success", "params": sim_instance.params})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 400

@app.route('/api/stream')
def stream():
    req_session_id = request.args.get("sessionId")
    if not req_session_id or req_session_id != session_id:
        return "Invalid session", 400
        
    def event_generator():
        global sim_instance
        dt = 0.02
        try:
            while True:
                if sim_instance.params["isPlaying"]:
                    for _ in range(5):
                        sim_instance.step(dt)
                        
                with sim_instance.lock:
                    payload = {
                        "coords": sim_instance.coords.tolist(),
                        "temperature": sim_instance.params["temperature"],
                        "logs": list(sim_instance.logs),
                        "hydrophobicExposure": sim_instance.hydrophobic_exposure,
                        "explicitWaterEnabled": sim_instance.params["explicitWaterEnabled"],
                        "waterCoords": sim_instance.water_coords.tolist() if len(sim_instance.waters) > 0 else [],
                        "waters": sim_instance.waters,
                        "waterEntropies": sim_instance.water_entropies,
                        "waterWeights": sim_instance.water_weights,
                        "informationCaptureIndex": sim_instance.information_capture_index,
                        "activeInformationPaths": sim_instance.active_information_paths,
                        "hydrophobicContacts": sim_instance.hydrophobic_contacts,
                        "globalWaterBeltContraction": sim_instance.params["globalWaterBeltContraction"],
                        "autoContractEnabled": sim_instance.params["autoContractEnabled"]
                    }
                    sim_instance.logs = [] # Clear logs after streaming
                    
                yield f"data: {json.dumps(payload)}\n\n"
                time.sleep(0.06)
        except GeneratorExit:
            print("Client disconnected from stream.")
            
    return Response(event_generator(), mimetype="text/event-stream")

if __name__ == "__main__":
    # Binds on port 5004 for lipid simulations
    app.run(host="0.0.0.0", port=5004, debug=False)
