import sys
import os
import uuid
import json
import time
import threading
from flask import Flask, request, jsonify, Response
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import heapq

app = Flask(__name__)

# CORS middleware for local development
@app.after_request
def add_cors_headers(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

# Helper to check if a bond is an amide bond (peptide bond)
def is_amide_bond(mol, bond):
    if bond.GetBondType() != Chem.BondType.SINGLE:
        return False
    a1 = bond.GetBeginAtom()
    a2 = bond.GetEndAtom()
    if {a1.GetSymbol(), a2.GetSymbol()} == {'C', 'N'}:
        # Amide carbons are bonded to double-bond oxygens
        c_atom = a1 if a1.GetSymbol() == 'C' else a2
        for nbr in c_atom.GetNeighbors():
            if nbr.GetSymbol() == 'O':
                b = mol.GetBondBetweenAtoms(c_atom.GetIdx(), nbr.GetIdx())
                if b and b.GetBondType() == Chem.BondType.DOUBLE:
                    return True
    return False

class PeptideSimulation:
    def __init__(self, sequence):
        self.lock = threading.Lock()
        self.sequence = sequence
        self.step_counter = 0
        self.logs = []
        
        # 1. Parse peptide sequence to RDKit molecule
        self.mol = self.parse_molecule(sequence)
        
        # 2. Generate 3D coordinates
        self.initialize_coordinates()
        
        # 3. Extract topology (atoms & bonds)
        self.extract_topology()
        
        # 4. Group atoms into residues
        self.assign_residues()
        
        # Identify rotatable bonds (joints)
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
        
        # 5. Center conformation
        self.center_around_first_residue()
        
        # 6. Setup parameters
        self.params = {
            "temperature": 0.25,
            "isPlaying": True,
            "explicitWaterEnabled": True,
            "waterWaterStrength": 0.65,
            "waterPeptideStrength": 0.75,
            "bulkSolventStrength": 0.45,
            "waterBeltContraction": 0.0,
            "watersPerAtom": 4,
            "kCapture": 0.70,
            "entropyBias": 1.2
        }
        
        self.hydrophobic_exposure = 0.0
        self.water_entropies = []
        self.water_weights = []
        self.information_capture_index = 0.0
        self.active_information_paths = []
        
        self.seq_folding = {
            "enabled": False,
            "activeResidue": 0,
            "stepsPerResidue": 500,
            "currentStep": 0,
            "status": "idle",
            "direction": 1
        }
        
        # 7. Initialize H2O water molecules
        self.initialize_waters()
        
    def parse_molecule(self, input_str):
        input_str = input_str.strip()
        
        # Option A: Try parsing as SMILES
        mol = Chem.MolFromSmiles(input_str)
        if mol:
            return mol
            
        # Option B: Try parsing as 3-letter amino acid code separated by hyphens (e.g. Ala-Cys-Gly)
        if '-' in input_str or input_str.istitle():
            parts = [p.strip().capitalize() for p in input_str.split('-') if p.strip()]
            mapping = {
                'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
                'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
                'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
                'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V'
            }
            seq = ""
            for p in parts:
                if p in mapping:
                    seq += mapping[p]
                else:
                    raise ValueError(f"Unknown amino acid residue: '{p}'")
            mol = Chem.MolFromSequence(seq)
            if mol:
                return mol
                
        # Option C: Try parsing as 1-letter amino acid sequence (e.g., AAAAA)
        mol = Chem.MolFromSequence(input_str)
        if mol:
            return mol
            
        raise ValueError("Could not parse sequence. Please provide a valid 1-letter, 3-letter sequence or SMILES.")

    def initialize_coordinates(self):
        mol_hs = Chem.AddHs(self.mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        embed_status = AllChem.EmbedMolecule(mol_hs, params)
        if embed_status == -1:
            AllChem.EmbedMolecule(mol_hs, useRandomCoords=True)
            
        try:
            AllChem.MMFFOptimizeMolecule(mol_hs, maxIters=500)
        except Exception:
            pass
            
        self.mol_heavy = Chem.RemoveHs(mol_hs)
        self.n_atoms = self.mol_heavy.GetNumAtoms()
        
        conf = self.mol_heavy.GetConformer()
        self.coords = np.zeros((self.n_atoms, 3))
        for i in range(self.n_atoms):
            pos = conf.GetAtomPosition(i)
            self.coords[i] = [pos.x, pos.y, pos.z]

    def extract_topology(self):
        self.atoms = []
        self.bonds = []
        self.adj = {i: [] for i in range(self.n_atoms)}
        
        # Heavy atoms properties
        for i in range(self.n_atoms):
            atom = self.mol_heavy.GetAtomWithIdx(i)
            symbol = atom.GetSymbol()
            
            # Simple classification for charge & hydrogen bonding capability
            h_bond = "none"
            if symbol in ("O", "N"):
                if symbol == "O":
                    h_bond = "acceptor"
                elif symbol == "N":
                    h_bond = "donor"
            
            # Hydrophobicity logP approximations
            hydrophobicity = 0.0
            if symbol == 'C':
                nbrs = [n.GetSymbol() for n in atom.GetNeighbors()]
                if len(nbrs) > 0 and set(nbrs).issubset({'C', 'H', 'S'}):
                    hydrophobicity = 0.45
                else:
                    hydrophobicity = 0.15
            elif symbol in ('F', 'Cl', 'Br', 'I', 'S'):
                hydrophobicity = 0.50
                
            # Steric radius
            radius = 1.7
            if symbol == 'C': radius = 1.7
            elif symbol == 'N': radius = 1.55
            elif symbol == 'O': radius = 1.52
            elif symbol == 'S': radius = 1.8
            
            info = atom.GetMonomerInfo()
            atom_name = info.GetName().strip() if info else symbol
            res_name = info.GetResidueName().strip() if info else "UNK"
            res_num = info.GetResidueNumber() if info else 1
            
            self.atoms.append({
                "id": i,
                "element": symbol,
                "symbol": symbol,
                "name": atom_name,
                "radius": radius,
                "h_bond": h_bond,
                "hydrophobicity": hydrophobicity,
                "res_name": res_name,
                "res_num": res_num
            })
            
        # Heavy bonds
        for b in self.mol_heavy.GetBonds():
            s = b.GetBeginAtomIdx()
            t = b.GetEndAtomIdx()
            if s >= self.n_atoms or t >= self.n_atoms:
                continue
            self.bonds.append({
                "source": s,
                "target": t,
                "type": str(b.GetBondType())
            })
            self.adj[s].append(t)
            self.adj[t].append(s)
            
        # Graph distances
        self.graph_dist = np.full((self.n_atoms, self.n_atoms), 999)
        np.fill_diagonal(self.graph_dist, 0)
        for b in self.bonds:
            self.graph_dist[b["source"], b["target"]] = 1
            self.graph_dist[b["target"], b["source"]] = 1
            
        for k in range(self.n_atoms):
            for i in range(self.n_atoms):
                for j in range(self.n_atoms):
                    if self.graph_dist[i, k] + self.graph_dist[k, j] < self.graph_dist[i, j]:
                        self.graph_dist[i, j] = self.graph_dist[i, k] + self.graph_dist[k, j]
                        
        # Identify rotatable peptide joints (excluding amide and terminal bonds)
        self.joints = []
        for b in self.mol_heavy.GetBonds():
            if is_amide_bond(self.mol_heavy, b):
                continue
            s = b.GetBeginAtomIdx()
            t = b.GetEndAtomIdx()
            if s >= self.n_atoms or t >= self.n_atoms:
                continue
                
            # Check terminal atoms
            if len(self.adj[s]) <= 1 or len(self.adj[t]) <= 1:
                continue
                
            # Split graph to find downstream atoms (D)
            visited = {s: True, t: True}
            downstream = []
            queue = [t]
            while queue:
                curr = queue.pop(0)
                downstream.append(curr)
                for nbr in self.adj[curr]:
                    if nbr not in visited:
                        visited[nbr] = True
                        queue.append(nbr)
                        
            if downstream and len(downstream) < self.n_atoms - 1:
                self.joints.append({
                    "u_idx": s,
                    "d_idx": t,
                    "downstream_atoms": downstream
                })
                
        self.logs.append(f"PeptideModel: Defined {len(self.atoms)} heavy atoms, {len(self.bonds)} bonds, and {len(self.joints)} rotatable joints.")

    def assign_residues(self):
        self.atom_residues = {}
        self.residue_labels = []
        n_term_candidates = [i for i, a in enumerate(self.atoms) if a["element"] == 'N' and len(self.adj[i]) <= 3]
        n_term = n_term_candidates[0] if n_term_candidates else 0
        distances_from_n = self.graph_dist[n_term]
        
        for i in range(self.n_atoms):
            res_idx = int(distances_from_n[i] / 3)
            self.atom_residues[i] = res_idx
            
        self.n_residues = max(self.atom_residues.values()) + 1 if self.atom_residues else 1
        self.residue_labels = [f"Res{r+1}" for r in range(self.n_residues)]

    def center_around_first_residue(self):
        res0_indices = [i for i in range(self.n_atoms) if self.atom_residues.get(i, 0) == 0]
        anchor_pos = np.mean(self.coords[res0_indices], axis=0) if res0_indices else self.coords[0]
        self.coords -= anchor_pos
        self.initial_coords = self.coords.copy()

    def initialize_waters(self):
        # Spawns water molecules around ALL peptide heavy atoms
        self.waters = []
        water_coords_list = []
        
        n_waters = int(self.params.get("watersPerAtom", 4))
        
        dirs = []
        for i in range(n_waters):
            phi = np.arccos(1.0 - 2.0 * (i + 0.5) / n_waters)
            theta = np.pi * (1.0 + 5.0 ** 0.5) * (i + 0.5)
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            dirs.append(np.array([x, y, z]))

        w_idx = 0
        for i in range(self.n_atoms):
            parent_pos = self.coords[i]
            res_idx = self.atom_residues.get(i, 0)
            is_hydrophobic = self.atoms[i]["hydrophobicity"] > 0.15
            
            for v in dirs:
                jitter = np.random.normal(0, 0.05, 3)
                final_dir = v + jitter
                final_dir /= np.linalg.norm(final_dir)
                
                pos_o = parent_pos + final_dir * 3.2
                
                b_dir = np.random.normal(0, 1, 3)
                b_dir /= np.linalg.norm(b_dir)
                
                aux = np.array([1.0, 0.0, 0.0]) if abs(b_dir[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                p_dir = np.cross(b_dir, aux)
                p_dir /= np.linalg.norm(p_dir)
                
                theta_half = np.radians(104.5 / 2.0)
                c = np.cos(theta_half)
                s = np.sin(theta_half)
                v_h1 = c * b_dir + s * p_dir
                v_h2 = c * b_dir - s * p_dir
                
                pos_h1 = pos_o + 0.96 * v_h1
                pos_h2 = pos_o + 0.96 * v_h2
                
                water_coords_list.append([pos_o, pos_h1, pos_h2])
                
                self.waters.append({
                    "id": w_idx,
                    "parent_atom_id": i,
                    "residue_index": res_idx,
                    "is_hydrophobic_neighborhood": is_hydrophobic,
                    "state": "searching"
                })
                w_idx += 1
                
        if water_coords_list:
            self.water_coords = np.array(water_coords_list)
        else:
            self.water_coords = np.zeros((0, 3, 3))

    def step_explicit_waters(self, dt):
        if not self.params.get("explicitWaterEnabled", True) or len(self.waters) == 0:
            self.water_entropies = []
            self.water_weights = []
            self.information_capture_index = 0.0
            self.active_information_paths = []
            return np.zeros((self.n_atoms, 3))
            
        num_waters = len(self.waters)
        forces_water = np.zeros((num_waters, 3, 3)) # Oxygen, Hydrogen 1, Hydrogen 2
        forces_reaction_on_peptide = np.zeros((self.n_atoms, 3))
        
        # Scaling variables
        k_pw = self.params.get("waterPeptideStrength", 0.60) * 15.0
        k_ww = self.params.get("waterWaterStrength", 0.50) * 25.0
        k_capture = self.params.get("kCapture", 0.60) * 12.0
        
        # 1. Internal Geometry Constraints (V-shape springs)
        k_int = 250.0
        for i in range(num_waters):
            pos_o = self.water_coords[i, 0]
            pos_h1 = self.water_coords[i, 1]
            pos_h2 = self.water_coords[i, 2]
            
            # O - H1 (0.96 Å)
            diff_h1 = pos_h1 - pos_o
            d_h1 = np.linalg.norm(diff_h1)
            if d_h1 > 1e-3:
                f = (d_h1 - 0.96) * k_int * (diff_h1 / d_h1)
                forces_water[i, 1] -= f
                forces_water[i, 0] += f
                
            # O - H2 (0.96 Å)
            diff_h2 = pos_h2 - pos_o
            d_h2 = np.linalg.norm(diff_h2)
            if d_h2 > 1e-3:
                f = (d_h2 - 0.96) * k_int * (diff_h2 / d_h2)
                forces_water[i, 2] -= f
                forces_water[i, 0] += f
                
            # H1 - H2 (1.52 Å)
            diff_hh = pos_h2 - pos_h1
            d_hh = np.linalg.norm(diff_hh)
            if d_hh > 1e-3:
                f = (d_hh - 1.52) * k_int * (diff_hh / d_hh)
                forces_water[i, 2] -= f
                forces_water[i, 1] += f

        # 2. Covalent parent atom tracking force
        for i in range(num_waters):
            parent_id = self.waters[i]["parent_atom_id"]
            pos_p = self.coords[parent_id]
            pos_o = self.water_coords[i, 0]
            
            diff = pos_o - pos_p
            d = np.linalg.norm(diff)
            if d > 1e-3:
                # Spring rest length = 3.2 Å
                f = (d - 3.2) * k_pw * (diff / d)
                forces_water[i, 0] -= f
                forces_reaction_on_peptide[parent_id] += f

        # 3. Dynamic Entropy & Connectivity Calculations
        polar_indices = [idx for idx in range(self.n_atoms) if self.atoms[idx]["h_bond"] in ("donor", "acceptor", "both")]
        hydrophobic_indices = [idx for idx in range(self.n_atoms) if self.atoms[idx]["hydrophobicity"] > 0.15]
        
        entropy_bias = self.params.get("entropyBias", 1.0)
        
        water_entropies = []
        water_weights = []
        
        for w_idx in range(num_waters):
            pos_o = self.water_coords[w_idx, 0]
            
            # Distance to closest polar and hydrophobic peptide atoms
            d_polar = min([np.linalg.norm(pos_o - self.coords[p]) for p in polar_indices]) if polar_indices else 99.0
            d_hydro = min([np.linalg.norm(pos_o - self.coords[h]) for h in hydrophobic_indices]) if hydrophobic_indices else 99.0
            
            f_polar = np.exp(- (d_polar ** 2) / (2 * 2.2 ** 2))
            f_hydro = np.exp(- (d_hydro ** 2) / (2 * 2.2 ** 2))
            
            # Hydrophilic: anchored, low entropy (0.1). Hydrophobic: loose, high entropy (1.0). Bulk: 0.6.
            S = 0.6 + 0.4 * f_hydro * entropy_bias - 0.5 * f_polar * entropy_bias
            S = np.clip(S, 0.05, 1.2)
            water_entropies.append(S)
            
            # Hydrophilic: high network connectivity (2.5). Hydrophobic: low connectivity (0.15). Bulk: 1.0.
            W = 1.0 + 1.5 * f_polar - 0.85 * f_hydro
            W = np.clip(W, 0.05, 4.0)
            water_weights.append(W)
            
        self.water_entropies = water_entropies
        self.water_weights = water_weights

        # 4. Graph Construction & Dijkstra Solver for I_capture
        # Nodes:
        # 0 ... num_waters - 1 : Water molecules
        # num_waters ... num_waters + len(polar_indices) - 1 : Peptide polar atoms
        num_graph_nodes = num_waters + len(polar_indices)
        graph_edges = {u: [] for u in range(num_graph_nodes)}
        
        # Build Water-Water edges (Oxygen-Oxygen distance < 3.5 Å)
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
                    
        # Build Water-Peptide polar edges (Oxygen-Polar distance < 3.5 Å)
        for i in range(num_waters):
            pos_o = self.water_coords[i, 0]
            for p_idx, p_node in enumerate(polar_indices):
                dist = np.linalg.norm(pos_o - self.coords[p_node])
                if dist < 3.5:
                    # Polar atom has a constant high weight factor of 2.5
                    w_ip = np.sqrt(water_weights[i] * 2.5)
                    cost = 1.0 / (w_ip + 1e-5)
                    graph_node_p = num_waters + p_idx
                    graph_edges[i].append((graph_node_p, cost))
                    graph_edges[graph_node_p].append((i, cost))

        # Dijkstra shortest path solver
        def dijkstra(src):
            dist = {src: 0.0}
            parent = {src: None}
            pq = [(0.0, src)]
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u]:
                    continue
                for v, cost in graph_edges[u]:
                    if dist.get(v, float('inf')) > d + cost:
                        dist[v] = d + cost
                        parent[v] = u
                        heapq.heappush(pq, (dist[v], v))
            return dist, parent

        # Calculate connectivity between all pairs of polar peptide atoms
        polar_graph_nodes = [num_waters + idx for idx in range(len(polar_indices))]
        paths_by_src = {}
        for src in polar_graph_nodes:
            paths_by_src[src] = dijkstra(src)
            
        I_capture = 0.0
        active_paths = []
        path_edges_set = set()
        
        for idx_a in range(len(polar_indices)):
            node_a = polar_graph_nodes[idx_a]
            dists, parents = paths_by_src[node_a]
            
            for idx_b in range(idx_a + 1, len(polar_indices)):
                node_b = polar_graph_nodes[idx_b]
                d_ab = dists.get(node_b, float('inf'))
                
                if d_ab < 999.0:
                    I_capture += 1.0 / d_ab
                    
                    # Backtrace shortest path to find routing edges
                    curr = node_b
                    while curr is not None:
                        prev = parents[curr]
                        if prev is not None:
                            edge = (min(curr, prev), max(curr, prev))
                            path_edges_set.add(edge)
                        curr = prev

        self.information_capture_index = I_capture
        
        # Convert path edges to actual indices/positions for rendering
        active_path_list = []
        for u, v in path_edges_set:
            # Map graph node indices back to client structure
            # Type: 0 = water-water, 1 = water-peptide
            if u < num_waters and v < num_waters:
                active_path_list.append({
                    "u": u, "v": v, "type": 0,
                    "posU": self.water_coords[u, 0].tolist(),
                    "posV": self.water_coords[v, 0].tolist()
                })
            elif u < num_waters and v >= num_waters:
                p_idx = v - num_waters
                p_node = polar_indices[p_idx]
                active_path_list.append({
                    "u": u, "v": p_node, "type": 1,
                    "posU": self.water_coords[u, 0].tolist(),
                    "posV": self.coords[p_node].tolist()
                })
            elif u >= num_waters and v < num_waters:
                p_idx = u - num_waters
                p_node = polar_indices[p_idx]
                active_path_list.append({
                    "u": p_node, "v": v, "type": 1,
                    "posU": self.coords[p_node].tolist(),
                    "posV": self.water_coords[v, 0].tolist()
                })
        self.active_information_paths = active_path_list

        # 5. Apply Information Routing Path Contraction Forces
        # For each edge in the active paths, apply attractive forces
        for edge in active_path_list:
            posU = np.array(edge["posU"])
            posV = np.array(edge["posV"])
            diff = posV - posU
            d = np.linalg.norm(diff)
            if d > 0.05:
                f_dir = diff / d
                # Linear attraction scaled by capture drive strength
                f = k_capture * 0.4
                
                # Apply forces to water O
                if edge["type"] == 0:
                    forces_water[edge["u"], 0] += f * f_dir
                    forces_water[edge["v"], 0] -= f * f_dir
                else: # water-peptide
                    # u is water, v is peptide polar atom
                    forces_water[edge["u"], 0] += f * f_dir
                    forces_reaction_on_peptide[edge["v"]] -= f * f_dir
                    
        # 6. Unconnected long-range search force (exploration)
        # Pull waters toward midpoints of unconnected polar pairs to bridge gaps
        unconnected_pairs = []
        for idx_a in range(len(polar_indices)):
            node_a = polar_graph_nodes[idx_a]
            dists, _ = paths_by_src[node_a]
            for idx_b in range(idx_a + 1, len(polar_indices)):
                node_b = polar_graph_nodes[idx_b]
                if dists.get(node_b, float('inf')) > 999.0:
                    unconnected_pairs.append((polar_indices[idx_a], polar_indices[idx_b]))
                    
        if unconnected_pairs and num_waters > 0:
            for p1, p2 in unconnected_pairs:
                midpoint = 0.5 * (self.coords[p1] + self.coords[p2])
                for i in range(num_waters):
                    pos_o = self.water_coords[i, 0]
                    diff = midpoint - pos_o
                    d = np.linalg.norm(diff)
                    if d > 0.1 and d < 8.0:
                        forces_water[i, 0] += (diff / d) * k_capture * 0.03

        # 7. Classical Steric Repulsions (O-O and O-Peptide)
        # Water-Water steric repulsion (Oxygen-Oxygen)
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

        # Peptide-Water Oxygen steric repulsion
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

        # Integrate H2O equations of motion (Euler integration with damping)
        damping = 0.82
        for i in range(num_waters):
            for atom_type in range(3): # Oxygen, H1, H2
                f = forces_water[i, atom_type]
                # Dynamic noise
                noise = np.random.normal(0, 0.05, 3)
                self.water_coords[i, atom_type] += (f + noise) * dt * dt
                
        return forces_reaction_on_peptide

    def step(self, dt):
        with self.lock:
            self.step_counter += 1
            temp = self.params.get("temperature", 0.08)
            
            # Step the water solvent agent first to compute information-theoretic forces
            forces_reaction = np.zeros((self.n_atoms, 3))
            if self.params.get("explicitWaterEnabled", True):
                forces_reaction = self.step_explicit_waters(dt)
                
            # If sequential folding is enabled
            R = 0
            if self.seq_folding["enabled"]:
                R = self.seq_folding["activeResidue"]
                direction = self.seq_folding.get("direction", 1)
                self.seq_folding["currentStep"] += 1
                self.params["waterBeltContraction"] = 0.0
                
                if self.seq_folding["currentStep"] >= self.seq_folding["stepsPerResidue"]:
                    self.seq_folding["currentStep"] = 0
                    
                    if direction == 1:
                        if R < self.n_residues - 1:
                            self.seq_folding["activeResidue"] += 1
                            R = self.seq_folding["activeResidue"]
                            self.logs.append(f"System: Residue {R + 1} ({self.residue_labels[R]}) folding started (N -> C). Previous residues frozen.")
                        else:
                            # Reached C-terminus, reverse to C->N
                            self.seq_folding["direction"] = -1
                            self.seq_folding["activeResidue"] = self.n_residues - 2 if self.n_residues > 1 else 0
                            R = self.seq_folding["activeResidue"]
                            self.logs.append(f"System: Reached C-terminus. Reversing direction to C -> N. Residue {R + 1} ({self.residue_labels[R]}) folding started.")
                    else: # direction == -1
                        if R > 0:
                            self.seq_folding["activeResidue"] -= 1
                            R = self.seq_folding["activeResidue"]
                            self.logs.append(f"System: Residue {R + 1} ({self.residue_labels[R]}) folding started (C -> N). Subsequent residues frozen.")
                        else:
                            # Reached N-terminus, folding is completed!
                            self.seq_folding["status"] = "completed"
                            self.seq_folding["enabled"] = False
                            self.params["temperature"] = 0.02 # Cool down automatically!
                            self.params["waterBeltContraction"] = self.seq_folding.get("maxContraction", 0.50)
                            self.logs.append("System: Sequential folding completed! Automatically cooling down to stabilize the final folded structure.")
            
            # Compute distance matrix
            diff = self.coords[:, np.newaxis, :] - self.coords[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            
            forces_other = np.zeros((self.n_atoms, 3))
            self.active_hbonds = []
            
            # Thermal Noise (applied to active residue in sequential mode)
            if temp > 0:
                noise = np.random.normal(0, temp * 3.5, size=(self.n_atoms, 3))
                if self.seq_folding["enabled"]:
                    for i in range(self.n_atoms):
                        res_i = self.atom_residues.get(i, 0)
                        if res_i == R:
                            forces_other[i] += noise[i]
                else:
                    forces_other += noise
                
            # Bulk solvent pressure effect
            bulk_solv = self.params.get("bulkSolventStrength", 0.0)
            if bulk_solv > 1e-4:
                com = np.mean(self.coords, axis=0)
                for i in range(self.n_atoms):
                    is_h = self.atoms[i]["hydrophobicity"] > 0.15
                    is_p = self.atoms[i]["h_bond"] in ("donor", "acceptor", "both")
                    if is_h:
                        dir_com = com - self.coords[i]
                        d_com = np.linalg.norm(dir_com)
                        if d_com > 0.1:
                            forces_other[i] += (dir_com / d_com) * bulk_solv * 6.0
                    elif is_p:
                        dir_com = self.coords[i] - com
                        d_com = np.linalg.norm(dir_com)
                        if d_com > 0.1:
                            forces_other[i] += (dir_com / d_com) * bulk_solv * 4.0
                            
            # Pairwise interactions: steric repulsion and hydrophobic attraction
            for i in range(self.n_atoms):
                for j in range(i + 1, self.n_atoms):
                    if self.graph_dist[i, j] <= 2:
                        continue
                        
                    d = dists[i, j]
                    if d < 1e-3:
                        continue
                        
                    min_dist = (self.atoms[i]["radius"] + self.atoms[j]["radius"]) * 0.45
                    
                    is_h1 = self.atoms[i]["hydrophobicity"] > 0.15
                    is_h2 = self.atoms[j]["hydrophobicity"] > 0.15
                    if is_h1 and is_h2:
                        if d > min_dist and d < 6.0:
                            attract_pull = (6.0 - d) * 1.5
                            dir_ij = (self.coords[j] - self.coords[i]) / d
                            forces_other[i] += dir_ij * attract_pull
                            forces_other[j] -= dir_ij * attract_pull
                            
                    if d < min_dist:
                        rep_push = (min_dist - d) * 15.0
                        dir_ij = (self.coords[j] - self.coords[i]) / d
                        forces_other[i] -= dir_ij * rep_push
                        forces_other[j] += dir_ij * rep_push

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

            # Apply rotations based on torques
            for joint in self.joints:
                u_idx = joint["u_idx"]
                d_idx = joint["d_idx"]
                D = joint["downstream_atoms"]
                d_res = self.atom_residues.get(d_idx, 0)

                # If sequential folding is enabled, only active residue's joints can rotate
                if self.seq_folding["enabled"]:
                    if self.seq_folding["status"] == "completed":
                        continue
                    if d_res != R:
                        continue
                else:
                    if d_res == 0:
                        continue
                
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
                
                # Reduced damping and inertia scaling for highly dynamic conformation updates
                damping_const = 0.4
                effective_inertia = max(0.5, inertia * 0.05)
                d_theta = (dt * torque) / (damping_const * effective_inertia)
                
                # Clamp rotation angle to prevent numerical instabilities
                d_theta = np.clip(d_theta, -0.4, 0.4)
                
                # Apply rotation to downstream coordinates
                cos_t = np.cos(d_theta)
                sin_t = np.sin(d_theta)
                # Rodrigues rotation formula
                for idx in D:
                    r_vec = self.coords[idx] - pos_B
                    rotated = r_vec * cos_t + np.cross(axis, r_vec) * sin_t + axis * np.dot(axis, r_vec) * (1 - cos_t)
                    self.coords[idx] = pos_B + rotated
                    
            # Compute Peptide H-bonds for rendering
            # N-H donor and C=O acceptor within 3.5 Å and linear
            # Find N, H, C, O atoms
            n_atoms_idx = [i for i in range(self.n_atoms) if self.atoms[i]["element"] == 'N']
            o_atoms_idx = [i for i in range(self.n_atoms) if self.atoms[i]["element"] == 'O']
            
            for n_idx in n_atoms_idx:
                for o_idx in o_atoms_idx:
                    # Ignore if in same residue or close in graph
                    if self.graph_dist[n_idx, o_idx] < 4:
                        continue
                        
                    dist = dists[n_idx, o_idx]
                    if dist < 3.5:
                        self.active_hbonds.append([int(n_idx), int(o_idx)])
                        
            # Dynamic calculation of hydrophobic exposure
            # Defined as average distance between hydrophobic atoms and center of mass
            h_atoms = [i for i in range(self.n_atoms) if self.atoms[i]["hydrophobicity"] > 0.15]
            if h_atoms:
                com = np.mean(self.coords, axis=0)
                d_h_com = [np.linalg.norm(self.coords[i] - com) for i in h_atoms]
                self.hydrophobic_exposure = float(np.mean(d_h_com))
            else:
                self.hydrophobic_exposure = 0.0

simulations = {}

@app.route('/api/initialize', methods=['POST'])
def initialize_simulation():
    try:
        data = request.json or {}
        sequence = data.get("sequence", "Ala-Ala-Ala")
        
        sim = PeptideSimulation(sequence)
        
        # Override initial params if sent
        if "temperature" in data:
            sim.params["temperature"] = float(data["temperature"])
        if "explicitWaterEnabled" in data:
            sim.params["explicitWaterEnabled"] = bool(data["explicitWaterEnabled"])
        if "waterWaterStrength" in data:
            sim.params["waterWaterStrength"] = float(data["waterWaterStrength"])
        if "waterPeptideStrength" in data:
            sim.params["waterPeptideStrength"] = float(data["waterPeptideStrength"])
        if "bulkSolventStrength" in data:
            sim.params["bulkSolventStrength"] = float(data["bulkSolventStrength"])
        if "waterBeltContraction" in data:
            sim.params["waterBeltContraction"] = float(data["waterBeltContraction"])
        if "watersPerAtom" in data:
            sim.params["watersPerAtom"] = int(data["watersPerAtom"])
            sim.initialize_waters()
        if "kCapture" in data:
            sim.params["kCapture"] = float(data["kCapture"])
        if "entropyBias" in data:
            sim.params["entropyBias"] = float(data["entropyBias"])
            
        session_id = str(uuid.uuid4())
        simulations[session_id] = sim
        
        return jsonify({
            "status": "success",
            "sessionId": session_id,
            "atoms": sim.atoms,
            "bonds": sim.bonds,
            "initialCoords": sim.initial_coords.tolist(),
            "atomResidues": sim.atom_residues,
            "residueLabels": sim.residue_labels,
            "waters": sim.waters,
            "waterCoords": sim.water_coords.tolist() if len(sim.waters) > 0 else []
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "error": str(e)}), 400

@app.route('/api/control', methods=['POST'])
def control_simulation():
    data = request.json or {}
    session_id = data.get("sessionId")
    if not session_id or session_id not in simulations:
        return jsonify({"status": "error", "error": "Invalid session ID"}), 400
        
    sim = simulations[session_id]
    
    with sim.lock:
        if "isPlaying" in data:
            sim.params["isPlaying"] = bool(data["isPlaying"])
        if "temperature" in data:
            sim.params["temperature"] = float(data["temperature"])
        if "explicitWaterEnabled" in data:
            sim.params["explicitWaterEnabled"] = bool(data["explicitWaterEnabled"])
        if "waterWaterStrength" in data:
            sim.params["waterWaterStrength"] = float(data["waterWaterStrength"])
        if "waterPeptideStrength" in data:
            sim.params["waterPeptideStrength"] = float(data["waterPeptideStrength"])
        if "bulkSolventStrength" in data:
            sim.params["bulkSolventStrength"] = float(data["bulkSolventStrength"])
        if "kCapture" in data:
            sim.params["kCapture"] = float(data["kCapture"])
        if "entropyBias" in data:
            sim.params["entropyBias"] = float(data["entropyBias"])
        if "waterBeltContraction" in data:
            val = float(data["waterBeltContraction"])
            if sim.seq_folding["enabled"]:
                sim.seq_folding["maxContraction"] = val
                sim.params["waterBeltContraction"] = 0.0
            else:
                sim.params["waterBeltContraction"] = val
        if "watersPerAtom" in data:
            old_val = sim.params.get("watersPerAtom", 4)
            new_val = int(data["watersPerAtom"])
            if old_val != new_val:
                sim.params["watersPerAtom"] = new_val
                sim.initialize_waters()
            
        # Sequential folding control
        if "seqFoldingEnabled" in data:
            sim.seq_folding["enabled"] = bool(data["seqFoldingEnabled"])
            if sim.seq_folding["enabled"]:
                sim.seq_folding["activeResidue"] = 0
                sim.seq_folding["currentStep"] = 0
                sim.seq_folding["status"] = "folding"
                sim.seq_folding["direction"] = 1
                max_c = sim.params.get("waterBeltContraction", 0.50)
                if max_c < 0.01:
                    max_c = 0.50
                sim.seq_folding["maxContraction"] = max_c
                sim.params["waterBeltContraction"] = 0.0
                sim.logs.append("System: Sequential folding mode enabled. Folding started from Residue 1.")
            else:
                sim.seq_folding["status"] = "idle"
                sim.logs.append("System: Sequential folding mode disabled. All joints freed.")
                
        if "seqFoldingActiveResidue" in data:
            res_idx = int(data["seqFoldingActiveResidue"])
            if 0 <= res_idx < sim.n_residues:
                sim.seq_folding["activeResidue"] = res_idx
                sim.seq_folding["currentStep"] = 0
                sim.seq_folding["status"] = "folding"
                sim.seq_folding["direction"] = 1 if res_idx == 0 else sim.seq_folding.get("direction", 1)
                sim.params["waterBeltContraction"] = 0.0
                sim.logs.append(f"System: Set active folding residue to Residue {res_idx + 1} ({sim.residue_labels[res_idx]}). Other parts frozen.")
        if "seqFoldingReset" in data:
            sim.seq_folding["activeResidue"] = 0
            sim.seq_folding["currentStep"] = 0
            sim.seq_folding["status"] = "folding"
            sim.seq_folding["direction"] = 1
            sim.params["isPlaying"] = True
            sim.params["waterBeltContraction"] = 0.0
            sim.logs.append("System: Sequential folding reset. Restarting from Residue 1.")
            
    return jsonify({
        "status": "success", 
        "params": sim.params,
        "seqFolding": sim.seq_folding
    })

@app.route('/api/stream')
def stream_simulation():
    session_id = request.args.get("sessionId")
    if not session_id or session_id not in simulations:
        return Response("Invalid session ID", status=400)
        
    sim = simulations[session_id]
    
    def event_generator():
        dt = 0.02
        try:
            while True:
                if sim.params["isPlaying"]:
                    for _ in range(5):
                        sim.step(dt)
                        
                with sim.lock:
                    payload = {
                        "coords": sim.coords.tolist(),
                        "temperature": sim.params["temperature"],
                        "logs": list(sim.logs),
                        "activeHBonds": list(sim.active_hbonds),
                        "hydrophobicExposure": sim.hydrophobic_exposure,
                        "explicitWaterEnabled": sim.params["explicitWaterEnabled"],
                        "waterBeltContraction": sim.params["waterBeltContraction"],
                        "waterCoords": sim.water_coords.tolist() if len(sim.waters) > 0 else [],
                        "waters": sim.waters,
                        "waterEntropies": sim.water_entropies,
                        "waterWeights": sim.water_weights,
                        "informationCaptureIndex": sim.information_capture_index,
                        "activeInformationPaths": sim.active_information_paths,
                        "seqFolding": {
                            "enabled": sim.seq_folding["enabled"],
                            "activeResidue": sim.seq_folding["activeResidue"],
                            "currentStep": sim.seq_folding["currentStep"],
                            "stepsPerResidue": sim.seq_folding["stepsPerResidue"],
                            "status": sim.seq_folding["status"],
                            "nResidues": sim.n_residues,
                            "residueLabels": sim.residue_labels,
                            "direction": sim.seq_folding.get("direction", 1)
                        }
                    }
                    sim.logs = []
                
                yield f"data: {json.dumps(payload)}\n\n"
                time.sleep(0.033)
        except GeneratorExit:
            pass
            
    return Response(event_generator(), mimetype="text/event-stream")

if __name__ == '__main__':
    # Start on port 5003 for Entropy & Information capture mode
    app.run(host='0.0.0.0', port=5003, debug=False)
