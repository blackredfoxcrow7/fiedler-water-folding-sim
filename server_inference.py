import sys
import os
import uuid
import json
import time
import threading
import heapq
from flask import Flask, request, jsonify, Response
from rdkit import Chem
import numpy as np

# Import our Peptide Agent
from peptide_agent import PeptideAgent

app = Flask(__name__)

# CORS headers middleware
@app.after_request
def add_cors_headers(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

class PeptideInferenceSimulation:
    def __init__(self, sequence):
        self.lock = threading.Lock()
        self.sequence = sequence
        self.step_counter = 0
        self.logs = []
        
        # 1. Initialize Peptide Agent
        self.peptide = PeptideAgent(sequence)
        
        # Mirror peptide properties for easy access
        self.n_atoms = self.peptide.n_atoms
        self.atoms = self.peptide.atoms
        self.bonds = self.peptide.bonds
        self.joints = self.peptide.joints
        self.atom_residues = self.peptide.atom_residues
        self.residue_labels = self.peptide.residue_labels
        self.n_residues = self.peptide.n_residues
        self.graph_dist = self.peptide.graph_dist
        
        # Link coordinates directly to the peptide agent's coordinates
        # So modifications to self.coords affect self.peptide.coords
        self.coords = self.peptide.coords
        
        # Simulation parameters
        self.params = {
            "temperature": 0.15,
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
        
        # Sequential Step control
        self.folding_status = "idle"         # "idle", "learning", "folding", "completed"
        self.learning_mode = False           # Sequential C-to-N unfolding
        self.inference_mode = False          # Sequential N-to-C folding
        self.active_residue = 0
        self.step_counter_residue = 0
        self.steps_per_residue = 300         # Time window per amino acid stage
        
        # Solvation hysteresis trajectory log
        self.trajectory = []
        
        # Caching variables for Dijkstra path updates
        self.cached_active_paths = []
        self.cached_I_capture = 0.0
        self.cached_water_entropies = []
        self.cached_water_weights = []
        self.cached_unconnected_pairs = []
        
        # Initialize explicit solvent waters
        self.initialize_waters()

    def initialize_waters(self):
        self.waters = []
        water_coords_list = []
        for i in range(0, self.n_atoms, 2):
            pos = self.coords[i]
            is_h = self.atoms[i]["hydrophobicity"] > 0.15
            is_p = self.atoms[i]["h_bond"] in ("donor", "acceptor", "both")
            if is_h or is_p:
                n_w = 4
                for w in range(n_w):
                    w_id = len(self.waters)
                    r_shell = 3.8 + np.random.uniform(-0.3, 0.3)
                    theta = np.random.uniform(0, 2*np.pi)
                    phi = np.random.uniform(0, np.pi)
                    o_pos = pos + np.array([
                        r_shell * np.sin(phi) * np.cos(theta),
                        r_shell * np.cos(phi),
                        r_shell * np.sin(phi) * np.sin(theta)
                    ])
                    u = np.random.normal(0, 1, 3)
                    u /= np.linalg.norm(u)
                    v = np.random.normal(0, 1, 3)
                    v = v - np.dot(v, u)*u
                    v /= np.linalg.norm(v)
                    h1_pos = o_pos + 0.96 * u
                    h2_pos = o_pos + 0.96 * (np.cos(104.5 * np.pi / 180) * u + np.sin(104.5 * np.pi / 180) * v)
                    self.waters.append({"id": w_id, "parent_atom_id": i})
                    water_coords_list.append([o_pos.tolist(), h1_pos.tolist(), h2_pos.tolist()])
        self.water_coords = np.array(water_coords_list)

    def load_pdb_conformation(self, pdb_path):
        try:
            pdb_mol = Chem.MolFromPDBFile(pdb_path, removeHs=True)
            if not pdb_mol:
                return False
            pdb_conf = pdb_mol.GetConformer()
            for i in range(min(self.n_atoms, pdb_mol.GetNumAtoms())):
                pos = pdb_conf.GetAtomPosition(i)
                self.coords[i] = [pos.x, pos.y, pos.z]
            self.peptide.center_around_first_residue()
            self.peptide.align_joint_angles_to_coords(self.coords)
            self.initialize_waters()
            return True
        except Exception as e:
            self.logs.append(f"System Error: Failed to load PDB - {str(e)}")
            return False

    def step_explicit_waters(self, dt):
        if not self.params.get("explicitWaterEnabled", True) or len(self.waters) == 0:
            self.water_entropies = []
            self.water_weights = []
            self.information_capture_index = 0.0
            self.active_information_paths = []
            return np.zeros((self.n_atoms, 3))
            
        num_waters = len(self.waters)
        forces_water = np.zeros((num_waters, 3, 3))
        forces_reaction_on_peptide = np.zeros((self.n_atoms, 3))
        
        k_pw = self.params.get("waterPeptideStrength", 0.60) * 15.0
        k_ww = self.params.get("waterWaterStrength", 0.50) * 25.0
        k_capture = self.params.get("kCapture", 0.70) * 12.0
        
        # 1. Internal Geometry Constraints
        k_int = 250.0
        pos_o = self.water_coords[:, 0]
        pos_h1 = self.water_coords[:, 1]
        pos_h2 = self.water_coords[:, 2]
        
        # O - H1
        diff_h1 = pos_h1 - pos_o
        d_h1 = np.linalg.norm(diff_h1, axis=1, keepdims=True)
        safe_d_h1 = np.where(d_h1 > 1e-3, d_h1, 1.0)
        f_h1 = (d_h1 - 0.96) * k_int * (diff_h1 / safe_d_h1)
        f_h1 = np.where(d_h1 > 1e-3, f_h1, 0.0)
        forces_water[:, 1] -= f_h1
        forces_water[:, 0] += f_h1
        
        # O - H2
        diff_h2 = pos_h2 - pos_o
        d_h2 = np.linalg.norm(diff_h2, axis=1, keepdims=True)
        safe_d_h2 = np.where(d_h2 > 1e-3, d_h2, 1.0)
        f_h2 = (d_h2 - 0.96) * k_int * (diff_h2 / safe_d_h2)
        f_h2 = np.where(d_h2 > 1e-3, f_h2, 0.0)
        forces_water[:, 2] -= f_h2
        forces_water[:, 0] += f_h2
        
        # H1 - H2
        diff_hh = pos_h2 - pos_h1
        d_hh = np.linalg.norm(diff_hh, axis=1, keepdims=True)
        safe_d_hh = np.where(d_hh > 1e-3, d_hh, 1.0)
        f_hh = (d_hh - 1.52) * k_int * (diff_hh / safe_d_hh)
        f_hh = np.where(d_hh > 1e-3, f_hh, 0.0)
        forces_water[:, 2] -= f_hh
        forces_water[:, 1] += f_hh

        # 2. Covalent parent tracking force (Spacious: 3.8 Å)
        parent_ids = np.array([w["parent_atom_id"] for w in self.waters])
        pos_p = self.coords[parent_ids]
        diff_parent = pos_o - pos_p
        d_parent = np.linalg.norm(diff_parent, axis=1, keepdims=True)
        safe_d_parent = np.where(d_parent > 1e-3, d_parent, 1.0)
        f_parent = (d_parent - 3.8) * k_pw * (diff_parent / safe_d_parent)
        f_parent = np.where(d_parent > 1e-3, f_parent, 0.0)
        
        forces_water[:, 0] -= f_parent
        for i in range(num_waters):
            forces_reaction_on_peptide[parent_ids[i]] += f_parent[i]

        # 3. Dynamic Entropy & Connectivity
        w_pos = self.water_coords[:, 0]
        run_dijkstra = (self.step_counter % 3 == 0) or not self.cached_water_entropies
        polar_indices = [idx for idx in range(self.n_atoms) if self.atoms[idx]["h_bond"] in ("donor", "acceptor", "both")]
        
        if run_dijkstra:
            hydrophobic_indices = [idx for idx in range(self.n_atoms) if self.atoms[idx]["hydrophobicity"] > 0.15]
            entropy_bias = self.params.get("entropyBias", 1.2)
            
            p_coords = self.coords[polar_indices] if polar_indices else np.empty((0, 3))
            h_coords = self.coords[hydrophobic_indices] if hydrophobic_indices else np.empty((0, 3))
            
            if len(polar_indices) > 0:
                d_polar_matrix = np.linalg.norm(w_pos[:, np.newaxis, :] - p_coords[np.newaxis, :, :], axis=2)
                d_polar_min = np.min(d_polar_matrix, axis=1)
            else:
                d_polar_matrix = np.empty((num_waters, 0))
                d_polar_min = np.full(num_waters, 99.0)
                
            if len(hydrophobic_indices) > 0:
                d_hydro_matrix = np.linalg.norm(w_pos[:, np.newaxis, :] - h_coords[np.newaxis, :, :], axis=2)
                d_hydro_min = np.min(d_hydro_matrix, axis=1)
            else:
                d_hydro_min = np.full(num_waters, 99.0)
                
            f_polar = np.exp(- (d_polar_min ** 2) / (2 * 2.2 ** 2))
            f_hydro = np.exp(- (d_hydro_min ** 2) / (2 * 2.2 ** 2))
            
            water_entropies = 0.6 + 0.4 * f_hydro * entropy_bias - 0.5 * f_polar * entropy_bias
            water_entropies = np.clip(water_entropies, 0.05, 1.2).tolist()
            
            water_weights = 1.0 + 1.5 * f_polar - 0.85 * f_hydro
            water_weights = np.clip(water_weights, 0.05, 4.0).tolist()
            
            self.water_entropies = water_entropies
            self.water_weights = water_weights
            self.cached_water_entropies = water_entropies
            self.cached_water_weights = water_weights
            
            # 4. Graph Construction & Dijkstra Solver
            num_graph_nodes = num_waters + len(polar_indices)
            graph_edges = {u: [] for u in range(num_graph_nodes)}
            
            dist_ww = np.linalg.norm(w_pos[:, np.newaxis, :] - w_pos[np.newaxis, :, :], axis=2)
            ii, jj = np.where((dist_ww < 3.5) & (np.arange(num_waters)[:, np.newaxis] < np.arange(num_waters)[np.newaxis, :]))
            for i, j in zip(ii, jj):
                w_ij = np.sqrt(water_weights[i] * water_weights[j])
                cost = 1.0 / (w_ij + 1e-5)
                graph_edges[i].append((j, cost))
                graph_edges[j].append((i, cost))
                
            if len(polar_indices) > 0:
                ii, jj = np.where(d_polar_matrix < 3.5)
                for i, p_idx in zip(ii, jj):
                    w_ip = np.sqrt(water_weights[i] * 2.5)
                    cost = 1.0 / (w_ip + 1e-5)
                    graph_node_p = num_waters + p_idx
                    graph_edges[i].append((graph_node_p, cost))
                    graph_edges[graph_node_p].append((i, cost))

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

            polar_graph_nodes = [num_waters + idx for idx in range(len(polar_indices))]
            paths_by_src = {}
            for src in polar_graph_nodes:
                paths_by_src[src] = dijkstra(src)
                
            I_capture = 0.0
            path_edges_set = set()
            
            for idx_a in range(len(polar_indices)):
                node_a = polar_graph_nodes[idx_a]
                dists, parents = paths_by_src[node_a]
                for idx_b in range(idx_a + 1, len(polar_indices)):
                    node_b = polar_graph_nodes[idx_b]
                    d_ab = dists.get(node_b, float('inf'))
                    if d_ab < 999.0:
                        I_capture += 1.0 / d_ab
                        curr = node_b
                        while curr is not None:
                            prev = parents[curr]
                            if prev is not None:
                                edge = (min(curr, prev), max(curr, prev))
                                path_edges_set.add(edge)
                            curr = prev

            self.information_capture_index = I_capture
            self.cached_I_capture = I_capture
            
            active_path_list = []
            for u, v in path_edges_set:
                u_val = int(u)
                v_val = int(v)
                if u_val < num_waters and v_val < num_waters:
                    active_path_list.append({
                        "u": u_val, "v": v_val, "type": 0,
                        "posU": self.water_coords[u_val, 0].tolist(),
                        "posV": self.water_coords[v_val, 0].tolist()
                    })
                elif u_val < num_waters and v_val >= num_waters:
                    p_idx = v_val - num_waters
                    p_node = int(polar_indices[p_idx])
                    active_path_list.append({
                        "u": u_val, "v": p_node, "type": 1,
                        "posU": self.water_coords[u_val, 0].tolist(),
                        "posV": self.coords[p_node].tolist()
                    })
                elif u_val >= num_waters and v_val < num_waters:
                    p_idx = u_val - num_waters
                    p_node = int(polar_indices[p_idx])
                    active_path_list.append({
                        "u": p_node, "v": v_val, "type": 1,
                        "posU": self.coords[p_node].tolist(),
                        "posV": self.water_coords[v_val, 0].tolist()
                    })
            self.active_information_paths = active_path_list
            self.cached_active_paths = active_path_list
        else:
            self.water_entropies = self.cached_water_entropies
            self.water_weights = self.cached_water_weights
            self.information_capture_index = self.cached_I_capture
            
            # Map coordinate references dynamically using cached paths
            active_path_list = []
            for edge in self.cached_active_paths:
                u, v, t = int(edge["u"]), int(edge["v"]), int(edge["type"])
                if t == 0:
                    posU = self.water_coords[u, 0].tolist()
                    posV = self.water_coords[v, 0].tolist()
                else:
                    posU = self.water_coords[u, 0].tolist() if u < num_waters else self.coords[u].tolist()
                    posV = self.coords[v].tolist() if v < self.n_atoms else self.water_coords[v, 0].tolist()
                active_path_list.append({
                    "u": u, "v": v, "type": t,
                    "posU": posU,
                    "posV": posV
                })
            self.active_information_paths = active_path_list

        # 5. Apply Information Routing Path Contraction Forces
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

        # 6. Unconnected search force
        if run_dijkstra:
            unconnected_pairs = []
            for idx_a in range(len(polar_indices)):
                node_a = polar_graph_nodes[idx_a]
                dists, _ = paths_by_src[node_a]
                for idx_b in range(idx_a + 1, len(polar_indices)):
                    node_b = polar_graph_nodes[idx_b]
                    if dists.get(node_b, float('inf')) > 999.0:
                        unconnected_pairs.append((polar_indices[idx_a], polar_indices[idx_b]))
            self.cached_unconnected_pairs = unconnected_pairs
        else:
            unconnected_pairs = self.cached_unconnected_pairs
                    
        if unconnected_pairs and num_waters > 0:
            p1_indices = np.array([p[0] for p in unconnected_pairs])
            p2_indices = np.array([p[1] for p in unconnected_pairs])
            midpoints = 0.5 * (self.coords[p1_indices] + self.coords[p2_indices])
            
            diff_mid = midpoints[:, np.newaxis, :] - w_pos[np.newaxis, :, :]
            d_mid = np.linalg.norm(diff_mid, axis=2)
            mid_mask = (d_mid > 0.1) & (d_mid < 8.0)
            
            if np.any(mid_mask):
                safe_d_mid = np.where(d_mid > 1e-5, d_mid, 1.0)
                f_dir_mid = diff_mid / safe_d_mid[:, :, np.newaxis]
                f_mag = k_capture * 0.03
                forces_to_apply = f_dir_mid * (f_mag * mid_mask)[:, :, np.newaxis]
                forces_water[:, 0] += np.sum(forces_to_apply, axis=0)

        # 7. Classical Steric Repulsions
        dist_ww = np.linalg.norm(w_pos[:, np.newaxis, :] - w_pos[np.newaxis, :, :], axis=2)
        ii, jj = np.where((dist_ww < 2.8) & (np.arange(num_waters)[:, np.newaxis] < np.arange(num_waters)[np.newaxis, :]))
        for i, j in zip(ii, jj):
            diff = w_pos[j] - w_pos[i]
            d = dist_ww[i, j]
            if d > 1e-3:
                rep = (2.8 - d) * 30.0 * (diff / d)
                forces_water[i, 0] -= rep
                forces_water[j, 0] += rep

        diff_wp = self.coords[np.newaxis, :, :] - w_pos[:, np.newaxis, :]
        dist_wp = np.linalg.norm(diff_wp, axis=2)
        
        radii = np.array([a["radius"] for a in self.atoms])
        min_dists = radii + 1.2
        
        overlap_mask = dist_wp < min_dists[np.newaxis, :]
        ww, pp = np.where(overlap_mask)
        for w_idx, p_idx in zip(ww, pp):
            d = dist_wp[w_idx, p_idx]
            if d > 1e-3:
                diff = diff_wp[w_idx, p_idx]
                rep = (min_dists[p_idx] - d) * 35.0 * (diff / d)
                forces_water[w_idx, 0] -= rep
                forces_reaction_on_peptide[p_idx] += rep

        # Euler step for waters
        noise = np.random.normal(0, 0.05, size=(num_waters, 3, 3))
        self.water_coords += (forces_water + noise) * (dt * dt)
                
        return forces_reaction_on_peptide

    def calculate_radius_of_gyration(self):
        com = np.mean(self.coords, axis=0)
        sq_dist = np.sum((self.coords - com) ** 2)
        return np.sqrt(sq_dist / self.n_atoms)

    def step(self, dt):
        with self.lock:
            self.step_counter += 1
            temp = self.params.get("temperature", 0.15)
            
            # 1. Step explicit water solvers
            forces_reaction = np.zeros((self.n_atoms, 3))
            if self.params.get("explicitWaterEnabled", True):
                forces_reaction = self.step_explicit_waters(dt)
                
            # 2. Classical intramolecular peptide forces
            diff = self.coords[:, np.newaxis, :] - self.coords[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            
            forces_other = np.zeros((self.n_atoms, 3))
            
            # Thermal noise
            if temp > 0:
                noise = np.random.normal(0, temp * 3.5, size=(self.n_atoms, 3))
                if self.learning_mode or self.inference_mode:
                    active_mask = np.array([self.atom_residues.get(i, 0) == self.active_residue for i in range(self.n_atoms)])
                    forces_other[active_mask] += noise[active_mask]
                else:
                    forces_other += noise

            # Bulk Solvent Pressure (Exclusion force)
            bulk_solv = self.params.get("bulkSolventStrength", 0.45)
            if bulk_solv > 1e-4:
                com = np.mean(self.coords, axis=0)
                dir_com = com - self.coords
                d_com = np.linalg.norm(dir_com, axis=1, keepdims=True)
                
                hydrophobic_mask = np.array([a["hydrophobicity"] > 0.15 for a in self.atoms])
                polar_mask = np.array([a["h_bond"] in ("donor", "acceptor", "both") for a in self.atoms])
                valid = (d_com.flatten() > 0.1)
                
                h_indices = np.where(hydrophobic_mask & valid)[0]
                forces_other[h_indices] += (dir_com[h_indices] / d_com[h_indices]) * bulk_solv * 6.0
                
                p_indices = np.where(polar_mask & valid)[0]
                forces_other[p_indices] -= (dir_com[p_indices] / d_com[p_indices]) * bulk_solv * 4.0

            # Pairwise steric & hydrophobic attraction
            radii = np.array([a["radius"] for a in self.atoms])
            min_dist_matrix = (radii[:, np.newaxis] + radii[np.newaxis, :]) * 0.45
            interact_mask = (self.graph_dist > 2) & (dists > 1e-3)
            
            hydrophobic_mask = np.array([a["hydrophobicity"] > 0.15 for a in self.atoms])
            hydro_both_mask = interact_mask & hydrophobic_mask[:, np.newaxis] & hydrophobic_mask[np.newaxis, :]
            steric_mask = interact_mask & (dists < min_dist_matrix)
            
            safe_dists = np.where(dists > 1e-5, dists, 1.0)
            dir_ij = diff / safe_dists[:, :, np.newaxis]
            
            if np.any(hydro_both_mask):
                attract_pull = (6.0 - dists) * 1.5
                hydro_attract_mask = hydro_both_mask & (dists > min_dist_matrix) & (dists < 6.0)
                attract_forces = dir_ij * (attract_pull * hydro_attract_mask)[:, :, np.newaxis]
                forces_other += np.sum(attract_forces, axis=1)
                forces_other -= np.sum(attract_forces, axis=0)
                
            if np.any(steric_mask):
                rep_push = (min_dist_matrix - dists) * 15.0
                rep_forces = dir_ij * (rep_push * steric_mask)[:, :, np.newaxis]
                forces_other -= np.sum(rep_forces, axis=1)
                forces_other += np.sum(rep_forces, axis=0)

            # Combine forces
            forces = forces_other + forces_reaction
            
            # Clamp forces
            force_norms = np.linalg.norm(forces, axis=1)
            max_f = 25.0
            mask_large_f = force_norms > max_f
            if np.any(mask_large_f):
                scale_factors = np.ones(self.n_atoms)
                scale_factors[mask_large_f] = max_f / force_norms[mask_large_f]
                forces_other *= scale_factors[:, np.newaxis]
                forces_reaction *= scale_factors[:, np.newaxis]
                forces = forces_other + forces_reaction

            # 3. Apply Joint Rotations via Peptide Agent
            if self.learning_mode:
                # Sequential Reverse learning (C -> N)
                self.peptide.apply_rotations(dt, forces_reaction, forces_other, active_residue=self.active_residue, reverse=True, use_inference=False)
                self.step_counter_residue += 1
                
                # Check if learning time window for current residue is finished
                if self.step_counter_residue >= self.steps_per_residue:
                    self.peptide.learn_residue_step(self.active_residue)
                    self.step_counter_residue = 0
                    self.active_residue -= 1
                    
                    if self.active_residue < 0:
                        # Unfolding and learning finished!
                        self.learning_mode = False
                        self.folding_status = "completed"
                        self.logs.append("AI Learning: Completed sequential reverse learning cycle! Extended state re-initialized.")
                        # Reset to extended state ready for inference folding
                        self.peptide.reset_to_initial()
                        self.initialize_waters()
                    else:
                        self.logs.append(f"AI Learning: Active Residue shifted to {self.active_residue + 1} ({self.residue_labels[self.active_residue]})")
            
            elif self.inference_mode:
                # Sequential Forward inference (N -> C)
                self.peptide.apply_rotations(dt, forces_reaction, forces_other, active_residue=self.active_residue, reverse=False, use_inference=True)
                self.step_counter_residue += 1
                
                # Check if inference window for current residue is finished
                if self.step_counter_residue >= self.steps_per_residue:
                    self.step_counter_residue = 0
                    self.active_residue += 1
                    
                    if self.active_residue >= self.n_residues:
                        # Folding completed!
                        self.inference_mode = False
                        self.folding_status = "completed"
                        self.params["temperature"] = 0.02 # Cool down to stabilize
                        self.logs.append("AI Inference: Sequential folding completed! Automatically cooling down to stabilize.")
                    else:
                        self.logs.append(f"AI Inference: Active Residue shifted to {self.active_residue + 1} ({self.residue_labels[self.active_residue]})")
            
            else:
                # Ordinary physics mode (all residues free)
                self.peptide.apply_rotations(dt, forces_reaction, forces_other, active_residue=None, reverse=False, use_inference=False)

            # Record trajectory coordinates for hysteresis plot
            Rg = self.calculate_radius_of_gyration()
            mode_label = "learning" if self.learning_mode else ("inference" if self.inference_mode else "physics")
            self.trajectory.append({
                "rg": float(Rg),
                "I_capture": float(self.information_capture_index),
                "mode": mode_label
            })
            # Limit trajectory log size
            if len(self.trajectory) > 5000:
                self.trajectory.pop(0)

# Global simulator instance
sim_instance = None
session_id = None
recent_session_ids = []

@app.route('/api/initialize', methods=['POST'])
def initialize():
    global sim_instance, session_id, recent_session_ids
    try:
        req_data = request.json or {}
        sequence = req_data.get("sequence", "GYDPETGTWG") # Chignolin validation default
        
        sim_instance = PeptideInferenceSimulation(sequence)
        import uuid
        session_id = str(uuid.uuid4())
        recent_session_ids.append(session_id)
        if len(recent_session_ids) > 10:
            recent_session_ids.pop(0)
        
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

@app.route('/api/train_reverse', methods=['POST'])
def train_reverse():
    global sim_instance
    if not sim_instance:
        return jsonify({"status": "error", "error": "Simulation not initialized"}), 400
    try:
        with sim_instance.lock:
            # 1. Force the conformation into the folded state from validation PDB (if 1UAO)
            has_pdb = False
            if sim_instance.sequence == "GYDPETGTWG" or "1UAO" in sim_instance.sequence:
                if os.path.exists("1UAO.pdb"):
                    has_pdb = sim_instance.load_pdb_conformation("1UAO.pdb")
                    
            if not has_pdb:
                # If no PDB, we just reset to the current coordinates as the folded state reference
                sim_instance.initialize_waters()
                
            # 2. Setup sequential learning parameters
            sim_instance.learning_mode = True
            sim_instance.inference_mode = False
            sim_instance.folding_status = "learning"
            sim_instance.active_residue = sim_instance.n_residues - 1 # Start from C-terminus
            sim_instance.step_counter_residue = 0
            sim_instance.trajectory = [] # Reset hysteresis plot trajectory
            
            sim_instance.logs.append(f"AI Learning: Commencing C-to-N sequential reverse learning. Active residue: {sim_instance.active_residue + 1} ({sim_instance.residue_labels[sim_instance.active_residue]})")
            
        return jsonify({"status": "success", "foldingStatus": sim_instance.folding_status})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 400

@app.route('/api/infer_forward', methods=['POST'])
def infer_forward():
    global sim_instance
    if not sim_instance:
        return jsonify({"status": "error", "error": "Simulation not initialized"}), 400
    try:
        with sim_instance.lock:
            # 1. Reset coordinates to extended conformation
            sim_instance.peptide.reset_to_initial()
            sim_instance.initialize_waters()
            
            # 2. Setup sequential folding parameters
            sim_instance.learning_mode = False
            sim_instance.inference_mode = True
            sim_instance.folding_status = "folding"
            sim_instance.active_residue = 0 # Start from N-terminus
            sim_instance.step_counter_residue = 0
            sim_instance.params["temperature"] = 0.15 # Set optimal folding noise
            
            sim_instance.logs.append(f"AI Inference: Commencing N-to-C sequential autonomous folding. Active residue: {sim_instance.active_residue + 1} ({sim_instance.residue_labels[sim_instance.active_residue]})")
            
        return jsonify({"status": "success", "foldingStatus": sim_instance.folding_status})
    except Exception as e:
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
            if "clearTrajectory" in req_data:
                sim_instance.trajectory = []
                
        return jsonify({"status": "success", "params": sim_instance.params})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 400

@app.route('/api/stream')
def stream():
    req_session_id = request.args.get("sessionId")
    if not req_session_id or req_session_id not in recent_session_ids:
        return "Invalid session", 400
        
    def event_generator():
        global sim_instance
        dt = 0.02
        try:
            while True:
                if sim_instance.params["isPlaying"]:
                    # Run 5 physics steps per stream packet
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
                        
                        # Active residue step and learning status
                        "foldingStatus": sim_instance.folding_status,
                        "learningMode": sim_instance.learning_mode,
                        "inferenceMode": sim_instance.inference_mode,
                        "activeResidue": sim_instance.active_residue,
                        "stepProgress": float(sim_instance.step_counter_residue) / sim_instance.steps_per_residue,
                        
                        # Hysteresis trajectory data
                        "trajectory": sim_instance.trajectory[-300:] # Send last 300 points for the chart
                    }
                    sim_instance.logs = [] # Clear logs after streaming
                    
                yield f"data: {json.dumps(payload)}\n\n"
                time.sleep(0.06)
        except GeneratorExit:
            pass
            
    return Response(event_generator(), mimetype="text/event-stream")

if __name__ == "__main__":
    # Port 5005 for the autonomous inference simulator
    app.run(host="0.0.0.0", port=5005, debug=False)
