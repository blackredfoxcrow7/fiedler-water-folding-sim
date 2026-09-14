import json
import time
import uuid
import threading
import numpy as np
from flask import Flask, jsonify, request, Response
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

app = Flask(__name__)

# CORS middleware for local development
@app.after_request
def add_cors_headers(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

def is_amide_bond(mol, bond):
    """
    Checks if a bond is a peptide (amide) bond.
    Amide bond is a C-N single bond where the carbon is double-bonded to an oxygen.
    """
    if bond.GetBondType() != Chem.BondType.SINGLE:
        return False
    a1 = bond.GetBeginAtom()
    a2 = bond.GetEndAtom()
    if {a1.GetSymbol(), a2.GetSymbol()} == {'C', 'N'}:
        c_atom = a1 if a1.GetSymbol() == 'C' else a2
        for nbr in c_atom.GetNeighbors():
            if nbr.GetSymbol() == 'O':
                b = mol.GetBondBetweenAtoms(c_atom.GetIdx(), nbr.GetIdx())
                if b.GetBondType() == Chem.BondType.DOUBLE:
                    return True
    return False

class PeptideSimulation:
    def __init__(self, sequence_or_smiles):
        # 1. Parse and build molecule
        self.mol = self.parse_molecule(sequence_or_smiles)
        # 2. Add Hs, Embed 3D, MMFF Optimize, Remove Hs (maintaining heavy coords)
        self.initialize_coordinates()
        # 3. Extract topology, bonds, and joints
        self.extract_topology()
        # 4. Assign atoms to residues and create names
        self.assign_residues()
        
        # Mark rotatable bonds and their associated residue
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
        
        # Center coordinates around the first residue to make it a stationary anchor
        self.center_around_first_residue()
        
        # Simulation parameters
        self.params = {
            "temperature": 0.08,
            "isPlaying": True,
            "explicitWaterEnabled": True,
            "waterWaterStrength": 0.35,
            "waterPeptideStrength": 0.40,
            "waterFoldingTransition": False,
            "bulkSolventStrength": 0.0,
            "waterBeltContraction": 0.0,
            "watersPerAtom": 2
        }
        self.hydrophobic_exposure = 0.0
        
        # Sequential folding parameters
        self.seq_folding = {
            "enabled": False,
            "activeResidue": 0,       # 0-indexed active residue index
            "stepsPerResidue": 500,   # Number of simulation steps before auto-transitioning
            "currentStep": 0,         # Current step counter
            "status": "idle",         # "folding", "completed", "idle"
            "direction": 1            # 1 for N->C, -1 for C->N
        }
        
        self.logs = []
        self.active_hbonds = []
        self.active_water_bonds = []
        self.active_water_peptide_bonds = []
        self.lock = threading.Lock()
        self.step_counter = 0
        
        # 5. Initialize water particles
        self.initialize_waters()
        
        # Initial log message
        self.logs.append("PeptideModel: Model loaded successfully. Ready to interact with Water Solvent Agent.")
        self.logs.append(f"PeptideModel: Defined {len(self.atoms)} heavy atoms, {len(self.bonds)} bonds, and {len(self.joints)} rotatable joints.")
        self.logs.append(f"PeptideModel: Identified {self.n_residues} residues: {', '.join(self.residue_labels)}.")

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

        # Center coordinates initially around (0,0,0)
        init_com = np.mean(self.coords, axis=0)
        self.coords -= init_com
        self.initial_coords = self.coords.copy()

    def extract_topology(self):
        self.atoms = []
        crippen_contribs = rdMolDescriptors._CalcCrippenContribs(self.mol_heavy)
        
        # Heavy atom smarts for donors (N or O with at least one implicit/explicit H)
        donor_query = Chem.MolFromSmarts("[N,O;!H0]")
        acceptor_query = Chem.MolFromSmarts("[N,O]")
        
        donors = {m[0] for m in self.mol_heavy.GetSubstructMatches(donor_query)}
        acceptors = {m[0] for m in self.mol_heavy.GetSubstructMatches(acceptor_query)}
        
        vdw_radii = {"C": 1.7, "N": 1.55, "O": 1.5, "S": 1.8, "P": 1.8}
        
        for i in range(self.n_atoms):
            atom = self.mol_heavy.GetAtomWithIdx(i)
            element = atom.GetSymbol()
            logp = crippen_contribs[i][0]
            
            h_bond = "none"
            is_d = i in donors
            is_a = i in acceptors
            if is_d and is_a:
                h_bond = "both"
            elif is_d:
                h_bond = "donor"
            elif is_a:
                h_bond = "acceptor"
                
            radius = vdw_radii.get(element, 1.5)
            self.atoms.append({
                "id": i,
                "element": element,
                "hydrophobicity": float(logp),
                "h_bond": h_bond,
                "radius": radius
            })
            
        self.bonds = []
        for bond in self.mol_heavy.GetBonds():
            source = bond.GetBeginAtomIdx()
            target = bond.GetEndAtomIdx()
            is_amide = is_amide_bond(self.mol_heavy, bond)
            
            pos_s = self.coords[source]
            pos_t = self.coords[target]
            length = np.linalg.norm(pos_s - pos_t)
            
            self.bonds.append({
                "source": source,
                "target": target,
                "length": float(length),
                "is_amide": is_amide
            })
            
        self.adj = {i: [] for i in range(self.n_atoms)}
        for b in self.bonds:
            self.adj[b['source']].append(b['target'])
            self.adj[b['target']].append(b['source'])
            
        self.graph_dist = self.compute_graph_distances()
        
        # Build and order rotatable joints
        self.joints = []
        visited = {0}
        queue = [0]
        
        while queue:
            curr = queue.pop(0)
            for nbr in self.adj[curr]:
                if nbr not in visited:
                    visited.add(nbr)
                    
                    bond_obj = self.mol_heavy.GetBondBetweenAtoms(curr, nbr)
                    is_rot = False
                    if bond_obj:
                        is_single = bond_obj.GetBondType() == Chem.BondType.SINGLE
                        in_ring = bond_obj.IsInRing()
                        deg_curr = self.mol_heavy.GetAtomWithIdx(curr).GetDegree()
                        deg_nbr = self.mol_heavy.GetAtomWithIdx(nbr).GetDegree()
                        is_amide = is_amide_bond(self.mol_heavy, bond_obj)
                        
                        if is_single and not in_ring and deg_curr > 1 and deg_nbr > 1 and not is_amide:
                            is_rot = True
                            
                    if is_rot:
                        ds_atoms = self.get_downstream_atoms(nbr, curr)
                        self.joints.append({
                            "u_idx": curr,
                            "d_idx": nbr,
                            "downstream_atoms": ds_atoms
                        })
                    queue.append(nbr)

    def compute_graph_distances(self):
        dist_matrix = np.full((self.n_atoms, self.n_atoms), 999, dtype=int)
        for i in range(self.n_atoms):
            dist_matrix[i, i] = 0
            queue = [i]
            visited = {i}
            while queue:
                curr = queue.pop(0)
                d = dist_matrix[i, curr]
                for nbr in self.adj[curr]:
                    if nbr not in visited:
                        visited.add(nbr)
                        dist_matrix[i, nbr] = d + 1
                        queue.append(nbr)
        return dist_matrix

    def get_downstream_atoms(self, start, avoid):
        visited = set()
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for nbr in self.adj[node]:
                if nbr != avoid and nbr not in visited:
                    queue.append(nbr)
        return list(visited)

    def assign_residues(self):
        self.atom_residues = {}
        has_monomer_info = False
        res_nums = []
        for i in range(self.n_atoms):
            atom = self.mol_heavy.GetAtomWithIdx(i)
            info = atom.GetMonomerInfo()
            if info:
                res_num = info.GetResidueNumber()
                self.atom_residues[i] = res_num
                res_nums.append(res_num)
                has_monomer_info = True
                
        if has_monomer_info:
            unique_res = sorted(list(set(res_nums)))
            mapping = {r: idx for idx, r in enumerate(unique_res)}
            for i in range(self.n_atoms):
                self.atom_residues[i] = mapping[self.atom_residues[i]]
            self.n_residues = len(unique_res)
        else:
            self.atom_residues = {i: 0 for i in range(self.n_atoms)}
            amide_bonds = set()
            for bond in self.mol_heavy.GetBonds():
                if is_amide_bond(self.mol_heavy, bond):
                    source = bond.GetBeginAtomIdx()
                    target = bond.GetEndAtomIdx()
                    amide_bonds.add((source, target))
                    amide_bonds.add((target, source))
                    
            visited = {0}
            queue = [(0, 0)]
            max_res = 0
            while queue:
                curr, res = queue.pop(0)
                self.atom_residues[curr] = res
                max_res = max(max_res, res)
                for nbr in self.adj[curr]:
                    if nbr not in visited:
                        visited.add(nbr)
                        next_res = res + 1 if (curr, nbr) in amide_bonds else res
                        queue.append((nbr, next_res))
            self.n_residues = max_res + 1

        # Look up residue names from monomer info
        residue_names = {}
        for i in range(self.n_atoms):
            atom = self.mol_heavy.GetAtomWithIdx(i)
            info = atom.GetMonomerInfo()
            if info:
                res_idx = self.atom_residues[i]
                res_name = info.GetResidueName().strip().capitalize()
                residue_names[res_idx] = res_name
                
        self.residue_labels = []
        for i in range(self.n_residues):
            name = residue_names.get(i, "Res")
            self.residue_labels.append(f"{name}{i+1}")

    def center_around_first_residue(self):
        res0_indices = [i for i in range(self.n_atoms) if self.atom_residues.get(i, 0) == 0]
        anchor_pos = np.mean(self.coords[res0_indices], axis=0) if res0_indices else self.coords[0]
        self.coords -= anchor_pos
        self.initial_coords = self.coords.copy()

    def initialize_waters(self):
        # Identify all hydrophilic atoms (H-bond donor/acceptor/both)
        self.waters = []
        water_coords_list = []
        
        # We sort hydrophilic atoms by their residue index, then by atom index to keep a sequential belt.
        hydrophilic_atom_indices = []
        for i in range(self.n_atoms):
            a_i = self.atoms[i]
            if a_i["h_bond"] in ("donor", "acceptor", "both"):
                hydrophilic_atom_indices.append((self.atom_residues.get(i, 0), i))
        
        # Sort by residue index, then atom index
        hydrophilic_atom_indices.sort()
        
        n_waters = int(self.params.get("watersPerAtom", 2))
        
        # Generate N points uniformly on a sphere
        dirs = []
        for i in range(n_waters):
            # Fibonacci spiral on sphere
            phi = np.arccos(1.0 - 2.0 * (i + 0.5) / n_waters)
            theta = np.pi * (1.0 + 5.0 ** 0.5) * (i + 0.5)
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            dirs.append(np.array([x, y, z]))

        w_idx = 0
        for (res_idx, atom_idx) in hydrophilic_atom_indices:
            parent_pos = self.coords[atom_idx]
            for v in dirs:
                jitter = np.random.normal(0, 0.05, 3)
                final_dir = v + jitter
                final_dir /= np.linalg.norm(final_dir)
                
                water_pos = parent_pos + final_dir * 2.6
                water_coords_list.append(water_pos)
                
                self.waters.append({
                    "id": w_idx,
                    "parent_atom_id": atom_idx,
                    "residue_index": res_idx,
                    "state": "searching"
                })
                w_idx += 1
            
        if water_coords_list:
            self.water_coords = np.array(water_coords_list)
        else:
            self.water_coords = np.zeros((0, 3))

    def step_explicit_waters(self, dt):
        if not self.params.get("explicitWaterEnabled", True) or len(self.waters) == 0:
            self.active_water_bonds = []
            self.active_water_peptide_bonds = []
            return np.zeros((self.n_atoms, 3))
            
        num_waters = len(self.waters)
        forces_water = np.zeros((num_waters, 3))
        forces_reaction_on_peptide = np.zeros((self.n_atoms, 3))
        
        k_pw = self.params.get("waterPeptideStrength", 0.40) * 15.0 # Scaled for simulation stability
        k_ww = self.params.get("waterWaterStrength", 0.35) * 20.0 # Strongly bond water-water
        
        # 1. Peptide-Water Attraction: spring attraction pulling waters toward parent hydrophilic atoms
        for w_idx, water in enumerate(self.waters):
            parent_idx = water["parent_atom_id"]
            pos_w = self.water_coords[w_idx]
            pos_p = self.coords[parent_idx]
            
            diff = pos_p - pos_w
            d = np.linalg.norm(diff)
            if d > 1e-3:
                f_mag = (d - 2.5) * k_pw
                f_dir = diff / d
                force = f_dir * f_mag
                forces_water[w_idx] += force
                forces_reaction_on_peptide[parent_idx] -= force

        # 2. Water-Water Attraction: spring attraction between water molecules (forming a closed loop)
        c_rate = self.params.get("waterBeltContraction", 0.0)
        target_ww_dist = 2.8 * (1.0 - c_rate)
        
        # Build water-water pairs
        pairs = []
        for w_idx in range(num_waters):
            next_w_idx = (w_idx + 1) % num_waters
            parent_idx1 = self.waters[w_idx]["parent_atom_id"]
            parent_idx2 = self.waters[next_w_idx]["parent_atom_id"]
            if parent_idx1 != parent_idx2:
                pairs.append((w_idx, next_w_idx, parent_idx1, parent_idx2))

        for idx1, idx2, parent_idx1, parent_idx2 in pairs:
            # Clathrate effect: water-water bonds are stronger (1.8x) near hydrophobic atoms
            near_hydrophobic = False
            if self.atoms[parent_idx1]["hydrophobicity"] > 0.15 or self.atoms[parent_idx2]["hydrophobicity"] > 0.15:
                near_hydrophobic = True
            else:
                pos_w1 = self.water_coords[idx1]
                pos_w2 = self.water_coords[idx2]
                for p_idx in range(self.n_atoms):
                    if self.atoms[p_idx]["hydrophobicity"] > 0.15:
                        d1 = np.linalg.norm(pos_w1 - self.coords[p_idx])
                        d2 = np.linalg.norm(pos_w2 - self.coords[p_idx])
                        if d1 < 4.5 or d2 < 4.5:
                            near_hydrophobic = True
                            break
                            
            pair_k_ww = k_ww * 1.8 if near_hydrophobic else k_ww
            
            pos_w1 = self.water_coords[idx1]
            pos_w2 = self.water_coords[idx2]
            
            diff = pos_w2 - pos_w1
            d = np.linalg.norm(diff)
            if d > 1e-3:
                f_mag = (d - target_ww_dist) * pair_k_ww
                f_dir = diff / d
                force = f_dir * f_mag
                forces_water[idx1] += force
                forces_water[idx2] -= force
                
        # 3. Steric Repulsion between water particles
        for i in range(num_waters):
            for j in range(i + 1, num_waters):
                pos_w1 = self.water_coords[i]
                pos_w2 = self.water_coords[j]
                diff = pos_w2 - pos_w1
                d = np.linalg.norm(diff)
                if d > 0.01 and d < 2.2: # Steric radius of water ~1.1Å, diameter 2.2Å
                    rep = (2.2 - d) * 10.0
                    f_dir = diff / d
                    forces_water[i] -= f_dir * rep
                    forces_water[j] += f_dir * rep
                    
        # 4. Steric Repulsion and exclusion between water particles and peptide atoms (except parent)
        for w_idx, water in enumerate(self.waters):
            parent_idx = water["parent_atom_id"]
            pos_w = self.water_coords[w_idx]
            for p_idx in range(self.n_atoms):
                if p_idx == parent_idx:
                    continue
                pos_p = self.coords[p_idx]
                diff = pos_p - pos_w
                d = np.linalg.norm(diff)
                
                is_hydrophobic = self.atoms[p_idx]["hydrophobicity"] > 0.15
                # Strong exclusion of water from hydrophobic atoms (1.6x radius, 2.0x strength)
                scale_radius = 1.6 if is_hydrophobic else 1.0
                scale_strength = 2.0 if is_hydrophobic else 1.0
                
                r_comb = (1.1 + self.atoms[p_idx]["radius"] * 0.23) * 1.1 * scale_radius
                if d > 0.01 and d < r_comb:
                    rep = (r_comb - d) * 12.0 * scale_strength
                    f_dir = diff / d
                    forces_water[w_idx] -= f_dir * rep
                    forces_reaction_on_peptide[p_idx] += f_dir * rep
                    
        # 5. Apply forces to water coordinates (Euler integration with damping)
        damping = 5.0
        temp = self.params.get("temperature", 0.08)
        for w_idx in range(num_waters):
            noise = np.random.normal(0, temp * 2.5, 3) if temp > 0 else np.zeros(3)
            disp = (forces_water[w_idx] / damping) * dt + noise * dt
            disp_norm = np.linalg.norm(disp)
            if disp_norm > 0.15:
                disp = (disp / disp_norm) * 0.15
            self.water_coords[w_idx] += disp

        # Determine water states for UI display
        self.active_water_bonds = []
        for idx1, idx2, parent_idx1, parent_idx2 in pairs:
            pos_w1 = self.water_coords[idx1]
            pos_w2 = self.water_coords[idx2]
            d = np.linalg.norm(pos_w2 - pos_w1)
            if d < 3.2:
                self.active_water_bonds.append((idx1, idx2))

        self.active_water_peptide_bonds = []
        for w_idx, water in enumerate(self.waters):
            parent_idx = water["parent_atom_id"]
            d = np.linalg.norm(self.water_coords[w_idx] - self.coords[parent_idx])
            if d < 2.9:
                self.active_water_peptide_bonds.append((w_idx, parent_idx))

        for w_idx, water in enumerate(self.waters):
            parent_idx = water["parent_atom_id"]
            d_p = np.linalg.norm(self.water_coords[w_idx] - self.coords[parent_idx])
            if d_p < 2.9:
                is_bonded = False
                for u, v in self.active_water_bonds:
                    if u == w_idx or v == w_idx:
                        is_bonded = True
                        break
                water["state"] = "bonded" if is_bonded else "adsorbed"
            else:
                water["state"] = "searching"
            
        return forces_reaction_on_peptide

    def step(self, dt):
        with self.lock:
            self.step_counter += 1
            temp = self.params.get("temperature", 0.08)
            
            # Step the water solvent agent first to compute water forces
            forces_reaction = np.zeros((self.n_atoms, 3))
            if self.params.get("explicitWaterEnabled", True):
                forces_reaction = self.step_explicit_waters(dt)
                
            # If sequential folding is enabled
            R = 0
            if self.seq_folding["enabled"]:
                R = self.seq_folding["activeResidue"]
                direction = self.seq_folding.get("direction", 1)
                
                # Advance step counter
                self.seq_folding["currentStep"] += 1
                
                # Keep water belt completely relaxed during folding
                self.params["waterBeltContraction"] = 0.0
                
                # Check water-bond-triggered transition
                water_triggered = False
                if self.params.get("waterFoldingTransition", False) and self.params.get("explicitWaterEnabled", True) and len(self.waters) > 0:
                    target_res = R + 1 if direction == 1 else R - 1
                    if 0 <= target_res < self.n_residues:
                        waters_R = [w_idx for w_idx, w in enumerate(self.waters) if w["residue_index"] == R]
                        waters_target = [w_idx for w_idx, w in enumerate(self.waters) if w["residue_index"] == target_res]
                        
                        for w1 in waters_R:
                            for w2 in waters_target:
                                dist = np.linalg.norm(self.water_coords[w1] - self.water_coords[w2])
                                if dist < 3.2:
                                    water_triggered = True
                                    break
                            if water_triggered:
                                break
                
                if self.seq_folding["currentStep"] >= self.seq_folding["stepsPerResidue"] or (water_triggered and self.seq_folding["currentStep"] >= 80):
                    if water_triggered:
                        if direction == 1:
                            self.logs.append(f"System: Water-mediated bond formed between Residue {R+1} and {R+2}!")
                        else:
                            self.logs.append(f"System: Water-mediated bond formed between Residue {R+1} and {R}!")
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
                            self.params["waterBeltContraction"] = self.seq_folding.get("maxContraction", 0.50)
                            self.logs.append("System: Sequential folding completed! All residues are now folded and joints are freed.")
            
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
                    
                    # Hydrophobic attraction between hydrophobic atoms (logP > 0.15)
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
                    # Freeze N-terminal residue when sequential folding is disabled
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
                    
                    # Reaction force torque
                    t_react = np.cross(r, forces_reaction[idx])
                    torque_reaction += np.dot(t_react, axis)
                    
                    # Other forces torque (noise, steric)
                    t_other = np.cross(r, forces_other[idx])
                    torque_other += np.dot(t_other, axis)
                    
                    # Moment of inertia
                    cross_r_axis = np.cross(r, axis)
                    inertia += np.dot(cross_r_axis, cross_r_axis)
                    
                torque = torque_reaction + torque_other
                
                if abs(torque) < 1e-5:
                    continue
                
                damping = 1.5
                effective_inertia = max(1.0, inertia)
                
                d_theta = (dt * torque) / (damping * effective_inertia)
                
                max_rot = 0.08
                d_theta = np.clip(d_theta, -max_rot, max_rot)
                
                if abs(d_theta) > 0.005 and self.step_counter % 150 == 0:
                    res_label = self.residue_labels[d_res] if d_res < len(self.residue_labels) else f"Res{d_res+1}"
                    self.logs.append(f"PeptideModel: Rotated joint ({u_idx} -> {d_idx}) of residue {res_label} by {d_theta:.3f} rad.")
                
                cos_t = np.cos(d_theta)
                sin_t = np.sin(d_theta)
                
                # Apply Rodrigues' rotation
                for idx in D:
                    v = self.coords[idx] - pos_B
                    v_rot = v * cos_t + np.cross(axis, v) * sin_t + axis * np.dot(axis, v) * (1.0 - cos_t)
                    self.coords[idx] = pos_B + v_rot
                    
                    # Co-rotate the corresponding water particles
                    if self.params.get("explicitWaterEnabled", True) and len(self.waters) > 0:
                        for w_idx, water in enumerate(self.waters):
                            if water["parent_atom_id"] == idx:
                                v_w = self.water_coords[w_idx] - pos_B
                                v_w_rot = v_w * cos_t + np.cross(axis, v_w) * sin_t + axis * np.dot(axis, v_w) * (1.0 - cos_t)
                                self.water_coords[w_idx] = pos_B + v_w_rot

            # Center coordinates around the first residue to make it a stationary anchor
            res0_indices = [i for i in range(self.n_atoms) if self.atom_residues.get(i, 0) == 0]
            anchor_pos = np.mean(self.coords[res0_indices], axis=0) if res0_indices else self.coords[0]
            self.coords -= anchor_pos
            if len(self.waters) > 0 and self.water_coords.shape[0] > 0:
                self.water_coords -= anchor_pos

            # Calculate hydrophobic exposure
            hydrophobic_indices = [i for i, a in enumerate(self.atoms) if a["hydrophobicity"] > 0.15]
            if not hydrophobic_indices or len(self.waters) == 0:
                self.hydrophobic_exposure = 0.0
            else:
                exposures = []
                for idx in hydrophobic_indices:
                    w_dists = np.linalg.norm(self.water_coords - self.coords[idx], axis=1)
                    d_min = np.min(w_dists)
                    exp_val = np.clip((5.5 - d_min) / (5.5 - 3.0), 0.0, 1.0)
                    exposures.append(exp_val)
                self.hydrophobic_exposure = float(np.mean(exposures))

            # Physics-based logging (once every 150 steps)
            if self.step_counter % 150 == 0:
                # 1. Pulled hydrophilic
                if self.active_water_peptide_bonds:
                    w_idx, parent_idx = self.active_water_peptide_bonds[0]
                    res_idx = self.atom_residues.get(parent_idx, 0)
                    res_label = self.residue_labels[res_idx] if res_idx < len(self.residue_labels) else f"Res{res_idx+1}"
                    self.logs.append(f"WaterSolvent: Pulled hydrophilic atom {parent_idx} ({self.atoms[parent_idx]['element']}) of {res_label} closer to water.")
                
                # 2. Formed H-bond
                if self.active_water_bonds:
                    w1, w2 = self.active_water_bonds[0]
                    self.logs.append(f"WaterSolvent: Formed H-bond between water molecules {w1} and {w2}.")
                
                # 3. Pushed hydrophobic
                if hydrophobic_indices and self.params.get("explicitWaterEnabled", True):
                    max_rep_f = 0.0
                    max_rep_idx = -1
                    for idx in hydrophobic_indices:
                        f_mag = np.linalg.norm(forces_reaction[idx])
                        if f_mag > max_rep_f:
                            max_rep_f = f_mag
                            max_rep_idx = idx
                    if max_rep_f > 0.5:
                        res_idx = self.atom_residues.get(max_rep_idx, 0)
                        res_label = self.residue_labels[res_idx] if res_idx < len(self.residue_labels) else f"Res{res_idx+1}"
                        self.logs.append(f"WaterSolvent: Pushed hydrophobic atom {max_rep_idx} ({self.atoms[max_rep_idx]['element']}) of {res_label} away (force: {max_rep_f:.2f}).")
                
                # 4. Hydrophobic collapse
                if hydrophobic_indices:
                    collapsed_pairs = []
                    for i in hydrophobic_indices:
                        for j in hydrophobic_indices:
                            if i >= j:
                                continue
                            if self.atom_residues[i] == self.atom_residues[j]:
                                continue
                            d = dists[i, j]
                            if d < 4.5:
                                collapsed_pairs.append((i, j, d))
                    if collapsed_pairs:
                        collapsed_pairs.sort(key=lambda x: x[2])
                        i, j, d = collapsed_pairs[0]
                        res_i = self.atom_residues.get(i, 0)
                        res_j = self.atom_residues.get(j, 0)
                        label_i = self.residue_labels[res_i] if res_i < len(self.residue_labels) else f"Res{res_i+1}"
                        label_j = self.residue_labels[res_j] if res_j < len(self.residue_labels) else f"Res{res_j+1}"
                        self.logs.append(f"WaterSolvent: Hydrophobic collapse! Hydrophobic atoms {i} ({label_i}) and {j} ({label_j}) attracted together (dist: {d:.2f} Å).")
                
                # 5. Steric clash
                clashes = []
                for i in range(self.n_atoms):
                    for j in range(i + 1, self.n_atoms):
                        if self.graph_dist[i, j] <= 2:
                            continue
                        d = dists[i, j]
                        min_dist = (self.atoms[i]["radius"] + self.atoms[j]["radius"]) * 0.45
                        if d < min_dist:
                            clashes.append((i, j, d, min_dist))
                if clashes:
                    clashes.sort(key=lambda x: x[2])
                    i, j, d, min_dist = clashes[0]
                    self.logs.append(f"PeptideModel: Steric clash between atom {i} ({self.atoms[i]['element']}) and {j} ({self.atoms[j]['element']}) (dist: {d:.2f} Å < {min_dist:.2f} Å).")

            # Store logs
            if len(self.logs) > 60:
                self.logs = self.logs[-60:]

    def get_new_logs(self):
        with self.lock:
            logs = list(self.logs)
            self.logs = []
            return logs


# Global state store
simulations = {}

@app.route('/api/initialize', methods=['POST'])
def initialize_peptide():
    data = request.json or {}
    sequence = data.get("sequence", "Ala-Ala-Ala")
    try:
        sim = PeptideSimulation(sequence)
        # Apply any parameters passed in payload
        for k, v in data.items():
            if k in sim.params:
                sim.params[k] = v
                
        session_id = str(uuid.uuid4())
        simulations[session_id] = sim
        
        # Serialize initial topology
        return jsonify({
            "sessionId": session_id,
            "atoms": sim.atoms,
            "bonds": sim.bonds,
            "initialCoords": sim.coords.tolist(),
            "atomResidues": sim.atom_residues,
            "residueLabels": sim.residue_labels,
            "waters": sim.waters,
            "initialWaterCoords": sim.water_coords.tolist() if len(sim.waters) > 0 else []
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

@app.route('/api/control', methods=['POST'])
def control_simulation():
    data = request.json or {}
    session_id = data.get("sessionId")
    if not session_id or session_id not in simulations:
        return jsonify({"error": "Invalid session ID"}), 400
        
    sim = simulations[session_id]
    with sim.lock:
        if "temperature" in data:
            sim.params["temperature"] = float(data["temperature"])
        if "isPlaying" in data:
            sim.params["isPlaying"] = bool(data["isPlaying"])
        if "explicitWaterEnabled" in data:
            sim.params["explicitWaterEnabled"] = bool(data["explicitWaterEnabled"])
        if "waterWaterStrength" in data:
            sim.params["waterWaterStrength"] = float(data["waterWaterStrength"])
        if "waterPeptideStrength" in data:
            sim.params["waterPeptideStrength"] = float(data["waterPeptideStrength"])
        if "bulkSolventStrength" in data:
            sim.params["bulkSolventStrength"] = float(data["bulkSolventStrength"])
        if "waterBeltContraction" in data:
            val = float(data["waterBeltContraction"])
            if sim.seq_folding["enabled"]:
                sim.seq_folding["maxContraction"] = val
                sim.params["waterBeltContraction"] = 0.0
            else:
                sim.params["waterBeltContraction"] = val
        if "watersPerAtom" in data:
            old_val = sim.params.get("watersPerAtom", 2)
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
                # Run steps if playing
                if sim.params["isPlaying"]:
                    # Run 5 micro-steps per frame to ensure stability and smooth physics
                    for _ in range(5):
                        sim.step(dt)
                        
                # Prepare payload under lock to ensure thread-safety
                with sim.lock:
                    payload = {
                        "coords": sim.coords.tolist(),
                        "logs": list(sim.logs),
                        "activeHBonds": list(sim.active_hbonds),
                        "hydrophobicExposure": sim.hydrophobic_exposure,
                        "explicitWaterEnabled": sim.params["explicitWaterEnabled"],
                        "waterBeltContraction": sim.params["waterBeltContraction"],
                        "waterCoords": sim.water_coords.tolist() if len(sim.waters) > 0 else [],
                        "waters": sim.waters,
                        "activeWaterBonds": list(sim.active_water_bonds) if len(sim.waters) > 0 else [],
                        "activeWaterPeptideBonds": list(sim.active_water_peptide_bonds) if len(sim.waters) > 0 else [],
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
                time.sleep(0.033) # Stream at ~30 FPS
        except GeneratorExit:
            # Client disconnected
            pass
            
    return Response(event_generator(), mimetype="text/event-stream")

if __name__ == '__main__':
    # Start on port 5000
    app.run(host='0.0.0.0', port=5000, debug=False)
