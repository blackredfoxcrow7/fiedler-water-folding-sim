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
        self.logs = []
        self.active_hbonds = []
        self.active_water_bonds = []
        self.active_water_peptide_bonds = []
        self.lock = threading.Lock()
        self.step_counter = 0

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
        
        # Simulation parameters (Optimized default values tailored for H2O model)
        self.params = {
            "temperature": 0.08,
            "isPlaying": True,
            "explicitWaterEnabled": True,
            "waterWaterStrength": 0.50,      # Dynamic H-bond water-water scaling
            "waterPeptideStrength": 0.60,    # Dynamic H-bond water-peptide scaling
            "waterFoldingTransition": False,
            "bulkSolventStrength": 0.35,     # Pressure effect
            "waterBeltContraction": 0.0,     # Relaxed during folding
            "watersPerAtom": 4               # Target waters per peptide atom (hydrophobic + hydrophilic)
        }
        self.hydrophobic_exposure = 0.0
        
        # Sequential folding parameters
        self.seq_folding = {
            "enabled": False,
            "activeResidue": 0,
            "stepsPerResidue": 500,
            "currentStep": 0,
            "status": "idle",
            "direction": 1
        }
        # 5. Initialize H2O water molecules
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
            atom = self.mol.GetAtomWithIdx(i)
            symbol = atom.GetSymbol()
            
            # Simple classification for charge & hydrogen bonding capability
            h_bond = "none"
            if symbol in ("O", "N"):
                # Check if it has bonded hydrogens (donor) or is acceptor
                # Nitrogen in peptide is donor, Carbonyl oxygen is acceptor
                # For simplified rule: Oxygen is acceptor, Nitrogen is donor
                if symbol == "O":
                    h_bond = "acceptor"
                elif symbol == "N":
                    h_bond = "donor"
            
            # Hydrophobicity logP approximations
            hydrophobicity = 0.0
            if symbol == 'C':
                # Carbon in side chain is hydrophobic
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
            atom_name = info.GetName().strip() if info else ""
            self.atoms.append({
                "id": i,
                "symbol": symbol,
                "name": atom_name,
                "radius": radius,
                "h_bond": h_bond,
                "hydrophobicity": hydrophobicity
            })
            
        # Heavy bonds
        for b in self.mol.GetBonds():
            s = b.GetBeginAtomIdx()
            t = b.GetEndAtomIdx()
            if s >= self.n_atoms or t >= self.n_atoms:
                continue
            self.bonds.append({
                "source": s,
                "target": t,
                "is_rotatable": False
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
        for b in self.mol.GetBonds():
            if is_amide_bond(self.mol, b):
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
        # Group atoms by residues
        self.atom_residues = {}
        self.residue_labels = []
        
        # Find all amino acid residues
        # Since it's condensation sequence: N-C-C(=O)-N-C-C(=O)
        # We can map residue index based on shortest path distance from first Nitrogen atom
        # Nitrogen 0 is typically N-terminus.
        n_term_candidates = [i for i, a in enumerate(self.atoms) if a["symbol"] == 'N' and len(self.adj[i]) <= 3]
        n_term = n_term_candidates[0] if n_term_candidates else 0
        
        distances_from_n = self.graph_dist[n_term]
        
        for i in range(self.n_atoms):
            # Roughly 3 graph bonds per residue backbone
            res_idx = int(distances_from_n[i] / 3)
            self.atom_residues[i] = res_idx
            
        self.n_residues = max(self.atom_residues.values()) + 1 if self.atom_residues else 1
        
        # Extract amino acid labels from sequence input or assign defaults
        self.residue_labels = [f"Res{r+1}" for r in range(self.n_residues)]

    def center_around_first_residue(self):
        res0_indices = [i for i in range(self.n_atoms) if self.atom_residues.get(i, 0) == 0]
        anchor_pos = np.mean(self.coords[res0_indices], axis=0) if res0_indices else self.coords[0]
        self.coords -= anchor_pos
        self.initial_coords = self.coords.copy()

    def initialize_waters(self):
        # Spawns water molecules around ALL peptide heavy atoms (hydrophilic and hydrophobic)
        self.waters = []
        water_coords_list = []
        
        n_waters = int(self.params.get("watersPerAtom", 4))
        
        # Fibonacci spiral on sphere for uniform spawn directions
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
                # Add a bit of jitter to avoid perfect symmetry
                jitter = np.random.normal(0, 0.05, 3)
                final_dir = v + jitter
                final_dir /= np.linalg.norm(final_dir)
                
                # Spawn Oxygen at a distance of 3.2 Å from peptide atom
                pos_o = parent_pos + final_dir * 3.2
                
                # Generate two Hydrogens with H-O-H angle 104.5 degrees and O-H distance 0.96 Å
                # Choose random bisector direction
                b_dir = np.random.normal(0, 1, 3)
                b_dir /= np.linalg.norm(b_dir)
                
                # Perpendicular direction
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
            self.water_coords = np.array(water_coords_list) # shape: (n_waters, 3, 3)
        else:
            self.water_coords = np.zeros((0, 3, 3))

    def step_explicit_waters(self, dt):
        if not self.params.get("explicitWaterEnabled", True) or len(self.waters) == 0:
            self.active_water_bonds = []
            self.active_water_peptide_bonds = []
            return np.zeros((self.n_atoms, 3))
            
        num_waters = len(self.waters)
        forces_water = np.zeros((num_waters, 3, 3)) # Oxygen, Hydrogen 1, Hydrogen 2
        forces_reaction_on_peptide = np.zeros((self.n_atoms, 3))
        
        # Scaling variables
        k_pw = self.params.get("waterPeptideStrength", 0.60) * 15.0 # Parent spring
        k_ww = self.params.get("waterWaterStrength", 0.50) * 25.0    # Dynamic H-bonds
        
        # 1. Internal Geometry Constraints (Stiff harmonic springs for rigid V-shape)
        k_int = 250.0
        for i in range(num_waters):
            pos_o = self.water_coords[i, 0]
            pos_h1 = self.water_coords[i, 1]
            pos_h2 = self.water_coords[i, 2]
            
            # O - H1 (target: 0.96 Å)
            diff_h1 = pos_h1 - pos_o
            d_h1 = np.linalg.norm(diff_h1)
            if d_h1 > 1e-3:
                f = (d_h1 - 0.96) * k_int * (diff_h1 / d_h1)
                forces_water[i, 1] -= f
                forces_water[i, 0] += f
                
            # O - H2 (target: 0.96 Å)
            diff_h2 = pos_h2 - pos_o
            d_h2 = np.linalg.norm(diff_h2)
            if d_h2 > 1e-3:
                f = (d_h2 - 0.96) * k_int * (diff_h2 / d_h2)
                forces_water[i, 2] -= f
                forces_water[i, 0] += f
                
            # H1 - H2 (target: 1.52 Å)
            diff_h12 = pos_h2 - pos_h1
            d_h12 = np.linalg.norm(diff_h12)
            if d_h12 > 1e-3:
                f = (d_h12 - 1.52) * k_int * (diff_h12 / d_h12)
                forces_water[i, 2] -= f
                forces_water[i, 1] += f

        # 2. Peptide-Water Parent springs (only for hydrophilic neighborhoods)
        for i, water in enumerate(self.waters):
            if not water["is_hydrophobic_neighborhood"]:
                parent_idx = water["parent_atom_id"]
                pos_o = self.water_coords[i, 0]
                pos_p = self.coords[parent_idx]
                
                diff = pos_p - pos_o
                d = np.linalg.norm(diff)
                if d > 1e-3:
                    f_mag = (d - 2.8) * k_pw
                    f_dir = diff / d
                    force = f_dir * f_mag
                    forces_water[i, 0] += force
                    forces_reaction_on_peptide[parent_idx] -= force

        # 3. Dynamic Water-Water & Water-Peptide Hydrogen Bonds
        self.active_water_bonds = []
        self.active_water_peptide_bonds = []
        
        c_rate = self.params.get("waterBeltContraction", 0.0)
        target_ww_hbond = 1.9 * (1.0 - c_rate) # Contraction scales the target H-bond length
        
        # Water-Water dynamic Hydrogen Bonds (O_i attracts H_j,k)
        for i in range(num_waters):
            pos_o_i = self.water_coords[i, 0]
            
            for j in range(num_waters):
                if i == j:
                    continue
                    
                for h_idx in (1, 2):
                    pos_h_jk = self.water_coords[j, h_idx]
                    
                    diff = pos_h_jk - pos_o_i
                    d = np.linalg.norm(diff)
                    
                    if d < 3.2:
                        # Directional linearity check (O_j - H_jk ... O_i)
                        pos_o_j = self.water_coords[j, 0]
                        donor_dir = pos_h_jk - pos_o_j
                        donor_len = np.linalg.norm(donor_dir)
                        
                        if donor_len > 1e-3:
                            donor_dir /= donor_len
                            h_to_o_dir = pos_o_i - pos_h_jk
                            h_to_o_len = np.linalg.norm(h_to_o_dir)
                            
                            if h_to_o_len > 1e-3:
                                h_to_o_dir /= h_to_o_len
                                cos_angle = np.dot(donor_dir, h_to_o_dir)
                                
                                if cos_angle > 0.5: # within 60 degrees of linear 180 degrees
                                    # Strengthen bonds near hydrophobic areas (clathrate effect)
                                    near_h = self.waters[i]["is_hydrophobic_neighborhood"] or self.waters[j]["is_hydrophobic_neighborhood"]
                                    pair_k = k_ww * 1.8 if near_h else k_ww
                                    
                                    f_mag = (d - target_ww_hbond) * pair_k * cos_angle
                                    force = (diff / d) * f_mag
                                    
                                    forces_water[i, 0] += force
                                    forces_water[j, h_idx] -= force
                                    
                                    if len(self.active_water_bonds) < 150:
                                        self.active_water_bonds.append((i, j, h_idx))

        # Water-Peptide dynamic Hydrogen Bonds
        k_wp = k_pw * 0.8
        for w_idx in range(num_waters):
            # Water Oxygen (O) attracts peptide donors (labeled "donor")
            pos_o = self.water_coords[w_idx, 0]
            # Water Hydrogens (H) attract peptide acceptors (labeled "acceptor")
            pos_h1 = self.water_coords[w_idx, 1]
            pos_h2 = self.water_coords[w_idx, 2]
            
            for p_idx in range(self.n_atoms):
                h_bond_type = self.atoms[p_idx]["h_bond"]
                if h_bond_type == "none":
                    continue
                    
                pos_p = self.coords[p_idx]
                
                if h_bond_type == "donor":
                    # Peptide donor attracts Water Oxygen
                    diff = pos_p - pos_o
                    d = np.linalg.norm(diff)
                    if d < 3.5:
                        f_mag = (d - 2.8) * k_wp
                        force = (diff / d) * f_mag
                        forces_water[w_idx, 0] += force
                        forces_reaction_on_peptide[p_idx] -= force
                        self.active_water_peptide_bonds.append((w_idx, p_idx, 0)) # Type 0: O to donor
                        
                elif h_bond_type == "acceptor":
                    # Peptide acceptor attracts Water Hydrogens
                    for h_idx, pos_h in ((1, pos_h1), (2, pos_h2)):
                        diff = pos_p - pos_h
                        d = np.linalg.norm(diff)
                        if d < 3.2:
                            f_mag = (d - 1.8) * k_wp
                            force = (diff / d) * f_mag
                            forces_water[w_idx, h_idx] += force
                            forces_reaction_on_peptide[p_idx] -= force
                            self.active_water_peptide_bonds.append((w_idx, p_idx, h_idx)) # Type 1/2: H to acceptor

        # 4. Steric Repulsions
        # Water-Water Oxygen steric repulsion
        for i in range(num_waters):
            pos_o_i = self.water_coords[i, 0]
            for j in range(i + 1, num_waters):
                pos_o_j = self.water_coords[j, 0]
                diff = pos_o_j - pos_o_i
                d = np.linalg.norm(diff)
                if d < 2.8: # Sum of oxygen VdW radii
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
                                # Distance between Oxygen of w1 and Oxygen of w2
                                dist = np.linalg.norm(self.water_coords[w1, 0] - self.water_coords[w2, 0])
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
                                # Rotate all 3 atoms of the water molecule (O, H1, H2)
                                for atom_k in range(3):
                                    v_w = self.water_coords[w_idx, atom_k] - pos_B
                                    v_w_rot = v_w * cos_t + np.cross(axis, v_w) * sin_t + axis * np.dot(axis, v_w) * (1.0 - cos_t)
                                    self.water_coords[w_idx, atom_k] = pos_B + v_w_rot

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
                    # Hydrophobic exposure is defined by local water density around the hydrophobic atom
                    # Real water: count how many water Oxygens are within 5.5 Å
                    dists_to_waters = np.linalg.norm(self.water_coords[:, 0] - self.coords[idx], axis=1)
                    nearby_count = np.sum(dists_to_waters < 5.5)
                    exposures.append(nearby_count)
                # Average number of nearby water molecules around hydrophobic groups
                self.hydrophobic_exposure = float(np.mean(exposures)) if exposures else 0.0

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
            pass
            
    return Response(event_generator(), mimetype="text/event-stream")

if __name__ == '__main__':
    # Start on port 5002 for H2O model
    app.run(host='0.0.0.0', port=5002, debug=False)
