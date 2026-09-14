import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

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
                if b and b.GetBondType() == Chem.BondType.DOUBLE:
                    return True
    return False

class PeptideAgent:
    def __init__(self, sequence):
        self.sequence = sequence.strip()
        self.logs = []
        
        # 1. Parse peptide sequence to RDKit molecule
        self.mol = self.parse_molecule(self.sequence)
        
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
        
        # Save initial extended conformation as reference coordinates
        self.initial_coords = self.coords.copy()
        
        # Initialize joint angles (integrated rotations)
        self.joint_angles = np.zeros(len(self.joints))
        
        # Learned databases for local motifs
        self.learned_angles = {}         # joint_idx -> target_angle
        self.learned_by_type = {}        # residue_name -> target_angle
        self.learned_by_motif = {}       # (prev_name, curr_name, next_name) -> target_angle

    def parse_molecule(self, input_str):
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
        
        # Calculate distance matrix on the chemical graph for Dijkstra usage later
        self.graph_dist = np.full((self.n_atoms, self.n_atoms), 999.0)
        for i in range(self.n_atoms):
            self.graph_dist[i, i] = 0.0
            
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
            self.graph_dist[s, t] = 1.0
            self.graph_dist[t, s] = 1.0
            
        # Floyd-Warshall for graph distances
        for k in range(self.n_atoms):
            for i in range(self.n_atoms):
                for j in range(self.n_atoms):
                    if self.graph_dist[i, k] + self.graph_dist[k, j] < self.graph_dist[i, j]:
                        self.graph_dist[i, j] = self.graph_dist[i, k] + self.graph_dist[k, j]
                        
        # Identify rotatable peptide joints
        self.joints = []
        for b in self.mol_heavy.GetBonds():
            if is_amide_bond(self.mol_heavy, b):
                continue
            s = b.GetBeginAtomIdx()
            t = b.GetEndAtomIdx()
            if s >= self.n_atoms or t >= self.n_atoms:
                continue
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
                        
            # Orient joint so downstream_atoms always represents the C-terminal side.
            # If the N-terminal anchor atom (index 0) is in the downstream split,
            # we swap s and t and re-split to get the C-terminal side.
            if 0 in downstream:
                s, t = t, s
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

    def assign_residues(self):
        # Find all unique residue numbers in order of appearance
        res_nums = []
        for i in range(self.n_atoms):
            num = self.atoms[i]["res_num"]
            if num not in res_nums:
                res_nums.append(num)
                
        res_map = {num: idx for idx, num in enumerate(res_nums)}
        
        self.atom_residues = {}
        for i in range(self.n_atoms):
            self.atom_residues[i] = res_map[self.atoms[i]["res_num"]]
            
        self.n_residues = len(res_nums)
        self.residue_labels = []
        for r in range(self.n_residues):
            res_atoms = [i for i in range(self.n_atoms) if self.atom_residues[i] == r]
            res_name = self.atoms[res_atoms[0]]["res_name"] if res_atoms else "UNK"
            self.residue_labels.append(f"{res_name}-{r+1}")

    def center_around_first_residue(self):
        first_res_atoms = [i for i, r in self.atom_residues.items() if r == 0]
        if first_res_atoms:
            com = np.mean(self.coords[first_res_atoms], axis=0)
            self.coords -= com
        else:
            com = np.mean(self.coords, axis=0)
            self.coords -= com

    def reset_to_initial(self):
        self.coords = self.initial_coords.copy()
        self.joint_angles = np.zeros(len(self.joints))

    def learn_residue_step(self, residue_idx):
        """
        Record the current joint angles of the active residue and associate them
        with the local residue type and dipeptide/tripeptide motifs.
        """
        self.logs.append(f"AI Learning: Capturing folded dihedrals for residue {residue_idx + 1} ({self.residue_labels[residue_idx]})")
        
        res_name = self.residue_labels[residue_idx].split('-')[0]
        prev_name = self.residue_labels[residue_idx - 1].split('-')[0] if residue_idx > 0 else "N-TERM"
        next_name = self.residue_labels[residue_idx + 1].split('-')[0] if residue_idx < self.n_residues - 1 else "C-TERM"
        motif = (prev_name, res_name, next_name)
        
        for j_idx, joint in enumerate(self.joints):
            d_idx = joint["d_idx"]
            d_res = self.atom_residues.get(d_idx, 0)
            if d_res == residue_idx:
                angle = self.joint_angles[j_idx]
                
                # 1. Direct index memory (exact sequence)
                self.learned_angles[j_idx] = angle
                
                # 2. Residue type generalized memory
                self.learned_by_type[res_name] = angle
                
                # 3. Tripeptide motif local memory
                self.learned_by_motif[motif] = angle

    def apply_rotations(self, dt, forces_reaction, forces_other, active_residue=None, reverse=False, use_inference=False):
        """
        Computes joint torques and rotates downstream coordinates.
        Supports:
        - `reverse`: Flips torque direction.
        - `use_inference`: Blends a steering force drawing the joint towards the learned angles.
        """
        for j_idx, joint in enumerate(self.joints):
            u_idx = joint["u_idx"]
            d_idx = joint["d_idx"]
            D = joint["downstream_atoms"]
            d_res = self.atom_residues.get(d_idx, 0)
            
            # If sequential active residue is set, only rotate its joints
            if active_residue is not None:
                if d_res != active_residue:
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
            
            # Calculate physical torque in vectorized format
            r = self.coords[D] - pos_B
            t_react = np.cross(r, forces_reaction[D])
            torque_reaction = np.sum(t_react @ axis)
            
            t_other = np.cross(r, forces_other[D])
            torque_other = np.sum(t_other @ axis)
            
            cross_r_axis = np.cross(r, axis)
            inertia = np.sum(cross_r_axis * cross_r_axis)
                
            torque = torque_reaction + torque_other
            
            # Apply reverse unfolding guidance bias in learning mode
            if reverse:
                # Steering force pulling active residue joints towards 0.0 (unfolded state)
                k_unfold = 25.0
                unfold_torque = -k_unfold * self.joint_angles[j_idx]
                torque += unfold_torque

            # Apply autonomous inference guidance bias in forward folding mode
            if use_inference:
                # Look up target angle using local motif hierarchy
                res_name = self.residue_labels[d_res].split('-')[0]
                prev_name = self.residue_labels[d_res - 1].split('-')[0] if d_res > 0 else "N-TERM"
                next_name = self.residue_labels[d_res + 1].split('-')[0] if d_res < self.n_residues - 1 else "C-TERM"
                motif = (prev_name, res_name, next_name)
                
                target = 0.0
                if j_idx in self.learned_angles:
                    target = self.learned_angles[j_idx]
                elif motif in self.learned_by_motif:
                    target = self.learned_by_motif[motif]
                elif res_name in self.learned_by_type:
                    target = self.learned_by_type[res_name]
                    
                # Harmonic spring torque pulling to target dihedral
                k_bias = 20.0
                bias_torque = -k_bias * (self.joint_angles[j_idx] - target)
                torque += bias_torque
                
            # Rigid-body dynamics integration with high damping to prevent oscillations
            damping_const = 4.0
            effective_inertia = max(0.5, inertia * 0.05)
            d_theta = (dt * torque) / (damping_const * effective_inertia)
            
            # Clamp step rotation to maintain stability
            max_d_theta = 0.05
            d_theta = np.clip(d_theta, -max_d_theta, max_d_theta)
            self.joint_angles[j_idx] += d_theta
            
            # Rodrigues rotation update in vectorized format
            cos_t = np.cos(d_theta)
            sin_t = np.sin(d_theta)
            
            r_vec = self.coords[D] - pos_B
            cross_axis_r = np.cross(axis, r_vec)
            dot_axis_r = r_vec @ axis
            
            rotated = r_vec * cos_t + cross_axis_r * sin_t + axis[np.newaxis, :] * (dot_axis_r * (1.0 - cos_t))[:, np.newaxis]
            self.coords[D] = pos_B + rotated

    def align_joint_angles_to_coords(self, target_coords):
        """
        Calculates the joint angles required to transition from the initial
        extended conformation to the target coordinates, and updates self.joint_angles.
        """
        def get_dihedral(coords, p, u, d, n):
            v1 = coords[u] - coords[p]
            v2 = coords[d] - coords[u]
            v3 = coords[n] - coords[d]
            n1 = np.cross(v1, v2)
            n2 = np.cross(v2, v3)
            n1_len = np.linalg.norm(n1)
            n2_len = np.linalg.norm(n2)
            if n1_len < 1e-6 or n2_len < 1e-6:
                return 0.0
            n1 /= n1_len
            n2 /= n2_len
            m1 = np.cross(n1, v2 / np.linalg.norm(v2))
            return np.arctan2(np.dot(m1, n2), np.dot(n1, n2))

        new_angles = np.zeros(len(self.joints))
        for j_idx, joint in enumerate(self.joints):
            u = joint["u_idx"]
            d = joint["d_idx"]
            u_neighbors = [a.GetIdx() for a in self.mol.GetAtomWithIdx(u).GetNeighbors() if a.GetIdx() != d]
            d_neighbors = [a.GetIdx() for a in self.mol.GetAtomWithIdx(d).GetNeighbors() if a.GetIdx() != u]
            
            if u_neighbors and d_neighbors:
                p = u_neighbors[0]
                n = d_neighbors[0]
                theta_init = get_dihedral(self.initial_coords, p, u, d, n)
                theta_target = get_dihedral(target_coords, p, u, d, n)
                diff = theta_target - theta_init
                diff = (diff + np.pi) % (2 * np.pi) - np.pi
                new_angles[j_idx] = diff
        self.joint_angles = new_angles
