import time
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem

# Minimal replicate of server's PeptideAgent
class MinimalPeptide:
    def __init__(self, sequence):
        self.sequence = sequence.strip().upper()
        self.mol = Chem.MolFromSequence(self.sequence)
        self.mol = Chem.AddHs(self.mol)
        AllChem.EmbedMolecule(self.mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(self.mol)
        self.mol = Chem.RemoveHs(self.mol)
        self.n_atoms = self.mol.GetNumAtoms()
        
        self.atoms = []
        for atom in self.mol.GetAtoms():
            info = atom.GetPDBResidueInfo()
            symbol = info.GetName().strip() if info else atom.GetSymbol()
            self.atoms.append({"id": atom.GetIdx(), "symbol": symbol, "element": atom.GetSymbol()})
            
        self.bonds = [{"source": b.GetBeginAtomIdx(), "target": b.GetEndAtomIdx()} for b in self.mol.GetBonds()]
        
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
        
        # nx graph
        G = nx.Graph()
        G.add_nodes_from(range(self.n_atoms))
        for b in self.bonds:
            G.add_edge(int(b["source"]), int(b["target"]))
        self.graph_dist = nx.floyd_warshall_numpy(G)
        
        # dihedrals
        self.joints = []
        for bond in self.mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            info1 = a1.GetPDBResidueInfo()
            info2 = a2.GetPDBResidueInfo()
            if info1 and info2:
                name1 = info1.GetName().strip()
                name2 = info2.GetName().strip()
                is_phi = (name1 == "N" and name2 == "CA") or (name1 == "CA" and name2 == "N")
                is_psi = (name1 == "CA" and name2 == "C") or (name1 == "C" and name2 == "CA")
                if is_phi or is_psi:
                    u, v = a1.GetIdx(), a2.GetIdx()
                    G_temp = G.copy()
                    G_temp.remove_edge(u, v)
                    components = list(nx.connected_components(G_temp))
                    if len(components) == 2:
                        comp1, comp2 = components
                        downstream = comp2 if v in comp2 else comp1
                        self.joints.append({
                            "u_idx": u, "d_idx": v, "downstream_atoms": list(downstream)
                        })

print("Initializing YYDPETGTWY...")
pep = MinimalPeptide("YYDPETGTWY")
print("Atoms:", pep.n_atoms)
print("Joints:", len(pep.joints))

# Let's count how many times we would run eigenvalues in a single cycle
# inward sweep: for each res, for each joint in res, for each candidate angle:
#   we build a graph and get eigenvalue.
res_arr = np.array(pep.atom_residues)
joints_by_res = {r: [] for r in range(pep.n_residues)}
for joint in pep.joints:
    r = pep.atom_residues[joint["d_idx"]]
    joints_by_res[r].append(joint)

candidate_angles = [-0.5, -0.25, -0.1, -0.05, 0.05, 0.1, 0.25, 0.5]

t0 = time.time()
n_evals = 0
for res_idx in reversed(range(pep.n_residues)):
    for joint in joints_by_res[res_idx]:
        for d_theta in candidate_angles:
            # simple mock of graph and eigenvalue calculation
            G_pep_cand = nx.Graph()
            G_pep_cand.add_nodes_from(range(pep.n_atoms))
            for covalent_bond in pep.bonds:
                G_pep_cand.add_edge(int(covalent_bond["source"]), int(covalent_bond["target"]), weight=10.0)
            
            # compute Fiedler
            L = nx.laplacian_matrix(G_pep_cand, weight='weight').toarray()
            eigenvals = np.linalg.eigvalsh(L)
            fiedler = float(eigenvals[1])
            n_evals += 1

t1 = time.time()
print(f"Executed {n_evals} simple eigenvalue calculations in {t1 - t0:.3f} seconds.")
