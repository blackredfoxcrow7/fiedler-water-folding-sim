from rdkit import Chem
from rdkit.Chem import AllChem

n_monomers = 10
smiles = "CC(C)NC(=O)C" + "C(C(=O)NC(C)C)C" * (n_monomers - 1) + "C"
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
params = AllChem.ETKDGv3()
params.useRandomCoords = True
AllChem.EmbedMolecule(mol, params)

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

n_atoms = mol.GetNumAtoms()
adj = {i: [] for i in range(n_atoms)}
for bond in mol.GetBonds():
    s = bond.GetBeginAtomIdx()
    t = bond.GetEndAtomIdx()
    adj[s].append(t)
    adj[t].append(s)

joints = []
for bond in mol.GetBonds():
    if bond.GetBondType() != Chem.BondType.SINGLE:
        continue
    if is_amide_bond(mol, bond):
        continue
    s = bond.GetBeginAtomIdx()
    t = bond.GetEndAtomIdx()
    if len(adj[s]) <= 1 or len(adj[t]) <= 1:
        continue
        
    visited = {s: True, t: True}
    downstream = []
    queue = [t]
    while queue:
        curr = queue.pop(0)
        downstream.append(curr)
        for nbr in adj[curr]:
            if nbr not in visited:
                visited[nbr] = True
                queue.append(nbr)
                
    if 0 in downstream:
        s, t = t, s
        visited = {s: True, t: True}
        downstream = []
        queue = [t]
        while queue:
            curr = queue.pop(0)
            downstream.append(curr)
            for nbr in adj[curr]:
                if nbr not in visited:
                    visited[nbr] = True
                    queue.append(nbr)
                    
    if downstream and len(downstream) < n_atoms - 1:
        joints.append((s, t, len(downstream)))

print(f"Total Atoms (including H): {n_atoms}")
print(f"Total Rotatable Joints Found: {len(joints)}")
for j in joints[:10]:
    print(f"Joint: {j[0]} - {j[1]} | Downstream Atoms: {j[2]}")
