import sys
from rdkit import Chem
from rdkit.Chem import AllChem

sequence = "YYDPETGTWY"
print("Building molecule...")
mol = Chem.MolFromSequence(sequence)
print("Adding H...")
mol = Chem.AddHs(mol)

print("Embedding conformer with ETKDGv3 and useRandomCoords=True...")
params = AllChem.ETKDGv3()
params.randomSeed = 42
params.useRandomCoords = True
params.maxAttempts = 100

status = AllChem.EmbedMolecule(mol, params)
print("Embed status:", status)
if status < 0:
    print("Embedding failed, trying with basic distance geometry...")
    status = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
    print("Backup embed status:", status)

print("Optimizing conformer...")
AllChem.MMFFOptimizeMolecule(mol)
print("Removing H...")
mol = Chem.RemoveHs(mol)
print("Number of atoms:", mol.GetNumAtoms())
print("Success!")
