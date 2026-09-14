import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

n_monomers = 10
smiles = "CC(C)NC(=O)C" + "C(C(=O)NC(C)C)C" * (n_monomers - 1) + "C"
print("SMILES:", smiles)

mol = Chem.MolFromSmiles(smiles)
if mol is None:
    print("FAILED to parse SMILES!")
else:
    print("Successfully parsed SMILES!")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.useRandomCoords = True
    embed_status = AllChem.EmbedMolecule(mol, params)
    print("Embed Status:", embed_status)
    if embed_status != -1:
        AllChem.MMFFOptimizeMolecule(mol)
        print("Successfully optimized 3D conformer!")
