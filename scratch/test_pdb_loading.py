import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server_inference import PeptideInferenceSimulation
from rdkit import Chem

def main():
    print("Testing 1UAO.pdb loading...")
    sim = PeptideInferenceSimulation("GYDPETGTWG")
    
    pdb_path = "1UAO.pdb"
    pdb_mol = Chem.MolFromPDBFile(pdb_path, removeHs=True)
    if not pdb_mol:
        print("❌ Error: Chem.MolFromPDBFile returned None!")
        return
        
    print(f"✓ PDB parsed successfully. Atom count: {pdb_mol.GetNumAtoms()}")
    print(f"PeptideAgent heavy atom count: {sim.n_atoms}")
    
    success = sim.load_pdb_conformation(pdb_path)
    print(f"load_pdb_conformation result: {success}")
    
if __name__ == "__main__":
    main()
