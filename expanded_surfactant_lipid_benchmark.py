import numpy as np
import pandas as pd
import json
from rdkit import Chem
from rdkit.Chem import AllChem

class SurfactantLipidAnalyzer:
    """
    Comprehensive Fiedler Spectral Analyzer for Surfactants, Zwitterions, 
    and Biological Membrane Lipids (Phospholipids, Cholesterol, Ceramides).
    """
    def __init__(self):
        self.molecules = [
            # --- 1. Industrial & Synthetic Surfactants ---
            {"category": "Anionic Surfactant", "name": "SDS (Sodium Dodecyl Sulfate)", "smiles": "CCCCCCCCCCCCOS(=O)(=O)[O-]"},
            {"category": "Anionic Surfactant", "name": "Sodium Laurate (Soap)", "smiles": "CCCCCCCCCCCC(=O)[O-]"},
            {"category": "Cationic Surfactant", "name": "CTAB (Cetyltrimethylammonium)", "smiles": "CCCCCCCCCCCCCCCC[N+](C)(C)C"},
            {"category": "Non-ionic Surfactant", "name": "Triton X-100 Head-Chain", "smiles": "CC(C)(C)CC(C)(C)c1ccc(OCCOCCOCCO)cc1"},
            {"category": "Zwitterionic Surfactant", "name": "Lauryl Betaine", "smiles": "CCCCCCCCCCCC[N+](C)(C)CC(=O)[O-]"},
            
            # --- 2. Biological Membrane Lipids & Bio-amphiphiles ---
            {"category": "Membrane Phospholipid", "name": "DPPC (Dipalmitoylphosphatidylcholine)", "smiles": "CCCCCCCCCCCCCCCC(=O)OCC(COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCC"},
            {"category": "Membrane Phospholipid", "name": "POPE (Phosphatidylethanolamine)", "smiles": "CCCCCCCCCCCCCCCC(=O)OCC(COP(=O)([O-])OCCN)OC(=O)CCCCCCCC=======C".replace("=", "")},
            {"category": "Steroid Membrane Lipid", "name": "Cholesterol", "smiles": "CC(C)CCCC(C)C1CCC2C1(CCC3C2C(CC4C3(CCC(C4)O)C)C)C"},
            {"category": "Barrier Sphingolipid", "name": "Ceramide (C16:0)", "smiles": "CCCCCCCCCCCCCC/C=C/[C@@H](O)[C@@H](CO)NC(=O)CCCCCCCCCCCCCC"},
            {"category": "Bile Acid Surfactant", "name": "Sodium Cholates (Cholic Acid)", "smiles": "CC(CCC(=O)[O-])C1CCC2C1(C(CC3C2C(CC4C3(CCC(C4)O)C)O)O)C"},
        ]

    def analyze_molecule(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        
        adj = Chem.GetAdjacencyMatrix(mol, useBO=True)
        deg = np.diag(np.sum(adj, axis=1))
        laplacian = deg - adj
        
        evals, evecs = np.linalg.eigh(laplacian)
        idx = np.argsort(evals)
        evals = evals[idx]
        evecs = evecs[:, idx]
        
        lambda2 = float(evals[1])
        v2 = evecs[:, 1]
        
        # Orient v2 so that polar heteroatoms (O, N, S, P) have positive sign (+)
        polar_atoms = [i for i, a in enumerate(mol.GetAtoms()) if a.GetSymbol() in ["O", "N", "S", "P"]]
        if len(polar_atoms) > 0 and np.mean(v2[polar_atoms]) < 0:
            v2 = -v2
            
        head_count = int(np.sum(v2 >= 0))
        tail_count = int(np.sum(v2 < 0))
        total_atoms = mol.GetNumAtoms()
        
        # Fiedler Hydrophobicity Index (FHI)
        c_count = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
        h_count = sum(1 for a in mol.GetAtoms() if a.GetSymbol() in ["O", "N", "S", "P"])
        fhi = (c_count / max(h_count, 1)) * (1.0 / max(lambda2, 0.001)) * 0.15
        
        return {
            "lambda2": lambda2,
            "head_atoms": head_count,
            "tail_atoms": tail_count,
            "total_atoms": total_atoms,
            "head_pct": round(head_count / total_atoms * 100, 1),
            "fhi": round(fhi, 3)
        }

    def run_benchmark(self):
        print("==================================================")
        print("  EXPANDED SURFACTANT & BIOMEMBRANE LIPID BENCHMARK")
        print("  Fiedler Spectral Analysis Across Surfactants & Lipids")
        print("==================================================")

        records = []
        for item in self.molecules:
            res = self.analyze_molecule(item["smiles"])
            if res is not None:
                records.append({
                    "Category": item["category"],
                    "Name": item["name"],
                    "Total_Atoms": res["total_atoms"],
                    "Head_Atoms_(+)": res["head_atoms"],
                    "Tail_Atoms_(-)" : res["tail_atoms"],
                    "Head_Ratio_(%)": res["head_pct"],
                    "Monomer_lambda2": round(res["lambda2"], 4),
                    "FHI_Hydrophobicity": res["fhi"],
                })

        df = pd.DataFrame(records)

        print("\n[Table 1: Fiedler Vector Atom Partition & Hydrophobicity Index (FHI)]")
        print(df.to_string(index=False))

        # Save to JSON
        with open("surfactant_lipid_benchmark_results.json", "w") as f:
            json.dump(records, f, indent=2)

        print("\n==================================================")
        print("  BENCHMARK COMPLETE: Biological lipids & surfactants")
        print("  successfully quantified via Fiedler Graph Spectrum!")
        print("==================================================")

if __name__ == "__main__":
    analyzer = SurfactantLipidAnalyzer()
    analyzer.run_benchmark()
