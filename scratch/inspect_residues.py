import sys
sys.path.append(".")
from server import PeptideSimulation
from download_pdb import parse_heavy_atoms_pdb

sim = PeptideSimulation("Gly-Tyr-Asp-Pro-Glu-Thr-Gly-Thr-Trp-Gly")
pdb_coords, pdb_info = parse_heavy_atoms_pdb("1UAO.pdb")

print("--- SIMULATION ATOMS ---")
print(f"Total atoms: {sim.n_atoms}")
for res_idx in range(sim.n_residues):
    atoms = [i for i in range(sim.n_atoms) if sim.atom_residues.get(i) == res_idx]
    elements = [sim.atoms[i]["element"] for i in atoms]
    print(f"Residue {res_idx} ({sim.residue_labels[res_idx]}): {len(atoms)} atoms, elements: {elements}")

print("\n--- PDB ATOMS ---")
print(f"Total PDB heavy atoms: {len(pdb_info)}")
pdb_res_indices = sorted(list(set(info["res_num"] for info in pdb_info)))
for res_num in pdb_res_indices:
    atoms = [i for i, info in enumerate(pdb_info) if info["res_num"] == res_num]
    elements = [pdb_info[i]["element"] for i in atoms]
    res_name = pdb_info[atoms[0]]["res_name"]
    print(f"PDB Residue {res_num} ({res_name}): {len(atoms)} atoms, elements: {elements}")
