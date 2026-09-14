import sys
sys.path.append(".")
from server import PeptideSimulation
from download_pdb import parse_heavy_atoms_pdb
import numpy as np

def align_rdkit_to_pdb(sim, pdb_coords, pdb_info):
    mapped_coords = np.zeros((sim.n_atoms, 3))
    mapped_count = 0
    
    for res_idx in range(sim.n_residues):
        sim_atom_indices = [i for i in range(sim.n_atoms) if sim.atom_residues.get(i, -1) == res_idx]
        pdb_atom_indices = [i for i, info in enumerate(pdb_info) if info["res_num"] == res_idx + 1]
        
        # 1. Identify PDB atoms
        pdb_atoms = {}
        for idx in pdb_atom_indices:
            name = pdb_info[idx]["atom_name"]
            pdb_atoms[name] = idx
            
        # 2. Identify Simulation atoms based on rules
        sim_atoms = {}
        
        # Elements in this residue
        res_atoms = {i: sim.atoms[i] for i in sim_atom_indices}
        
        # Backbone N:
        # It's a Nitrogen. For normal residues, it's the only N, or the one connected to CA.
        # Let's find all Nitrogens in this residue
        res_nitrogens = [i for i in sim_atom_indices if sim.atoms[i]["element"] == 'N']
        # Backbone O:
        # It's an Oxygen connected to a Carbon with degree 3.
        # Let's find all Oxygens
        res_oxygens = [i for i in sim_atom_indices if sim.atoms[i]["element"] == 'O']
        
        # Let's find CA: it's connected to a Nitrogen and a Carbon.
        # Let's find Backbone C: it is connected to Backbone O.
        
        # We can find Backbone O:
        backbone_o_candidates = [i for i in res_oxygens if len(sim.adj[i]) == 1 and sim.atoms[sim.adj[i][0]]["element"] == 'C']
        if len(backbone_o_candidates) >= 1:
            # The one connected to a Carbon of degree 3 (backbone C)
            for o_cand in backbone_o_candidates:
                c_cand = sim.adj[o_cand][0]
                if len(sim.adj[c_cand]) >= 2:
                    sim_atoms["O"] = o_cand
                    sim_atoms["C"] = c_cand
                    break
        
        # If C-terminal, we might have OXT (second terminal oxygen)
        if "O" in sim_atoms:
            c_backbone = sim_atoms["C"]
            oxt_cands = [i for i in res_oxygens if i != sim_atoms["O"] and c_backbone in sim.adj[i]]
            if oxt_cands:
                sim_atoms["OXT"] = oxt_cands[0]
                
        # Backbone CA is connected to Backbone C
        if "C" in sim_atoms:
            c_backbone = sim_atoms["C"]
            ca_cands = [i for i in sim.adj[c_backbone] if i in sim_atom_indices and i != sim_atoms.get("O") and i != sim_atoms.get("OXT")]
            if ca_cands:
                sim_atoms["CA"] = ca_cands[0]
                
        # Backbone N is connected to CA
        if "CA" in sim_atoms:
            ca_atom = sim_atoms["CA"]
            n_cands = [i for i in sim.adj[ca_atom] if i in sim_atom_indices and sim.atoms[i]["element"] == 'N']
            if n_cands:
                sim_atoms["N"] = n_cands[0]
                
        # If N was not found, fallback
        if "N" not in sim_atoms and res_nitrogens:
            sim_atoms["N"] = res_nitrogens[0]
            
        # Sidechain atoms:
        # Let's match remaining atoms by mapping their PDB names
        matched_sim_indices = set(sim_atoms.values())
        matched_pdb_names = set(sim_atoms.keys())
        
        # Map by distance to CA
        if "CA" in sim_atoms:
            ca_idx = sim_atoms["CA"]
            unmatched_sim = [i for i in sim_atom_indices if i not in matched_sim_indices]
            unmatched_pdb = [name for name in pdb_atoms if name not in matched_pdb_names]
            
            # Group by element
            for elem in ['C', 'O', 'N', 'S']:
                elem_sim = [i for i in unmatched_sim if sim.atoms[i]["element"] == elem]
                elem_pdb = [name for name in unmatched_pdb if pdb_info[pdb_atoms[name]]["element"] == elem]
                
                if not elem_sim:
                    continue
                    
                # Sort sim by graph distance to CA
                sim_dists = [sim.graph_dist[ca_idx, i] for i in elem_sim]
                sorted_sim = [x for _, x in sorted(zip(sim_dists, elem_sim))]
                
                # Sort PDB by name depth (B=1, G=2, D=3, E=4, Z=5, H=6)
                def name_depth(name):
                    if len(name) < 2: return 9
                    code = name[1]
                    depths = {'B': 1, 'G': 2, 'D': 3, 'E': 4, 'Z': 5, 'H': 6}
                    # Handle numbers like CE1, CD2
                    val = depths.get(code, 7)
                    if len(name) > 2 and name[2].isdigit():
                        val += int(name[2]) * 0.01
                    return val
                    
                sorted_pdb = sorted(elem_pdb, key=name_depth)
                
                for s_i, p_name in zip(sorted_sim, sorted_pdb):
                    sim_atoms[p_name] = s_i
                    
        # Apply the mapping to mapped_coords
        for name, s_idx in sim_atoms.items():
            if name in pdb_atoms:
                p_idx = pdb_atoms[name]
                mapped_coords[s_idx] = pdb_coords[p_idx]
                mapped_count += 1
            else:
                print(f"Warning: atom {name} in sim but not in PDB!")
                
    print(f"Mapped {mapped_count} out of {sim.n_atoms} atoms.")
    return mapped_coords

if __name__ == "__main__":
    sim = PeptideSimulation("Gly-Tyr-Asp-Pro-Glu-Thr-Gly-Thr-Trp-Gly")
    pdb_coords, pdb_info = parse_heavy_atoms_pdb("1UAO.pdb")
    mapped = align_rdkit_to_pdb(sim, pdb_coords, pdb_info)
