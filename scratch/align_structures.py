import numpy as np
import sys
sys.path.append(".")
from server import PeptideSimulation
from download_pdb import parse_heavy_atoms_pdb

def align_rdkit_to_pdb(sim, pdb_coords, pdb_info):
    """
    Map sim.atoms to pdb_info.
    Returns pdb_mapped_coords of shape (sim.n_atoms, 3) aligned to sim.atoms order.
    """
    mapped_coords = np.zeros((sim.n_atoms, 3))
    mapped_count = 0
    
    # Residues in sim are 0-indexed: 0 to 9
    # Residues in pdb are 1-indexed: 1 to 10
    
    # For each residue, match atoms
    for res_idx in range(sim.n_residues):
        # Atoms in simulation belonging to this residue
        sim_atom_indices = [i for i in range(sim.n_atoms) if sim.atom_residues.get(i, -1) == res_idx]
        
        # Atoms in PDB belonging to this residue (1-indexed)
        pdb_atom_indices = [i for i, info in enumerate(pdb_info) if info["res_num"] == res_idx + 1]
        
        # Group by element
        sim_by_elem = {}
        for idx in sim_atom_indices:
            elem = sim.atoms[idx]["element"]
            sim_by_elem.setdefault(elem, []).append(idx)
            
        pdb_by_elem = {}
        for idx in pdb_atom_indices:
            elem = pdb_info[idx]["element"]
            pdb_by_elem.setdefault(elem, []).append(idx)
            
        # Match atoms
        for elem in sim_by_elem:
            sim_idxs = sim_by_elem[elem]
            pdb_idxs = pdb_by_elem.get(elem, [])
            
            if len(sim_idxs) != len(pdb_idxs):
                print(f"Warning: Element {elem} count mismatch in residue {res_idx+1}! Sim: {len(sim_idxs)}, PDB: {len(pdb_idxs)}")
                
            # If count is 1, it's a direct match (e.g. N, CA, CB in Ala)
            if len(sim_idxs) == 1 and len(pdb_idxs) == 1:
                mapped_coords[sim_idxs[0]] = pdb_coords[pdb_idxs[0]]
                mapped_count += 1
            else:
                # If multiple (like Carbons in Tyrosine), we can match them using a topological alignment
                # Or for simplicity, match by backbone name or distance/connectivity
                # Let's check atom names in PDB
                # Backbone atoms: N, CA, C, O
                # Sidechain: CB, etc.
                # In RDKit:
                # We can trace connections:
                # CA is connected to N, C, CB.
                # C is connected to CA, O.
                # N is connected to CA, and previous C.
                # Let's write a simple rule-based matcher for residue atoms:
                matched_sim = set()
                matched_pdb = set()
                
                # Match backbone N, CA, C, O by tracing degrees and connectivity
                # In simulation:
                # - O has degree 1, connected to a Carbon which is C.
                # - C has degree 3, connected to O and N and CA.
                # - N has degree 2 (or 3 for Proline), connected to CA and previous C.
                # - CA has degree 3 or 4, connected to N, C, CB, and H.
                
                # Let's find sim O and PDB O
                sim_o = [i for i in sim_idxs if sim.atoms[i]["element"] == 'O' and len(sim.adj[i]) == 1]
                pdb_o = [i for i in pdb_idxs if pdb_info[i]["atom_name"] == 'O']
                if sim_o and pdb_o:
                    mapped_coords[sim_o[0]] = pdb_coords[pdb_o[0]]
                    matched_sim.add(sim_o[0])
                    matched_pdb.add(pdb_o[0])
                    mapped_count += 1
                    
                # Match C (backbone C is connected to the matched O)
                if sim_o:
                    sim_c_backbone = sim.adj[sim_o[0]][0]
                    pdb_c_backbone = [i for i in pdb_idxs if pdb_info[i]["atom_name"] == 'C']
                    if sim_c_backbone in sim_idxs and pdb_c_backbone:
                        mapped_coords[sim_c_backbone] = pdb_coords[pdb_c_backbone[0]]
                        matched_sim.add(sim_c_backbone)
                        matched_pdb.add(pdb_c_backbone[0])
                        mapped_count += 1
                        
                # Match N
                pdb_n = [i for i in pdb_idxs if pdb_info[i]["atom_name"] == 'N']
                # N in sim is connected to CA and previous C.
                # Let's find N in sim_idxs: it is a Nitrogen.
                sim_n = [i for i in sim_idxs if sim.atoms[i]["element"] == 'N']
                if sim_n and pdb_n:
                    mapped_coords[sim_n[0]] = pdb_coords[pdb_n[0]]
                    matched_sim.add(sim_n[0])
                    matched_pdb.add(pdb_n[0])
                    mapped_count += 1
                    
                # Match CA
                pdb_ca = [i for i in pdb_idxs if pdb_info[i]["atom_name"] == 'CA']
                # CA is connected to N.
                if sim_n:
                    sim_ca_candidates = [i for i in sim.adj[sim_n[0]] if i in sim_idxs and sim.atoms[i]["element"] == 'C']
                    if sim_ca_candidates and pdb_ca:
                        mapped_coords[sim_ca_candidates[0]] = pdb_coords[pdb_ca[0]]
                        matched_sim.add(sim_ca_candidates[0])
                        matched_pdb.add(pdb_ca[0])
                        mapped_count += 1
                
                # Match the rest by topological distance from CA!
                # For remaining unmatched atoms, we compute their graph distance in simulation to CA,
                # and in PDB to CA (using PDB bonds or simple naming scheme).
                # Naming scheme in PDB is standard:
                # CB: distance 1 from CA
                # CG/CG1/CG2: distance 2 from CA
                # CD/CD1/CD2: distance 3 from CA
                # CE/CE1/CE2: distance 4 from CA
                # CZ: distance 5 from CA
                # Let's get the graph distance from CA in simulation
                sim_ca = [i for i in sim_idxs if i in matched_sim and sim.atoms[i]["element"] == 'C'] # Should be the CA
                if sim_ca:
                    ca_idx = sim_ca[0]
                    # remaining unmatched sim
                    unmatched_sim_idxs = [i for i in sim_idxs if i not in matched_sim]
                    unmatched_pdb_idxs = [i for i in pdb_idxs if i not in matched_pdb]
                    
                    # Sort sim by graph distance to CA
                    sim_dists = [sim.graph_dist[ca_idx, i] for i in unmatched_sim_idxs]
                    sorted_sim_unmatched = [x for _, x in sorted(zip(sim_dists, unmatched_sim_idxs))]
                    
                    # Sort PDB by name suffix depth (CB=1, CG=2, CD=3, CE=4, CZ=5, OH/OD=6, etc.)
                    def pdb_depth(idx):
                        name = pdb_info[idx]["atom_name"]
                        if len(name) < 2: return 0
                        code = name[1]
                        depths = {'B': 1, 'G': 2, 'D': 3, 'E': 4, 'Z': 5, 'H': 6}
                        return depths.get(code, 7)
                        
                    sorted_pdb_unmatched = sorted(unmatched_pdb_idxs, key=pdb_depth)
                    
                    for s_i, p_i in zip(sorted_sim_unmatched, sorted_pdb_unmatched):
                        mapped_coords[s_i] = pdb_coords[p_i]
                        matched_sim.add(s_i)
                        matched_pdb.add(p_i)
                        mapped_count += 1
                        
    print(f"Mapped {mapped_count} out of {sim.n_atoms} atoms.")
    return mapped_coords

if __name__ == "__main__":
    sim = PeptideSimulation("Gly-Tyr-Asp-Pro-Glu-Thr-Gly-Thr-Trp-Gly")
    pdb_coords, pdb_info = parse_heavy_atoms_pdb("1UAO.pdb")
    mapped_coords = align_rdkit_to_pdb(sim, pdb_coords, pdb_info)
