import json
import os
import numpy as np

# Import the simulation engines to extract chemical structures (bonds, types) dynamically
from pnipam_single_chain_crosslinked import PNIPAMSingleCrosslinkedSimulation
from pnipam_branched_dimer import PNIPAMBranchedDimerSimulation
from pnipam_differentiable_folding import PNIPAMTorsionAgent

def compile_data():
    unified_data = {}
    
    # 1. PEPTIDE SYSTEMS
    peptide_files = {
        "Deca-alanine (AAAAAAAAAA) - Differentiable": "aaaaaaaaaa_differentiable_folded.json",
        "Deca-alanine (AAAAAAAAAA) - Hybrid": "aaaaaaaaaa_hybrid_folded.json",
        "Chignolin Mutant (CLN025) - Differentiable": "yydpetgtwy_differentiable_folded.json",
        "Chignolin Mutant (CLN025) - Hybrid": "yydpetgtwy_hybrid_folded.json",
        "Wild-type Chignolin (1UAO) - Differentiable": "gydpetgtwg_differentiable_folded.json",
        "Wild-type Chignolin (1UAO) - Hybrid": "gydpetgtwg_hybrid_folded.json",
        "Trp-cage (1L2Y) - Differentiable": "nlyiqwlkdggpssgrppps_differentiable_folded.json",
        "Trp-cage (1L2Y) - Hybrid": "nlyiqwlkdggpssgrppps_hybrid_folded.json"
    }
    
    for label, filename in peptide_files.items():
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data = json.load(f)
                # Format to be consistent
                unified_data[label] = {
                    "system_type": "peptide",
                    "sequence": data.get("sequence", ""),
                    "atoms": data.get("atoms", []),
                    "atomResidues": data.get("atomResidues", []),
                    "foldedCoords": data.get("foldedCoords", [])
                }
            print(f"Compiled Peptide: {label}")
        else:
            print(f"Warning: Peptide file {filename} not found, skipping.")
            
    # 2. PNIPAM POLYMER SYSTEMS
    print("\nExtracting PNIPAM Ring Polymer metadata...")
    sim_ring = PNIPAMSingleCrosslinkedSimulation(n_monomers=15, n_water=0)
    ring_bonds = sim_ring.bonds
    ring_types = list(sim_ring.types)
    ring_hydro = sim_ring.is_hydro_c.tolist()
    ring_polar_acc = sim_ring.is_polar_acc.tolist()
    ring_polar_don = sim_ring.is_polar_don_h.tolist()
    
    if os.path.exists("pnipam_single_crosslinked_results.json"):
        with open("pnipam_single_crosslinked_results.json", "r") as f:
            raw_ring = json.load(f)
            unified_data["PNIPAM Ring Polymer (15-mer Loop)"] = {
                "system_type": "polymer",
                "name": "PNIPAM Single-Chain Self-Crosslinked Ring",
                "n_poly": sim_ring.n_poly,
                "bonds": ring_bonds,
                "types": ring_types,
                "is_hydro_c": ring_hydro,
                "is_polar_acc": ring_polar_acc,
                "is_polar_don_h": ring_polar_don,
                "scenarios": raw_ring
            }
        print("Compiled PNIPAM Ring Polymer")
        
    print("\nExtracting PNIPAM Branched Polymer metadata...")
    sim_branched = PNIPAMBranchedDimerSimulation(n_monomers=15, n_water=0)
    branched_bonds = sim_branched.bonds
    branched_types = list(sim_branched.types)
    branched_hydro = sim_branched.is_hydro_c.tolist()
    branched_polar_acc = sim_branched.is_polar_acc.tolist()
    branched_polar_don = sim_branched.is_polar_don_h.tolist()
    
    if os.path.exists("pnipam_branched_dimer_results.json"):
        with open("pnipam_branched_dimer_results.json", "r") as f:
            raw_branched = json.load(f)
            unified_data["PNIPAM Branched Polymer (H-Dimer)"] = {
                "system_type": "polymer",
                "name": "PNIPAM H-Shaped Branched Dimer",
                "n_poly": sim_branched.n_poly,
                "bonds": branched_bonds,
                "types": branched_types,
                "is_hydro_c": branched_hydro,
                "is_polar_acc": branched_polar_acc,
                "is_polar_don_h": branched_polar_don,
                "scenarios": raw_branched
            }
        print("Compiled PNIPAM Branched Polymer")

    print("\nExtracting PNIPAM Differentiable Folding metadata...")
    agent_diff = PNIPAMTorsionAgent(n_monomers=12)
    diff_bonds = agent_diff.bonds
    diff_types = [agent_diff.mol.GetAtomWithIdx(i).GetSymbol() for i in range(agent_diff.n_atoms)]
    diff_hydro = agent_diff.is_hydro_c.tolist()
    diff_polar_acc = agent_diff.is_polar_acc.tolist()
    diff_polar_don = agent_diff.is_polar_don_h.tolist()
    
    if os.path.exists("pnipam_differentiable_folding_results.json"):
        with open("pnipam_differentiable_folding_results.json", "r") as f:
            raw_diff = json.load(f)
            # Reformat to match scenarios key
            scenarios = {}
            for k, val in raw_diff.items():
                scenarios[k] = {
                    "n_water": val.get("n_water", int(k)),
                    "scale_hydro": 1.0 - (int(k) / 180.0),
                    "rg": val.get("rg", 0.0),
                    "l2": val.get("l2", 0.0),
                    "hbonds": val.get("h_bonds", 0),
                    "hydro": val.get("hydro_contacts", 0),
                    "coords": val.get("coords", [])
                }
            unified_data["PNIPAM Torsion-Angle Single Chain (12-mer)"] = {
                "system_type": "polymer",
                "name": "PNIPAM Torsion-Angle Differentiable Folding",
                "n_poly": agent_diff.n_atoms,
                "bonds": diff_bonds,
                "types": diff_types,
                "is_hydro_c": diff_hydro,
                "is_polar_acc": diff_polar_acc,
                "is_polar_don_h": diff_polar_don,
                "scenarios": scenarios
            }
        print("Compiled PNIPAM Torsion-Angle Single Chain")

    # Output to viewer_data.js
    with open("viewer_data.js", "w") as f:
        f.write("const VIEWER_DATA = ")
        json.dump(unified_data, f, indent=2)
        f.write(";\n")
        
    print("\nSuccessfully compiled all data to viewer_data.js!")

if __name__ == "__main__":
    compile_data()
