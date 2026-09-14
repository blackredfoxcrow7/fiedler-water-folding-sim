import sys
import os
import numpy as np
import networkx as nx
from flask import Flask, jsonify, request
from rdkit import Chem

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from peptide_agent import PeptideAgent

app = Flask(__name__)

# CORS headers middleware
@app.after_request
def add_cors_headers(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

def center_coordinates(coords):
    """Centers coordinates around the center of mass."""
    com = np.mean(coords, axis=0)
    return (coords - com).tolist()

def calculate_fiedler_metrics(atoms, bonds, coords, threshold=4.5):
    """
    Constructs a spatial graph network where nodes are atoms,
    edges are covalent bonds + distance-based contact edges (d < threshold).
    Returns (fiedler_value, num_edges, density).
    """
    G = nx.Graph()
    
    # 1. Add all atoms as nodes
    for atom in atoms:
        G.add_node(atom["id"], element=atom["element"])
        
    # 2. Add covalent bond edges
    for bond in bonds:
        G.add_edge(bond["source"], bond["target"], type="covalent")
        
    # 3. Add distance-based non-covalent contact edges
    n_atoms = len(atoms)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            pos_i = np.array(coords[i])
            pos_j = np.array(coords[j])
            dist = np.linalg.norm(pos_i - pos_j)
            if dist < threshold:
                G.add_edge(i, j, type="contact")
                
    # 4. Calculate Fiedler Value (algebraic connectivity)
    try:
        fiedler = nx.algebraic_connectivity(G)
    except Exception as e:
        print(f"Error calculating Fiedler value: {e}")
        fiedler = 0.0
        
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    # Graph density: 2 * E / (V * (V - 1))
    density = (2 * num_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0.0
    
    return fiedler, num_edges, density

@app.route('/api/structure', methods=['GET', 'POST'])
def get_structure():
    try:
        # Support user-provided sequence, default to Chignolin validation sequence
        sequence = "GYDPETGTWG"
        if request.is_json:
            req_data = request.json or {}
            sequence = req_data.get("sequence", "GYDPETGTWG")
        else:
            sequence = request.args.get("sequence", "GYDPETGTWG")
            
        # 1. Initialize PeptideAgent for topology and initial coordinates (unfolded)
        peptide = PeptideAgent(sequence)
        unfolded_coords = center_coordinates(peptide.initial_coords)
        
        # 2. Parse folded coordinates from 1UAO.pdb if sequence is the validation sequence
        folded_coords = []
        has_folded = False
        
        if sequence.strip().upper() == "GYDPETGTWG":
            pdb_path = "1UAO.pdb"
            if os.path.exists(pdb_path):
                try:
                    pdb_mol = Chem.MolFromPDBFile(pdb_path, removeHs=True)
                    if pdb_mol:
                        pdb_conf = pdb_mol.GetConformer()
                        raw_folded = np.zeros_like(peptide.initial_coords)
                        for i in range(min(peptide.n_atoms, pdb_mol.GetNumAtoms())):
                            pos = pdb_conf.GetAtomPosition(i)
                            raw_folded[i] = [pos.x, pos.y, pos.z]
                        folded_coords = center_coordinates(raw_folded)
                        has_folded = True
                except Exception as e:
                    print(f"Error reading PDB file: {e}")
                    
        # If no validation folded conformation exists, we copy the unfolded coords
        if not has_folded:
            folded_coords = unfolded_coords
            
        # 3. Calculate Fiedler values and topological metrics for both states
        threshold = 4.5 # Angstroms
        unfolded_fiedler, unfolded_edges, unfolded_density = calculate_fiedler_metrics(
            peptide.atoms, peptide.bonds, unfolded_coords, threshold
        )
        folded_fiedler, folded_edges, folded_density = calculate_fiedler_metrics(
            peptide.atoms, peptide.bonds, folded_coords, threshold
        )
        
        # 4. Print results to terminal console
        print(f"\n========================================================")
        print(f"  PEPTIDE NETWORK GRAPH COMPARISON (Sequence: {sequence})")
        print(f"  Distance Threshold for Contacts: {threshold} Angstroms")
        print(f"--------------------------------------------------------")
        print(f"  UNFOLDED STATE:")
        print(f"    Fiedler Value (Algebraic Connectivity): {unfolded_fiedler:.6f}")
        print(f"    Total Graph Edges (Bonds + Contacts):    {unfolded_edges}")
        print(f"    Graph Density:                           {unfolded_density:.4f}")
        print(f"  FOLDED STATE:")
        print(f"    Fiedler Value (Algebraic Connectivity): {folded_fiedler:.6f}")
        print(f"    Total Graph Edges (Bonds + Contacts):    {folded_edges}")
        print(f"    Graph Density:                           {folded_density:.4f}")
        print(f"========================================================\n")
        sys.stdout.flush() # Ensure it outputs to terminal instantly
        
        return jsonify({
            "status": "success",
            "sequence": sequence,
            "atoms": peptide.atoms,
            "bonds": peptide.bonds,
            "unfoldedCoords": unfolded_coords,
            "foldedCoords": folded_coords,
            "hasFolded": has_folded,
            "atomResidues": peptide.atom_residues,
            "residueLabels": peptide.residue_labels,
            # Network metrics
            "unfoldedFiedler": unfolded_fiedler,
            "unfoldedEdges": unfolded_edges,
            "unfoldedDensity": unfolded_density,
            "foldedFiedler": folded_fiedler,
            "foldedEdges": folded_edges,
            "foldedDensity": folded_density,
            "contactThreshold": threshold
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

if __name__ == "__main__":
    # Start server on port 5007
    app.run(host="0.0.0.0", port=5007, debug=False)
