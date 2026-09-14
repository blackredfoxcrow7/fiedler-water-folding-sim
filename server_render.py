import sys
import os
import numpy as np
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
            
        return jsonify({
            "status": "success",
            "sequence": sequence,
            "atoms": peptide.atoms,
            "bonds": peptide.bonds,
            "unfoldedCoords": unfolded_coords,
            "foldedCoords": folded_coords,
            "hasFolded": has_folded,
            "atomResidues": peptide.atom_residues,
            "residueLabels": peptide.residue_labels
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

if __name__ == "__main__":
    # Start server on port 5006
    app.run(host="0.0.0.0", port=5006, debug=False)
