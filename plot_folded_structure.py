import json
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_peptide_structure(json_path, output_png_path):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return
        
    with open(json_path, "r") as f:
        data = json.load(f)
        
    sequence = data["sequence"]
    coords = np.array(data["foldedCoords"])
    # Use PeptideAgent to find proper backbone indices and residues
    from differentiable_folding_directional import PeptideAgent
    peptide = PeptideAgent(sequence)
    
    ca_indices = []
    ca_labels = []
    o_indices = []
    n_indices = []
    atom_residues = peptide.atom_residues
    
    for idx in range(peptide.n_atoms):
        res_info = peptide.mol.GetAtomWithIdx(idx).GetPDBResidueInfo()
        if res_info is not None:
            name = res_info.GetName().strip()
            if name == "CA":
                ca_indices.append(idx)
                ca_labels.append(f"{sequence[atom_residues[idx]]}{atom_residues[idx]+1}")
            elif name == "O":
                o_indices.append(idx)
            elif name == "N":
                n_indices.append(idx)
                
    ca_coords = coords[ca_indices]
    h_bonds = []
    
    for o_idx in o_indices:
        for n_idx in n_indices:
            # Check residue distance to avoid covalent connections (graph dist)
            if abs(atom_residues[o_idx] - atom_residues[n_idx]) < 3:
                continue
            d = np.linalg.norm(coords[o_idx] - coords[n_idx])
            if d < 4.5:
                h_bonds.append((coords[o_idx], coords[n_idx], d))
                
    # Calculate Radius of Gyration
    center = np.mean(coords, axis=0)
    rg = np.sqrt(np.mean(np.sum((coords - center)**2, axis=-1)))
    
    # Create 3D plot
    fig = plt.figure(figsize=(8, 7), facecolor='#111111')
    ax = fig.add_subplot(111, projection='3d', facecolor='#111111')
    
    # Plot C-alpha trace with a gradient color (N to C terminal)
    n_ca = len(ca_coords)
    colors = plt.cm.cool(np.linspace(0, 1, n_ca))
    
    # Draw backbone segments with gradient color
    for idx in range(n_ca - 1):
        ax.plot(
            ca_coords[idx:idx+2, 0], 
            ca_coords[idx:idx+2, 1], 
            ca_coords[idx:idx+2, 2], 
            color=colors[idx], 
            linewidth=4, 
            solid_capstyle='round',
            alpha=0.9
        )
        
    # Draw C-alpha residue markers
    ax.scatter(
        ca_coords[:, 0], ca_coords[:, 1], ca_coords[:, 2], 
        color=colors, 
        s=120, 
        edgecolors='white', 
        linewidths=1.2,
        alpha=1.0,
        zorder=10
    )
    
    # Label N-terminal and C-terminal
    ax.text(
        ca_coords[0, 0], ca_coords[0, 1], ca_coords[0, 2] + 0.5, 
        "N-term", color='#00e1ff', fontsize=10, weight='bold', ha='center'
    )
    ax.text(
        ca_coords[-1, 0], ca_coords[-1, 1], ca_coords[-1, 2] + 0.5, 
        "C-term", color='#ff4d4d', fontsize=10, weight='bold', ha='center'
    )
    
    # Draw hydrogen bonds as cyan dashed lines
    for start, end, d in h_bonds:
        ax.plot(
            [start[0], end[0]], 
            [start[1], end[1]], 
            [start[2], end[2]], 
            color='#00ffd5', 
            linestyle='--', 
            linewidth=1.5, 
            alpha=0.7
        )
        # Mark midpoints with a small dot
        mid = (start + end) / 2.0
        ax.scatter([mid[0]], [mid[1]], [mid[2]], color='#00ffd5', s=15, alpha=0.8)
        
    # Aesthetic settings
    ax.set_title(
        f"Folded Structure: {sequence}\nRadius of Gyration: {rg:.3f} Å | H-bonds: {len(h_bonds)}", 
        color='white', 
        fontsize=13, 
        weight='bold',
        pad=15
    )
    
    # Make axis grid dark and clean
    ax.xaxis.set_pane_color((0.08, 0.08, 0.08, 1.0))
    ax.yaxis.set_pane_color((0.08, 0.08, 0.08, 1.0))
    ax.zaxis.set_pane_color((0.08, 0.08, 0.08, 1.0))
    
    ax.grid(color='#333333', linestyle=':', linewidth=0.5)
    
    # Force ticks and labels to white
    ax.tick_params(colors='white', labelsize=8)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    
    # Remove axis labels for a cleaner look
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    # Equal aspect ratio scaling (box layout helper)
    all_coords = np.vstack([ca_coords, coords])
    max_range = np.array([
        all_coords[:, 0].max() - all_coords[:, 0].min(), 
        all_coords[:, 1].max() - all_coords[:, 1].min(), 
        all_coords[:, 2].max() - all_coords[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (all_coords[:, 0].max() + all_coords[:, 0].min()) / 2.0
    mid_y = (all_coords[:, 1].max() + all_coords[:, 1].min()) / 2.0
    mid_z = (all_coords[:, 2].max() + all_coords[:, 2].min()) / 2.0
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=150, facecolor='#111111', edgecolor='none')
    plt.close()
    print(f"Successfully generated 3D plot: {output_png_path}")

if __name__ == "__main__":
    art_dir = "/home/eldenring/.gemini/antigravity/brain/d1e57b48-f8cc-4dd3-9357-bbffb4c85cbd"
    os.makedirs(art_dir, exist_ok=True)
    
    peptides = [
        ("aaaaaaaaaa_differentiable_folded.json", "aaaaaaaaaa_structure.png"),
        ("aaaaaaaaaa_hybrid_folded.json", "aaaaaaaaaa_hybrid_structure.png"),
        ("yydpetgtwy_differentiable_folded.json", "yydpetgtwy_structure.png"),
        ("yydpetgtwy_hybrid_folded.json", "yydpetgtwy_hybrid_structure.png"),
        ("gydpetgtwg_differentiable_folded.json", "gydpetgtwg_structure.png"),
        ("gydpetgtwg_hybrid_folded.json", "gydpetgtwg_hybrid_structure.png"),
        ("nlyiqwlkdggpssgrppps_differentiable_folded.json", "nlyiqwlkdggpssgrppps_structure.png"),
        ("nlyiqwlkdggpssgrppps_hybrid_folded.json", "nlyiqwlkdggpssgrppps_hybrid_structure.png")
    ]
    
    for json_file, png_file in peptides:
        if os.path.exists(json_file):
            plot_peptide_structure(json_file, os.path.join(art_dir, png_file))
        else:
            print(f"Skipping missing file: {json_file}")
