import numpy as np
import networkx as nx
import os
import json
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem

# --- PNIPAM MOLECULAR TORSION AGENT ---
class PNIPAMTorsionAgent:
    def __init__(self, n_monomers=12):
        self.n_monomers = n_monomers
        # Programmatic SMILES for N-mer PNIPAM
        self.smiles = "CC(C)NC(=O)C" + "C(C(=O)NC(C)C)C" * (n_monomers - 1) + "C"
        
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.mol = Chem.AddHs(self.mol)
        
        # Embed 3D coordinates
        params = AllChem.ETKDGv3()
        params.useRandomCoords = True
        AllChem.EmbedMolecule(self.mol, params)
        AllChem.MMFFOptimizeMolecule(self.mol)
        
        self.n_atoms = self.mol.GetNumAtoms()
        conf = self.mol.GetConformer()
        self.initial_coords = np.array([list(conf.GetAtomPosition(i)) for i in range(self.n_atoms)])
        
        # Parse all bonds
        self.bonds = []
        for bond in self.mol.GetBonds():
            self.bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            
        # Compute graph distances
        adj = np.zeros((self.n_atoms, self.n_atoms))
        for src, tgt in self.bonds:
            adj[src, tgt] = 1
            adj[tgt, src] = 1
        G = nx.from_numpy_array(adj)
        self.graph_dist = dict(nx.all_pairs_shortest_path_length(G))
        self.graph_dist = np.array([[self.graph_dist[i].get(p, 999) for p in range(self.n_atoms)] for i in range(self.n_atoms)])
        
        # Identify heavy atoms for topological Laplacian calculations
        self.heavy_atoms = [i for i in range(self.n_atoms) if self.mol.GetAtomWithIdx(i).GetSymbol() != "H"]
        self.heavy_map = {idx: i for i, idx in enumerate(self.heavy_atoms)}
        self.n_heavy = len(self.heavy_atoms)
        
        # Pre-map neighbors for hydrogen bond geometric calculations
        self.hydrogen_n_map = {}
        for i in range(self.n_atoms):
            atom = self.mol.GetAtomWithIdx(i)
            if atom.GetSymbol() == "H":
                for nbr in atom.GetNeighbors():
                    if nbr.GetSymbol() == "N":
                        self.hydrogen_n_map[i] = nbr.GetIdx()
                        
        self.oxygen_c_map = {}
        for i in range(self.n_atoms):
            atom = self.mol.GetAtomWithIdx(i)
            if atom.GetSymbol() == "O":
                for nbr in atom.GetNeighbors():
                    if nbr.GetSymbol() == "C":
                        self.oxygen_c_map[i] = nbr.GetIdx()
                        
        # Identify atom categories
        self.is_polar_acc = np.zeros(self.n_atoms, dtype=bool) # Amide Oxygen
        self.is_polar_don_h = np.zeros(self.n_atoms, dtype=bool) # Amide Hydrogen
        self.is_hydro_c = np.zeros(self.n_atoms, dtype=bool) # Isopropyl Methyl Carbons
        
        for idx in range(self.n_atoms):
            atom = self.mol.GetAtomWithIdx(idx)
            sym = atom.GetSymbol()
            if sym == "O":
                if idx in self.oxygen_c_map:
                    self.is_polar_acc[idx] = True
            elif sym == "H":
                if idx in self.hydrogen_n_map:
                    self.is_polar_don_h[idx] = True
            elif sym == "C":
                c_nbrs = [nbr for nbr in atom.GetNeighbors() if nbr.GetSymbol() == "C"]
                h_nbrs = [nbr for nbr in atom.GetNeighbors() if nbr.GetSymbol() == "H"]
                if len(c_nbrs) == 1 and len(h_nbrs) == 3:
                    self.is_hydro_c[idx] = True
                    
        # Identify rotatable single bonds (joints)
        self.joints = self.identify_rotatable_joints()
        
        # Use natural coiled conformer as reference coordinates
        self.extended_coords = self.initial_coords.copy()
        
    def identify_rotatable_joints(self):
        def is_amide_bond(mol, bond):
            if bond.GetBondType() != Chem.BondType.SINGLE:
                return False
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            if {a1.GetSymbol(), a2.GetSymbol()} == {'C', 'N'}:
                c_atom = a1 if a1.GetSymbol() == 'C' else a2
                for nbr in c_atom.GetNeighbors():
                    if nbr.GetSymbol() == 'O':
                        b = mol.GetBondBetweenAtoms(c_atom.GetIdx(), nbr.GetIdx())
                        if b and b.GetBondType() == Chem.BondType.DOUBLE:
                            return True
            return False

        n_atoms = self.n_atoms
        adj = {i: [] for i in range(n_atoms)}
        for s, t in self.bonds:
            adj[s].append(t)
            adj[t].append(s)
            
        joints = []
        for bond in self.mol.GetBonds():
            if bond.GetBondType() != Chem.BondType.SINGLE:
                continue
            if is_amide_bond(self.mol, bond):
                continue
            s = bond.GetBeginAtomIdx()
            t = bond.GetEndAtomIdx()
            if len(adj[s]) <= 1 or len(adj[t]) <= 1:
                continue
                
            # Find downstream split
            visited = {s: True, t: True}
            downstream = []
            queue = [t]
            while queue:
                curr = queue.pop(0)
                downstream.append(curr)
                for nbr in adj[curr]:
                    if nbr not in visited:
                        visited[nbr] = True
                        queue.append(nbr)
                        
            # Keep index 0 upstream
            if 0 in downstream:
                s, t = t, s
                visited = {s: True, t: True}
                downstream = []
                queue = [t]
                while queue:
                    curr = queue.pop(0)
                    downstream.append(curr)
                    for nbr in adj[curr]:
                        if nbr not in visited:
                            visited[nbr] = True
                            queue.append(nbr)
                            
            if downstream and len(downstream) < n_atoms - 1:
                joints.append({
                    "u_idx": s,
                    "p_idx": t,
                    "downstream_atoms": downstream,
                    "name": f"joint_{s}_{t}"
                })
        return joints

    def generate_stretched_reference_coords(self):
        coords = self.initial_coords.copy()
        mean = np.mean(coords, axis=0)
        coords_centered = coords - mean
        U, S, Vt = np.linalg.svd(coords_centered)
        
        # Stretch along first principal component
        proj = coords_centered @ Vt
        proj[:, 0] *= 3.5
        coords_stretched = proj @ Vt.T + mean
        
        # Relax covalent bonds and steric repulsion to keep normal bond lengths/angles
        relaxed = coords_stretched.copy()
        k_b = 90.0
        d_cov = 1.5
        for _ in range(300):
            grad = np.zeros_like(relaxed)
            for u, v in self.bonds:
                diff = relaxed[u] - relaxed[v]
                d = np.linalg.norm(diff)
                if d > 0.01:
                    g = k_b * (d - d_cov) * (diff / d)
                    grad[u] += g
                    grad[v] -= g
                    
            # Steric repulsion between graph-distant atoms
            diff_all = relaxed[:, np.newaxis, :] - relaxed[np.newaxis, :, :]
            d_all = np.linalg.norm(diff_all, axis=-1)
            for i in range(self.n_atoms):
                for p in range(i + 1, self.n_atoms):
                    if self.graph_dist[i, p] >= 3 and d_all[i, p] < 2.4:
                        diff = relaxed[i] - relaxed[p]
                        d = d_all[i, p]
                        if d < 0.1: d = 0.1
                        g = -80.0 * (d - 0.2)**7 / (((d - 0.2)**8 + 0.01)**2) * (diff / d)
                        grad[i] += g
                        grad[p] -= g
                        
            # Step update
            grad_norms = np.linalg.norm(grad, axis=-1, keepdims=True)
            grad = np.where(grad_norms > 15.0, grad / (grad_norms + 1e-6) * 15.0, grad)
            relaxed -= 0.01 * grad
            
        return relaxed

def rotate_joint(coords, joint, angle_rad):
    u = coords[joint["u_idx"]]
    p = coords[joint["p_idx"]]
    axis = p - u
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-5:
        return coords
    axis = axis / axis_len
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    ux, uy, uz = axis
    K = np.array([
        [0, -uz, uy],
        [uz, 0, -ux],
        [-uy, ux, 0]
    ])
    R = np.eye(3) + sin_a * K + (1.0 - cos_a) * np.dot(K, K)
    new_coords = coords.copy()
    downstream = joint["downstream_atoms"]
    for idx in downstream:
        new_coords[idx] = u + np.dot(R, coords[idx] - u)
    return new_coords

# --- DIFFERENTIABLE TORSION-ANGLE HYBRID FOLDING SIMULATOR ---
def run_pnipam_differentiable_folding(n_monomers=12, steps_per_run=150, learning_rate=0.08):
    print("==================================================")
    print("  PNIPAM DIFFEENTIABLE TORSION-ANGLE FOLDING")
    print("  Implicit Solvent with Dehydration Screening")
    print("==================================================")
    
    agent = PNIPAMTorsionAgent(n_monomers=n_monomers)
    n_atoms = agent.n_atoms
    joints = agent.joints
    n_joints = len(joints)
    
    print(f" Monomers: {n_monomers} | Total Atoms: {n_atoms} | Joints: {n_joints}")
    
    # Dehydration sweep water scenarios
    water_scenarios = [180, 120, 60, 30, 10, 0]
    results = {}
    
    # Starting joint angles (zeros = reference stretched extended conformation)
    theta = np.zeros(n_joints)
    
    # Save sweep coordinates for 3D visualization comparison
    coords_water_180 = None
    coords_water_0 = None
    
    sweep_history = []
    
    for n_water in water_scenarios:
        # scale_hydro: 0.0 at N_W=180 (no hydrophobic force), 1.0 at N_W=0 (max hydrophobic/folding force)
        scale_hydro = 1.0 - (n_water / 180.0)
        print(f"\n--- Running Scenario: N_Water = {n_water} | Hydrophobic scale = {scale_hydro:.3f} ---")
        
        for step in range(steps_per_run):
            # Compute current coordinates
            curr_coords = agent.extended_coords.copy()
            for k, joint in enumerate(joints):
                curr_coords = rotate_joint(curr_coords, joint, theta[k])
                
            # Pairwise distances
            dists = np.linalg.norm(curr_coords[:, None, :] - curr_coords[None, :, :], axis=-1)
            
            # Find active contacts
            active_bonds = []
            
            # A. Amide-Amide Hydrogen Bonds (connecting amide N and carbonyl O)
            for h_idx in range(n_atoms):
                if not agent.is_polar_don_h[h_idx]: continue
                n_idx = agent.hydrogen_n_map[h_idx]
                res_n = h_idx // 19 # approximate monomer index
                
                for o_idx in range(n_atoms):
                    if not agent.is_polar_acc[o_idx]: continue
                    c_idx = agent.oxygen_c_map[o_idx]
                    res_o = o_idx // 19
                    if abs(res_n - res_o) < 2: continue # skip local
                    
                    d_ho = dists[h_idx, o_idx]
                    if d_ho < 4.5:
                        # Angle directionality
                        u_nh = (curr_coords[h_idx] - curr_coords[n_idx]) / (np.linalg.norm(curr_coords[h_idx] - curr_coords[n_idx]) + 1e-6)
                        u_ho = (curr_coords[o_idx] - curr_coords[h_idx]) / (d_ho + 1e-6)
                        u_oc = (curr_coords[c_idx] - curr_coords[o_idx]) / (np.linalg.norm(curr_coords[c_idx] - curr_coords[o_idx]) + 1e-6)
                        
                        cos_theta = np.dot(u_nh, u_ho)
                        cos_phi = np.dot(-u_ho, u_oc)
                        
                        dir_factor = max(0, cos_theta)**2 * max(0, cos_phi)**2
                        # Hydrogen bond strength decreases as dehydration proceeds (polymer contracts and hides amides)
                        hb_scale = 1.0 - 0.4 * scale_hydro
                        w = hb_scale * dir_factor * (20.0 / (d_ho**2 + 0.1))
                        if w > 0.05:
                            active_bonds.append((n_idx, o_idx, w, "hbond", h_idx, o_idx))
                            
            # B. Isopropyl-Isopropyl Hydrophobic Contacts (connecting methyl carbons)
            # These are only active when scale_hydro > 0
            if scale_hydro > 0.01:
                for i in range(n_atoms):
                    if not agent.is_hydro_c[i]: continue
                    for p in range(i + 1, n_atoms):
                        if not agent.is_hydro_c[p]: continue
                        if agent.graph_dist[i, p] < 4: continue
                        
                        d = dists[i, p]
                        if d < 5.0:
                            # Hydrophobic attraction scaled by hydration screening factor
                            w = scale_hydro * (35.0 / (d**2 + 0.1))
                            if w > 0.05:
                                active_bonds.append((i, p, w, "hydro", i, p))
                                
            # Calculate clashes
            clash_count = 0
            clash_pairs = []
            for i in range(n_atoms):
                for p in range(i + 1, n_atoms):
                    if agent.graph_dist[i, p] < 4: continue
                    d = dists[i, p]
                    sym_i = agent.mol.GetAtomWithIdx(i).GetSymbol()
                    sym_p = agent.mol.GetAtomWithIdx(p).GetSymbol()
                    thresh = 1.8 if (sym_i == "H" or sym_p == "H") else 2.5
                    if d < thresh:
                        clash_count += 1
                        clash_pairs.append((i, p, d, thresh))
                        
            # Build Laplacian on heavy atoms only
            G = nx.Graph()
            G.add_nodes_from(range(agent.n_heavy))
            # Covalent heavy bonds
            for src, tgt in agent.bonds:
                if src in agent.heavy_map and tgt in agent.heavy_map:
                    G.add_edge(agent.heavy_map[src], agent.heavy_map[tgt], weight=10.0)
            # Active non-covalent contacts
            for s_idx, t_idx, w, _, _, _ in active_bonds:
                if s_idx in agent.heavy_map and t_idx in agent.heavy_map:
                    G.add_edge(agent.heavy_map[s_idx], agent.heavy_map[t_idx], weight=w)
                    
            # Compute Fiedler algebraic connectivity
            l2 = nx.algebraic_connectivity(G, method='lanczos', tol=1e-5)
            
            # Get Fiedler Vector v2
            L_matrix = nx.laplacian_matrix(G).toarray()
            eigvals, eigvecs = np.linalg.eigh(L_matrix)
            v2 = eigvecs[:, 1]
            
            # Compute Radius of Gyration
            center = np.mean(curr_coords, axis=0)
            rg = np.sqrt(np.mean(np.sum((curr_coords - center)**2, axis=-1)))
            
            # Print status periodically
            if step % 30 == 0 or step == steps_per_run - 1:
                hb_cnt = len([b for b in active_bonds if b[3] == "hbond"])
                hp_cnt = len([b for b in active_bonds if b[3] == "hydro"])
                print(f"   Step {step:<4} | Fiedler (L2): {l2:.5f} | Rg: {rg:.3f} Å | H-Bonds: {hb_cnt} | Hydro-Contacts: {hp_cnt} | Clashes: {clash_count}")
                
            # --- BACKWARD PASS ---
            # Coordinate Jacobians
            dr_dtheta = np.zeros((n_joints, n_atoms, 3))
            for k, joint in enumerate(joints):
                u_idx = joint["u_idx"]
                p_idx = joint["p_idx"]
                axis = curr_coords[p_idx] - curr_coords[u_idx]
                axis = axis / (np.linalg.norm(axis) + 1e-6)
                downstream = joint["downstream_atoms"]
                for idx in downstream:
                    dr_dtheta[k, idx] = np.cross(axis, curr_coords[idx] - curr_coords[u_idx])
                    
            # 1. Fiedler Gradient
            grad_l2 = np.zeros(n_joints)
            # Only apply folding Fiedler gradient if scale_hydro > 0 (during dehydration)
            if scale_hydro > 0.05 and l2 > 1e-6:
                for s_idx, t_idx, w, b_type, d_src, d_tgt in active_bonds:
                    s_heavy = agent.heavy_map[s_idx]
                    t_heavy = agent.heavy_map[t_idx]
                    dv_sq = (v2[s_heavy] - v2[t_heavy])**2
                    d = dists[d_src, d_tgt]
                    
                    # Weight derivative
                    dw_dd = -2.0 * w / (d + 1e-6)
                    
                    for k in range(n_joints):
                        dr_1 = dr_dtheta[k, d_src]
                        dr_2 = dr_dtheta[k, d_tgt]
                        dd_dtheta = (1.0 / max(d, 0.1)) * np.dot(curr_coords[d_src] - curr_coords[d_tgt], dr_1 - dr_2)
                        grad_l2[k] += dv_sq * dw_dd * dd_dtheta
                        
            # 2. Steric Repulsion Gradient
            grad_steric = np.zeros(n_joints)
            for i, p, d, thresh in clash_pairs:
                de_dd = -18.0 / ((max(d - (thresh - 0.1), 0.05))**3)
                for k in range(n_joints):
                    dr_i = dr_dtheta[k, i]
                    dr_p = dr_dtheta[k, p]
                    dd_dtheta = (1.0 / max(d, 0.1)) * np.dot(curr_coords[i] - curr_coords[p], dr_i - dr_p)
                    grad_steric[k] += de_dd * dd_dtheta
                    
            # Normalize gradients
            grad_l2_norm = np.linalg.norm(grad_l2)
            if grad_l2_norm > 1.0: grad_l2 /= grad_l2_norm
            
            grad_steric_norm = np.linalg.norm(grad_steric)
            if grad_steric_norm > 1.0: grad_steric /= grad_steric_norm
            
            # Gradient Update
            # If scale_hydro = 0, we just relax clashes and let thermal noise fluctuate the chain
            if scale_hydro < 0.05:
                gradient = -0.8 * grad_steric
            else:
                gradient = 0.5 * grad_l2 - 0.8 * grad_steric
                
            theta += learning_rate * gradient
            # Add thermal noise to explore conformations
            theta += np.random.normal(0.0, 0.006, n_joints)
            
        # End of scenario results
        hb_final = len([b for b in active_bonds if b[3] == "hbond"])
        hp_final = len([b for b in active_bonds if b[3] == "hydro"])
        
        results[n_water] = {
            "n_water": n_water,
            "rg": float(rg),
            "l2": float(l2),
            "h_bonds": hb_final,
            "hydro_contacts": hp_final,
            "clashes": clash_count,
            "coords": curr_coords.tolist()
        }
        sweep_history.append((n_water, rg, l2, hb_final, hp_final))
        
        if n_water == 180:
            coords_water_180 = curr_coords.copy()
        if n_water == 0:
            coords_water_0 = curr_coords.copy()
            
    # --- ANALYSIS & VISUALIZATION ---
    print("\n==================================================")
    print("  SIMULATION RESULTS SUMMARY")
    print("==================================================")
    print(f" {'N_Water':<8} | {'Rg (Å)':<8} | {'Fiedler (L2)':<15} | {'H-Bonds':<8} | {'Hydro-Contacts':<15}")
    print("--------------------------------------------------")
    for n_water, rg, l2, hb, hp in sweep_history:
        print(f" {n_water:<8} | {rg:<8.3f} | {l2:<15.5f} | {hb:<8} | {hp:<15}")
        
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='#111111')
    ax1.set_facecolor('#111111')
    ax2.set_facecolor('#111111')
    
    n_waters = [item[0] for item in sweep_history]
    rgs = [item[1] for item in sweep_history]
    l2s = [item[2] for item in sweep_history]
    
    # Plot Rg contraction
    ax1.plot(n_waters, rgs, color='#d500ff', marker='o', linewidth=2.5, label='Radius of Gyration (Rg)')
    ax1.invert_xaxis() # 180 -> 0
    ax1.set_xlabel("Water Molecule Count (N_Water)", color='#e5e5ed', fontsize=12)
    ax1.set_ylabel("Radius of Gyration Rg (Å)", color='#e5e5ed', fontsize=12)
    ax1.set_title("PNIPAM Chain Contraction (Coil-Globule)", color='#e5e5ed', fontsize=14, fontweight='bold')
    ax1.tick_params(colors='#e5e5ed')
    ax1.grid(color='#333333', linestyle='--')
    ax1.legend(facecolor='#222222', edgecolor='none', labelcolor='#e5e5ed')
    
    # Plot Fiedler Connectivity
    ax2.plot(n_waters, l2s, color='#00ffd5', marker='s', linewidth=2.5, label='Fiedler Connectivity (λ2)')
    ax2.invert_xaxis()
    ax2.set_xlabel("Water Molecule Count (N_Water)", color='#e5e5ed', fontsize=12)
    ax2.set_ylabel("Laplacian Algebraic Connectivity (λ2)", color='#e5e5ed', fontsize=12)
    ax2.set_title("Topological Contact Network Connectivity", color='#e5e5ed', fontsize=14, fontweight='bold')
    ax2.tick_params(colors='#e5e5ed')
    ax2.grid(color='#333333', linestyle='--')
    ax2.legend(facecolor='#222222', edgecolor='none', labelcolor='#e5e5ed')
    
    plt.suptitle("Torsion-Angle Hybrid Differentiable Folding of PNIPAM Chain", color='#e5e5ed', fontsize=16, fontweight='bold', y=0.97)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    plot_dir = "/home/eldenring/.gemini/antigravity/brain/d1e57b48-f8cc-4dd3-9357-bbffb4c85cbd"
    os.makedirs(plot_dir, exist_ok=True)
    curve_path = os.path.join(plot_dir, "pnipam_differentiable_folding_transition.png")
    plt.savefig(curve_path, dpi=150, facecolor='#111111', edgecolor='none')
    plt.close()
    
    # Generate 3D side-by-side plot of PNIPAM structure
    fig_3d = plt.figure(figsize=(14, 7), facecolor='#111111')
    
    def plot_3d_state(coords, title, subplot_idx):
        ax = fig_3d.add_subplot(1, 2, subplot_idx, projection='3d')
        ax.set_facecolor('#111111')
        
        # Heavy atoms color code
        bb_idx = []
        am_idx = []
        ip_idx = []
        
        # Let's simple check atoms index mapping
        for idx in range(n_atoms):
            atom = agent.mol.GetAtomWithIdx(idx)
            sym = atom.GetSymbol()
            if sym == "O" or sym == "N":
                am_idx.append(idx)
            elif idx in agent.heavy_atoms:
                # If terminal methyl
                if agent.is_hydro_c[idx]:
                    ip_idx.append(idx)
                else:
                    bb_idx.append(idx)
                    
        # Plot heavy atoms
        ax.scatter(coords[bb_idx, 0], coords[bb_idx, 1], coords[bb_idx, 2], color='#8e8e9f', s=60, label='Backbone C')
        ax.scatter(coords[am_idx, 0], coords[am_idx, 1], coords[am_idx, 2], color='#00ffd5', s=75, label='Amide N/O')
        ax.scatter(coords[ip_idx, 0], coords[ip_idx, 1], coords[ip_idx, 2], color='#d500ff', s=90, label='Isopropyl C')
        
        # Draw bonds between heavy atoms
        for u, v in agent.bonds:
            if u in agent.heavy_map and v in agent.heavy_map:
                ax.plot([coords[u, 0], coords[v, 0]], 
                        [coords[u, 1], coords[v, 1]], 
                        [coords[u, 2], coords[v, 2]], color='#555555', linewidth=2.0)
                
        ax.set_title(title, color='#e5e5ed', fontsize=14, fontweight='bold', pad=10)
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        # Find limits
        lim = 18.0
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        
    plot_3d_state(coords_water_180, "Fully Hydrated Extended State (N_Water = 180)\nSwollen Coil (Rg = {:.3f} Å)".format(sweep_history[0][1]), 1)
    plot_3d_state(coords_water_0, "Fully Dehydrated Collapsed State (N_Water = 0)\nCompact Globule (Rg = {:.3f} Å)".format(sweep_history[-1][1]), 2)
    
    handles, labels = fig_3d.axes[0].get_legend_handles_labels()
    fig_3d.legend(handles, labels, loc='lower center', ncol=3, facecolor='#222222', edgecolor='none', labelcolor='#e5e5ed')
    plt.suptitle("3D Conformation: PNIPAM Torsion-Angle Differentiable Folding", color='#e5e5ed', fontsize=16, fontweight='bold', y=0.96)
    plt.tight_layout(rect=[0, 0.08, 1, 0.94])
    
    structure_path = os.path.join(plot_dir, "pnipam_differentiable_folding_structure.png")
    plt.savefig(structure_path, dpi=150, facecolor='#111111', edgecolor='none')
    plt.close()
    
    # Save results to JSON
    json_path = "pnipam_differentiable_folding_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nSaved transition plots to:\n - {curve_path}\n - {structure_path}")
    print(f"Saved results database to {json_path}")
    print("==================================================")

if __name__ == "__main__":
    run_pnipam_differentiable_folding(n_monomers=12, steps_per_run=150, learning_rate=0.08)
