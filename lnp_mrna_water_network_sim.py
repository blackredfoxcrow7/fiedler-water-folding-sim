"""
Lipid Nanoparticle (LNP) mRNA Encapsulation & pH-Triggered Water Channel Release Simulator
Graph-Spectral Topological Analysis of LNP Internal Water Networks
Concept by Yoshihiro Honda (本多 義弘)

Models:
1. Neutral Storage State (pH 7.4): Unprotonated ionizable lipids, stable water network, mRNA locked in LNP core.
2. Endosomal Activation State (pH 5.5): Protonated ionizable lipids (NH+), rapid expansion of interconnected water channels (percolation jump in λ2), driving endosomal membrane disruption and mRNA release.
"""

import numpy as np
import scipy.linalg as la
import json

def build_lnp_mrna_core(ph_level=7.4, n_lipids=60, n_water=120, n_mrna_nodes=20, core_radius=12.0):
    """
    Builds graph model of LNP internal core containing:
    - Ionizable Lipids (SM-102 / ALC-0315 type)
    - Helper Lipids (DSPC) & Cholesterol
    - Encapsulated mRNA strand
    - Internal Water Network (bound/channel water)
    """
    np.random.seed(42 if ph_level > 6.5 else 101)
    
    pos = []
    labels = []
    
    # 1. Encapsulated mRNA backbone in core center (Helical strand)
    mrna_indices = []
    t_vals = np.linspace(-np.pi * 2, np.pi * 2, n_mrna_nodes)
    for i, t in enumerate(t_vals):
        x = 3.0 * np.cos(t)
        y = 3.0 * np.sin(t)
        z = (i / (n_mrna_nodes - 1)) * 14.0 - 7.0
        pos.append([x, y, z])
        labels.append("mRNA_Phosphate")
        mrna_indices.append(i)
        
    # 2. Lipid polar heads around mRNA and in core matrix
    # At pH 5.5, ionizable lipids become protonated (positively charged NH+), causing repulsion and water influx
    is_protonated = (ph_level < 6.0)
    
    lipid_indices = []
    for i in range(n_lipids):
        r = np.random.uniform(4.0, core_radius)
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        pos.append([x, y, z])
        if i % 3 == 0:
            labels.append("Ionizable_Lipid_NH+" if is_protonated else "Ionizable_Lipid_N")
        elif i % 3 == 1:
            labels.append("DSPC_Head")
        else:
            labels.append("Cholesterol")
        lipid_indices.append(len(pos) - 1)
        
    # 3. Water molecules forming internal network
    # Protonation at pH 5.5 triggers high water influx & formation of continuous water percolation channels
    water_indices = []
    effective_n_water = int(n_water * (1.4 if is_protonated else 1.0))
    
    for i in range(effective_n_water):
        if is_protonated:
            # At pH 5.5, water forms continuous 1D/3D percolation channels through the core
            # Water molecules align along channels connecting mRNA to LNP surface
            z = np.random.uniform(-core_radius, core_radius)
            r = np.random.uniform(1.0, core_radius * 0.95)
            angle = np.random.uniform(0, 2 * np.pi)
            x = r * np.cos(angle)
            y = r * np.sin(angle)
        else:
            # At pH 7.4, water molecules are localized around mRNA & lipid heads
            r = np.random.uniform(2.0, core_radius * 0.7)
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            
        pos.append([x, y, z])
        labels.append("H2O_channel" if is_protonated else "H2O_bound")
        water_indices.append(len(pos) - 1)
        
    pos = np.array(pos)
    N = len(pos)
    adj = np.zeros((N, N))
    
    # Build Adjacency Matrix with interaction weights
    for i in range(N):
        for j in range(i+1, N):
            p1, p2 = pos[i], pos[j]
            dist = np.linalg.norm(p1 - p2)
            
            cutoff = 5.2 if is_protonated else 4.2
            if dist < cutoff:
                w = np.exp(-dist**2 / 12.0)
                
                # Enhanced water-water hydrogen bonding network at pH 5.5
                if "H2O" in labels[i] and "H2O" in labels[j]:
                    w *= (2.2 if is_protonated else 1.0)
                    
                # Electrostatic binding between protonated lipid (NH+) and mRNA phosphate (PO4-)
                if ("NH+" in labels[i] and "mRNA" in labels[j]) or ("NH+" in labels[j] and "mRNA" in labels[i]):
                    w *= 3.0 # Strong electrostatic complexation
                    
                adj[i, j] = w
                adj[j, i] = w
                
    return pos, adj, labels, water_indices, mrna_indices

def analyze_lnp_water_percolation(pos, adj, labels, water_indices):
    """Compute Fiedler Spectrum and Water Network Percolation Density."""
    N = len(pos)
    deg = np.diag(np.sum(adj, axis=1))
    L = deg - adj
    
    # Global Fiedler Eigenvalue
    evals = la.eigvalsh(L)
    nonzero_evals = [e for e in evals if e > 1e-5]
    lambda2_global = nonzero_evals[0] if len(nonzero_evals) > 0 else 0.0
    
    # Subgraph for internal water network
    n_w = len(water_indices)
    w_adj = adj[np.ix_(water_indices, water_indices)]
    w_deg = np.diag(np.sum(w_adj, axis=1))
    w_L = w_deg - w_adj
    w_evals = la.eigvalsh(w_L)
    w_nonzero = [e for e in w_evals if e > 1e-5]
    lambda2_water = w_nonzero[0] if len(w_nonzero) > 0 else 0.0
    
    # Water Percolation Index (Mean degree of water network)
    water_percolation_index = float(np.sum(w_adj) / (n_w + 1e-6))
    
    # mRNA Release Kinetic Factor (Proportional to water channel connectivity and ionic repulsion)
    release_kinetic_factor = lambda2_water * water_percolation_index
    
    return {
        "lambda2_global": float(lambda2_global),
        "lambda2_water_subgraph": float(lambda2_water),
        "water_percolation_index": water_percolation_index,
        "release_kinetic_factor": float(release_kinetic_factor)
    }

def main():
    print("==========================================================================")
    print("  LNP mRNA Encapsulation & Water Network Release Simulator (Honda Concept)  ")
    print("==========================================================================")
    
    # 1. Neutral Storage State (pH 7.4)
    pos_74, adj_74, labels_74, w_idx_74, _ = build_lnp_mrna_core(ph_level=7.4)
    res_74 = analyze_lnp_water_percolation(pos_74, adj_74, labels_74, w_idx_74)
    
    print("\n--- 1. Neutral Storage State (pH 7.4 - Physiological / Storage) ---")
    print(f"Global Fiedler Value (λ2): {res_74['lambda2_global']:.4f}")
    print(f"Water Network Subgraph Fiedler (λ2_water): {res_74['lambda2_water_subgraph']:.4f}")
    print(f"Water Channel Percolation Index: {res_74['water_percolation_index']:.4f} (Localized / Isolated)")
    print(f"mRNA Release Kinetic Factor: {res_74['release_kinetic_factor']:.4f} (Stable Encapsulation)")
    
    # 2. Endosomal Activation State (pH 5.5)
    pos_55, adj_55, labels_55, w_idx_55, _ = build_lnp_mrna_core(ph_level=5.5)
    res_55 = analyze_lnp_water_percolation(pos_55, adj_55, labels_55, w_idx_55)
    
    print("\n--- 2. Endosomal Activation State (pH 5.5 - Protonated / Release Phase) ---")
    print(f"Global Fiedler Value (λ2): {res_55['lambda2_global']:.4f}")
    print(f"Water Network Subgraph Fiedler (λ2_water): {res_55['lambda2_water_subgraph']:.4f} (+{((res_55['lambda2_water_subgraph']/res_74['lambda2_water_subgraph'])-1)*100:.1f}% Jump!)")
    print(f"Water Channel Percolation Index: {res_55['water_percolation_index']:.4f} (Continuous Water Channels Formed)")
    print(f"mRNA Release Kinetic Factor: {res_55['release_kinetic_factor']:.4f} (Rapid Endosomal Escape & Release!)")

    results = {
        "ph_7_4_storage": res_74,
        "ph_5_5_endosomal": res_55
    }
    
    with open("lnp_mrna_water_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nSimulation results successfully saved to lnp_mrna_water_results.json.")

if __name__ == "__main__":
    main()
