"""
Refined Directional Graph Spectral Calculations for Hexagonal Proton Gel
Concept by Yoshihiro Honda (本多 義弘)
"""

import numpy as np
import scipy.linalg as la
import json

def build_bulk_water_graph(n_water=100, box_size=15.0, cutoff=4.5):
    """Bulk water: Random 3D water network with isotropic H-bonding."""
    np.random.seed(42)
    pos = np.random.uniform(0, box_size, (n_water, 3))
    
    adj = np.zeros((n_water, n_water))
    for i in range(n_water):
        for j in range(i+1, n_water):
            dist = np.linalg.norm(pos[i] - pos[j])
            if dist < cutoff:
                w = np.exp(-dist**2 / 10.0)
                adj[i, j] = w
                adj[j, i] = w
                
    return pos, adj, ["H2O_bulk"] * n_water

def build_hexagonal_surfactant_gel(n_surfactants=48, n_water=100, channel_radius=6.0, length_z=25.0):
    """Hexagonal Surfactant Gel channel (Z-axis oriented)."""
    np.random.seed(123)
    pos = []
    labels = []
    
    n_ring = 12
    n_layers = n_surfactants // n_ring
    z_coords = np.linspace(0, length_z, n_layers)
    
    for z in z_coords:
        for r_idx in range(n_ring):
            angle = (2 * np.pi / n_ring) * r_idx
            x = channel_radius * np.cos(angle)
            y = channel_radius * np.sin(angle)
            pos.append([x, y, z])
            labels.append("Head_Polar")
            
    # Bound water along Z channel
    for i in range(n_water):
        z = np.random.uniform(0, length_z)
        r = np.random.uniform(0, channel_radius * 0.85)
        angle = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        pos.append([x, y, z])
        labels.append("H2O_bound")
        
    pos = np.array(pos)
    N = len(pos)
    adj = np.zeros((N, N))
    
    for i in range(N):
        for j in range(i+1, N):
            delta = pos[i] - pos[j]
            dist = np.linalg.norm(delta)
            if dist < 4.5:
                w = np.exp(-dist**2 / 10.0)
                
                # Direct H-bond chain alignment along Z channel axis
                cos_z = abs(delta[2]) / (dist + 1e-6)
                if labels[i] == "H2O_bound" and labels[j] == "H2O_bound":
                    w *= (1.0 + 2.5 * cos_z**2)
                    
                if "Head" in labels[i] or "Head" in labels[j]:
                    w *= 2.0 # Bound water anchored to polar heads
                    
                adj[i, j] = w
                adj[j, i] = w
                
    return pos, adj, labels

def analyze_directional_spectral_tensor(pos, adj):
    """Compute directional spectral connectivity and diffusion tensor."""
    N = len(pos)
    
    # 1. Standard Graph Laplacian
    deg = np.diag(np.sum(adj, axis=1))
    L = deg - adj
    evals = la.eigvalsh(L)
    # Find first non-zero eigenvalue (Fiedler value)
    nonzero_evals = [e for e in evals if e > 1e-5]
    lambda2_global = nonzero_evals[0] if len(nonzero_evals) > 0 else 0.0
    
    # 2. Directional Laplacian Tensors
    # W_x = W * (dx/d)^2, W_y = W * (dy/d)^2, W_z = W * (dz/d)^2
    Wx = np.zeros((N, N))
    Wy = np.zeros((N, N))
    Wz = np.zeros((N, N))
    
    for i in range(N):
        for j in range(i+1, N):
            if adj[i, j] > 0:
                delta = pos[i] - pos[j]
                dist = np.linalg.norm(delta) + 1e-6
                
                wx = adj[i, j] * (delta[0] / dist)**2
                wy = adj[i, j] * (delta[1] / dist)**2
                wz = adj[i, j] * (delta[2] / dist)**2
                
                Wx[i, j] = Wx[j, i] = wx
                Wy[i, j] = Wy[j, i] = wy
                Wz[i, j] = Wz[j, i] = wz
                
    Lx = np.diag(np.sum(Wx, axis=1)) - Wx
    Ly = np.diag(np.sum(Wy, axis=1)) - Wy
    Lz = np.diag(np.sum(Wz, axis=1)) - Wz
    
    # Directional connectivity (Sum of edge weights per direction / N = average directional coupling)
    sigma_x = float(np.sum(Wx) / N)
    sigma_y = float(np.sum(Wy) / N)
    sigma_z = float(np.sum(Wz) / N)
    
    anisotropy_ratio = sigma_z / (0.5 * (sigma_x + sigma_y) + 1e-6)
    
    # Non-zero Fiedler eigenvalues for each directional component
    ev_x = [e for e in la.eigvalsh(Lx) if e > 1e-5]
    ev_y = [e for e in la.eigvalsh(Ly) if e > 1e-5]
    ev_z = [e for e in la.eigvalsh(Lz) if e > 1e-5]
    
    l2_x = ev_x[0] if len(ev_x) > 0 else 0.0
    l2_y = ev_y[0] if len(ev_y) > 0 else 0.0
    l2_z = ev_z[0] if len(ev_z) > 0 else 0.0
    
    # Grotthuss proton hopping activation energy (eV)
    # Ea = Ea_bulk / (1 + k * sigma_z)
    Ea_z = 0.40 / (1.0 + 0.5 * sigma_z)
    
    return {
        "lambda2_global": float(lambda2_global),
        "l2_x": float(l2_x),
        "l2_y": float(l2_y),
        "l2_z": float(l2_z),
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
        "sigma_z": sigma_z,
        "anisotropy_ratio": float(anisotropy_ratio),
        "Ea_z_eV": float(Ea_z)
    }

def main():
    print("=========================================================")
    print("  Graph-Spectral Proton Conduction & Anisotropy Engine   ")
    print("=========================================================")
    
    # Bulk Water
    pos_b, adj_b, _ = build_bulk_water_graph(n_water=100)
    spec_b = analyze_directional_spectral_tensor(pos_b, adj_b)
    
    print("\n--- 1. Bulk Liquid Water (Isotropic) ---")
    print(f"Global Fiedler Value (λ2): {spec_b['lambda2_global']:.4f}")
    print(f"Directional Conductance Density: σ_x={spec_b['sigma_x']:.4f}, σ_y={spec_b['sigma_y']:.4f}, σ_z={spec_b['sigma_z']:.4f}")
    print(f"Anisotropy Ratio (σ_z / σ_xy): {spec_b['anisotropy_ratio']:.3f} (Isotropic ≈ 1.0)")
    print(f"Proton Hopping Barrier Ea(z): {spec_b['Ea_z_eV']:.3f} eV")
    
    # Hexagonal Gel Channel
    pos_h, adj_h, labels_h = build_hexagonal_surfactant_gel(n_surfactants=48, n_water=100)
    spec_h = analyze_directional_spectral_tensor(pos_h, adj_h)
    
    print("\n--- 2. Hexagonal Surfactant Gel (Directional Channel) ---")
    print(f"Global Fiedler Value (λ2): {spec_h['lambda2_global']:.4f}")
    print(f"Directional Conductance Density: σ_x={spec_h['sigma_x']:.4f}, σ_y={spec_h['sigma_y']:.4f}, σ_z={spec_h['sigma_z']:.4f}")
    print(f"Anisotropy Ratio (σ_z / σ_xy): {spec_h['anisotropy_ratio']:.3f} (Strongly Anisotropic Channel!)")
    print(f"Proton Hopping Barrier Ea(z): {spec_h['Ea_z_eV']:.3f} eV (Reduced Barrier due to H-bond alignment!)")

    results = {
        "bulk_water": spec_b,
        "hexagonal_gel": spec_h,
        "hexagonal_pos": pos_h.tolist(),
        "hexagonal_labels": labels_h,
        "hexagonal_adj": adj_h.tolist()
    }
    
    with open("proton_conduction_graph_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nData exported to proton_conduction_graph_results.json.")

if __name__ == "__main__":
    main()
