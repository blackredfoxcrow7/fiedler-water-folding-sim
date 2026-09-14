import numpy as np
import networkx as nx
import json
import os
import matplotlib.pyplot as plt

# --- COARSE-GRAINED NIPA GEL MODEL ---
class NIPAGelSimulator:
    def __init__(self, n_side=4):
        self.n_side = n_side
        self.n_beads = n_side ** 3
        self.initial_coords = self.initialize_lattice()
        self.backbone_bonds = self.generate_crosslinks()
        
    def initialize_lattice(self):
        # 4x4x4 cubic lattice with minor random perturbations
        coords = []
        for x in range(self.n_side):
            for y in range(self.n_side):
                for z in range(self.n_side):
                    # Lattice spacing = 3.0 Angstroms
                    px = x * 3.0 + np.random.normal(0, 0.1)
                    py = y * 3.0 + np.random.normal(0, 0.1)
                    pz = z * 3.0 + np.random.normal(0, 0.1)
                    coords.append([px, py, pz])
        return np.array(coords)

    def generate_crosslinks(self):
        # Link nearest neighbors in the lattice (backbone bonds)
        bonds = []
        for i in range(self.n_beads):
            x1, y1, z1 = self.index_to_grid(i)
            for j in range(i + 1, self.n_beads):
                x2, y2, z2 = self.index_to_grid(j)
                dist = abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2)
                if dist == 1: # Nearest neighbor
                    bonds.append((i, j))
        return bonds

    def index_to_grid(self, idx):
        z = idx % self.n_side
        y = (idx // self.n_side) % self.n_side
        x = idx // (self.n_side ** 2)
        return x, y, z

    def compute_energy_and_gradients(self, coords, temp_c):
        n_beads = self.n_beads
        grad = np.zeros_like(coords)
        total_energy = 0.0
        
        # Temperature-dependent parameters (LCST = 32 C)
        # Sigmoidal response curves
        alpha = 1.0 / (1.0 + np.exp(-(temp_c - 32.0) / 1.5))  # Hydrophobic attraction coefficient
        beta = 1.0 - alpha                                  # Swelling repulsion coefficient
        
        # 1. Covalent backbone springs
        k_b = 20.0
        d_0 = 3.0
        for i, p in self.backbone_bonds:
            diff = coords[i] - coords[p]
            d = np.linalg.norm(diff)
            if d > 0.01:
                e = 0.5 * k_b * (d - d_0)**2
                total_energy += e
                g_val = k_b * (d - d_0) * (diff / d)
                grad[i] += g_val
                grad[p] -= g_val
                
        # 2. Non-covalent pairwise potentials (Hydrophobic attraction, swell repulsion, steric repulsion)
        for i in range(n_beads):
            for p in range(i + 1, n_beads):
                is_backbone = (i, p) in self.backbone_bonds or (p, i) in self.backbone_bonds
                diff = coords[i] - coords[p]
                d = np.linalg.norm(diff)
                if d < 0.1:
                    d = 0.1
                
                # Steric repulsion (universal)
                e_steric = 10.0 / (d**10 + 0.01)
                total_energy += e_steric
                g_steric = -100.0 / (d**11) * (diff / d)
                grad[i] += g_steric
                grad[p] -= g_steric
                
                if not is_backbone:
                    # Hydrophobic attraction (attracts more at higher temperature)
                    e_hydro = -alpha * (50.0 / (d**2 + 0.1))
                    total_energy += e_hydro
                    g_hydro = alpha * (100.0 * d / (d**2 + 0.1)**2) * (diff / d)
                    grad[i] += g_hydro
                    grad[p] -= g_hydro
                    
                    # Swelling hydration repulsion (swells more at lower temperature)
                    e_swell = beta * (15.0 / (d**4 + 0.1))
                    total_energy += e_swell
                    g_swell = -beta * (60.0 * (d**3) / (d**4 + 0.1)**2) * (diff / d)
                    grad[i] += g_swell
                    grad[p] -= g_swell
                    
        return total_energy, grad

    def run_relaxation(self, temp_c, steps=150, lr=0.01):
        coords = self.initial_coords.copy()
        
        for step in range(steps):
            energy, grad = self.compute_energy_and_gradients(coords, temp_c)
            
            # Gradient clipping per bead to prevent numerical explosion
            grad_norms = np.linalg.norm(grad, axis=-1, keepdims=True)
            grad = np.where(grad_norms > 10.0, grad / (grad_norms + 1e-6) * 10.0, grad)
            
            # Gradient descent step
            coords -= lr * grad
            
        return coords

def run_gel_phase_transition_sweep():
    sim = NIPAGelSimulator(n_side=4)
    temperatures = [20.0, 23.0, 26.0, 28.0, 30.0, 31.0, 32.0, 33.0, 34.0, 36.0, 39.0, 42.0, 45.0]
    
    results = []
    
    print("==================================================")
    print("  NIPA GEL VOLUME PHASE TRANSITION SIMULATOR")
    print("  Lower Critical Solution Temperature (LCST): 32 C")
    print("==================================================")
    print(f" {'Temp (C)':<10} | {'Radius of Gyration (Rg)':<25} | {'Fiedler Value (L2)':<20} | {'Contacts':<10}")
    print("--------------------------------------------------")
    
    for temp in temperatures:
        # Relax hydrogel coordinates at target temperature
        coords = sim.run_relaxation(temp, steps=150, lr=0.01)
        
        # Calculate Radius of Gyration (Rg) as a proxy for Volume
        center = np.mean(coords, axis=0)
        rg = np.sqrt(np.mean(np.sum((coords - center)**2, axis=-1)))
        
        # Build Contact Network (contact if distance < 4.0 A)
        G = nx.Graph()
        G.add_nodes_from(range(sim.n_beads))
        for i, p in sim.backbone_bonds:
            G.add_edge(i, p, weight=1.0)
            
        n_contacts = 0
        for i in range(sim.n_beads):
            for p in range(i + 1, sim.n_beads):
                if (i, p) not in sim.backbone_bonds:
                    d = np.linalg.norm(coords[i] - coords[p])
                    if d < 4.0:
                        G.add_edge(i, p, weight=0.5)
                        n_contacts += 1
                        
        # Compute Fiedler value
        l2 = nx.algebraic_connectivity(G, method='lanczos', tol=1e-5)
        
        print(f" {temp:<10.1f} | {rg:<25.4f} | {l2:<20.5e} | {n_contacts:<10}")
        
        results.append({
            "temp": temp,
            "rg": rg,
            "fiedler": l2,
            "contacts": n_contacts,
            "coords": coords.tolist()
        })
        
    # Save sweep results to JSON
    with open("nipa_gel_results.json", "w") as f:
        json.dump(results, f)
        
    # Plot transition curves
    art_dir = "/home/eldenring/.gemini/antigravity/brain/d1e57b48-f8cc-4dd3-9357-bbffb4c85cbd"
    os.makedirs(art_dir, exist_ok=True)
    
    temps = [r["temp"] for r in results]
    rgs = [r["rg"] for r in results]
    fiedlers = [r["fiedler"] for r in results]
    
    fig, ax1 = plt.subplots(figsize=(8, 5), facecolor='#111111')
    ax1.set_facecolor('#111111')
    
    # Rg curve (left axis)
    color = '#ff4d4d'
    ax1.set_xlabel('Temperature (°C)', color='#e5e5ed', fontsize=12)
    ax1.set_ylabel('Radius of Gyration Rg (Å) [Volume Proxy]', color=color, fontsize=11)
    line1 = ax1.plot(temps, rgs, color=color, marker='o', linewidth=2.5, label='Rg (Volume)')
    ax1.tick_params(axis='y', labelcolor=color, colors='#8e8e9f')
    ax1.tick_params(axis='x', colors='#8e8e9f')
    ax1.grid(color='#222222', linestyle='--')
    
    # Fiedler curve (right axis)
    ax2 = ax1.twinx()
    color2 = '#00ffd5'
    ax2.set_ylabel('Fiedler Connectivity Value (λ2)', color=color2, fontsize=11)
    line2 = ax2.plot(temps, fiedlers, color=color2, marker='s', linewidth=2.5, linestyle='--', label='Fiedler Value (λ2)')
    ax2.tick_params(axis='y', labelcolor=color2, colors='#8e8e9f')
    
    # LCST boundary line
    ax1.axvline(x=32.0, color='#8800ff', linestyle=':', linewidth=2.0)
    ax1.text(32.5, min(rgs) + 0.1, 'LCST (32°C)', color='#8800ff', fontsize=10, fontweight='bold')
    
    plt.title('NIPA Hydrogel Volume Phase Transition vs. Fiedler Spectral Connectivity', color='#e5e5ed', fontsize=13, fontweight='bold', pad=15)
    
    # Legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', facecolor='#222222', edgecolor='none', labelcolor='#e5e5ed')
    
    plt.tight_layout()
    plot_path = os.path.join(art_dir, "nipa_gel_transition.png")
    plt.savefig(plot_path, dpi=150, facecolor='#111111', edgecolor='none')
    plt.close()
    
    print("==================================================")
    print(f" Saved transition plot to {plot_path}")
    print(" Saved simulation data to nipa_gel_results.json")
    print("==================================================")

if __name__ == "__main__":
    run_gel_phase_transition_sweep()
