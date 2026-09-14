import numpy as np
import os
import json
import matplotlib.pyplot as plt

# --- DETAILED 3-BEAD PNIPAM + EXPLICIT WATER MODEL ---
class DetailedNIPASimulator:
    def __init__(self, n_chains=6, chain_len=4):
        self.n_chains = n_chains
        self.chain_len = chain_len
        self.n_monomers = n_chains * chain_len
        self.n_poly_beads = self.n_monomers * 3 # BB, AM, IP per monomer
        self.n_water = 200
        self.n_total_atoms = self.n_poly_beads + self.n_water
        
        # Atom types
        self.types = [] # "BB", "AM", "IP", "W"
        self.monomer_residues = []
        
        # Build Polymer Chains
        self.backbone_bonds = []
        self.sidechain_bonds = []
        
        coords = []
        
        # Initialize polymer beads
        bead_idx = 0
        for c in range(n_chains):
            # Chain origin
            ox = np.random.uniform(-4.0, 4.0)
            oy = np.random.uniform(-4.0, 4.0)
            oz = np.random.uniform(-8.0, 8.0)
            
            for m in range(chain_len):
                # Monomer index
                mon_idx = c * chain_len + m
                
                # BB bead
                coords.append([ox, oy, oz + m * 3.0])
                self.types.append("BB")
                self.monomer_residues.append(mon_idx)
                bb_idx = bead_idx
                bead_idx += 1
                
                # AM bead (offset from BB)
                coords.append([ox + 1.2, oy + 1.2, oz + m * 3.0])
                self.types.append("AM")
                self.monomer_residues.append(mon_idx)
                am_idx = bead_idx
                bead_idx += 1
                
                # IP bead (offset from AM)
                coords.append([ox + 2.4, oy + 2.4, oz + m * 3.0])
                self.types.append("IP")
                self.monomer_residues.append(mon_idx)
                ip_idx = bead_idx
                bead_idx += 1
                
                # Bonds
                self.sidechain_bonds.append((bb_idx, am_idx))
                self.sidechain_bonds.append((am_idx, ip_idx))
                
                if m > 0:
                    # Link backbones
                    prev_bb = bb_idx - 3
                    self.backbone_bonds.append((prev_bb, bb_idx))
                    
        # Crosslinks between chains to form a gel network
        self.crosslinks = []
        for c in range(n_chains - 1):
            # Link end of chain c to start of chain c+1
            end_bb = (c * chain_len + (chain_len - 1)) * 3
            start_bb = ((c + 1) * chain_len) * 3
            self.crosslinks.append((end_bb, start_bb))
            
        # Add Explicit Water beads (W) randomly in a 3D box of size L=20 A
        for w in range(self.n_water):
            coords.append([
                np.random.uniform(-10.0, 10.0),
                np.random.uniform(-10.0, 10.0),
                np.random.uniform(-10.0, 10.0)
            ])
            self.types.append("W")
            self.monomer_residues.append(-1)
            
        self.initial_coords = np.array(coords)
        
    def compute_energy_and_gradients(self, coords, temp_c):
        n_atoms = self.n_total_atoms
        grad = np.zeros_like(coords)
        total_energy = 0.0
        
        # Temperature responsive parameters
        # Hydrogen bond factor (strong at low temp, breaks above LCST)
        gamma_hb = 1.0 / (1.0 + np.exp((temp_c - 32.0) / 1.5))
        # Hydrophobic attraction factor (strong at high temp)
        gamma_hydro = 1.0 / (1.0 + np.exp(-(temp_c - 32.0) / 1.5))
        # Water cage hydration shell factor (water cage around IP at low temp)
        beta_cage = gamma_hb
        
        # 1. Covalent bonds (Backbone + Sidechains + Crosslinks)
        k_b = 30.0
        d_cov = 1.5
        all_covalent = self.backbone_bonds + self.sidechain_bonds + self.crosslinks
        for i, p in all_covalent:
            diff = coords[i] - coords[p]
            d = np.linalg.norm(diff)
            if d > 0.01:
                e = 0.5 * k_b * (d - d_cov)**2
                total_energy += e
                g_val = k_b * (d - d_cov) * (diff / d)
                grad[i] += g_val
                grad[p] -= g_val
                
        # 2. Non-covalent pairwise interactions
        for i in range(n_atoms):
            type_i = self.types[i]
            for p in range(i + 1, n_atoms):
                type_p = self.types[p]
                
                # Skip if covalently bonded
                if (i, p) in all_covalent or (p, i) in all_covalent:
                    continue
                    
                diff = coords[i] - coords[p]
                d = np.linalg.norm(diff)
                if d < 0.1:
                    d = 0.1
                    
                # Steric repulsion (universal)
                e_steric = 5.0 / (d**10 + 0.01)
                total_energy += e_steric
                g_steric = -50.0 / (d**11 + 1e-6) * (diff / d)
                grad[i] += g_steric
                grad[p] -= g_steric
                
                # --- Specific Non-Covalent Interactions ---
                
                # A. Amide-Water (AM-W) Hydrogen Bonding
                if (type_i == "AM" and type_p == "W") or (type_i == "W" and type_p == "AM"):
                    # Strong hydrophilic attraction below LCST
                    e_hb = -gamma_hb * (45.0 / (d**3 + 0.1))
                    total_energy += e_hb
                    g_hb = gamma_hb * (135.0 * d / (d**3 + 0.1)**2) * (diff / d)
                    grad[i] += g_hb
                    grad[p] -= g_hb
                    
                # B. Isopropyl-Isopropyl (IP-IP) Hydrophobic Attraction
                elif type_i == "IP" and type_p == "IP":
                    # Strong hydrophobic attraction above LCST
                    e_hydro = -gamma_hydro * (80.0 / (d**2 + 0.1))
                    total_energy += e_hydro
                    g_hydro = gamma_hydro * (160.0 * d / (d**2 + 0.1)**2) * (diff / d)
                    grad[i] += g_hydro
                    grad[p] -= g_hydro
                    
                # C. Isopropyl-Water (IP-W) Hydration Cage
                elif (type_i == "IP" and type_p == "W") or (type_i == "W" and type_p == "IP"):
                    # Low Temp: Water forms a structured cage at d ~ 3.5 A
                    # High Temp: Water is repelled (cage breaks, hydrophobic exclusion)
                    if temp_c < 32.0:
                        # Double well or simple harmonic potential centered at 3.5 A
                        e_cage = beta_cage * 15.0 * (d - 3.5)**2
                        total_energy += e_cage
                        g_cage = beta_cage * 30.0 * (d - 3.5) * (diff / d)
                        grad[i] += g_cage
                        grad[p] -= g_cage
                    else:
                        # Pure hydrophobic repulsion
                        e_rep = (1.0 - beta_cage) * (15.0 / (d**4 + 0.1))
                        total_energy += e_rep
                        g_rep = -(1.0 - beta_cage) * (60.0 * d**2 / (d**4 + 0.1)**2) * (diff / d)
                        grad[i] += g_rep
                        grad[p] -= g_rep
                        
        # 3. Boundary walls to keep water inside the simulation box (L = 22 A)
        box_limit = 11.0
        for i in range(n_atoms):
            for axis in range(3):
                val = coords[i, axis]
                if val > box_limit:
                    grad[i, axis] += 50.0 * (val - box_limit)
                elif val < -box_limit:
                    grad[i, axis] += 50.0 * (val + box_limit)
                    
        return total_energy, grad

    def run_relaxation(self, temp_c, steps=250, lr=0.008):
        coords = self.initial_coords.copy()
        
        for step in range(steps):
            energy, grad = self.compute_energy_and_gradients(coords, temp_c)
            
            # Gradient clipping per bead to prevent numerical explosions
            grad_norms = np.linalg.norm(grad, axis=-1, keepdims=True)
            grad = np.where(grad_norms > 12.0, grad / (grad_norms + 1e-6) * 12.0, grad)
            
            coords -= lr * grad
            
        return coords

def run_detailed_simulation():
    print("==================================================")
    print("  NIPA DETAILED 3-BEAD + EXPLICIT WATER SIMULATOR")
    print("==================================================")
    
    sim = DetailedNIPASimulator(n_chains=6, chain_len=4)
    
    # 1. Simulate below LCST (20 C)
    print(" Sweeping Temperature: 20.0 C (Below LCST - Hydrated & Swollen)")
    coords_cold = sim.run_relaxation(20.0, steps=250)
    
    # 2. Simulate above LCST (40 C)
    print(" Sweeping Temperature: 40.0 C (Above LCST - Dehydrated & Collapsed)")
    coords_hot = sim.run_relaxation(40.0, steps=250)
    
    # --- ANALYSIS OF HYDROGEN BONDS & HYDROPHOBIC CONTACTS ---
    def analyze_structure(coords, temp):
        n_atoms = sim.n_total_atoms
        hb_count = 0
        hydro_count = 0
        cage_count = 0
        
        for i in range(n_atoms):
            type_i = sim.types[i]
            for p in range(i + 1, n_atoms):
                type_p = sim.types[p]
                
                # Skip covalent
                all_cov = sim.backbone_bonds + sim.sidechain_bonds + sim.crosslinks
                if (i, p) in all_cov or (p, i) in all_cov:
                    continue
                    
                d = np.linalg.norm(coords[i] - coords[p])
                
                # Amide - Water Hydrogen Bonds (< 3.5 A)
                if (type_i == "AM" and type_p == "W") or (type_i == "W" and type_p == "AM"):
                    if d < 3.5:
                        hb_count += 1
                # Isopropyl - Isopropyl Hydrophobic Bonds (< 4.5 A)
                elif type_i == "IP" and type_p == "IP":
                    if d < 4.5:
                        hydro_count += 1
                # Isopropyl - Water Hydration Shell Contacts (3.0 - 4.2 A)
                elif (type_i == "IP" and type_p == "W") or (type_i == "W" and type_p == "IP"):
                    if 3.0 <= d <= 4.2:
                        cage_count += 1
                        
        # Radius of gyration of polymer network only
        poly_coords = coords[:sim.n_poly_beads]
        center = np.mean(poly_coords, axis=0)
        rg = np.sqrt(np.mean(np.sum((poly_coords - center)**2, axis=-1)))
        
        print(f"\n Results at {temp}°C:")
        print(f"   - Polymer Radius of Gyration (Rg): {rg:.3f} A")
        print(f"   - Amide-Water H-Bonds (AM-W): {hb_count} contacts")
        print(f"   - Isopropyl-Isopropyl Hydrophobic Bonds (IP-IP): {hydro_count} contacts")
        print(f"   - Isopropyl-Water Hydration Shell (IP-W): {cage_count} contacts")
        return rg, hb_count, hydro_count, cage_count

    rg_c, hb_c, hydro_c, cage_c = analyze_structure(coords_cold, 20.0)
    rg_h, hb_h, hydro_h, cage_h = analyze_structure(coords_hot, 40.0)
    
    # Save statistics
    detailed_data = {
        "cold": {
            "temp": 20.0,
            "rg": rg_c,
            "hbonds": hb_c,
            "hydro": hydro_c,
            "cage": cage_c,
            "coords": coords_cold.tolist()
        },
        "hot": {
            "temp": 40.0,
            "rg": rg_h,
            "hbonds": hb_h,
            "hydro": hydro_h,
            "cage": cage_h,
            "coords": coords_hot.tolist()
        },
        "types": sim.types
    }
    with open("nipa_detailed_results.json", "w") as f:
        json.dump(detailed_data, f)
        
    # --- GENERATE 3D SIDE-BY-SIDE PLOT ---
    art_dir = "/home/eldenring/.gemini/antigravity/brain/d1e57b48-f8cc-4dd3-9357-bbffb4c85cbd"
    os.makedirs(art_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(14, 7), facecolor='#111111')
    
    def plot_subset(coords, title, subplot_idx):
        ax = fig.add_subplot(1, 2, subplot_idx, projection='3d')
        ax.set_facecolor('#111111')
        
        # Plot Water beads (semi-transparent light blue)
        w_coords = coords[sim.n_poly_beads:]
        ax.scatter(w_coords[:, 0], w_coords[:, 1], w_coords[:, 2], 
                   color='#00e1ff', alpha=0.15, s=25, label='Water Solvent')
        
        # Plot Polymer beads
        poly_coords = coords[:sim.n_poly_beads]
        
        bb_idx = [i for i, t in enumerate(sim.types[:sim.n_poly_beads]) if t == "BB"]
        am_idx = [i for i, t in enumerate(sim.types[:sim.n_poly_beads]) if t == "AM"]
        ip_idx = [i for i, t in enumerate(sim.types[:sim.n_poly_beads]) if t == "IP"]
        
        # Plot Backbone (Grey)
        ax.scatter(poly_coords[bb_idx, 0], poly_coords[bb_idx, 1], poly_coords[bb_idx, 2], 
                   color='#8e8e9f', s=60, label='Backbone')
        
        # Plot Amide (Cyan)
        ax.scatter(poly_coords[am_idx, 0], poly_coords[am_idx, 1], poly_coords[am_idx, 2], 
                   color='#00ffd5', s=70, label='Amide (Hydrophilic)')
        
        # Plot Isopropyl (Purple)
        ax.scatter(poly_coords[ip_idx, 0], poly_coords[ip_idx, 1], poly_coords[ip_idx, 2], 
                   color='#d500ff', s=80, label='Isopropyl (Hydrophobic)')
        
        # Draw backbone bonds
        for bond in sim.backbone_bonds:
            i, p = bond
            ax.plot([coords[i, 0], coords[p, 0]], 
                    [coords[i, 1], coords[p, 1]], 
                    [coords[i, 2], coords[p, 2]], color='#555555', linewidth=2.0)
            
        # Draw sidechain bonds
        for bond in sim.sidechain_bonds:
            i, p = bond
            ax.plot([coords[i, 0], coords[p, 0]], 
                    [coords[i, 1], coords[p, 1]], 
                    [coords[i, 2], coords[p, 2]], color='#8e8e9f', linewidth=1.0, linestyle=':')
            
        ax.set_title(title, color='#e5e5ed', fontsize=14, fontweight='bold', pad=10)
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_xlim(-12, 12)
        ax.set_ylim(-12, 12)
        ax.set_zlim(-12, 12)
        
    plot_subset(coords_cold, "Swollen Hydrated State (20°C)\nHigh Polymer-Water H-Bonding", 1)
    plot_subset(coords_hot, "Collapsed Dehydrated State (40°C)\nHigh Isopropyl Hydrophobic Aggregation", 2)
    
    # Single Legend
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, facecolor='#222222', edgecolor='none', labelcolor='#e5e5ed')
    
    plt.suptitle("Molecular Volume Phase Transition of NIPA Hydrogel", color='#e5e5ed', fontsize=16, fontweight='bold', y=0.96)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    plot_path = os.path.join(art_dir, "nipa_gel_detailed_transition.png")
    plt.savefig(plot_path, dpi=150, facecolor='#111111', edgecolor='none')
    plt.close()
    
    print("==================================================")
    print(f" Saved molecular detailed plot to {plot_path}")
    print(" Saved detailed data to nipa_detailed_results.json")
    print("==================================================")

if __name__ == "__main__":
    run_detailed_simulation()
