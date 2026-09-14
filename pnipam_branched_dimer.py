import numpy as np
import networkx as nx
import os
import json
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem

# --- SINGLE CHAIN PNIPAM AGENT ---
class PNIPAMAgent:
    def __init__(self, n_monomers=15):
        self.n_monomers = n_monomers
        self.smiles = "CC(C)NC(=O)C" + "C(C(=O)NC(C)C)C" * (n_monomers - 1) + "C"
        
        self.mol = Chem.MolFromSmiles(self.smiles)
        self.mol = Chem.AddHs(self.mol)
        
        params = AllChem.ETKDGv3()
        params.useRandomCoords = True
        AllChem.EmbedMolecule(self.mol, params)
        AllChem.MMFFOptimizeMolecule(self.mol)
        
        self.n_atoms = self.mol.GetNumAtoms()
        conf = self.mol.GetConformer()
        self.initial_coords = np.array([list(conf.GetAtomPosition(i)) for i in range(self.n_atoms)])
        
        self.types = [atom.GetSymbol() for atom in self.mol.GetAtoms()]
        self.is_polar_acc = np.zeros(self.n_atoms, dtype=bool)
        self.is_polar_don_h = np.zeros(self.n_atoms, dtype=bool)
        self.is_hydro_c = np.zeros(self.n_atoms, dtype=bool)
        
        self.bonds = []
        for bond in self.mol.GetBonds():
            self.bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            
        for idx in range(self.n_atoms):
            atom = self.mol.GetAtomWithIdx(idx)
            sym = atom.GetSymbol()
            if sym == "O":
                for nbr in atom.GetNeighbors():
                    if nbr.GetSymbol() == "C":
                        self.is_polar_acc[idx] = True
            elif sym == "H":
                for nbr in atom.GetNeighbors():
                    if nbr.GetSymbol() == "N":
                        self.is_polar_don_h[idx] = True
            elif sym == "C":
                n_c_nbrs = len([nbr for nbr in atom.GetNeighbors() if nbr.GetSymbol() == "C"])
                n_h_nbrs = len([nbr for nbr in atom.GetNeighbors() if nbr.GetSymbol() == "H"])
                if n_c_nbrs == 1 and n_h_nbrs == 3:
                    self.is_hydro_c[idx] = True

# --- H-SHAPED BRANCHED DIMER SIMULATION ENGINE ---
class PNIPAMBranchedDimerSimulation:
    def __init__(self, n_monomers=15, n_water=150):
        self.agentA = PNIPAMAgent(n_monomers=n_monomers)
        self.agentB = PNIPAMAgent(n_monomers=n_monomers)
        self.n_monomers = n_monomers
        self.n_water = n_water
        
        # Identify the middle Nitrogen on each chain (index 7 for 15-mer)
        nitrogensA = [atom.GetIdx() for atom in self.agentA.mol.GetAtoms() if atom.GetSymbol() == "N"]
        self.N_crossA = nitrogensA[7]
        
        nitrogensB = [atom.GetIdx() for atom in self.agentB.mol.GetAtoms() if atom.GetSymbol() == "N"]
        self.N_crossB = nitrogensB[7]
        
        # Find isopropyl group to remove for Chain A N_crossA
        C_ipA = None
        for nbr in self.agentA.mol.GetAtomWithIdx(self.N_crossA).GetNeighbors():
            if nbr.GetSymbol() == "C":
                has_O_nbr = any(n.GetSymbol() == "O" for n in nbr.GetNeighbors())
                if not has_O_nbr:
                    C_ipA = nbr.GetIdx()
                    break
        to_removeA = set()
        queue = [C_ipA]
        to_removeA.add(C_ipA)
        visited = {self.N_crossA, C_ipA}
        while queue:
            curr = queue.pop(0)
            for nbr in self.agentA.mol.GetAtomWithIdx(curr).GetNeighbors():
                nbr_idx = nbr.GetIdx()
                if nbr_idx not in visited:
                    visited.add(nbr_idx)
                    to_removeA.add(nbr_idx)
                    queue.append(nbr_idx)
                    
        # Find isopropyl group to remove for Chain B N_crossB
        C_ipB = None
        for nbr in self.agentB.mol.GetAtomWithIdx(self.N_crossB).GetNeighbors():
            if nbr.GetSymbol() == "C":
                has_O_nbr = any(n.GetSymbol() == "O" for n in nbr.GetNeighbors())
                if not has_O_nbr:
                    C_ipB = nbr.GetIdx()
                    break
        to_removeB = set()
        queue = [C_ipB]
        to_removeB.add(C_ipB)
        visited = {self.N_crossB, C_ipB}
        while queue:
            curr = queue.pop(0)
            for nbr in self.agentB.mol.GetAtomWithIdx(curr).GetNeighbors():
                nbr_idx = nbr.GetIdx()
                if nbr_idx not in visited:
                    visited.add(nbr_idx)
                    to_removeB.add(nbr_idx)
                    queue.append(nbr_idx)
                    
        # Filter and map Chain A
        self.mapA = {}
        idx_new = 0
        for i in range(self.agentA.n_atoms):
            if i in to_removeA:
                continue
            self.mapA[i] = idx_new
            idx_new += 1
        self.n_poly1 = len(self.mapA)
        
        # Filter and map Chain B
        self.mapB = {}
        for i in range(self.agentB.n_atoms):
            if i in to_removeB:
                continue
            self.mapB[i] = idx_new
            idx_new += 1
        self.n_poly2 = len(self.mapB)
        
        self.n_poly = self.n_poly1 + self.n_poly2 + 1 # Chain A + Chain B + 1 MBAA Linker Carbon
        self.n_total = self.n_poly + self.n_water
        
        # Coordinates initialization from natural coiled conformers
        coordsA = self.agentA.initial_coords.copy()
        coordsA -= np.mean(coordsA, axis=0)
        coordsA += np.array([-3.5, 0.0, 0.0]) # slightly offset chains
        coordsA_filtered = np.delete(coordsA, list(to_removeA), axis=0)
        
        coordsB = self.agentB.initial_coords.copy()
        coordsB -= np.mean(coordsB, axis=0)
        coordsB += np.array([3.5, 0.0, 0.0])
        coordsB_filtered = np.delete(coordsB, list(to_removeB), axis=0)
        
        # Linker coordinate (midpoint of the two Nitrogen crosslinks)
        N_cA_coords = coordsA_filtered[self.mapA[self.N_crossA]]
        N_cB_coords = coordsB_filtered[self.mapB[self.N_crossB] - self.n_poly1]
        coords_linker = (N_cA_coords + N_cB_coords) / 2.0
        coords_linker = coords_linker.reshape(1, 3)
        
        # Add water coordinates surrounding the polymer H-dimer
        if n_water > 0:
            water_coords = []
            poly_coords = np.vstack([coordsA_filtered, coordsB_filtered, coords_linker])
            while len(water_coords) < n_water:
                candidate = np.random.uniform(-16.0, 16.0, 3)
                if np.all(np.linalg.norm(poly_coords - candidate, axis=1) > 2.5):
                    if len(water_coords) == 0 or np.all(np.linalg.norm(np.array(water_coords) - candidate, axis=1) > 2.0):
                        water_coords.append(candidate)
            water_coords = np.array(water_coords)
        else:
            water_coords = np.zeros((0, 3))
            
        self.coords = np.vstack([coordsA_filtered, coordsB_filtered, coords_linker, water_coords])
        
        # Properties
        typesA = [self.agentA.types[i] for i in range(self.agentA.n_atoms) if i not in to_removeA]
        typesB = [self.agentB.types[i] for i in range(self.agentB.n_atoms) if i not in to_removeB]
        self.types = typesA + typesB + ["C"] + ["W"] * n_water
        
        pa_A = [self.agentA.is_polar_acc[i] for i in range(self.agentA.n_atoms) if i not in to_removeA]
        pa_B = [self.agentB.is_polar_acc[i] for i in range(self.agentB.n_atoms) if i not in to_removeB]
        self.is_polar_acc = np.concatenate([pa_A, pa_B, [False]])
        
        pd_H_A = [self.agentA.is_polar_don_h[i] for i in range(self.agentA.n_atoms) if i not in to_removeA]
        pd_H_B = [self.agentB.is_polar_don_h[i] for i in range(self.agentB.n_atoms) if i not in to_removeB]
        self.is_polar_don_h = np.concatenate([pd_H_A, pd_H_B, [False]])
        
        hc_A = [self.agentA.is_hydro_c[i] for i in range(self.agentA.n_atoms) if i not in to_removeA]
        hc_B = [self.agentB.is_hydro_c[i] for i in range(self.agentB.n_atoms) if i not in to_removeB]
        self.is_hydro_c = np.concatenate([hc_A, hc_B, [True]])
        
        # Bonds list
        bondsA = []
        for u, v in self.agentA.bonds:
            if u not in to_removeA and v not in to_removeA:
                bondsA.append((self.mapA[u], self.mapA[v]))
        bondsB = []
        for u, v in self.agentB.bonds:
            if u not in to_removeB and v not in to_removeB:
                bondsB.append((self.mapB[u], self.mapB[v]))
                
        # Linker index
        idx_linker = self.n_poly - 1
        # H-shaped dimer bonds: Chain A bonds + Chain B bonds + MBAA crosslinks
        self.bonds = bondsA + bondsB + [
            (self.mapA[self.N_crossA], idx_linker),
            (self.mapB[self.N_crossB], idx_linker)
        ]
        
        # Compute graph distances
        adj = np.zeros((self.n_poly, self.n_poly))
        for u, v in self.bonds:
            adj[u, v] = 1
            adj[v, u] = 1
        G = nx.from_numpy_array(adj)
        self.graph_dist = dict(nx.all_pairs_shortest_path_length(G))
        self.graph_dist = np.array([[self.graph_dist[i].get(p, 999) for p in range(self.n_poly)] for i in range(self.n_poly)])
        
        # Run brief relaxation to normalize covalent crosslink lengths
        self.coords = self.relax_covalent_conformation(self.coords)
        
    def relax_covalent_conformation(self, coords):
        relaxed = coords.copy()
        k_b = 85.0
        d_cov = 1.5
        for _ in range(250):
            grad = np.zeros_like(relaxed)
            # Covalent bonds
            for u, v in self.bonds:
                diff = relaxed[u] - relaxed[v]
                d = np.linalg.norm(diff)
                if d > 0.01:
                    g = k_b * (d - d_cov) * (diff / d)
                    grad[u] += g
                    grad[v] -= g
            # Steric repulsion
            diff_all = relaxed[:self.n_poly, np.newaxis, :] - relaxed[np.newaxis, :self.n_poly, :]
            d_all = np.linalg.norm(diff_all, axis=-1)
            for i in range(self.n_poly):
                for p in range(i + 1, self.n_poly):
                    if self.graph_dist[i, p] >= 3 and d_all[i, p] < 2.4:
                        diff = relaxed[i] - relaxed[p]
                        d = d_all[i, p]
                        if d < 0.1: d = 0.1
                        g = -80.0 * (d - 0.2)**7 / (((d - 0.2)**8 + 0.01)**2) * (diff / d)
                        grad[i] += g
                        grad[p] -= g
            grad_norms = np.linalg.norm(grad, axis=-1, keepdims=True)
            grad = np.where(grad_norms > 15.0, grad / (grad_norms + 1e-6) * 15.0, grad)
            relaxed[:self.n_poly] -= 0.01 * grad[:self.n_poly]
        return relaxed

    def compute_energy_and_forces(self, coords, scale_hydro):
        n_poly = self.n_poly
        n_total = self.n_total
        grad = np.zeros_like(coords)
        total_energy = 0.0
        
        # 1. Covalent bonds
        k_b = 85.0
        d_cov = 1.5
        for u, v in self.bonds:
            diff = coords[u] - coords[v]
            d = np.linalg.norm(diff)
            if d > 0.01:
                total_energy += 0.5 * k_b * (d - d_cov)**2
                g = k_b * (d - d_cov) * (diff / d)
                grad[u] += g
                grad[v] -= g
                
        # 2. Steric overlap repulsion
        for i in range(n_total):
            for p in range(i + 1, n_total):
                if i < n_poly and p < n_poly and self.graph_dist[i, p] < 3:
                    continue
                diff = coords[i] - coords[p]
                d = np.linalg.norm(diff)
                if d < 0.1: d = 0.1
                
                thresh = 1.8 if (self.types[i] == "H" or self.types[p] == "H") else 2.6
                if d < thresh:
                    e = 40.0 * (d - thresh)**4
                    total_energy += e
                    g = 160.0 * (d - thresh)**3 * (diff / d)
                    grad[i] += g
                    grad[p] -= g
                    
        # 3. Amide-Water Hydrogen Bonding
        hb_strength = 20.0 * (1.0 - 0.4 * scale_hydro)
        for i in range(n_poly):
            is_acc = self.is_polar_acc[i]
            is_don = self.is_polar_don_h[i]
            if not (is_acc or is_don):
                continue
            for p in range(n_poly, n_total):
                diff = coords[i] - coords[p]
                d = np.linalg.norm(diff)
                if d < 4.0:
                    e = -hb_strength / (d**2 + 0.1)
                    total_energy += e
                    g = hb_strength * (2.0 * d / (d**2 + 0.1)**2) * (diff / d)
                    grad[i] += g
                    grad[p] -= g
                    
        # 4. Isopropyl-Isopropyl Hydrophobic Attraction
        hydro_strength = 35.0 * scale_hydro
        if hydro_strength > 0.01:
            for i in range(n_poly):
                if not self.is_hydro_c[i]: continue
                for p in range(i + 1, n_poly):
                    if not self.is_hydro_c[p]: continue
                    if self.graph_dist[i, p] < 4: continue
                    diff = coords[i] - coords[p]
                    d = np.linalg.norm(diff)
                    if d < 5.0:
                        e = -hydro_strength / (d**2 + 0.1)
                        total_energy += e
                        g = hydro_strength * (2.0 * d / (d**2 + 0.1)**2) * (diff / d)
                        grad[i] += g
                        grad[p] -= g
                        
        # 5. Fiedler Spectral Folding Force
        if scale_hydro > 0.05:
            active_contacts = []
            for i in range(n_poly):
                if not self.is_hydro_c[i]: continue
                for p in range(i + 1, n_poly):
                    if not self.is_hydro_c[p]: continue
                    if self.graph_dist[i, p] < 4: continue
                    d = np.linalg.norm(coords[i] - coords[p])
                    if d < 5.0:
                        w = scale_hydro * (30.0 / (d**2 + 0.1))
                        if w > 0.05:
                            active_contacts.append((i, p, w))
                            
            if active_contacts:
                G = nx.Graph()
                G.add_nodes_from(range(n_poly))
                for u, v in self.bonds:
                    G.add_edge(u, v, weight=10.0)
                for u, v, w in active_contacts:
                    G.add_edge(u, v, weight=w)
                    
                l2 = nx.algebraic_connectivity(G, method='lanczos', tol=1e-5)
                if l2 > 1e-5:
                    L_matrix = nx.laplacian_matrix(G).toarray()
                    eigvals, eigvecs = np.linalg.eigh(L_matrix)
                    v2 = eigvecs[:, 1]
                    
                    k_fiedler = 8.0 * scale_hydro
                    for u, v, w in active_contacts:
                        dv_sq = (v2[u] - v2[v])**2
                        diff = coords[u] - coords[v]
                        d = np.linalg.norm(diff)
                        if d > 0.1:
                            dw_dd = -2.0 * w / d
                            f_val = k_fiedler * dv_sq * dw_dd * (diff / d)
                            grad[u] += f_val
                            grad[v] -= f_val
                            
        # 6. Boundary wall to keep water inside box L = 26 A
        box_limit = 13.0
        for i in range(n_total):
            for axis in range(3):
                val = coords[i, axis]
                if val > box_limit:
                    grad[i, axis] += 40.0 * (val - box_limit)
                elif val < -box_limit:
                    grad[i, axis] += 40.0 * (val + box_limit)
                    
        return total_energy, grad

    def run_relaxation(self, scale_hydro, steps=300, lr=0.008):
        coords = self.coords.copy()
        for step in range(steps):
            energy, grad = self.compute_energy_and_forces(coords, scale_hydro)
            grad_norms = np.linalg.norm(grad, axis=-1, keepdims=True)
            grad = np.where(grad_norms > 12.0, grad / (grad_norms + 1e-6) * 12.0, grad)
            coords -= lr * grad
        return coords

# --- RUNNING THE DEHYDRATION SWEEP ---
def run_branched_dimer_sweep():
    print("==================================================")
    print("  H-SHAPED BRANCHED DIMER DEHYDRATION SWEEP (MBAA)")
    print("  Monomer 7 Single-Crosslinked Dimer")
    print("==================================================")
    
    water_scenarios = [180, 120, 60, 30, 10, 0]
    results = {}
    sweep_history = []
    
    coords_wet = None
    coords_dry = None
    
    for n_water in water_scenarios:
        sim = PNIPAMBranchedDimerSimulation(n_monomers=15, n_water=n_water)
        scale_hydro = 1.0 - (n_water / 180.0)
        
        print(f"\n Running Simulation with {n_water} Water Molecules (scale_hydro = {scale_hydro:.3f})...")
        final_coords = sim.run_relaxation(scale_hydro, steps=300)
        
        # Analyze results
        poly_coords = final_coords[:sim.n_poly]
        center = np.mean(poly_coords, axis=0)
        rg = np.sqrt(np.mean(np.sum((poly_coords - center)**2, axis=-1)))
        
        # Separation between chain centers
        poly1_coords = poly_coords[:sim.n_poly1]
        poly2_coords = poly_coords[sim.n_poly1:sim.n_poly1 + sim.n_poly2]
        com1 = np.mean(poly1_coords, axis=0)
        com2 = np.mean(poly2_coords, axis=0)
        separation = np.linalg.norm(com1 - com2)
        
        # Amide-Water contacts
        hb_count = 0
        for i in range(sim.n_poly):
            is_acc = sim.is_polar_acc[i]
            is_don = sim.is_polar_don_h[i]
            if not (is_acc or is_don): continue
            for p in range(sim.n_poly, sim.n_total):
                d = np.linalg.norm(final_coords[i] - final_coords[p])
                if d < 3.5:
                    hb_count += 1
                    
        # Inter-chain hydrophobic contacts
        hydro_count = 0
        for i in range(sim.n_poly1):
            if not sim.is_hydro_c[i]: continue
            for p in range(sim.n_poly1, sim.n_poly1 + sim.n_poly2):
                if not sim.is_hydro_c[p]: continue
                d = np.linalg.norm(final_coords[i] - final_coords[p])
                if d < 4.5:
                    hydro_count += 1
                    
        # Fiedler algebraic connectivity
        G = nx.Graph()
        G.add_nodes_from(range(sim.n_poly))
        for u, v in sim.bonds:
            G.add_edge(u, v, weight=10.0)
        for i in range(sim.n_poly):
            if not sim.is_hydro_c[i]: continue
            for p in range(i + 1, sim.n_poly):
                if not sim.is_hydro_c[p]: continue
                if sim.graph_dist[i, p] < 4: continue
                d = np.linalg.norm(final_coords[i] - final_coords[p])
                if d < 5.0:
                    w = scale_hydro * (30.0 / (d**2 + 0.1))
                    if w > 0.05:
                        G.add_edge(i, p, weight=w)
        l2 = nx.algebraic_connectivity(G, method='lanczos', tol=1e-5)
        
        print(f"   * Final Rg: {rg:.3f} Å | Separation: {separation:.3f} Å")
        print(f"   * Fiedler Connectivity (λ2): {l2:.5f}")
        print(f"   * Amide-Water Contacts: {hb_count}")
        print(f"   * Inter-chain Hydrophobic Contacts: {hydro_count}")
        
        results[n_water] = {
            "n_water": n_water,
            "scale_hydro": scale_hydro,
            "rg": rg,
            "separation": separation,
            "l2": l2,
            "hbonds": hb_count,
            "hydro": hydro_count,
            "coords": final_coords.tolist()
        }
        sweep_history.append((n_water, rg, separation, l2, hb_count, hydro_count))
        
        if n_water == 180:
            coords_wet = final_coords.copy()
            sim_wet = sim
        if n_water == 0:
            coords_dry = final_coords.copy()
            sim_dry = sim
            
    # --- VISUALIZATION ---
    plot_dir = "/home/eldenring/.gemini/antigravity/brain/d1e57b48-f8cc-4dd3-9357-bbffb4c85cbd"
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. 3D Side-by-Side conformation plot
    fig_3d = plt.figure(figsize=(14, 7), facecolor='#111111')
    
    def plot_3d_state(coords, sim, title, subplot_idx):
        ax = fig_3d.add_subplot(1, 2, subplot_idx, projection='3d')
        ax.set_facecolor('#111111')
        
        # Plot water
        if sim.n_water > 0:
            w_coords = coords[sim.n_poly:]
            ax.scatter(w_coords[:, 0], w_coords[:, 1], w_coords[:, 2], 
                       color='#00e1ff', alpha=0.15, s=25, label='Water Solvent')
                       
        # Plot polymer
        poly_coords = coords[:sim.n_poly]
        
        # Colors: Orange for Chain A, Magenta for Chain B, Neon Green for MBAA linker
        bb_idxA = [i for i in range(sim.n_poly1) if sim.types[i] == "C" and not sim.is_hydro_c[i]]
        am_idxA = [i for i in range(sim.n_poly1) if sim.types[i] == "O" or sim.types[i] == "N"]
        ip_idxA = [i for i in range(sim.n_poly1) if sim.is_hydro_c[i]]
        
        bb_idxB = [i + sim.n_poly1 for i in range(sim.n_poly2) if sim.types[i + sim.n_poly1] == "C" and not sim.is_hydro_c[i + sim.n_poly1]]
        am_idxB = [i + sim.n_poly1 for i in range(sim.n_poly2) if sim.types[i + sim.n_poly1] == "O" or sim.types[i + sim.n_poly1] == "N"]
        ip_idxB = [i + sim.n_poly1 for i in range(sim.n_poly2) if sim.is_hydro_c[i + sim.n_poly1]]
        
        idx_linker = sim.n_poly - 1
        
        # Scatter chain A
        ax.scatter(poly_coords[bb_idxA, 0], poly_coords[bb_idxA, 1], poly_coords[bb_idxA, 2], 
                   color='#ff7f0e', s=50, label='Chain A Backbone C')
        ax.scatter(poly_coords[am_idxA, 0], poly_coords[am_idxA, 1], poly_coords[am_idxA, 2], 
                   color='#00ffd5', s=60, label='Chain A Amide N/O')
        ax.scatter(poly_coords[ip_idxA, 0], poly_coords[ip_idxA, 1], poly_coords[ip_idxA, 2], 
                   color='#d500ff', s=70, label='Chain A Isopropyl C')
                   
        # Scatter chain B
        ax.scatter(poly_coords[bb_idxB, 0], poly_coords[bb_idxB, 1], poly_coords[bb_idxB, 2], 
                   color='#ff00ff', s=50, label='Chain B Backbone C')
        ax.scatter(poly_coords[am_idxB, 0], poly_coords[am_idxB, 1], poly_coords[am_idxB, 2], 
                   color='#00ffd5', s=60)
        ax.scatter(poly_coords[ip_idxB, 0], poly_coords[ip_idxB, 1], poly_coords[ip_idxB, 2], 
                   color='#d500ff', s=70)
                   
        # Scatter Linker
        ax.scatter(poly_coords[idx_linker, 0], poly_coords[idx_linker, 1], poly_coords[idx_linker, 2], 
                   color='#00ff00', s=130, label='MBAA Linker C', edgecolors='#ffffff')
                   
        # Draw bonds
        for u, v in sim.bonds:
            b_color = '#00ff00' if (u == idx_linker or v == idx_linker) else '#555555'
            ax.plot([coords[u, 0], coords[v, 0]], 
                    [coords[u, 1], coords[v, 1]], 
                    [coords[u, 2], coords[v, 2]], color=b_color, linewidth=2.0 if b_color=='#00ff00' else 1.5)
                    
        ax.set_title(title, color='#e5e5ed', fontsize=14, fontweight='bold', pad=10)
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        lim = 16.0
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        
    plot_3d_state(coords_wet, sim_wet, "Wet Swollen State (N_Water = 180)\nBranched H-Conformer (Rg = {:.3f} Å)".format(sweep_history[0][1]), 1)
    plot_3d_state(coords_dry, sim_dry, "Dry Collapsed State (N_Water = 0)\nCompact H-Globule (Rg = {:.3f} Å)".format(sweep_history[-1][1]), 2)
    
    handles, labels = fig_3d.axes[0].get_legend_handles_labels()
    fig_3d.legend(handles, labels, loc='lower center', ncol=4, facecolor='#222222', edgecolor='none', labelcolor='#e5e5ed')
    plt.suptitle("H-Shaped Single-Crosslinked PNIPAM Dimer: Wet vs Dry States", color='#e5e5ed', fontsize=16, fontweight='bold', y=0.96)
    plt.tight_layout(rect=[0, 0.08, 1, 0.94])
    
    structure_path = os.path.join(plot_dir, "pnipam_branched_dimer_structure.png")
    plt.savefig(structure_path, dpi=150, facecolor='#111111', edgecolor='none')
    plt.close()
    
    # 2. Transition curves plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='#111111')
    ax1.set_facecolor('#111111')
    ax2.set_facecolor('#111111')
    
    n_waters = [item[0] for item in sweep_history]
    rgs = [item[1] for item in sweep_history]
    seps = [item[2] for item in sweep_history]
    l2s = [item[3] for item in sweep_history]
    
    ax1.plot(n_waters, rgs, color='#d500ff', marker='o', linewidth=2.5, label='Radius of Gyration (Rg)')
    ax1.plot(n_waters, seps, color='#ff7f0e', marker='^', linewidth=2.0, linestyle='--', label='Chain Separation')
    ax1.invert_xaxis()
    ax1.set_xlabel("Water Molecule Count (N_Water)", color='#e5e5ed', fontsize=12)
    ax1.set_ylabel("Distance / Size (Å)", color='#e5e5ed', fontsize=12)
    ax1.set_title("H-Dimer Contraction & Chain Proximity", color='#e5e5ed', fontsize=13, fontweight='bold')
    ax1.tick_params(colors='#e5e5ed')
    ax1.grid(color='#333333', linestyle='--')
    ax1.legend(facecolor='#222222', edgecolor='none', labelcolor='#e5e5ed')
    
    ax2.plot(n_waters, l2s, color='#00ff00', marker='s', linewidth=2.5, label='Fiedler Connectivity (λ2)')
    ax2.invert_xaxis()
    ax2.set_xlabel("Water Molecule Count (N_Water)", color='#e5e5ed', fontsize=12)
    ax2.set_ylabel("Fiedler Connectivity (λ2)", color='#e5e5ed', fontsize=12)
    ax2.set_title("Topological Contact Network Connectivity", color='#e5e5ed', fontsize=13, fontweight='bold')
    ax2.tick_params(colors='#e5e5ed')
    ax2.grid(color='#333333', linestyle='--')
    ax2.legend(facecolor='#222222', edgecolor='none', labelcolor='#e5e5ed')
    
    plt.suptitle("MBAA Middle-Crosslinked H-Dimer Dehydration Sweep", color='#e5e5ed', fontsize=16, fontweight='bold', y=0.97)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    transition_path = os.path.join(plot_dir, "pnipam_branched_dimer_transition.png")
    plt.savefig(transition_path, dpi=150, facecolor='#111111', edgecolor='none')
    plt.close()
    
    # Save results to JSON
    json_path = "pnipam_branched_dimer_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nSaved transition plots to:\n - {structure_path}\n - {transition_path}")
    print(f"Saved results database to {json_path}")
    print("==================================================")

if __name__ == "__main__":
    run_branched_dimer_sweep()
