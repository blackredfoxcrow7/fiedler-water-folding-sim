import numpy as np
import networkx as nx
import os
import json
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem

# --- PNIPAM MOLECULAR AGENT ---
class PNIPAMAgent:
    def __init__(self, n_monomers=15):
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
        
        # Identify specific atom groups
        self.types = [atom.GetSymbol() for atom in self.mol.GetAtoms()]
        self.is_polar_acc = np.zeros(self.n_atoms, dtype=bool) # Amide Oxygen
        self.is_polar_don_h = np.zeros(self.n_atoms, dtype=bool) # Amide Hydrogen
        self.is_hydro_c = np.zeros(self.n_atoms, dtype=bool) # Isopropyl Methyl Carbons
        
        # Parse bonds
        self.bonds = []
        for bond in self.mol.GetBonds():
            self.bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            
        # Map atoms
        for idx in range(self.n_atoms):
            atom = self.mol.GetAtomWithIdx(idx)
            sym = atom.GetSymbol()
            
            # Find Amide Oxygen
            if sym == "O":
                # Check if neighbor is carbonyl Carbon
                for nbr in atom.GetNeighbors():
                    if nbr.GetSymbol() == "C":
                        self.is_polar_acc[idx] = True
                        
            # Find Amide Hydrogen (bonded to N)
            if sym == "H":
                for nbr in atom.GetNeighbors():
                    if nbr.GetSymbol() == "N":
                        self.is_polar_don_h[idx] = True
                        
            # Find Isopropyl Methyl Carbons
            if sym == "C":
                n_c_nbrs = len([nbr for nbr in atom.GetNeighbors() if nbr.GetSymbol() == "C"])
                n_h_nbrs = len([nbr for nbr in atom.GetNeighbors() if nbr.GetSymbol() == "H"])
                # Terminal methyl group: bonded to 1 Carbon and 3 Hydrogens
                if n_c_nbrs == 1 and n_h_nbrs == 3:
                    self.is_hydro_c[idx] = True
                    
        # Compute graph distances
        adj = np.zeros((self.n_atoms, self.n_atoms))
        for src, tgt in self.bonds:
            adj[src, tgt] = 1
            adj[tgt, src] = 1
        G = nx.from_numpy_array(adj)
        self.graph_dist = dict(nx.all_pairs_shortest_path_length(G))
        self.graph_dist = np.array([[self.graph_dist[i].get(p, 999) for p in range(self.n_atoms)] for i in range(self.n_atoms)])

# --- UNIFIED SOLVENT-COUPLED SIMULATION ENGINE ---
class PNIPAMSolventSimulation:
    def __init__(self, n_monomers=15, n_water=150):
        self.agent = PNIPAMAgent(n_monomers=n_monomers)
        self.n_poly = self.agent.n_atoms
        self.n_water = n_water
        self.n_total = self.n_poly + self.n_water
        
        # Initialize coordinates: polymer in center, water surrounding
        poly_coords = self.agent.initial_coords.copy()
        # Shift polymer to origin
        poly_coords -= np.mean(poly_coords, axis=0)
        
        # Random water coordinates in a box L = 26 A
        if n_water > 0:
            water_coords = []
            for _ in range(n_water):
                water_coords.append([
                    np.random.uniform(-13.0, 13.0),
                    np.random.uniform(-13.0, 13.0),
                    np.random.uniform(-13.0, 13.0)
                ])
            water_coords = np.array(water_coords)
        else:
            water_coords = np.zeros((0, 3))
            
        self.coords = np.vstack([poly_coords, water_coords])
        
    def compute_energy_and_forces(self, coords):
        n_poly = self.n_poly
        n_total = self.n_total
        grad = np.zeros_like(coords)
        total_energy = 0.0
        
        # Constant biophysical coefficients (temperature-independent)
        gamma_hb = 0.5
        gamma_hydro = 0.5
        
        # 1. Polymer Covalent Bonds (harmonic spring constraints)
        k_b = 80.0
        d_cov = 1.5
        for i, p in self.agent.bonds:
            diff = coords[i] - coords[p]
            d = np.linalg.norm(diff)
            if d > 0.01:
                total_energy += 0.5 * k_b * (d - d_cov)**2
                g_val = k_b * (d - d_cov) * (diff / d)
                grad[i] += g_val
                grad[p] -= g_val
                
        # 2. Universal Steric Repulsion (Polymer-Polymer, Polymer-Water, Water-Water)
        for i in range(n_total):
            sym_i = self.agent.types[i] if i < n_poly else "O" # Water is O
            for p in range(i + 1, n_total):
                # Skip covalent
                if i < n_poly and p < n_poly and (self.agent.graph_dist[i, p] < 3):
                    continue
                    
                diff = coords[i] - coords[p]
                d = np.linalg.norm(diff)
                if d < 0.1:
                    d = 0.1
                    
                sym_p = self.agent.types[p] if p < n_poly else "O"
                thresh = 1.8 if (sym_i == "H" or sym_p == "H") else 2.4
                
                # Steric Potential
                if d < thresh:
                    e_steric = 10.0 / ((d - 0.2)**8 + 0.01)
                    total_energy += e_steric
                    g_steric = -80.0 * (d - 0.2)**7 / (((d - 0.2)**8 + 0.01)**2) * (diff / d)
                    grad[i] += g_steric
                    grad[p] -= g_steric
                    
        # 3. Non-Covalent Specific Forces
        active_bonds = [] # for Fiedler calculation
        
        # A. Amide-Water (AM-W) Hydrogen bonding
        for i in range(n_poly):
            # We look at Amide Oxygen or Amide Hydrogen
            is_amide = self.agent.is_polar_acc[i] or self.agent.is_polar_don_h[i]
            if not is_amide:
                continue
            for w in range(n_poly, n_total):
                diff = coords[i] - coords[w]
                d = np.linalg.norm(diff)
                if d < 4.0:
                    w_edge = gamma_hb * (20.0 / (d**2 + 0.1))
                    total_energy -= w_edge
                    g_hb = gamma_hb * (40.0 * d / (d**2 + 0.1)**2) * (diff / d)
                    grad[i] += g_hb
                    grad[w] -= g_hb
                    
                    if w_edge > 0.1:
                        active_bonds.append((i, w, w_edge))
                        
        # B. Isopropyl-Isopropyl (IP-IP) Hydrophobic attraction
        for i in range(n_poly):
            if not self.agent.is_hydro_c[i]:
                continue
            for p in range(i + 1, n_poly):
                if not self.agent.is_hydro_c[p]:
                    continue
                if self.agent.graph_dist[i, p] < 4:
                    continue
                    
                diff = coords[i] - coords[p]
                d = np.linalg.norm(diff)
                if d < 4.5:
                    w_edge = gamma_hydro * (35.0 / (d**2 + 0.1))
                    total_energy -= w_edge
                    g_hydro = gamma_hydro * (70.0 * d / (d**2 + 0.1)**2) * (diff / d)
                    grad[i] += g_hydro
                    grad[p] -= g_hydro
                    
                    if w_edge > 0.1:
                        active_bonds.append((i, p, w_edge))
                        
        # C. Isopropyl-Water (IP-W) Hydrophobic Exclusion (Repulsion at high temp)
        for i in range(n_poly):
            if not self.agent.is_hydro_c[i]:
                continue
            for w in range(n_poly, n_total):
                diff = coords[i] - coords[w]
                d = np.linalg.norm(diff)
                if d < 4.0:
                    e_rep = gamma_hydro * (15.0 / (d**4 + 0.1))
                    total_energy += e_rep
                    g_rep = -gamma_hydro * (60.0 * d**3 / (d**4 + 0.1)**2) * (diff / d)
                    grad[i] += g_rep
                    grad[w] -= g_rep
                    
        # 4. Box Boundary Wall (Harmonic potential)
        box_limit = 13.0
        for i in range(n_total):
            for axis in range(3):
                val = coords[i, axis]
                if val > box_limit:
                    grad[i, axis] += 80.0 * (val - box_limit)
                elif val < -box_limit:
                    grad[i, axis] += 80.0 * (val + box_limit)
                    
        # 5. Differentiable Fiedler Gradient (only if there are active bonds)
        l2 = 0.0
        if len(active_bonds) > 3:
            G_sys = nx.Graph()
            G_sys.add_nodes_from(range(n_total))
            
            # Covalent bonds
            for src, tgt in self.agent.bonds:
                G_sys.add_edge(src, tgt, weight=10.0)
                
            # Dynamic bonds
            for i, p, w_edge in active_bonds:
                G_sys.add_edge(i, p, weight=w_edge)
                
            try:
                l2 = nx.algebraic_connectivity(G_sys, method='lanczos', tol=1e-5)
                # Compute eigenvectors
                L_matrix = nx.laplacian_matrix(G_sys).toarray()
                _, eigvecs = np.linalg.eigh(L_matrix)
                v2 = eigvecs[:, 1]
                
                # Apply Fiedler attraction gradient to coordinates
                fiedler_force_coeff = 2.5
                for i, p, w_edge in active_bonds:
                    diff = coords[i] - coords[p]
                    d = np.linalg.norm(diff)
                    if d > 0.1:
                        dv_sq = (v2[i] - v2[p])**2
                        dw_dd = -2.0 * w_edge / d
                        g_fiedler = fiedler_force_coeff * dv_sq * dw_dd * (diff / d)
                        grad[i] += g_fiedler
                        grad[p] -= g_fiedler
            except:
                pass
                
        return total_energy, grad, l2

    def run_simulation(self, steps=200, lr=0.008):
        coords = self.coords.copy()
        
        for step in range(steps):
            energy, grad, l2 = self.compute_energy_and_forces(coords)
            
            # Gradient clipping to prevent exploding steps
            grad_norms = np.linalg.norm(grad, axis=-1, keepdims=True)
            grad = np.where(grad_norms > 12.0, grad / (grad_norms + 1e-6) * 12.0, grad)
            
            coords -= lr * grad
            
        return coords, l2

# --- MULTI-CHAIN (TWO-CHAIN) SIMULATION ENGINE ---
class PNIPAMTwoChainSimulation:
    def __init__(self, n_monomers=15, n_water=150):
        self.agentA = PNIPAMAgent(n_monomers=n_monomers)
        self.agentB = PNIPAMAgent(n_monomers=n_monomers)
        self.n_poly1 = self.agentA.n_atoms
        self.n_poly2 = self.agentB.n_atoms
        self.n_poly = self.n_poly1 + self.n_poly2
        self.n_water = n_water
        self.n_total = self.n_poly + self.n_water
        
        # Shift Chain A to left, Chain B to right (4 A separation)
        coordsA = self.agentA.initial_coords.copy()
        coordsA -= np.mean(coordsA, axis=0)
        coordsA += np.array([-2.0, 0.0, 0.0])
        
        coordsB = self.agentB.initial_coords.copy()
        coordsB -= np.mean(coordsB, axis=0)
        coordsB += np.array([2.0, 0.0, 0.0])
        
        # Random water coordinates surrounding both chains in box L = 28 A
        if n_water > 0:
            water_coords = []
            for _ in range(n_water):
                water_coords.append([
                    np.random.uniform(-14.0, 14.0),
                    np.random.uniform(-14.0, 14.0),
                    np.random.uniform(-14.0, 14.0)
                ])
            water_coords = np.array(water_coords)
        else:
            water_coords = np.zeros((0, 3))
            
        self.coords = np.vstack([coordsA, coordsB, water_coords])
        
        # Merge types and properties
        self.types = self.agentA.types + self.agentB.types + ["W"] * n_water
        self.is_polar_acc = np.concatenate([self.agentA.is_polar_acc, self.agentB.is_polar_acc])
        self.is_polar_don_h = np.concatenate([self.agentA.is_polar_don_h, self.agentB.is_polar_don_h])
        self.is_hydro_c = np.concatenate([self.agentA.is_hydro_c, self.agentB.is_hydro_c])
        
        # Build merged bonds list (Chain B indices are shifted by n_poly1)
        self.bonds = list(self.agentA.bonds)
        for u, v in self.agentB.bonds:
            self.bonds.append((u + self.n_poly1, v + self.n_poly1))
            
        # Build merged graph distance table (between-chain distances are infinite/999)
        self.graph_dist = np.full((self.n_poly, self.n_poly), 999)
        self.graph_dist[:self.n_poly1, :self.n_poly1] = self.agentA.graph_dist
        self.graph_dist[self.n_poly1:, self.n_poly1:] = self.agentB.graph_dist
        
        # Precompute steric masks and parameters for vectorized force calculation
        self.thresh_matrix = np.full((self.n_total, self.n_total), 2.4)
        is_H = np.array([sym == "H" for sym in self.types])
        self.thresh_matrix[is_H, :] = 1.8
        self.thresh_matrix[:, is_H] = 1.8
        
        self.skip_mask = np.zeros((self.n_total, self.n_total), dtype=bool)
        for i in range(self.n_poly):
            for p in range(self.n_poly):
                if self.graph_dist[i, p] < 3:
                    self.skip_mask[i, p] = True
                    self.skip_mask[p, i] = True
                    
    def compute_energy_and_forces(self, coords):
        n_poly = self.n_poly
        n_total = self.n_total
        grad = np.zeros_like(coords)
        total_energy = 0.0
        
        gamma_hb = 0.5
        gamma_hydro = 0.5
        
        # 1. Covalent bonds for both chains (keep loop since bonds are small, ~500)
        k_b = 80.0
        d_cov = 1.5
        for i, p in self.bonds:
            diff = coords[i] - coords[p]
            d = np.linalg.norm(diff)
            if d > 0.01:
                total_energy += 0.5 * k_b * (d - d_cov)**2
                g_val = k_b * (d - d_cov) * (diff / d)
                grad[i] += g_val
                grad[p] -= g_val
                
        # Compute all pairwise vectors and distances using NumPy
        diff_all = coords[:, np.newaxis, :] - coords[np.newaxis, :, :] # (N, N, 3)
        d_all = np.linalg.norm(diff_all, axis=-1) # (N, N)
        
        # 2. Steric Repulsion (Vectorized)
        steric_mask = (d_all < self.thresh_matrix) & (~self.skip_mask)
        steric_mask = np.triu(steric_mask, k=1)
        
        if np.any(steric_mask):
            rows, cols = np.where(steric_mask)
            d_val = d_all[steric_mask]
            d_val = np.maximum(d_val, 0.1) # avoid division by zero
            
            e_steric = 10.0 / ((d_val - 0.2)**8 + 0.01)
            total_energy += np.sum(e_steric)
            
            g_coeff = -80.0 * (d_val - 0.2)**7 / (((d_val - 0.2)**8 + 0.01)**2 * d_val)
            g_vecs = g_coeff[:, np.newaxis] * diff_all[steric_mask]
            
            np.add.at(grad, rows, g_vecs)
            np.subtract.at(grad, cols, g_vecs)
            
        # 3. Non-Covalent Forces (Hydrogen bonds and Hydrophobic bonds)
        active_bonds = []
        
        # A. Amide-Water (AM-W) Hydrogen bonding (Vectorized)
        amide_indices = np.where(self.is_polar_acc | self.is_polar_don_h)[0]
        water_indices = np.arange(self.n_poly, self.n_total)
        
        if len(amide_indices) > 0 and len(water_indices) > 0:
            d_slice = d_all[amide_indices[:, np.newaxis], water_indices[np.newaxis, :]]
            hb_mask = d_slice < 4.0
            
            if np.any(hb_mask):
                rows_idx, cols_idx = np.where(hb_mask)
                a_idx = amide_indices[rows_idx]
                w_idx = water_indices[cols_idx]
                
                d_val = d_slice[hb_mask]
                d_val = np.maximum(d_val, 0.1)
                
                w_edge = gamma_hb * (20.0 / (d_val**2 + 0.1))
                total_energy -= np.sum(w_edge)
                
                g_coeff = gamma_hb * (40.0 / (d_val**2 + 0.1)**2)
                diff_vecs = coords[a_idx] - coords[w_idx]
                g_hb = g_coeff[:, np.newaxis] * diff_vecs
                
                np.add.at(grad, a_idx, g_hb)
                np.subtract.at(grad, w_idx, g_hb)
                
                for idx_bond in range(len(a_idx)):
                    if w_edge[idx_bond] > 0.1:
                        active_bonds.append((a_idx[idx_bond], w_idx[idx_bond], w_edge[idx_bond]))
                        
        # B. Isopropyl-Isopropyl (IP-IP) Hydrophobic attraction (Intra- and Inter-chain, Vectorized)
        hydro_indices = np.where(self.is_hydro_c)[0]
        if len(hydro_indices) > 1:
            d_slice = d_all[hydro_indices[:, np.newaxis], hydro_indices[np.newaxis, :]]
            gd_slice = self.graph_dist[hydro_indices[:, np.newaxis], hydro_indices[np.newaxis, :]]
            
            hydro_mask = (d_slice < 4.5) & (gd_slice >= 4)
            hydro_mask = np.triu(hydro_mask, k=1)
            
            if np.any(hydro_mask):
                rows_idx, cols_idx = np.where(hydro_mask)
                i_idx = hydro_indices[rows_idx]
                p_idx = hydro_indices[cols_idx]
                
                d_val = d_slice[hydro_mask]
                d_val = np.maximum(d_val, 0.1)
                
                w_edge = gamma_hydro * (35.0 / (d_val**2 + 0.1))
                total_energy -= np.sum(w_edge)
                
                g_coeff = gamma_hydro * (70.0 / (d_val**2 + 0.1)**2)
                diff_vecs = coords[i_idx] - coords[p_idx]
                g_hydro = g_coeff[:, np.newaxis] * diff_vecs
                
                np.add.at(grad, i_idx, g_hydro)
                np.subtract.at(grad, p_idx, g_hydro)
                
                for idx_bond in range(len(i_idx)):
                    if w_edge[idx_bond] > 0.1:
                        active_bonds.append((i_idx[idx_bond], p_idx[idx_bond], w_edge[idx_bond]))
                        
        # C. Isopropyl-Water (IP-W) Hydrophobic Exclusion (Vectorized)
        if len(hydro_indices) > 0 and len(water_indices) > 0:
            d_slice = d_all[hydro_indices[:, np.newaxis], water_indices[np.newaxis, :]]
            ipw_mask = d_slice < 4.0
            
            if np.any(ipw_mask):
                rows_idx, cols_idx = np.where(ipw_mask)
                i_idx = hydro_indices[rows_idx]
                w_idx = water_indices[cols_idx]
                
                d_val = d_slice[ipw_mask]
                d_val = np.maximum(d_val, 0.1)
                
                e_rep = gamma_hydro * (15.0 / (d_val**4 + 0.1))
                total_energy += np.sum(e_rep)
                
                g_coeff = -gamma_hydro * (60.0 * d_val**2 / (d_val**4 + 0.1)**2)
                diff_vecs = coords[i_idx] - coords[w_idx]
                g_rep = g_coeff[:, np.newaxis] * diff_vecs
                
                np.add.at(grad, i_idx, g_rep)
                np.subtract.at(grad, w_idx, g_rep)
                
        # 4. Box Boundary Wall (Harmonic potential)
        box_limit = 14.0
        for i in range(n_total):
            for axis in range(3):
                val = coords[i, axis]
                if val > box_limit:
                    grad[i, axis] += 80.0 * (val - box_limit)
                elif val < -box_limit:
                    grad[i, axis] += 80.0 * (val + box_limit)
                    
        # 5. Differentiable Fiedler Gradient
        l2 = 0.0
        if len(active_bonds) > 3:
            G_sys = nx.Graph()
            G_sys.add_nodes_from(range(n_total))
            for src, tgt in self.bonds:
                G_sys.add_edge(src, tgt, weight=10.0)
            for i, p, w_edge in active_bonds:
                G_sys.add_edge(i, p, weight=w_edge)
                
            try:
                l2 = nx.algebraic_connectivity(G_sys, method='lanczos', tol=1e-5)
                L_matrix = nx.laplacian_matrix(G_sys).toarray()
                _, eigvecs = np.linalg.eigh(L_matrix)
                v2 = eigvecs[:, 1]
                
                fiedler_force_coeff = 2.5
                for i, p, w_edge in active_bonds:
                    diff = coords[i] - coords[p]
                    d = np.linalg.norm(diff)
                    if d > 0.1:
                        dv_sq = (v2[i] - v2[p])**2
                        dw_dd = -2.0 * w_edge / d
                        g_fiedler = fiedler_force_coeff * dv_sq * dw_dd * (diff / d)
                        grad[i] += g_fiedler
                        grad[p] -= g_fiedler
            except:
                pass
                
        return total_energy, grad, l2

# --- HELPER FUNCTION TO STRETCH POLYMER CHAIN TO EXTENDED STATE ---
def stretch_agent(agent):
    coords = agent.initial_coords.copy()
    mean = np.mean(coords, axis=0)
    coords_centered = coords - mean
    U, S, Vt = np.linalg.svd(coords_centered)
    
    # Project onto principal axes
    proj = coords_centered @ Vt
    
    # Scale the long axis (first principal component)
    proj[:, 0] *= 3.0
    
    # Reconstruct
    coords_stretched = proj @ Vt.T + mean
    
    # Relax coordinates with only covalent and steric forces
    bonds = agent.bonds
    k_b = 80.0
    d_cov = 1.5
    
    coords_relaxed = coords_stretched.copy()
    for _ in range(250):
        grad = np.zeros_like(coords_relaxed)
        for u, v in bonds:
            diff = coords_relaxed[u] - coords_relaxed[v]
            d = np.linalg.norm(diff)
            if d > 0.01:
                g = k_b * (d - d_cov) * (diff / d)
                grad[u] += g
                grad[v] -= g
                
        diff_all = coords_relaxed[:, np.newaxis, :] - coords_relaxed[np.newaxis, :, :]
        d_all = np.linalg.norm(diff_all, axis=-1)
        for i in range(agent.n_atoms):
            for p in range(i + 1, agent.n_atoms):
                if agent.graph_dist[i, p] >= 3 and d_all[i, p] < 2.4:
                    diff = coords_relaxed[i] - coords_relaxed[p]
                    d = d_all[i, p]
                    if d < 0.1: d = 0.1
                    g = -80.0 * (d - 0.2)**7 / (((d - 0.2)**8 + 0.01)**2) * (diff / d)
                    grad[i] += g
                    grad[p] -= g
                    
        grad_norms = np.linalg.norm(grad, axis=-1, keepdims=True)
        grad = np.where(grad_norms > 15.0, grad / (grad_norms + 1e-6) * 15.0, grad)
        coords_relaxed -= 0.012 * grad
    return coords_relaxed

# --- COVALENTLY CROSSLINKED DIMER (MBAA) SIMULATION ENGINE ---
class PNIPAMCrosslinkedSimulation:
    def __init__(self, n_monomers=15, n_water=150):
        self.agentA = PNIPAMAgent(n_monomers=n_monomers)
        self.agentB = PNIPAMAgent(n_monomers=n_monomers)
        self.n_monomers = n_monomers
        self.n_water = n_water
        
        # 1. Identify crosslinking nitrogens (1st and 15th nitrogen atoms, i.e. index 0 and 14)
        nitrogensA = [atom.GetIdx() for atom in self.agentA.mol.GetAtoms() if atom.GetSymbol() == "N"]
        N_crossA1 = nitrogensA[0]
        N_crossA2 = nitrogensA[14]
        
        # BFS to find isopropyl atoms for N_crossA1
        C_ipA1 = None
        for nbr in self.agentA.mol.GetAtomWithIdx(N_crossA1).GetNeighbors():
            if nbr.GetSymbol() == "C":
                has_O_nbr = any(n.GetSymbol() == "O" for n in nbr.GetNeighbors())
                if not has_O_nbr:
                    C_ipA1 = nbr.GetIdx()
                    break
        to_removeA1 = set()
        queue = [C_ipA1]
        to_removeA1.add(C_ipA1)
        visited = {N_crossA1, C_ipA1}
        while queue:
            curr = queue.pop(0)
            for nbr in self.agentA.mol.GetAtomWithIdx(curr).GetNeighbors():
                nbr_idx = nbr.GetIdx()
                if nbr_idx not in visited:
                    visited.add(nbr_idx)
                    to_removeA1.add(nbr_idx)
                    queue.append(nbr_idx)
                    
        # BFS to find isopropyl atoms for N_crossA2
        C_ipA2 = None
        for nbr in self.agentA.mol.GetAtomWithIdx(N_crossA2).GetNeighbors():
            if nbr.GetSymbol() == "C":
                has_O_nbr = any(n.GetSymbol() == "O" for n in nbr.GetNeighbors())
                if not has_O_nbr:
                    C_ipA2 = nbr.GetIdx()
                    break
        to_removeA2 = set()
        queue = [C_ipA2]
        to_removeA2.add(C_ipA2)
        visited = {N_crossA2, C_ipA2}
        while queue:
            curr = queue.pop(0)
            for nbr in self.agentA.mol.GetAtomWithIdx(curr).GetNeighbors():
                nbr_idx = nbr.GetIdx()
                if nbr_idx not in visited:
                    visited.add(nbr_idx)
                    to_removeA2.add(nbr_idx)
                    queue.append(nbr_idx)
                    
        to_removeA = to_removeA1.union(to_removeA2)
        
        # Map Chain A atoms
        self.mapA = {}
        idx_new = 0
        for i in range(self.agentA.n_atoms):
            if i in to_removeA:
                continue
            self.mapA[i] = idx_new
            idx_new += 1
        self.n_poly1 = len(self.mapA)
        
        # For Chain B
        nitrogensB = [atom.GetIdx() for atom in self.agentB.mol.GetAtoms() if atom.GetSymbol() == "N"]
        N_crossB1 = nitrogensB[0]
        N_crossB2 = nitrogensB[14]
        
        # BFS to find isopropyl atoms for N_crossB1
        C_ipB1 = None
        for nbr in self.agentB.mol.GetAtomWithIdx(N_crossB1).GetNeighbors():
            if nbr.GetSymbol() == "C":
                has_O_nbr = any(n.GetSymbol() == "O" for n in nbr.GetNeighbors())
                if not has_O_nbr:
                    C_ipB1 = nbr.GetIdx()
                    break
        to_removeB1 = set()
        queue = [C_ipB1]
        to_removeB1.add(C_ipB1)
        visited = {N_crossB1, C_ipB1}
        while queue:
            curr = queue.pop(0)
            for nbr in self.agentB.mol.GetAtomWithIdx(curr).GetNeighbors():
                nbr_idx = nbr.GetIdx()
                if nbr_idx not in visited:
                    visited.add(nbr_idx)
                    to_removeB1.add(nbr_idx)
                    queue.append(nbr_idx)
                    
        # BFS to find isopropyl atoms for N_crossB2
        C_ipB2 = None
        for nbr in self.agentB.mol.GetAtomWithIdx(N_crossB2).GetNeighbors():
            if nbr.GetSymbol() == "C":
                has_O_nbr = any(n.GetSymbol() == "O" for n in nbr.GetNeighbors())
                if not has_O_nbr:
                    C_ipB2 = nbr.GetIdx()
                    break
        to_removeB2 = set()
        queue = [C_ipB2]
        to_removeB2.add(C_ipB2)
        visited = {N_crossB2, C_ipB2}
        while queue:
            curr = queue.pop(0)
            for nbr in self.agentB.mol.GetAtomWithIdx(curr).GetNeighbors():
                nbr_idx = nbr.GetIdx()
                if nbr_idx not in visited:
                    visited.add(nbr_idx)
                    to_removeB2.add(nbr_idx)
                    queue.append(nbr_idx)
                    
        to_removeB = to_removeB1.union(to_removeB2)
        
        self.mapB = {}
        for i in range(self.agentB.n_atoms):
            if i in to_removeB:
                continue
            self.mapB[i] = idx_new
            idx_new += 1
        self.n_poly2 = len(self.mapB)
        self.n_poly = self.n_poly1 + self.n_poly2 + 2 # Chain A + Chain B + 2 Linkers
        self.n_total = self.n_poly + self.n_water
        
        # Initial coords (use stretched conformation!)
        coordsA = stretch_agent(self.agentA)
        coordsA -= np.mean(coordsA, axis=0)
        coordsA += np.array([-2.0, 0.0, 0.0])
        coordsA_filtered = np.delete(coordsA, list(to_removeA), axis=0)
        
        coordsB = stretch_agent(self.agentB)
        coordsB -= np.mean(coordsB, axis=0)
        coordsB += np.array([2.0, 0.0, 0.0])
        coordsB_filtered = np.delete(coordsB, list(to_removeB), axis=0)
        
        # Linker 1 coordinate (midpoint between N_crossA1 and N_crossB1)
        coords_link1 = (coordsA_filtered[self.mapA[N_crossA1]] + coordsB_filtered[self.mapB[N_crossB1] - self.n_poly1]) / 2.0
        coords_link1 = coords_link1.reshape(1, 3)
        
        # Linker 2 coordinate (midpoint between N_crossA2 and N_crossB2)
        coords_link2 = (coordsA_filtered[self.mapA[N_crossA2]] + coordsB_filtered[self.mapB[N_crossB2] - self.n_poly1]) / 2.0
        coords_link2 = coords_link2.reshape(1, 3)
        
        if n_water > 0:
            water_coords = []
            poly_coords = np.vstack([coordsA_filtered, coordsB_filtered, coords_link1, coords_link2])
            while len(water_coords) < n_water:
                candidate = np.array([
                    np.random.uniform(-14.0, 14.0),
                    np.random.uniform(-14.0, 14.0),
                    np.random.uniform(-14.0, 14.0)
                ])
                # Ensure no overlap with polymer atoms (d > 2.5 A)
                dists_poly = np.linalg.norm(poly_coords - candidate, axis=1)
                if np.all(dists_poly > 2.5):
                    # Ensure no major overlap with other water molecules (d > 2.0 A)
                    if len(water_coords) == 0 or np.all(np.linalg.norm(np.array(water_coords) - candidate, axis=1) > 2.0):
                        water_coords.append(candidate)
            water_coords = np.array(water_coords)
        else:
            water_coords = np.zeros((0, 3))
            
        self.coords = np.vstack([coordsA_filtered, coordsB_filtered, coords_link1, coords_link2, water_coords])
        
        # Types and properties
        typesA = [self.agentA.types[i] for i in range(self.agentA.n_atoms) if i not in to_removeA]
        typesB = [self.agentB.types[i] for i in range(self.agentB.n_atoms) if i not in to_removeB]
        self.types = typesA + typesB + ["C", "C"] + ["W"] * n_water
        
        pa_A = self.agentA.is_polar_acc[np.array([i not in to_removeA for i in range(self.agentA.n_atoms)])]
        pa_B = self.agentB.is_polar_acc[np.array([i not in to_removeB for i in range(self.agentB.n_atoms)])]
        self.is_polar_acc = np.concatenate([pa_A, pa_B, [False, False]])
        
        pd_H_A = self.agentA.is_polar_don_h[np.array([i not in to_removeA for i in range(self.agentA.n_atoms)])]
        pd_H_B = self.agentB.is_polar_don_h[np.array([i not in to_removeB for i in range(self.agentB.n_atoms)])]
        self.is_polar_don_h = np.concatenate([pd_H_A, pd_H_B, [False, False]])
        
        hc_A = self.agentA.is_hydro_c[np.array([i not in to_removeA for i in range(self.agentA.n_atoms)])]
        hc_B = self.agentB.is_hydro_c[np.array([i not in to_removeB for i in range(self.agentB.n_atoms)])]
        self.is_hydro_c = np.concatenate([hc_A, hc_B, [True, True]])
        
        # Build bonds
        bondsA = []
        for u, v in self.agentA.bonds:
            if u not in to_removeA and v not in to_removeA:
                bondsA.append((self.mapA[u], self.mapA[v]))
        bondsB = []
        for u, v in self.agentB.bonds:
            if u not in to_removeB and v not in to_removeB:
                bondsB.append((self.mapB[u], self.mapB[v]))
                
        # Connect to linkers
        idx_link1 = self.n_poly - 2 # 291
        idx_link2 = self.n_poly - 1 # 292
        self.bonds = bondsA + bondsB + [
            (self.mapA[N_crossA1], idx_link1), (self.mapB[N_crossB1], idx_link1),
            (self.mapA[N_crossA2], idx_link2), (self.mapB[N_crossB2], idx_link2)
        ]
        
        # Compute graph distances using NetworkX
        G_poly = nx.Graph()
        G_poly.add_nodes_from(range(self.n_poly))
        for u, v in self.bonds:
            G_poly.add_edge(u, v)
        path_lengths = dict(nx.all_pairs_shortest_path_length(G_poly))
        self.graph_dist = np.full((self.n_poly, self.n_poly), 999)
        for i in range(self.n_poly):
            for p in range(self.n_poly):
                if p in path_lengths[i]:
                    self.graph_dist[i, p] = path_lengths[i][p]
                    
        # Precompute steric masks and parameters
        self.thresh_matrix = np.full((self.n_total, self.n_total), 2.4)
        is_H = np.array([sym == "H" for sym in self.types])
        self.thresh_matrix[is_H, :] = 1.8
        self.thresh_matrix[:, is_H] = 1.8
        
        self.skip_mask = np.zeros((self.n_total, self.n_total), dtype=bool)
        for i in range(self.n_poly):
            for p in range(self.n_poly):
                if self.graph_dist[i, p] < 3:
                    self.skip_mask[i, p] = True
                    self.skip_mask[p, i] = True
                    
    def compute_energy_and_forces(self, coords):
        n_poly = self.n_poly
        n_total = self.n_total
        grad = np.zeros_like(coords)
        total_energy = 0.0
        
        gamma_hb = 0.5
        # Scale hydrophobic interaction by the level of dehydration
        scale_hydro = max(0.0, min(1.0, 1.0 - self.n_water / 180.0))
        gamma_hydro = 0.5 * scale_hydro
        
        # 1. Covalent bonds (including the crosslinker)
        k_b = 80.0
        d_cov = 1.5
        for i, p in self.bonds:
            diff = coords[i] - coords[p]
            d = np.linalg.norm(diff)
            if d > 0.01:
                total_energy += 0.5 * k_b * (d - d_cov)**2
                g_val = k_b * (d - d_cov) * (diff / d)
                grad[i] += g_val
                grad[p] -= g_val
                
        # Compute all pairwise vectors and distances
        diff_all = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        d_all = np.linalg.norm(diff_all, axis=-1)
        
        # 2. Steric Repulsion (Vectorized)
        steric_mask = (d_all < self.thresh_matrix) & (~self.skip_mask)
        steric_mask = np.triu(steric_mask, k=1)
        
        if np.any(steric_mask):
            rows, cols = np.where(steric_mask)
            d_val = d_all[steric_mask]
            d_val = np.maximum(d_val, 0.3)
            
            e_steric = 10.0 / ((d_val - 0.2)**8 + 0.01)
            total_energy += np.sum(e_steric)
            
            g_coeff = -80.0 * (d_val - 0.2)**7 / (((d_val - 0.2)**8 + 0.01)**2 * d_val)
            g_vecs = g_coeff[:, np.newaxis] * diff_all[steric_mask]
            
            np.add.at(grad, rows, g_vecs)
            np.subtract.at(grad, cols, g_vecs)
            
        # 3. Non-Covalent Forces
        active_bonds = []
        
        # A. Amide-Water (AM-W) Hydrogen bonding (Vectorized)
        amide_indices = np.where(self.is_polar_acc | self.is_polar_don_h)[0]
        water_indices = np.arange(self.n_poly, self.n_total)
        
        if len(amide_indices) > 0 and len(water_indices) > 0:
            d_slice = d_all[amide_indices[:, np.newaxis], water_indices[np.newaxis, :]]
            hb_mask = d_slice < 4.0
            
            if np.any(hb_mask):
                rows_idx, cols_idx = np.where(hb_mask)
                a_idx = amide_indices[rows_idx]
                w_idx = water_indices[cols_idx]
                
                d_val = d_slice[hb_mask]
                d_val = np.maximum(d_val, 0.1)
                
                w_edge = gamma_hb * (20.0 / (d_val**2 + 0.1))
                total_energy -= np.sum(w_edge)
                
                g_coeff = gamma_hb * (40.0 / (d_val**2 + 0.1)**2)
                diff_vecs = coords[a_idx] - coords[w_idx]
                g_hb = g_coeff[:, np.newaxis] * diff_vecs
                
                np.add.at(grad, a_idx, g_hb)
                np.subtract.at(grad, w_idx, g_hb)
                
                for idx_bond in range(len(a_idx)):
                    if w_edge[idx_bond] > 0.1:
                        active_bonds.append((a_idx[idx_bond], w_idx[idx_bond], w_edge[idx_bond]))
                        
        # B. Isopropyl-Isopropyl (IP-IP) Hydrophobic attraction (Vectorized)
        hydro_indices = np.where(self.is_hydro_c)[0]
        if len(hydro_indices) > 1:
            d_slice = d_all[hydro_indices[:, np.newaxis], hydro_indices[np.newaxis, :]]
            gd_slice = self.graph_dist[hydro_indices[:, np.newaxis], hydro_indices[np.newaxis, :]]
            
            hydro_mask = (d_slice < 4.5) & (gd_slice >= 4)
            hydro_mask = np.triu(hydro_mask, k=1)
            
            if np.any(hydro_mask):
                rows_idx, cols_idx = np.where(hydro_mask)
                i_idx = hydro_indices[rows_idx]
                p_idx = hydro_indices[cols_idx]
                
                d_val = d_slice[hydro_mask]
                d_val = np.maximum(d_val, 0.1)
                
                w_edge = gamma_hydro * (35.0 / (d_val**2 + 0.1))
                total_energy -= np.sum(w_edge)
                
                g_coeff = gamma_hydro * (70.0 / (d_val**2 + 0.1)**2)
                diff_vecs = coords[i_idx] - coords[p_idx]
                g_hydro = g_coeff[:, np.newaxis] * diff_vecs
                
                np.add.at(grad, i_idx, g_hydro)
                np.subtract.at(grad, p_idx, g_hydro)
                
                for idx_bond in range(len(i_idx)):
                    if w_edge[idx_bond] > 0.1:
                        active_bonds.append((i_idx[idx_bond], p_idx[idx_bond], w_edge[idx_bond]))
                        
        # C. Isopropyl-Water (IP-W) Hydrophobic Exclusion (Vectorized)
        if len(hydro_indices) > 0 and len(water_indices) > 0:
            d_slice = d_all[hydro_indices[:, np.newaxis], water_indices[np.newaxis, :]]
            ipw_mask = d_slice < 4.0
            
            if np.any(ipw_mask):
                rows_idx, cols_idx = np.where(ipw_mask)
                i_idx = hydro_indices[rows_idx]
                w_idx = water_indices[cols_idx]
                
                d_val = d_slice[ipw_mask]
                d_val = np.maximum(d_val, 0.1)
                
                e_rep = gamma_hydro * (15.0 / (d_val**4 + 0.1))
                total_energy += np.sum(e_rep)
                
                g_coeff = -gamma_hydro * (60.0 * d_val**2 / (d_val**4 + 0.1)**2)
                diff_vecs = coords[i_idx] - coords[w_idx]
                g_rep = g_coeff[:, np.newaxis] * diff_vecs
                
                np.add.at(grad, i_idx, g_rep)
                np.subtract.at(grad, w_idx, g_rep)
                
        # 4. Box Boundary Wall (Harmonic potential)
        box_limit = 14.0
        for i in range(n_total):
            for axis in range(3):
                val = coords[i, axis]
                if val > box_limit:
                    grad[i, axis] += 80.0 * (val - box_limit)
                elif val < -box_limit:
                    grad[i, axis] += 80.0 * (val + box_limit)
                    
        # 5. Differentiable Fiedler Gradient
        l2 = 0.0
        if len(active_bonds) > 3:
            G_sys = nx.Graph()
            G_sys.add_nodes_from(range(n_total))
            for src, tgt in self.bonds:
                G_sys.add_edge(src, tgt, weight=10.0)
            for i, p, w_edge in active_bonds:
                G_sys.add_edge(i, p, weight=w_edge)
                
            try:
                l2 = nx.algebraic_connectivity(G_sys, method='lanczos', tol=1e-5)
                L_matrix = nx.laplacian_matrix(G_sys).toarray()
                _, eigvecs = np.linalg.eigh(L_matrix)
                v2 = eigvecs[:, 1]
                
                fiedler_force_coeff = 2.5 * scale_hydro
                for i, p, w_edge in active_bonds:
                    diff = coords[i] - coords[p]
                    d = np.linalg.norm(diff)
                    if d > 0.1:
                        dv_sq = (v2[i] - v2[p])**2
                        # Correct analytical derivative of regularized w_edge = C / (d^2 + 0.1)
                        # dw/dd = -2.0 * d * w_edge / (d^2 + 0.1)
                        # force_vector = dw/dd * (diff / d) = -2.0 * w_edge / (d^2 + 0.1) * diff
                        g_fiedler = fiedler_force_coeff * dv_sq * (-2.0 * w_edge / (d**2 + 0.1)) * diff
                        grad[i] += g_fiedler
                        grad[p] -= g_fiedler
            except:
                pass
                
        return total_energy, grad, l2

    def run_simulation(self, steps=250, lr=0.008):
        coords = self.coords.copy()
        for step in range(steps):
            energy, grad, l2 = self.compute_energy_and_forces(coords)
            grad_norms = np.linalg.norm(grad, axis=-1, keepdims=True)
            grad = np.where(grad_norms > 12.0, grad / (grad_norms + 1e-6) * 12.0, grad)
            coords -= lr * grad
        return coords, l2

def run_pnipam_simulations():
    print("==================================================")
    print("  15-mer PNIPAM MBAA-CROSSLINKED DIMER DEHYDRATION SWEEP")
    print("  Differentiable Cartesian Fiedler Force Coupling")
    print("==================================================")
    
    n_water_list = [180, 120, 60, 30, 10, 0]
    sweep_results = []
    plot_states = {}
    
    for n_water in n_water_list:
        print(f"\n Running Crosslinked Simulation with {n_water} Water Molecules...")
        sim = PNIPAMCrosslinkedSimulation(n_monomers=15, n_water=n_water)
        
        # Relax system (250 steps)
        coords_final, l2 = sim.run_simulation(steps=250, lr=0.008)
        
        # Analyze final state
        n_poly1 = sim.n_poly1
        n_poly = sim.n_poly
        
        poly_coordsA = coords_final[:n_poly1]
        poly_coordsB = coords_final[n_poly1:n_poly - 2] # exclude both linkers
        
        c_A = np.mean(poly_coordsA, axis=0)
        c_B = np.mean(poly_coordsB, axis=0)
        
        # Radius of gyration
        rg_A = np.sqrt(np.mean(np.sum((poly_coordsA - c_A)**2, axis=-1)))
        rg_B = np.sqrt(np.mean(np.sum((poly_coordsB - c_B)**2, axis=-1)))
        
        # Center-of-mass separation
        separation = np.linalg.norm(c_A - c_B)
        
        # AM-W H-bonds
        hb_count = 0
        w_coords = coords_final[n_poly:]
        if n_water > 0:
            for i in range(n_poly):
                is_amide = sim.is_polar_acc[i] or sim.is_polar_don_h[i]
                if is_amide:
                    for w in range(n_water):
                        d = np.linalg.norm(coords_final[i] - w_coords[w])
                        if d < 3.5:
                            hb_count += 1
                            
        # IP-IP Hydrophobic bonds: Inter-chain vs Intra-chain
        inter_hydro = 0
        intra_hydro = 0
        for i in range(n_poly):
            if sim.is_hydro_c[i] and i < (n_poly - 2): # exclude linkers
                for p in range(i + 1, n_poly):
                    if sim.is_hydro_c[p] and p < (n_poly - 2): # exclude linkers
                        is_inter = (i < n_poly1 and p >= n_poly1) or (i >= n_poly1 and p < n_poly1)
                        if is_inter:
                            d = np.linalg.norm(coords_final[i] - coords_final[p])
                            if d < 4.5:
                                inter_hydro += 1
                        else:
                            if sim.graph_dist[i, p] >= 4:
                                d = np.linalg.norm(coords_final[i] - coords_final[p])
                                if d < 4.5:
                                    intra_hydro += 1
                                    
        print(f"   * Final Separation: {separation:.3f} Å")
        print(f"   * Chain A Rg: {rg_A:.3f} Å | Chain B Rg: {rg_B:.3f} Å")
        print(f"   * Fiedler Connectivity (λ2): {l2:.5f}")
        print(f"   * Amide-Water H-Bonds (AM-W): {hb_count}")
        print(f"   * Inter-chain Hydrophobic Contacts: {inter_hydro}")
        print(f"   * Intra-chain Hydrophobic Contacts: {intra_hydro}")
        
        sweep_results.append({
            "n_water": n_water,
            "separation": separation,
            "rg_A": rg_A,
            "rg_B": rg_B,
            "l2": l2,
            "hb": hb_count,
            "inter_hydro": inter_hydro,
            "intra_hydro": intra_hydro
        })
        
        # Save Wet (180) and Dry (0) for 3D plotting
        if n_water in [180, 0]:
            name = "Fully Solvated Wet State (180 Waters)" if n_water == 180 else "Fully Dehydrated Dry State (0 Waters)"
            plot_states[name] = {
                "coords": coords_final.tolist(),
                "types": sim.types,
                "n_poly": sim.n_poly,
                "n_poly1": sim.n_poly1,
                "sep": separation,
                "l2": l2,
                "hb": hb_count,
                "inter_hydro": inter_hydro,
                "intra_hydro": intra_hydro,
                "bonds": sim.bonds,
                "is_hydro_c": sim.is_hydro_c.tolist(),
                "is_polar_acc": sim.is_polar_acc.tolist(),
                "is_polar_don_h": sim.is_polar_don_h.tolist()
            }
            
    # Save sweep results to JSON
    with open("pnipam_two_chain_dehydration_results.json", "w") as f:
        json.dump({
            "sweep": sweep_results,
            "plot_states": plot_states
        }, f)
        
    # --- PLOT 1: SIDE-BY-SIDE 3D STRUCTURAL PLOT ---
    art_dir = "/home/eldenring/.gemini/antigravity/brain/d1e57b48-f8cc-4dd3-9357-bbffb4c85cbd"
    os.makedirs(art_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 7.5), facecolor='#111111')
    
    for idx, (name, res) in enumerate(plot_states.items()):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
        ax.set_facecolor('#111111')
        
        coords = np.array(res["coords"])
        n_poly = res["n_poly"]
        n_poly1 = res["n_poly1"]
        types = res["types"]
        
        poly_coords = coords[:n_poly]
        w_coords = coords[n_poly:]
        
        # Plot Water beads (semi-transparent cyan)
        if len(w_coords) > 0:
            ax.scatter(w_coords[:, 0], w_coords[:, 1], w_coords[:, 2], 
                       color='#00e1ff', alpha=0.08, s=12, label='Water' if idx==0 else "")
            
        # Distinguish beads
        is_hydro_c = res["is_hydro_c"]
        is_polar_acc = res["is_polar_acc"]
        is_polar_don_h = res["is_polar_don_h"]
        
        # Plot Backbone (Chain A: Neon Orange, Chain B: Neon Magenta)
        bbA_idx = [i for i in range(n_poly1) if types[i] == "C" and not is_hydro_c[i] and i < (n_poly - 2)]
        bbB_idx = [i for i in range(n_poly1, n_poly - 2) if types[i] == "C" and not is_hydro_c[i]]
        
        ax.scatter(poly_coords[bbA_idx, 0], poly_coords[bbA_idx, 1], poly_coords[bbA_idx, 2], 
                   color='#ff8000', s=45, label='Chain A Backbone' if idx==0 else "")
        ax.scatter(poly_coords[bbB_idx, 0], poly_coords[bbB_idx, 1], poly_coords[bbB_idx, 2], 
                   color='#ff00ff', s=45, label='Chain B Backbone' if idx==0 else "")
        
        # Plot Amide (Green) and Isopropyl (Purple)
        am_idx = [i for i in range(n_poly - 2) if types[i] in ["N", "O"] and (is_polar_acc[i] or is_polar_don_h[i])]
        ip_idx = [i for i in range(n_poly - 2) if is_hydro_c[i]]
        
        ax.scatter(poly_coords[am_idx, 0], poly_coords[am_idx, 1], poly_coords[am_idx, 2], 
                   color='#00ffd5', s=50, label='Amide (Polar)' if idx==0 else "")
        ax.scatter(poly_coords[ip_idx, 0], poly_coords[ip_idx, 1], poly_coords[ip_idx, 2], 
                   color='#d500ff', s=55, label='Isopropyl (Hydrophobic)' if idx==0 else "")
        
        # Plot MBAA Crosslinker beads (Neon Green, larger sphere)
        ax.scatter(poly_coords[n_poly - 2:, 0], poly_coords[n_poly - 2:, 1], poly_coords[n_poly - 2:, 2], 
                   color='#00ff00', s=120, label='MBAA Crosslinker' if idx==0 else "", edgecolors='#ffffff', linewidths=1.0)
        
        # Draw bonds
        for b_src, b_tgt in res["bonds"]:
            if b_src >= (n_poly - 2) or b_tgt >= (n_poly - 2):
                b_color = '#00ff00' # Neon Green for MBAA Crosslinks
                b_width = 2.5
            else:
                b_color = '#ff8000' if b_src < n_poly1 else '#ff00ff'
                b_width = 1.5
            ax.plot([poly_coords[b_src, 0], poly_coords[b_tgt, 0]], 
                    [poly_coords[b_src, 1], poly_coords[b_tgt, 1]], 
                    [poly_coords[b_src, 2], poly_coords[b_tgt, 2]], color=b_color, linewidth=b_width, alpha=0.8)
            
        ax.set_title(f"{name}\nSeparation: {res['sep']:.2f}Å | λ2: {res['l2']:.4f}\nInter-Hydro: {res['inter_hydro']} | Intra-Hydro: {res['intra_hydro']}", 
                     color='#e5e5ed', fontsize=11, pad=5)
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_xlim(-16, 16)
        ax.set_ylim(-16, 16)
        ax.set_zlim(-16, 16)
        
    fig.legend(loc='lower center', ncol=6, facecolor='#222222', edgecolor='none', labelcolor='#e5e5ed')
    plt.suptitle("15-mer PNIPAM MBAA-Crosslinked Dimer (Double Linker): Wet vs Dry States", color='#e5e5ed', fontsize=15, fontweight='bold', y=0.96)
    plt.tight_layout(rect=[0, 0.08, 1, 0.94])
    
    plot_path_3d = os.path.join(art_dir, "pnipam_folding_transition.png")
    plt.savefig(plot_path_3d, dpi=150, facecolor='#111111', edgecolor='none')
    plt.close()
    
    # --- PLOT 2: TWO-CHAIN ASSEMBLY SWEEP CURVE ---
    water_counts = [r["n_water"] for r in sweep_results]
    separations = [r["separation"] for r in sweep_results]
    fiedlers = [r["l2"] for r in sweep_results]
    
    fig, ax1 = plt.subplots(figsize=(8, 5), facecolor='#111111')
    ax1.set_facecolor('#111111')
    
    # Fiedler Connectivity Curve (Left axis)
    color = '#00ff00'
    ax1.set_xlabel('Number of Water Molecules (Dehydration Sweep)', color='#e5e5ed', fontsize=12)
    ax1.set_ylabel('Fiedler Connectivity (λ2)', color=color, fontsize=11)
    line1 = ax1.plot(water_counts, fiedlers, color=color, marker='o', linewidth=2.5, label='Fiedler Connectivity (λ2)')
    ax1.tick_params(axis='y', labelcolor=color, colors='#8e8e9f')
    ax1.tick_params(axis='x', colors='#8e8e9f')
    ax1.grid(color='#222222', linestyle='--')
    ax1.invert_xaxis() # Show drying process from left (180) to right (0)
    
    # Chain-Chain Separation (Right axis)
    ax2 = ax1.twinx()
    color2 = '#ff8000'
    ax2.set_ylabel('Chain-Chain Separation (Å)', color=color2, fontsize=11)
    line2 = ax2.plot(water_counts, separations, color=color2, marker='s', linewidth=2.5, linestyle='--', label='Chain-Chain Separation')
    ax2.tick_params(axis='y', labelcolor=color2, colors='#8e8e9f')
    
    plt.title('MBAA-Crosslinked PNIPAM Dehydration-Induced Gel Compaction', color='#e5e5ed', fontsize=13, fontweight='bold', pad=15)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', facecolor='#222222', edgecolor='none', labelcolor='#e5e5ed')
    
    plt.tight_layout()
    plot_path_curve = os.path.join(art_dir, "pnipam_dehydration_transition.png")
    plt.savefig(plot_path_curve, dpi=150, facecolor='#111111', edgecolor='none')
    plt.close()
    
    print("==================================================")
    print(f" Saved 3D comparison plot to {plot_path_3d}")
    print(f" Saved transition curve plot to {plot_path_curve}")
    print(" Saved dehydration sweep results to pnipam_two_chain_dehydration_results.json")
    print("==================================================")

if __name__ == "__main__":
    run_pnipam_simulations()
