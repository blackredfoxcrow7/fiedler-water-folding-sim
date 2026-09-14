import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server_inference import PeptideInferenceSimulation
import numpy as np

def profile():
    sim = PeptideInferenceSimulation("GYDPETGTWG")
    dt = 0.02
    
    # Run once to warm up
    sim.step(dt)
    
    print("\n--- Profiling step ---")
    
    # 1. step_explicit_waters profile
    t0 = time.time()
    num_waters = len(sim.waters)
    forces_water = np.zeros((num_waters, 3, 3))
    forces_reaction_on_peptide = np.zeros((sim.n_atoms, 3))
    
    k_pw = sim.params.get("waterPeptideStrength", 0.60) * 15.0
    k_capture = sim.params.get("kCapture", 0.70) * 12.0
    
    # 1.1 Internal constraints
    t_start = time.time()
    k_int = 250.0
    pos_o = sim.water_coords[:, 0]
    pos_h1 = sim.water_coords[:, 1]
    pos_h2 = sim.water_coords[:, 2]
    
    diff_h1 = pos_h1 - pos_o
    d_h1 = np.linalg.norm(diff_h1, axis=1, keepdims=True)
    safe_d_h1 = np.where(d_h1 > 1e-3, d_h1, 1.0)
    f_h1 = (d_h1 - 0.96) * k_int * (diff_h1 / safe_d_h1)
    f_h1 = np.where(d_h1 > 1e-3, f_h1, 0.0)
    forces_water[:, 1] -= f_h1
    forces_water[:, 0] += f_h1
    
    diff_h2 = pos_h2 - pos_o
    d_h2 = np.linalg.norm(diff_h2, axis=1, keepdims=True)
    safe_d_h2 = np.where(d_h2 > 1e-3, d_h2, 1.0)
    f_h2 = (d_h2 - 0.96) * k_int * (diff_h2 / safe_d_h2)
    f_h2 = np.where(d_h2 > 1e-3, f_h2, 0.0)
    forces_water[:, 2] -= f_h2
    forces_water[:, 0] += f_h2
    
    diff_hh = pos_h2 - pos_h1
    d_hh = np.linalg.norm(diff_hh, axis=1, keepdims=True)
    safe_d_hh = np.where(d_hh > 1e-3, d_hh, 1.0)
    f_hh = (d_hh - 1.52) * k_int * (diff_hh / safe_d_hh)
    f_hh = np.where(d_hh > 1e-3, f_hh, 0.0)
    forces_water[:, 2] -= f_hh
    forces_water[:, 1] += f_hh
    print(f"  Internal constraints: {1000 * (time.time() - t_start):.2f} ms")
    
    # 1.2 Covalent parent tracking
    t_start = time.time()
    parent_ids = np.array([w["parent_atom_id"] for w in sim.waters])
    pos_p = sim.coords[parent_ids]
    diff_parent = pos_o - pos_p
    d_parent = np.linalg.norm(diff_parent, axis=1, keepdims=True)
    safe_d_parent = np.where(d_parent > 1e-3, d_parent, 1.0)
    f_parent = (d_parent - 3.8) * k_pw * (diff_parent / safe_d_parent)
    f_parent = np.where(d_parent > 1e-3, f_parent, 0.0)
    
    forces_water[:, 0] -= f_parent
    for i in range(num_waters):
        forces_reaction_on_peptide[parent_ids[i]] += f_parent[i]
    print(f"  Parent tracking: {1000 * (time.time() - t_start):.2f} ms")
    
    # 1.3 Entropy calculation (Dijkstra)
    t_start = time.time()
    # Let's run Dijkstra path updates (run_dijkstra = True)
    w_pos = sim.water_coords[:, 0]
    polar_indices = [idx for idx in range(sim.n_atoms) if sim.atoms[idx]["h_bond"] in ("donor", "acceptor", "both")]
    hydrophobic_indices = [idx for idx in range(sim.n_atoms) if sim.atoms[idx]["hydrophobicity"] > 0.15]
    entropy_bias = sim.params.get("entropyBias", 1.2)
    
    p_coords = sim.coords[polar_indices] if polar_indices else np.empty((0, 3))
    h_coords = sim.coords[hydrophobic_indices] if hydrophobic_indices else np.empty((0, 3))
    
    d_polar_matrix = np.linalg.norm(w_pos[:, np.newaxis, :] - p_coords[np.newaxis, :, :], axis=2)
    d_polar_min = np.min(d_polar_matrix, axis=1)
    
    d_hydro_matrix = np.linalg.norm(w_pos[:, np.newaxis, :] - h_coords[np.newaxis, :, :], axis=2)
    d_hydro_min = np.min(d_hydro_matrix, axis=1)
    
    f_polar = np.exp(- (d_polar_min ** 2) / (2 * 2.2 ** 2))
    f_hydro = np.exp(- (d_hydro_min ** 2) / (2 * 2.2 ** 2))
    
    water_entropies = 0.6 + 0.4 * f_hydro * entropy_bias - 0.5 * f_polar * entropy_bias
    water_entropies = np.clip(water_entropies, 0.05, 1.2).tolist()
    water_weights = 1.0 + 1.5 * f_polar - 0.85 * f_hydro
    water_weights = np.clip(water_weights, 0.05, 4.0).tolist()
    
    num_graph_nodes = num_waters + len(polar_indices)
    import collections, heapq
    graph_edges = {u: [] for u in range(num_graph_nodes)}
    
    dist_ww = np.linalg.norm(w_pos[:, np.newaxis, :] - w_pos[np.newaxis, :, :], axis=2)
    ii, jj = np.where((dist_ww < 3.5) & (np.arange(num_waters)[:, np.newaxis] < np.arange(num_waters)[np.newaxis, :]))
    for i, j in zip(ii, jj):
        w_ij = np.sqrt(water_weights[i] * water_weights[j])
        cost = 1.0 / (w_ij + 1e-5)
        graph_edges[i].append((j, cost))
        graph_edges[j].append((i, cost))
        
    ii, jj = np.where(d_polar_matrix < 3.5)
    for i, p_idx in zip(ii, jj):
        w_ip = np.sqrt(water_weights[i] * 2.5)
        cost = 1.0 / (w_ip + 1e-5)
        graph_node_p = num_waters + p_idx
        graph_edges[i].append((graph_node_p, cost))
        graph_edges[graph_node_p].append((i, cost))

    def dijkstra(src):
        dist = {src: 0.0}
        parent = {src: None}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, cost in graph_edges[u]:
                if dist.get(v, float('inf')) > d + cost:
                    dist[v] = d + cost
                    parent[v] = u
                    heapq.heappush(pq, (dist[v], v))
        return dist, parent

    polar_graph_nodes = [num_waters + idx for idx in range(len(polar_indices))]
    paths_by_src = {}
    for src in polar_graph_nodes:
        paths_by_src[src] = dijkstra(src)
        
    I_capture = 0.0
    path_edges_set = set()
    
    for idx_a in range(len(polar_indices)):
        node_a = polar_graph_nodes[idx_a]
        dists, parents = paths_by_src[node_a]
        for idx_b in range(idx_a + 1, len(polar_indices)):
            node_b = polar_graph_nodes[idx_b]
            d_ab = dists.get(node_b, float('inf'))
            if d_ab < 999.0:
                I_capture += 1.0 / d_ab
                curr = node_b
                while curr is not None:
                    prev = parents[curr]
                    if prev is not None:
                        edge = (min(curr, prev), max(curr, prev))
                        path_edges_set.add(edge)
                    curr = prev
    print(f"  Entropy + Dijkstra paths: {1000 * (time.time() - t_start):.2f} ms")
    
    # 1.4 Unconnected pairs force
    t_start = time.time()
    unconnected_pairs = []
    for idx_a in range(len(polar_indices)):
        node_a = polar_graph_nodes[idx_a]
        dists, _ = paths_by_src[node_a]
        for idx_b in range(idx_a + 1, len(polar_indices)):
            node_b = polar_graph_nodes[idx_b]
            if dists.get(node_b, float('inf')) > 999.0:
                unconnected_pairs.append((polar_indices[idx_a], polar_indices[idx_b]))
    if unconnected_pairs and num_waters > 0:
        p1_indices = np.array([p[0] for p in unconnected_pairs])
        p2_indices = np.array([p[1] for p in unconnected_pairs])
        midpoints = 0.5 * (sim.coords[p1_indices] + sim.coords[p2_indices])
        
        diff_mid = midpoints[:, np.newaxis, :] - w_pos[np.newaxis, :, :]
        d_mid = np.linalg.norm(diff_mid, axis=2)
        mid_mask = (d_mid > 0.1) & (d_mid < 8.0)
        
        if np.any(mid_mask):
            safe_d_mid = np.where(d_mid > 1e-5, d_mid, 1.0)
            f_dir_mid = diff_mid / safe_d_mid[:, :, np.newaxis]
            f_mag = k_capture * 0.03
            forces_to_apply = f_dir_mid * (f_mag * mid_mask)[:, :, np.newaxis]
            forces_water[:, 0] += np.sum(forces_to_apply, axis=0)
    print(f"  Unconnected pairs force: {1000 * (time.time() - t_start):.2f} ms")
    
    # 1.5 Steric repulsions
    t_start = time.time()
    dist_ww = np.linalg.norm(w_pos[:, np.newaxis, :] - w_pos[np.newaxis, :, :], axis=2)
    ii, jj = np.where((dist_ww < 2.8) & (np.arange(num_waters)[:, np.newaxis] < np.arange(num_waters)[np.newaxis, :]))
    for i, j in zip(ii, jj):
        diff = w_pos[j] - w_pos[i]
        d = dist_ww[i, j]
        if d > 1e-3:
            rep = (2.8 - d) * 30.0 * (diff / d)
            forces_water[i, 0] -= rep
            forces_water[j, 0] += rep

    diff_wp = sim.coords[np.newaxis, :, :] - w_pos[:, np.newaxis, :]
    dist_wp = np.linalg.norm(diff_wp, axis=2)
    
    radii = np.array([a["radius"] for a in sim.atoms])
    min_dists = radii + 1.2
    
    overlap_mask = dist_wp < min_dists[np.newaxis, :]
    ww, pp = np.where(overlap_mask)
    for w_idx, p_idx in zip(ww, pp):
        d = dist_wp[w_idx, p_idx]
        if d > 1e-3:
            diff = diff_wp[w_idx, p_idx]
            rep = (min_dists[p_idx] - d) * 35.0 * (diff / d)
            forces_water[w_idx, 0] -= rep
            forces_reaction_on_peptide[p_idx] += rep
    print(f"  Steric repulsions: {1000 * (time.time() - t_start):.2f} ms")

    # 1.6 Euler step
    t_start = time.time()
    noise = np.random.normal(0, 0.05, size=(num_waters, 3, 3))
    sim.water_coords += (forces_water + noise) * (dt * dt)
    print(f"  Water Euler step: {1000 * (time.time() - t_start):.2f} ms")
    
    # 2. Peptide intramolecular step
    t_start = time.time()
    # 2.1 Calculate diffs
    diff_p = sim.coords[:, np.newaxis, :] - sim.coords[np.newaxis, :, :]
    dists_p = np.linalg.norm(diff_p, axis=-1)
    
    forces_other = np.zeros((sim.n_atoms, 3))
    temp = sim.params.get("temperature", 0.15)
    if temp > 0:
        noise = np.random.normal(0, temp * 3.5, size=(sim.n_atoms, 3))
        forces_other += noise
        
    bulk_solv = sim.params.get("bulkSolventStrength", 0.45)
    if bulk_solv > 1e-4:
        com = np.mean(sim.coords, axis=0)
        dir_com = com - sim.coords
        d_com = np.linalg.norm(dir_com, axis=1, keepdims=True)
        hydrophobic_mask = np.array([a["hydrophobicity"] > 0.15 for a in sim.atoms])
        polar_mask = np.array([a["h_bond"] in ("donor", "acceptor", "both") for a in sim.atoms])
        valid = (d_com.flatten() > 0.1)
        h_indices = np.where(hydrophobic_mask & valid)[0]
        forces_other[h_indices] += (dir_com[h_indices] / d_com[h_indices]) * bulk_solv * 6.0
        p_indices = np.where(polar_mask & valid)[0]
        forces_other[p_indices] -= (dir_com[p_indices] / d_com[p_indices]) * bulk_solv * 4.0
        
    radii = np.array([a["radius"] for a in sim.atoms])
    min_dist_matrix = (radii[:, np.newaxis] + radii[np.newaxis, :]) * 0.45
    interact_mask = (sim.graph_dist > 2) & (dists_p > 1e-3)
    hydrophobic_mask = np.array([a["hydrophobicity"] > 0.15 for a in sim.atoms])
    hydro_both_mask = interact_mask & hydrophobic_mask[:, np.newaxis] & hydrophobic_mask[np.newaxis, :]
    steric_mask = interact_mask & (dists_p < min_dist_matrix)
    
    safe_dists = np.where(dists_p > 1e-5, dists_p, 1.0)
    dir_ij = diff_p / safe_dists[:, :, np.newaxis]
    
    if np.any(hydro_both_mask):
        attract_pull = (6.0 - dists_p) * 1.5
        hydro_attract_mask = hydro_both_mask & (dists_p > min_dist_matrix) & (dists_p < 6.0)
        attract_forces = dir_ij * (attract_pull * hydro_attract_mask)[:, :, np.newaxis]
        forces_other += np.sum(attract_forces, axis=1)
        forces_other -= np.sum(attract_forces, axis=0)
        
    if np.any(steric_mask):
        rep_push = (min_dist_matrix - dists_p) * 15.0
        rep_forces = dir_ij * (rep_push * steric_mask)[:, :, np.newaxis]
        forces_other -= np.sum(rep_forces, axis=1)
        forces_other += np.sum(rep_forces, axis=0)
    print(f"  Peptide intramolecular forces: {1000 * (time.time() - t_start):.2f} ms")

    # 3. Apply rotations
    t_start = time.time()
    sim.peptide.apply_rotations(dt, forces_reaction_on_peptide, forces_other, active_residue=None, reverse=False, use_inference=False)
    print(f"  Peptide apply rotations: {1000 * (time.time() - t_start):.2f} ms")

if __name__ == "__main__":
    profile()
