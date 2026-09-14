import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server_inference import PeptideInferenceSimulation
import numpy as np

def main():
    sim = PeptideInferenceSimulation("GYDPETGTWG")
    dt = 0.02
    
    print("Initial coords of atom 0:", sim.coords[0])
    
    # Let's inspect step 1 details
    print("\n--- Running step 1 ---")
    
    # 1. Step waters
    forces_reaction = sim.step_explicit_waters(dt)
    print("Max forces_reaction:", np.max(np.abs(forces_reaction)))
    
    # 2. Check other forces
    temp = sim.params.get("temperature", 0.15)
    forces_other = np.zeros((sim.n_atoms, 3))
    if temp > 0:
        noise = np.random.normal(0, temp * 3.5, size=(sim.n_atoms, 3))
        forces_other += noise
    print("Max forces_other (noise):", np.max(np.abs(forces_other)))
    
    # 3. Before rotations
    coords_before = sim.coords.copy()
    
    # Apply rotations
    print("\nApplying rotations...")
    for j_idx, joint in enumerate(sim.peptide.joints):
        u_idx = joint["u_idx"]
        d_idx = joint["d_idx"]
        D = joint["downstream_atoms"]
        
        pos_A = sim.coords[u_idx]
        pos_B = sim.coords[d_idx]
        axis = pos_B - pos_A
        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-4:
            continue
        axis /= axis_len
        
        torque_reaction = 0.0
        torque_other = 0.0
        inertia = 0.0
        
        for idx in D:
            r = sim.coords[idx] - pos_B
            t_react = np.cross(r, forces_reaction[idx])
            torque_reaction += np.dot(t_react, axis)
            
            t_other = np.cross(r, forces_other[idx])
            torque_other += np.dot(t_other, axis)
            
            cross_r_axis = np.cross(r, axis)
            inertia += np.dot(cross_r_axis, cross_r_axis)
            
        torque = torque_reaction + torque_other
        damping_const = 0.4
        effective_inertia = max(0.5, inertia * 0.05)
        d_theta = (dt * torque) / (damping_const * effective_inertia)
        d_theta = np.clip(d_theta, -0.4, 0.4)
        
        if abs(d_theta) > 0.01:
            print(f"  Joint {j_idx} (atoms {u_idx}->{d_idx}): d_theta={d_theta:.4f}, torque={torque:.4f}, inertia={inertia:.4f}")
            
        cos_t = np.cos(d_theta)
        sin_t = np.sin(d_theta)
        for idx in D:
            r_vec = sim.coords[idx] - pos_B
            rotated = r_vec * cos_t + np.cross(axis, r_vec) * sin_t + axis * np.dot(axis, r_vec) * (1.0 - cos_t)
            sim.coords[idx] = pos_B + rotated
            
    coords_after = sim.coords.copy()
    diff = np.max(np.abs(coords_after - coords_before))
    print(f"\nMax coordinate change during rotations: {diff:.6f} Å")
    print("Coords of atom 0 after rotations:", sim.coords[0])

if __name__ == "__main__":
    main()
