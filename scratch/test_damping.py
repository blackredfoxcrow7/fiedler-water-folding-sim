import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server_inference import PeptideInferenceSimulation

def test_combination(k_unfold, damping_const, max_d_theta):
    sim = PeptideInferenceSimulation("GYDPETGTWG")
    sim.load_pdb_conformation("1UAO.pdb")
    
    # Temporarily patch apply_rotations params in sim
    # We will run 300 steps and measure the change
    sim.learning_mode = True
    sim.reverse = True
    sim.active_residue = sim.n_residues - 1
    sim.step_counter_residue = 0
    
    # We will modify the peptide agent methods on the fly
    original_apply_rotations = sim.peptide.apply_rotations
    
    def patched_apply_rotations(dt, forces_reaction, forces_other, active_residue=None, reverse=False, use_inference=False):
        for j_idx, joint in enumerate(sim.peptide.joints):
            u_idx = joint["u_idx"]
            d_idx = joint["d_idx"]
            D = joint["downstream_atoms"]
            d_res = sim.peptide.atom_residues.get(d_idx, 0)
            
            if active_residue is not None:
                if d_res != active_residue:
                    continue
            else:
                if d_res == 0:
                    continue
                    
            pos_A = sim.peptide.coords[u_idx]
            pos_B = sim.peptide.coords[d_idx]
            axis = pos_B - pos_A
            axis_len = np.linalg.norm(axis)
            if axis_len < 1e-4:
                continue
            axis /= axis_len
            
            r = sim.peptide.coords[D] - pos_B
            t_react = np.cross(r, forces_reaction[D])
            torque_reaction = np.sum(t_react @ axis)
            
            t_other = np.cross(r, forces_other[D])
            torque_other = np.sum(t_other @ axis)
            
            cross_r_axis = np.cross(r, axis)
            inertia = np.sum(cross_r_axis * cross_r_axis)
                
            torque = torque_reaction + torque_other
            
            if reverse:
                unfold_torque = -k_unfold * sim.peptide.joint_angles[j_idx]
                torque += unfold_torque
                
            effective_inertia = max(0.5, inertia * 0.05)
            d_theta = (dt * torque) / (damping_const * effective_inertia)
            
            d_theta = np.clip(d_theta, -max_d_theta, max_d_theta)
            sim.peptide.joint_angles[j_idx] += d_theta
            
            cos_t = np.cos(d_theta)
            sin_t = np.sin(d_theta)
            r_vec = sim.peptide.coords[D] - pos_B
            cross_axis_r = np.cross(axis, r_vec)
            dot_axis_r = r_vec @ axis
            rotated = r_vec * cos_t + cross_axis_r * sin_t + axis[np.newaxis, :] * (dot_axis_r * (1.0 - cos_t))[:, np.newaxis]
            sim.peptide.coords[D] = pos_B + rotated

    sim.peptide.apply_rotations = patched_apply_rotations
    
    init_angle_34 = sim.peptide.joint_angles[34]
    for _ in range(150):
        sim.step(0.02)
        
    final_angle_34 = sim.peptide.joint_angles[34]
    delta = final_angle_34 - init_angle_34
    
    print(f"k_unfold={k_unfold:4.1f}, damping={damping_const:3.1f}, max_d_theta={max_d_theta:.3f} | Final angle 34: {final_angle_34:.4f} (delta: {delta:+.4f})")
    return np.abs(final_angle_34)

def main():
    print("Testing combinations to optimize unfolding...")
    test_combination(k_unfold=15.0, damping_const=0.4, max_d_theta=0.150)
    test_combination(k_unfold=15.0, damping_const=1.5, max_d_theta=0.150)
    test_combination(k_unfold=15.0, damping_const=3.0, max_d_theta=0.150)
    
    test_combination(k_unfold=15.0, damping_const=2.0, max_d_theta=0.080)
    test_combination(k_unfold=15.0, damping_const=3.0, max_d_theta=0.060)
    test_combination(k_unfold=25.0, damping_const=4.0, max_d_theta=0.050)
    test_combination(k_unfold=30.0, damping_const=5.0, max_d_theta=0.040)
    test_combination(k_unfold=40.0, damping_const=6.0, max_d_theta=0.040)

if __name__ == "__main__":
    main()
