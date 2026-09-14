import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from peptide_agent import PeptideAgent
import numpy as np

def test_peptide_agent_initialization():
    print("Testing PeptideAgent initialization...")
    # Initialize with Chignolin sequence
    seq = "GYDPETGTWG"
    agent = PeptideAgent(seq)
    
    assert agent.n_atoms > 0, "Atoms list should not be empty"
    assert len(agent.bonds) > 0, "Bonds list should not be empty"
    assert len(agent.joints) > 0, "Should identify rotatable joints"
    assert agent.n_residues == 10, f"Expected 10 residues, got {agent.n_residues}"
    
    print(f"✓ Success: Parsed {agent.n_atoms} atoms, {len(agent.bonds)} bonds, {len(agent.joints)} joints.")

def test_sequential_learning_and_inference():
    print("Testing sequential residue learning and inference...")
    seq = "GYDPETGTWG"
    agent = PeptideAgent(seq)
    
    # 1. Simulate learning step for a residue (e.g. residue 5)
    # Set mock folded angles for joints of residue 5
    res_5_joints = []
    for j_idx, joint in enumerate(agent.joints):
        d_res = agent.atom_residues[joint["d_idx"]]
        if d_res == 5:
            agent.joint_angles[j_idx] = 1.25 # Mock folded angle
            res_5_joints.append(j_idx)
            
    assert len(res_5_joints) > 0, "Should find joints for residue 5"
    
    # Run learning step
    agent.learn_residue_step(5)
    
    # Verify that the angles are recorded
    res_5_name = agent.residue_labels[5].split('-')[0]
    assert agent.learned_by_type[res_5_name] == 1.25, "Should record target angle by residue type"
    
    # 2. Reset coordinates and test inference torque direction
    agent.reset_to_initial()
    assert np.all(agent.joint_angles == 0.0), "Angles should be reset to 0.0"
    
    # Mock forces
    forces_reaction = np.zeros((agent.n_atoms, 3))
    forces_other = np.zeros((agent.n_atoms, 3))
    
    # Apply forward rotations with inference enabled for active residue 5
    agent.apply_rotations(
        dt=0.02, 
        forces_reaction=forces_reaction, 
        forces_other=forces_other, 
        active_residue=5, 
        reverse=False, 
        use_inference=True
    )
    
    # Check that joint angles of residue 5 moved towards the target 1.25
    for j_idx in res_5_joints:
        angle = agent.joint_angles[j_idx]
        assert angle > 0.0, f"Joint {j_idx} angle should have biased towards target 1.25, got {angle}"
        
    print("✓ Success: Verified step-by-step learning capture and inference steering bias force.")

if __name__ == "__main__":
    try:
        test_peptide_agent_initialization()
        test_sequential_learning_and_inference()
        print("\nAll unit tests passed successfully!")
    except AssertionError as e:
        print(f"\nAssertion Error: {e}")
        sys.exit(1)
