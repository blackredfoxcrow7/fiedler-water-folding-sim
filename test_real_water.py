import numpy as np
from server_real_water import PeptideSimulation

def test_h2o_geometry_conservation():
    print("=== Running H2O Geometry Conservation Test ===")
    sim = PeptideSimulation("Ala-Ala-Ala")
    sim.params["explicitWaterEnabled"] = True
    
    # 1. Run 100 simulation steps
    dt = 0.02
    for _ in range(100):
        sim.step(dt)
        
    # 2. Check each water molecule's O-H and H-H distance deviations
    for i in range(len(sim.waters)):
        pos_o = sim.water_coords[i, 0]
        pos_h1 = sim.water_coords[i, 1]
        pos_h2 = sim.water_coords[i, 2]
        
        d_oh1 = np.linalg.norm(pos_h1 - pos_o)
        d_oh2 = np.linalg.norm(pos_h2 - pos_o)
        d_h12 = np.linalg.norm(pos_h2 - pos_h1)
        
        # Stiff springs should keep these distances extremely close to their target values
        assert abs(d_oh1 - 0.96) < 0.08, f"Water {i} O-H1 distance drifted to {d_oh1:.3f} Å"
        assert abs(d_oh2 - 0.96) < 0.08, f"Water {i} O-H2 distance drifted to {d_oh2:.3f} Å"
        assert abs(d_h12 - 1.52) < 0.08, f"Water {i} H1-H2 distance drifted to {d_h12:.3f} Å"
        
    print("✓ Water molecule rigid V-shape geometry conservation verified successfully!")

def test_dynamic_hydrogen_bonds():
    print("=== Running Dynamic Hydrogen Bond Network Test ===")
    sim = PeptideSimulation("Ala-Ala")
    sim.params["explicitWaterEnabled"] = True
    sim.params["waterWaterStrength"] = 1.0
    
    # Manually place water 0 and water 1 close to each other in a hydrogen bonding layout
    # Water 0 Oxygen at [0, 0, 0]
    sim.water_coords[0, 0] = np.array([0.0, 0.0, 0.0])
    # Water 0 Hydrogen 1 at [0.96, 0, 0]
    sim.water_coords[0, 1] = np.array([0.96, 0.0, 0.0])
    # Water 0 Hydrogen 2 at [-0.24, 0.93, 0]
    sim.water_coords[0, 2] = np.array([-0.24, 0.93, 0.0])
    
    # Water 1 Oxygen at [2.8, 0, 0]
    sim.water_coords[1, 0] = np.array([2.8, 0.0, 0.0])
    # Water 1 Hydrogen 1 at [2.8 + 0.96, 0, 0]
    sim.water_coords[1, 1] = np.array([3.76, 0.0, 0.0])
    # Water 1 Hydrogen 2 at [2.8 - 0.24, 0.93, 0]
    sim.water_coords[1, 2] = np.array([2.56, 0.93, 0.0])
    
    # In this configuration:
    # Water 0 Hydrogen 1 (at [0.96, 0, 0]) is pointing directly towards Water 1 Oxygen (at [2.8, 0, 0])
    # Distance = 1.84 Å. Linearity is perfect (along X axis).
    # This should form a strong hydrogen bond!
    
    # Step simulation
    sim.step(0.02)
    
    # Verify that water 1 Oxygen and water 0 Hydrogen 1 have an active hydrogen bond
    print(f"Active water bonds: {sim.active_water_bonds}")
    assert (1, 0, 1) in sim.active_water_bonds or (0, 1, 1) in sim.active_water_bonds, "Hydrogen bond failed to form dynamically!"
    print("✓ Dynamic hydrogen bond formation verified successfully!")

if __name__ == '__main__':
    test_h2o_geometry_conservation()
    test_dynamic_hydrogen_bonds()
