import numpy as np
from server import PeptideSimulation

def test_water_agent_logic():
    print("=== Running Water Agent and Sequential Folding Logic Tests ===")
    sim = PeptideSimulation("Ala-Ser-Asp")
    sim.params["explicitWaterEnabled"] = True
    
    # 1. Verify exactly 2 waters spawned per hydrophilic atom (N or O)
    n_philic = sum(1 for a in sim.atoms if a["h_bond"] in ("donor", "acceptor", "both"))
    assert len(sim.waters) == n_philic * 2, f"Expected {n_philic * 2} waters (2 per hydrophilic atom), got {len(sim.waters)}"
    print(f"✓ Spawning of exactly 2 waters per hydrophilic atom verified ({len(sim.waters)} waters for {n_philic} atoms).")
    
    # 2. Verify default searching state on initialization
    for w in sim.waters:
        assert w["state"] == "searching", f"Initial state should be searching, got {w['state']}"
    print("✓ Initial searching states verified.")
    
    # 3. Step physics and verify some waters transition to adsorbed (since they start at 2.6Å near parents)
    sim.step(0.02)
    adsorbed_count = sum(1 for w in sim.waters if w["state"] == "adsorbed")
    print(f"  After 1 step, {adsorbed_count}/{len(sim.waters)} waters are adsorbed.")
    assert adsorbed_count > 0, "No waters adsorbed after stepping!"
    
    # 4. Simulate cooperative bonded state by manually placing parent atoms close and waters close
    w1 = 0
    w2 = 1
    while w2 < len(sim.waters) and sim.waters[w2]["parent_atom_id"] == sim.waters[w1]["parent_atom_id"]:
        w1 += 1
        w2 += 1
    parent_idx1 = sim.waters[w1]["parent_atom_id"]
    parent_idx2 = sim.waters[w2]["parent_atom_id"]
    
    # Place parents 7.0Å apart (close enough for H-bond spacing)
    sim.coords[parent_idx2] = sim.coords[parent_idx1] + np.array([7.0, 0.0, 0.0])
    # Place waters 2.3Å from parents, pointing towards each other (distance = 2.4Å)
    sim.water_coords[w1] = sim.coords[parent_idx1] + np.array([2.3, 0.0, 0.0])
    sim.water_coords[w2] = sim.coords[parent_idx2] - np.array([2.3, 0.0, 0.0])
    
    # Step simulation to trigger state evaluation
    sim.step(0.02)
    
    # Check if waters are bonded
    assert sim.waters[w1]["state"] == "bonded", f"w1 state should be bonded, got {sim.waters[w1]['state']}"
    assert sim.waters[w2]["state"] == "bonded", f"w2 state should be bonded, got {sim.waters[w2]['state']}"
    print("✓ Cooperative bonded states verified.")
    assert (w1, w2) in sim.active_water_bonds, f"Expected (w1, w2) in active_water_bonds: {sim.active_water_bonds}"
    
    # 5. Verify sequential folding completion transition (reversal to C->N)
    sim.seq_folding["enabled"] = True
    sim.seq_folding["activeResidue"] = sim.n_residues - 1
    sim.seq_folding["currentStep"] = sim.seq_folding["stepsPerResidue"] - 1
    sim.seq_folding["direction"] = 1
    
    # Step simulation to trigger completion transition
    sim.step(0.02)
    
    assert sim.seq_folding["direction"] == -1, f"Direction should have reversed to -1, got {sim.seq_folding['direction']}"
    assert sim.seq_folding["activeResidue"] == sim.n_residues - 2, f"Active residue should have decremented to {sim.n_residues - 2}, got {sim.seq_folding['activeResidue']}"
    print("✓ Sequential folding bi-directional transition verified.")
    
    print("\n=== ALL WATER AGENT TESTS PASSED SUCCESSFULLY! ===")

if __name__ == '__main__':
    test_water_agent_logic()
