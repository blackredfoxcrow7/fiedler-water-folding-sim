import numpy as np
import json
from rdkit import Chem
from rdkit.Chem import AllChem

def build_fiedler_visualization_data():
    molecules = [
        {"id": "sds", "name": "SDS (Sodium Dodecyl Sulfate)", "smiles": "CCCCCCCCCCCCOS(=O)(=O)[O-]"},
        {"id": "laurate", "name": "Sodium Laurate (Soap)", "smiles": "CCCCCCCCCCCC(=O)[O-]"},
        {"id": "butanol1", "name": "1-Butanol (Linear)", "smiles": "CCCCO"},
        {"id": "tbutanol", "name": "tert-Butanol (Spherical)", "smiles": "CC(C)(C)O"},
        {"id": "maleic", "name": "Maleic Acid (cis)", "smiles": "O=C(O)/C=C\\C(=O)O"},
        {"id": "fumaric", "name": "Fumaric Acid (trans)", "smiles": "O=C(O)/C=C/C(=O)O"},
        {"id": "dppc", "name": "DPPC Membrane Lipid", "smiles": "CCCCCCCCCCCCCCCC(=O)OCC(COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCC"},
        {"id": "cholesterol", "name": "Cholesterol", "smiles": "CC(C)CCCC(C)C1CCC2C1(CCC3C2C(CC4C3(CCC(C4)O)C)C)C"},
    ]

    mol_data = {}

    for item in molecules:
        mol = Chem.MolFromSmiles(item["smiles"])
        mol = Chem.AddHs(mol)
        cid = AllChem.EmbedMolecule(mol, randomSeed=42)
        if cid < 0:
            AllChem.EmbedMolecule(mol, useRandomCoords=True)
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            pass
        
        conf = mol.GetConformer()
        adj = Chem.GetAdjacencyMatrix(mol, useBO=True)
        deg = np.diag(np.sum(adj, axis=1))
        laplacian = deg - adj
        
        evals, evecs = np.linalg.eigh(laplacian)
        idx = np.argsort(evals)
        evals = evals[idx]
        evecs = evecs[:, idx]
        
        lambda2 = float(evals[1])
        v2 = evecs[:, 1]
        
        # Orient v2 so that heteroatoms have positive sign (+)
        polar_atoms = [i for i, a in enumerate(mol.GetAtoms()) if a.GetSymbol() in ["O", "N", "S", "P"]]
        if len(polar_atoms) > 0 and np.mean(v2[polar_atoms]) < 0:
            v2 = -v2
            
        atoms = []
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            symbol = atom.GetSymbol()
            v2_val = float(v2[i])
            sign_type = "Head (+) Polar" if v2_val >= 0 else "Tail (-) Hydrophobic"
            color = "#ef4444" if v2_val >= 0 else "#3b82f6"  # Red for (+), Blue for (-)
            
            atoms.append({
                "idx": i,
                "symbol": symbol,
                "x": float(pos.x),
                "y": float(pos.y),
                "z": float(pos.z),
                "v2_val": round(v2_val, 4),
                "type": sign_type,
                "color": color
            })
            
        # Bonds
        bonds = []
        for bond in mol.GetBonds():
            bonds.append({
                "source": bond.GetBeginAtomIdx(),
                "target": bond.GetEndAtomIdx()
            })
            
        mol_data[item["id"]] = {
            "name": item["name"],
            "smiles": item["smiles"],
            "lambda2": round(lambda2, 4),
            "atoms": atoms,
            "bonds": bonds
        }

    return mol_data

def generate_interactive_html():
    data = build_fiedler_visualization_data()
    
    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>Molecular Fiedler Vector Sign Visualizer</title>
    <style>
        body {{ margin: 0; background: #0f172a; color: #f8fafc; font-family: system-ui, sans-serif; overflow: hidden; }}
        #panel {{ position: absolute; top: 20px; left: 20px; z-index: 10; background: rgba(15, 23, 42, 0.85); padding: 20px 24px; border-radius: 14px; border: 1px solid #334155; backdrop-filter: blur(10px); width: 360px; }}
        h1 {{ font-size: 1.2rem; margin: 0 0 10px 0; color: #38bdf8; }}
        select {{ width: 100%; padding: 8px 12px; background: #1e293b; border: 1px solid #475569; color: #fff; border-radius: 8px; font-size: 0.95rem; margin-bottom: 12px; cursor: pointer; }}
        .legend {{ font-size: 0.88rem; line-height: 1.6; color: #cbd5e1; }}
        .badge-red {{ display: inline-block; width: 12px; height: 12px; background: #ef4444; border-radius: 50%; margin-right: 6px; }}
        .badge-blue {{ display: inline-block; width: 12px; height: 12px; background: #3b82f6; border-radius: 50%; margin-right: 6px; }}
        .stat {{ margin-top: 12px; padding: 10px; background: rgba(51, 65, 85, 0.5); border-radius: 8px; font-size: 0.9rem; color: #4ade80; font-weight: bold; }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="panel">
        <h1>分子内 Fiedler 固有ベクトル 3D見える化</h1>
        <label for="molSelect"><b>分子を選択:</b></label>
        <select id="molSelect" onchange="loadMolecule(this.value)">
            <option value="sds">SDS (ラウリル硫酸塩)</option>
            <option value="laurate">Sodium Laurate (石鹸成分)</option>
            <option value="butanol1">1-Butanol (直鎖アルコール)</option>
            <option value="tbutanol">tert-Butanol (球状アルコール)</option>
            <option value="maleic">Maleic Acid (cis体)</option>
            <option value="fumaric">Fumaric Acid (trans体)</option>
            <option value="dppc">DPPC (細胞膜リン脂質)</option>
            <option value="cholesterol">Cholesterol (コレステロール)</option>
        </select>

        <div class="legend">
            <p><span class="badge-red"></span><b>赤色 (v₂ ≥ 0)</b>: 親水基・極性ドメイン (Head)</p>
            <p><span class="badge-blue"></span><b>青色 (v₂ &lt; 0)</b>: 疎水基・非極性アルキル鎖 (Tail)</p>
        </div>

        <div id="stats" class="stat">
            Fiedler 連結度 λ₂: 0.0164
        </div>
    </div>

    <script>
        const molData = {json.dumps(data)};
        
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0f172a);
        
        const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(0, 0, 30);
        
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);
        
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        
        scene.add(new THREE.AmbientLight(0xffffff, 0.8));
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
        dirLight.position.set(20, 20, 20);
        scene.add(dirLight);
        
        let currentGroup = new THREE.Group();
        scene.add(currentGroup);
        
        function loadMolecule(id) {{
            scene.remove(currentGroup);
            currentGroup = new THREE.Group();
            
            const data = molData[id];
            document.getElementById('stats').innerHTML = "代数的連結度 λ₂ = " + data.lambda2 + "<br>原子数: " + data.atoms.length;
            
            // Center calculation
            let cx = 0, cy = 0, cz = 0;
            data.atoms.forEach(a => {{ cx += a.x; cy += a.y; cz += a.z; }});
            cx /= data.atoms.length; cy /= data.atoms.length; cz /= data.atoms.length;
            
            // Add Atoms
            data.atoms.forEach(a => {{
                const size = a.symbol === "H" ? 0.35 : 0.7;
                const geom = new THREE.SphereGeometry(size, 32, 32);
                const mat = new THREE.MeshStandardMaterial({{ color: a.color, roughness: 0.3 }});
                const mesh = new THREE.Mesh(geom, mat);
                mesh.position.set(a.x - cx, a.y - cy, a.z - cz);
                currentGroup.add(mesh);
            }});
            
            // Add Bonds
            data.bonds.forEach(b => {{
                const a1 = data.atoms[b.source];
                const a2 = data.atoms[b.target];
                const p1 = new THREE.Vector3(a1.x - cx, a1.y - cy, a1.z - cz);
                const p2 = new THREE.Vector3(a2.x - cx, a2.y - cy, a2.z - cz);
                
                const dir = new THREE.Vector3().subVectors(p2, p1);
                const len = dir.length();
                const geom = new THREE.CylinderGeometry(0.15, 0.15, len, 16);
                const mat = new THREE.MeshStandardMaterial({{ color: 0x64748b }});
                const mesh = new THREE.Mesh(geom, mat);
                
                mesh.position.copy(p1).add(p2).multiplyScalar(0.5);
                mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
                currentGroup.add(mesh);
            }});
            
            scene.add(currentGroup);
        }}
        
        loadMolecule('sds');
        
        function animate() {{
            requestAnimationFrame(animate);
            if(currentGroup) currentGroup.rotation.y += 0.005;
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
        
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>"""
    
    with open("fiedler_molecular_sign_visualizer.html", "w") as f:
        f.write(html)
    print("Generated fiedler_molecular_sign_visualizer.html successfully!")

if __name__ == "__main__":
    generate_interactive_html()
