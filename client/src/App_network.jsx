import React, { useState, useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { Sparkles, Layers, Box, RotateCw, ZoomIn, Eye, HelpCircle } from 'lucide-react';

const BACKEND_URL = 'http://localhost:5007';

// Modern amino acid metadata
const AMINO_ACID_INFO = {
  G: { name: 'Glycine', type: 'Hydrophobic', color: '#64748b', hydrophobicity: 0.16 },
  Y: { name: 'Tyrosine', type: 'Polar/Aromatic', color: '#a855f7', hydrophobicity: 0.21 },
  D: { name: 'Aspartic Acid', type: 'Acidic (Charged)', color: '#ef4444', hydrophobicity: 0.02 },
  P: { name: 'Proline', type: 'Hydrophobic', color: '#f59e0b', hydrophobicity: 0.19 },
  E: { name: 'Glutamic Acid', type: 'Acidic (Charged)', color: '#ef4444', hydrophobicity: 0.04 },
  T: { name: 'Threonine', type: 'Polar', color: '#10b981', hydrophobicity: 0.11 },
  W: { name: 'Tryptophan', type: 'Hydrophobic/Aromatic', color: '#ec4899', hydrophobicity: 0.24 }
};

export default function AppNetwork() {
  const [sequence, setSequence] = useState('GYDPETGTWG');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [moleculeData, setMoleculeData] = useState(null);
  const [activeResidue, setActiveResidue] = useState(null);
  const [debugLogs, setDebugLogs] = useState([]);

  useEffect(() => {
    const handleWindowError = (event) => {
      setDebugLogs(prev => [...prev, `ERROR: ${event.message} at ${event.filename}:${event.lineno}`].slice(-30));
    };
    window.addEventListener('error', handleWindowError);

    const originalConsoleError = console.error;
    const originalConsoleWarn = console.warn;
    const originalConsoleLog = console.log;

    console.error = (...args) => {
      setDebugLogs(prev => [...prev, `ERR: ${args.map(a => typeof a === 'object' ? JSON.stringify(a) : a).join(' ')}`].slice(-30));
      originalConsoleError.apply(console, args);
    };
    console.warn = (...args) => {
      setDebugLogs(prev => [...prev, `WARN: ${args.map(a => typeof a === 'object' ? JSON.stringify(a) : a).join(' ')}`].slice(-30));
      originalConsoleWarn.apply(console, args);
    };
    console.log = (...args) => {
      setDebugLogs(prev => [...prev, `LOG: ${args.map(a => typeof a === 'object' ? JSON.stringify(a) : a).join(' ')}`].slice(-30));
      originalConsoleLog.apply(console, args);
    };

    return () => {
      window.removeEventListener('error', handleWindowError);
      console.error = originalConsoleError;
      console.warn = originalConsoleWarn;
      console.log = originalConsoleLog;
    };
  }, []);
  
  // Style configurations
  const [representation, setRepresentation] = useState('ball-stick'); // 'ball-stick' | 'vdw' | 'ribbon'
  const [colorTheme, setColorTheme] = useState('element'); // 'element' | 'hydrophobicity'
  const [syncCameras, setSyncCameras] = useState(true);
  const [showLabels, setShowLabels] = useState(false);
  const [scenesReady, setScenesReady] = useState(false);

  // Viewport DOM refs
  const mountFoldedRef = useRef(null);
  const mountUnfoldedRef = useRef(null);

  // Three.js instances for Folded Viewport
  const sceneFolded = useRef(null);
  const rendererFolded = useRef(null);
  const cameraFolded = useRef(null);
  const controlsFolded = useRef(null);
  const meshesFolded = useRef({ atoms: [], bonds: [], ribbon: null, labels: [] });

  // Three.js instances for Unfolded Viewport
  const sceneUnfolded = useRef(null);
  const rendererUnfolded = useRef(null);
  const cameraUnfolded = useRef(null);
  const controlsUnfolded = useRef(null);
  const meshesUnfolded = useRef({ atoms: [], bonds: [], ribbon: null, labels: [] });

  // Flags to prevent synchronization loop recursion
  const isSyncingRef = useRef(false);

  // 1. Fetch structure coordinates and topology from backend
  const fetchStructure = async (seqToLoad) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${BACKEND_URL}/api/structure`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequence: seqToLoad })
      });
      if (!response.ok) {
        throw new Error('Failed to retrieve peptide structure coordinates.');
      }
      const data = await response.json();
      if (data.status === 'error') {
        throw new Error(data.error);
      }
      
      // Print Fiedler comparison in the browser console
      console.log("\n========================================================");
      console.log(`  PEPTIDE NETWORK GRAPH COMPARISON (Sequence: ${data.sequence})`);
      console.log(`  Distance Threshold for Contacts: ${data.contactThreshold} Angstroms`);
      console.log(`--------------------------------------------------------`);
      console.log(`  UNFOLDED STATE:`);
      console.log(`    Fiedler Value (Algebraic Connectivity): ${data.unfoldedFiedler.toFixed(6)}`);
      console.log(`    Total Graph Edges (Bonds + Contacts):    ${data.unfoldedEdges}`);
      console.log(`    Graph Density:                           ${(data.unfoldedDensity * 100).toFixed(2)}%`);
      console.log(`  FOLDED STATE:`);
      console.log(`    Fiedler Value (Algebraic Connectivity): ${data.foldedFiedler.toFixed(6)}`);
      console.log(`    Total Graph Edges (Bonds + Contacts):    ${data.foldedEdges}`);
      console.log(`    Graph Density:                           ${(data.foldedDensity * 100).toFixed(2)}%`);
      console.log(`========================================================\n`);

      setMoleculeData(data);
    } catch (err) {
      setError(err.message);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStructure(sequence);
  }, []);

  // Helpers for building cylinders (bonds)
  const updateCylinder = (mesh, vStart, vEnd) => {
    const distance = vStart.distanceTo(vEnd);
    mesh.scale.set(1, distance, 1);
    
    // Position cylinder in the center of the two points
    const position = new THREE.Vector3().addVectors(vStart, vEnd).multiplyScalar(0.5);
    mesh.position.copy(position);
    
    // Rotate cylinder to align with the bond vector
    const direction = new THREE.Vector3().subVectors(vEnd, vStart).normalize();
    const up = new THREE.Vector3(0, 1, 0);
    const quaternion = new THREE.Quaternion().setFromUnitVectors(up, direction);
    mesh.setRotationFromQuaternion(quaternion);
  };

  // Helper to determine atom color
  const getAtomColor = (atom, theme) => {
    if (theme === 'hydrophobicity') {
      const resIdx = moleculeData?.atomResidues[atom.id];
      const resChar = sequence[resIdx] || 'G';
      const info = AMINO_ACID_INFO[resChar] || { color: '#64748b' };
      return new THREE.Color(info.color);
    } else {
      // Element theme
      if (atom.symbol === 'N') return new THREE.Color('#3b82f6'); // Nitrogen blue
      if (atom.symbol === 'O') return new THREE.Color('#ef4444'); // Oxygen red
      if (atom.symbol === 'S') return new THREE.Color('#eab308'); // Sulfur yellow
      return new THREE.Color('#475569'); // Carbon grey/graphite
    }
  };

  // Helper to determine atom size/radius
  const getAtomRadius = (atom, rep) => {
    if (rep === 'vdw') {
      if (atom.symbol === 'N') return 1.55 * 0.45;
      if (atom.symbol === 'O') return 1.52 * 0.45;
      if (atom.symbol === 'S') return 1.80 * 0.45;
      return 1.70 * 0.45; // Carbon
    } else {
      // Ball & stick size
      if (atom.symbol === 'N') return 0.28;
      if (atom.symbol === 'O') return 0.28;
      if (atom.symbol === 'S') return 0.35;
      return 0.25; // Carbon
    }
  };

  // 2. Rebuild Three.js Scene Contents
  const rebuildViewport = (scene, camera, renderer, mountNode, meshesRef, coords, label) => {
    console.log(`rebuildViewport [${label}] triggered. scene:`, !!scene, "coords length:", coords ? coords.length : 0);
    if (!scene || !camera || !renderer || !mountNode || !moleculeData || !coords) {
      console.warn(`rebuildViewport [${label}] skipped: missing required objects.`);
      return;
    }

    try {
      // Self-correct renderer size to handle initial mount layout delays
      const width = mountNode.clientWidth;
      const height = mountNode.clientHeight;
      console.log(`rebuildViewport [${label}] layout dimensions: ${width}x${height}`);
      if (width > 0 && height > 0) {
        renderer.setSize(width, height);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
      }

      // Clear old meshes
      meshesRef.current.atoms.forEach(m => scene.remove(m));
      meshesRef.current.atoms = [];
      meshesRef.current.bonds.forEach(m => scene.remove(m));
      meshesRef.current.bonds = [];
      meshesRef.current.labels.forEach(m => scene.remove(m));
      meshesRef.current.labels = [];
      if (meshesRef.current.ribbon) {
        scene.remove(meshesRef.current.ribbon);
        meshesRef.current.ribbon = null;
      }

      const { atoms, bonds } = moleculeData;

      // 1. Build Atoms
      if (representation !== 'ribbon') {
        atoms.forEach((atom) => {
          const radius = getAtomRadius(atom, representation);
          const color = getAtomColor(atom, colorTheme);

          const isHighlighted = activeResidue === null || moleculeData.atomResidues[atom.id] === activeResidue;
          const opacity = isHighlighted ? 1.0 : 0.18;

          const geo = new THREE.SphereGeometry(radius, 24, 24);
          const mat = new THREE.MeshStandardMaterial({
            color: color,
            roughness: 0.15,
            metalness: 0.1,
            transparent: true,
            opacity: opacity,
            emissive: isHighlighted && activeResidue !== null ? color : new THREE.Color('#000000'),
            emissiveIntensity: isHighlighted && activeResidue !== null ? 0.25 : 0.0
          });

          const mesh = new THREE.Mesh(geo, mat);
          mesh.position.fromArray(coords[atom.id]);
          scene.add(mesh);
          meshesRef.current.atoms.push(mesh);
        });
      }

      // 2. Build Bonds
      if (representation === 'ball-stick') {
        bonds.forEach((bond) => {
          const posA = new THREE.Vector3().fromArray(coords[bond.source]);
          const posB = new THREE.Vector3().fromArray(coords[bond.target]);

          const resA = moleculeData.atomResidues[bond.source];
          const resB = moleculeData.atomResidues[bond.target];
          const isHighlighted = activeResidue === null || (resA === activeResidue && resB === activeResidue);
          const opacity = isHighlighted ? 0.9 : 0.15;

          const geo = new THREE.CylinderGeometry(0.06, 0.06, 1, 12);
          const mat = new THREE.MeshStandardMaterial({
            color: 0x94a3b8,
            roughness: 0.5,
            metalness: 0.1,
            transparent: true,
            opacity: opacity
          });
          const mesh = new THREE.Mesh(geo, mat);
          updateCylinder(mesh, posA, posB);
          scene.add(mesh);
          meshesRef.current.bonds.push(mesh);
        });
      }

      // 3. Build Ribbon / Backbone Tube
      if (representation === 'ribbon' || representation === 'ball-stick') {
        // Find CA atoms in residue order
        const caCoords = [];
        atoms.forEach((atom) => {
          if (atom.name === 'CA' || (atom.element === 'C' && atom.name.includes('CA'))) {
            caCoords.push(new THREE.Vector3().fromArray(coords[atom.id]));
          }
        });

        if (caCoords.length > 1) {
          // Create curve passing through CA atoms
          const curve = new THREE.CatmullRomCurve3(caCoords);
          const tubeGeo = new THREE.TubeGeometry(curve, 64, 0.14, 12, false);
          
          // Ribbon material
          const mat = new THREE.MeshStandardMaterial({
            color: representation === 'ribbon' ? 0xa855f7 : 0xd8b4fe, // glowing purple
            roughness: 0.1,
            metalness: 0.2,
            transparent: true,
            opacity: activeResidue === null ? 0.95 : 0.35, // dim ribbon if highlighting specific atoms
            emissive: representation === 'ribbon' ? 0x6b21a8 : 0x000000,
            emissiveIntensity: 0.3
          });
          
          const mesh = new THREE.Mesh(tubeGeo, mat);
          scene.add(mesh);
          meshesRef.current.ribbon = mesh;
        }
      }
    } catch (err) {
      console.error(`Error in rebuildViewport [${label}]:`, err);
    }
  };

  // Rebuild both viewports when representation, colors, highlight, or structure changes
  useEffect(() => {
    if (!moleculeData || !scenesReady) return;
    rebuildViewport(sceneFolded.current, cameraFolded.current, rendererFolded.current, mountFoldedRef.current, meshesFolded, moleculeData.foldedCoords, "Folded");
    rebuildViewport(sceneUnfolded.current, cameraUnfolded.current, rendererUnfolded.current, mountUnfoldedRef.current, meshesUnfolded, moleculeData.unfoldedCoords, "Unfolded");
  }, [moleculeData, scenesReady, representation, colorTheme, activeResidue]);

  // 3. Initialize Viewport Three.js Engines
  const initEngine = (mountNode, isFoldedScene) => {
    if (!mountNode) return null;

    const width = mountNode.clientWidth || 300;
    const height = mountNode.clientHeight || 300;
    const aspect = width / height;

    // A. Setup Scene and Fog
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x0b0f19, 0.015);

    // B. Setup Camera
    const camera = new THREE.PerspectiveCamera(40, aspect, 0.1, 1000);
    camera.position.set(0, 10, 25);

    // C. Setup Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    // Clear any existing child elements to prevent duplicate canvases from StrictMode/HMR
    while (mountNode.firstChild) {
      mountNode.removeChild(mountNode.firstChild);
    }
    mountNode.appendChild(renderer.domElement);

    // D. Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight1.position.set(10, 15, 10);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xa78bfa, 0.4); // soft purple fill
    dirLight2.position.set(-10, -5, -10);
    scene.add(dirLight2);

    const pointLight = new THREE.PointLight(0xffffff, 0.5, 30);
    pointLight.position.set(0, 0, 5);
    scene.add(pointLight);

    // E. Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 45;
    controls.minDistance = 5;

    // F. Platform floor grid
    const gridHelper = new THREE.GridHelper(30, 30, 0x1e293b, 0x0f172a);
    gridHelper.position.y = -8;
    scene.add(gridHelper);

    return { scene, camera, renderer, controls };
  };

  useEffect(() => {
    const foldedEngine = initEngine(mountFoldedRef.current, true);
    if (foldedEngine) {
      sceneFolded.current = foldedEngine.scene;
      cameraFolded.current = foldedEngine.camera;
      rendererFolded.current = foldedEngine.renderer;
      controlsFolded.current = foldedEngine.controls;
    }

    const unfoldedEngine = initEngine(mountUnfoldedRef.current, false);
    if (unfoldedEngine) {
      sceneUnfolded.current = unfoldedEngine.scene;
      cameraUnfolded.current = unfoldedEngine.camera;
      rendererUnfolded.current = unfoldedEngine.renderer;
      controlsUnfolded.current = unfoldedEngine.controls;
    }

    // Set scenes as ready
    setScenesReady(true);

    // Render loop
    let animationId;
    const animate = () => {
      animationId = requestAnimationFrame(animate);

      if (controlsFolded.current) controlsFolded.current.update();
      if (controlsUnfolded.current) controlsUnfolded.current.update();

      if (rendererFolded.current && sceneFolded.current && cameraFolded.current) {
        rendererFolded.current.render(sceneFolded.current, cameraFolded.current);
      }
      if (rendererUnfolded.current && sceneUnfolded.current && cameraUnfolded.current) {
        rendererUnfolded.current.render(sceneUnfolded.current, cameraUnfolded.current);
      }
    };
    animate();

    // Resize Handler
    const handleResize = () => {
      [
        { mount: mountFoldedRef.current, cam: cameraFolded.current, ren: rendererFolded.current },
        { mount: mountUnfoldedRef.current, cam: cameraUnfolded.current, ren: rendererUnfolded.current }
      ].forEach(({ mount, cam, ren }) => {
        if (!mount || !cam || !ren) return;
        const w = mount.clientWidth || 300;
        const h = mount.clientHeight || 300;
        cam.aspect = w / h;
        cam.updateProjectionMatrix();
        ren.setSize(w, h);
      });
    };
    window.addEventListener('resize', handleResize);

    // Clean up
    return () => {
      setScenesReady(false);
      cancelAnimationFrame(animationId);
      window.removeEventListener('resize', handleResize);

      if (rendererFolded.current && mountFoldedRef.current) {
        mountFoldedRef.current.removeChild(rendererFolded.current.domElement);
      }
      if (rendererUnfolded.current && mountUnfoldedRef.current) {
        mountUnfoldedRef.current.removeChild(rendererUnfolded.current.domElement);
      }
    };
  }, []);

  // 4. Synchronize camera movement between viewports in real-time
  useEffect(() => {
    const ctrlF = controlsFolded.current;
    const ctrlU = controlsUnfolded.current;
    const camF = cameraFolded.current;
    const camU = cameraUnfolded.current;

    if (!ctrlF || !ctrlU || !camF || !camU) return;

    const syncToU = () => {
      if (!syncCameras || isSyncingRef.current) return;
      isSyncingRef.current = true;
      camU.position.copy(camF.position);
      camU.quaternion.copy(camF.quaternion);
      camU.zoom = camF.zoom;
      ctrlU.target.copy(ctrlF.target);
      isSyncingRef.current = false;
    };

    const syncToF = () => {
      if (!syncCameras || isSyncingRef.current) return;
      isSyncingRef.current = true;
      camF.position.copy(camU.position);
      camF.quaternion.copy(camU.quaternion);
      camF.zoom = camU.zoom;
      ctrlF.target.copy(ctrlU.target);
      isSyncingRef.current = false;
    };

    if (syncCameras) {
      ctrlF.addEventListener('change', syncToU);
      ctrlU.addEventListener('change', syncToF);
      
      // Perform initial alignment
      syncToU();
    }

    return () => {
      ctrlF.removeEventListener('change', syncToU);
      ctrlU.removeEventListener('change', syncToF);
    };
  }, [syncCameras]);

  const handleSequenceChange = (e) => {
    setSequence(e.target.value.toUpperCase().replace(/[^A-Z]/g, ''));
  };

  const handleSubmitSequence = (e) => {
    e.preventDefault();
    if (sequence.trim()) {
      fetchStructure(sequence);
    }
  };

  return (
    <div className="render-app-container">
      {/* 1. Header Bar */}
      <header className="app-header">
        <div className="header-brand">
          <Sparkles className="brand-icon" />
          <div className="brand-text">
            <h1>Peptide Conformation Analyzer</h1>
            <p>High-Fidelity 3D Validation & Graph Topology Viewer</p>
          </div>
        </div>
        <form onSubmit={handleSubmitSequence} className="sequence-form">
          <input
            type="text"
            value={sequence}
            onChange={handleSequenceChange}
            placeholder="Enter sequence (e.g. GYDPETGTWG)"
            className="sequence-input"
          />
          <button type="submit" className="sequence-btn">
            Load Conformation
          </button>
        </form>
      </header>

      {/* 2. Main Double-Viewport Workspace */}
      <main className="workspace-main">
        {loading && (
          <div className="loading-overlay">
            <div className="loading-card">
              <div className="spinner"></div>
              <h3>Analyzing Structure & Network</h3>
              <p>Calculating coordinates, mapping topology, and running graph Laplacian eigensolvers...</p>
            </div>
          </div>
        )}

        {error && (
          <div className="error-overlay">
            <div className="error-card">
              <h3>Simulation Error</h3>
              <p>{error}</p>
              <button onClick={() => fetchStructure(sequence)} className="retry-btn">
                Retry Load
              </button>
            </div>
          </div>
        )}

        <div className="viewports-split">
          {/* A. Folded Conformation Viewport */}
          <div className="viewport-container card-glass">
            <div className="viewport-label">
              <Layers className="label-icon" />
              <span>Folded Conformation (1UAO Validation PDB)</span>
            </div>
            <div className="canvas-mount" ref={mountFoldedRef} />
          </div>

          {/* B. Unfolded Conformation Viewport */}
          <div className="viewport-container card-glass">
            <div className="viewport-label">
              <Box className="label-icon" />
              <span>Unfolded Conformation (Extended State)</span>
            </div>
            <div className="canvas-mount" ref={mountUnfoldedRef} />
          </div>
        </div>
      </main>

      {/* 3. Control Center Overlay */}
      <section className="control-overlay-panel card-glass">
        <div className="panel-section">
          <h4>Representation</h4>
          <div className="btn-group">
            <button
              onClick={() => setRepresentation('ball-stick')}
              className={representation === 'ball-stick' ? 'active' : ''}
            >
              Ball & Stick
            </button>
            <button
              onClick={() => setRepresentation('vdw')}
              className={representation === 'vdw' ? 'active' : ''}
            >
              Space Fill (VDW)
            </button>
            <button
              onClick={() => setRepresentation('ribbon')}
              className={representation === 'ribbon' ? 'active' : ''}
            >
              Ribbon Tube
            </button>
          </div>
        </div>

        <div className="panel-section">
          <h4>Color Theme</h4>
          <div className="btn-group">
            <button
              onClick={() => setColorTheme('element')}
              className={colorTheme === 'element' ? 'active' : ''}
            >
              Element
            </button>
            <button
              onClick={() => setColorTheme('hydrophobicity')}
              className={colorTheme === 'hydrophobicity' ? 'active' : ''}
            >
              Residue Type
            </button>
          </div>
        </div>

        <div className="panel-section flex-row-item">
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={syncCameras}
              onChange={(e) => setSyncCameras(e.target.checked)}
            />
            <span>Sync Viewport Cameras</span>
          </label>
        </div>

        {/* New Graph network statistics panel */}
        {moleculeData && (
          <div className="panel-section network-stats">
            <h4>Graph Network Metrics</h4>
            <div className="stats-grid">
              <div className="stats-col">
                <span className="stats-label">Folded State</span>
                <div className="stats-val">λ₂: {moleculeData.foldedFiedler !== undefined ? moleculeData.foldedFiedler.toFixed(5) : 'N/A'}</div>
                <div className="stats-sub">Edges: {moleculeData.foldedEdges}</div>
                <div className="stats-sub">Density: {moleculeData.foldedDensity !== undefined ? (moleculeData.foldedDensity * 100).toFixed(1) + '%' : 'N/A'}</div>
              </div>
              <div className="stats-col border-left">
                <span className="stats-label">Unfolded State</span>
                <div className="stats-val">λ₂: {moleculeData.unfoldedFiedler !== undefined ? moleculeData.unfoldedFiedler.toFixed(5) : 'N/A'}</div>
                <div className="stats-sub">Edges: {moleculeData.unfoldedEdges}</div>
                <div className="stats-sub">Density: {moleculeData.unfoldedDensity !== undefined ? (moleculeData.unfoldedDensity * 100).toFixed(1) + '%' : 'N/A'}</div>
              </div>
            </div>
            <div className="network-explanation">
              The Fiedler value (λ₂) represents algebraic connectivity. The compact folded conformation has higher connectivity due to non-covalent contacts (d &lt; {moleculeData.contactThreshold}Å) acting as network shortcuts, raising λ₂ compared to the unfolded linear state.
            </div>
          </div>
        )}
      </section>

      {/* 4. Sequence Inspector & Atom Highlighting (At the bottom) */}
      <footer className="sequence-inspector card-glass">
        <div className="sequence-label">
          <Eye className="seq-icon" />
          <span>Interactive Sequence Mapper</span>
          <span className="tooltip-hint">(Hover or click residue to highlight atoms)</span>
        </div>
        <div className="amino-acid-row">
          {sequence.split('').map((char, index) => {
            const info = AMINO_ACID_INFO[char] || { name: 'Unknown', type: 'Neutral', color: '#64748b', hydrophobicity: 0.0 };
            const isActive = activeResidue === index;
            return (
              <div
                key={index}
                className={`amino-card ${isActive ? 'active' : ''}`}
                style={{ '--card-color': info.color }}
                onMouseEnter={() => setActiveResidue(index)}
                onMouseLeave={() => setActiveResidue(null)}
                onClick={() => setActiveResidue(activeResidue === index ? null : index)}
              >
                <span className="residue-idx">{index + 1}</span>
                <span className="residue-char">{char}</span>
                <div className="residue-tooltip card-glass">
                  <strong>{info.name}</strong>
                  <span>Type: {info.type}</span>
                  <span>Hydrophobicity: {info.hydrophobicity}</span>
                </div>
              </div>
            );
          })}
        </div>
      </footer>

      {/* Floating Debug Panel */}
      <div className="debug-panel card-glass">
        <h4 className="debug-title">System Diagnostics</h4>
        <div className="debug-console">
          {debugLogs.map((log, i) => (
            <div key={i} className={`debug-line ${log.startsWith('ERR') ? 'error-line' : log.startsWith('WARN') ? 'warn-line' : ''}`}>
              {log}
            </div>
          ))}
          {debugLogs.length === 0 && <div className="debug-line text-muted">No logs recorded.</div>}
        </div>
      </div>

      {/* Scoped Visual Styles */}
      <style>{`
        .render-app-container {
          width: 100vw;
          height: 100vh;
          background: radial-gradient(circle at top left, #0f172a, #020617);
          color: #f8fafc;
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          display: flex;
          flex-direction: column;
          overflow: hidden;
          position: relative;
        }

        .app-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 1.25rem 2rem;
          background: rgba(15, 23, 42, 0.45);
          backdrop-filter: blur(12px);
          border-bottom: 1px solid rgba(255, 255, 255, 0.08);
          z-index: 10;
        }

        .header-brand {
          display: flex;
          align-items: center;
          gap: 0.75rem;
        }

        .brand-icon {
          color: #a78bfa;
          width: 2rem;
          height: 2rem;
          animation: float 3s ease-in-out infinite;
        }

        .brand-text h1 {
          font-size: 1.25rem;
          font-weight: 700;
          letter-spacing: -0.025em;
          margin: 0;
          background: linear-gradient(135deg, #f8fafc, #c084fc);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        .brand-text p {
          font-size: 0.75rem;
          color: #94a3b8;
          margin: 2px 0 0 0;
        }

        .sequence-form {
          display: flex;
          gap: 0.5rem;
        }

        .sequence-input {
          background: rgba(15, 23, 42, 0.85);
          border: 1px solid rgba(255, 255, 255, 0.12);
          border-radius: 8px;
          padding: 0.5rem 1rem;
          color: #fff;
          font-size: 0.875rem;
          font-family: monospace;
          min-width: 240px;
          outline: none;
          transition: border-color 0.2s, box-shadow 0.2s;
        }

        .sequence-input:focus {
          border-color: #a78bfa;
          box-shadow: 0 0 0 3px rgba(167, 139, 250, 0.2);
        }

        .sequence-btn {
          background: linear-gradient(135deg, #a78bfa, #7c3aed);
          border: none;
          border-radius: 8px;
          padding: 0.5rem 1.25rem;
          color: #fff;
          font-size: 0.875rem;
          font-weight: 600;
          cursor: pointer;
          transition: filter 0.2s;
        }

        .sequence-btn:hover {
          filter: brightness(1.1);
        }

        .workspace-main {
          flex: 1;
          position: relative;
          padding: 1.5rem 2rem;
          display: flex;
          flex-direction: column;
          overflow: hidden;
        }

        .viewports-split {
          flex: 1;
          display: flex;
          gap: 1.5rem;
          height: 100%;
        }

        .viewport-container {
          flex: 1;
          display: flex;
          flex-direction: column;
          overflow: hidden;
          position: relative;
        }

        .viewport-label {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.75rem 1.25rem;
          background: rgba(255, 255, 255, 0.03);
          border-bottom: 1px solid rgba(255, 255, 255, 0.05);
          font-size: 0.75rem;
          font-weight: 600;
          letter-spacing: 0.05em;
          text-transform: uppercase;
          color: #cbd5e1;
        }

        .label-icon {
          width: 0.875rem;
          height: 0.875rem;
          color: #a78bfa;
        }

        .canvas-mount {
          flex: 1;
          width: 100%;
          height: 100%;
          cursor: grab;
        }

        .canvas-mount:active {
          cursor: grabbing;
        }

        .card-glass {
          background: rgba(15, 23, 42, 0.45);
          backdrop-filter: blur(16px);
          -webkit-backdrop-filter: blur(16px);
          border: 1px solid rgba(255, 255, 255, 0.08);
          border-radius: 12px;
          box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        }

        /* 3. Control panel overlay */
        .control-overlay-panel {
          position: absolute;
          top: 7rem;
          left: 3.5rem;
          width: 290px;
          padding: 1.25rem;
          display: flex;
          flex-direction: column;
          gap: 1rem;
          z-index: 8;
        }

        .panel-section h4 {
          font-size: 0.65rem;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.075em;
          color: #94a3b8;
          margin: 0 0 0.5rem 0;
        }

        .btn-group {
          display: flex;
          background: rgba(15, 23, 42, 0.75);
          border: 1px solid rgba(255, 255, 255, 0.06);
          border-radius: 6px;
          padding: 2px;
        }

        .btn-group button {
          flex: 1;
          background: transparent;
          border: none;
          border-radius: 4px;
          padding: 0.35rem 0.5rem;
          color: #94a3b8;
          font-size: 0.7rem;
          font-weight: 600;
          cursor: pointer;
          transition: background 0.15s, color 0.15s;
        }

        .btn-group button.active {
          background: rgba(255, 255, 255, 0.08);
          color: #fff;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .toggle-label {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          cursor: pointer;
          font-size: 0.75rem;
          font-weight: 500;
          color: #cbd5e1;
        }

        .toggle-label input {
          accent-color: #a78bfa;
        }

        /* Network Stats Styling */
        .network-stats {
          border-top: 1px solid rgba(255, 255, 255, 0.08);
          padding-top: 0.75rem;
          margin-top: 0.5rem;
        }
        
        .stats-grid {
          display: flex;
          margin-top: 0.5rem;
          gap: 0.75rem;
        }
        
        .stats-col {
          flex: 1;
          display: flex;
          flex-direction: column;
          gap: 2px;
        }
        
        .stats-col.border-left {
          border-left: 1px solid rgba(255, 255, 255, 0.08);
          padding-left: 0.75rem;
        }
        
        .stats-label {
          font-size: 0.65rem;
          color: #94a3b8;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }
        
        .stats-val {
          font-size: 0.85rem;
          font-weight: 700;
          color: #a78bfa;
        }
        
        .stats-sub {
          font-size: 0.65rem;
          color: #64748b;
        }
        
        .network-explanation {
          margin-top: 0.5rem;
          font-size: 0.6rem;
          color: #64748b;
          line-height: 1.35;
        }

        /* 4. Sequence Inspector */
        .sequence-inspector {
          margin: 0 2rem 2rem 2rem;
          padding: 1.25rem;
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
          z-index: 10;
        }

        .sequence-label {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          font-size: 0.75rem;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          color: #cbd5e1;
        }

        .seq-icon {
          width: 0.9rem;
          height: 0.9rem;
          color: #a78bfa;
        }

        .tooltip-hint {
          font-size: 0.7rem;
          font-weight: 400;
          text-transform: none;
          color: #64748b;
          letter-spacing: 0;
        }

        .amino-acid-row {
          display: flex;
          gap: 0.5rem;
        }

        .amino-card {
          flex: 1;
          height: 60px;
          border-radius: 8px;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.05);
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          position: relative;
          transition: background 0.2s, border-color 0.2s, transform 0.2s;
        }

        .amino-card:hover, .amino-card.active {
          background: rgba(255, 255, 255, 0.06);
          border-color: var(--card-color);
          transform: translateY(-2px);
          box-shadow: 0 0 10px rgba(167, 139, 250, 0.15);
        }

        .residue-idx {
          font-size: 0.65rem;
          color: #64748b;
          margin-bottom: 2px;
        }

        .residue-char {
          font-size: 1.15rem;
          font-weight: 700;
          color: var(--card-color);
        }

        .residue-tooltip {
          position: absolute;
          bottom: 75px;
          left: 50%;
          transform: translateX(-50%);
          width: 180px;
          padding: 0.75rem;
          display: none;
          flex-direction: column;
          gap: 3px;
          z-index: 100;
          pointer-events: none;
          font-size: 0.7rem;
        }

        .amino-card:hover .residue-tooltip {
          display: flex;
        }

        .residue-tooltip strong {
          color: #fff;
          font-size: 0.75rem;
          margin-bottom: 2px;
        }

        /* Overlays & spinner */
        .loading-overlay, .error-overlay {
          position: absolute;
          inset: 0;
          background: rgba(2, 6, 23, 0.75);
          backdrop-filter: blur(8px);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 100;
        }

        .loading-card, .error-card {
          background: rgba(15, 23, 42, 0.85);
          border: 1px solid rgba(255, 255, 255, 0.12);
          border-radius: 12px;
          padding: 2.5rem;
          text-align: center;
          max-width: 400px;
          box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
        }

        .spinner {
          width: 40px;
          height: 40px;
          border: 3px solid rgba(167, 139, 250, 0.2);
          border-top-color: #a78bfa;
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
          margin: 0 auto 1.25rem auto;
        }

        .retry-btn {
          background: #ef4444;
          border: none;
          color: white;
          padding: 0.5rem 1.25rem;
          border-radius: 6px;
          cursor: pointer;
          font-weight: 600;
          margin-top: 1rem;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        @keyframes float {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-4px); }
        }

        .debug-panel {
          position: absolute;
          bottom: 1.5rem;
          right: 1.5rem;
          width: 340px;
          height: 180px;
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
          padding: 0.75rem;
          z-index: 10000;
          font-family: monospace;
          font-size: 0.65rem;
          background: rgba(2, 6, 23, 0.9) !important;
        }

        .debug-title {
          font-size: 0.7rem;
          font-weight: 700;
          color: #a78bfa;
          margin: 0;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
          padding-bottom: 4px;
          text-transform: uppercase;
          letter-spacing: 0.05em;
        }

        .debug-console {
          flex: 1;
          overflow-y: auto;
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .debug-line {
          white-space: pre-wrap;
          word-break: break-all;
          color: #cbd5e1;
        }

        .error-line {
          color: #f87171 !important;
        }

        .warn-line {
          color: #fbbf24 !important;
        }

        .text-muted {
          color: #64748b;
        }
      `}</style>
    </div>
  );
}
