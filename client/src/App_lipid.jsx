import React, { useState, useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { 
  Play, Pause, RefreshCw, Layers, Sliders, Dna, FileText, ChevronLeft, ChevronRight, Activity
} from 'lucide-react';

const BACKEND_URL = 'http://localhost:5004';

// Helper to calculate cylinder orientation
const updateCylinder = (mesh, posA, posB) => {
  const mid = new THREE.Vector3().addVectors(posA, posB).multiplyScalar(0.5);
  mesh.position.copy(mid);
  const dir = new THREE.Vector3().subVectors(posB, posA);
  const len = dir.length();
  mesh.scale.set(1, len, 1);
  dir.normalize();
  const quaternion = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
  mesh.setRotationFromQuaternion(quaternion);
};

export default function AppLipid() {
  // Input sequence defaults to DPPC phospholipid SMILES
  const [sequence, setSequence] = useState('CCCCCCCCCCCCCCCC(=O)OCC(COP(=O)([O-])OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCC');
  const [moleculeCount, setMoleculeCount] = useState(3);
  const [sessionId, setSessionId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Panel state
  const [showSidebar, setShowSidebar] = useState(true);
  const [showLogs, setShowLogs] = useState(true);

  // Simulation parameter states
  const [temperature, setTemperature] = useState(0.20);
  const [isPlaying, setIsPlaying] = useState(true);
  const [explicitWaterEnabled, setExplicitWaterEnabled] = useState(true);
  const [waterWaterStrength, setWaterWaterStrength] = useState(0.65);
  const [waterPeptideStrength, setWaterPeptideStrength] = useState(0.75);
  const [bulkSolventStrength, setBulkSolventStrength] = useState(0.45);
  const [watersPerAtom, setWatersPerAtom] = useState(3);
  const [kCapture, setKCapture] = useState(0.70);
  const [entropyBias, setEntropyBias] = useState(1.2);
  const [globalWaterBeltContraction, setGlobalWaterBeltContraction] = useState(0.0);
  const [autoContractEnabled, setAutoContractEnabled] = useState(true);
  const [initialLayout, setInitialLayout] = useState('dispersed');

  // Dynamic calculated stats
  const [gyrationRadius, setGyrationRadius] = useState(0.0);
  const [hydrophobicExposure, setHydrophobicExposure] = useState(0.0);
  const [informationCaptureIndex, setInformationCaptureIndex] = useState(0.0);
  const [averageWaterEntropy, setAverageWaterEntropy] = useState(0.6);
  const [hydrophobicContacts, setHydrophobicContacts] = useState(0);
  const [atomCount, setAtomCount] = useState(0);
  const [showWaterVisuals, setShowWaterVisuals] = useState(true);
  const [residueLabels, setResidueLabels] = useState([]);

  // Terminal log lines
  const [logs, setLogs] = useState([]);

  // Three.js DOM container
  const mountRef = useRef(null);

  // References to keep Three.js updated in real-time
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const controlsRef = useRef(null);

  // Tracking meshes
  const atomMeshesRef = useRef([]);
  const bondMeshesRef = useRef([]);
  const waterMeshesRef = useRef([]);
  const hbondLinesRef = useRef(null);
  const waterBondsGroupRef = useRef(null);
  const activePathsGroupRef = useRef(null);

  // SSE streaming buffers
  const latestCoordsRef = useRef(null);
  const latestWaterCoordsRef = useRef(null);
  const latestWatersRef = useRef(null);
  const latestWaterBondsRef = useRef([]);
  const latestWaterEntropiesRef = useRef([]);
  const latestActiveInformationPathsRef = useRef([]);

  // Synchronize state values to refs for the animation loop
  const isPlayingRef = useRef(isPlaying);
  const explicitWaterEnabledRef = useRef(explicitWaterEnabled);
  const showWaterVisualsRef = useRef(showWaterVisuals);
  const atomResiduesRef = useRef({});
  const moleculeDataRef = useRef(null);
  
  // React state synchronization to refs
  useEffect(() => { isPlayingRef.current = isPlaying; }, [isPlaying]);
  useEffect(() => { explicitWaterEnabledRef.current = explicitWaterEnabled; }, [explicitWaterEnabled]);
  useEffect(() => { showWaterVisualsRef.current = showWaterVisuals; }, [showWaterVisuals]);

  // Initialize peptide and load topology from backend
  const handleInitialize = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${BACKEND_URL}/api/initialize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequence, count: moleculeCount, initialLayout })
      });
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || 'Failed to initialize system.');
      }
      const data = await response.json();
      setSessionId(data.sessionId);
      setAtomCount(data.atoms.length);
      setResidueLabels(data.residueLabels || []);
      
      const residueMap = {};
      data.atoms.forEach(atom => {
        residueMap[atom.id] = data.atomResidues[atom.id] || 0;
      });
      atomResiduesRef.current = residueMap;
      moleculeDataRef.current = data;

      // Render Three.js meshes
      rebuildScene(data.atoms, data.bonds, data.initialCoords, data);
      
      setLogs([{ text: `System: Initialized ${moleculeCount} copy(ies) of lipid structure.`, type: 'system' }]);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    handleInitialize();
    return () => {
      if (rendererRef.current) {
        rendererRef.current.dispose();
      }
    };
  }, []);

  // Update control parameter to backend
  const syncControls = async (paramsObj) => {
    if (!sessionId) return;
    try {
      await fetch(`${BACKEND_URL}/api/control`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(paramsObj)
      });
    } catch (err) {
      console.error("Control synchronization error:", err);
    }
  };

  const updateParameter = (type, val) => {
    if (type === 'temperature') {
      setTemperature(val);
      syncControls({ temperature: val });
    } else if (type === 'waterWaterStrength') {
      setWaterWaterStrength(val);
      syncControls({ waterWaterStrength: val });
    } else if (type === 'waterPeptideStrength') {
      setWaterPeptideStrength(val);
      syncControls({ waterPeptideStrength: val });
    } else if (type === 'bulkSolventStrength') {
      setBulkSolventStrength(val);
      syncControls({ bulkSolventStrength: val });
    } else if (type === 'kCapture') {
      setKCapture(val);
      syncControls({ kCapture: val });
    } else if (type === 'entropyBias') {
      setEntropyBias(val);
      syncControls({ entropyBias: val });
    }
  };

  const togglePlayback = () => {
    const next = !isPlaying;
    setIsPlaying(next);
    syncControls({ isPlaying: next });
  };

  const toggleWaterBelt = () => {
    const next = !explicitWaterEnabled;
    setExplicitWaterEnabled(next);
    syncControls({ explicitWaterEnabled: next });
  };

  // Build/Rebuild Three.js scene meshes
  const rebuildScene = (atoms, bonds, initialCoords, data) => {
    try {
      const scene = sceneRef.current;
      if (!scene) return;

      // Clean old meshes
      atomMeshesRef.current.forEach(m => scene.remove(m));
      bondMeshesRef.current.forEach(item => scene.remove(item.mesh));
      waterMeshesRef.current.forEach(w => {
        scene.remove(w.oxygen);
        scene.remove(w.hydrogen1);
        scene.remove(w.hydrogen2);
        scene.remove(w.bond1);
        scene.remove(w.bond2);
      });

      atomMeshesRef.current = [];
      bondMeshesRef.current = [];
      waterMeshesRef.current = [];

      // 1. Build Atom Spheres
      atoms.forEach((atom, idx) => {
        const coord = initialCoords[idx];
        if (!coord) {
          throw new Error(`initialCoords[${idx}] is undefined (total atoms: ${atoms.length})`);
        }
        
        let color = 0x9ca3af; // Grey for carbon
        if (atom.symbol === 'O') color = 0xef4444; // Red
        else if (atom.symbol === 'N') color = 0x3b82f6; // Blue
        else if (atom.symbol === 'P') color = 0xeab308; // Yellow/Orange for phosphorus
        else if (atom.symbol === 'S') color = 0xeab308; // Yellow
        
        // Highlight hydrophobic tails in orange
        if (atom.hydrophobicity > 0.15) {
          color = 0xf97316; // Orange
        }

        const geom = new THREE.SphereGeometry(atom.radius * 0.45, 20, 20);
        const mat = new THREE.MeshPhongMaterial({
          color: color,
          shininess: 70,
          specular: 0x111111
        });
        const mesh = new THREE.Mesh(geom, mat);
        mesh.position.set(coord[0], coord[1], coord[2]);
        scene.add(mesh);
        atomMeshesRef.current.push(mesh);
      });

      // 2. Build Bonds
      bonds.forEach(bond => {
        if (bond.source === undefined || bond.target === undefined) {
          throw new Error(`bond.source or bond.target is undefined`);
        }
        const posA = new THREE.Vector3(...initialCoords[bond.source]);
        const posB = new THREE.Vector3(...initialCoords[bond.target]);
        
        const thickness = bond.is_rotatable ? 0.08 : 0.05;
        const color = bond.is_rotatable ? 0x10b981 : 0xd1d5db; // Green for rotatable, grey for rigid
        
        const geom = new THREE.CylinderGeometry(thickness, thickness, 1, 12);
        const mat = new THREE.MeshPhongMaterial({ color: color });
        const mesh = new THREE.Mesh(geom, mat);
        
        updateCylinder(mesh, posA, posB);
        
        scene.add(mesh);
        bondMeshesRef.current.push({
          mesh,
          source: bond.source,
          target: bond.target
        });
      });

      // 3. Build Water Belt Molecules
      const initialWaterCoords = data.waterCoords || [];
      initialWaterCoords.forEach((coord, idx) => {
        if (!coord || coord.length < 3) {
          throw new Error(`initialWaterCoords[${idx}] is invalid`);
        }
        
        const geomO = new THREE.SphereGeometry(0.20, 10, 10);
        const geomH = new THREE.SphereGeometry(0.12, 8, 8);
        
        const matO = new THREE.MeshPhongMaterial({
          color: 0x38bdf8,
          shininess: 50,
          specular: 0x050505
        });
        const matH = new THREE.MeshPhongMaterial({
          color: 0xf8fafc,
          shininess: 15
        });
        const matBond = new THREE.MeshBasicMaterial({ color: 0x475569 });
        
        const oMesh = new THREE.Mesh(geomO, matO);
        const hMesh1 = new THREE.Mesh(geomH, matH);
        const hMesh2 = new THREE.Mesh(geomH, matH);
        
        const bGeom1 = new THREE.CylinderGeometry(0.03, 0.03, 0.96, 4);
        const bGeom2 = new THREE.CylinderGeometry(0.03, 0.03, 0.96, 4);
        const bMesh1 = new THREE.Mesh(bGeom1, matBond);
        const bMesh2 = new THREE.Mesh(bGeom2, matBond);
        
        oMesh.position.set(...coord[0]);
        hMesh1.position.set(...coord[1]);
        hMesh2.position.set(...coord[2]);
        
        updateCylinder(bMesh1, new THREE.Vector3(...coord[0]), new THREE.Vector3(...coord[1]));
        updateCylinder(bMesh2, new THREE.Vector3(...coord[0]), new THREE.Vector3(...coord[2]));
        
        scene.add(oMesh);
        scene.add(hMesh1);
        scene.add(hMesh2);
        scene.add(bMesh1);
        scene.add(bMesh2);
        
        waterMeshesRef.current.push({
          oxygen: oMesh,
          hydrogen1: hMesh1,
          hydrogen2: hMesh2,
          bond1: bMesh1,
          bond2: bMesh2
        });
      });

      // Reset buffers
      latestCoordsRef.current = initialCoords;
      latestWaterCoordsRef.current = initialWaterCoords;
      latestWaterBondsRef.current = [];
      latestWaterEntropiesRef.current = [];
      latestActiveInformationPathsRef.current = [];

      // Camera focus
      const box = new THREE.Box3();
      initialCoords.forEach(c => box.expandByPoint(new THREE.Vector3(...c)));
      const center = new THREE.Vector3();
      box.getCenter(center);
      if (controlsRef.current) {
        controlsRef.current.target.copy(center);
      }
    } catch (err) {
      console.error("Error in rebuildScene:", err);
      setLogs(prev => [...prev, { text: `rebuildScene error: ${err.message}`, type: 'error' }]);
    }
  };

  // Setup Three.js scene canvas
  useEffect(() => {
    if (!mountRef.current) return;
    
    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x070b19); // Deep space blue-black
    sceneRef.current = scene;
    
    // Camera
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 150);
    camera.position.set(0, 8, 30);
    cameraRef.current = camera;
    
    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    const initW = mountRef.current.clientWidth || 300;
    const initH = mountRef.current.clientHeight || 300;
    renderer.setSize(initW, initH);
    if (mountRef.current.clientHeight > 0) {
      camera.aspect = initW / initH;
      camera.updateProjectionMatrix();
    }
    renderer.setPixelRatio(window.devicePixelRatio);
    while (mountRef.current.firstChild) {
      mountRef.current.removeChild(mountRef.current.firstChild);
    }
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;
    
    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controlsRef.current = controls;
    
    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.40);
    scene.add(ambientLight);
    
    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.85);
    dirLight1.position.set(15, 20, 15);
    scene.add(dirLight1);
    
    const dirLight2 = new THREE.DirectionalLight(0x0ea5e9, 0.25); // Cyan fill
    dirLight2.position.set(-15, -5, -15);
    scene.add(dirLight2);
    
    // Add lines groups
    const wHBGroup = new THREE.Group();
    scene.add(wHBGroup);
    waterBondsGroupRef.current = wHBGroup;

    const activePaths = new THREE.Group();
    scene.add(activePaths);
    activePathsGroupRef.current = activePaths;

    // Resize observer
    const resizeObserver = new ResizeObserver((entries) => {
      for (let entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          camera.aspect = width / height;
          camera.updateProjectionMatrix();
          renderer.setSize(width, height);
        }
      }
    });
    resizeObserver.observe(mountRef.current);
    
    // Animation Loop
    let animId;
    const animate = () => {
      try {
        animId = requestAnimationFrame(animate);
        controls.update();
        
        const coords = latestCoordsRef.current;
        if (coords) {
          // 1. Update atoms
          coords.forEach((coord, idx) => {
            const mesh = atomMeshesRef.current[idx];
            if (mesh) {
              mesh.position.set(coord[0], coord[1], coord[2]);
            }
          });
          
          // 2. Update bonds
          bondMeshesRef.current.forEach(item => {
            const posA = new THREE.Vector3(...coords[item.source]);
            const posB = new THREE.Vector3(...coords[item.target]);
            updateCylinder(item.mesh, posA, posB);
          });

          // 3. Update waters
          const wCoords = latestWaterCoordsRef.current || [];
          const entropies = latestWaterEntropiesRef.current || [];
          wCoords.forEach((coord, idx) => {
            const w = waterMeshesRef.current[idx];
            if (w) {
              w.oxygen.position.set(...coord[0]);
              w.hydrogen1.position.set(...coord[1]);
              w.hydrogen2.position.set(...coord[2]);
              
              updateCylinder(w.bond1, new THREE.Vector3(...coord[0]), new THREE.Vector3(...coord[1]));
              updateCylinder(w.bond2, new THREE.Vector3(...coord[0]), new THREE.Vector3(...coord[2]));
              
              const visible = explicitWaterEnabledRef.current && showWaterVisualsRef.current;
              w.oxygen.visible = visible;
              w.hydrogen1.visible = visible;
              w.hydrogen2.visible = visible;
              w.bond1.visible = visible;
              w.bond2.visible = visible;
              
              // Color oxygen based on local entropy
              const S = entropies[idx] !== undefined ? entropies[idx] : 0.6;
              const color = new THREE.Color();
              if (S < 0.6) {
                const t = (S - 0.1) / 0.5;
                color.lerpColors(new THREE.Color(0.14, 0.83, 0.93), new THREE.Color(0.58, 0.64, 0.72), Math.max(0, Math.min(1, t)));
              } else {
                const t = (S - 0.6) / 0.4;
                color.lerpColors(new THREE.Color(0.58, 0.64, 0.72), new THREE.Color(0.96, 0.62, 0.04), Math.max(0, Math.min(1, t)));
              }
              if (w.oxygen.material) {
                w.oxygen.material.color.copy(color);
              }
            }
          });

          // 4. Draw active routing channels (golden pipelines)
          if (activePathsGroupRef.current) {
            while (activePathsGroupRef.current.children.length > 0) {
              const child = activePathsGroupRef.current.children[0];
              activePathsGroupRef.current.remove(child);
              if (child.geometry) child.geometry.dispose();
              if (child.material) child.material.dispose();
            }

            const paths = latestActiveInformationPathsRef.current || [];
            if (explicitWaterEnabledRef.current && showWaterVisualsRef.current && wCoords.length > 0) {
              paths.forEach(p => {
                const posA = new THREE.Vector3(...p.posU);
                const posB = new THREE.Vector3(...p.posV);
                
                const dir = new THREE.Vector3().subVectors(posB, posA);
                const len = dir.length();
                
                const geom = new THREE.CylinderGeometry(0.045, 0.045, len, 4);
                const mat = new THREE.MeshBasicMaterial({
                  color: 0xeab308, 
                  transparent: true,
                  opacity: 0.85
                });
                const mesh = new THREE.Mesh(geom, mat);
                
                const mid = new THREE.Vector3().addVectors(posA, posB).multiplyScalar(0.5);
                mesh.position.copy(mid);
                dir.normalize();
                const quaternion = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
                mesh.setRotationFromQuaternion(quaternion);
                
                activePathsGroupRef.current.add(mesh);
              });
            }
          }

          // Calculate local radius of gyration
          const com = new THREE.Vector3();
          coords.forEach(c => com.add(new THREE.Vector3(...c)));
          com.divideScalar(coords.length);
          let sqDistSum = 0;
          coords.forEach(c => {
            const v = new THREE.Vector3(...c);
            sqDistSum += v.distanceToSquared(com);
          });
          setGyrationRadius(Math.sqrt(sqDistSum / coords.length));
        }
        
        renderer.render(scene, camera);
      } catch (err) {
        console.error("Error in animate loop:", err);
        if (!window.loggedAnimateError) {
          window.loggedAnimateError = true;
          setLogs(prev => [...prev, { text: `animate loop error: ${err.message}`, type: 'error' }]);
        }
      }
    };
    animate();
    
    const mountContainer = mountRef.current;
    return () => {
      cancelAnimationFrame(animId);
      resizeObserver.disconnect();
      controls.dispose();
      renderer.dispose();
      if (mountContainer) {
        while (mountContainer.firstChild) {
          mountContainer.removeChild(mountContainer.firstChild);
        }
      }
    };
  }, []);

  // Subscribe to SSE stream from server
  useEffect(() => {
    if (!sessionId) return;

    const eventSource = new EventSource(`${BACKEND_URL}/api/stream?sessionId=${sessionId}`);
    
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      latestCoordsRef.current = data.coords;
      latestWaterCoordsRef.current = data.waterCoords || [];
      latestWaterEntropiesRef.current = data.waterEntropies || [];
      latestActiveInformationPathsRef.current = data.activeInformationPaths || [];

      if (data.temperature !== undefined) {
        setTemperature(data.temperature);
      }
      if (data.explicitWaterEnabled !== undefined) {
        setExplicitWaterEnabled(data.explicitWaterEnabled);
      }
      if (data.globalWaterBeltContraction !== undefined) {
        setGlobalWaterBeltContraction(data.globalWaterBeltContraction);
      }
      if (data.autoContractEnabled !== undefined) {
        setAutoContractEnabled(data.autoContractEnabled);
      }
      if (data.informationCaptureIndex !== undefined) {
        setInformationCaptureIndex(data.informationCaptureIndex);
      }
      if (data.hydrophobicExposure !== undefined) {
        setHydrophobicExposure(data.hydrophobicExposure);
      }
      if (data.hydrophobicContacts !== undefined) {
        setHydrophobicContacts(data.hydrophobicContacts);
      }
      if (data.waterEntropies && data.waterEntropies.length > 0) {
        const avg = data.waterEntropies.reduce((a, b) => a + b, 0) / data.waterEntropies.length;
        setAverageWaterEntropy(avg);
      }

      if (data.logs && data.logs.length > 0) {
        const newLogs = data.logs.map(logText => {
          let type = 'system';
          if (logText.includes('coalescence') || logText.includes('association') || logText.includes('lipid')) {
            type = 'water-agent';
          }
          return { text: logText, type };
        });
        setLogs(prev => [...prev, ...newLogs].slice(-100));
      }
    };

    eventSource.onerror = (err) => {
      console.error("SSE Connection error:", err);
      setLogs(prev => [...prev, { text: "Error: Disconnected from physics stream. Trying to reconnect...", type: "error" }]);
    };

    return () => {
      eventSource.close();
    };
  }, [sessionId]);

  return (
    <div className="app-layout">
      {/* 1. Header Control Bar */}
      <header className="app-header glass-panel">
        <div className="header-left">
          <div className="logo-container">
            <Activity className="logo-icon pulsing" />
            <span className="logo-text">H2O Lipid Coalescence Simulator</span>
          </div>
          <span className="mode-badge">MULTI-MOLECULE LIPID SELF-ASSEMBLY</span>
        </div>
        
        <div className="header-controls">
          <button 
            onClick={togglePlayback} 
            className={`play-btn ${isPlaying ? 'playing' : 'paused'}`}
            style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {isPlaying ? '一時停止' : 'シミュレーション開始'}
          </button>
          
          <button 
            onClick={handleInitialize} 
            className="secondary-btn" 
            style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
          >
            <RefreshCw className="w-4 h-4" /> 再リセット
          </button>
        </div>
      </header>

      {/* 2. Main Body Area */}
      <div className="app-body">
        {/* Left Control Sidebar */}
        <div className={`panel-sidebar ${showSidebar ? '' : 'collapsed'}`}>
          {/* Copy Count & SMILES Input */}
          <div className="glass-panel" style={{ padding: '14px' }}>
            <h2 className="panel-title">
              <Dna className="w-4 h-4 text-cyan-400" /> 分子構成・コピー数入力
            </h2>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '6px' }}>
              <button 
                onClick={handleInitialize} 
                className="primary" 
                disabled={loading}
                style={{ width: '100%', padding: '8px 14px', fontSize: '13px', whiteSpace: 'nowrap', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}
              >
                {loading ? '読み込み中...' : '初期構造モデル生成'}
              </button>
              
              {/* Molecule copy count slider */}
              <div className="slider-group" style={{ margin: '4px 0' }}>
                <div className="slider-header" style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span className="slider-title" style={{ fontSize: '11px', color: '#94a3b8' }}>👥 分子コピー数 (1〜5個)</span>
                  <span className="slider-value blue" style={{ fontWeight: 'bold' }}>{moleculeCount}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="5"
                  step="1"
                  value={moleculeCount}
                  onChange={(e) => setMoleculeCount(parseInt(e.target.value))}
                  style={{ width: '100%', marginTop: '4px' }}
                />
              </div>

              {/* Initial Layout Selection */}
              <div className="slider-group" style={{ margin: '4px 0' }}>
                <span className="slider-title" style={{ fontSize: '11px', color: '#94a3b8', display: 'block', marginBottom: '4px' }}>🏁 初期配置モード</span>
                <select
                  value={initialLayout}
                  onChange={(e) => setInitialLayout(e.target.value)}
                  style={{
                    width: '100%',
                    background: 'rgba(15, 23, 42, 0.9)',
                    border: '1px solid rgba(255, 255, 255, 0.12)',
                    borderRadius: '6px',
                    padding: '6px 10px',
                    color: 'white',
                    fontSize: '12px',
                    cursor: 'pointer'
                  }}
                >
                  <option value="dispersed">分散（自己組織化を観察）</option>
                  <option value="assembled">凝集（固体・ミセル会合相）</option>
                 </select>
              </div>

               <span style={{ fontSize: '11px', color: '#64748b', alignSelf: 'flex-start' }}>脂質SMILES入力:</span>
              <input
                type="text"
                value={sequence}
                onChange={(e) => setSequence(e.target.value)}
                placeholder="SMILES / 例: CCCCCCCCCCCC(=O)[O-]"
                style={{
                  width: '100%',
                  background: 'rgba(255, 255, 255, 0.04)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  borderRadius: '6px',
                  padding: '8px 10px',
                  color: 'white',
                  fontSize: '13px',
                  boxSizing: 'border-box'
                }}
              />
            </div>
            {error && <div className="error-message" style={{ marginTop: '8px', color: '#ef4444', fontSize: '12px' }}>{error}</div>}
          </div>

          {/* Section 2: Membrane Coalescence & Stats */}
          <div className="glass-panel" style={{ padding: '14px' }}>
            <h2 className="panel-title">
              <Activity className="w-4 h-4 text-cyan-400" /> 自己組織化・凝集度の統計情報
            </h2>
            
            <div className="stats-grid" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginTop: '6px' }}>
              <div className="stat-card" style={{ padding: '8px', background: 'rgba(255,255,255,0.02)', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.04)' }}>
                <span className="stat-label" style={{ fontSize: '10px', color: '#64748b', display: 'block' }}>重原子総数</span>
                <span className="stat-value text-cyan-400" style={{ fontSize: '16px', fontWeight: 'bold' }}>{atomCount} <span style={{ fontSize: '10px', color: '#475569' }}>atoms</span></span>
              </div>
              <div className="stat-card" style={{ padding: '8px', background: 'rgba(255,255,255,0.02)', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.04)' }}>
                <span className="stat-label" style={{ fontSize: '10px', color: '#64748b', display: 'block' }}>全体回転半径 (Rg)</span>
                <span className="stat-value text-blue-400" style={{ fontSize: '16px', fontWeight: 'bold' }}>{gyrationRadius.toFixed(2)} <span style={{ fontSize: '10px', color: '#475569' }}>Å</span></span>
              </div>
              <div className="stat-card" style={{ padding: '8px', background: 'rgba(255,255,255,0.02)', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.04)', gridColumn: 'span 2' }}>
                <span className="stat-label" style={{ fontSize: '10px', color: '#64748b', display: 'block' }}>🤝 疎水性分子間コンタクト数</span>
                <span className={`stat-value ${hydrophobicContacts > 0 ? 'text-amber-500 pulsing' : 'text-slate-500'}`} style={{ fontSize: '18px', fontWeight: 'bold' }}>
                  {hydrophobicContacts} <span style={{ fontSize: '10px', color: '#475569' }}>contacts (&lt; 5.0 Å)</span>
                </span>
                <span style={{ fontSize: '10px', color: '#475569', display: 'block', marginTop: '2px' }}>自己組織化・脂質二重層化の強さを示します</span>
              </div>
              <div className="stat-card" style={{ padding: '8px', background: 'rgba(255,255,255,0.02)', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.04)' }}>
                <span className="stat-label" style={{ fontSize: '10px', color: '#64748b', display: 'block' }}>情報捕獲能力 (Ic)</span>
                <span className="stat-value text-amber-500" style={{ fontSize: '15px', fontWeight: 'bold' }}>{informationCaptureIndex.toFixed(1)}</span>
              </div>
              <div className="stat-card" style={{ padding: '8px', background: 'rgba(255,255,255,0.02)', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.04)' }}>
                <span className="stat-label" style={{ fontSize: '10px', color: '#64748b', display: 'block' }}>平均水エントロピー (S)</span>
                <span className="stat-value text-indigo-400" style={{ fontSize: '15px', fontWeight: 'bold' }}>{averageWaterEntropy.toFixed(3)}</span>
              </div>
            </div>
          </div>

          {/* Section 3: Water Belt parameter controls */}
          <div className="glass-panel" style={{ padding: '14px' }}>
            <h2 className="panel-title">
              <Sliders className="w-4 h-4 text-cyan-400" /> 水分子ネットワークパラメータ
            </h2>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginTop: '4px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="slider-title" style={{ fontSize: '12px', color: '#94a3b8' }}>水分子の描画表示</span>
                <button
                  onClick={() => setShowWaterVisuals(!showWaterVisuals)}
                  className={showWaterVisuals ? "primary" : "secondary"}
                  style={{
                    padding: '4px 10px',
                    fontSize: '11px',
                    borderRadius: '6px',
                    background: showWaterVisuals ? 'linear-gradient(135deg, #0ea5e9, #2563eb)' : 'rgba(255,255,255,0.06)',
                    borderColor: showWaterVisuals ? 'rgba(255,255,255,0.2)' : 'rgba(255,255,255,0.08)',
                    color: 'white',
                    fontWeight: 'bold',
                    margin: 0
                  }}
                >
                  {showWaterVisuals ? "表示" : "非表示"}
                </button>
              </div>

              {/* Slider: Temperature */}
              <div className="slider-group">
                <div className="slider-header">
                  <span className="slider-title">🌡️ 熱運動ノイズ (温度)</span>
                  <span className="slider-value blue">{temperature.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="0.80"
                  step="0.02"
                  value={temperature}
                  onChange={(e) => updateParameter('temperature', parseFloat(e.target.value))}
                />
              </div>
              
              {/* Toggle Global Water Belt Auto-Contraction */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="slider-title" style={{ fontSize: '11px', color: '#94a3b8' }}>🌊 外殻水ベルト自動収縮</span>
                <button
                  onClick={() => {
                    const next = !autoContractEnabled;
                    setAutoContractEnabled(next);
                    syncControls({ autoContractEnabled: next });
                  }}
                  className={autoContractEnabled ? "primary" : "secondary"}
                  style={{
                    padding: '4px 10px',
                    fontSize: '11px',
                    borderRadius: '6px',
                    background: autoContractEnabled ? 'linear-gradient(135deg, #10b981, #059669)' : 'rgba(255,255,255,0.06)',
                    borderColor: autoContractEnabled ? 'rgba(255,255,255,0.2)' : 'rgba(255,255,255,0.08)',
                    color: 'white',
                    fontWeight: 'bold',
                    margin: 0
                  }}
                >
                  {autoContractEnabled ? "有効" : "無効"}
                </button>
              </div>

              {/* Slider: Global Water Belt Contraction Force */}
              <div className="slider-group">
                <div className="slider-header">
                  <span className="slider-title">🧲 外殻水ベルト収縮力 (k_contract)</span>
                  <span className="slider-value cyan">{globalWaterBeltContraction.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.00"
                  max="2.50"
                  step="0.05"
                  value={globalWaterBeltContraction}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    setGlobalWaterBeltContraction(val);
                    syncControls({ globalWaterBeltContraction: val });
                  }}
                />
              </div>

              {/* Slider: Water-Water Attraction */}
              <div className="slider-group">
                <div className="slider-header">
                  <span className="slider-title">💧 水分子同士の結合力 (k_ww)</span>
                  <span className="slider-value indg">{waterWaterStrength.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.10"
                  max="2.00"
                  step="0.05"
                  value={waterWaterStrength}
                  onChange={(e) => updateParameter('waterWaterStrength', parseFloat(e.target.value))}
                />
              </div>

              {/* Slider: Water-Peptide Attraction */}
              <div className="slider-group">
                <div className="slider-header">
                  <span className="slider-title">🧬 水-親水基結合力 (k_pw)</span>
                  <span className="slider-value indg">{waterPeptideStrength.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.10"
                  max="2.00"
                  step="0.05"
                  value={waterPeptideStrength}
                  onChange={(e) => updateParameter('waterPeptideStrength', parseFloat(e.target.value))}
                />
              </div>

              {/* Slider: Bulk Solvent Strength */}
              <div className="slider-group">
                <div className="slider-header">
                  <span className="slider-title">🌊 バルク溶媒の排除圧</span>
                  <span className="slider-value cyan">{bulkSolventStrength.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.00"
                  max="1.50"
                  step="0.05"
                  value={bulkSolventStrength}
                  onChange={(e) => updateParameter('bulkSolventStrength', parseFloat(e.target.value))}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Toggle buttons */}
        <button
          className={`toggle-btn-left ${showSidebar ? '' : 'collapsed'}`}
          onClick={() => setShowSidebar(!showSidebar)}
          style={{ margin: 0 }}
        >
          {showSidebar ? <ChevronLeft /> : <ChevronRight />}
        </button>

        <button
          className={`toggle-btn-right ${showLogs ? '' : 'collapsed'}`}
          onClick={() => setShowLogs(!showLogs)}
          style={{ margin: 0 }}
        >
          {showLogs ? <ChevronRight /> : <ChevronLeft />}
        </button>

        {/* WebGL Canvas Container */}
        <div className="canvas-container" ref={mountRef}></div>

        {/* Molecule copy indicator legend overlay */}
        <div className="folding-overlay glass-panel" style={{ position: 'absolute', top: '16px', left: showSidebar ? '360px' : '64px', padding: '10px', fontSize: '11px', pointerEvents: 'none', background: 'rgba(15,23,42,0.85)', zIndex: 10, transition: 'left 0.3s ease' }}>
          <span style={{ fontWeight: 'bold', color: '#22d3ee', display: 'block', marginBottom: '6px' }}>分子系統:</span>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', maxWidth: '200px' }}>
            {residueLabels.map((label, idx) => (
              <span key={idx} style={{ display: 'flex', alignItems: 'center', gap: '4px', background: 'rgba(255,255,255,0.04)', padding: '2px 6px', borderRadius: '4px', border: '1px solid rgba(255,255,255,0.08)' }}>
                <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: '#f97316' }} />
                {label}
              </span>
            ))}
          </div>
        </div>

        {/* Right Terminal Log panel */}
        <div className={`panel-logs ${showLogs ? '' : 'collapsed'}`}>
          <div className="terminal-header">
            <span className="terminal-title">🖥️ 物理・自己組織化シミュレータログ</span>
          </div>
          <div className="terminal-body" style={{ flex: 1, overflowY: 'auto', padding: '8px', fontSize: '11px', fontFamily: 'monospace', lineHeight: '1.4' }}>
            {logs.map((log, idx) => (
              <div key={idx} className={`log-line ${log.type}`} style={{ marginBottom: '4px' }}>
                <span className="timestamp" style={{ color: '#475569', marginRight: '6px' }}>[{new Date().toLocaleTimeString()}]</span>
                <span className="text">{log.text}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
