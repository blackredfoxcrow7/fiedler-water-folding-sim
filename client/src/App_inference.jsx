import React, { useState, useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { 
  Play, 
  Pause, 
  RotateCcw, 
  Activity, 
  Sliders, 
  Terminal, 
  Compass, 
  Share2,
  Thermometer, 
  Droplet, 
  Dna,
  Link,
  ChevronRight,
  ChevronLeft,
  Shield,
  HelpCircle,
  Cpu
} from 'lucide-react';
import './App.css';

const BACKEND_HOST = window.location.hostname || 'localhost';
const BACKEND_URL = `http://${BACKEND_HOST}:5005`;

// Helper to align cylinders between two points
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

export default function AppInference() {
  const [sequence, setSequence] = useState('GYDPETGTWG'); // Chignolin validation sequence
  const [sessionId, setSessionId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // UI Panels Toggle State
  const [showSidebar, setShowSidebar] = useState(true);
  const [showLogs, setShowLogs] = useState(true);

  // Simulation parameter states
  const [temperature, setTemperature] = useState(0.15);
  const [isPlaying, setIsPlaying] = useState(true);
  const [explicitWaterEnabled, setExplicitWaterEnabled] = useState(true);
  const [waterWaterStrength, setWaterWaterStrength] = useState(0.65);
  const [waterPeptideStrength, setWaterPeptideStrength] = useState(0.75);
  const [bulkSolventStrength, setBulkSolventStrength] = useState(0.45);
  const [kCapture, setKCapture] = useState(0.70);
  const [entropyBias, setEntropyBias] = useState(1.2);

  // Dynamic calculated stats
  const [gyrationRadius, setGyrationRadius] = useState(0.0);
  const [informationCaptureIndex, setInformationCaptureIndex] = useState(0.0);
  const [averageWaterEntropy, setAverageWaterEntropy] = useState(0.6);
  const [atomCount, setAtomCount] = useState(0);
  const [showWaterVisuals, setShowWaterVisuals] = useState(true);
  const [residueLabels, setResidueLabels] = useState([]);
  
  // Sequential Step States
  const [foldingStatus, setFoldingStatus] = useState('idle'); // 'idle', 'learning', 'folding', 'completed'
  const [learningMode, setLearningMode] = useState(false);
  const [inferenceMode, setInferenceMode] = useState(false);
  const [activeResidue, setActiveResidue] = useState(0);
  const [stepProgress, setStepProgress] = useState(0.0);
  const [trajectory, setTrajectory] = useState([]);

  // Logs
  const [logs, setLogs] = useState([]);
  const logEndRef = useRef(null);

  // Three.js refs
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const controlsRef = useRef(null);
  
  // Scene objects refs
  const atomMeshesRef = useRef([]);      // Peptide heavy atoms
  const bondMeshesRef = useRef([]);      // Peptide heavy bonds
  const waterMeshesRef = useRef([]);     // H2O objects
  const waterPeptideLinesRef = useRef(null); // Group for water-peptide lines
  const activePathsGroupRef = useRef(null); // Group for active information paths
  const backboneTubeRef = useRef(null);    // Smooth ribbon for backbone
  
  // Atom residue mapping
  const [atomResidues, setAtomResidues] = useState({});
  const [moleculeData, setMoleculeData] = useState(null);
  const moleculeDataRef = useRef(moleculeData);
  useEffect(() => {
    moleculeDataRef.current = moleculeData;
  }, [moleculeData]);
  const [isSceneReady, setIsSceneReady] = useState(false);

  const explicitWaterEnabledRef = useRef(explicitWaterEnabled);
  const showWaterVisualsRef = useRef(showWaterVisuals);
  const atomResiduesRef = useRef(atomResidues);

  useEffect(() => { explicitWaterEnabledRef.current = explicitWaterEnabled; }, [explicitWaterEnabled]);
  useEffect(() => { showWaterVisualsRef.current = showWaterVisuals; }, [showWaterVisuals]);
  useEffect(() => { atomResiduesRef.current = atomResidues; }, [atomResidues]);

  useEffect(() => {
    if (moleculeData && isSceneReady) {
      rebuildScene(moleculeData.atoms, moleculeData.bonds, moleculeData.initialCoords, moleculeData);
    }
  }, [moleculeData, isSceneReady]);

  // Physics streams buffers
  const latestCoordsRef = useRef(null);
  const latestWaterCoordsRef = useRef([]);
  const latestWatersRef = useRef([]);
  const latestWaterEntropiesRef = useRef([]);
  const latestActiveInformationPathsRef = useRef([]);

  // Initialize peptide and load topology from backend
  const handleInitialize = async () => {
    if (!sequence || !sequence.trim()) return;
    setLoading(true);
    setError(null);
    
    // Clear physics buffers immediately to prevent transition snapping
    latestCoordsRef.current = null;
    latestWaterCoordsRef.current = [];
    latestWatersRef.current = [];
    latestWaterEntropiesRef.current = [];
    latestActiveInformationPathsRef.current = [];

    try {
      const response = await fetch(`${BACKEND_URL}/api/initialize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequence })
      });
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || 'Failed to initialize system.');
      }
      const data = await response.json();
      setSessionId(data.sessionId);
      setAtomCount(data.atoms.length);
      setResidueLabels(data.residueLabels || []);
      setAtomResidues(data.atomResidues || {});
      setTrajectory([]); // Clear trajectory
      
      // Save molecule topology data locally to trigger rebuild
      setMoleculeData(data);
      setLogs([{ text: `System: Initialized sequence '${sequence}' with ${data.atoms.length} atoms.`, type: 'system' }]);
    } catch (err) {
      setError(err.message);
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Launch sequential C-to-N reverse learning
  const handleTrainReverse = async () => {
    if (!sessionId) return;
    try {
      const response = await fetch(`${BACKEND_URL}/api/train_reverse`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      if (response.ok) {
        setLogs(prev => [...prev, { text: "AI Learning: Running sequential reverse learning on validation folded structure...", type: "system" }]);
      }
    } catch (err) {
      console.error(err);
    }
  };

  // Launch sequential N-to-C forward folding inference
  const handleInferForward = async () => {
    if (!sessionId) return;
    try {
      const response = await fetch(`${BACKEND_URL}/api/infer_forward`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      if (response.ok) {
        setLogs(prev => [...prev, { text: "AI Inference: Running sequential autonomous folding on unknown sequence...", type: "system" }]);
      }
    } catch (err) {
      console.error(err);
    }
  };

  // Synchronize parameter changes to backend
  const syncControls = async (updatedParams) => {
    if (!sessionId) return;
    try {
      await fetch(`${BACKEND_URL}/api/control`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updatedParams)
      });
    } catch (err) {
      console.error("Control sync failed:", err);
    }
  };

  const updateParameter = (name, value) => {
    const setters = {
      temperature: setTemperature,
      isPlaying: setIsPlaying,
      explicitWaterEnabled: setExplicitWaterEnabled,
      waterWaterStrength: setWaterWaterStrength,
      waterPeptideStrength: setWaterPeptideStrength,
      bulkSolventStrength: setBulkSolventStrength,
      kCapture: setKCapture,
      entropyBias: setEntropyBias
    };
    if (setters[name]) {
      setters[name](value);
      syncControls({ [name]: value });
    }
  };

  // Clear Trajectory Plot
  const handleClearTrajectory = () => {
    setTrajectory([]);
    syncControls({ clearTrajectory: true });
  };

  // Initialize Three.js Scene
  useEffect(() => {
    if (!mountRef.current) return;

    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0f19);
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(0, 20, 35);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 150;
    controls.minDistance = 5;
    controlsRef.current = controls;

    // Ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.35);
    scene.add(ambientLight);

    // Directional light
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.85);
    dirLight.position.set(20, 40, 20);
    scene.add(dirLight);

    // Point lights
    const pointLight1 = new THREE.PointLight(0x0ea5e9, 0.4, 100);
    pointLight1.position.set(-15, 15, -15);
    scene.add(pointLight1);

    const pointLight2 = new THREE.PointLight(0xf59e0b, 0.3, 100);
    pointLight2.position.set(15, -15, 15);
    scene.add(pointLight2);

    // Water hydrogen-bonding lines
    const wpLinesGeo = new THREE.BufferGeometry();
    const wpLinesMat = new THREE.LineBasicMaterial({
      color: 0x0ea5e9,
      transparent: true,
      opacity: 0.30,
      blending: THREE.AdditiveBlending
    });
    const wpLines = new THREE.LineSegments(wpLinesGeo, wpLinesMat);
    scene.add(wpLines);
    waterPeptideLinesRef.current = wpLines;

    // Active information capture paths (gold cylinders)
    const activePathsGroup = new THREE.Group();
    scene.add(activePathsGroup);
    activePathsGroupRef.current = activePathsGroup;

    setIsSceneReady(true);

    // Resize observer
    const resizeObserver = new ResizeObserver((entries) => {
      for (let entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          if (cameraRef.current) {
            cameraRef.current.aspect = width / height;
            cameraRef.current.updateProjectionMatrix();
          }
          if (rendererRef.current) {
            rendererRef.current.setSize(width, height);
          }
        }
      }
    });
    resizeObserver.observe(mountRef.current);

    // Render loop
    let animationId;
    const animate = () => {
      animationId = requestAnimationFrame(animate);
      
      // Apply coordinate updates from the SSE buffer smoothly
      updateCoordinatesFromBuffer();
      
      if (controlsRef.current) controlsRef.current.update();
      if (rendererRef.current && sceneRef.current && cameraRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current);
      }
    };
    animate();

    const mountContainer = mountRef.current;
    return () => {
      cancelAnimationFrame(animationId);
      resizeObserver.disconnect();
      if (mountContainer && renderer.domElement) {
        mountContainer.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  // Initialize peptide simulation structure
  useEffect(() => {
    handleInitialize();
  }, []);

  // Subscribe to SSE stream from port 5005
  useEffect(() => {
    if (!sessionId) return;

    const eventSource = new EventSource(`${BACKEND_URL}/api/stream?sessionId=${sessionId}`);
    
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log("SSE update received, coords length:", data.coords ? data.coords.length : 0, "first coord:", data.coords ? data.coords[0] : null);
      
      latestCoordsRef.current = data.coords;
      latestWaterCoordsRef.current = data.waterCoords || [];
      latestWatersRef.current = data.waters || [];
      latestWaterEntropiesRef.current = data.waterEntropies || [];
      latestActiveInformationPathsRef.current = data.activeInformationPaths || [];

      if (data.temperature !== undefined) setTemperature(data.temperature);
      if (data.explicitWaterEnabled !== undefined) setExplicitWaterEnabled(data.explicitWaterEnabled);
      if (data.informationCaptureIndex !== undefined) setInformationCaptureIndex(data.informationCaptureIndex);
      if (data.foldingStatus !== undefined) setFoldingStatus(data.foldingStatus);
      if (data.learningMode !== undefined) setLearningMode(data.learningMode);
      if (data.inferenceMode !== undefined) setInferenceMode(data.inferenceMode);
      if (data.activeResidue !== undefined) setActiveResidue(data.activeResidue);
      if (data.stepProgress !== undefined) setStepProgress(data.stepProgress);
      if (data.trajectory !== undefined) {
        setTrajectory(data.trajectory);
        if (data.trajectory.length > 0) {
          setGyrationRadius(data.trajectory[data.trajectory.length - 1].rg);
        }
      }
      
      if (data.waterEntropies && data.waterEntropies.length > 0) {
        const avg = data.waterEntropies.reduce((a, b) => a + b, 0) / data.waterEntropies.length;
        setAverageWaterEntropy(avg);
      }

      if (data.logs && data.logs.length > 0) {
        const newLogs = data.logs.map(logText => {
          let type = 'system';
          if (logText.includes('AI Learning') || logText.includes('AI Inference')) {
            type = 'ai';
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

  // Scroll to log end
  useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  // Rebuild Three.js meshes
  const rebuildScene = (atoms, bonds, coords, rawData) => {
    const scene = sceneRef.current;
    if (!scene) return;

    // 1. Remove old peptide meshes
    atomMeshesRef.current.forEach(m => scene.remove(m));
    atomMeshesRef.current = [];
    bondMeshesRef.current.forEach(m => scene.remove(m));
    bondMeshesRef.current = [];
    
    if (backboneTubeRef.current) {
      scene.remove(backboneTubeRef.current);
      backboneTubeRef.current = null;
    }

    // 2. Build heavy atoms
    atoms.forEach((atom) => {
      const isPolar = atom.h_bond !== "none";
      const isHydrophobic = atom.hydrophobicity > 0.15;
      
      let color = 0x64748b; // Carbon (grey)
      if (atom.symbol === 'N') color = 0x3b82f6; // Nitrogen (blue)
      if (atom.symbol === 'O') color = 0xef4444; // Oxygen (red)
      if (atom.symbol === 'S') color = 0xeab308; // Sulfur (yellow)
      
      // Override for polar/hydrophobic highlighting
      if (isHydrophobic && atom.symbol === 'C') color = 0xf97316; // Hydrophobic orange
      
      const size = atom.symbol === 'C' ? 0.38 : 0.42;
      const geo = new THREE.SphereGeometry(size, 16, 16);
      
      // Glowing look for polar atoms
      const mat = new THREE.MeshStandardMaterial({
        color: color,
        roughness: 0.2,
        metalness: 0.1,
        emissive: isPolar ? color : 0x000000,
        emissiveIntensity: isPolar ? 0.15 : 0.0
      });
      
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.fromArray(coords[atom.id]);
      scene.add(mesh);
      atomMeshesRef.current.push(mesh);
    });

    // 3. Build heavy bonds
    bonds.forEach((bond) => {
      const posA = new THREE.Vector3().fromArray(coords[bond.source]);
      const posB = new THREE.Vector3().fromArray(coords[bond.target]);
      
      const geo = new THREE.CylinderGeometry(0.08, 0.08, 1, 8);
      const mat = new THREE.MeshStandardMaterial({
        color: 0x475569,
        roughness: 0.6
      });
      const mesh = new THREE.Mesh(geo, mat);
      updateCylinder(mesh, posA, posB);
      scene.add(mesh);
      bondMeshesRef.current.push(mesh);
    });

    // 4. Rebuild water molecule structures
    waterMeshesRef.current.forEach(w => {
      scene.remove(w.oxygen);
      scene.remove(w.hydrogen1);
      scene.remove(w.hydrogen2);
      scene.remove(w.bond1);
      scene.remove(w.bond2);
    });
    waterMeshesRef.current = [];

    const numWaters = rawData.waters ? rawData.waters.length : 0;
    if (numWaters > 0 && rawData.waterCoords) {
      for (let i = 0; i < numWaters; i++) {
        const w_coords = rawData.waterCoords[i];
        const pos_o = new THREE.Vector3().fromArray(w_coords[0]);
        const pos_h1 = new THREE.Vector3().fromArray(w_coords[1]);
        const pos_h2 = new THREE.Vector3().fromArray(w_coords[2]);

        const o_geo = new THREE.SphereGeometry(0.24, 8, 8);
        const o_mat = new THREE.MeshBasicMaterial({ color: 0x00d8ff });
        const o_mesh = new THREE.Mesh(o_geo, o_mat);
        o_mesh.position.copy(pos_o);
        scene.add(o_mesh);

        const h_geo = new THREE.SphereGeometry(0.12, 6, 6);
        const h_mat = new THREE.MeshBasicMaterial({ color: 0xffffff });
        
        const h1_mesh = new THREE.Mesh(h_geo, h_mat);
        h1_mesh.position.copy(pos_h1);
        scene.add(h1_mesh);

        const h2_mesh = new THREE.Mesh(h_geo, h_mat);
        h2_mesh.position.copy(pos_h2);
        scene.add(h2_mesh);

        const b_geo = new THREE.CylinderGeometry(0.03, 0.03, 1, 4);
        const b_mat = new THREE.MeshBasicMaterial({ color: 0x64748b });
        
        const b1_mesh = new THREE.Mesh(b_geo, b_mat);
        updateCylinder(b1_mesh, pos_o, pos_h1);
        scene.add(b1_mesh);

        const b2_mesh = new THREE.Mesh(b_geo, b_mat);
        updateCylinder(b2_mesh, pos_o, pos_h2);
        scene.add(b2_mesh);

        waterMeshesRef.current.push({
          oxygen: o_mesh,
          hydrogen1: h1_mesh,
          hydrogen2: h2_mesh,
          bond1: b1_mesh,
          bond2: b2_mesh
        });
      }
    }

    // 5. Build backbone ribbon tube
    rebuildBackboneTube(coords);
    
    // Position camera to fit the structure
    adjustCamera(coords);
  };

  // Draw smooth ribbon tube around C-alpha atoms
  const rebuildBackboneTube = (coords) => {
    const scene = sceneRef.current;
    if (!scene) return;

    if (backboneTubeRef.current) {
      scene.remove(backboneTubeRef.current);
      if (backboneTubeRef.current.geometry) backboneTubeRef.current.geometry.dispose();
      if (backboneTubeRef.current.material) {
        if (Array.isArray(backboneTubeRef.current.material)) {
          backboneTubeRef.current.material.forEach(m => m.dispose());
        } else {
          backboneTubeRef.current.material.dispose();
        }
      }
    }

    // Filter C-alpha coordinates
    const caCoords = [];
    const currentMoleculeData = moleculeDataRef.current;
    if (!currentMoleculeData) return;
    currentMoleculeData.atoms.forEach((atom) => {
      if (atom.name === 'CA' || atom.element === 'C' && atom.name.includes('CA')) {
        caCoords.push(new THREE.Vector3().fromArray(coords[atom.id]));
      }
    });

    if (caCoords.length < 2) return;

    // Interpolate path using CatmullRomCurve3
    const curve = new THREE.CatmullRomCurve3(caCoords);
    const tubeGeo = new THREE.TubeGeometry(curve, 64, 0.18, 8, false);
    const tubeMat = new THREE.MeshStandardMaterial({
      color: 0x4f46e5,
      roughness: 0.1,
      metalness: 0.2,
      transparent: true,
      opacity: 0.85
    });

    const tubeMesh = new THREE.Mesh(tubeGeo, tubeMat);
    scene.add(tubeMesh);
    backboneTubeRef.current = tubeMesh;
  };

  const adjustCamera = (coords) => {
    if (coords.length === 0 || !cameraRef.current || !controlsRef.current) return;
    const center = new THREE.Vector3();
    coords.forEach(c => center.add(new THREE.Vector3().fromArray(c)));
    center.divideScalar(coords.length);
    controlsRef.current.target.copy(center);
  };

  // Interpolate and update coordinate buffers smoothly
  const updateCoordinatesFromBuffer = () => {
    try {
      const scene = sceneRef.current;
      const currentMoleculeData = moleculeDataRef.current;
      if (!scene || !latestCoordsRef.current || !currentMoleculeData) return;

      const coords = latestCoordsRef.current;
      const atoms = currentMoleculeData.atoms;
      const bonds = currentMoleculeData.bonds;

      // 1. Update heavy atoms
      atoms.forEach((atom, idx) => {
        const mesh = atomMeshesRef.current[idx];
        if (mesh && coords[idx]) {
          mesh.position.fromArray(coords[idx]);
        }
      });

      // 2. Update heavy bonds
      bonds.forEach((bond, idx) => {
        const mesh = bondMeshesRef.current[idx];
        if (mesh && coords[bond.source] && coords[bond.target]) {
          const posA = new THREE.Vector3().fromArray(coords[bond.source]);
          const posB = new THREE.Vector3().fromArray(coords[bond.target]);
          updateCylinder(mesh, posA, posB);
        }
      });

      // 3. Update backbone tube ribbon
      rebuildBackboneTube(coords);

      // Track camera target on molecule center of mass
      adjustCamera(coords);

      // 4. Update water coordinates & color-code by entropy
      const waterCoords = latestWaterCoordsRef.current;
      const waters = latestWatersRef.current;
      const entropies = latestWaterEntropiesRef.current;

      const showWaters = showWaterVisualsRef.current && explicitWaterEnabledRef.current;

      waterMeshesRef.current.forEach((w, idx) => {
        if (idx < waterCoords.length && showWaters) {
          const w_coords = waterCoords[idx];
          if (w_coords && w_coords.length >= 3) {
            const pos_o = new THREE.Vector3().fromArray(w_coords[0]);
            const pos_h1 = new THREE.Vector3().fromArray(w_coords[1]);
            const pos_h2 = new THREE.Vector3().fromArray(w_coords[2]);

            w.oxygen.position.copy(pos_o);
            w.hydrogen1.position.copy(pos_h1);
            w.hydrogen2.position.copy(pos_h2);

            updateCylinder(w.bond1, pos_o, pos_h1);
            updateCylinder(w.bond2, pos_o, pos_h2);

            w.oxygen.visible = true;
            w.hydrogen1.visible = true;
            w.hydrogen2.visible = true;
            w.bond1.visible = true;
            w.bond2.visible = true;

            // Dynamic color representing solvation entropy
            if (idx < entropies.length) {
              const S = entropies[idx]; // Ranges 0.05 to 1.2
              // Interp color: Cyan (0x00f2ff) for low entropy to Gold (0xffa600) for high entropy
              const t = (S - 0.05) / 1.15;
              w.oxygen.material.color.setHSL(0.55 - t * 0.45, 1.0, 0.5);
            }
          }
        } else {
          w.oxygen.visible = false;
          w.hydrogen1.visible = false;
          w.hydrogen2.visible = false;
          w.bond1.visible = false;
          w.bond2.visible = false;
        }
      });

      // 5. Render hydrogen bonds & active capture pathways
      const activePaths = latestActiveInformationPathsRef.current;
      if (activePathsGroupRef.current) {
        const group = activePathsGroupRef.current;
        
        // Clear old path meshes
        while(group.children.length > 0) { 
          const child = group.children[0];
          group.remove(child); 
          if (child.geometry) child.geometry.dispose();
          if (child.material) {
            if (Array.isArray(child.material)) {
              child.material.forEach(m => m.dispose());
            } else {
              child.material.dispose();
            }
          }
        }

        if (showWaters && activePaths) {
          activePaths.forEach((path) => {
            if (path && path.posU && path.posV) {
              const posU = new THREE.Vector3().fromArray(path.posU);
              const posV = new THREE.Vector3().fromArray(path.posV);
              
              // Draw connecting tubes representing active information pathways
              const geo = new THREE.CylinderGeometry(0.04, 0.04, 1, 6);
              const mat = new THREE.MeshBasicMaterial({
                color: path.type === 0 ? 0x06b6d4 : 0xf59e0b, // Cyan for H2O-H2O, Gold for H2O-Peptide
                transparent: true,
                opacity: 0.65
              });
              const cylinder = new THREE.Mesh(geo, mat);
              updateCylinder(cylinder, posU, posV);
              group.add(cylinder);
            }
          });
        }
      }
    } catch (err) {
      console.error("Error in updateCoordinatesFromBuffer:", err);
    }
  };

  return (
    <div className="app-layout">
      {/* 1. Header Toolbar */}
      <header className="app-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div className="logo-badge" style={{ background: 'linear-gradient(135deg, #a855f7, #6366f1)' }}>
            <Cpu className="w-5 h-5 text-white animate-pulse" />
          </div>
          <div>
            <h1 className="header-title" style={{ background: 'linear-gradient(to right, #c084fc, #818cf8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
              H2O Residue-Sequential Autonomous Inference Simulator
            </h1>
            <p className="header-subtitle">相補的アミノ酸ステップ学習・立体配座推論と水和エントロピー可視化</p>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          <button 
            onClick={() => updateParameter('isPlaying', !isPlaying)} 
            className={isPlaying ? "secondary-btn" : "primary-btn"}
            style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {isPlaying ? '一時停止' : '再開'}
          </button>
          
          <button 
            onClick={handleInitialize} 
            className="secondary-btn" 
            style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
          >
            <RotateCcw className="w-4 h-4" /> 再リセット
          </button>
        </div>
      </header>

      {/* 2. Main Body Area */}
      <div className="app-body" style={{ position: 'relative', width: '100%', height: 'calc(100vh - 75px)', overflow: 'hidden', display: 'block' }}>
        {/* Left Control Sidebar */}
        <div className={`panel-sidebar ${showSidebar ? '' : 'collapsed'}`} style={{
          position: 'absolute',
          top: '14px',
          left: '14px',
          bottom: '14px',
          width: '330px',
          minWidth: '330px',
          display: 'flex',
          flexDirection: 'column',
          gap: '14px',
          padding: '14px',
          background: 'rgba(15, 23, 42, 0.85)',
          backdropFilter: 'blur(16px)',
          WebkitBackdropFilter: 'blur(16px)',
          border: '1.5px solid rgba(255, 255, 255, 0.15)',
          borderRadius: '12px',
          zIndex: 10,
          overflowY: 'auto',
          boxSizing: 'border-box',
          boxShadow: '0 12px 36px rgba(0, 0, 0, 0.55)',
          transition: 'transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.3s ease',
          transform: showSidebar ? 'translateX(0)' : 'translateX(-110%)',
          opacity: showSidebar ? 1 : 0,
          pointerEvents: showSidebar ? 'auto' : 'none'
        }}>
          
          {/* Active Residue Step-Wise Progress Bar */}
          <div className="glass-panel" style={{ padding: '14px', border: '1px solid rgba(168, 85, 247, 0.25)' }}>
            <h2 className="panel-title" style={{ color: '#c084fc' }}>
              <Compass className="w-4 h-4 text-purple-400" /> アミノ酸ステップ進捗
            </h2>
            
            <div style={{ marginTop: '8px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                <span style={{ color: '#94a3b8' }}>状態:</span>
                <span className="cyan" style={{ fontWeight: 'bold' }}>
                  {foldingStatus === 'learning' ? '🔄 逆再生ステップ学習中 (C➔N)' : 
                   foldingStatus === 'folding' ? '🤖 自律ステップ推論中 (N➔C)' : 
                   foldingStatus === 'completed' ? '✅ シーケンシャルプロセス完了' : '💤 待機中 (物理シミュレーション)'}
                </span>
              </div>

              {(learningMode || inferenceMode) && residueLabels.length > 0 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', margin: '8px 0' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#e2e8f0' }}>
                    <span>アクティブ残基: <strong>{residueLabels[activeResidue]}</strong></span>
                    <span>ステップ進捗: {(stepProgress * 100).toFixed(0)}%</span>
                  </div>
                  {/* Step progress bar */}
                  <div style={{ height: '6px', background: 'rgba(255,255,255,0.06)', borderRadius: '3px', overflow: 'hidden' }}>
                    <div style={{ 
                      height: '100%', 
                      width: `${stepProgress * 100}%`, 
                      background: learningMode ? 'linear-gradient(90deg, #f97316, #ea580c)' : 'linear-gradient(90deg, #06b6d4, #0891b2)',
                      transition: 'width 0.1s linear'
                    }} />
                  </div>
                </div>
              )}

              {/* Progress dots for each residue */}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginTop: '10px' }}>
                {residueLabels.map((lbl, idx) => {
                  let color = 'rgba(255,255,255,0.1)';
                  let isCurrent = idx === activeResidue && (learningMode || inferenceMode);
                  if (isCurrent) {
                    color = learningMode ? '#f97316' : '#06b6d4';
                  } else if ((learningMode && idx > activeResidue) || (inferenceMode && idx < activeResidue)) {
                    color = 'rgba(16, 185, 129, 0.4)'; // Completed step
                  }
                  return (
                    <div 
                      key={idx}
                      title={lbl}
                      style={{
                        padding: '2px 6px',
                        fontSize: '9px',
                        borderRadius: '4px',
                        background: color,
                        border: isCurrent ? '1px solid white' : '1px solid transparent',
                        color: isCurrent ? '#fff' : '#64748b',
                        fontWeight: isCurrent ? 'bold' : 'normal'
                      }}
                    >
                      {lbl.split('-')[0]}
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* AI Learning & Inference Cooperative controls */}
          <div className="glass-panel" style={{ padding: '14px' }}>
            <h2 className="panel-title">
              <Cpu className="w-4 h-4 text-cyan-400" /> 自律ステップ制御
            </h2>
            <p style={{ fontSize: '11px', color: '#64748b', margin: '4px 0 10px 0' }}>
              アミノ酸残基を単位ステップとし、水ベルトとペプチドの相補的配座を学習・推論させます。
            </p>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <button 
                onClick={handleTrainReverse}
                disabled={loading || learningMode || inferenceMode}
                className="primary"
                style={{ 
                  width: '100%', 
                  padding: '8px 14px', 
                  fontSize: '12px',
                  background: 'linear-gradient(135deg, #f97316, #ea580c)',
                  borderColor: 'rgba(255,255,255,0.1)'
                }}
              >
                🔄 ステップごとに局所規則を学習 (逆再生 C➔N)
              </button>

              <button 
                onClick={handleInferForward}
                disabled={loading || learningMode || inferenceMode}
                className="primary"
                style={{ 
                  width: '100%', 
                  padding: '8px 14px', 
                  fontSize: '12px',
                  background: 'linear-gradient(135deg, #0ea5e9, #2563eb)',
                  borderColor: 'rgba(255,255,255,0.1)'
                }}
              >
                🤖 自律ステップ推論 (フォールディング N➔C)
              </button>
            </div>
          </div>

          {/* Molecule copy count & SMILES Input */}
          <div className="glass-panel" style={{ padding: '14px' }}>
            <h2 className="panel-title">
              <Dna className="w-4 h-4 text-cyan-400" /> 対象ペプチド配列入力
            </h2>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '6px' }}>
              <input
                type="text"
                value={sequence}
                onChange={(e) => setSequence(e.target.value)}
                placeholder="配列例: GYDPETGTWG or Ala-Cys-Gly"
                style={{
                  width: '100%',
                  background: 'rgba(255, 255, 255, 0.04)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  borderRadius: '6px',
                  padding: '6px 10px',
                  color: 'white',
                  fontSize: '12px'
                }}
              />
              <button 
                onClick={handleInitialize} 
                className="secondary-btn" 
                disabled={loading || learningMode || inferenceMode}
                style={{ width: '100%', padding: '6px 14px', fontSize: '12px' }}
              >
                {loading ? '読み込み中...' : '初期化・再ロード'}
              </button>
            </div>
          </div>

          {/* Parameter Sliders */}
          <div className="glass-panel" style={{ padding: '14px' }}>
            <h2 className="panel-title">
              <Sliders className="w-4 h-4 text-cyan-400" /> 水和物理パラメータ
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

              {/* Slider: Solvation Drive strength */}
              <div className="slider-group">
                <div className="slider-header">
                  <span className="slider-title">🧲 情報キャプチャ誘導力 (K_capture)</span>
                  <span className="slider-value cyan">{kCapture.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="1.50"
                  step="0.05"
                  value={kCapture}
                  onChange={(e) => updateParameter('kCapture', parseFloat(e.target.value))}
                />
              </div>

              {/* Slider: Bulk Solvent Exclusion force */}
              <div className="slider-group">
                <div className="slider-header">
                  <span className="slider-title">🌊 バルク溶媒排除圧 (K_solvent)</span>
                  <span className="slider-value blue">{bulkSolventStrength.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.0"
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
        <div className="canvas-container" ref={mountRef} style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          zIndex: 1,
          overflow: 'hidden'
        }}></div>

        {/* Solvation Hysteresis Loop Chart Overlay */}
        <div className="glass-panel" style={{ 
          position: 'absolute', 
          bottom: '20px', 
          left: showSidebar ? '360px' : '20px', 
          width: '290px', 
          padding: '14px', 
          zIndex: 10,
          pointerEvents: 'auto',
          background: 'rgba(11, 15, 25, 0.85)',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          transition: 'left 0.3s cubic-bezier(0.4, 0, 0.2, 1)'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <h2 className="panel-title" style={{ color: '#e2e8f0', margin: 0 }}>
              <Activity className="w-4 h-4 text-cyan-400" /> 水和ヒステリシスループ
            </h2>
            <button 
              onClick={handleClearTrajectory}
              className="secondary-btn"
              style={{ padding: '2px 6px', fontSize: '9px', margin: 0 }}
            >
              クリア
            </button>
          </div>
          
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '9px', color: '#64748b', marginBottom: '4px' }}>
            <span>縦軸: 情報キャプチャ (I_capture)</span>
            <span>横軸: 回転半径 (Rg)</span>
          </div>

          <div style={{ 
            height: '110px', 
            width: '100%', 
            background: 'rgba(0,0,0,0.3)', 
            borderRadius: '6px', 
            border: '1px solid rgba(255,255,255,0.05)',
            position: 'relative'
          }}>
            {trajectory && trajectory.length > 0 ? (
              <svg style={{ width: '100%', height: '100%', overflow: 'visible' }}>
                {/* Grid Lines */}
                <line x1="10%" y1="10%" x2="90%" y2="10%" stroke="rgba(255,255,255,0.03)" />
                <line x1="10%" y1="50%" x2="90%" y2="50%" stroke="rgba(255,255,255,0.03)" />
                <line x1="10%" y1="90%" x2="90%" y2="90%" stroke="rgba(255,255,255,0.03)" />
                
                {(() => {
                  const rgs = trajectory.map(t => t.rg);
                  const ics = trajectory.map(t => t.I_capture);
                  const minRg = Math.min(...rgs);
                  const maxRg = Math.max(...rgs);
                  const minIc = Math.min(...ics);
                  const maxIc = Math.max(...ics);
                  
                  const rgRange = (maxRg - minRg) || 1.0;
                  const icRange = (maxIc - minIc) || 1.0;
                  
                  // Convert trajectory points into SVG path string
                  let learnPts = [];
                  let inferPts = [];
                  
                  trajectory.forEach(pt => {
                    const x = 20 + 220 * (pt.rg - minRg) / rgRange;
                    const y = 95 - 80 * (pt.I_capture - minIc) / icRange;
                    
                    if (pt.mode === 'learning') {
                      learnPts.push(`${x},${y}`);
                    } else if (pt.mode === 'inference') {
                      inferPts.push(`${x},${y}`);
                    }
                  });
                  
                  return (
                    <>
                      {learnPts.length > 1 && (
                        <polyline
                          fill="none"
                          stroke="#f97316"
                          strokeWidth="2"
                          strokeDasharray="2,2"
                          points={learnPts.join(' ')}
                        />
                      )}
                      {inferPts.length > 1 && (
                        <polyline
                          fill="none"
                          stroke="#06b6d4"
                          strokeWidth="2.5"
                          points={inferPts.join(' ')}
                        />
                      )}
                      
                      {/* Legend Overlay */}
                      <g style={{ fontSize: '8px', fill: '#94a3b8' }} transform="translate(10, 10)">
                        <rect x="0" y="0" width="8" height="4" fill="#f97316" />
                        <text x="12" y="4">逆学習 (C➔N)</text>
                        
                        <rect x="100" y="0" width="8" height="4" fill="#06b6d4" />
                        <text x="112" y="4">自律推論 (N➔C)</text>
                      </g>
                    </>
                  );
                })()}
              </svg>
            ) : (
              <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#475569', fontSize: '10px' }}>
                軌跡データなし (学習/推論を実行してください)
              </div>
            )}
          </div>
        </div>

        {/* Stats Dashboard Overlays */}
        <div className="stats-overlay" style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(4, 1fr)', 
          gap: '10px', 
          width: showSidebar ? (showLogs ? 'calc(100% - 700px)' : 'calc(100% - 380px)') : (showLogs ? 'calc(100% - 380px)' : 'calc(100% - 60px)'),
          left: showSidebar ? '360px' : '64px', 
          top: '20px', 
          background: 'rgba(11, 15, 25, 0.7)',
          position: 'absolute',
          zIndex: 10,
          transition: 'left 0.3s cubic-bezier(0.4, 0, 0.2, 1), width 0.3s cubic-bezier(0.4, 0, 0.2, 1)'
        }}>
          <div className="stat-card">
            <span className="stat-label">🧬 重原子数 / 水分子数</span>
            <span className="stat-value text-indigo-400">{atomCount} / {waterMeshesRef.current.length}</span>
          </div>
          
          <div className="stat-card">
            <span className="stat-label">📐 回転半径 (Rg)</span>
            <span className="stat-value text-blue-400">{gyrationRadius.toFixed(2)} Å</span>
          </div>

          <div className="stat-card">
            <span className="stat-label">🕸️ 情報キャプチャ (I_capture)</span>
            <span className="stat-value text-amber-400">{informationCaptureIndex.toFixed(3)}</span>
          </div>

          <div className="stat-card">
            <span className="stat-label">🌊 水エントロピー (S_water)</span>
            <span className="stat-value text-cyan-400">{averageWaterEntropy.toFixed(3)}</span>
          </div>
        </div>

        {/* Right Console Logs Sidebar */}
        <div className={`panel-logs ${showLogs ? '' : 'collapsed'}`} style={{
          position: 'absolute',
          top: '14px',
          right: '14px',
          bottom: '14px',
          width: '330px',
          minWidth: '330px',
          display: 'flex',
          flexDirection: 'column',
          padding: '14px',
          background: 'rgba(15, 23, 42, 0.85)',
          backdropFilter: 'blur(16px)',
          WebkitBackdropFilter: 'blur(16px)',
          border: '1.5px solid rgba(255, 255, 255, 0.15)',
          borderRadius: '12px',
          zIndex: 10,
          overflow: 'hidden',
          boxSizing: 'border-box',
          boxShadow: '0 12px 36px rgba(0, 0, 0, 0.55)',
          transition: 'transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.3s ease',
          transform: showLogs ? 'translateX(0)' : 'translateX(110%)',
          opacity: showLogs ? 1 : 0,
          pointerEvents: showLogs ? 'auto' : 'none'
        }}>
          <div className="terminal-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
            <h3 style={{ display: 'flex', alignItems: 'center', gap: '6px', margin: 0, fontSize: '0.85rem' }}>
              <Terminal className="w-4 h-4 text-cyan-400" /> エージェント意思決定ログ
            </h3>
            <button 
              onClick={() => setLogs([])}
              style={{
                padding: '2px 8px',
                fontSize: '10px',
                borderRadius: '4px',
                background: 'rgba(255,255,255,0.06)',
                border: '1px solid rgba(255,255,255,0.1)',
                color: '#94a3b8',
                cursor: 'pointer',
                margin: 0
              }}
            >
              ログ消去
            </button>
          </div>
          
          <div className="terminal-body" style={{ flex: 1, overflowY: 'auto', padding: '8px', fontSize: '11px', fontFamily: 'monospace', lineHeight: '1.4' }}>
            {logs.map((log, idx) => (
              <div key={idx} className={`log-line ${log.type}`} style={{ marginBottom: '4px' }}>
                <span className="timestamp" style={{ color: '#475569', marginRight: '6px' }}>[{new Date().toLocaleTimeString()}]</span>
                <span className="text">{log.text}</span>
              </div>
            ))}
            <div ref={logEndRef} />
          </div>
        </div>
      </div>
    </div>
  );
}
