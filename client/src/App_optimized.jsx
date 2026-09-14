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
  Thermometer, 
  Droplet, 
  Dna,
  Link,
  ChevronRight,
  ChevronLeft,
  Shield
} from 'lucide-react';

const BACKEND_HOST = window.location.hostname || 'localhost';
const BACKEND_URL = `http://${BACKEND_HOST}:5001`;

export default function App() {
  const [jsErrors, setJsErrors] = useState([]);
  const [showSidebar, setShowSidebar] = useState(true);
  const [showLogs, setShowLogs] = useState(true);
  const [viewportSize, setViewportSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const originalConsoleError = console.error;
    console.error = (...args) => {
      originalConsoleError.apply(console, args);
      // Skip framing/Vite HMR logs to avoid noise
      const msg = args.map(arg => typeof arg === 'object' ? JSON.stringify(arg) : String(arg)).join(' ');
      if (!msg.includes('[vite]') && !msg.includes('HMR')) {
        setJsErrors((prev) => [...prev, `Console Error: ${msg}`].slice(-15));
      }
    };

    const handleError = (event) => {
      const errorMsg = event.error ? event.error.stack || event.error.message : event.message;
      setJsErrors((prev) => [...prev, `Error: ${errorMsg}`].slice(-15));
    };
    const handleRejection = (event) => {
      const reason = event.reason ? event.reason.stack || event.reason.message || String(event.reason) : 'Unhandled promise rejection';
      setJsErrors((prev) => [...prev, `Promise Rejection: ${reason}`].slice(-15));
    };
    window.addEventListener('error', handleError);
    window.addEventListener('unhandledrejection', handleRejection);
    return () => {
      console.error = originalConsoleError;
      window.removeEventListener('error', handleError);
      window.removeEventListener('unhandledrejection', handleRejection);
    };
  }, []);

  // Input sequence / SMILES state
  const [sequence, setSequence] = useState('Ala-Ala-Ala-Val-Leu-Ser-Asp-Lys');
  const [sessionId, setSessionId] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [moleculeData, setMoleculeData] = useState(null);
  const [isSceneReady, setIsSceneReady] = useState(false);

  // Sequential folding state
  const [atomResidues, setAtomResidues] = useState({});
  const [residueLabels, setResidueLabels] = useState([]);
  const [seqFolding, setSeqFolding] = useState({
    enabled: false,
    activeResidue: 0,
    currentStep: 0,
    stepsPerResidue: 400,
    status: 'idle',
    nResidues: 0,
    residueLabels: []
  });

  // Renderer mode selection
  const [rendererType, setRendererType] = useState('canvas2d'); // Default to canvas2d for compatibility
  const canvas2DRef = useRef(null);

  // Mouse & touch drag/zoom refs for 2D Canvas
  const rotXRef = useRef(0.3);
  const rotYRef = useRef(0.5);
  const zoomRef = useRef(40);
  const isDraggingRef = useRef(false);
  const previousMousePositionRef = useRef({ x: 0, y: 0 });

  // Cached stats for avoiding state updates spam
  const lastRgRef = useRef(0);
  const lastHBondCountRef = useRef(-1);

  // Simulation parameters (real-time synchronized with backend)
  const [temperature, setTemperature] = useState(0.08);
  const [isPlaying, setIsPlaying] = useState(true);
  const [hydrophobicExposure, setHydrophobicExposure] = useState(0.0);
  const [explicitWaterEnabled, setExplicitWaterEnabled] = useState(true);
  const [showWaterVisuals, setShowWaterVisuals] = useState(true);
  const [waterWaterStrength, setWaterWaterStrength] = useState(0.50);
  const [waterPeptideStrength, setWaterPeptideStrength] = useState(0.60);
  const [bulkSolventStrength, setBulkSolventStrength] = useState(0.35);
  const [waterBeltContraction, setWaterBeltContraction] = useState(0.50);
  const [watersPerAtom, setWatersPerAtom] = useState(4);
  const [waters, setWaters] = useState([]);
  const [activeWaterBonds, setActiveWaterBonds] = useState([]);
  const [activeWaterPeptideBonds, setActiveWaterPeptideBonds] = useState([]);
  const [waterBeltRatio, setWaterBeltRatio] = useState(0);

  // Live statistics
  const [atomCount, setAtomCount] = useState(0);
  const [jointCount, setJointCount] = useState(0);
  const [activeHBondCount, setActiveHBondCount] = useState(0);
  const [gyrationRadius, setGyrationRadius] = useState(0.0);

  // Panel states
  const [logs, setLogs] = useState([]);
  const logsEndRef = useRef(null);

  // Three.js and SSE refs
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const controlsRef = useRef(null);
  const atomMeshesRef = useRef([]);
  const bondMeshesRef = useRef([]);
  const hbondLinesRef = useRef(null); // Group for active H-bond lines
  const comMeshRef = useRef(null);    // Group/Mesh for Center of Mass
  const rotatingHaloRef = useRef(null); // Glowing spin ring for the active folding bond
  const initialCenterRef = useRef([0, 0, 0]);

  const latestCoordsRef = useRef(null);
  const latestHBondsRef = useRef([]);
  const waterMeshesRef = useRef([]);
  const waterBondMeshesRef = useRef([]);
  const waterPeptideLinesRef = useRef(null);
  const latestWaterCoordsRef = useRef([]);
  const latestWatersRef = useRef([]);
  const latestWaterBondsRef = useRef([]);
  const latestWaterPeptideBondsRef = useRef([]);
  const seqFoldingRef = useRef(seqFolding);
  const explicitWaterEnabledRef = useRef(explicitWaterEnabled);
  const showWaterVisualsRef = useRef(showWaterVisuals);
  const moleculeDataRef = useRef(moleculeData);
  const atomResiduesRef = useRef(atomResidues);

  useEffect(() => {
    seqFoldingRef.current = seqFolding;
  }, [seqFolding]);

  useEffect(() => {
    explicitWaterEnabledRef.current = explicitWaterEnabled;
  }, [explicitWaterEnabled]);

  useEffect(() => {
    showWaterVisualsRef.current = showWaterVisuals;
  }, [showWaterVisuals]);

  useEffect(() => {
    moleculeDataRef.current = moleculeData;
  }, [moleculeData]);

  useEffect(() => {
    atomResiduesRef.current = atomResidues;
  }, [atomResidues]);

  const updateStats = (rg, hBondCount, numWaterBonds = 0) => {
    if (Math.abs(lastRgRef.current - rg) > 0.001) {
      lastRgRef.current = rg;
      setGyrationRadius(rg);
    }
    if (lastHBondCountRef.current !== hBondCount) {
      lastHBondCountRef.current = hBondCount;
      setActiveHBondCount(hBondCount);
    }
    if (waters && waters.length > 1) {
      const maxBonds = waters.length - 1;
      const ratio = numWaterBonds / maxBonds;
      setWaterBeltRatio(ratio);
    } else {
      setWaterBeltRatio(0);
    }
  };

  const handleMouseDown = (e) => {
    isDraggingRef.current = true;
    previousMousePositionRef.current = { x: e.clientX, y: e.clientY };
  };

  const handleMouseMove = (e) => {
    if (!isDraggingRef.current) return;
    const deltaX = e.clientX - previousMousePositionRef.current.x;
    const deltaY = e.clientY - previousMousePositionRef.current.y;
    previousMousePositionRef.current = { x: e.clientX, y: e.clientY };

    rotYRef.current += deltaX * 0.01;
    rotXRef.current += deltaY * 0.01;
    rotXRef.current = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, rotXRef.current));
  };

  const handleMouseUp = () => {
    isDraggingRef.current = false;
  };

  const handleWheel = (e) => {
    zoomRef.current = Math.max(10, Math.min(200, zoomRef.current - e.deltaY * 0.05));
  };

  const handleTouchStart = (e) => {
    if (e.touches.length === 1) {
      isDraggingRef.current = true;
      previousMousePositionRef.current = { x: e.touches[0].clientX, y: e.touches[0].clientY };
    }
  };

  const handleTouchMove = (e) => {
    if (!isDraggingRef.current || e.touches.length !== 1) return;
    const deltaX = e.touches[0].clientX - previousMousePositionRef.current.x;
    const deltaY = e.touches[0].clientY - previousMousePositionRef.current.y;
    previousMousePositionRef.current = { x: e.touches[0].clientX, y: e.touches[0].clientY };

    rotYRef.current += deltaX * 0.01;
    rotXRef.current += deltaY * 0.01;
    rotXRef.current = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, rotXRef.current));
  };

  const handleTouchEnd = () => {
    isDraggingRef.current = false;
  };

  // Auto-scroll logs panel
  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  // Initial loading on mount
  useEffect(() => {
    handleInitialize();
  }, []);

  // Rebuild scene when molecule data is loaded and Three.js scene is ready
  useEffect(() => {
    if (moleculeData && isSceneReady) {
      rebuildScene(moleculeData.atoms, moleculeData.bonds, moleculeData.initialCoords);
    }
  }, [moleculeData, isSceneReady]);

  // Initialize peptide and load topology from backend
  const handleInitialize = async () => {
    if (!sequence || !sequence.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${BACKEND_URL}/api/initialize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          sequence,
          temperature,
          isPlaying,
          explicitWaterEnabled,
          waterWaterStrength,
          waterPeptideStrength,
          bulkSolventStrength,
          waterBeltContraction,
          watersPerAtom
        })
      });
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || 'Failed to initialize molecule.');
      }
      const data = await response.json();
      setSessionId(data.sessionId);
      setAtomCount(data.atoms.length);
      // Convert residues map keys from string (JSON default) to numbers for direct indexing
      const resMap = {};
      if (data.atomResidues) {
        Object.entries(data.atomResidues).forEach(([k, v]) => {
          resMap[Number(k)] = v;
        });
      }
      setAtomResidues(resMap);
      setResidueLabels(data.residueLabels || []);
      
      setSeqFolding({
        enabled: false,
        activeResidue: 0,
        currentStep: 0,
        stepsPerResidue: 400,
        status: 'idle',
        nResidues: data.residueLabels ? data.residueLabels.length : 0,
        residueLabels: data.residueLabels || []
      });
      
      setActiveHBondCount(0);
      setGyrationRadius(0.0);
      setWaters(data.waters || []);
      setActiveWaterBonds([]);
      setActiveWaterPeptideBonds([]);
      setWaterBeltRatio(0);
      setLogs([
        { text: "System: Peptide initialized successfully.", type: "system" },
        { text: `System: Session ID is ${data.sessionId}`, type: "system" }
      ]);

      // Calculate initial bounding box center to center the view in canvas2d
      if (data.initialCoords && data.initialCoords.length > 0) {
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        let minZ = Infinity, maxZ = -Infinity;
        data.initialCoords.forEach(c => {
          if (c[0] < minX) minX = c[0];
          if (c[0] > maxX) maxX = c[0];
          if (c[1] < minY) minY = c[1];
          if (c[1] > maxY) maxY = c[1];
          if (c[2] < minZ) minZ = c[2];
          if (c[2] > maxZ) maxZ = c[2];
        });
        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;
        const centerZ = (minZ + maxZ) / 2;
        initialCenterRef.current = [centerX, centerY, centerZ];
      } else {
        initialCenterRef.current = [0, 0, 0];
      }

      // Store in state to trigger useEffect drawing
      setMoleculeData(data);
    } catch (err) {
      setError(err.message);
      setLogs(prev => [...prev, { text: `Error: ${err.message}`, type: "clash" }]);
    } finally {
      setLoading(false);
    }
  };

  // Synchronize parameter updates to the backend
  const syncControls = async (paramsUpdate) => {
    if (!sessionId) return;
    try {
      await fetch(`${BACKEND_URL}/api/control`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId, ...paramsUpdate })
      });
    } catch (err) {
      console.error("Failed to sync controls:", err);
    }
  };

  // Handle controls updates
  const updateParameter = (type, val) => {
    if (type === 'temperature') {
      setTemperature(val);
      syncControls({ temperature: val });
    } else if (type === 'isPlaying') {
      setIsPlaying(val);
      syncControls({ isPlaying: val });
    } else if (type === 'explicitWaterEnabled') {
      setExplicitWaterEnabled(val);
      syncControls({ explicitWaterEnabled: val });
    } else if (type === 'waterWaterStrength') {
      setWaterWaterStrength(val);
      syncControls({ waterWaterStrength: val });
    } else if (type === 'waterPeptideStrength') {
      setWaterPeptideStrength(val);
      syncControls({ waterPeptideStrength: val });
    } else if (type === 'bulkSolventStrength') {
      setBulkSolventStrength(val);
      syncControls({ bulkSolventStrength: val });
    } else if (type === 'waterBeltContraction') {
      setWaterBeltContraction(val);
      syncControls({ waterBeltContraction: val });
    } else if (type === 'watersPerAtom') {
      setWatersPerAtom(val);
      syncControls({ watersPerAtom: val });
    }
  };

  const updateSeqFolding = async (controlAction) => {
    if (!sessionId) return;
    try {
      const response = await fetch(`${BACKEND_URL}/api/control`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId, ...controlAction })
      });
      if (response.ok) {
        const data = await response.json();
        if (data.seqFolding) {
          setSeqFolding(data.seqFolding);
        }
        if (data.params && data.params.waterBeltContraction !== undefined) {
          setWaterBeltContraction(data.params.waterBeltContraction);
        }
      }
    } catch (err) {
      console.error("Failed to update seq folding:", err);
    }
  };

  // Re-run from current state (reset parameters, or trigger re-init)
  const handleReset = () => {
    handleInitialize();
  };

  // Build/Rebuild the Three.js scene objects
  const rebuildScene = (atoms, bonds, initialCoords) => {
    const scene = sceneRef.current;
    if (!scene) return;

    // 1. Clear previous atom meshes, bond meshes, and H-bond lines
    atomMeshesRef.current.forEach(mesh => scene.remove(mesh));
    atomMeshesRef.current = [];

    bondMeshesRef.current.forEach(mesh => scene.remove(mesh.mesh));
    bondMeshesRef.current = [];

    if (hbondLinesRef.current) {
      scene.remove(hbondLinesRef.current);
    }
    hbondLinesRef.current = new THREE.Group();
    scene.add(hbondLinesRef.current);

    // Clear previous water meshes
    waterMeshesRef.current.forEach(mesh => scene.remove(mesh));
    waterMeshesRef.current = [];

    waterBondMeshesRef.current.forEach(mesh => scene.remove(mesh));
    waterBondMeshesRef.current = [];

    if (waterPeptideLinesRef.current) {
      scene.remove(waterPeptideLinesRef.current);
    }
    waterPeptideLinesRef.current = new THREE.Group();
    scene.add(waterPeptideLinesRef.current);

    if (comMeshRef.current) {
      scene.remove(comMeshRef.current);
    }
    if (rotatingHaloRef.current) {
      scene.remove(rotatingHaloRef.current);
    }

    // Element color mapping
    const elementColors = {
      'C': 0x334155, // Dark slate
      'N': 0x3b82f6, // Soft blue
      'O': 0xef4444, // Vibrant coral
      'S': 0xf59e0b  // Warm amber
    };

    // 2. Create atom spheres
    atoms.forEach((atom, idx) => {
      const radius = atom.radius * 0.23;
      const geom = new THREE.SphereGeometry(radius, 32, 32);
      const color = elementColors[atom.element] || 0x94a3b8;
      
      const mat = new THREE.MeshLambertMaterial({
        color: color,
      });
      const mesh = new THREE.Mesh(geom, mat);
      
      const initPos = initialCoords[idx];
      mesh.position.set(initPos[0], initPos[1], initPos[2]);
      scene.add(mesh);
      atomMeshesRef.current.push(mesh);

      // Add visual glow/indicator halos
      if (atom.h_bond !== 'none') {
        // Hydrophilic Cyan aura
        const haloGeom = new THREE.SphereGeometry(radius * 1.45, 16, 16);
        const haloMat = new THREE.MeshBasicMaterial({
          color: 0x22d3ee,
          wireframe: true,
          transparent: true,
          opacity: 0.08
        });
        const halo = new THREE.Mesh(haloGeom, haloMat);
        mesh.add(halo);
      } else if (atom.hydrophobicity > 0.1) {
        // Hydrophobic Orange aura
        const haloGeom = new THREE.SphereGeometry(radius * 1.45, 16, 16);
        const haloMat = new THREE.MeshBasicMaterial({
          color: 0xfb923c,
          wireframe: true,
          transparent: true,
          opacity: 0.08
        });
        const halo = new THREE.Mesh(haloGeom, haloMat);
        mesh.add(halo);
      }
    });

    // 3. Create bond cylinders
    bonds.forEach(bond => {
      const geom = new THREE.CylinderGeometry(0.04, 0.04, 1, 8);
      const color = bond.is_amide ? 0x10b981 : 0x475569; // Amide is Emerald Green, regular is Dark Gray
      const mat = new THREE.MeshLambertMaterial({
        color: color
      });
      const mesh = new THREE.Mesh(geom, mat);
      scene.add(mesh);
      bondMeshesRef.current.push({
        mesh: mesh,
        source: bond.source,
        target: bond.target,
        is_amide: bond.is_amide,
        is_rotatable: bond.is_rotatable,
        residue: bond.residue
      });
    });

    // 4. Create Center of Mass indicator (subtle glowing core)
    const comGeom = new THREE.SphereGeometry(0.12, 16, 16);
    const comMat = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.25
    });
    comMeshRef.current = new THREE.Mesh(comGeom, comMat);
    scene.add(comMeshRef.current);

    // 5. Create active rotating bond indicator (neon glowing spinning ring)
    const haloGeom = new THREE.TorusGeometry(0.35, 0.05, 8, 24);
    const haloMat = new THREE.MeshBasicMaterial({
      color: 0xffea00,
      transparent: true,
      opacity: 0.8,
      wireframe: false
    });
    rotatingHaloRef.current = new THREE.Mesh(haloGeom, haloMat);
    rotatingHaloRef.current.visible = false;
    scene.add(rotatingHaloRef.current);

    // Create water spheres
    const initialWaterCoords = data.initialWaterCoords || [];
    const watersMetadata = data.waters || [];
    watersMetadata.forEach((water, wIdx) => {
      const wGeom = new THREE.SphereGeometry(0.18, 16, 16);
      const wMat = new THREE.MeshLambertMaterial({
        color: 0x22d3ee,
        transparent: true,
        opacity: 0.75
      });
      const mesh = new THREE.Mesh(wGeom, wMat);
      
      const haloGeom = new THREE.SphereGeometry(0.18 * 1.45, 16, 16);
      const haloMat = new THREE.MeshBasicMaterial({
        color: 0x22d3ee,
        wireframe: true,
        transparent: true,
        opacity: 0.08
      });
      const halo = new THREE.Mesh(haloGeom, haloMat);
      mesh.add(halo);
      
      const initPos = initialWaterCoords[wIdx] || [0, 0, 0];
      mesh.position.set(initPos[0], initPos[1], initPos[2]);
      mesh.visible = explicitWaterEnabled && showWaterVisuals;
      scene.add(mesh);
      waterMeshesRef.current.push(mesh);
    });

    // Reset ref coordinate stream
    latestCoordsRef.current = initialCoords;
    latestHBondsRef.current = [];

    // Focus camera on center
    const box = new THREE.Box3();
    initialCoords.forEach(c => box.expandByPoint(new THREE.Vector3(...c)));
    const center = new THREE.Vector3();
    box.getCenter(center);
    if (controlsRef.current) {
      controlsRef.current.target.copy(center);
    }
  };

  // SSE Stream connection effect
  useEffect(() => {
    if (!sessionId) return;

    const eventSource = new EventSource(`${BACKEND_URL}/api/stream?sessionId=${sessionId}`);
    
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      // Diagnostic check for NaN in coordinates
      if (data.coords) {
        const hasNaN = data.coords.some(coord => coord.some(val => isNaN(val)));
        if (hasNaN) {
          console.error("THREE: Coordinates contain NaN values from backend!");
        }
      }

      latestCoordsRef.current = data.coords;
      latestHBondsRef.current = data.activeHBonds;
      latestWaterCoordsRef.current = data.waterCoords || [];
      latestWatersRef.current = data.waters || [];
      latestWaterBondsRef.current = data.activeWaterBonds || [];
      latestWaterPeptideBondsRef.current = data.activeWaterPeptideBonds || [];

      if (data.explicitWaterEnabled !== undefined) {
        setExplicitWaterEnabled(data.explicitWaterEnabled);
      }
      if (data.waterBeltContraction !== undefined) {
        setWaterBeltContraction(data.waterBeltContraction);
      }

      if (data.waters !== undefined) {
        setWaters(data.waters);
      }
      if (data.activeWaterBonds !== undefined) {
        setActiveWaterBonds(data.activeWaterBonds);
      }
      if (data.activeWaterPeptideBonds !== undefined) {
        setActiveWaterPeptideBonds(data.activeWaterPeptideBonds);
      }

      if (data.seqFolding) {
        setSeqFolding(data.seqFolding);
      }

      if (data.hydrophobicExposure !== undefined) {
        setHydrophobicExposure(data.hydrophobicExposure);
      }

      // Handle logs
      if (data.logs && data.logs.length > 0) {
        const newLogs = data.logs.map(logText => {
          let type = 'system';
          if (logText.includes('WaterAgent')) {
            type = 'water-agent';
          } else if (logText.includes('WaterSolvent: Pulled hydrophilic') || logText.includes('WaterSolvent: Formed H-bond')) {
            type = 'hydrophilic';
          } else if (logText.includes('WaterSolvent: Pushed hydrophobic') || logText.includes('WaterSolvent: Hydrophobic collapse')) {
            type = 'hydrophobic';
          } else if (logText.includes('PeptideModel: Rotated joint')) {
            type = 'joint';
          } else if (logText.includes('PeptideModel: Steric clash')) {
            type = 'clash';
          }
          return { text: logText, type };
        });
        setLogs(prev => {
          const combined = [...prev, ...newLogs];
          return combined.slice(-200);
        });
      }
    };

    eventSource.onerror = (err) => {
      console.error("SSE stream error:", err);
      eventSource.close();
    };

    return () => {
      eventSource.close();
    };
  }, [sessionId]);

  // Set up Three.js Canvas and Loop
  useEffect(() => {
    if (rendererType !== 'threejs') {
      setIsSceneReady(false);
      return;
    }
    if (!mountRef.current) return;

    // 1. Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x060913); // Deep space dark blue
    sceneRef.current = scene;

    // 2. Camera setup (start with aspect ratio 1, will resize immediately on observer trigger)
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    camera.position.set(0, 4, 12);

    // 3. Renderer setup
    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true });
    } catch (e) {
      console.error("WebGL Creation Failed:", e);
      setError("WebGL initialization failed. Please make sure WebGL is enabled: " + e.message);
      setIsSceneReady(false);
      return;
    }
    renderer.shadowMap.enabled = true;
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // 4. Controls setup
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controlsRef.current = controls;

    // 5. Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.35);
    scene.add(ambientLight);

    const mainLight = new THREE.DirectionalLight(0xffffff, 0.95);
    mainLight.position.set(10, 15, 10);
    scene.add(mainLight);

    const blueLight = new THREE.DirectionalLight(0x22d3ee, 0.45); // Cyan solvent glow
    blueLight.position.set(-10, -5, -5);
    scene.add(blueLight);

    // 6. Grid floor (subtle background grid)
    const gridHelper = new THREE.GridHelper(30, 30, 0x1e293b, 0x0f172a);
    gridHelper.position.y = -4;
    scene.add(gridHelper);

    // 7. Resize Observer for robust resizing and handling initial layout shifts
    const resizeObserver = new ResizeObserver((entries) => {
      for (let entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          camera.aspect = width / height;
          camera.updateProjectionMatrix();
          renderer.setSize(width, height);
          setViewportSize({ width: Math.round(width), height: Math.round(height) });
        }
      }
    });
    resizeObserver.observe(mountRef.current);

    // Expose for console debugging
    window.scene = scene;
    window.camera = camera;
    window.renderer = renderer;
    window.controls = controls;

    // Signal that canvas is fully initialized
    setIsSceneReady(true);

    // 8. Animation Loop
    let animId;
    const tick = () => {
      animId = requestAnimationFrame(tick);

      const seqFolding = seqFoldingRef.current;
      const explicitWaterEnabled = explicitWaterEnabledRef.current;
      const showWaterVisuals = showWaterVisualsRef.current;
      const moleculeData = moleculeDataRef.current;
      const atomResidues = atomResiduesRef.current;

      // Handle stream coords
      if (latestCoordsRef.current && atomMeshesRef.current.length === latestCoordsRef.current.length) {
        const coords = latestCoordsRef.current;
        
        // Compute Center of Mass
        const com = new THREE.Vector3(0, 0, 0);
        coords.forEach(c => com.add(new THREE.Vector3(...c)));
        com.divideScalar(coords.length);

        if (comMeshRef.current) {
          comMeshRef.current.position.copy(com);
        }

        // Calculate Radius of Gyration
        let sqDistsSum = 0;
        coords.forEach(c => {
          const v = new THREE.Vector3(...c);
          sqDistsSum += v.distanceToSquared(com);
        });
        const rg = Math.sqrt(sqDistsSum / coords.length);

        // Update atom positions and colors
        const elementColors = {
          'C': 0x334155, // Dark slate
          'N': 0x3b82f6, // Soft blue
          'O': 0xef4444, // Vibrant coral
          'S': 0xf59e0b  // Warm amber
        };

        coords.forEach((coord, i) => {
          const mesh = atomMeshesRef.current[i];
          if (mesh) {
            mesh.position.set(coord[0], coord[1], coord[2]);
            
            if (seqFolding && seqFolding.enabled && atomResidues && moleculeData) {
              const resIdx = atomResidues[i];
              const dir = seqFolding.direction || 1;
              const isFrozen = dir === 1 ? resIdx < seqFolding.activeResidue : resIdx > seqFolding.activeResidue;
              const isFuture = dir === 1 ? resIdx > seqFolding.activeResidue + 1 : resIdx < seqFolding.activeResidue - 1;
              
              if (isFrozen) {
                // Frozen atom: desaturated dark slate and semi-transparent
                mesh.material.color.setHex(0x475569);
                mesh.material.opacity = 0.55;
                mesh.material.transparent = true;
              } else if (isFuture) {
                // Future atom: desaturated dark blue and highly transparent
                mesh.material.color.setHex(0x1e293b);
                mesh.material.opacity = 0.15;
                mesh.material.transparent = true;
              } else {
                // Active atom: full vibrant element color
                const baseColor = elementColors[moleculeData.atoms[i].element] || 0x94a3b8;
                mesh.material.color.setHex(baseColor);
                mesh.material.opacity = 1.0;
                mesh.material.transparent = false;
              }
            } else {
              // Regular: full vibrant element color
              if (moleculeData && moleculeData.atoms && moleculeData.atoms[i]) {
                const baseColor = elementColors[moleculeData.atoms[i].element] || 0x94a3b8;
                mesh.material.color.setHex(baseColor);
                mesh.material.opacity = 1.0;
                mesh.material.transparent = false;
              }
            }
          }
        });

        let activeRotatingBondPos = null;
        let activeRotatingBondDir = null;

        // Update bond positions, rotations, and colors
        bondMeshesRef.current.forEach(bondObj => {
          const mesh = bondObj.mesh;
          const posA = new THREE.Vector3(...coords[bondObj.source]);
          const posB = new THREE.Vector3(...coords[bondObj.target]);
          
          // Midpoint
          const midPoint = new THREE.Vector3().addVectors(posA, posB).multiplyScalar(0.5);
          mesh.position.copy(midPoint);

          // Direction and length
          const dir = new THREE.Vector3().subVectors(posB, posA);
          const len = dir.length();
          mesh.scale.set(1, len, 1);

          dir.normalize();
          const bondDirection = dir.clone();
          
          const alignAxis = new THREE.Vector3(0, 1, 0);
          const quaternion = new THREE.Quaternion().setFromUnitVectors(alignAxis, dir);
          mesh.setRotationFromQuaternion(quaternion);

          if (seqFolding && seqFolding.enabled && atomResidues) {
            const resA = atomResidues[bondObj.source];
            const resB = atomResidues[bondObj.target];
            const dir = seqFolding.direction || 1;
            
            const isA_Frozen = dir === 1 ? resA < seqFolding.activeResidue : resA > seqFolding.activeResidue;
            const isB_Frozen = dir === 1 ? resB < seqFolding.activeResidue : resB > seqFolding.activeResidue;
            
            const isA_Future = dir === 1 ? resA > seqFolding.activeResidue + 1 : resA < seqFolding.activeResidue - 1;
            const isB_Future = dir === 1 ? resB > seqFolding.activeResidue + 1 : resB < seqFolding.activeResidue - 1;
            
            const isActiveRotating = bondObj.is_rotatable && bondObj.residue === seqFolding.activeResidue;

            if (isActiveRotating) {
              // Active rotating bond: glowing vibrant yellow-gold, strongly pulsating
              mesh.material.color.setHex(0xffeb3b);
              mesh.material.opacity = 1.0;
              mesh.material.transparent = false;
              if (mesh.material.emissive) {
                mesh.material.emissive.setHex(0xffea00);
                mesh.material.emissiveIntensity = 1.5;
              }
              const pulse = 2.5 + 0.8 * Math.sin(Date.now() * 0.02);
              mesh.scale.set(pulse, len, pulse);

              activeRotatingBondPos = midPoint.clone();
              activeRotatingBondDir = bondDirection;
            } else {
              if (mesh.material.emissive) {
                mesh.material.emissive.setHex(0x000000);
                mesh.material.emissiveIntensity = 0;
              }
              mesh.scale.set(1.0, len, 1.0);

              if (isA_Frozen && isB_Frozen) {
                // Frozen bond: dark slate
                mesh.material.color.setHex(0x334155);
                mesh.material.opacity = 0.4;
                mesh.material.transparent = true;
              } else if (isA_Future || isB_Future) {
                // Future bond: very light/invisible
                mesh.material.color.setHex(0x1e293b);
                mesh.material.opacity = 0.1;
                mesh.material.transparent = true;
              } else {
                // Active bond (non-rotating: e.g. amide)
                const baseColor = bondObj.is_amide ? 0x10b981 : 0x475569;
                mesh.material.color.setHex(baseColor);
                mesh.material.opacity = 1.0;
                mesh.material.transparent = false;
              }
            }
          } else {
            // Regular mode: no highlights on rotatable bonds
            if (mesh.material.emissive) {
              mesh.material.emissive.setHex(0x000000);
              mesh.material.emissiveIntensity = 0;
            }
            mesh.scale.set(1.0, len, 1.0);

            const baseColor = bondObj.is_amide ? 0x10b981 : 0x475569;
            mesh.material.color.setHex(baseColor);
            mesh.material.opacity = 1.0;
            mesh.material.transparent = false;
          }
        });

        // Update rotating halo position and rotation
        if (rotatingHaloRef.current) {
          if (activeRotatingBondPos && seqFolding && seqFolding.enabled) {
            rotatingHaloRef.current.position.copy(activeRotatingBondPos);
            
            // Align torus normal (0, 0, 1) to the bond direction vector
            const alignAxis = new THREE.Vector3(0, 0, 1);
            const quaternion = new THREE.Quaternion().setFromUnitVectors(alignAxis, activeRotatingBondDir);
            rotatingHaloRef.current.setRotationFromQuaternion(quaternion);
            
            // Spin ring around the bond
            rotatingHaloRef.current.rotateZ(Date.now() * 0.005);
            
            // Pulsate diameter
            const scale = 1.0 + 0.3 * Math.sin(Date.now() * 0.02);
            rotatingHaloRef.current.scale.set(scale, scale, 1.0);
            
            rotatingHaloRef.current.visible = true;
          } else {
            rotatingHaloRef.current.visible = false;
          }
        }

        // Update active H-bond lines
        if (hbondLinesRef.current) {
          // Clear previous lines
          while (hbondLinesRef.current.children.length > 0) {
            const line = hbondLinesRef.current.children[0];
            hbondLinesRef.current.remove(line);
          }

          const hbPairs = latestHBondsRef.current || [];
          hbPairs.forEach(pair => {
            const p1 = new THREE.Vector3(...coords[pair[0]]);
            const p2 = new THREE.Vector3(...coords[pair[1]]);
            
            const geom = new THREE.BufferGeometry().setFromPoints([p1, p2]);
            const mat = new THREE.LineDashedMaterial({
              color: 0x22d3ee, // cyan-400
              dashSize: 0.15,
              gapSize: 0.1,
            });
            const line = new THREE.Line(geom, mat);
            line.computeLineDistances();
            hbondLinesRef.current.add(line);
          });
        }

        // Update water spheres positions and colors based on agent states
        const wCoords = latestWaterCoordsRef.current || [];
        const watersData = latestWatersRef.current || [];
        wCoords.forEach((coord, idx) => {
          const mesh = waterMeshesRef.current[idx];
          if (mesh) {
            mesh.position.set(coord[0], coord[1], coord[2]);
            mesh.visible = explicitWaterEnabled && showWaterVisuals;
            
            // Apply state-based visual feedback
            const waterData = watersData[idx];
            if (waterData && waterData.state) {
              const state = waterData.state;
              let bodyColor = 0x22d3ee; // default adsorbed (cyan)
              let haloColor = 0x22d3ee;
              let opacity = 0.75;
              
              if (state === 'searching') {
                bodyColor = 0x3b82f6; // blue
                haloColor = 0x3b82f6;
                opacity = 0.6;
              } else if (state === 'bonded') {
                bodyColor = 0xf0fdfa; // glowing mint-white
                haloColor = 0x22d3ee; // cyan halo
                opacity = 0.95;
              }
              
              if (mesh.material) {
                mesh.material.color.setHex(bodyColor);
                mesh.material.opacity = opacity;
              }
              if (mesh.children[0] && mesh.children[0].material) {
                mesh.children[0].material.color.setHex(haloColor);
              }
            }
          }
        });

        // Clear and rebuild water-water cylinders
        waterBondMeshesRef.current.forEach(mesh => scene.remove(mesh));
        waterBondMeshesRef.current = [];

        const activeWB = latestWaterBondsRef.current || [];
        if (explicitWaterEnabled && showWaterVisuals && wCoords.length > 0) {
          activeWB.forEach(pair => {
            const posA = new THREE.Vector3(...wCoords[pair[0]]);
            const posB = new THREE.Vector3(...wCoords[pair[1]]);
            
            const geom = new THREE.CylinderGeometry(0.05, 0.05, 1, 8);
            const mat = new THREE.MeshLambertMaterial({
              color: 0x06b6d4, // Cyan-500
              emissive: 0x0891b2,
              emissiveIntensity: 0.4
            });
            const mesh = new THREE.Mesh(geom, mat);
            
            const mid = new THREE.Vector3().addVectors(posA, posB).multiplyScalar(0.5);
            mesh.position.copy(mid);
            
            const dir = new THREE.Vector3().subVectors(posB, posA);
            const len = dir.length();
            mesh.scale.set(1, len, 1);
            dir.normalize();
            
            const quaternion = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
            mesh.setRotationFromQuaternion(quaternion);
            
            scene.add(mesh);
            waterBondMeshesRef.current.push(mesh);
          });
        }

        // Clear and rebuild water-peptide dashed lines
        if (waterPeptideLinesRef.current) {
          while (waterPeptideLinesRef.current.children.length > 0) {
            const line = waterPeptideLinesRef.current.children[0];
            waterPeptideLinesRef.current.remove(line);
          }
          
          const activeWPB = latestWaterPeptideBondsRef.current || [];
          if (explicitWaterEnabled && showWaterVisuals && wCoords.length > 0) {
            activeWPB.forEach(pair => {
              const p1 = new THREE.Vector3(...wCoords[pair[0]]);
              const p2 = new THREE.Vector3(...coords[pair[1]]);
              
              const geom = new THREE.BufferGeometry().setFromPoints([p1, p2]);
              const mat = new THREE.LineDashedMaterial({
                color: 0x0891b2,
                dashSize: 0.1,
                gapSize: 0.08
              });
              const line = new THREE.Line(geom, mat);
              line.computeLineDistances();
              waterPeptideLinesRef.current.add(line);
            });
          }
        }

        const hbPairs = latestHBondsRef.current || [];
        updateStats(rg, hbPairs.length, activeWB.length);
      }

      controls.update();
      renderer.render(scene, camera);
    };

    tick();

    return () => {
      cancelAnimationFrame(animId);
      resizeObserver.disconnect();
      controls.dispose();
      renderer.dispose();
      setIsSceneReady(false);
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
    };
  }, [rendererType]);

  const drawMoleculeCanvas2D = (ctx, w, h) => {
    const seqFolding = seqFoldingRef.current;
    const explicitWaterEnabled = explicitWaterEnabledRef.current;
    const showWaterVisuals = showWaterVisualsRef.current;
    const moleculeData = moleculeDataRef.current;

    const coords = latestCoordsRef.current;
    if (!coords || !moleculeData || coords.length === 0) {
      ctx.fillStyle = '#94a3b8';
      ctx.font = '14px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(loading ? '構造モデル読み込み中...' : 'ペプチドデータがありません', w / 2, h / 2);
      return;
    }

    const atoms = moleculeData.atoms;
    const bonds = moleculeData.bonds;
    const hbonds = latestHBondsRef.current || [];

    // 1. Calculate Center of Mass (COM)
    const com = [0, 0, 0];
    coords.forEach(c => {
      com[0] += c[0];
      com[1] += c[1];
      com[2] += c[2];
    });
    com[0] /= coords.length;
    com[1] /= coords.length;
    com[2] /= coords.length;

    // Calculate Anchor (first residue average position)
    const anchor = [0, 0, 0];
    let anchorCount = 0;
    const atomResidues = moleculeData.atomResidues || {};
    coords.forEach((c, idx) => {
      if (atomResidues[idx] === 0) {
        anchor[0] += c[0];
        anchor[1] += c[1];
        anchor[2] += c[2];
        anchorCount++;
      }
    });
    if (anchorCount > 0) {
      anchor[0] /= anchorCount;
      anchor[1] /= anchorCount;
      anchor[2] /= anchorCount;
    } else {
      anchor[0] = com[0];
      anchor[1] = com[1];
      anchor[2] = com[2];
    }

    // Calculate Radius of Gyration
    let sqDistsSum = 0;
    coords.forEach(c => {
      const dx = c[0] - com[0];
      const dy = c[1] - com[1];
      const dz = c[2] - com[2];
      sqDistsSum += dx*dx + dy*dy + dz*dz;
    });
    const rg = Math.sqrt(sqDistsSum / coords.length);

    const activeWB = latestWaterBondsRef.current || [];
    // Update stats using our helper
    updateStats(rg, hbonds.length, activeWB.length);

    // 2. Project coordinates: translate by anchor and rotate
    const thetaX = rotXRef.current;
    const thetaY = rotYRef.current;
    const cosX = Math.cos(thetaX);
    const sinX = Math.sin(thetaX);
    const cosY = Math.cos(thetaY);
    const sinY = Math.sin(thetaY);

    const centerOffset = initialCenterRef.current || [0, 0, 0];

    // Project Center of Mass (COM) relative to anchor + initialCenter
    const comX = com[0] - anchor[0] - centerOffset[0];
    const comY = com[1] - anchor[1] - centerOffset[1];
    const comZ = com[2] - anchor[2] - centerOffset[2];

    const comX1 = comX * cosY - comZ * sinY;
    const comZ1 = comX * sinY + comZ * cosY;
    const comY1 = comY;

    const comX2 = comX1;
    const comY2 = comY1 * cosX - comZ1 * sinX;

    const comScreenX = w / 2 + comX2 * zoomRef.current;
    const comScreenY = h / 2 - comY2 * zoomRef.current;

    // Project Anchor position relative to anchor + initialCenter
    const anchorX = -centerOffset[0];
    const anchorY = -centerOffset[1];
    const anchorZ = -centerOffset[2];

    const anchorX1 = anchorX * cosY - anchorZ * sinY;
    const anchorZ1 = anchorX * sinY + anchorZ * cosY;
    const anchorY1 = anchorY;

    const anchorX2 = anchorX1;
    const anchorY2 = anchorY1 * cosX - anchorZ1 * sinX;

    const anchorScreenX = w / 2 + anchorX2 * zoomRef.current;
    const anchorScreenY = h / 2 - anchorY2 * zoomRef.current;

    const projected = coords.map((c) => {
      // Translate relative to anchor + centerOffset
      const x = c[0] - anchor[0] - centerOffset[0];
      const y = c[1] - anchor[1] - centerOffset[1];
      const z = c[2] - anchor[2] - centerOffset[2];

      // Rotate around Y axis (left/right)
      const x1 = x * cosY - z * sinY;
      const z1 = x * sinY + z * cosY;
      const y1 = y;

      // Rotate around X axis (up/down)
      const x2 = x1;
      const y2 = y1 * cosX - z1 * sinX;
      const z2 = y1 * sinX + z1 * cosX;

      return {
        x: x2,
        y: y2,
        z: z2, // Depth
        screenX: w / 2 + x2 * zoomRef.current,
        screenY: h / 2 - y2 * zoomRef.current // Flip Y for canvas coords
      };
    });

    const projectedWaters = (latestWaterCoordsRef.current || []).map((c) => {
      // Translate relative to anchor + centerOffset
      const x = c[0] - anchor[0] - centerOffset[0];
      const y = c[1] - anchor[1] - centerOffset[1];
      const z = c[2] - anchor[2] - centerOffset[2];

      const x1 = x * cosY - z * sinY;
      const z1 = x * sinY + z * cosY;
      const y1 = y;

      const x2 = x1;
      const y2 = y1 * cosX - z1 * sinX;
      const z2 = y1 * sinX + z1 * cosX;

      return {
        x: x2,
        y: y2,
        z: z2, // Depth
        screenX: w / 2 + x2 * zoomRef.current,
        screenY: h / 2 - y2 * zoomRef.current
      };
    });

    // 3. Build a list of drawable items (Atoms, Bonds, Hbonds)
    const drawItems = [];

    // Add water spheres and water bonds
    if (explicitWaterEnabled && showWaterVisuals && projectedWaters.length > 0) {
      projectedWaters.forEach((proj, idx) => {
        if (!proj) return;
        const radius = 0.16; // Water radius
        drawItems.push({
          type: 'water',
          depth: proj.z,
          draw: () => {
            const rad = radius * zoomRef.current;
            const state = latestWatersRef.current[idx]?.state || 'searching';
            let haloColor = 'rgba(59, 130, 246, 0.2)';
            let centerColor = '#ffffff';
            let midColor = '#3b82f6'; // blue-500
            let edgeColor = '#1d4ed8'; // blue-700
            let haloRadMult = 1.45;

            if (state === 'adsorbed') {
              haloColor = 'rgba(34, 211, 238, 0.2)';
              midColor = '#22d3ee'; // cyan-400
              edgeColor = '#0891b2'; // cyan-700
            } else if (state === 'bonded') {
              haloColor = 'rgba(34, 211, 238, 0.5)';
              midColor = '#f0fdfa'; // glowing mint-white
              edgeColor = '#22d3ee'; // cyan-400
              haloRadMult = 1.6;
            }

            // Draw water halo
            ctx.beginPath();
            ctx.arc(proj.screenX, proj.screenY, rad * haloRadMult, 0, Math.PI * 2);
            ctx.strokeStyle = haloColor;
            ctx.lineWidth = 1.5;
            ctx.stroke();

            // Shaded sphere
            ctx.beginPath();
            ctx.arc(proj.screenX, proj.screenY, rad, 0, Math.PI * 2);
            const grad = ctx.createRadialGradient(
              proj.screenX - rad * 0.3, proj.screenY - rad * 0.3, rad * 0.1,
              proj.screenX, proj.screenY, rad
            );
            grad.addColorStop(0, centerColor);
            grad.addColorStop(0.3, midColor);
            grad.addColorStop(1, edgeColor);
            ctx.fillStyle = grad;
            ctx.fill();

            // Water label
            ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
            ctx.font = `bold ${Math.max(7, Math.round(rad * 0.75))}px sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('W', proj.screenX, proj.screenY);
          }
        });
      });

      // Add water-water bonds
      activeWB.forEach(pair => {
        const projA = projectedWaters[pair[0]];
        const projB = projectedWaters[pair[1]];
        if (!projA || !projB) return;
        const avgZ = (projA.z + projB.z) / 2;

        drawItems.push({
          type: 'water-bond',
          depth: avgZ,
          draw: () => {
            ctx.beginPath();
            ctx.moveTo(projA.screenX, projA.screenY);
            ctx.lineTo(projB.screenX, projB.screenY);
            ctx.strokeStyle = '#06b6d4';
            ctx.lineWidth = Math.max(2, 0.05 * zoomRef.current);
            ctx.lineCap = 'round';
            ctx.stroke();
          }
        });
      });

      // Add water-peptide bonds
      const activeWPB = latestWaterPeptideBondsRef.current || [];
      activeWPB.forEach(pair => {
        const projW = projectedWaters[pair[0]];
        const projP = projected[pair[1]];
        if (!projW || !projP) return;
        const avgZ = (projW.z + projP.z) / 2;

        drawItems.push({
          type: 'water-peptide-bond',
          depth: avgZ,
          draw: () => {
            ctx.beginPath();
            ctx.moveTo(projW.screenX, projW.screenY);
            ctx.lineTo(projP.screenX, projP.screenY);
            ctx.strokeStyle = 'rgba(8, 145, 178, 0.7)';
            ctx.lineWidth = Math.max(1, 0.03 * zoomRef.current);
            ctx.setLineDash([3, 3]);
            ctx.stroke();
            ctx.setLineDash([]);
          }
        });
      });
    }

    // Element color mapping
    const elementColors = {
      'C': '#334155', // Dark slate
      'N': '#3b82f6', // Soft blue
      'O': '#ef4444', // Vibrant coral
      'S': '#f59e0b'  // Warm amber
    };

    const lightColors = {
      'C': '#475569',
      'N': '#60a5fa',
      'O': '#f87171',
      'S': '#fbbf24'
    };

    // Add atoms
    atoms.forEach((atom, idx) => {
      const proj = projected[idx];
      if (!proj) return;
      const radius = atom.radius * 0.23;
      let baseColor = elementColors[atom.element] || '#94a3b8';
      let lightColor = lightColors[atom.element] || '#cbd5e1';

      let isFrozen = false;
      let isFuture = false;

      if (seqFolding && seqFolding.enabled && atomResidues) {
        const resIdx = atomResidues[idx];
        const dir = seqFolding.direction || 1;
        const isResFrozen = dir === 1 ? resIdx < seqFolding.activeResidue : resIdx > seqFolding.activeResidue;
        const isResFuture = dir === 1 ? resIdx > seqFolding.activeResidue + 1 : resIdx < seqFolding.activeResidue - 1;
        
        if (isResFrozen) {
          isFrozen = true;
          baseColor = '#475569';
          lightColor = '#64748b';
        } else if (isResFuture) {
          isFuture = true;
          baseColor = 'rgba(30, 41, 59, 0.15)';
          lightColor = 'rgba(71, 85, 105, 0.15)';
        }
      }

      drawItems.push({
        type: 'atom',
        depth: proj.z,
        index: idx,
        draw: () => {
          const rad = radius * zoomRef.current;
          
          // Draw halo indicator (cyan for hydrophilic / orange for hydrophobic)
          if (!isFuture && (atom.h_bond !== 'none' || atom.hydrophobicity > 0.1)) {
            ctx.beginPath();
            ctx.arc(proj.screenX, proj.screenY, rad * 1.4, 0, Math.PI * 2);
            ctx.strokeStyle = atom.h_bond !== 'none' ? 'rgba(34, 211, 238, 0.25)' : 'rgba(251, 146, 60, 0.25)';
            ctx.lineWidth = 1.5;
            if (atom.h_bond !== 'none') {
              ctx.setLineDash([3, 3]);
            } else {
              ctx.setLineDash([]);
            }
            ctx.stroke();
            ctx.setLineDash([]); // Reset line dash
          }

          // Shaded sphere using radial gradient
          ctx.beginPath();
          ctx.arc(proj.screenX, proj.screenY, rad, 0, Math.PI * 2);
          
          const grad = ctx.createRadialGradient(
            proj.screenX - rad * 0.3, proj.screenY - rad * 0.3, rad * 0.1,
            proj.screenX, proj.screenY, rad
          );
          
          if (isFuture) {
            grad.addColorStop(0, 'rgba(255, 255, 255, 0.2)');
            grad.addColorStop(0.3, 'rgba(71, 85, 105, 0.15)');
            grad.addColorStop(1, 'rgba(30, 41, 59, 0.10)');
            ctx.fillStyle = grad;
          } else if (isFrozen) {
            grad.addColorStop(0, 'rgba(255, 255, 255, 0.4)');
            grad.addColorStop(0.3, lightColor);
            grad.addColorStop(1, baseColor);
            ctx.fillStyle = grad;
          } else {
            grad.addColorStop(0, '#ffffff');
            grad.addColorStop(0.3, lightColor);
            grad.addColorStop(1, baseColor);
            ctx.fillStyle = grad;
          }
          ctx.fill();

          // Element label overlay for nitrogen, oxygen, sulfur (if zoom is large enough and not future)
          if (!isFuture && zoomRef.current > 25) {
            ctx.fillStyle = isFrozen ? 'rgba(255, 255, 255, 0.4)' : 'rgba(255, 255, 255, 0.7)';
            ctx.font = `bold ${Math.max(9, Math.round(rad * 0.9))}px sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(atom.element, proj.screenX, proj.screenY);
          }
        }
      });
    });

    // Add bonds
    bonds.forEach((bond) => {
      const projA = projected[bond.source];
      const projB = projected[bond.target];
      if (!projA || !projB) return;
      const avgZ = (projA.z + projB.z) / 2;

      let strokeColor = bond.is_amide ? '#10b981' : '#475569';
      let isFutureBond = false;
      let isRotating = false;

      if (seqFolding && seqFolding.enabled && atomResidues) {
        const resA = atomResidues[bond.source];
        const resB = atomResidues[bond.target];
        const dir = seqFolding.direction || 1;
        
        const isA_Frozen = dir === 1 ? resA < seqFolding.activeResidue : resA > seqFolding.activeResidue;
        const isB_Frozen = dir === 1 ? resB < seqFolding.activeResidue : resB > seqFolding.activeResidue;
        
        const isA_Future = dir === 1 ? resA > seqFolding.activeResidue + 1 : resA < seqFolding.activeResidue - 1;
        const isB_Future = dir === 1 ? resB > seqFolding.activeResidue + 1 : resB < seqFolding.activeResidue - 1;
        
        if (bond.is_rotatable && bond.residue === seqFolding.activeResidue) {
          isRotating = true;
          strokeColor = '#f59e0b';
        } else if (isA_Frozen && isB_Frozen) {
          // Frozen bond: dark slate
          strokeColor = '#334155';
        } else if (isA_Future || isB_Future) {
          // Future bond: very light
          strokeColor = 'rgba(30, 41, 59, 0.1)';
          isFutureBond = true;
        }
      } else {
        strokeColor = bond.is_amide ? '#10b981' : '#475569';
      }

      drawItems.push({
        type: 'bond',
        depth: avgZ,
        draw: () => {
          ctx.beginPath();
          ctx.moveTo(projA.screenX, projA.screenY);
          ctx.lineTo(projB.screenX, projB.screenY);
          ctx.strokeStyle = strokeColor;
          
          let width = isFutureBond ? 1 : Math.max(2, 0.06 * zoomRef.current);
          if (isRotating) {
            const pulse = 2.5 + 0.8 * Math.sin(Date.now() * 0.02);
            width = width * pulse;
            ctx.shadowColor = '#ffea00';
            ctx.shadowBlur = 15;
            strokeColor = '#ffea00';
          }
          
          ctx.lineWidth = width;
          ctx.lineCap = 'round';
          ctx.stroke();
          ctx.shadowBlur = 0;

          if (isRotating) {
            // Draw a spinning dashed halo around the midpoint of the rotating bond
            const midX = (projA.screenX + projB.screenX) / 2;
            const midY = (projA.screenY + projB.screenY) / 2;
            
            ctx.beginPath();
            ctx.strokeStyle = '#ffea00';
            ctx.lineWidth = 2.0;
            ctx.setLineDash([6, 6]);
            ctx.lineDashOffset = (Date.now() / 25) % 12;
            
            const radius = 16 + 5 * Math.sin(Date.now() * 0.02);
            ctx.arc(midX, midY, radius, 0, Math.PI * 2);
            ctx.stroke();
            ctx.setLineDash([]); // Reset line dash
          }
        }
      });
    });

    // Add hydrogen bonds
    hbonds.forEach((pair) => {
      const projA = projected[pair[0]];
      const projB = projected[pair[1]];
      if (!projA || !projB) return;
      const avgZ = (projA.z + projB.z) / 2;

      drawItems.push({
        type: 'hbond',
        depth: avgZ,
        draw: () => {
          ctx.beginPath();
          ctx.moveTo(projA.screenX, projA.screenY);
          ctx.lineTo(projB.screenX, projB.screenY);
          ctx.strokeStyle = 'rgba(34, 211, 238, 0.85)';
          ctx.lineWidth = Math.max(1.5, 0.04 * zoomRef.current);
          ctx.setLineDash([4, 4]);
          ctx.stroke();
          ctx.setLineDash([]); // Reset line dash
        }
      });
    });

    // 4. Sort all items by depth: Painter's Algorithm (draw back-to-front)
    drawItems.sort((a, b) => a.depth - b.depth);

    // 5. Draw all items
    drawItems.forEach(item => item.draw());

    // 6. Draw Center of Mass (COM) marker at its projected screen position
    ctx.beginPath();
    ctx.arc(comScreenX, comScreenY, 4, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.fill();

    // Subtle crosshair at anchor position
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(anchorScreenX - 8, anchorScreenY);
    ctx.lineTo(anchorScreenX + 8, anchorScreenY);
    ctx.moveTo(anchorScreenX, anchorScreenY - 8);
    ctx.lineTo(anchorScreenX, anchorScreenY + 8);
    ctx.stroke();

    // Draw a small cyan dot at the anchor point to indicate it is locked
    ctx.beginPath();
    ctx.arc(anchorScreenX, anchorScreenY, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = '#22d3ee';
    ctx.fill();
  };

  // Canvas 2D Loop
  useEffect(() => {
    if (rendererType !== 'canvas2d') return;

    let animId;
    const canvas = canvas2DRef.current;
    if (!canvas) return;

    // Handle high DPI displays
    const resizeCanvas = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      setViewportSize({ width: Math.round(rect.width), height: Math.round(rect.height) });
    };

    // Initial resize
    resizeCanvas();

    // Resize observer
    const resizeObserver = new ResizeObserver(() => {
      resizeCanvas();
    });
    resizeObserver.observe(canvas);

    const render = () => {
      animId = requestAnimationFrame(render);
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const dpr = window.devicePixelRatio || 1;
      const width = canvas.width;
      const height = canvas.height;

      // Clear with background color
      ctx.fillStyle = '#060913';
      ctx.fillRect(0, 0, width, height);

      // Save state for scaling (high DPI support)
      ctx.save();
      ctx.scale(dpr, dpr);

      // Run our drawing function
      drawMoleculeCanvas2D(ctx, width / dpr, height / dpr);

      ctx.restore();
    };

    render();

    return () => {
      cancelAnimationFrame(animId);
      resizeObserver.disconnect();
    };
  }, [rendererType, moleculeData]);

  return (
    <div className="app-layout">
      {/* Header */}
      <header className="app-header">
        <div className="header-brand">
          <div className="brand-icon-wrapper">
            <Dna className="w-5 h-5 text-white" />
          </div>
          <div className="brand-title-group">
            <h1 className="brand-title">
              水溶媒エージェント × ペプチド分子模型
              <span className="brand-badge">対話型3Dシミュレーター</span>
            </h1>
            <p className="brand-subtitle">
              RDKit解析 ＆ ねじれ角空間(Torsion-space)動力学による軽量折り畳みプロトタイプ
            </p>
          </div>
        </div>

        {/* Metrics */}
        <div className="header-metrics">
          <div className="metric-item">
            <span className="metric-label">慣性半径 (Rg)</span>
            <span className="metric-value indigo">
              {gyrationRadius > 0 ? `${gyrationRadius.toFixed(3)} Å` : '計測中...'}
            </span>
          </div>
          <div className="metric-divider" />
          <div className="metric-item">
            <span className="metric-label">水素結合数</span>
            <span className="metric-value cyan">
              {activeHBondCount}
            </span>
          </div>
          <div className="metric-divider" />
          <div className="metric-item">
            <span className="metric-label">疎水基水接触度</span>
            <span className={`metric-value ${
              hydrophobicExposure > 0.6 ? 'red' : hydrophobicExposure > 0.35 ? 'orange' : 'green'
            }`}>
              {(hydrophobicExposure * 100).toFixed(0)}%
            </span>
          </div>
          <div className="metric-divider" />
          <div className="metric-item">
            <span className="metric-label">水の帯形成率</span>
            <span className="metric-value cyan" style={{ color: '#22d3ee' }}>
              {(waterBeltRatio * 100).toFixed(0)}% ({activeWaterBonds.length}/{waters.length > 1 ? waters.length - 1 : 0})
            </span>
          </div>
          <div className="metric-divider" />
          {/* Renderer Toggle Segmented Button */}
          <div className="renderer-toggle" style={{
            display: 'flex',
            backgroundColor: 'rgba(15, 23, 42, 0.6)',
            borderRadius: '8px',
            padding: '3px',
            border: '1px solid rgba(255, 255, 255, 0.08)'
          }}>
            <button
              onClick={() => setRendererType('canvas2d')}
              style={{
                padding: '4px 10px',
                borderRadius: '6px',
                fontSize: '11px',
                fontWeight: '600',
                border: 'none',
                cursor: 'pointer',
                backgroundColor: rendererType === 'canvas2d' ? 'rgba(34, 211, 238, 0.2)' : 'transparent',
                color: rendererType === 'canvas2d' ? '#22d3ee' : '#94a3b8',
                transition: 'all 0.2s ease',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                margin: 0
              }}
            >
              <span>2D Canvas</span>
            </button>
            <button
              onClick={() => setRendererType('threejs')}
              style={{
                padding: '4px 10px',
                borderRadius: '6px',
                fontSize: '11px',
                fontWeight: '600',
                border: 'none',
                cursor: 'pointer',
                backgroundColor: rendererType === 'threejs' ? 'rgba(59, 130, 246, 0.2)' : 'transparent',
                color: rendererType === 'threejs' ? '#3b82f6' : '#94a3b8',
                transition: 'all 0.2s ease',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                margin: 0
              }}
            >
              <span>3D WebGL</span>
            </button>
          </div>
          <div className="metric-divider" />
          <div className="status-badge">
            <div className={`pulsing-dot ${isPlaying ? '' : 'paused'}`} />
            <span className="status-badge-text">
              {isPlaying ? 'Running' : 'Paused'}
            </span>
          </div>
        </div>
      </header>

      {/* Main View Area */}
      <div className="app-body">
        {/* Left Sidebar */}
        <div className={`panel-sidebar ${showSidebar ? '' : 'collapsed'}`}>
          {/* Section 1: Simulations Controller (Moved to Top for Visibility) */}
          <div className="glass-panel" style={{ padding: '14px' }}>
            <h2 className="panel-title">
              <Activity className="w-4 h-4 text-emerald-400" /> シミュレーション制御
            </h2>
            <div className="control-buttons">
              <button
                onClick={() => updateParameter('isPlaying', !isPlaying)}
                className="primary"
                style={{
                  flex: 1,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '6px',
                  background: isPlaying ? 'linear-gradient(135deg, #ef4444, #dc2626)' : 'linear-gradient(135deg, #10b981, #059669)',
                  color: 'white',
                  boxShadow: isPlaying ? '0 4px 14px rgba(239, 68, 68, 0.4)' : '0 4px 14px rgba(16, 185, 129, 0.4)',
                  borderColor: 'rgba(255,255,255,0.2)'
                }}
              >
                {isPlaying ? (
                  <>
                    <Pause className="w-4 h-4" /> シミュレーション一時停止
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" /> シミュレーション再開
                  </>
                )}
              </button>
              <button
                onClick={handleReset}
                className="secondary"
                style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '10px' }}
                title="シミュレーションを初期状態に戻す"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Section 1.5: Sequential Folding Controller */}
          <div className="glass-panel" style={{ padding: '14px' }}>
            <h2 className="panel-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Compass className="w-4 h-4 text-cyan-400" /> 逐次折り畳みプロセス (N→C末端)
            </h2>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginTop: '4px' }}>
              {/* Toggle Mode */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="slider-title" style={{ fontSize: '12px', color: '#94a3b8' }}>逐次折り畳みモード</span>
                <button
                  onClick={() => updateSeqFolding({ seqFoldingEnabled: !seqFolding.enabled })}
                  className={seqFolding.enabled ? "primary" : "secondary"}
                  style={{
                    padding: '4px 10px',
                    fontSize: '11px',
                    borderRadius: '6px',
                    background: seqFolding.enabled ? 'linear-gradient(135deg, #06b6d4, #0891b2)' : 'rgba(255,255,255,0.06)',
                    borderColor: seqFolding.enabled ? 'rgba(255,255,255,0.2)' : 'rgba(255,255,255,0.08)',
                    color: 'white',
                    fontWeight: 'bold',
                    boxShadow: seqFolding.enabled ? '0 2px 8px rgba(6, 182, 212, 0.4)' : 'none',
                    margin: 0
                  }}
                >
                  {seqFolding.enabled ? "ON (有効)" : "OFF (無効)"}
                </button>
              </div>

              {seqFolding.enabled && (
                <>
                  {/* Status Banner */}
                  <div style={{
                    backgroundColor: seqFolding.status === 'completed' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(34, 211, 238, 0.1)',
                    border: seqFolding.status === 'completed' ? '1px dashed #10b981' : '1px dashed #22d3ee',
                    padding: '8px',
                    borderRadius: '8px',
                    fontSize: '11px',
                    color: seqFolding.status === 'completed' ? '#34d399' : '#22d3ee',
                    textAlign: 'center',
                    fontFamily: 'monospace'
                  }}>
                    {seqFolding.status === 'completed' ? (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', alignItems: 'center' }}>
                        <span>🎉 全アミノ酸の逐次折り畳みが完了しました！</span>
                        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', justifyContent: 'center', marginTop: '2px' }}>
                          <button
                            onClick={() => setShowWaterVisuals(!showWaterVisuals)}
                            className={showWaterVisuals ? "primary" : "secondary"}
                            style={{
                              padding: '4px 10px',
                              fontSize: '10px',
                              borderRadius: '6px',
                              fontWeight: 'bold',
                              cursor: 'pointer',
                              background: showWaterVisuals ? 'linear-gradient(135deg, #0ea5e9, #2563eb)' : 'rgba(255,255,255,0.06)',
                              border: '1px solid rgba(255,255,255,0.15)',
                              color: 'white',
                              margin: 0
                            }}
                          >
                            {showWaterVisuals ? "💧 水分子を非表示にする" : "💧 水分子を表示する"}
                          </button>
                          <button
                            onClick={() => updateSeqFolding({ seqFoldingEnabled: true })}
                            className="primary"
                            style={{
                              padding: '4px 10px',
                              fontSize: '10px',
                              borderRadius: '6px',
                              fontWeight: 'bold',
                              cursor: 'pointer',
                              background: 'linear-gradient(135deg, #10b981, #059669)',
                              border: '1px solid rgba(255,255,255,0.15)',
                              color: 'white',
                              margin: 0
                            }}
                          >
                            🔄 現在の構造からさらに折り畳む（ループ実行）
                          </button>
                        </div>
                      </div>
                    ) : (
                      <span>
                        ⚙️ {seqFolding.activeResidue + 1}番目のアミノ酸を折り畳み中...
                        <br />
                        ({seqFolding.currentStep} / {seqFolding.stepsPerResidue} ステップ)
                      </span>
                    )}
                  </div>

                  {/* Progress Bar */}
                  {seqFolding.status !== 'completed' && (
                    <div style={{
                      width: '100%',
                      height: '4px',
                      backgroundColor: 'rgba(255, 255, 255, 0.08)',
                      borderRadius: '2px',
                      overflow: 'hidden'
                    }}>
                      <div style={{
                        width: `${(seqFolding.currentStep / seqFolding.stepsPerResidue) * 100}%`,
                        height: '100%',
                        backgroundColor: '#22d3ee',
                        boxShadow: '0 0 8px #22d3ee',
                        transition: 'width 0.1s linear'
                      }} />
                    </div>
                  )}

                  {/* Timeline representation */}
                  <div style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: '6px',
                    padding: '4px 0',
                    justifyContent: 'flex-start'
                  }}>
                    {residueLabels.map((label, idx) => {
                      const dir = seqFolding.direction || 1;
                      const isResFrozen = dir === 1 ? idx < seqFolding.activeResidue : idx > seqFolding.activeResidue;
                      const isResActive = idx === seqFolding.activeResidue;
                      const isResNextActive = dir === 1 ? idx === seqFolding.activeResidue + 1 : idx === seqFolding.activeResidue - 1;
                      const isResFuture = dir === 1 ? idx > seqFolding.activeResidue + 1 : idx < seqFolding.activeResidue - 1;

                      let bgColor = 'rgba(255, 255, 255, 0.04)';
                      let textColor = '#64748b';
                      let borderColor = 'rgba(255, 255, 255, 0.08)';
                      let shadow = 'none';

                      if (isResFrozen) {
                        bgColor = 'rgba(71, 85, 105, 0.2)';
                        textColor = '#94a3b8';
                        borderColor = '#475569';
                      } else if (isResActive) {
                        bgColor = 'rgba(6, 182, 212, 0.25)';
                        textColor = '#22d3ee';
                        borderColor = '#06b6d4';
                        shadow = '0 0 8px rgba(6, 182, 212, 0.5)';
                      } else if (isResNextActive) {
                        bgColor = 'rgba(139, 92, 246, 0.15)'; // Soft purple
                        textColor = '#a78bfa'; // Soft purple-ish
                        borderColor = 'rgba(139, 92, 246, 0.4)';
                        shadow = 'none';
                      } else if (isResFuture) {
                        bgColor = 'transparent';
                        textColor = '#475569';
                        borderColor = 'rgba(255, 255, 255, 0.04)';
                      }

                      return (
                        <button
                          key={idx}
                          onClick={() => updateSeqFolding({ seqFoldingActiveResidue: idx })}
                          style={{
                            padding: '3px 8px',
                            borderRadius: '6px',
                            fontSize: '10px',
                            fontFamily: 'monospace',
                            fontWeight: isResActive ? 'bold' : 'normal',
                            background: bgColor,
                            color: textColor,
                            border: `1.5px solid ${borderColor}`,
                            cursor: 'pointer',
                            boxShadow: shadow,
                            transition: 'all 0.2s ease',
                            margin: 0
                          }}
                          title={`${label}を折り畳む（以前の部分は固定）`}
                        >
                          {label}
                        </button>
                      );
                    })}
                  </div>

                  {/* Navigation controls */}
                  <div style={{ display: 'flex', gap: '6px', width: '100%' }}>
                    <button
                      onClick={() => updateSeqFolding({ seqFoldingActiveResidue: Math.max(0, seqFolding.activeResidue - 1) })}
                      disabled={seqFolding.activeResidue === 0}
                      className="secondary"
                      style={{ flex: 1, padding: '6px', fontSize: '11px', borderRadius: '6px', margin: 0 }}
                    >
                      ◀ 前
                    </button>
                    <button
                      onClick={() => updateSeqFolding({ seqFoldingActiveResidue: Math.min(seqFolding.nResidues - 1, seqFolding.activeResidue + 1) })}
                      disabled={seqFolding.activeResidue >= seqFolding.nResidues - 1}
                      className="secondary"
                      style={{ flex: 1, padding: '6px', fontSize: '11px', borderRadius: '6px', margin: 0 }}
                    >
                      次 ▶
                    </button>
                    <button
                      onClick={() => updateSeqFolding({ seqFoldingReset: true })}
                      className="secondary"
                      style={{ padding: '6px 10px', fontSize: '11px', borderRadius: '6px', margin: 0 }}
                      title="逐次折り畳みをリセット"
                    >
                      🔄
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Section 1.6: Explicit Water Solvent Config */}
          <div className="glass-panel" style={{ padding: '14px' }}>
            <h2 className="panel-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Droplet className="w-4 h-4 text-cyan-400" /> 明示的水溶媒ベルト制御
            </h2>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginTop: '4px' }}>
              {/* Toggle Explicit Water */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="slider-title" style={{ fontSize: '12px', color: '#94a3b8' }}>明示的水シミュレーション</span>
                <button
                  onClick={() => updateParameter('explicitWaterEnabled', !explicitWaterEnabled)}
                  className={explicitWaterEnabled ? "primary" : "secondary"}
                  style={{
                    padding: '4px 10px',
                    fontSize: '11px',
                    borderRadius: '6px',
                    background: explicitWaterEnabled ? 'linear-gradient(135deg, #0ea5e9, #2563eb)' : 'rgba(255,255,255,0.06)',
                    borderColor: explicitWaterEnabled ? 'rgba(255,255,255,0.2)' : 'rgba(255,255,255,0.08)',
                    color: 'white',
                    fontWeight: 'bold',
                    margin: 0
                  }}
                >
                  {explicitWaterEnabled ? "ON (有効)" : "OFF (無効)"}
                </button>
              </div>

              {/* Toggle Water Visibility */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '4px' }}>
                <span className="slider-title" style={{ fontSize: '12px', color: '#94a3b8' }}>水分子 of 描画（表示）</span>
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

              <div style={{
                padding: '10px',
                borderRadius: '8px',
                background: 'rgba(255,255,255,0.03)',
                border: '1px solid rgba(255,255,255,0.05)',
                fontSize: '11px',
                display: 'flex',
                flexDirection: 'column',
                gap: '8px',
                color: '#94a3b8',
                marginTop: '4px'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>バルク溶媒圧効果 (K_solv):</span>
                  <strong className="orange" style={{ color: '#fb923c' }}>{bulkSolventStrength.toFixed(2)}</strong>
                </div>
                {explicitWaterEnabled && (
                  <>
                    <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid rgba(255,255,255,0.05)', paddingTop: '8px' }}>
                      <span>水の帯の結合強度 (K_ww):</span>
                      <strong className="cyan" style={{ color: '#22d3ee' }}>{waterWaterStrength.toFixed(2)}</strong>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>水の帯の収縮率 (K_contract):</span>
                      <strong className="cyan" style={{ color: '#22d3ee' }}>{waterBeltContraction.toFixed(2)}</strong>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>親水基あたりの水分子数:</span>
                      <strong className="cyan" style={{ color: '#22d3ee' }}>{watersPerAtom} 分子</strong>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>水-親水基の引力強度 (K_wp):</span>
                      <strong className="blue" style={{ color: '#60a5fa' }}>{waterPeptideStrength.toFixed(2)}</strong>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>


          {/* Section 2: Peptide Input */}
          <div className="glass-panel" style={{ padding: '14px' }}>
            <h2 className="panel-title">
              <Dna className="w-4 h-4 text-blue-400" /> ペプチド鎖の定義
            </h2>
            <div className="form-group">
              <label className="form-label">アミノ酸配列 (1文字/3文字) / SMILES</label>
              <input
                type="text"
                value={sequence}
                onChange={(e) => setSequence(e.target.value)}
                placeholder="例: Ala-Ala-Ala / ACDEF / CC(=O)N..."
                disabled={loading}
              />
              <button
                onClick={handleInitialize}
                disabled={loading}
                className="primary"
                style={{ marginTop: '4px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}
              >
                {loading ? '解析中...' : '構造モデルの生成 ＆ 読込'}
              </button>
            </div>
            {error && (
              <div className="error-message" style={{ marginTop: '8px' }}>
                {error}
              </div>
            )}
          </div>

          {/* Section 3: Parameters Info */}
          <div className="glass-panel" style={{ padding: '14px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <h2 className="panel-title">
              <Sliders className="w-4 h-4 text-cyan-400" /> シミュレーション環境情報
            </h2>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              fontSize: '11px',
              color: '#94a3b8',
              padding: '4px 0'
            }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <Thermometer className="w-3.5 h-3.5 text-blue-400" /> 🌡️ 溶媒温度 (熱ゆらぎ)
              </span>
              <strong className="blue" style={{ color: '#60a5fa' }}>{temperature.toFixed(2)}</strong>
            </div>
            <span style={{ fontSize: '10px', color: '#64748b' }}>
              全原子にランダムな力（ホワイトノイズ）を加え、分子の熱ゆらぎをシミュレートしています。
            </span>
          </div>

          {/* Legends */}
          <div className="legend-box">
            <span className="legend-title">凡例:</span>
            <div className="legend-item">
              <span className="legend-color-dot" style={{ backgroundColor: '#3b82f6' }} />
              <span>窒素 (N) / <span className="legend-color-dot" style={{ backgroundColor: '#ef4444' }} /> 酸素 (O)</span>
            </div>
            <div className="legend-item">
              <span className="legend-color-dot" style={{ backgroundColor: '#334155' }} />
              <span>炭素 (C) / <span className="legend-color-dot" style={{ backgroundColor: '#f59e0b' }} /> 硫黄 (S)</span>
            </div>
            <div className="legend-item">
              <span className="legend-color-ring" style={{ borderColor: '#22d3ee', backgroundColor: 'rgba(6,182,212,0.05)' }} />
              <span>親水基 (シアンの網掛けオーラ)</span>
            </div>
            <div className="legend-item">
              <span className="legend-color-ring" style={{ borderColor: '#fb923c', backgroundColor: 'rgba(249,115,22,0.05)' }} />
              <span>疎水基 (オレンジの網掛けオーラ)</span>
            </div>
            <div className="legend-item">
              <span className="legend-line" style={{ backgroundColor: '#10b981' }} />
              <span>アミド結合 (緑: 平面固定・非可動)</span>
            </div>
            <div className="legend-item">
              <span className="legend-line" style={{ backgroundColor: '#f59e0b', height: '4px', boxShadow: '0 0 6px #f59e0b' }} />
              <span>回転中の結合 (黄: 脈動・光彩)</span>
            </div>
            <div className="legend-item">
              <span className="legend-line-dashed" style={{ borderColor: '#22d3ee' }} />
              <span>水素結合 (シアン of 破線)</span>
            </div>
            {explicitWaterEnabled && (
              <>
                <div className="legend-item">
                  <span className="legend-color-dot" style={{ backgroundColor: '#22d3ee', width: '10px', height: '10px', borderRadius: '50%', border: '1px solid rgba(255,255,255,0.15)', display: 'inline-block' }} />
                  <span>水分子エージェント (W: シアン小球)</span>
                </div>
                <div className="legend-item">
                  <span className="legend-line" style={{ backgroundColor: '#06b6d4', height: '3px' }} />
                  <span>水の帯 (水-水結合)</span>
                </div>
                <div className="legend-item">
                  <span className="legend-line-dashed" style={{ borderColor: 'rgba(8, 145, 178, 0.7)' }} />
                  <span>水-親水基結合 (破線)</span>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Toggle Sidebar Button */}
        <button
          onClick={() => setShowSidebar(!showSidebar)}
          className={`toggle-btn-left ${showSidebar ? '' : 'collapsed'}`}
          title={showSidebar ? "サイドバーをたたむ" : "サイドバーを開く"}
        >
          {showSidebar ? <ChevronLeft className="w-5 h-5" /> : <ChevronRight className="w-5 h-5" />}
        </button>

        {/* Canvas Renderers: Three.js (WebGL) or Canvas 2D (Compatibility Fallback) */}
        {rendererType === 'threejs' ? (
          <div ref={mountRef} className="canvas-container" key="threejs-mount" />
        ) : (
          <canvas 
            ref={canvas2DRef} 
            className="canvas-container" 
            key="canvas2d-mount"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
            onWheel={handleWheel}
            onTouchStart={handleTouchStart}
            onTouchMove={handleTouchMove}
            onTouchEnd={handleTouchEnd}
          />
        )}

        {/* Viewport Size Overlay */}
        <div style={{
          position: 'absolute',
          bottom: '20px',
          left: showSidebar ? '360px' : '20px',
          backgroundColor: 'rgba(15, 23, 42, 0.75)',
          color: '#94a3b8',
          padding: '6px 12px',
          borderRadius: '6px',
          fontSize: '11px',
          fontFamily: 'monospace',
          zIndex: 5,
          pointerEvents: 'none',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          transition: 'left 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          display: 'flex',
          alignItems: 'center',
          gap: '6px'
        }}>
          <span style={{ display: 'inline-block', width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#10b981' }} />
          <span>Viewport: {viewportSize.width} × {viewportSize.height}px</span>
        </div>

        {/* Toggle Logs Button */}
        <button
          onClick={() => setShowLogs(!showLogs)}
          className={`toggle-btn-right ${showLogs ? '' : 'collapsed'}`}
          title={showLogs ? "ログをたたむ" : "ログを開く"}
        >
          {showLogs ? <ChevronRight className="w-5 h-5" /> : <ChevronLeft className="w-5 h-5" />}
        </button>

        {/* Right Sidebar Logs */}
        <div className={`panel-logs ${showLogs ? '' : 'collapsed'}`}>
          <div className="log-monitor glass-panel">
            <div className="log-monitor-header">
              <h2 className="log-monitor-title">
                <Terminal className="w-4 h-4 text-indigo-400" /> エージェントの対話モニター
              </h2>
              <span className="log-monitor-meta">Real-time</span>
            </div>
            
            <div className="log-monitor-body">
              {logs.map((log, idx) => {
                let colorClass = 'log-system';
                if (log.type === 'hydrophilic') colorClass = 'log-water-hydrophilic';
                else if (log.type === 'hydrophobic') colorClass = 'log-water-hydrophobic';
                else if (log.type === 'joint') colorClass = 'log-peptide-joint';
                else if (log.type === 'clash') colorClass = 'log-peptide-clash';
                else if (log.type === 'water-agent') colorClass = 'log-water-agent';
                
                return (
                  <div key={idx} className={`log-line ${colorClass}`} style={{ display: 'flex', alignItems: 'flex-start', gap: '4px' }}>
                    <ChevronRight className="w-3 h-3" style={{ marginTop: '2px', opacity: 0.4, flexShrink: 0 }} />
                    <span>{log.text}</span>
                  </div>
                );
              })}
              <div ref={logsEndRef} />
            </div>
          </div>
        </div>
      </div>

      {/* Floating JavaScript Error Overlay for Debugging */}
      {jsErrors.length > 0 && (
        <div style={{
          position: 'fixed',
          bottom: '20px',
          left: '20px',
          right: '20px',
          backgroundColor: 'rgba(220, 38, 38, 0.95)',
          color: 'white',
          padding: '16px',
          borderRadius: '8px',
          zIndex: 9999,
          maxHeight: '200px',
          overflowY: 'auto',
          boxShadow: '0 10px 25px rgba(0, 0, 0, 0.5)',
          border: '2px solid #ef4444',
          fontFamily: 'monospace',
          fontSize: '12px'
        }}>
          <h3 style={{ margin: '0 0 8px 0', fontSize: '14px', fontWeight: 'bold' }}>Uncaught Browser Errors:</h3>
          {jsErrors.map((err, i) => (
            <pre key={i} style={{ margin: '0 0 6px 0', whiteSpace: 'pre-wrap' }}>{err}</pre>
          ))}
          <button 
            onClick={() => setJsErrors([])} 
            style={{ 
              marginTop: '8px', 
              backgroundColor: 'white', 
              color: '#dc2626', 
              border: 'none', 
              padding: '4px 10px', 
              borderRadius: '4px', 
              fontWeight: 'bold', 
              cursor: 'pointer' 
            }}
          >
            Clear Errors
          </button>
        </div>
      )}
    </div>
  );
}
