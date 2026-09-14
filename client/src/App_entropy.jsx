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
  Shield
} from 'lucide-react';
import './App.css';

const BACKEND_HOST = window.location.hostname || 'localhost';
const BACKEND_URL = `http://${BACKEND_HOST}:5003`;

// Helper to align cylinder meshes between two points
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

export default function AppEntropy() {
  const [sequence, setSequence] = useState('Ala-Ala-Ala-Ala-Ala');
  const [sessionId, setSessionId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // UI Panels Toggle State
  const [showSidebar, setShowSidebar] = useState(true);
  const [showLogs, setShowLogs] = useState(true);

  // Simulation parameter states
  const [temperature, setTemperature] = useState(0.25);
  const [isPlaying, setIsPlaying] = useState(true);
  const [explicitWaterEnabled, setExplicitWaterEnabled] = useState(true);
  const [waterWaterStrength, setWaterWaterStrength] = useState(0.65);
  const [waterPeptideStrength, setWaterPeptideStrength] = useState(0.75);
  const [bulkSolventStrength, setBulkSolventStrength] = useState(0.45);
  const [waterBeltContraction, setWaterBeltContraction] = useState(0.0);
  const [watersPerAtom, setWatersPerAtom] = useState(4);
  const [kCapture, setKCapture] = useState(0.70);
  const [entropyBias, setEntropyBias] = useState(1.2);

  // Dynamic calculated stats
  const [activeHBondCount, setActiveHBondCount] = useState(0);
  const [gyrationRadius, setGyrationRadius] = useState(0.0);
  const [hydrophobicExposure, setHydrophobicExposure] = useState(0.0);
  const [informationCaptureIndex, setInformationCaptureIndex] = useState(0.0);
  const [averageWaterEntropy, setAverageWaterEntropy] = useState(0.6);
  const [atomCount, setAtomCount] = useState(0);
  const [showWaterVisuals, setShowWaterVisuals] = useState(true);
  const [residueLabels, setResidueLabels] = useState([]);

  // Logs
  const [logs, setLogs] = useState([]);
  const logEndRef = useRef(null);

  // Sequential Folding State
  const [seqFolding, setSeqFolding] = useState({
    enabled: false,
    activeResidue: 0,
    currentStep: 0,
    stepsPerResidue: 500,
    status: 'idle',
    nResidues: 0,
    residueLabels: [],
    direction: 1
  });

  // Three.js refs
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const controlsRef = useRef(null);
  
  // Scene objects refs
  const atomMeshesRef = useRef([]);      // Peptide heavy atoms
  const bondMeshesRef = useRef([]);      // Peptide heavy bonds
  const waterMeshesRef = useRef([]);     // H2O objects { oxygen, hydrogen1, hydrogen2, bond1, bond2 }
  const hbondLinesRef = useRef(null);    // Group for peptide H-bonds
  const waterBondsGroupRef = useRef(null); // Group for water-water hydrogen bonds
  const waterPeptideLinesRef = useRef(null); // Group for water-peptide hydrogen bonds
  const activePathsGroupRef = useRef(null); // Group for active information paths (gold cylinders)
  const backboneTubeRef = useRef(null);    // Smooth ribbon/tube for backbone
  
  // Atom residue mapping
  const [atomResidues, setAtomResidues] = useState({});

  const [moleculeData, setMoleculeData] = useState(null);
  const [isSceneReady, setIsSceneReady] = useState(false);

  const explicitWaterEnabledRef = useRef(explicitWaterEnabled);
  const showWaterVisualsRef = useRef(showWaterVisuals);
  const atomResiduesRef = useRef(atomResidues);
  const seqFoldingRef = useRef(seqFolding);

  useEffect(() => { explicitWaterEnabledRef.current = explicitWaterEnabled; }, [explicitWaterEnabled]);
  useEffect(() => { showWaterVisualsRef.current = showWaterVisuals; }, [showWaterVisuals]);
  useEffect(() => { atomResiduesRef.current = atomResidues; }, [atomResidues]);
  useEffect(() => { seqFoldingRef.current = seqFolding; }, [seqFolding]);

  useEffect(() => {
    if (moleculeData && isSceneReady) {
      rebuildScene(moleculeData.atoms, moleculeData.bonds, moleculeData.initialCoords, moleculeData);
    }
  }, [moleculeData, isSceneReady]);

  // Physics streams buffers
  const latestCoordsRef = useRef(null);
  const latestHBondsRef = useRef(null);
  const latestWaterCoordsRef = useRef([]);
  const latestWatersRef = useRef([]);
  const latestWaterBondsRef = useRef([]);
  const latestWaterPeptideBondsRef = useRef([]);
  const latestWaterEntropiesRef = useRef([]);
  const latestActiveInformationPathsRef = useRef([]);

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
          watersPerAtom,
          kCapture,
          entropyBias
        })
      });
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || 'Failed to initialize molecule.');
      }
      const data = await response.json();
      setSessionId(data.sessionId);
      setAtomCount(data.atoms.length);
      
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
        stepsPerResidue: 500,
        status: 'idle',
        nResidues: data.residueLabels ? data.residueLabels.length : 0,
        residueLabels: data.residueLabels || []
      });
      
      setActiveHBondCount(0);
      setGyrationRadius(0.0);
      setInformationCaptureIndex(0.0);
      setAverageWaterEntropy(0.6);
      
      setMoleculeData(data);
      setLogs([{ text: `System: Initialized entropy simulation for peptide: ${sequence}`, type: 'system' }]);
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

  // Update a single control parameter to Flask backend
  const syncControls = async (paramsObj) => {
    if (!sessionId) return;
    try {
      await fetch(`${BACKEND_URL}/api/control`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sessionId, ...paramsObj })
      });
    } catch (err) {
      console.error("Failed to sync controls:", err);
    }
  };

  // Sync state dynamically with slider changes
  const updateParameter = (type, val) => {
    if (type === 'temperature') {
      setTemperature(val);
      syncControls({ temperature: val });
    } else if (type === 'kCapture') {
      setKCapture(val);
      syncControls({ kCapture: val });
    } else if (type === 'entropyBias') {
      setEntropyBias(val);
      syncControls({ entropyBias: val });
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

  const handleReset = () => {
    handleInitialize();
  };

  // Build/Rebuild the Three.js scene objects
  const rebuildScene = (atoms, bonds, initialCoords, data) => {
    try {
      const scene = sceneRef.current;
      if (!scene) return;

      if (backboneTubeRef.current) {
        scene.remove(backboneTubeRef.current);
      }

      // 1. Remove all old meshes
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

      // 2. Rebuild peptide atom meshes
      atoms.forEach((atom, idx) => {
        const coord = initialCoords[idx];
        if (!coord) {
          throw new Error(`initialCoords[${idx}] is undefined (total atoms: ${atoms.length})`);
        }
        let color = 0x9ca3af; // Grey for carbon
        if (atom.symbol === 'O') color = 0xef4444; // Red
        else if (atom.symbol === 'N') color = 0x3b82f6; // Blue
        else if (atom.symbol === 'S') color = 0xeab308; // Yellow
        
        // Override hydrophobic residues with distinctive orange color
        if (atom.hydrophobicity > 0.15) {
          color = 0xf97316; // Orange
        }

        const geom = new THREE.SphereGeometry(atom.radius * 0.45, 24, 24);
        const mat = new THREE.MeshPhongMaterial({
          color: color,
          shininess: 80,
          specular: 0x111111
        });
        const mesh = new THREE.Mesh(geom, mat);
        mesh.position.set(coord[0], coord[1], coord[2]);
        scene.add(mesh);
        atomMeshesRef.current.push(mesh);
      });

      // 3. Rebuild peptide bond geometries
      bonds.forEach(bond => {
        if (bond.source === undefined || bond.target === undefined) {
          throw new Error(`bond.source or bond.target is undefined in bond: ${JSON.stringify(bond)}`);
        }
        const posA = new THREE.Vector3(...initialCoords[bond.source]);
        const posB = new THREE.Vector3(...initialCoords[bond.target]);
        
        const thickness = bond.is_rotatable ? 0.08 : 0.05;
        const color = bond.is_rotatable ? 0x10b981 : 0xd1d5db; // Green for rotatable, grey for rigid
        
        const geom = new THREE.CylinderGeometry(thickness, thickness, 1, 16);
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

      // 4. Add H2O water molecules (Oxygen sphere + two Hydrogen spheres + two covalent cylinders)
      const initialWaterCoords = data.waterCoords || [];
      const watersMetadata = data.waters || [];
      
      initialWaterCoords.forEach((coord, idx) => {
        if (!coord || coord.length < 3) {
          throw new Error(`initialWaterCoords[${idx}] is invalid`);
        }
        const geomO = new THREE.SphereGeometry(0.20, 12, 12);
        const geomH = new THREE.SphereGeometry(0.12, 10, 10);
        
        const matO = new THREE.MeshPhongMaterial({
          color: 0x38bdf8, // Ice-blue default
          shininess: 60,
          specular: 0x050505
        });
        const matH = new THREE.MeshPhongMaterial({
          color: 0xf8fafc, // White
          shininess: 20
        });
        const matBond = new THREE.MeshBasicMaterial({ color: 0x475569 });
        
        const oMesh = new THREE.Mesh(geomO, matO);
        const hMesh1 = new THREE.Mesh(geomH, matH);
        const hMesh2 = new THREE.Mesh(geomH, matH);
        
        const bGeom1 = new THREE.CylinderGeometry(0.03, 0.03, 0.96, 6);
        const bGeom2 = new THREE.CylinderGeometry(0.03, 0.03, 0.96, 6);
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

      // Reset stream buffers
      latestCoordsRef.current = initialCoords;
      latestHBondsRef.current = [];
      latestWaterCoordsRef.current = initialWaterCoords;
      latestWatersRef.current = watersMetadata;
      latestWaterBondsRef.current = [];
      latestWaterPeptideBondsRef.current = [];
      latestWaterEntropiesRef.current = [];
      latestActiveInformationPathsRef.current = [];

      // Focus camera
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
    scene.background = new THREE.Color(0x070b13); // Deep cyber black-blue
    sceneRef.current = scene;
    
    // Camera
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    camera.position.set(0, 5, 20);
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
    dirLight1.position.set(10, 15, 10);
    scene.add(dirLight1);
    
    const dirLight2 = new THREE.DirectionalLight(0x0ea5e9, 0.20); // Cyan fill
    dirLight2.position.set(-10, -5, -10);
    scene.add(dirLight2);
    
    // Add lines groups
    const hbLines = new THREE.Group();
    scene.add(hbLines);
    hbondLinesRef.current = hbLines;

    const wHBGroup = new THREE.Group();
    scene.add(wHBGroup);
    waterBondsGroupRef.current = wHBGroup;

    const wpLines = new THREE.Group();
    scene.add(wpLines);
    waterPeptideLinesRef.current = wpLines;

    const activePaths = new THREE.Group();
    scene.add(activePaths);
    activePathsGroupRef.current = activePaths;

    // Handle Resize using ResizeObserver
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
    
    setIsSceneReady(true);
    
    // Animation Loop
    let animId;
    const animate = () => {
      try {
        animId = requestAnimationFrame(animate);
        controls.update();
        
        const coords = latestCoordsRef.current;
        if (coords) {
          const sf = seqFoldingRef.current;
          const isCompleted = sf && sf.status === 'completed';

          // 1. Update peptide atom positions
          coords.forEach((coord, idx) => {
            const mesh = atomMeshesRef.current[idx];
            if (mesh) {
              mesh.position.set(coord[0], coord[1], coord[2]);
              
              const atomData = moleculeData ? moleculeData.atoms[idx] : null;
              const isBackboneAtom = atomData && (atomData.name === 'N' || atomData.name === 'CA' || atomData.name === 'C' || atomData.name === 'O' || atomData.name === 'OXT');
              
              if (isCompleted) {
                if (isBackboneAtom) {
                  mesh.material.transparent = true;
                  mesh.material.opacity = 0.95;
                  mesh.scale.set(1.05, 1.05, 1.05);
                } else {
                  mesh.material.transparent = true;
                  mesh.material.opacity = 0.20; 
                  mesh.scale.set(0.7, 0.7, 0.7);
                }
              } else {
                mesh.material.transparent = false;
                mesh.material.opacity = 1.0;
                
                const atomResId = atomResiduesRef.current[idx];
                if (sf && sf.enabled && sf.status === 'folding' && atomResId === sf.activeResidue) {
                  mesh.scale.set(1.15, 1.15, 1.15);
                } else {
                  mesh.scale.set(1.0, 1.0, 1.0);
                }
              }
            }
          });
          
          // 2. Update peptide bond geometries
          bondMeshesRef.current.forEach(item => {
            const posA = new THREE.Vector3(...coords[item.source]);
            const posB = new THREE.Vector3(...coords[item.target]);
            updateCylinder(item.mesh, posA, posB);
            
            const atomSrc = moleculeData ? moleculeData.atoms[item.source] : null;
            const atomTgt = moleculeData ? moleculeData.atoms[item.target] : null;
            const isBackboneBond = atomSrc && atomTgt &&
              (atomSrc.name === 'N' || atomSrc.name === 'CA' || atomSrc.name === 'C' || atomSrc.name === 'O' || atomSrc.name === 'OXT') &&
              (atomTgt.name === 'N' || atomTgt.name === 'CA' || atomTgt.name === 'C' || atomTgt.name === 'O' || atomTgt.name === 'OXT');
              
            if (isCompleted) {
              if (isBackboneBond) {
                item.mesh.material.transparent = true;
                item.mesh.material.opacity = 0.8;
              } else {
                item.mesh.material.transparent = true;
                item.mesh.material.opacity = 0.15;
              }
            } else {
              item.mesh.material.transparent = false;
              item.mesh.material.opacity = 1.0;
            }
          });

          // 3. Rebuild peptide H-bond dashed lines (Faint red pulse)
          if (hbondLinesRef.current) {
            while (hbondLinesRef.current.children.length > 0) {
              hbondLinesRef.current.remove(hbondLinesRef.current.children[0]);
            }
            
            const hbonds = latestHBondsRef.current || [];
            setActiveHBondCount(hbonds.length);
            
            hbonds.forEach(bond => {
              const p1 = new THREE.Vector3(...coords[bond[0]]);
              const p2 = new THREE.Vector3(...coords[bond[1]]);
              
              const geom = new THREE.BufferGeometry().setFromPoints([p1, p2]);
              const mat = new THREE.LineDashedMaterial({
                color: 0xef4444, 
                dashSize: 0.12,
                gapSize: 0.08,
              });
              const line = new THREE.Line(geom, mat);
              line.computeLineDistances();
              hbondLinesRef.current.add(line);
            });
          }

          // 4. Update smooth backbone ribbon/tube
          const showRibbon = false; // Tube is optional
          if (showRibbon) {
            const points = [];
            moleculeData.atoms.forEach((atom, idx) => {
              if (atom.name === 'N' || atom.name === 'CA' || atom.name === 'C') {
                if (coords[idx]) {
                  points.push(new THREE.Vector3(...coords[idx]));
                }
              }
            });
            if (points.length > 1) {
              const curve = new THREE.CatmullRomCurve3(points);
              const tubeGeom = new THREE.TubeGeometry(curve, 64, 0.12, 8, false);
              if (backboneTubeRef.current) {
                scene.remove(backboneTubeRef.current);
              }
              const tubeMat = new THREE.MeshPhongMaterial({
                color: 0x06b6d4,
                emissive: 0x0891b2,
                emissiveIntensity: 0.3,
                shininess: 90,
                transparent: true,
                opacity: 0.75
              });
              const tubeMesh = new THREE.Mesh(tubeGeom, tubeMat);
              scene.add(tubeMesh);
              backboneTubeRef.current = tubeMesh;
            }
          } else {
            if (backboneTubeRef.current) {
              scene.remove(backboneTubeRef.current);
              backboneTubeRef.current = null;
            }
          }

          // 5. Update H2O water spheres, hydrogens and covalent O-H bonds
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
              
              // Dynamic color based on local entropy
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

          // 6. Draw regular H-bonds as semi-transparent grey background lines
          if (waterBondsGroupRef.current) {
            while (waterBondsGroupRef.current.children.length > 0) {
              waterBondsGroupRef.current.remove(waterBondsGroupRef.current.children[0]);
            }
            
            const activeWB = latestWaterBondsRef.current || [];
            if (explicitWaterEnabledRef.current && showWaterVisualsRef.current && wCoords.length > 0) {
              activeWB.forEach(pair => {
                const w1_idx = pair[0];
                const w2_idx = pair[1];
                const h_idx = pair[2];
                
                if (wCoords[w1_idx] && wCoords[w2_idx] && wCoords[w2_idx][h_idx]) {
                  const posA = new THREE.Vector3(...wCoords[w1_idx][0]); 
                  const posB = new THREE.Vector3(...wCoords[w2_idx][h_idx]); 
                  
                  const geom = new THREE.BufferGeometry().setFromPoints([posA, posB]);
                  const mat = new THREE.LineDashedMaterial({
                    color: 0x475569, 
                    dashSize: 0.08,
                    gapSize: 0.12,
                    transparent: true,
                    opacity: 0.30
                  });
                  const line = new THREE.Line(geom, mat);
                  line.computeLineDistances();
                  waterBondsGroupRef.current.add(line);
                }
              });
            }
          }

          // 7. Redraw active information flow channels (Thick gold pipelines)
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

          // Calculate radius of gyration locally
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
      setIsSceneReady(false);
      if (backboneTubeRef.current) {
        scene.remove(backboneTubeRef.current);
        backboneTubeRef.current = null;
      }
      if (mountContainer) {
        while (mountContainer.firstChild) {
          mountContainer.removeChild(mountContainer.firstChild);
        }
      }
    };
  }, []);

  // Subscribe to SSE stream from Flask server
  useEffect(() => {
    if (!sessionId) return;

    const eventSource = new EventSource(`${BACKEND_URL}/api/stream?sessionId=${sessionId}`);
    
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      latestCoordsRef.current = data.coords;
      latestHBondsRef.current = data.activeHBonds;
      latestWaterCoordsRef.current = data.waterCoords || [];
      latestWatersRef.current = data.waters || [];
      latestWaterBondsRef.current = data.activeWaterBonds || [];
      latestWaterPeptideBondsRef.current = data.activeWaterPeptideBonds || [];
      latestWaterEntropiesRef.current = data.waterEntropies || [];
      latestActiveInformationPathsRef.current = data.activeInformationPaths || [];

      if (data.explicitWaterEnabled !== undefined) {
        setExplicitWaterEnabled(data.explicitWaterEnabled);
      }
      if (data.temperature !== undefined) {
        setTemperature(data.temperature);
      }
      if (data.waterBeltContraction !== undefined) {
        setWaterBeltContraction(data.waterBeltContraction);
      }
      if (data.informationCaptureIndex !== undefined) {
        setInformationCaptureIndex(data.informationCaptureIndex);
      }
      if (data.waterEntropies && data.waterEntropies.length > 0) {
        const avg = data.waterEntropies.reduce((a, b) => a + b, 0) / data.waterEntropies.length;
        setAverageWaterEntropy(avg);
      }

      if (data.seqFolding) {
        setSeqFolding(data.seqFolding);
      }

      if (data.hydrophobicExposure !== undefined) {
        setHydrophobicExposure(data.hydrophobicExposure);
      }

      if (data.logs && data.logs.length > 0) {
        const newLogs = data.logs.map(logText => {
          let type = 'system';
          if (logText.includes('Network') || logText.includes('information')) {
            type = 'water-agent';
          } else if (logText.includes('bond formed')) {
            type = 'h-bond';
          }
          return { text: logText, type };
        });
        setLogs(prev => [...prev, ...newLogs].slice(-100)); // cap at 100 logs
      }
    };

    eventSource.onerror = () => {
      setError("EventSource connection lost. Attempting reconnect...");
    };

    return () => {
      eventSource.close();
    };
  }, [sessionId]);

  // Autoscroll logs
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  return (
    <div className="app-layout">
      {/* 1. Header Toolbar */}
      <header className="app-header">
        <div className="header-left">
          <Droplet className="w-6 h-6 text-cyan-400 animate-pulse" />
          <h1 className="header-title">エントロピー＆情報フロー水分子ベルトシミュレーター</h1>
          <span className="badge">Entropy Mode</span>
        </div>
        
        {/* Real-time floating telemetry in Header to keep it sleek */}
        <div className="telemetry-bar" style={{ display: 'flex', gap: '16px', alignItems: 'center', marginLeft: 'auto', marginRight: '24px', fontSize: '11px', fontFamily: 'monospace' }}>
          <div>
            <span style={{ color: '#94a3b8' }}>I_capture:</span>{' '}
            <span style={{ color: '#fbbf24', fontWeight: 'bold' }}>{informationCaptureIndex.toFixed(4)}</span>
          </div>
          <div style={{ width: '1px', height: '14px', backgroundColor: 'rgba(255,255,255,0.1)' }} />
          <div>
            <span style={{ color: '#94a3b8' }}>S_avg:</span>{' '}
            <span style={{ color: '#22d3ee', fontWeight: 'bold' }}>{averageWaterEntropy.toFixed(4)}</span>
          </div>
          <div style={{ width: '1px', height: '14px', backgroundColor: 'rgba(255,255,255,0.1)' }} />
          <div>
            <span style={{ color: '#94a3b8' }}>Rg:</span>{' '}
            <span style={{ color: '#34d399', fontWeight: 'bold' }}>{gyrationRadius.toFixed(2)} Å</span>
          </div>
          <div style={{ width: '1px', height: '14px', backgroundColor: 'rgba(255,255,255,0.1)' }} />
          <div>
            <span style={{ color: '#94a3b8' }}>H_exposure:</span>{' '}
            <span style={{ color: '#fb923c', fontWeight: 'bold' }}>{hydrophobicExposure.toFixed(2)} Å</span>
          </div>
        </div>
      </header>

      {/* 2. Main Work Area */}
      <div className="app-body">
        {/* Left Sidebar controls */}
        <div className={`panel-sidebar ${showSidebar ? '' : 'collapsed'}`}>
          {/* Section 1: Sequence Initializer */}
          <div className="glass-panel" style={{ padding: '14px' }}>
            <h2 className="panel-title">
              <Dna className="w-4 h-4 text-cyan-400" /> アミノ酸配列入力
            </h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '6px' }}>
              <button 
                onClick={handleInitialize} 
                className="primary" 
                disabled={loading}
                style={{ width: '100%', padding: '8px 14px', fontSize: '13px', whiteSpace: 'nowrap', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}
              >
                {loading ? '読み込み中...' : '初期化'}
              </button>
              <input
                type="text"
                value={sequence}
                onChange={(e) => setSequence(e.target.value)}
                placeholder="例: Gly-Tyr-Asp-Pro-Glu-Thr-Gly-Thr-Trp-Gly"
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
          </div>

          {/* Section 2: Simulations Controller */}
          <div className="glass-panel" style={{ padding: '14px' }}>
            <h2 className="panel-title">
              <Activity className="w-4 h-4 text-cyan-400" /> シミュレーション制御
            </h2>
            <div style={{ display: 'flex', gap: '8px', marginTop: '4px' }}>
              <button
                onClick={() => updateParameter('isPlaying', !isPlaying)}
                className="primary"
                style={{
                  flex: 1,
                  margin: 0,
                  padding: '8px 10px',
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                  gap: '6px',
                  background: isPlaying ? 'linear-gradient(135deg, #ef4444, #dc2626)' : 'linear-gradient(135deg, #10b981, #059669)',
                  color: 'white',
                  boxShadow: isPlaying ? '0 4px 14px rgba(239, 68, 68, 0.4)' : '0 4px 14px rgba(16, 185, 129, 0.4)',
                  borderColor: 'rgba(255,255,255,0.2)'
                }}
              >
                {isPlaying ? (
                  <>
                    <Pause className="w-4 h-4" /> 一時停止
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4" /> 再生開始
                  </>
                )}
              </button>
              <button
                onClick={handleReset}
                className="secondary"
                style={{ margin: 0, padding: '8px 12px' }}
              >
                <RotateCcw className="w-4 h-4" /> リセット
              </button>
            </div>
          </div>

          {/* Section 3: Information & Entropy Parameters */}
          <div className="glass-panel" style={{ padding: '14px' }}>
            <h2 className="panel-title">
              <Share2 className="w-4 h-4 text-yellow-400" /> 情報キャプチャ＆エントロピー
            </h2>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginTop: '4px' }}>
              {/* Slider: kCapture */}
              <div className="slider-group">
                <div className="slider-header">
                  <span className="slider-title">🔗 情報キャプチャ駆動強度 (K_capture)</span>
                  <span className="slider-value yellow">{kCapture.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="1.5"
                  step="0.05"
                  value={kCapture}
                  onChange={(e) => updateParameter('kCapture', parseFloat(e.target.value))}
                  className="slider-yellow"
                />
                <div style={{ fontSize: '10px', color: '#64748b', marginTop: '2px' }}>
                  水分子ネットワークがペプチドの親水基構造を「キャプチャ」してフォールディングを引き寄せる力。
                </div>
              </div>

              {/* Slider: entropyBias */}
              <div className="slider-group">
                <div className="slider-header">
                  <span className="slider-title">🌡️ エントロピーコントラスト倍率</span>
                  <span className="slider-value cyan">{entropyBias.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="2.0"
                  step="0.1"
                  value={entropyBias}
                  onChange={(e) => updateParameter('entropyBias', parseFloat(e.target.value))}
                  className="slider-cyan"
                />
                <div style={{ fontSize: '10px', color: '#64748b', marginTop: '2px' }}>
                  親水基結合水（低エントロピー・氷青色）と疎水基周囲水（高エントロピー・黄金色）のコントラスト差。
                </div>
              </div>
            </div>
          </div>

          {/* Section 4: Sequential Folding Controller */}
          <div className="glass-panel" style={{ padding: '14px' }}>
            <h2 className="panel-title" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Compass className="w-4 h-4 text-cyan-400" /> 逐次折り畳みプロセス
            </h2>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', marginTop: '4px' }}>
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
                    margin: 0
                  }}
                >
                  {seqFolding.enabled ? "ON (有効)" : "OFF (無効)"}
                </button>
              </div>

              {seqFolding.enabled && (
                <>
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
                        <span>🎉 順次折り畳みが完了しました！水ベルトの収縮率を有効化中。</span>
                        <button
                          onClick={() => updateSeqFolding({ seqFoldingReset: true })}
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
                          🔄 現在の構造から再折り畳み(ループ)
                        </button>
                      </div>
                    ) : (
                      <span>
                        ⚙️ {seqFolding.activeResidue + 1}番目の残基を折り畳み中...
                        <br />
                        ({seqFolding.currentStep} / {seqFolding.stepsPerResidue} steps)
                      </span>
                    )}
                  </div>

                  {seqFolding.status !== 'completed' && (
                    <div style={{ width: '100%', height: '4px', backgroundColor: 'rgba(255, 255, 255, 0.08)', borderRadius: '2px', overflow: 'hidden' }}>
                      <div style={{
                        width: `${(seqFolding.currentStep / seqFolding.stepsPerResidue) * 100}%`,
                        height: '100%',
                        backgroundColor: '#22d3ee',
                        boxShadow: '0 0 8px #22d3ee',
                        transition: 'width 0.1s linear'
                      }} />
                    </div>
                  )}

                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', padding: '4px 0' }}>
                    {residueLabels.map((label, idx) => {
                      const isResActive = idx === seqFolding.activeResidue;
                      return (
                        <button
                          key={idx}
                          onClick={() => updateSeqFolding({ seqFoldingActiveResidue: idx })}
                          style={{
                            padding: '3px 8px',
                            borderRadius: '6px',
                            fontSize: '10px',
                            fontFamily: 'monospace',
                            background: isResActive ? 'rgba(6, 182, 212, 0.25)' : 'rgba(255, 255, 255, 0.04)',
                            color: isResActive ? '#22d3ee' : '#94a3b8',
                            border: `1.5px solid ${isResActive ? '#06b6d4' : 'rgba(255, 255, 255, 0.08)'}`,
                            cursor: 'pointer',
                            margin: 0
                          }}
                        >
                          {label}
                        </button>
                      );
                    })}
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Section 5: Standard H2O Solvent Parameter Sliders */}
          <div className="glass-panel" style={{ padding: '14px' }}>
            <h2 className="panel-title">
              <Sliders className="w-4 h-4 text-cyan-400" /> 水分子ネットワークパラメータ
            </h2>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginTop: '4px' }}>
              {/* Toggle Water Visibility */}
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
                  max="0.4"
                  step="0.01"
                  value={temperature}
                  onChange={(e) => updateParameter('temperature', parseFloat(e.target.value))}
                  className="slider-blue"
                />
              </div>

              {/* Slider: Bulk Solvent Pressure */}
              <div className="slider-group">
                <div className="slider-header">
                  <span className="slider-title">🌊 バルク溶媒効果圧強度 (K_solv)</span>
                  <span className="slider-value orange">{bulkSolventStrength.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.05"
                  value={bulkSolventStrength}
                  onChange={(e) => updateParameter('bulkSolventStrength', parseFloat(e.target.value))}
                  className="slider-orange"
                />
              </div>

              {/* Slider: Water-Water Strength */}
              <div className="slider-group">
                <div className="slider-header">
                  <span className="slider-title">🔗 水分子間水素結合強度 (K_ww)</span>
                  <span className="slider-value cyan">{waterWaterStrength.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1.5"
                  step="0.05"
                  value={waterWaterStrength}
                  onChange={(e) => updateParameter('waterWaterStrength', parseFloat(e.target.value))}
                  className="slider-cyan"
                />
              </div>

              {/* Slider: Water-Peptide Strength */}
              <div className="slider-group">
                <div className="slider-header">
                  <span className="slider-title">🧬 水分子-アミノ酸結合強度 (K_wp)</span>
                  <span className="slider-value blue">{waterPeptideStrength.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1.5"
                  step="0.05"
                  value={waterPeptideStrength}
                  onChange={(e) => updateParameter('waterPeptideStrength', parseFloat(e.target.value))}
                  className="slider-blue"
                />
              </div>

              {/* Slider: Water Belt Contraction */}
              <div className="slider-group">
                <div className="slider-header">
                  <span className="slider-title">🪢 水ベルトの収縮率</span>
                  <span className="slider-value cyan">{waterBeltContraction.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="0.80"
                  step="0.05"
                  value={waterBeltContraction}
                  onChange={(e) => updateParameter('waterBeltContraction', parseFloat(e.target.value))}
                  className="slider-cyan"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Floating Sidebar Toggle Buttons */}
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

        {/* Right Log Panel */}
        <div className={`panel-logs ${showLogs ? '' : 'collapsed'}`}>
          <div className="terminal-header">
            <h3 style={{ display: 'flex', alignItems: 'center', gap: '6px', margin: 0, fontSize: '0.85rem' }}>
              <Terminal className="w-4 h-4 text-cyan-400" /> シミュレーション物理ログ
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
              クリア
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






