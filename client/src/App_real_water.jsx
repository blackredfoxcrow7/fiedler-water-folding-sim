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
import './App.css';

const BACKEND_HOST = window.location.hostname || 'localhost';
const BACKEND_URL = `http://${BACKEND_HOST}:5002`;

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

export default function AppRealWater() {
  const [sequence, setSequence] = useState('Ala-Ala-Ala-Ala-Ala');
  const [sessionId, setSessionId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // UI Panels Toggle State
  const [showSidebar, setShowSidebar] = useState(true);
  const [showLogs, setShowLogs] = useState(true);

  // Simulation parameter states
  const [temperature, setTemperature] = useState(0.08);
  const [isPlaying, setIsPlaying] = useState(true);
  const [explicitWaterEnabled, setExplicitWaterEnabled] = useState(true);
  const [waterWaterStrength, setWaterWaterStrength] = useState(0.50);
  const [waterPeptideStrength, setWaterPeptideStrength] = useState(0.60);
  const [bulkSolventStrength, setBulkSolventStrength] = useState(0.35);
  const [waterBeltContraction, setWaterBeltContraction] = useState(0.0);
  const [watersPerAtom, setWatersPerAtom] = useState(4);
  const [showWaterVisuals, setShowWaterVisuals] = useState(true);

  // Read-only parameters
  const [atomCount, setAtomCount] = useState(0);
  const [hydrophobicExposure, setHydrophobicExposure] = useState(0.0);
  const [activeHBondCount, setActiveHBondCount] = useState(0);
  const [gyrationRadius, setGyrationRadius] = useState(0.0);
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

  // Physics streams buffer
  const latestCoordsRef = useRef(null);
  const latestHBondsRef = useRef([]);
  const latestWaterCoordsRef = useRef([]);
  const latestWatersRef = useRef([]);
  const latestWaterBondsRef = useRef([]);
  const latestWaterPeptideBondsRef = useRef([]);

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
      
      // Store molecule data to trigger rebuildScene reactively once scene is ready
      setMoleculeData(data);
      
      setLogs([{ text: `System: Initialized peptide sequence: ${sequence}`, type: 'system' }]);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    handleInitialize();
    return () => {
      // Clean up Three.js scene on unmount
      if (rendererRef.current) {
        rendererRef.current.dispose();
      }
    };
  }, []);

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

  // Re-run from current state
  const handleReset = () => {
    handleInitialize();
  };

  // Build/Rebuild the Three.js scene objects
  const rebuildScene = (atoms, bonds, initialCoords, data) => {
    const scene = sceneRef.current;
    if (!scene) return;

    if (backboneTubeRef.current) {
      scene.remove(backboneTubeRef.current);
      backboneTubeRef.current = null;
    }

    // 1. Clear previous peptide atom meshes, bond meshes
    atomMeshesRef.current.forEach(mesh => scene.remove(mesh));
    atomMeshesRef.current = [];

    bondMeshesRef.current.forEach(mesh => scene.remove(mesh.mesh));
    bondMeshesRef.current = [];

    // 2. Rebuild peptide atom meshes
    atoms.forEach((atom, idx) => {
      const coord = initialCoords[idx];
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

    // 4. Clear and rebuild explicit H2O molecules
    waterMeshesRef.current.forEach(w => {
      scene.remove(w.oxygen);
      scene.remove(w.hydrogen1);
      scene.remove(w.hydrogen2);
      scene.remove(w.bond1);
      scene.remove(w.bond2);
    });
    waterMeshesRef.current = [];

    const initialWaterCoords = data.waterCoords || [];
    const watersMetadata = data.waters || [];
    watersMetadata.forEach((water, wIdx) => {
      const initPos = initialWaterCoords[wIdx] || [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      
      // Oxygen atom (Red Sphere)
      const oGeom = new THREE.SphereGeometry(0.20, 16, 16);
      const oMat = new THREE.MeshPhongMaterial({ 
        color: 0xef4444, 
        transparent: true, 
        opacity: 0.85,
        shininess: 90 
      });
      const oMesh = new THREE.Mesh(oGeom, oMat);
      oMesh.position.set(...initPos[0]);
      oMesh.visible = explicitWaterEnabled && showWaterVisuals;
      scene.add(oMesh);
      
      // Hydrogen atoms (Light Grey Spheres)
      const hGeom = new THREE.SphereGeometry(0.12, 12, 12);
      const hMat = new THREE.MeshPhongMaterial({ 
        color: 0xf3f4f6, 
        transparent: true, 
        opacity: 0.85 
      });
      const hMesh1 = new THREE.Mesh(hGeom, hMat);
      const hMesh2 = new THREE.Mesh(hGeom, hMat);
      hMesh1.position.set(...initPos[1]);
      hMesh2.position.set(...initPos[2]);
      hMesh1.visible = explicitWaterEnabled && showWaterVisuals;
      hMesh2.visible = explicitWaterEnabled && showWaterVisuals;
      scene.add(hMesh1);
      scene.add(hMesh2);
      
      // Covalent O-H Cylinders
      const bGeom = new THREE.CylinderGeometry(0.03, 0.03, 1, 8);
      const bMat = new THREE.MeshLambertMaterial({ 
        color: 0x9ca3af, 
        transparent: true, 
        opacity: 0.6 
      });
      const bMesh1 = new THREE.Mesh(bGeom, bMat);
      const bMesh2 = new THREE.Mesh(bGeom, bMat);
      
      updateCylinder(bMesh1, new THREE.Vector3(...initPos[0]), new THREE.Vector3(...initPos[1]));
      updateCylinder(bMesh2, new THREE.Vector3(...initPos[0]), new THREE.Vector3(...initPos[2]));
      
      bMesh1.visible = explicitWaterEnabled && showWaterVisuals;
      bMesh2.visible = explicitWaterEnabled && showWaterVisuals;
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

    // Focus camera
    const box = new THREE.Box3();
    initialCoords.forEach(c => box.expandByPoint(new THREE.Vector3(...c)));
    const center = new THREE.Vector3();
    box.getCenter(center);
    if (controlsRef.current) {
      controlsRef.current.target.copy(center);
    }
  };

  // Setup Three.js scene canvas
  useEffect(() => {
    if (!mountRef.current) return;
    
    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0f1d); // Sleek dark space blue
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
    // Clean up any existing canvas elements first to prevent duplicates
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
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.45);
    scene.add(ambientLight);
    
    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.85);
    dirLight1.position.set(10, 15, 10);
    scene.add(dirLight1);
    
    const dirLight2 = new THREE.DirectionalLight(0x3b82f6, 0.25); // Subtle blue fill light
    dirLight2.position.set(-10, -5, -10);
    scene.add(dirLight2);
    
    // Add lines group for H-bonds
    const hbLines = new THREE.Group();
    scene.add(hbLines);
    hbondLinesRef.current = hbLines;

    // Add lines group for Water-Water hydrogen bonds
    const wHBGroup = new THREE.Group();
    scene.add(wHBGroup);
    waterBondsGroupRef.current = wHBGroup;

    // Add lines group for Water-Peptide hydrogen bonds
    const wpLines = new THREE.Group();
    scene.add(wpLines);
    waterPeptideLinesRef.current = wpLines;

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
    
    // Expose for console debugging
    window.scene = scene;
    window.camera = camera;
    window.renderer = renderer;
    window.controls = controls;

    setIsSceneReady(true);
    
    // Animation Loop
    let animId;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      controls.update();
      
      // Update coordinates from streaming buffers
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
                mesh.material.opacity = 0.20; // Fade out sidechains
                mesh.scale.set(0.7, 0.7, 0.7);
              }
            } else {
              mesh.material.transparent = false;
              mesh.material.opacity = 1.0;
              
              // Visual highlight for the active folding residue
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
              item.mesh.material.opacity = 0.15; // Fade out sidechain bonds
            }
          } else {
            item.mesh.material.transparent = false;
            item.mesh.material.opacity = 1.0;
          }
        });

        // 3. Rebuild peptide H-bond dashed lines
        if (hbondLinesRef.current) {
          while (hbondLinesRef.current.children.length > 0) {
            const line = hbondLinesRef.current.children[0];
            hbondLinesRef.current.remove(line);
          }
          
          const hbonds = latestHBondsRef.current || [];
          setActiveHBondCount(hbonds.length);
          
          hbonds.forEach(bond => {
            const p1 = new THREE.Vector3(...coords[bond[0]]);
            const p2 = new THREE.Vector3(...coords[bond[1]]);
            
            const geom = new THREE.BufferGeometry().setFromPoints([p1, p2]);
            
            // Animate / Pulse hydrogen bonds when completed
            const pulse = 1.0 + 0.3 * Math.sin(Date.now() * 0.008);
            const mat = new THREE.LineDashedMaterial({
              color: isCompleted ? 0xff3b30 : 0xef4444, // Glowing bright red
              dashSize: isCompleted ? 0.15 * pulse : 0.15,
              gapSize: isCompleted ? 0.1 / pulse : 0.1,
            });
            const line = new THREE.Line(geom, mat);
            line.computeLineDistances();
            hbondLinesRef.current.add(line);
          });
        }

        // Update smooth backbone ribbon/tube
        if (isCompleted && moleculeData) {
          const backboneIndices = [];
          moleculeData.atoms.forEach((atom, idx) => {
            if (atom.name === 'N' || atom.name === 'CA' || atom.name === 'C') {
              backboneIndices.push(idx);
            }
          });
          
          const points = [];
          backboneIndices.forEach(idx => {
            if (coords[idx]) {
              points.push(new THREE.Vector3(...coords[idx]));
            }
          });
          
          if (points.length > 1) {
            const curve = new THREE.CatmullRomCurve3(points);
            const tubeGeom = new THREE.TubeGeometry(curve, 64, 0.12, 8, false);
            
            if (backboneTubeRef.current) {
              scene.remove(backboneTubeRef.current);
            }
            
            const tubeMat = new THREE.MeshPhongMaterial({
              color: 0x06b6d4, // Cyan
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

        // 4. Update H2O water spheres, hydrogens and covalent O-H bonds
        const wCoords = latestWaterCoordsRef.current || [];
        const watersData = latestWatersRef.current || [];
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
            
            // State colors
            const waterData = watersData[idx];
            if (waterData) {
              let colorHex = 0xef4444; // Default red Oxygen
              if (waterData.is_hydrophobic_neighborhood) {
                colorHex = 0xf97316; // Orange-red Oxygen in hydrophobic neighborhood
              }
              if (w.oxygen.material) {
                w.oxygen.material.color.setHex(colorHex);
              }
            }
          }
        });

        // 5. Rebuild water-water hydrogen bonds (cyan dashed lines)
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
                const posA = new THREE.Vector3(...wCoords[w1_idx][0]); // Oxygen i
                const posB = new THREE.Vector3(...wCoords[w2_idx][h_idx]); // Hydrogen h_idx of water j
                
                const geom = new THREE.BufferGeometry().setFromPoints([posA, posB]);
                const mat = new THREE.LineDashedMaterial({
                  color: 0x22d3ee, // cyan
                  dashSize: 0.12,
                  gapSize: 0.08,
                });
                const line = new THREE.Line(geom, mat);
                line.computeLineDistances();
                waterBondsGroupRef.current.add(line);
              }
            });
          }
        }

        // 6. Rebuild water-peptide hydrogen bonds (purple dashed lines)
        if (waterPeptideLinesRef.current) {
          while (waterPeptideLinesRef.current.children.length > 0) {
            waterPeptideLinesRef.current.remove(waterPeptideLinesRef.current.children[0]);
          }
          
          const activeWPB = latestWaterPeptideBondsRef.current || [];
          if (explicitWaterEnabledRef.current && showWaterVisualsRef.current && wCoords.length > 0) {
            activeWPB.forEach(pair => {
              const w_idx = pair[0];
              const p_idx = pair[1];
              const type = pair[2]; // 0: O, 1: H1, 2: H2
              
              if (wCoords[w_idx] && wCoords[w_idx][type] && coords[p_idx]) {
                const posA = new THREE.Vector3(...wCoords[w_idx][type]);
                const posB = new THREE.Vector3(...coords[p_idx]);
                
                const geom = new THREE.BufferGeometry().setFromPoints([posA, posB]);
                const mat = new THREE.LineDashedMaterial({
                  color: 0xa855f7, // purple
                  dashSize: 0.1,
                  gapSize: 0.08,
                });
                const line = new THREE.Line(geom, mat);
                line.computeLineDistances();
                waterPeptideLinesRef.current.add(line);
              }
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

      if (data.explicitWaterEnabled !== undefined) {
        setExplicitWaterEnabled(data.explicitWaterEnabled);
      }
      if (data.waterBeltContraction !== undefined) {
        setWaterBeltContraction(data.waterBeltContraction);
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
          if (logText.includes('WaterAgent') || logText.includes('Water-mediated')) {
            type = 'water-agent';
          } else if (logText.includes('bond formed')) {
            type = 'hydrophilic';
          } else if (logText.includes('folding completed')) {
            type = 'hydrophobic';
          } else if (logText.includes('Rotated joint')) {
            type = 'joint';
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

  // Scroll to bottom of log terminal
  useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  return (
    <div className="app-layout">
      {/* Header */}
      <header className="app-header">
        <div className="header-brand">
          <div className="brand-icon-wrapper" style={{ background: 'linear-gradient(135deg, #0ea5e9, #0891b2)' }}>
            <Droplet className="w-5 h-5 text-white" />
          </div>
          <div className="brand-title-group">
            <h1 className="brand-title">
              実分子 H₂O 水分子ネットワーク
              <span className="brand-badge" style={{ color: '#06b6d4', borderColor: 'rgba(6, 182, 212, 0.4)', background: 'rgba(6, 182, 212, 0.1)' }}>
                3D H₂O Model
              </span>
            </h1>
            <p className="subtitle" style={{ color: '#94a3b8', margin: '2px 0 0' }}>
              酸素(O)・水素(H)を分子構造としてモデリングした高精細液相シミュレーター
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
            <span className="metric-label">主鎖水素結合</span>
            <span className="metric-value red">
              {activeHBondCount} 本
            </span>
          </div>
          <div className="metric-divider" />
          <div className="metric-item">
            <span className="metric-label">水分子間H-bonds</span>
            <span className="metric-value cyan">
              {latestWaterBondsRef.current.length} 本
            </span>
          </div>
          <div className="metric-divider" />
          <div className="metric-item">
            <span className="metric-label">平均疎水基水近接数</span>
            <span className="metric-value orange">
              {hydrophobicExposure.toFixed(2)}
            </span>
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
          {/* Section 1: Simulations Controller */}
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

          {/* Section 2: Sequential Folding Controller */}
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


          {/* Section 3: Explicit H2O Solver Parameter Sliders */}
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

          {/* Section 4: Peptide Definition Input */}
          <div className="glass-panel" style={{ padding: '14px' }}>
            <h2 className="panel-title">
              <Dna className="w-4 h-4 text-blue-400" /> ペプチド鎖の定義
            </h2>
            <div className="form-group">
              <input
                type="text"
                value={sequence}
                onChange={(e) => setSequence(e.target.value)}
                placeholder="例: Ala-Ala-Ala / SMILES"
                disabled={loading}
              />
              <button
                onClick={handleInitialize}
                disabled={loading}
                className="primary"
                style={{ marginTop: '6px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}
              >
                {loading ? '生成中...' : '初期構造モデル生成'}
              </button>
            </div>
            {error && <div className="error-message" style={{ marginTop: '8px' }}>{error}</div>}
          </div>

          {/* Legends */}
          <div className="legend-box">
            <span className="legend-title">凡例:</span>
            <div className="legend-item">
              <span className="legend-color-dot" style={{ backgroundColor: '#ef4444' }} />
              <span>酸素 (O) / <span className="legend-color-dot" style={{ backgroundColor: '#3b82f6' }} /> 窒素 (N)</span>
            </div>
            <div className="legend-item">
              <span className="legend-color-dot" style={{ backgroundColor: '#9ca3af' }} />
              <span>炭素 (C) / <span className="legend-color-dot" style={{ backgroundColor: '#f97316' }} /> 疎水基炭素 (C-H)</span>
            </div>
            <div className="legend-item">
              <span className="legend-color-dot" style={{ backgroundColor: '#06b6d4' }} />
              <span>主鎖リボン表示 (折り畳み完了時)</span>
            </div>
            <div className="legend-item">
              <span className="legend-color-dot" style={{ backgroundColor: '#ef4444', border: '1.5px solid #ffffff' }} />
              <span>水分子 (O: 赤球 / H: 白小球)</span>
            </div>
            <div className="legend-item">
              <span className="legend-line-dashed" style={{ borderColor: '#22d3ee' }} />
              <span>水-水 水素結合 (破線: シアン)</span>
            </div>
            <div className="legend-item">
              <span className="legend-line-dashed" style={{ borderColor: '#a855f7' }} />
              <span>水-アミノ酸結合 (破線: 紫)</span>
            </div>
          </div>
        </div>

        {/* Collapsible sidebar toggle buttons */}
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
              <Terminal className="w-4 h-4 text-emerald-400" /> シミュレーション物理ログ
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
                cursor: 'pointer'
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
