import React, { useState, useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { Sparkles, Layers, Box, Eye, TrendingUp, HelpCircle, ExternalLink } from 'lucide-react';

const BACKEND_URL = 'http://localhost:5009';

// Modern amino acid metadata
const AMINO_ACID_INFO = {
  A: { name: 'Alanine', type: 'Hydrophobic', color: '#10b981', hydrophobicity: 0.18 },
  R: { name: 'Arginine', type: 'Basic (Charged)', color: '#3b82f6', hydrophobicity: 0.04 },
  N: { name: 'Asparagine', type: 'Polar', color: '#a855f7', hydrophobicity: 0.07 },
  D: { name: 'Aspartic Acid', type: 'Acidic (Charged)', color: '#ef4444', hydrophobicity: 0.02 },
  C: { name: 'Cysteine', type: 'Sulfur-containing', color: '#fbbf24', hydrophobicity: 0.20 },
  Q: { name: 'Glutamine', type: 'Polar', color: '#a855f7', hydrophobicity: 0.08 },
  E: { name: 'Glutamic Acid', type: 'Acidic (Charged)', color: '#ef4444', hydrophobicity: 0.04 },
  G: { name: 'Glycine', type: 'Hydrophobic', color: '#64748b', hydrophobicity: 0.16 },
  H: { name: 'Histidine', type: 'Basic/Aromatic', color: '#3b82f6', hydrophobicity: 0.10 },
  I: { name: 'Isoleucine', type: 'Hydrophobic', color: '#10b981', hydrophobicity: 0.31 },
  L: { name: 'Leucine', type: 'Hydrophobic', color: '#10b981', hydrophobicity: 0.28 },
  K: { name: 'Lysine', type: 'Basic (Charged)', color: '#3b82f6', hydrophobicity: 0.05 },
  M: { name: 'Methionine', type: 'Sulfur-containing', color: '#fbbf24', hydrophobicity: 0.22 },
  F: { name: 'Phenylalanine', type: 'Aromatic', color: '#ec4899', hydrophobicity: 0.29 },
  P: { name: 'Proline', type: 'Hydrophobic', color: '#f59e0b', hydrophobicity: 0.19 },
  S: { name: 'Serine', type: 'Polar', color: '#a855f7', hydrophobicity: 0.08 },
  T: { name: 'Threonine', type: 'Polar', color: '#10b981', hydrophobicity: 0.11 },
  W: { name: 'Tryptophan', type: 'Hydrophobic/Aromatic', color: '#ec4899', hydrophobicity: 0.24 },
  Y: { name: 'Tyrosine', type: 'Polar/Aromatic', color: '#a855f7', hydrophobicity: 0.21 },
  V: { name: 'Valine', type: 'Hydrophobic', color: '#10b981', hydrophobicity: 0.26 }
};

export default function AppOverallFiedler() {
  const [sequence, setSequence] = useState('GYDPETGTWG');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [moleculeData, setMoleculeData] = useState(null);
  const [simulationHistory, setSimulationHistory] = useState([]);
  const [activeResidue, setActiveResidue] = useState(null);
  const [debugLogs, setDebugLogs] = useState([]);
  
  // Style configurations
  const [representation, setRepresentation] = useState('ball-stick'); // 'ball-stick' | 'vdw' | 'ribbon'
  const [colorTheme, setColorTheme] = useState('element'); // 'element' | 'hydrophobicity'
  const [syncCameras, setSyncCameras] = useState(true);
  const [chartMode, setChartMode] = useState('fiedler'); // 'fiedler' | 'entropy'
  const [scenesReady, setScenesReady] = useState(false);
  const [showWater, setShowWater] = useState(true);

  // Playback state
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  // Viewport DOM refs
  const mountFoldedRef = useRef(null); // Displays simulated trajectory frame
  const mountUnfoldedRef = useRef(null); // Displays initial state conformation

  // Three.js instances for Folded Viewport
  const sceneFolded = useRef(null);
  const rendererFolded = useRef(null);
  const cameraFolded = useRef(null);
  const controlsFolded = useRef(null);
  const meshesFolded = useRef({ atoms: [], bonds: [], ribbon: null, labels: [], waters: [], hbLines: [] });

  // Three.js instances for Unfolded Viewport
  const sceneUnfolded = useRef(null);
  const rendererUnfolded = useRef(null);
  const cameraUnfolded = useRef(null);
  const controlsUnfolded = useRef(null);
  const meshesUnfolded = useRef({ atoms: [], bonds: [], ribbon: null, labels: [], waters: [], hbLines: [] });

  // Flags
  const isSyncingRef = useRef(false);
  const chartWindowRef = useRef(null);

  // SSE data sync to popout window
  useEffect(() => {
    if (moleculeData && chartWindowRef.current && !chartWindowRef.current.closed) {
      if (chartWindowRef.current.updateData) {
        chartWindowRef.current.updateData(moleculeData.trajectory);
      }
    }
  }, [moleculeData]);

  // Handle Playback Interval Loop
  useEffect(() => {
    let timer;
    if (isPlaying && moleculeData && moleculeData.trajectory) {
      timer = setInterval(() => {
        setCurrentFrame(prev => {
          if (prev >= moleculeData.trajectory.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 150);
    }
    return () => clearInterval(timer);
  }, [isPlaying, moleculeData]);

  useEffect(() => {
    const handleWindowError = (event) => {
      setDebugLogs(prev => [...prev, `ERROR: ${event.message} at ${event.filename}:${event.lineno}`].slice(-30));
    };
    window.addEventListener('error', handleWindowError);

    const originalConsoleError = console.error;
    const originalConsoleWarn = console.warn;
    const originalConsoleLog = console.log;

    const sendLogToServer = (level, args) => {
      const message = args.map(a => {
        if (a instanceof Error) return a.message + "\n" + a.stack;
        return typeof a === 'object' ? JSON.stringify(a) : String(a);
      }).join(' ');
      fetch(`${BACKEND_URL}/api/log`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ level, message })
      }).catch(() => {});
    };

    console.error = (...args) => {
      setDebugLogs(prev => [...prev, `ERR: ${args.map(a => typeof a === 'object' ? JSON.stringify(a) : a).join(' ')}`].slice(-30));
      sendLogToServer('ERROR', args);
      originalConsoleError.apply(console, args);
    };
    console.warn = (...args) => {
      setDebugLogs(prev => [...prev, `WARN: ${args.map(a => typeof a === 'object' ? JSON.stringify(a) : a).join(' ')}`].slice(-30));
      sendLogToServer('WARN', args);
      originalConsoleWarn.apply(console, args);
    };
    console.log = (...args) => {
      setDebugLogs(prev => [...prev, `LOG: ${args.map(a => typeof a === 'object' ? JSON.stringify(a) : a).join(' ')}`].slice(-30));
      sendLogToServer('INFO', args);
      originalConsoleLog.apply(console, args);
    };

    return () => {
      window.removeEventListener('error', handleWindowError);
      console.error = originalConsoleError;
      console.warn = originalConsoleWarn;
      console.log = originalConsoleLog;
    };
  }, []);

  // Fetch simulation from backend
  const fetchStructure = async (seqToLoad) => {
    setLoading(true);
    setError(null);
    setIsPlaying(false);
    setCurrentFrame(0);
    try {
      const response = await fetch(`${BACKEND_URL}/api/structure`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequence: seqToLoad })
      });
      if (!response.ok) {
        throw new Error('Failed to run peptide folding simulation.');
      }
      const data = await response.json();
      if (data.status === 'error') {
        throw new Error(data.error);
      }
      
      console.log("\n========================================================");
      console.log(`  PEPTIDE SOLVENT-COUPLED OVERALL FOLDING (Sequence: ${data.sequence})`);
      console.log(`--------------------------------------------------------`);
      if (data.trajectory && data.trajectory.length > 0) {
        const init = data.trajectory[0];
        const last = data.trajectory[data.trajectory.length - 1];
        console.log(`  UNFOLDED STATE (Step 0):`);
        console.log(`    Overall Fiedler: ${init.overallFiedler.toFixed(6)} | Overall Entropy: ${init.overallEntropy.toFixed(4)}`);
        console.log(`  SIMULATED FOLDED STATE (Step ${last.step}):`);
        console.log(`    Overall Fiedler: ${last.overallFiedler.toFixed(6)} | Overall Entropy: ${last.overallEntropy.toFixed(4)}`);
        console.log(`    Peptide Ref:     ${last.peptideFiedler.toFixed(6)} | Water Ref:       ${last.waterFiedler.toFixed(6)}`);
      }
      console.log("========================================================\n");

      let finalFiedler = 0;
      let finalEntropy = 0;
      if (data.trajectory && data.trajectory.length > 0) {
        const last = data.trajectory[data.trajectory.length - 1];
        finalFiedler = last.overallFiedler;
        finalEntropy = last.overallEntropy;
      }
      
      const runId = `run-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      const newRun = { 
        ...data, 
        runId, 
        finalFiedler, 
        finalEntropy 
      };

      setSimulationHistory(prev => [newRun, ...prev]);
      setMoleculeData(newRun);
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

  // Cylinder alignment helper
  const updateCylinder = (mesh, vStart, vEnd) => {
    const distance = vStart.distanceTo(vEnd);
    mesh.scale.set(1, distance, 1);
    
    const position = new THREE.Vector3().addVectors(vStart, vEnd).multiplyScalar(0.5);
    mesh.position.copy(position);
    
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
      if (atom.symbol === 'N') return new THREE.Color('#3b82f6');
      if (atom.symbol === 'O') return new THREE.Color('#ef4444');
      if (atom.symbol === 'S') return new THREE.Color('#eab308');
      return new THREE.Color('#475569');
    }
  };

  const getAtomRadius = (atom, rep) => {
    if (rep === 'vdw') {
      if (atom.symbol === 'N') return 1.55 * 0.45;
      if (atom.symbol === 'O') return 1.52 * 0.45;
      if (atom.symbol === 'S') return 1.80 * 0.45;
      return 1.70 * 0.45;
    } else {
      if (atom.symbol === 'N') return 0.28;
      if (atom.symbol === 'O') return 0.28;
      if (atom.symbol === 'S') return 0.35;
      return 0.25;
    }
  };

  // Rebuild Three.js Scene
  const rebuildViewport = (scene, camera, renderer, mountNode, meshesRef, coords, label) => {
    if (!scene || !camera || !renderer || !mountNode || !moleculeData || !coords) {
      console.warn(`[rebuildViewport early exit] Label: ${label}, scene: ${!!scene}, camera: ${!!camera}, renderer: ${!!renderer}, mountNode: ${!!mountNode}, moleculeData: ${!!moleculeData}, coords: ${!!coords}`);
      return;
    }

    try {
      console.log(`[rebuildViewport run] Label: ${label}, Coords: ${coords.length}`);
      const meshes = meshesRef.current;
      
      // Clear old peptide meshes
      meshes.atoms.forEach(a => scene.remove(a));
      meshes.bonds.forEach(b => scene.remove(b));
      meshes.labels.forEach(l => scene.remove(l));
      if (meshes.ribbon) scene.remove(meshes.ribbon);

      meshes.atoms = [];
      meshes.bonds = [];
      meshes.labels = [];
      meshes.ribbon = null;

      // Clear old water meshes
      meshes.waters = meshes.waters || [];
      meshes.waters.forEach(w => {
        scene.remove(w.oxygen);
        scene.remove(w.hydrogen1);
        scene.remove(w.hydrogen2);
        scene.remove(w.bond1);
        scene.remove(w.bond2);
      });
      meshes.waters = [];

      // Clear old H-bonds dashed lines
      meshes.hbLines = meshes.hbLines || [];
      meshes.hbLines.forEach(l => scene.remove(l));
      meshes.hbLines = [];

      // 1. Draw Peptide Atoms
      if (representation === 'ball-stick' || representation === 'vdw') {
        moleculeData.atoms.forEach(atom => {
          const coord = coords[atom.id];
          if (!coord) return;

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
          mesh.position.set(coord[0], coord[1], coord[2]);
          scene.add(mesh);
          meshes.atoms.push(mesh);
        });
      }

      // 2. Draw Peptide Bonds
      if (representation === 'ball-stick') {
        moleculeData.bonds.forEach(bond => {
          const pStart = coords[bond.source];
          const pEnd = coords[bond.target];
          if (!pStart || !pEnd) return;

          const vStart = new THREE.Vector3(pStart[0], pStart[1], pStart[2]);
          const vEnd = new THREE.Vector3(pEnd[0], pEnd[1], pEnd[2]);

          const resA = moleculeData.atomResidues[bond.source];
          const resB = moleculeData.atomResidues[bond.target];
          const isHighlighted = activeResidue === null || (resA === activeResidue && resB === activeResidue);
          const opacity = isHighlighted ? 0.9 : 0.35;

          const geo = new THREE.CylinderGeometry(0.08, 0.08, 1, 12);
          const mat = new THREE.MeshStandardMaterial({
            color: 0x94a3b8,
            roughness: 0.5,
            metalness: 0.1,
            transparent: true,
            opacity: opacity
          });
          const mesh = new THREE.Mesh(geo, mat);
          updateCylinder(mesh, vStart, vEnd);
          scene.add(mesh);
          meshes.bonds.push(mesh);
        });
      }

      // 3. Draw Ribbon Tube
      if (representation === 'ribbon' || representation === 'ball-stick') {
        const caCoords = [];
        moleculeData.atoms.forEach((atom) => {
          if (atom.symbol === 'CA' || (atom.element === 'C' && atom.symbol === 'CA')) {
            const coord = coords[atom.id];
            if (coord) {
              caCoords.push(new THREE.Vector3(coord[0], coord[1], coord[2]));
            }
          }
        });

        if (caCoords.length > 1) {
          const curve = new THREE.CatmullRomCurve3(caCoords);
          const tubeGeo = new THREE.TubeGeometry(curve, 64, 0.25, 12, false);
          
          const mat = new THREE.MeshStandardMaterial({
            color: representation === 'ribbon' ? 0xa855f7 : 0xd8b4fe,
            roughness: 0.1,
            metalness: 0.2,
            transparent: true,
            opacity: activeResidue === null ? 0.95 : 0.6,
            emissive: representation === 'ribbon' ? 0x6b21a8 : 0x000000,
            emissiveIntensity: 0.3
          });
          
          const mesh = new THREE.Mesh(tubeGeo, mat);
          scene.add(mesh);
          meshes.ribbon = mesh;
        }
      }

      // 4. Draw Solvent Water Molecules (Realistic H-O-H geometry)
      const waterCoords = label === "Folded"
        ? (moleculeData.trajectory[currentFrame] ? moleculeData.trajectory[currentFrame].waterCoords : moleculeData.foldedWaters)
        : moleculeData.unfoldedWaters;

      if (showWater && waterCoords) {
        const oGeom = new THREE.SphereGeometry(0.20, 16, 16);
        const oMat = new THREE.MeshPhongMaterial({ color: 0xef4444, shininess: 90 });
        const hGeom = new THREE.SphereGeometry(0.12, 12, 12);
        const hMat = new THREE.MeshPhongMaterial({ color: 0xf3f4f6 });
        const bGeom = new THREE.CylinderGeometry(0.03, 0.03, 1, 8);
        const bMat = new THREE.MeshLambertMaterial({ color: 0x9ca3af, transparent: true, opacity: 0.6 });

        waterCoords.forEach(coord => {
          const oMesh = new THREE.Mesh(oGeom, oMat);
          oMesh.position.set(...coord[0]);
          scene.add(oMesh);

          const hMesh1 = new THREE.Mesh(hGeom, hMat);
          hMesh1.position.set(...coord[1]);
          scene.add(hMesh1);

          const hMesh2 = new THREE.Mesh(hGeom, hMat);
          hMesh2.position.set(...coord[2]);
          scene.add(hMesh2);

          const bMesh1 = new THREE.Mesh(bGeom, bMat);
          const bMesh2 = new THREE.Mesh(bGeom, bMat);
          updateCylinder(bMesh1, new THREE.Vector3(...coord[0]), new THREE.Vector3(...coord[1]));
          updateCylinder(bMesh2, new THREE.Vector3(...coord[0]), new THREE.Vector3(...coord[2]));
          scene.add(bMesh1);
          scene.add(bMesh2);

          meshes.waters.push({
            oxygen: oMesh,
            hydrogen1: hMesh1,
            hydrogen2: hMesh2,
            bond1: bMesh1,
            bond2: bMesh2
          });
        });
      }

      // 5. Draw Dynamic H-bonds and Hydrophobic Contacts
      const isFolded = label === "Folded";
      const activeFrame = isFolded 
        ? moleculeData.trajectory[currentFrame] 
        : moleculeData.trajectory[0];

      if (activeFrame) {
        // A1. Backbone-Backbone bonds (Red dashed lines)
        let hbonds = [];
        if (isFolded) {
          hbonds = activeFrame.activeHBonds || [];
        } else {
          hbonds = moleculeData.hasFoldedTarget 
            ? (moleculeData.targetHBonds || []) 
            : (activeFrame.activeHBonds || []);
        }
        hbonds.forEach(bond => {
          if (coords[bond[0]] && coords[bond[1]]) {
            const p1 = new THREE.Vector3(...coords[bond[0]]);
            const p2 = new THREE.Vector3(...coords[bond[1]]);
            const geom = new THREE.BufferGeometry().setFromPoints([p1, p2]);
            const mat = new THREE.LineDashedMaterial({ color: 0xef4444, dashSize: 0.15, gapSize: 0.1 });
            const line = new THREE.Line(geom, mat);
            line.computeLineDistances();
            scene.add(line);
            meshes.hbLines.push(line);
          }
        });

        // A2. Sidechain-Sidechain bonds (Orange dashed lines)
        let scBonds = [];
        if (isFolded) {
          scBonds = activeFrame.activeSidechainBonds || [];
        } else {
          scBonds = moleculeData.hasFoldedTarget 
            ? (moleculeData.targetSidechainBonds || []) 
            : (activeFrame.activeSidechainBonds || []);
        }
        scBonds.forEach(bond => {
          if (coords[bond[0]] && coords[bond[1]]) {
            const p1 = new THREE.Vector3(...coords[bond[0]]);
            const p2 = new THREE.Vector3(...coords[bond[1]]);
            const geom = new THREE.BufferGeometry().setFromPoints([p1, p2]);
            const mat = new THREE.LineDashedMaterial({ color: 0xf97316, dashSize: 0.15, gapSize: 0.1 }); // Orange
            const line = new THREE.Line(geom, mat);
            line.computeLineDistances();
            scene.add(line);
            meshes.hbLines.push(line);
          }
        });

        // A3. Backbone-Sidechain bonds (Yellow dashed lines)
        let bbScBonds = [];
        if (isFolded) {
          bbScBonds = activeFrame.activeBBScBonds || [];
        } else {
          bbScBonds = moleculeData.hasFoldedTarget 
            ? (moleculeData.targetBBScBonds || []) 
            : (activeFrame.activeBBScBonds || []);
        }
        bbScBonds.forEach(bond => {
          if (coords[bond[0]] && coords[bond[1]]) {
            const p1 = new THREE.Vector3(...coords[bond[0]]);
            const p2 = new THREE.Vector3(...coords[bond[1]]);
            const geom = new THREE.BufferGeometry().setFromPoints([p1, p2]);
            const mat = new THREE.LineDashedMaterial({ color: 0xeab308, dashSize: 0.15, gapSize: 0.1 }); // Yellow
            const line = new THREE.Line(geom, mat);
            line.computeLineDistances();
            scene.add(line);
            meshes.hbLines.push(line);
          }
        });

        // B. Water-water hydrogen bonds (cyan dashed lines)
        if (showWater) {
          let wBonds = [];
          if (isFolded) {
            wBonds = activeFrame.activeWaterBonds || [];
          } else {
            wBonds = moleculeData.hasFoldedTarget
              ? (moleculeData.targetWaterBonds || [])
              : (activeFrame.activeWaterBonds || []);
          }
          
          const wCoords = isFolded ? activeFrame.waterCoords : moleculeData.unfoldedWaters;
          
          wBonds.forEach(bond => {
            const w1 = bond[0];
            const w2 = bond[1];
            const h_idx = bond[2];
            if (wCoords[w1] && wCoords[w2] && wCoords[w2][h_idx]) {
              const p1 = new THREE.Vector3(...wCoords[w1][0]);
              const p2 = new THREE.Vector3(...wCoords[w2][h_idx]);
              const geom = new THREE.BufferGeometry().setFromPoints([p1, p2]);
              const mat = new THREE.LineDashedMaterial({ color: 0x22d3ee, dashSize: 0.12, gapSize: 0.08 });
              const line = new THREE.Line(geom, mat);
              line.computeLineDistances();
              scene.add(line);
              meshes.hbLines.push(line);
            }
          });
        }

        // C. Water-peptide hydrogen bonds (purple dashed lines)
        if (showWater) {
          let wpBonds = [];
          if (isFolded) {
            wpBonds = activeFrame.activeWaterPeptideBonds || [];
          } else {
            wpBonds = moleculeData.hasFoldedTarget
              ? (moleculeData.targetWaterPeptideBonds || [])
              : (activeFrame.activeWaterPeptideBonds || []);
          }
          
          const wCoords = isFolded ? activeFrame.waterCoords : moleculeData.unfoldedWaters;
          
          wpBonds.forEach(bond => {
            const w = bond[0];
            const p = bond[1];
            const t = bond[2];
            if (wCoords[w] && wCoords[w][t] && coords[p]) {
              const p1 = new THREE.Vector3(...wCoords[w][t]);
              const p2 = new THREE.Vector3(...coords[p]);
              const geom = new THREE.BufferGeometry().setFromPoints([p1, p2]);
              const mat = new THREE.LineDashedMaterial({ color: 0xa855f7, dashSize: 0.1, gapSize: 0.08 });
              const line = new THREE.Line(geom, mat);
              line.computeLineDistances();
              scene.add(line);
              meshes.hbLines.push(line);
            }
          });
        }
      }
    } catch (err) {
      console.error(`Error in rebuildViewport [${label}]:`, err);
    }
  };

  // Re-run viewports updates on frame changes
  useEffect(() => {
    if (!moleculeData || !scenesReady) return;
    
    // Unfolded Viewport
    rebuildViewport(sceneUnfolded.current, cameraUnfolded.current, rendererUnfolded.current, mountUnfoldedRef.current, meshesUnfolded, moleculeData.unfoldedCoords, "Unfolded");
    
    // Folded Viewport (representing current playback frame)
    const activeFrame = moleculeData.trajectory && moleculeData.trajectory[currentFrame];
    if (activeFrame) {
      rebuildViewport(sceneFolded.current, cameraFolded.current, rendererFolded.current, mountFoldedRef.current, meshesFolded, activeFrame.coords, "Folded");
    }
  }, [moleculeData, scenesReady, representation, colorTheme, activeResidue, sequence, currentFrame, showWater]);

  // Initialize viewports
  const initEngine = (mountNode) => {
    if (!mountNode) return null;

    const width = mountNode.clientWidth || 300;
    const height = mountNode.clientHeight || 300;
    const aspect = width / height;

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x0b0f19, 0.015);

    const camera = new THREE.PerspectiveCamera(40, aspect, 0.1, 1000);
    camera.position.set(0, 10, 25);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    while (mountNode.firstChild) {
      mountNode.removeChild(mountNode.firstChild);
    }
    mountNode.appendChild(renderer.domElement);

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight1.position.set(10, 15, 10);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xa78bfa, 0.4);
    dirLight2.position.set(-10, -5, -10);
    scene.add(dirLight2);

    const pointLight = new THREE.PointLight(0xffffff, 0.5, 30);
    pointLight.position.set(0, 0, 5);
    scene.add(pointLight);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 45;
    controls.minDistance = 5;

    const gridHelper = new THREE.GridHelper(30, 30, 0x1e293b, 0x0f172a);
    gridHelper.position.y = -8;
    scene.add(gridHelper);

    return { scene, camera, renderer, controls };
  };

  useEffect(() => {
    const foldedEngine = initEngine(mountFoldedRef.current);
    if (foldedEngine) {
      sceneFolded.current = foldedEngine.scene;
      cameraFolded.current = foldedEngine.camera;
      rendererFolded.current = foldedEngine.renderer;
      controlsFolded.current = foldedEngine.controls;
    }

    const unfoldedEngine = initEngine(mountUnfoldedRef.current);
    if (unfoldedEngine) {
      sceneUnfolded.current = unfoldedEngine.scene;
      cameraUnfolded.current = unfoldedEngine.camera;
      rendererUnfolded.current = unfoldedEngine.renderer;
      controlsUnfolded.current = unfoldedEngine.controls;
    }

    setScenesReady(true);

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

    return () => {
      cancelAnimationFrame(animationId);
      if (rendererFolded.current && mountFoldedRef.current) {
        mountFoldedRef.current.removeChild(rendererFolded.current.domElement);
      }
      if (rendererUnfolded.current && mountUnfoldedRef.current) {
        mountUnfoldedRef.current.removeChild(rendererUnfolded.current.domElement);
      }
      if (chartWindowRef.current && !chartWindowRef.current.closed) {
        chartWindowRef.current.close();
      }
    };
  }, []);

  // Synchronize cameras
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

  const handleSelectHistory = (run) => {
    setMoleculeData(run);
    setSequence(run.sequence);
    setCurrentFrame(0);
    setIsPlaying(false);
  };

  const openChartInNewWindow = () => {
    if (chartWindowRef.current && !chartWindowRef.current.closed) {
      chartWindowRef.current.focus();
      return;
    }
    const width = 600;
    const height = 450;
    const left = window.screenX + (window.outerWidth - width) / 2;
    const top = window.screenY + (window.outerHeight - height) / 2;
    const win = window.open('', 'FoldingSimulationChart', `width=${width},height=${height},left=${left},top=${top},menubar=no,toolbar=no,location=no,status=no,resizable=yes`);
    if (!win) {
      alert("Popup blocked! Please allow popups for this site.");
      return;
    }
    chartWindowRef.current = win;
    win.document.write(`
      <!DOCTYPE html>
      <html>
        <head>
          <title>Simulation History Chart</title>
          <style>
            body {
              background: #0b0f19;
              color: #f8fafc;
              font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
              margin: 0;
              padding: 20px;
              display: flex;
              flex-direction: column;
              height: 100vh;
              box-sizing: border-box;
              overflow: hidden;
            }
            .header {
              display: flex;
              justify-content: space-between;
              align-items: center;
              margin-bottom: 15px;
            }
            h2 {
              font-size: 1.1rem;
              margin: 0;
              color: #a78bfa;
              text-transform: uppercase;
              letter-spacing: 0.05em;
            }
            .tabs {
              display: flex;
              background: rgba(255,255,255,0.05);
              border: 1px solid rgba(255,255,255,0.1);
              border-radius: 6px;
              padding: 2px;
            }
            .tabs button {
              background: transparent;
              border: none;
              color: #94a3b8;
              font-size: 0.75rem;
              font-weight: 600;
              padding: 4px 10px;
              cursor: pointer;
              border-radius: 4px;
            }
            .tabs button.active {
              background: rgba(255,255,255,0.1);
              color: #fff;
            }
            .chart-wrapper {
              flex: 1;
              position: relative;
              background: rgba(15, 23, 42, 0.45);
              border: 1px solid rgba(255,255,255,0.08);
              border-radius: 8px;
              padding: 10px;
              display: flex;
              align-items: center;
              justify-content: center;
            }
            svg {
              width: 100%;
              height: 100%;
            }
            .legend {
              display: flex;
              justify-content: center;
              gap: 20px;
              margin-top: 12px;
            }
            .legend-item {
              display: flex;
              align-items: center;
              gap: 6px;
              font-size: 0.75rem;
              color: #cbd5e1;
            }
            .legend-dot {
              width: 8px;
              height: 8px;
              border-radius: 50%;
            }
            .legend-dot.sys { background: #c084fc; }
            .legend-dot.pep { background: #3b82f6; }
            .legend-dot.wat { background: #10b981; }
          </style>
        </head>
        <body>
          <div class="header">
            <h2 id="chart-title">Overall Simulation History</h2>
            <div class="tabs">
              <button id="btn-fiedler" class="active" onclick="setMode('fiedler')">Fiedler (λ₂)</button>
              <button id="btn-entropy" onclick="setMode('entropy')">Entropy (H)</button>
            </div>
          </div>
          <div class="chart-wrapper">
            <div id="svg-container" style="width:100%; height:100%;"></div>
          </div>
          <div class="legend">
            <span class="legend-item"><span class="legend-dot sys"></span>Combined System</span>
            <span class="legend-item"><span class="legend-dot pep"></span>Peptide Ref</span>
            <span class="legend-item"><span class="legend-dot wat"></span>Water Ref</span>
          </div>
          <script>
            let currentMode = 'fiedler';
            let trajData = [];
            window.updateData = function(traj) {
              trajData = traj;
              renderChart();
            };
            window.setMode = function(mode) {
              currentMode = mode;
              document.getElementById('btn-fiedler').className = mode === 'fiedler' ? 'active' : '';
              document.getElementById('btn-entropy').className = mode === 'entropy' ? 'active' : '';
              renderChart();
            };
            function renderChart() {
              const container = document.getElementById('svg-container');
              if (!trajData || trajData.length === 0) {
                container.innerHTML = '<div style="color:#64748b; font-size:0.8rem; text-align:center; padding-top:40px;">No trajectory data available.</div>';
                return;
              }
              const width = container.clientWidth || 540;
              const height = container.clientHeight || 280;
              const padding = 30;
              
              let pointsSys = [];
              let pointsPep = [];
              let pointsWat = [];
              
              let minVal = 0.0;
              let maxVal = 1.0;
              
              if (currentMode === 'fiedler') {
                const allVals = trajData.map(d => d.overallFiedler).concat(trajData.map(d => d.peptideFiedler)).concat(trajData.map(d => d.waterFiedler));
                minVal = Math.min(...allVals);
                maxVal = Math.max(...allVals, 0.2);
              } else {
                const allVals = trajData.map(d => d.overallEntropy).concat(trajData.map(d => d.peptideEntropy)).concat(trajData.map(d => d.waterEntropy));
                minVal = Math.min(...allVals);
                maxVal = Math.max(...allVals);
              }
              const valRange = (maxVal - minVal) || 1.0;
              const N = trajData.length;
              
              trajData.forEach((d, i) => {
                const x = padding + (i / (N - 1)) * (width - 2 * padding);
                const valSys = currentMode === 'fiedler' ? d.overallFiedler : d.overallEntropy;
                const valPep = currentMode === 'fiedler' ? d.peptideFiedler : d.peptideEntropy;
                const valWat = currentMode === 'fiedler' ? d.waterFiedler : d.waterEntropy;
                
                const ySys = padding + (1.0 - (valSys - minVal) / valRange) * (height - 2 * padding);
                const yPep = padding + (1.0 - (valPep - minVal) / valRange) * (height - 2 * padding);
                const yWat = padding + (1.0 - (valWat - minVal) / valRange) * (height - 2 * padding);
                
                pointsSys.push(x + ',' + ySys);
                pointsPep.push(x + ',' + yPep);
                pointsWat.push(x + ',' + yWat);
              });
              
              let svgHtml = '<svg viewBox="0 0 ' + width + ' ' + height + '">';
              const gridSteps = 4;
              for (let i = 0; i <= gridSteps; i++) {
                const ratio = i / gridSteps;
                const y = padding + ratio * (height - 2 * padding);
                const val = maxVal - ratio * valRange;
                svgHtml += '<line x1="' + padding + '" y1="' + y + '" x2="' + (width - padding) + '" y2="' + y + '" stroke="rgba(255,255,255,0.05)" stroke-width="1" />';
                svgHtml += '<text x="' + (padding - 8) + '" y="' + (y + 4) + '" fill="#64748b" font-size="9" text-anchor="end">' + val.toFixed(2) + '</text>';
              }
              
              const xSteps = Math.min(N, 5);
              for (let i = 0; i < xSteps; i++) {
                const idx = Math.floor((i / (xSteps - 1)) * (N - 1));
                const x = padding + (idx / (N - 1)) * (width - 2 * padding);
                svgHtml += '<text x="' + x + '" y="' + (height - 8) + '" fill="#64748b" font-size="9" text-anchor="middle">Step ' + idx + '</text>';
              }
              
              if (N > 1) {
                svgHtml += '<polyline fill="none" stroke="#c084fc" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="' + pointsSys.join(' ') + '" />';
                svgHtml += '<polyline fill="none" stroke="#3b82f6" stroke-width="1.5" stroke-dasharray="4,3" stroke-linecap="round" stroke-linejoin="round" points="' + pointsPep.join(' ') + '" />';
                svgHtml += '<polyline fill="none" stroke="#10b981" stroke-width="1.5" stroke-dasharray="2,2" stroke-linecap="round" stroke-linejoin="round" points="' + pointsWat.join(' ') + '" />';
              }
              svgHtml += '</svg>';
              container.innerHTML = svgHtml;
            }
            window.addEventListener('resize', renderChart);
          </script>
        </body>
      </html>
    `);
    win.document.close();
    if (moleculeData && moleculeData.trajectory) {
      setTimeout(() => {
        if (win.updateData) {
          win.updateData(moleculeData.trajectory);
        }
      }, 100);
    }
  };

  const renderSVGChart = () => {
    if (!moleculeData || !moleculeData.trajectory || moleculeData.trajectory.length === 0) return null;
    const traj = moleculeData.trajectory;
    const N = traj.length;

    const width = 280;
    const height = 150;
    const padding = 15;

    let pointsSys = [];
    let pointsPep = [];
    let pointsWat = [];

    let minVal = 0.0;
    let maxVal = 1.0;

    if (chartMode === 'fiedler') {
      const allVals = traj.map(d => d.overallFiedler).concat(traj.map(d => d.peptideFiedler)).concat(traj.map(d => d.waterFiedler));
      minVal = Math.min(...allVals);
      maxVal = Math.max(...allVals, 0.2);
    } else {
      const allVals = traj.map(d => d.overallEntropy).concat(traj.map(d => d.peptideEntropy)).concat(traj.map(d => d.waterEntropy));
      minVal = Math.min(...allVals);
      maxVal = Math.max(...allVals);
    }

    const valRange = (maxVal - minVal) || 1.0;

    traj.forEach((d, i) => {
      const x = padding + (i / (N - 1)) * (width - 2 * padding);
      const valSys = chartMode === 'fiedler' ? d.overallFiedler : d.overallEntropy;
      const valPep = chartMode === 'fiedler' ? d.peptideFiedler : d.peptideEntropy;
      const valWat = chartMode === 'fiedler' ? d.waterFiedler : d.waterEntropy;

      const ySys = padding + (1.0 - (valSys - minVal) / valRange) * (height - 2 * padding);
      const yPep = padding + (1.0 - (valPep - minVal) / valRange) * (height - 2 * padding);
      const yWat = padding + (1.0 - (valWat - minVal) / valRange) * (height - 2 * padding);

      pointsSys.push(`${x},${ySys}`);
      pointsPep.push(`${x},${yPep}`);
      pointsWat.push(`${x},${yWat}`);
    });

    return (
      <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} style={{ background: 'rgba(0,0,0,0.2)', borderRadius: '6px' }}>
        <line x1={padding} y1={padding} x2={width - padding} y2={padding} stroke="rgba(255,255,255,0.05)" />
        <line x1={padding} y1={height/2} x2={width - padding} y2={height/2} stroke="rgba(255,255,255,0.05)" />
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="rgba(255,255,255,0.05)" />
        
        <polyline fill="none" stroke="#c084fc" strokeWidth="2.5" points={pointsSys.join(' ')} />
        <polyline fill="none" stroke="#3b82f6" strokeWidth="1.2" strokeDasharray="3,2" points={pointsPep.join(' ')} />
        <polyline fill="none" stroke="#10b981" strokeWidth="1.2" strokeDasharray="1,1" points={pointsWat.join(' ')} />
        
        <text x={padding + 5} y={padding + 12} fill="#94a3b8" fontSize="8">Max: {maxVal.toFixed(3)}</text>
        <text x={padding + 5} y={height - padding - 4} fill="#94a3b8" fontSize="8">Min: {minVal.toFixed(3)}</text>
        <text x={width - padding - 40} y={height - padding - 4} fill="#64748b" fontSize="8">Steps →</text>
      </svg>
    );
  };

  const getFrameMetrics = () => {
    if (!moleculeData || !moleculeData.trajectory || moleculeData.trajectory.length === 0) return {};
    return moleculeData.trajectory[currentFrame] || {};
  };

  const frameMetrics = getFrameMetrics();

  return (
    <div className="render-app-container">
      {/* 1. Header Bar */}
      <header className="app-header">
        <div className="header-brand">
          <Sparkles className="brand-icon" />
          <div className="brand-text">
            <h1>Peptide Conformation Analyzer</h1>
            <p>Unified Solute-Solvent Overall Graph Fiedler Maximization</p>
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
            Run Folding Simulation
          </button>
        </form>
      </header>

      {/* 2. Main Sidebar & Viewports Workspace */}
      <main className="workspace-main">
        {loading && (
          <div className="loading-overlay">
            <div className="loading-card">
              <div className="spinner"></div>
              <h3>Running Unified Fiedler Folding Simulation</h3>
              <p>Attaching 2 H2O molecules per atom, running solvent relaxation, and optimizing overall Fiedler connectivity...</p>
            </div>
          </div>
        )}

        {error && (
          <div className="error-overlay">
            <div className="error-card">
              <h3>Simulation Error</h3>
              <p>{error}</p>
              <button onClick={() => fetchStructure(sequence)} className="retry-btn">
                Retry Simulation
              </button>
            </div>
          </div>
        )}

        {/* 3. Control Center Sidebar */}
        <section className="control-sidebar card-glass">
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

          <div className="panel-section flex-row-item" style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={syncCameras}
                onChange={(e) => setSyncCameras(e.target.checked)}
              />
              <span>Sync Viewport Cameras</span>
            </label>
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={showWater}
                onChange={(e) => setShowWater(e.target.checked)}
              />
              <span>Render Solvent Waters</span>
            </label>
          </div>

          {/* Dynamic Solute-Solvent Graph network statistics panel */}
          {moleculeData && (
            <div className="panel-section network-stats">
              <h4>System Unified Metrics</h4>
              <div className="stats-grid">
                <div className="stats-col">
                  <span className="stats-label" style={{ color: '#c084fc' }}>Combined Graph</span>
                  <div className="stats-val">λ₂: {frameMetrics.overallFiedler !== undefined ? frameMetrics.overallFiedler.toFixed(4) : 'N/A'}</div>
                  <div className="stats-sub">Entropy H: {frameMetrics.overallEntropy !== undefined ? frameMetrics.overallEntropy.toFixed(3) : 'N/A'}</div>
                </div>
              </div>
              <div className="stats-grid" style={{ marginTop: '0.5rem', paddingTop: '0.5rem', borderTop: '1px solid rgba(255,255,255,0.04)' }}>
                <div className="stats-col">
                  <span className="stats-label" style={{ color: '#3b82f6' }}>Peptide Ref</span>
                  <div className="stats-val" style={{ fontSize: '0.75rem' }}>λ₂: {frameMetrics.peptideFiedler !== undefined ? frameMetrics.peptideFiedler.toFixed(4) : 'N/A'}</div>
                </div>
                <div className="stats-col border-left">
                  <span className="stats-label" style={{ color: '#10b981' }}>Water Ref</span>
                  <div className="stats-val" style={{ fontSize: '0.75rem' }}>λ₂: {frameMetrics.waterFiedler !== undefined ? frameMetrics.waterFiedler.toFixed(4) : 'N/A'}</div>
                </div>
              </div>
            </div>
          )}

          {/* Custom SVG Line Chart showing folding process */}
          {moleculeData && (
            <div className="panel-section chart-section">
              <div className="chart-header">
                <h4>Simulation History</h4>
                <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                  <button 
                    onClick={openChartInNewWindow} 
                    className="popout-btn" 
                    title="Open chart in separate window"
                  >
                    <ExternalLink size={11} />
                  </button>
                  <div className="chart-tabs">
                    <button 
                      onClick={() => setChartMode('fiedler')} 
                      className={chartMode === 'fiedler' ? 'active' : ''}
                    >
                      Fiedler (λ₂)
                    </button>
                    <button 
                      onClick={() => setChartMode('entropy')} 
                      className={chartMode === 'entropy' ? 'active' : ''}
                    >
                      Entropy (H)
                    </button>
                  </div>
                </div>
              </div>
              {renderSVGChart()}
              <div className="chart-legend">
                <span className="legend-item"><span className="legend-dot sys"></span>System</span>
                <span className="legend-item"><span className="legend-dot pep"></span>Pep Ref</span>
                <span className="legend-item"><span className="legend-dot wat"></span>Wat Ref</span>
              </div>
              <div className="network-explanation">
                Each atom coordinates 2 water molecules. Hydrophobic Carbon side chains are highlighted orange. Hydrophilic contacts form H-bonds, and hydrophobic groups pack together, driving solvent exclusion.
              </div>
            </div>
          )}

          {/* Conformation Library / History */}
          <div className="panel-section history-section">
            <div className="history-header">
              <h4>Conformation Library</h4>
              {simulationHistory.length > 0 && (
                <button className="clear-btn" onClick={() => setSimulationHistory([])}>
                  Clear
                </button>
              )}
            </div>
            {simulationHistory.length === 0 ? (
              <div className="history-empty text-muted">
                No simulations run yet. Click "Run Folding Simulation" to start.
              </div>
            ) : (
              <div className="history-list">
                {simulationHistory
                  .slice()
                  .sort((a, b) => b.finalFiedler - a.finalFiedler)
                  .map((run, idx) => {
                    const isActive = moleculeData && moleculeData.runId === run.runId;
                    return (
                      <div 
                        key={run.runId} 
                        className={`history-card ${isActive ? 'active' : ''}`}
                        onClick={() => handleSelectHistory(run)}
                      >
                        <div className="history-rank">#{idx + 1}</div>
                        <div className="history-details">
                          <div className="history-seq">{run.sequence}</div>
                          <div className="history-metrics">
                            <span className="pep-val">Overall λ₂: {run.finalFiedler.toFixed(4)}</span>
                          </div>
                        </div>
                        {isActive && <div className="active-indicator">Active</div>}
                      </div>
                    );
                  })}
              </div>
            )}
          </div>
        </section>

        <div className="viewports-workspace">
          <div className="viewports-split">
            {/* A. Unfolded Viewport */}
            <div className="viewport-container card-glass">
              <div className="viewport-label">
                <Box className="label-icon" />
                <span>
                  {moleculeData?.hasFoldedTarget 
                    ? "Experimental Conformation (PDB Ground Truth)" 
                    : "Initial State (Extended Conformation)"}
                </span>
              </div>
              <div className="canvas-mount" ref={mountUnfoldedRef} />
            </div>

            {/* B. Simulated Conformation Viewport */}
            <div className="viewport-container card-glass">
              <div className="viewport-label">
                <Layers className="label-icon" />
                <span>Simulated Conformation (Playback)</span>
              </div>
              <div className="canvas-mount" ref={mountFoldedRef} />
            </div>
          </div>
          
          {moleculeData && (
            <div className="viewport-bonds-legend card-glass">
              <span className="legend-title">Viewport Bond Types:</span>
              <span className="legend-item"><span className="legend-dash bb-bb"></span>Backbone-Backbone (H-Bonds)</span>
              <span className="legend-item"><span className="legend-dash sc-sc"></span>Sidechain-Sidechain (Packing)</span>
              <span className="legend-item"><span className="legend-dash bb-sc"></span>Backbone-Sidechain</span>
              {showWater && (
                <>
                  <span className="legend-item"><span className="legend-dash wat-wat"></span>Water-Water H-Bonds</span>
                  <span className="legend-item"><span className="legend-dash wat-pep"></span>Water-Peptide Contacts</span>
                </>
              )}
            </div>
          )}
        </div>
      </main>

      {/* Playback Control Bar */}
      {moleculeData && moleculeData.trajectory && (
        <div className="playback-control-bar card-glass">
          <button 
            onClick={() => setIsPlaying(!isPlaying)} 
            className={`playback-toggle-btn ${isPlaying ? 'playing' : ''}`}
          >
            {isPlaying ? "Pause" : "Play Simulation"}
          </button>
          <button 
            onClick={() => { setCurrentFrame(0); setIsPlaying(false); }} 
            className="playback-reset-btn"
          >
            Reset
          </button>
          <input
            type="range"
            min="0"
            max={moleculeData.trajectory.length - 1}
            value={currentFrame}
            onChange={(e) => { setCurrentFrame(Number(e.target.value)); setIsPlaying(false); }}
            className="playback-slider"
          />
          <span className="playback-counter">
            Frame {currentFrame} / {moleculeData.trajectory.length - 1} (Sweep {Math.floor(frameMetrics.step / sequence.length) || 0})
          </span>
        </div>
      )}

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
          display: flex;
          flex-direction: row;
          overflow: hidden;
          padding: 1.5rem;
          gap: 1.5rem;
          background: #020617;
          flex: 1;
        }

        .viewports-workspace {
          flex: 1;
          display: flex;
          flex-direction: column;
          gap: 1rem;
          min-width: 0;
          height: 100%;
          overflow: hidden;
        }

        .viewports-split {
          flex: 1;
          display: flex;
          gap: 1.5rem;
          overflow: hidden;
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

        .control-sidebar {
          width: 320px;
          display: flex;
          flex-direction: column;
          gap: 0.85rem;
          padding: 1.25rem;
          overflow-y: auto;
          flex-shrink: 0;
          height: 100%;
          box-shadow: 5px 0 15px rgba(0, 0, 0, 0.2);
        }

        .popout-btn {
          background: transparent;
          border: none;
          color: #94a3b8;
          cursor: pointer;
          padding: 4px;
          border-radius: 4px;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: background 0.15s, color 0.15s;
          outline: none;
        }

        .popout-btn:hover {
          background: rgba(255, 255, 255, 0.08);
          color: #fff;
        }

        .history-section {
          border-top: 1px solid rgba(255, 255, 255, 0.08);
          padding-top: 0.75rem;
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .history-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .clear-btn {
          background: transparent;
          border: none;
          color: #ef4444;
          font-size: 0.65rem;
          font-weight: 600;
          cursor: pointer;
          padding: 2px 6px;
          border-radius: 4px;
          transition: background 0.15s;
          outline: none;
        }

        .clear-btn:hover {
          background: rgba(239, 68, 68, 0.1);
        }

        .history-empty {
          font-size: 0.65rem;
          line-height: 1.35;
          color: #64748b;
          text-align: center;
          padding: 1rem 0;
        }

        .history-list {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
          max-height: 200px;
          overflow-y: auto;
          padding-right: 4px;
        }

        .history-card {
          background: rgba(255, 255, 255, 0.02);
          border: 1px solid rgba(255, 255, 255, 0.04);
          border-radius: 6px;
          padding: 0.5rem;
          display: flex;
          align-items: center;
          gap: 0.5rem;
          cursor: pointer;
          transition: background 0.15s, border-color 0.15s;
        }

        .history-card:hover {
          background: rgba(255, 255, 255, 0.05);
          border-color: rgba(255, 255, 255, 0.1);
        }

        .history-card.active {
          background: rgba(167, 139, 250, 0.08);
          border-color: rgba(167, 139, 250, 0.3);
        }

        .history-rank {
          font-size: 0.75rem;
          font-weight: 700;
          color: #a78bfa;
          width: 20px;
          text-align: center;
        }

        .history-details {
          flex: 1;
          display: flex;
          flex-direction: column;
          gap: 2px;
        }

        .history-seq {
          font-size: 0.7rem;
          font-weight: 600;
          font-family: monospace;
          color: #cbd5e1;
        }

        .history-metrics {
          display: flex;
          gap: 8px;
          font-size: 0.6rem;
        }

        .history-metrics .pep-val {
          color: #c084fc;
          font-weight: 600;
        }

        .active-indicator {
          font-size: 0.55rem;
          background: #a78bfa;
          color: #000;
          padding: 2px 4px;
          border-radius: 3px;
          font-weight: 700;
          text-transform: uppercase;
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

        .network-stats {
          border-top: 1px solid rgba(255, 255, 255, 0.08);
          padding-top: 0.65rem;
        }
        
        .stats-grid {
          display: flex;
          margin-top: 0.4rem;
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
          text-transform: uppercase;
          letter-spacing: 0.05em;
          font-weight: 600;
        }
        
        .stats-val {
          font-size: 0.85rem;
          font-weight: 700;
          color: #f8fafc;
        }
        
        .stats-sub {
          font-size: 0.65rem;
          color: #64748b;
        }

        .chart-section {
          border-top: 1px solid rgba(255, 255, 255, 0.08);
          padding-top: 0.65rem;
        }

        .chart-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 0.4rem;
        }

        .chart-tabs {
          display: flex;
          background: rgba(0,0,0,0.3);
          border-radius: 4px;
          padding: 1px;
        }

        .chart-tabs button {
          background: transparent;
          border: none;
          color: #64748b;
          font-size: 0.6rem;
          font-weight: 600;
          padding: 2px 6px;
          cursor: pointer;
          border-radius: 3px;
        }

        .chart-tabs button.active {
          background: rgba(255,255,255,0.08);
          color: #fff;
        }

        .chart-legend {
          display: flex;
          justify-content: center;
          gap: 1rem;
          margin-top: 0.4rem;
        }

        .legend-dot.sys { background: #c084fc; }
        .legend-dot.pep { background: #3b82f6; }
        .legend-dot.wat { background: #10b981; }
        
        .network-explanation {
          margin-top: 0.5rem;
          font-size: 0.6rem;
          color: #64748b;
          line-height: 1.35;
        }

        /* Viewport Bonds Legend */
        .viewport-bonds-legend {
          margin: 0.5rem 2rem;
          padding: 0.5rem 1rem;
          display: flex;
          justify-content: center;
          align-items: center;
          gap: 1.5rem;
          flex-wrap: wrap;
          font-size: 0.75rem;
          color: #94a3b8;
          z-index: 10;
        }

        .legend-title {
          font-weight: 600;
          color: #f1f5f9;
        }

        .legend-dash {
          display: inline-block;
          width: 20px;
          height: 0px;
          border-top: 2px dashed;
          margin-right: 6px;
          vertical-align: middle;
        }

        .legend-dash.bb-bb { border-color: #ef4444; }
        .legend-dash.sc-sc { border-color: #f97316; }
        .legend-dash.bb-sc { border-color: #eab308; }
        .legend-dash.wat-wat { border-color: #22d3ee; }
        .legend-dash.wat-pep { border-color: #a855f7; }

        /* Playback Control Bar */
        .playback-control-bar {
          margin: 0 2rem 1rem 2rem;
          padding: 0.75rem 1.5rem;
          display: flex;
          align-items: center;
          gap: 1rem;
          z-index: 10;
        }

        .playback-toggle-btn {
          background: linear-gradient(135deg, #a78bfa, #7c3aed);
          border: none;
          border-radius: 6px;
          padding: 0.4rem 1rem;
          color: #fff;
          font-size: 0.75rem;
          font-weight: 600;
          cursor: pointer;
          min-width: 120px;
          transition: filter 0.15s;
        }

        .playback-toggle-btn.playing {
          background: #ef4444;
        }

        .playback-toggle-btn:hover {
          filter: brightness(1.1);
        }

        .playback-reset-btn {
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 6px;
          padding: 0.4rem 0.75rem;
          color: #fff;
          font-size: 0.75rem;
          font-weight: 600;
          cursor: pointer;
          transition: background 0.15s;
        }

        .playback-reset-btn:hover {
          background: rgba(255, 255, 255, 0.1);
        }

        .playback-slider {
          flex: 1;
          accent-color: #a78bfa;
          height: 6px;
          border-radius: 3px;
          outline: none;
          cursor: pointer;
        }

        .playback-counter {
          font-size: 0.75rem;
          font-weight: 600;
          color: #a78bfa;
          font-family: monospace;
          min-width: 150px;
          text-align: right;
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
          max-width: 450px;
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
          color: #94a3b8;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }

        .error-line {
          color: #ef4444;
        }

        .warn-line {
          color: #f59e0b;
        }
      `}</style>
    </div>
  );
}
