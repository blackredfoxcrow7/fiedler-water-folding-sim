import json
import os

def build_interactive_viewer():
    data_dict = {}
    
    files = {
        "Deca-alanine (AAAAAAAAAA)": "aaaaaaaaaa_differentiable_folded.json",
        "Deca-alanine Hybrid (AAAAAAAAAA)": "aaaaaaaaaa_hybrid_folded.json",
        "Chignolin Mutant (CLN025)": "yydpetgtwy_differentiable_folded.json",
        "Chignolin Mutant Hybrid (CLN025)": "yydpetgtwy_hybrid_folded.json",
        "Wild-type Chignolin (1UAO)": "gydpetgtwg_differentiable_folded.json",
        "Wild-type Chignolin Hybrid (1UAO)": "gydpetgtwg_hybrid_folded.json",
        "Trp-cage (1L2Y)": "nlyiqwlkdggpssgrppps_differentiable_folded.json",
        "Trp-cage Hybrid (1L2Y)": "nlyiqwlkdggpssgrppps_hybrid_folded.json"
    }
    
    for label, filename in files.items():
        if os.path.exists(filename):
            with open(filename, "r") as f:
                data_dict[label] = json.load(f)
            print(f"Loaded: {filename}")
        else:
            print(f"Warning: {filename} not found, skipping.")
            
    if not data_dict:
        print("No coordinate files found. Build failed.")
        return
        
    html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Spectral Folding Interactive Viewer</title>
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <!-- Plotly CDN -->
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <style>
        :root {{
            --bg-color: #0b0b0d;
            --card-bg: rgba(20, 20, 25, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --accent-cyan: #00ffd5;
            --accent-purple: #7000ff;
            --text-color: #e5e5ed;
            --text-muted: #8e8e9f;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            height: 100vh;
            overflow: hidden;
            display: flex;
        }}
        
        /* Dashboard Layout */
        .sidebar {{
            width: 360px;
            background: var(--card-bg);
            border-right: 1px solid var(--border-color);
            backdrop-filter: blur(20px);
            padding: 30px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            z-index: 10;
        }}
        
        .main-content {{
            flex-grow: 1;
            position: relative;
            background-color: #000;
        }}
        
        #plot-container {{
            width: 100%;
            height: 100%;
        }}
        
        /* Headers & Typography */
        h1 {{
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 8px;
            background: linear-gradient(90deg, #00ffd5, #8800ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .subtitle {{
            font-size: 12px;
            color: var(--text-muted);
            margin-bottom: 25px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        /* Select & Controls */
        .control-group {{
            margin-bottom: 25px;
        }}
        
        label {{
            font-size: 11px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            display: block;
            margin-bottom: 8px;
        }}
        
        select {{
            width: 100%;
            padding: 12px 16px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-color);
            font-size: 14px;
            font-weight: 600;
            outline: none;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        
        select:focus {{
            border-color: var(--accent-cyan);
            box-shadow: 0 0 10px rgba(0, 255, 213, 0.2);
        }}
        
        /* Info Cards */
        .info-card {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
        }}
        
        .stat-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 12px;
            font-size: 13px;
        }}
        
        .stat-row:last-child {{
            margin-bottom: 0;
        }}
        
        .stat-label {{
            color: var(--text-muted);
        }}
        
        .stat-val {{
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            color: #fff;
        }}
        
        .legend-card {{
            border-top: 1px solid var(--border-color);
            padding-top: 20px;
            margin-top: 20px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            font-size: 12px;
            margin-bottom: 8px;
            color: var(--text-muted);
        }}
        
        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 3px;
            margin-right: 8px;
        }}
        
        /* Interactive Instruction Overlay */
        .overlay-tip {{
            position: absolute;
            bottom: 25px;
            right: 25px;
            background: rgba(15, 15, 20, 0.85);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 10px 16px;
            font-size: 11px;
            color: var(--text-muted);
            pointer-events: none;
            backdrop-filter: blur(10px);
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .footer {{
            font-size: 10px;
            color: var(--text-muted);
            border-top: 1px solid var(--border-color);
            padding-top: 15px;
            text-align: center;
        }}
    </style>
</head>
<body>

    <div class="sidebar">
        <div>
            <h1>3D Spectral Folding</h1>
            <div class="subtitle">Interactive Viewer</div>
            
            <div class="control-group">
                <label for="peptide-select">ペプチドモデルの選択</label>
                <select id="peptide-select">
                    {"".join([f'<option value="{lbl}">{lbl}</option>' for lbl in data_dict.keys()])}
                </select>
            </div>
            
            <div class="info-card">
                <label>モデル情報</label>
                <div class="stat-row">
                    <span class="stat-label">アミノ酸配列</span>
                    <span class="stat-val" id="info-seq">-</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">原子数</span>
                    <span class="stat-val" id="info-atoms">-</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">慣性半径 (Rg)</span>
                    <span class="stat-val" id="info-rg" style="color: var(--accent-cyan);">-</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">水素結合数</span>
                    <span class="stat-val" id="info-hbonds">-</span>
                </div>
            </div>
            
            <div class="legend-card">
                <label>凡例</label>
                <div class="legend-item">
                    <div class="legend-color" style="background: linear-gradient(90deg, #00e1ff, #ff4d4d);"></div>
                    <span>主鎖 Cα トレース (N末端 ➡ C末端)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="border: 1px dashed var(--accent-cyan); background: transparent;"></div>
                    <span>水素結合 (O ··· N 距離 < 4.5 Å)</span>
                </div>
            </div>
        </div>
        
        <div class="footer">
            Differentiable Spectral Folding Engine &copy; 2026
        </div>
    </div>
    
    <div class="main-content">
        <div id="plot-container"></div>
        <div class="overlay-tip">
            <span>🖱️ ドラッグで回転</span>
            <span>|</span>
            <span>🔍 スクロールでズーム</span>
            <span>|</span>
            <span>✋ 右ドラッグで平行移動</span>
        </div>
    </div>

    <script>
        // Embed the coordinate data
        const peptideData = {json.dumps(data_dict)};
        
        function updatePlot() {{
            const select = document.getElementById("peptide-select");
            const modelName = select.value;
            const data = peptideData[modelName];
            
            if (!data) return;
            
            const sequence = data.sequence;
            const atoms = data.atoms;
            const coords = data.foldedCoords;
            const atomResidues = data.atomResidues;
            
            // Extract CA coordinates for backbone trace
            const caIndices = [];
            const caLabels = [];
            
            atoms.forEach((atom, idx) => {{
                if (atom.symbol === "CA") {{
                    caIndices.push(idx);
                    // Label residue name + index
                    caLabels.push(`${{sequence[atomResidues[idx]]}}${{atomResidues[idx] + 1}}`);
                }}
            }});
            
            const caX = caIndices.map(idx => coords[idx][0]);
            const caY = caIndices.map(idx => coords[idx][1]);
            const caZ = caIndices.map(idx => coords[idx][2]);
            
            // Find active hydrogen bonds
            const hBonds = [];
            const oIndices = [];
            const nIndices = [];
            
            atoms.forEach((atom, idx) => {{
                if (atom.symbol === "O") oIndices.push(idx);
                if (atom.symbol === "N") nIndices.push(idx);
            }});
            
            oIndices.forEach(oIdx => {{
                nIndices.forEach(nIdx => {{
                    if (Math.abs(atomResidues[oIdx] - atomResidues[nIdx]) >= 3) {{
                        const dist = Math.sqrt(
                            Math.pow(coords[oIdx][0] - coords[nIdx][0], 2) +
                            Math.pow(coords[oIdx][1] - coords[nIdx][1], 2) +
                            Math.pow(coords[oIdx][2] - coords[nIdx][2], 2)
                        );
                        if (dist < 4.5) {{
                            hBonds.push({{
                                start: coords[oIdx],
                                end: coords[nIdx],
                                distance: dist
                            }});
                        }}
                    }}
                }});
            }});
            
            // Calculate Rg
            let cx = 0, cy = 0, cz = 0;
            coords.forEach(c => {{
                cx += c[0];
                cy += c[1];
                cz += c[2];
            }});
            cx /= coords.length;
            cy /= coords.length;
            cz /= coords.length;
            
            let rgSq = 0;
            coords.forEach(c => {{
                rgSq += Math.pow(c[0] - cx, 2) + Math.pow(c[1] - cy, 2) + Math.pow(c[2] - cz, 2);
            }});
            const rg = Math.sqrt(rgSq / coords.length);
            
            // Update Sidebar stats
            document.getElementById("info-seq").innerText = sequence;
            document.getElementById("info-atoms").innerText = coords.length;
            document.getElementById("info-rg").innerText = `${{rg.toFixed(3)}} Å`;
            document.getElementById("info-hbonds").innerText = `${{hBonds.length}} 本`;
            
            // --- Plotly rendering ---
            const traces = [];
            
            // 1. C-alpha backbone ribbon
            traces.push({{
                type: 'scatter3d',
                mode: 'lines+markers',
                x: caX,
                y: caY,
                z: caZ,
                text: caLabels,
                hoverinfo: 'text+x+y+z',
                line: {{
                    width: 8,
                    color: caIndices.map((_, i) => i),
                    colorscale: 'Portland', // cool gradient from N (blue) to C (red)
                    reversescale: false
                }},
                marker: {{
                    size: 8,
                    color: caIndices.map((_, i) => i),
                    colorscale: 'Portland',
                    line: {{
                        color: '#ffffff',
                        width: 1.5
                    }}
                }},
                name: 'Cα Trace'
            }});
            
            // 2. Hydrogen bonds (dashed lines)
            hBonds.forEach((bond, idx) => {{
                traces.push({{
                    type: 'scatter3d',
                    mode: 'lines',
                    x: [bond.start[0], bond.end[0]],
                    y: [bond.start[1], bond.end[1]],
                    z: [bond.start[2], bond.end[2]],
                    line: {{
                        color: '#00ffd5',
                        width: 4,
                        dash: 'dash'
                    }},
                    hoverinfo: 'none',
                    showlegend: idx === 0, // show only the first in legend
                    name: '水素結合'
                }});
            }});
            
            // Auto aspect ratio calculation
            const allX = coords.map(c => c[0]);
            const allY = coords.map(c => c[1]);
            const allZ = coords.map(c => c[2]);
            
            const minX = Math.min(...allX), maxX = Math.max(...allX);
            const minY = Math.min(...allY), maxY = Math.max(...allY);
            const minZ = Math.min(...allZ), maxZ = Math.max(...allZ);
            
            const midX = (minX + maxX) / 2;
            const midY = (minY + maxY) / 2;
            const midZ = (minZ + maxZ) / 2;
            
            const maxRange = Math.max(maxX - minX, maxY - minY, maxZ - minZ) / 2 + 1;
            
            const layout = {{
                paper_bgcolor: '#000000',
                plot_bgcolor: '#000000',
                margin: {{ l: 0, r: 0, t: 0, b: 0 }},
                showlegend: false,
                scene: {{
                    xaxis: {{
                        range: [midX - maxRange, midX + maxRange],
                        backgroundcolor: '#000',
                        gridcolor: '#222',
                        showbackground: true,
                        zerolinecolor: '#333',
                        tickfont: {{ color: '#fff' }},
                        showticklabels: false,
                        title: ''
                    }},
                    yaxis: {{
                        range: [midY - maxRange, midY + maxRange],
                        backgroundcolor: '#00',
                        gridcolor: '#222',
                        showbackground: true,
                        zerolinecolor: '#333',
                        tickfont: {{ color: '#fff' }},
                        showticklabels: false,
                        title: ''
                    }},
                    zaxis: {{
                        range: [midZ - maxRange, midZ + maxRange],
                        backgroundcolor: '#00',
                        gridcolor: '#222',
                        showbackground: true,
                        zerolinecolor: '#333',
                        tickfont: {{ color: '#fff' }},
                        showticklabels: false,
                        title: ''
                    }},
                    aspectmode: 'manual',
                    aspectratio: {{ x: 1, y: 1, z: 1 }}
                }}
            }};
            
            Plotly.newPlot('plot-container', traces, layout, {{ responsive: true }});
        }}
        
        // Initial load
        document.getElementById("peptide-select").addEventListener("change", updatePlot);
        window.addEventListener("load", updatePlot);
    </script>
</body>
</html>
"""
    
    with open("viewer.html", "w") as f:
        f.write(html_content)
    print(" viewer.html was created successfully!")

if __name__ == "__main__":
    build_interactive_viewer()
