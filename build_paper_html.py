import re
import os

workspace = "/home/eldenring/newProject"
md_path = os.path.join(workspace, "paper_draft.md")
html_path = os.path.join(workspace, "paper_publication.html")

if not os.path.exists(md_path):
    print("paper_draft.md not found!")
    exit(1)

with open(md_path, "r", encoding="utf-8") as f:
    content = f.read()

# Convert markdown basic elements to HTML
html_body = content

# Replace code blocks/mermaid
html_body = re.sub(r'```mermaid\n([\s\S]*?)\n```', r'<pre class="mermaid">\1</pre>', html_body)
html_body = re.sub(r'```bibtex\n([\s\S]*?)\n```', r'<pre class="bibtex"><code>\1</code></pre>', html_body)
html_body = re.sub(r'```([\s\S]*?)\n```', r'<pre><code>\1</code></pre>', html_body)

# Convert Markdown tables to HTML tables
def convert_table(match):
    lines = match.group(0).strip().split('\n')
    if len(lines) < 2:
        return match.group(0)
    
    html = ['<table class="academic-table">']
    headers = [h.strip() for h in lines[0].split('|')[1:-1]]
    html.append('  <thead><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr></thead>')
    
    html.append('  <tbody>')
    for line in lines[2:]:
        if '|' in line:
            cells = [c.strip() for c in line.split('|')[1:-1]]
            html.append('    <tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')
    html.append('  </tbody>')
    html.append('</table>')
    return '\n'.join(html)

table_regex = r'(\|[^\n]+\|\n\|[-:\s|]+\|\n(\|[^\n]+\|\n?)+)'
html_body = re.sub(table_regex, convert_table, html_body)

# Convert headers
html_body = re.sub(r'^# (.*)$', r'<h1 class="paper-title">\1</h1>', html_body, flags=re.MULTILINE)
html_body = re.sub(r'^## (.*)$', r'<h2 class="section-title">\1</h2>', html_body, flags=re.MULTILINE)
html_body = re.sub(r'^### (.*)$', r'<h3 class="subsection-title">\1</h3>', html_body, flags=re.MULTILINE)

# Bold & Italics
html_body = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html_body)
html_body = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html_body)

# Paragraphs
paragraphs = html_body.split('\n\n')
html_processed = []
for p in paragraphs:
    p = p.strip()
    if p.startswith('<h') or p.startswith('<table') or p.startswith('<pre') or p.startswith('<hr'):
        html_processed.append(p)
    else:
        # replace newlines with br in paragraph
        p_clean = p.replace('\n', '<br>')
        html_processed.append(f'<p>{p_clean}</p>')

final_body = '\n\n'.join(html_processed)

full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Graph-Spectral Protein Folding: Simulating Peptide Self-Assembly via Laplacian Fiedler Vector Optimization</title>
    
    <!-- MathJax for Math LaTeX Rendering -->
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

    <style>
        @page {{
            size: A4;
            margin: 20mm 20mm 20mm 20mm;
        }}
        body {{
            font-family: 'Times New Roman', Times, serif, 'Georgia', Georgia;
            line-height: 1.6;
            color: #111;
            max-width: 850px;
            margin: 0 auto;
            padding: 40px 20px;
            background-color: #fff;
        }}
        .paper-title {{
            font-size: 24pt;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
            color: #000;
            line-height: 1.25;
        }}
        .author-block {{
            text-align: center;
            font-size: 11pt;
            margin-bottom: 30px;
            color: #333;
        }}
        h2.section-title {{
            font-size: 14pt;
            font-weight: bold;
            border-bottom: 1.5px solid #222;
            padding-bottom: 4px;
            margin-top: 30px;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        h3.subsection-title {{
            font-size: 12pt;
            font-weight: bold;
            margin-top: 20px;
            margin-bottom: 8px;
        }}
        p {{
            font-size: 11pt;
            text-align: justify;
            text-justify: inter-word;
            margin-bottom: 14px;
            text-indent: 1.5em;
        }}
        .academic-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 10pt;
        }}
        .academic-table th, .academic-table td {{
            border-top: 1px solid #111;
            border-bottom: 1px solid #111;
            padding: 8px 10px;
            text-align: center;
        }}
        .academic-table th {{
            font-weight: bold;
            background-color: #f8f9fa;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 12px;
            border-radius: 4px;
            font-family: 'Courier New', Courier, monospace;
            font-size: 9.5pt;
            overflow-x: auto;
        }}
        .figure-container {{
            text-align: center;
            margin: 25px 0;
        }}
        .figure-container img {{
            max-width: 95%;
            height: auto;
            border: 1px solid #ddd;
        }}
        .figure-caption {{
            font-size: 9.5pt;
            color: #444;
            margin-top: 8px;
            font-style: italic;
        }}
        .print-btn {{
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 18px;
            background-color: #0066cc;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            font-weight: bold;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            z-index: 1000;
        }}
        .print-btn:hover {{
            background-color: #004c99;
        }}
        @media print {{
            .print-btn {{ display: none; }}
            body {{ padding: 0; margin: 0; width: 100%; max-width: 100%; }}
        }}
    </style>
</head>
<body>
    <button class="print-btn" onclick="window.print()">🖨️ Save as PDF / Print</button>
    
    {final_body}
</body>
</html>
"""

with open(html_path, "w", encoding="utf-8") as f:
    f.write(full_html)

print(f"Successfully generated academic publication HTML at {html_path}")
