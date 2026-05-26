import re

def extract_summary_vals(html):
    vals = re.findall(r'<span class="summary-val"[^>]*>(\d+)</span>', html)
    return [int(v) for v in vals]

def extract_tbody(html):
    m = re.search(r'<tbody>(.*?)</tbody>', html, re.DOTALL)
    return m.group(1) if m else ''

with open(r'data\Email1_qc_report.html', encoding='utf-8') as f:
    e1 = f.read()

print("Reading Email2 (16MB)...")
with open(r'data\Email2_qc_report.html', encoding='utf-8') as f:
    e2 = f.read()

v1 = extract_summary_vals(e1)
v2 = extract_summary_vals(e2)
print('Email1 stats (total/pass/recovered/fail):', v1)
print('Email2 stats (total/pass/recovered/fail):', v2)

combined = [v1[i] + v2[i] for i in range(4)]
tbody1 = extract_tbody(e1)
tbody2 = extract_tbody(e2)

css = """
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #f4f7f6; }
.container { max-width: 1400px; margin: auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
h1 { color: #2c3e50; text-align: center; }
table { width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 0.9em; }
th, td { padding: 10px; border: 1px solid #e2e8f0; text-align: left; }
th { background-color: #f8f9fa; color: #2c3e50; position: sticky; top: 0; }
.status-pass { background-color: #27ae60; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
.status-recovered { background-color: #2980b9; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
.status-fail { background-color: #e74c3c; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
.summary { display: flex; justify-content: space-around; margin-bottom: 30px; background: #f8f9fa; padding: 20px; border-radius: 8px; }
.summary-item { text-align: center; }
.summary-val { font-size: 1.5em; font-weight: bold; display: block; }
.text-cell { max-width: 200px; word-break: break-word; font-size: 0.85em; }
.whisper-cell { max-width: 200px; word-break: break-word; font-size: 0.85em; color: #1a5276; }
.empty-whisper { color: #bdc3c7; font-style: italic; font-size: 0.8em; }
.tabs { display: flex; gap: 8px; margin-bottom: 20px; border-bottom: 2px solid #e2e8f0; }
.tab-btn { padding: 10px 24px; border: none; background: #f8f9fa; color: #2c3e50; cursor: pointer; border-radius: 6px 6px 0 0; font-size: 1em; font-weight: bold; transition: background 0.2s; }
.tab-btn.active { background: #2c3e50; color: white; }
.tab-content { display: none; }
.tab-content.active { display: block; }
.badge { display: inline-block; background: #e2e8f0; color: #2c3e50; border-radius: 12px; padding: 2px 10px; font-size: 0.85em; margin-left: 6px; }
"""

table_header = """
<thead>
<tr>
<th>ID</th><th>Speaker</th><th>Status</th><th>Reason</th>
<th>Rate (Syll/Sec)</th><th>Syllables</th><th>Duration (s)</th>
<th>AI Score</th><th>Script (Reference)</th><th>Whisper (Transcribed)</th><th>Audio</th>
</tr>
</thead>
"""

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>IQR Audio QC Pipeline Report</title>
<style>{css}</style>
</head>
<body>
<div class="container">
<h1>IQR Audio QC Pipeline Report</h1>
<div class="summary">
  <div class="summary-item">
    <span class="summary-val">{combined[0]:,}</span>
    <span class="summary-label">Total Audios</span>
  </div>
  <div class="summary-item">
    <span class="summary-val" style="color:#27ae60;">{combined[1]:,}</span>
    <span class="summary-label">Normal (PASS)</span>
  </div>
  <div class="summary-item">
    <span class="summary-val" style="color:#2980b9;">{combined[2]:,}</span>
    <span class="summary-label">AI Recovered</span>
  </div>
  <div class="summary-item">
    <span class="summary-val" style="color:#e74c3c;">{combined[3]:,}</span>
    <span class="summary-label">Failed</span>
  </div>
</div>

<div class="tabs">
  <button class="tab-btn active" onclick="switchTab('email1',this)">Email 1 <span class="badge">{v1[0]:,}</span></button>
  <button class="tab-btn" onclick="switchTab('email2',this)">Email 2 <span class="badge">{v2[0]:,}</span></button>
</div>

<div id="email1" class="tab-content active">
  <div class="summary">
    <div class="summary-item"><span class="summary-val">{v1[0]}</span><span class="summary-label">Total</span></div>
    <div class="summary-item"><span class="summary-val" style="color:#27ae60;">{v1[1]}</span><span class="summary-label">PASS</span></div>
    <div class="summary-item"><span class="summary-val" style="color:#2980b9;">{v1[2]}</span><span class="summary-label">AI Recovered</span></div>
    <div class="summary-item"><span class="summary-val" style="color:#e74c3c;">{v1[3]}</span><span class="summary-label">Failed</span></div>
  </div>
  <table>{table_header}<tbody>{tbody1}</tbody></table>
</div>

<div id="email2" class="tab-content">
  <div class="summary">
    <div class="summary-item"><span class="summary-val">{v2[0]:,}</span><span class="summary-label">Total</span></div>
    <div class="summary-item"><span class="summary-val" style="color:#27ae60;">{v2[1]:,}</span><span class="summary-label">PASS</span></div>
    <div class="summary-item"><span class="summary-val" style="color:#2980b9;">{v2[2]:,}</span><span class="summary-label">AI Recovered</span></div>
    <div class="summary-item"><span class="summary-val" style="color:#e74c3c;">{v2[3]:,}</span><span class="summary-label">Failed</span></div>
  </div>
  <table>{table_header}<tbody>{tbody2}</tbody></table>
</div>

</div>
<script>
function switchTab(id, btn) {{
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
}}
</script>
</body>
</html>"""

out_path = r'index.html'
print(f"Writing index.html...")
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)

import os
size_mb = os.path.getsize(out_path) / 1024 / 1024
print(f"Done! index.html = {size_mb:.1f} MB")
