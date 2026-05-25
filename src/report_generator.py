import os
import pandas as pd


def generate_single_html_report(audio_path, result_dict, comparison_results, output_path="qc_report.html"):
    """Generate an HTML QC report for a single audio file."""
    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>Audio QC Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #f4f7f6; }}
            .container {{ max-width: 900px; margin: auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .score-pass {{ color: #27ae60; font-weight: bold; font-size: 1.2em; }}
            .score-borderline {{ color: #f39c12; font-weight: bold; font-size: 1.2em; }}
            .score-fail {{ color: #c0392b; font-weight: bold; font-size: 1.2em; }}
            .diff-insert {{ color: #2980b9; text-decoration: underline; background-color: #ebf5fb; padding: 2px; border-radius: 3px; }}
            .diff-delete {{ color: #e74c3c; text-decoration: line-through; background-color: #fdedec; padding: 2px; border-radius: 3px; }}
            .diff-replace {{ color: #d35400; background-color: #fdf2e9; padding: 2px; border-radius: 3px; }}
            .box {{ border: 1px solid #e2e8f0; padding: 20px; border-radius: 8px; margin-bottom: 20px; background: #ffffff; }}
            .script-box {{ background: #f8f9f9; padding: 15px; border-left: 4px solid #bdc3c7; margin: 10px 0; font-size: 1.1em; }}
            audio {{ width: 100%; margin-top: 15px; outline: none; }}
            ul {{ list-style-type: none; padding-left: 0; }}
            li {{ margin-bottom: 10px; padding: 10px; background: #f8f9fa; border-radius: 5px; border-left: 3px solid #ccc; }}
            .type-replace {{ border-left-color: #f39c12; }}
            .type-delete {{ border-left-color: #e74c3c; }}
            .type-insert {{ border-left-color: #3498db; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎙️ Audio QC Report</h1>
            <div class="box">
                <h3>Audio File: {os.path.basename(audio_path)}</h3>
                <audio controls>
                    <source src="{audio_path}">
                </audio>
            </div>
    """

    if comparison_results:
        c = comparison_results
        ml_score = c['minilm_score']
        if ml_score >= 80:
            status_cls, status_txt = "score-pass", f"✅ PASS ({ml_score:.1f}%)"
        else:
            status_cls, status_txt = "score-fail", f"❌ FAIL ({ml_score:.1f}%)"

        html += f"""
            <div class="box">
                <h2>Comparison Results</h2>
                <p><strong>Semantic Score (MiniLM):</strong> <span class="{status_cls}">{status_txt}</span></p>
                <p><strong>Exact Match Score:</strong> {c['diff_score']:.1f}%</p>

                <h3>Original Script vs Transcribed Audio</h3>
                <div class="script-box">
                    <strong>📝 Script:</strong><br>{c['reference_text']}
                </div>
                <div class="script-box">
                    <strong>🗣️ Transcribed:</strong><br>{c['transcribed_text']}
                </div>

                <h3>Detailed Differences:</h3>
                <ul>
        """
        if not c['diffs']:
            html += "<li style='border-left-color: #2ecc71;'>✅ Perfect Match! No missing or extra words.</li>"
        else:
            for d in c['diffs']:
                if d['type'] == 'replace':
                    html += f"<li class='type-replace'>❌ <strong>Replaced:</strong> <span class='diff-delete'>{d['ref']}</span> &rarr; <span class='diff-replace'>{d['asr']}</span></li>"
                elif d['type'] == 'delete':
                    html += f"<li class='type-delete'>➖ <strong>Missing (Skipped):</strong> <span class='diff-delete'>{d['ref']}</span></li>"
                elif d['type'] == 'insert':
                    html += f"<li class='type-insert'>➕ <strong>Extra (Added):</strong> <span class='diff-insert'>{d['asr']}</span></li>"
        html += """
                </ul>
            </div>
        """
    else:
        html += f"""
            <div class="box">
                <h2>Transcription Only (No Script Provided)</h2>
                <div class="script-box">
                    {result_dict.get('text', '')}
                </div>
            </div>
        """

    html += """
        </div>
    </body>
    </html>
    """

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"📊 HTML Report generated at: {output_path}")


def generate_batch_html_report(batch_results, output_path="batch_report.html"):
    """Generate a unified HTML report for multiple audio QC results."""
    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>Batch Audio QC Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #f4f7f6; }}
            .container {{ max-width: 1200px; margin: auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; text-align: center; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 12px; border: 1px solid #e2e8f0; text-align: left; }}
            th {{ background-color: #f8f9fa; color: #2c3e50; }}
            .score-pass {{ color: #27ae60; font-weight: bold; }}
            .score-borderline {{ color: #f39c12; font-weight: bold; }}
            .score-fail {{ color: #c0392b; font-weight: bold; }}
            .status-tag {{ padding: 4px 8px; border-radius: 4px; font-size: 0.9em; color: white; display: inline-block; }}
            .tag-pass {{ background-color: #27ae60; }}
            .tag-borderline {{ background-color: #f39c12; }}
            .tag-fail {{ background-color: #c0392b; }}
            audio {{ width: 200px; height: 30px; }}
            .summary {{ display: flex; justify-content: space-around; margin-bottom: 30px; background: #f8f9fa; padding: 20px; border-radius: 8px; }}
            .summary-item {{ text-align: center; }}
            .summary-val {{ font-size: 1.5em; font-weight: bold; display: block; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Batch Audio QC Report</h1>
    """

    total = len(batch_results)
    valid_results = [r for r in batch_results if r and r.get('comparison')]
    passed = sum(1 for r in valid_results if r['comparison']['minilm_score'] >= 80)
    failed = total - passed

    html += f"""
            <div class="summary">
                <div class="summary-item">
                    <span class="summary-val">{total}</span>
                    <span class="summary-label">Total Files</span>
                </div>
                <div class="summary-item">
                    <span class="summary-val score-pass">{passed}</span>
                    <span class="summary-label">Pass (>=80%)</span>
                </div>
                <div class="summary-item">
                    <span class="summary-val score-fail">{failed}</span>
                    <span class="summary-label">Fail (<80%)</span>
                </div>
            </div>

            <table>
                <thead>
                    <tr>
                        <th>ID / Filename</th>
                        <th>Status</th>
                        <th>Score</th>
                        <th>Script (Normalized)</th>
                        <th>ASR (Normalized)</th>
                        <th>Audio</th>
                    </tr>
                </thead>
                <tbody>
    """

    for r in batch_results:
        if not r:
            continue
        c = r.get('comparison')
        audio_name = os.path.basename(r.get('audio_path', 'unknown'))

        if c:
            score = c['minilm_score']
            if score >= 80:
                tag, cls = "PASS", "tag-pass"
            else:
                tag, cls = "FAIL", "tag-fail"

            html += f"""
                <tr>
                    <td>{audio_name}</td>
                    <td><span class="status-tag {cls}">{tag}</span></td>
                    <td class="{cls.replace('tag','score')}">{score:.1f}%</td>
                    <td style="font-size: 0.85em;">{c.get('norm_s','-')}</td>
                    <td style="font-size: 0.85em;">{c.get('norm_a','-')}</td>
                    <td><audio controls preload="none"><source src="{r.get('audio_path','')}"></audio></td>
                </tr>
            """
        else:
            html += f"""
                <tr>
                    <td>{audio_name}</td>
                    <td colspan="4">No comparison available</td>
                    <td><audio controls preload="none"><source src="{r.get('audio_path','')}"></audio></td>
                </tr>
            """

    html += """
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\n✨ Unified Batch Report generated at: {output_path}")


def generate_pipeline_html_report(df, output_path):
    """Generate a clean HTML report of the IQR pipeline QC results."""
    print(f"Generating HTML report at {output_path}...", flush=True)

    total = len(df)
    passed = len(df[df['final_status'] == 'PASS'])
    recovered = len(df[df['final_status'] == 'PASS (AI Recovered)'])
    failed = total - passed - recovered

    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>IQR Audio QC Pipeline Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #f4f7f6; }}
            .container {{ max-width: 1200px; margin: auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; text-align: center; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 0.9em; }}
            th, td {{ padding: 10px; border: 1px solid #e2e8f0; text-align: left; }}
            th {{ background-color: #f8f9fa; color: #2c3e50; position: sticky; top: 0; }}
            .status-pass {{ background-color: #27ae60; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }}
            .status-recovered {{ background-color: #2980b9; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }}
            .status-fail {{ background-color: #e74c3c; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }}
            .summary {{ display: flex; justify-content: space-around; margin-bottom: 30px; background: #f8f9fa; padding: 20px; border-radius: 8px; }}
            .summary-item {{ text-align: center; }}
            .summary-val {{ font-size: 1.5em; font-weight: bold; display: block; }}
            .text-cell {{ max-width: 200px; word-break: break-word; font-size: 0.85em; }}
            .whisper-cell {{ max-width: 200px; word-break: break-word; font-size: 0.85em; color: #1a5276; }}
            .empty-whisper {{ color: #bdc3c7; font-style: italic; font-size: 0.8em; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>IQR Audio QC Pipeline Report</h1>
            <div class="summary">
                <div class="summary-item">
                    <span class="summary-val">{total}</span>
                    <span class="summary-label">Total Audios</span>
                </div>
                <div class="summary-item">
                    <span class="summary-val" style="color: #27ae60;">{passed}</span>
                    <span class="summary-label">Normal (PASS)</span>
                </div>
                <div class="summary-item">
                    <span class="summary-val" style="color: #2980b9;">{recovered}</span>
                    <span class="summary-label">AI Recovered</span>
                </div>
                <div class="summary-item">
                    <span class="summary-val" style="color: #e74c3c;">{failed}</span>
                    <span class="summary-label">Failed</span>
                </div>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Speaker</th>
                        <th>Status</th>
                        <th>Reason</th>
                        <th>Rate (Syll/Sec)</th>
                        <th>Syllables</th>
                        <th>Duration (s)</th>
                        <th>AI Score</th>
                        <th>Script (Reference)</th>
                        <th>Whisper (Transcribed)</th>
                        <th>Audio</th>
                    </tr>
                </thead>
                <tbody>
    """

    def sort_key(status):
        if status.startswith('FAIL'):
            return 0
        if status == 'PASS (AI Recovered)':
            return 1
        return 2

    df_sorted = df.copy()
    df_sorted['sort_val'] = df_sorted['final_status'].apply(sort_key)
    df_sorted = df_sorted.sort_values(['sort_val', 'user_id'])

    for idx, row in df_sorted.iterrows():
        status = row.get('final_status', 'UNKNOWN')
        if status == 'PASS':
            cls = 'status-pass'
        elif status == 'PASS (AI Recovered)':
            cls = 'status-recovered'
        else:
            cls = 'status-fail'

        audio_id = row.get('tid', row.get('id', row.get('audio_index', idx)))
        speaker = row.get('user_id', 'Unknown')
        reason = row.get('outlier_type', '-')
        rate = f"{row.get('speech_rate', 0):.2f}" if pd.notna(row.get('speech_rate')) else "-"
        syllables = row.get('syllable_count', 0)
        duration = f"{row.get('audio_duration', 0):.2f}" if pd.notna(row.get('audio_duration')) else "-"
        ai_score = f"{row.get('ai_score', 0):.1f}%" if pd.notna(row.get('ai_score')) else "-"
        audio_url = row.get('audio', row.get('audiourl', ''))
        script_text = str(row.get('text', '')).replace('<', '&lt;').replace('>', '&gt;')
        raw_whisper = row.get('whisper_text', '')
        if pd.isna(raw_whisper) or raw_whisper == '' or raw_whisper is None:
            whisper_cell = '<span class="empty-whisper">-</span>'
        else:
            whisper_cell = f'<span class="whisper-cell">{str(raw_whisper).replace("<", "&lt;").replace(">", "&gt;")}</span>'

        html += f"""
            <tr>
                <td>{audio_id}</td>
                <td style="max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="{speaker}">{speaker}</td>
                <td><span class="{cls}">{status}</span></td>
                <td>{reason}</td>
                <td>{rate}</td>
                <td>{syllables}</td>
                <td>{duration}</td>
                <td>{ai_score}</td>
                <td class="text-cell">{script_text}</td>
                <td>{whisper_cell}</td>
                <td>
                    <audio controls preload="none" style="height: 30px; width: 150px;">
                        <source src="{audio_url}">
                    </audio>
                </td>
            </tr>
        """

    html += """
                </tbody>
            </table>
    """

    # Whisper vs Script comparison section (only rows that went through AI check)
    ai_rows = df_sorted[df_sorted['ai_score'].notna()]
    if not ai_rows.empty:
        html += """
            <h2 style="color:#2c3e50; margin-top:40px;">&#127897; Whisper vs Script Comparison</h2>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Status</th>
                        <th>AI Score</th>
                        <th>Script (Reference)</th>
                        <th>Whisper (Transcribed)</th>
                    </tr>
                </thead>
                <tbody>
        """
        for _, row in ai_rows.iterrows():
            audio_id = row.get('tid', row.get('id', '-'))
            status = row.get('final_status', '')
            if status == 'PASS (AI Recovered)':
                cls = 'status-recovered'
            elif status == 'PASS':
                cls = 'status-pass'
            else:
                cls = 'status-fail'
            ai_score = f"{row.get('ai_score', 0):.1f}%"
            script_text = str(row.get('text', '')).replace('<', '&lt;').replace('>', '&gt;')
            whisper_text = str(row.get('whisper_text', '')).replace('<', '&lt;').replace('>', '&gt;')
            html += f"""
                <tr>
                    <td>{audio_id}</td>
                    <td><span class="{cls}">{status}</span></td>
                    <td style="font-weight:bold;">{ai_score}</td>
                    <td style="color:#555;">{script_text}</td>
                    <td style="color:#1a5276;">{whisper_text}</td>
                </tr>
            """
        html += """
                </tbody>
            </table>
        """

    html += """
        </div>
    </body>
    </html>
    """

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Done: Report saved to {output_path}", flush=True)
