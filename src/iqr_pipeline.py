import os
import sys

# Reconfigure stdout/stderr to support Unicode/Thai/Emojis on Windows
if sys.platform.startswith('win'):
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    if hasattr(sys.stderr, 'reconfigure'):
        try:
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
from tqdm import tqdm
from pythainlp.tokenize import syllable_tokenize
import argparse
import json
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Import our whisper engine
# Ensure the src folder is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from whisper_engine import stable_whisper_transcribe, load_whisper_model, transcribe_with_model
    from audio_processor import download_audio, trim_audio_silence, denoise_audio
    from report_generator import generate_pipeline_html_report
except ImportError as e:
    print(f"Error: Could not import required module: {e}", flush=True)
    print("Please ensure all module files exist in the same directory.", flush=True)
    sys.exit(1)

# ==================== Pipeline Config ====================
# Audio Enhancement (DeepFilterNet; pip install deepfilternet)
ENHANCE_AUDIO = True

# IQR Outlier Detection
IQR_MULTIPLIER     = 1.5   # Tukey standard; raise to 2.0–2.5 to be less aggressive
MIN_AUDIO_DURATION = 0.5   # seconds — shorter recordings auto-FAIL without IQR

# AI Recovery (Whisper + MiniLM)
# Binary scoring:
#   score >= AI_CONFIDENT_THRESHOLD → PASS (AI Recovered)      auto-pass
#   score <  AI_CONFIDENT_THRESHOLD → FAIL
AI_CONFIDENT_THRESHOLD  = 80.0  # score needed for auto-pass

def get_audio_duration(url):
    """Get audio duration using ffprobe without downloading the whole file."""
    if pd.isna(url) or not url:
        return None
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', str(url)
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass
    return None

def count_syllables(text):
    """Count syllables in Thai text using PyThaiNLP."""
    if pd.isna(text) or not str(text).strip():
        return 0
    # pythainlp's syllable_tokenize returns a list of syllables
    syllables = syllable_tokenize(str(text))
    return len(syllables)

def process_durations_multithreaded(df, max_workers=20, cache_file='data/duration_cache.json'):
    """Fetch audio durations with multithreading and caching."""
    durations = {}
    
    # Load cache if exists
    if os.path.exists(cache_file):
        print(f"Loading duration cache from {cache_file}...", flush=True)
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                durations = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load cache: {e}", flush=True)

    # Identify URLs that need fetching
    urls_to_fetch = []
    for idx, row in df.iterrows():
        # Check 'audio' or 'audiourl' columns
        url = row.get('audio', row.get('audiourl', None))
        if url and str(url) not in durations:
            urls_to_fetch.append((idx, url))
            
    if urls_to_fetch:
        print(f"Fetching duration for {len(urls_to_fetch)} new audio files...", flush=True)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(get_audio_duration, url): (idx, url) for idx, url in urls_to_fetch}
            
            for future in tqdm(as_completed(future_to_idx), total=len(urls_to_fetch), desc="Fetching Durations"):
                idx, url = future_to_idx[future]
                try:
                    duration = future.result()
                    durations[str(url)] = duration
                except Exception:
                    durations[str(url)] = None
                    
        # Save updated cache
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(durations, f, ensure_ascii=False, indent=2)
            print(f"Done: Updated duration cache at {cache_file}", flush=True)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}", flush=True)
            
    # Apply durations back to dataframe
    df['audio_duration'] = df.apply(lambda r: durations.get(str(r.get('audio', r.get('audiourl', ''))), None), axis=1)
    return df



def run_pipeline(input_csv, max_workers=20, is_test_mode=False, sample=None):
    print("="*70)
    print("Starting IQR Audio QC Pipeline")
    print("="*70, flush=True)
    
    # 1. Load Data
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"Error: Failed to load CSV: {e}", flush=True)
        sys.exit(1)
        
    print(f"Total records: {len(df)}")
    
    # Check if we are in testing mode (e.g. limit to 50 for testing)
    if is_test_mode:
        print("RUNNING IN TEST MODE: Only processing first 50 records", flush=True)
        df = df.head(50).copy()
    elif sample:
        print(f"RUNNING IN SAMPLE MODE: Only processing first {sample} records", flush=True)
        df = df.head(sample).copy()
    
    # 1b. Normalize column names across different CSV schemas
    col_aliases = {
        'user_id': ['uname', 'username', 'speaker_id'],
        'audio':   ['audiourl', 'audio_url', 'url'],
        'tid':     ['id', 'audio_id', 'index'],
    }
    for canonical, aliases in col_aliases.items():
        if canonical not in df.columns:
            for alias in aliases:
                if alias in df.columns:
                    df[canonical] = df[alias]
                    print(f"Column mapping: '{alias}' → '{canonical}'", flush=True)
                    break
    if 'user_id' not in df.columns:
        df['user_id'] = 'unknown'
        print("Warning: No speaker/user column found — grouping all as 'unknown'", flush=True)

    # 2. Extract Syllables & Durations
    print("\nCounting syllables using PyThaiNLP...", flush=True)
    df['syllable_count'] = df['text'].apply(count_syllables)
    
    df = process_durations_multithreaded(df, max_workers=max_workers)
    
    # 3. Calculate Speech Rate & IQR Outliers
    print("\nCalculating Speech Rate and IQR Outliers per Speaker...", flush=True)
    # Filter out rows missing duration or having 0 duration
    df['speech_rate'] = np.where(
        (df['audio_duration'].notna()) & (df['audio_duration'] > 0),
        df['syllable_count'] / df['audio_duration'],
        np.nan
    )
    
    df['outlier_type'] = 'Normal'
    df['final_status'] = 'PASS'
    df['ai_score'] = np.nan

    # Auto-fail rows with no reference text — nothing to compare against
    no_text_idx = df[df['text'].isna() | (df['text'].astype(str).str.strip() == '')].index
    if len(no_text_idx) > 0:
        df.loc[no_text_idx, 'outlier_type'] = 'No Script'
        df.loc[no_text_idx, 'final_status'] = 'FAIL'
        print(f"Auto-failed {len(no_text_idx)} rows with missing reference text (No Script)", flush=True)

    # Compute global IQR bounds as fallback for speakers with too few samples
    all_rates = df['speech_rate'].dropna()
    if len(all_rates) >= 4:
        _gq1 = all_rates.quantile(0.25)
        _gq3 = all_rates.quantile(0.75)
        _giqr = _gq3 - _gq1
        global_too_fast = _gq3 + IQR_MULTIPLIER * _giqr
        global_too_slow = max(0.0, _gq1 - IQR_MULTIPLIER * _giqr)
    else:
        global_too_fast = 8.0
        global_too_slow = 0.5
    print(f"Global IQR bounds (\u00d7{IQR_MULTIPLIER}): too_slow < {global_too_slow:.2f} | too_fast > {global_too_fast:.2f} syll/sec", flush=True)

    # Group by user_id
    speaker_groups = df.groupby('user_id')

    outlier_count = 0

    # Auto-fail recordings that are suspiciously short (likely incomplete captures)
    short_dur_idx = df[
        df['audio_duration'].notna() & (df['audio_duration'] < MIN_AUDIO_DURATION)
    ].index
    if len(short_dur_idx) > 0:
        df.loc[short_dur_idx, 'outlier_type'] = 'Too Short'
        df.loc[short_dur_idx, 'final_status'] = 'FAIL'
        outlier_count += len(short_dur_idx)
        print(f"Auto-failed {len(short_dur_idx)} recordings under {MIN_AUDIO_DURATION}s (Too Short)", flush=True)

    for speaker, group in speaker_groups:
        # Skip rows already FAIL (e.g. Too Short) so IQR only sees PASS candidates
        valid_group = group[group['final_status'] == 'PASS']
        if valid_group.empty:
            continue
        rates = valid_group['speech_rate'].dropna()
        if len(rates) < 4:
            # Too few samples for per-speaker IQR — use global bounds as fallback
            fast_idx = valid_group[valid_group['speech_rate'] > global_too_fast].index
            slow_idx = valid_group[valid_group['speech_rate'] < global_too_slow].index
            df.loc[fast_idx, 'outlier_type'] = 'Too Fast'
            df.loc[slow_idx, 'outlier_type'] = 'Too Slow'
            df.loc[fast_idx, 'final_status'] = 'FAIL'
            df.loc[slow_idx, 'final_status'] = 'FAIL'
            outlier_count += len(fast_idx) + len(slow_idx)
            continue

        q1 = rates.quantile(0.25)
        q3 = rates.quantile(0.75)
        iqr = q3 - q1

        too_fast_threshold = q3 + IQR_MULTIPLIER * iqr
        too_slow_threshold = q1 - IQR_MULTIPLIER * iqr

        # Apply bounds
        fast_idx = valid_group[valid_group['speech_rate'] > too_fast_threshold].index
        slow_idx = valid_group[valid_group['speech_rate'] < too_slow_threshold].index

        df.loc[fast_idx, 'outlier_type'] = 'Too Fast'
        df.loc[slow_idx, 'outlier_type'] = 'Too Slow'
        df.loc[fast_idx, 'final_status'] = 'FAIL'
        df.loc[slow_idx, 'final_status'] = 'FAIL'

        outlier_count += (len(fast_idx) + len(slow_idx))
        
    print(f"Warning: Found {outlier_count} speed outliers across {len(speaker_groups)} speakers.", flush=True)
    
    # 4. AI-Assisted Transcription for Outliers
    if outlier_count > 0:
        print(f"\nLoading Whisper model once for {outlier_count} outliers...", flush=True)
        whisper_model = load_whisper_model()

        print("Running Whisper AI fallback on outliers...", flush=True)
        # Skip 'No Script' only — Too Short goes through Whisper normally
        outliers = df[
            (df['final_status'] == 'FAIL') &
            (~df['outlier_type'].isin(['No Script']))
        ]
        os.makedirs("temp_audio", exist_ok=True)

        for idx, row in tqdm(outliers.iterrows(), total=len(outliers), desc="AI Recovery"):
            url = row.get('audio', row.get('audiourl', ''))
            ref_text = str(row.get('text', '')) if pd.notna(row.get('text')) else ""
            audio_id = row.get('tid', row.get('id', idx))

            if not url or pd.isna(url):
                continue

            local_audio_path = os.path.join("temp_audio", f"temp_{audio_id}.wav")
            local_trimmed_path = os.path.join("temp_audio", f"temp_{audio_id}_trimmed.wav")
            local_enhanced_path = os.path.join("temp_audio", f"temp_{audio_id}_enhanced.wav")

            if not download_audio(url, local_audio_path):
                print(f"Error: Failed to download audio for {audio_id}", flush=True)
                continue

            # Step 1: Trim silence + loudnorm + resample to 16 kHz mono
            trim_ok = trim_audio_silence(local_audio_path, local_trimmed_path)
            trim_out = local_trimmed_path if trim_ok else local_audio_path

            # Step 2: Neural denoising (DeepFilterNet → noisereduce → skip gracefully)
            if ENHANCE_AUDIO:
                enhance_ok = denoise_audio(trim_out, local_enhanced_path)
                whisper_input = local_enhanced_path if enhance_ok else trim_out
            else:
                whisper_input = trim_out

            try:
                comparison = transcribe_with_model(whisper_model, whisper_input, ref_text)
                minilm_score = comparison['minilm_score']
                df.loc[idx, 'ai_score'] = minilm_score
                df.loc[idx, 'whisper_text'] = comparison.get('transcribed_text', '')
                df.loc[idx, 'avg_word_prob'] = comparison.get('avg_word_prob', None)
                df.loc[idx, 'low_confidence'] = comparison.get('low_confidence', None)

                if minilm_score >= AI_CONFIDENT_THRESHOLD:
                    df.loc[idx, 'final_status'] = 'PASS (AI Recovered)'
                    print(f"Pass: Audio {audio_id} recovered! (Score: {minilm_score:.1f}%)", flush=True)
                else:
                    df.loc[idx, 'final_status'] = 'FAIL'
                    print(f"Fail: Audio {audio_id} failed AI check (Score: {minilm_score:.1f}%)", flush=True)
            except Exception as e:
                print(f"Warning: Error during AI check for {audio_id}: {e}", flush=True)
            finally:
                for _p in [local_audio_path, local_trimmed_path, local_enhanced_path]:
                    if os.path.exists(_p):
                        os.remove(_p)
                
    # 5. Summary
    total = len(df)
    n_pass     = (df['final_status'] == 'PASS').sum()
    n_recovered= (df['final_status'] == 'PASS (AI Recovered)').sum()
    n_fail     = (df['final_status'] == 'FAIL').sum()
    breakdown  = df[df['final_status'] == 'FAIL']['outlier_type'].value_counts()
    print("\n" + "="*70, flush=True)
    print("QC SUMMARY", flush=True)
    print("="*70, flush=True)
    print(f"  Total records        : {total}", flush=True)
    print(f"  PASS                 : {n_pass}  ({n_pass/total*100:.1f}%)", flush=True)
    print(f"  PASS (recovered)     : {n_recovered}  ({n_recovered/total*100:.1f}%)", flush=True)
    print(f"  FAIL                 : {n_fail}  ({n_fail/total*100:.1f}%)", flush=True)
    if not breakdown.empty:
        print("  FAIL breakdown:", flush=True)
        for reason, count in breakdown.items():
            print(f"    {reason:<20}: {count}", flush=True)
    print("="*70, flush=True)

    # 6. Output Results
    print("\nSaving results...", flush=True)
    output_csv = input_csv.replace(".csv", "_qc_results.csv")
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    output_html = input_csv.replace(".csv", "_qc_report.html")
    generate_pipeline_html_report(df, output_html)
    
    print("="*70)
    print("Pipeline Completed Successfully!")
    print("="*70, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IQR-based Audio QC Pipeline")
    parser.add_argument("--input", type=str, default="data/Email1.csv", help="Path to input CSV data")
    parser.add_argument("--workers", type=int, default=20, help="Number of multithreading workers for duration fetching")
    parser.add_argument("--test", action="store_true", help="Run on a small subset of 50 rows for testing")
    
    parser.add_argument("--sample", type=int, default=None, help="Limit to first N rows (e.g. 1000 for sampling)")
    
    args = parser.parse_args()
    
    # Note: If testing mode is enabled, we could modify run_pipeline to accept it
    run_pipeline(args.input, args.workers, args.test, args.sample)
