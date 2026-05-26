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
MIN_AUDIO_DURATION = 0.5   # seconds — shorter recordings auto-FAIL without IQR

# Hard absolute rate bounds — applied before any IQR (physically impossible rates)
HARD_MIN_RATE = 0.5   # syll/sec — below this is always Too Slow regardless of speaker
HARD_MAX_RATE = 12.0  # syll/sec — above this is always Too Fast regardless of email type

# New style-based buckets
NEW_BUCKETS = ['spelling', 'reading_short', 'reading_medium', 'reading_long']

SPELLING_TOKENS = {
    'เอ', 'บี', 'ซี', 'ดี', 'อี', 'เอฟ', 'จี', 'เอช', 'เฮช', 'ไอ', 'เจ', 'เค', 'แอล', 
    'เอ็ม', 'เอ็น', 'โอ', 'พี', 'คิว', 'อาร์', 'เอส', 'ที', 'ยู', 'วี', 'ดับบลิว', 
    'ดับเบิ้ลยู', 'เอ็กซ์', 'วาย', 'แซด', 'จุด', 'ขีด', 'ขีดล่าง', 'อันเดอร์สกอร์', 'ไฮเฟน'
}

def classify_style(text):
    if pd.isna(text):
        return 'reading'
    words = str(text).split()
    if not words:
        return 'reading'
    spelling_words = [w for w in words if w in SPELLING_TOKENS or len(w) == 1]
    ratio = len(spelling_words) / len(words)
    return 'spelling' if ratio > 0.4 else 'reading'

# Log-IQR multiplier for bucket-level global bounds (wider = less aggressive)
BUCKET_IQR_MULTIPLIER = 2.0

# Per-speaker IQR multiplier within same bucket (wider than bucket to reduce false positives)
SPEAKER_IQR_MULTIPLIER = 2.5
MIN_SPEAKER_BUCKET_SAMPLES = 20  # minimum samples in a bucket for per-speaker IQR to apply

# AI Recovery (Whisper + MiniLM)
# Binary scoring:
#   score >= AI_CONFIDENT_THRESHOLD → PASS (AI Recovered)      auto-pass
#   score <  AI_CONFIDENT_THRESHOLD → FAIL
AI_CONFIDENT_THRESHOLD  = 70.0  # score needed for auto-pass (lowered from 80 — borderline phonetic matches are close enough)

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

def get_trimmed_duration_remote(url):
    """Fetch active speech duration directly from S3 URL using ffmpeg silenceremove."""
    if pd.isna(url) or not url:
        return None
    try:
        silence_filter = (
            "silenceremove=start_periods=1:start_silence=0.3:start_threshold=-40dB"
            ",areverse"
            ",silenceremove=start_periods=1:start_silence=0.3:start_threshold=-40dB"
            ",areverse"
        )
        cmd = [
            "ffmpeg", "-y", "-i", str(url),
            "-af", silence_filter,
            "-f", "null", "-"
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)
        if result.returncode == 0:
            import re
            matches = re.findall(r'time=(\d+):(\d+):(\d+\.\d+)', result.stderr)
            if matches:
                h, m, s = matches[-1]
                duration = float(h) * 3600 + float(m) * 60 + float(s)
                return duration
    except Exception:
        pass
    return None

def get_syl_bucket(row):
    """Map script style and syllable count to bucket label."""
    text = row.get('text', '')
    syl = row.get('syllable_count', 0)
    style = classify_style(text)
    if style == 'spelling':
        return 'spelling'
    else:
        if syl <= 8:
            return 'reading_short'
        elif syl <= 15:
            return 'reading_medium'
        else:
            return 'reading_long'


def count_syllables(text):
    """Count syllables in Thai text using PyThaiNLP, ignoring spaces and punctuation."""
    if pd.isna(text) or not str(text).strip():
        return 0
    import re
    # 1. Clean punctuation (replace with space)
    cleaned = re.sub(r'[^\u0e00-\u0e7fA-Za-z0-9\s]', ' ', str(text))
    # 2. Tokenize using PyThaiNLP
    raw_tokens = syllable_tokenize(cleaned)
    # 3. Filter out whitespace and any punctuation/special characters tokens
    filtered_tokens = [t for t in raw_tokens if re.search(r'[\u0e00-\u0e7fA-Za-z0-9]', t)]
    return len(filtered_tokens)

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



def run_pipeline(input_csv, max_workers=20, is_test_mode=False, sample=None, enable_ai_recovery=True, bucket_iqr_multiplier=2.0, speaker_iqr_multiplier=2.5):
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
    df['syl_bucket'] = df.apply(get_syl_bucket, axis=1)

    # Auto-fail rows with no reference text — nothing to compare against
    no_text_idx = df[df['text'].isna() | (df['text'].astype(str).str.strip() == '')].index
    if len(no_text_idx) > 0:
        df.loc[no_text_idx, 'outlier_type'] = 'No Script'
        df.loc[no_text_idx, 'final_status'] = 'FAIL'
        print(f"Auto-failed {len(no_text_idx)} rows with missing reference text (No Script)", flush=True)

    outlier_count = 0

    # Auto-fail rows with missing duration (broken link or corrupted audio)
    missing_dur_idx = df[df['audio_duration'].isna()].index
    if len(missing_dur_idx) > 0:
        df.loc[missing_dur_idx, 'outlier_type'] = 'Duration Error'
        df.loc[missing_dur_idx, 'final_status'] = 'FAIL'
        outlier_count += len(missing_dur_idx)
        print(f"Auto-failed {len(missing_dur_idx)} rows due to missing/corrupted duration (Duration Error)", flush=True)

    # Auto-fail recordings that are suspiciously short (likely incomplete captures)
    short_dur_idx = df[
        df['audio_duration'].notna() & (df['audio_duration'] < MIN_AUDIO_DURATION)
    ].index
    if len(short_dur_idx) > 0:
        df.loc[short_dur_idx, 'outlier_type'] = 'Too Short'
        df.loc[short_dur_idx, 'final_status'] = 'FAIL'
        outlier_count += len(short_dur_idx)
        print(f"Auto-failed {len(short_dur_idx)} recordings under {MIN_AUDIO_DURATION}s (Too Short)", flush=True)

    # --- Step 1: Hard absolute rate bounds (physically impossible values) ---
    # Applied only to PASS records with a valid speech rate
    pass_mask = (df['final_status'] == 'PASS') & df['speech_rate'].notna()
    hard_fast_idx = df[pass_mask & (df['speech_rate'] > HARD_MAX_RATE)].index
    hard_slow_idx = df[pass_mask & (df['speech_rate'] < HARD_MIN_RATE)].index
    df.loc[hard_fast_idx, 'outlier_type'] = 'Too Fast'
    df.loc[hard_slow_idx, 'outlier_type'] = 'Too Slow'
    df.loc[hard_fast_idx.union(hard_slow_idx), 'final_status'] = 'FAIL'
    outlier_count += len(hard_fast_idx) + len(hard_slow_idx)
    print(f"Hard bounds (< {HARD_MIN_RATE} or > {HARD_MAX_RATE} syll/sec): {len(hard_fast_idx)} too fast, {len(hard_slow_idx)} too slow", flush=True)

    # --- Step 2: Global log-IQR per bucket & Speaker-level Shrunken Z-score ---
    # We will compute bounds on the PASS records
    valid_mask = (df['final_status'] == 'PASS') & df['speech_rate'].notna() & (df['speech_rate'] > 0)
    df['log_rate'] = np.log(df['speech_rate'])
    
    # Calculate global IQR bounds for each bucket
    bucket_bounds = {}
    print(f"\nBucket-stratified log-IQR (x{bucket_iqr_multiplier}) bounds:", flush=True)
    for label in NEW_BUCKETS:
        bucket_mask = valid_mask & (df['syl_bucket'] == label)
        rates = df[bucket_mask]['speech_rate']
        if len(rates) >= 10:
            log_rates = np.log(rates)
            q1 = log_rates.quantile(0.25)
            q3 = log_rates.quantile(0.75)
            iqr = q3 - q1
            lower = np.exp(q1 - bucket_iqr_multiplier * iqr)
            upper = np.exp(q3 + bucket_iqr_multiplier * iqr)
        else:
            lower, upper = HARD_MIN_RATE, HARD_MAX_RATE
        bucket_bounds[label] = (lower, upper)
        print(f"  {label:15s} (n={len(rates):5d}): {lower:.2f} - {upper:.2f} syll/sec", flush=True)

    # Calculate global parameters for Z-score
    global_stats = {}
    for label in NEW_BUCKETS:
        bucket_mask = valid_mask & (df['syl_bucket'] == label)
        rates = df[bucket_mask]['log_rate']
        if len(rates) > 2:
            global_stats[label] = {'mean': rates.mean(), 'std': max(rates.std(), 0.05)}
        else:
            global_stats[label] = {'mean': np.log(2.0), 'std': 0.3}

    # Calculate speaker statistics per bucket
    speaker_stats = {}
    for (speaker, label), group in df[valid_mask].groupby(['user_id', 'syl_bucket']):
        log_rates = group['log_rate']
        n = len(log_rates)
        m = log_rates.mean()
        s = log_rates.std() if n > 1 else 0.0
        speaker_stats[(speaker, label)] = {'n': n, 'mean': m, 'std': s}

    # Compute shrunken Z-score for all valid rows
    # Shrinkage priors: K_m=5.0, K_v=5.0
    K_m, K_v = 5.0, 5.0
    df['z_score'] = np.nan
    for idx, row in df[valid_mask].iterrows():
        speaker = row['user_id']
        bucket = row['syl_bucket']
        g_mean = global_stats[bucket]['mean']
        g_std = global_stats[bucket]['std']
        
        spk_info = speaker_stats.get((speaker, bucket), {'n': 0, 'mean': g_mean, 'std': g_std})
        n = spk_info['n']
        m = spk_info['mean']
        s = spk_info['std']
        
        mu_shrunken = (n * m + K_m * g_mean) / (n + K_m)
        if n > 1:
            s_raw_var = (n / (n - 1)) * (s ** 2)
            var_shrunken = ((n - 1) * s_raw_var + K_v * (g_std ** 2)) / ((n - 1) + K_v)
            std_shrunken = np.sqrt(var_shrunken)
        else:
            std_shrunken = g_std
            
        std_shrunken = max(std_shrunken, 0.05)
        z = (row['log_rate'] - mu_shrunken) / std_shrunken
        df.loc[idx, 'z_score'] = z

    # Outlier criteria: Outside Global bounds AND Speaker Z-score > speaker_iqr_multiplier
    df['is_global_outlier'] = False
    for label, (lower, upper) in bucket_bounds.items():
        b_mask = valid_mask & (df['syl_bucket'] == label)
        df.loc[b_mask & ((df['speech_rate'] < lower) | (df['speech_rate'] > upper)), 'is_global_outlier'] = True

    # speaker_iqr_multiplier is repurposed as z_thresh
    z_thresh = speaker_iqr_multiplier
    initial_outliers_idx = df[valid_mask & df['is_global_outlier'] & (df['z_score'].abs() > z_thresh)].index

    for idx in initial_outliers_idx:
        row = df.loc[idx]
        bucket = row['syl_bucket']
        lower, upper = bucket_bounds[bucket]
        if row['speech_rate'] < lower:
            df.loc[idx, 'outlier_type'] = 'Too Slow'
        else:
            df.loc[idx, 'outlier_type'] = 'Too Fast'
        df.loc[idx, 'final_status'] = 'FAIL'

    # Count outliers so far (excluding No Script)
    outlier_count = len(df[(df['final_status'] == 'FAIL') & (~df['outlier_type'].isin(['No Script']))])

    # --- Active Speech Verification (On-Demand Silence Trim) for outliers ---
    speed_outliers_mask = (df['final_status'] == 'FAIL') & df['outlier_type'].isin(['Too Fast', 'Too Slow'])
    speed_outliers = df[speed_outliers_mask]
    if len(speed_outliers) > 0:
        print(f"\n🔍 Verifying {len(speed_outliers)} speed outliers using active speech duration...", flush=True)
        trimmed_durations = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_idx = {
                executor.submit(get_trimmed_duration_remote, row.get('audio', row.get('audiourl', ''))): idx
                for idx, row in speed_outliers.iterrows()
                if row.get('audio', row.get('audiourl', '')) and not pd.isna(row.get('audio', row.get('audiourl', '')))
            }
            for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Trimming Silences"):
                idx = future_to_idx[future]
                try:
                    trimmed_durations[idx] = future.result()
                except Exception:
                    trimmed_durations[idx] = None

        # Apply trimmed duration and re-evaluate
        recovered_count = 0
        for idx, row in speed_outliers.iterrows():
            trimmed_dur = trimmed_durations.get(idx, None)
            if trimmed_dur is not None and trimmed_dur > 0.5:
                new_rate = row['syllable_count'] / trimmed_dur
                bucket = row['syl_bucket']
                lower, upper = bucket_bounds[bucket]
                
                is_ok_global = (lower <= new_rate <= upper)
                
                speaker = row['user_id']
                g_mean = global_stats[bucket]['mean']
                g_std = global_stats[bucket]['std']
                
                spk_info = speaker_stats.get((speaker, bucket), {'n': 0, 'mean': g_mean, 'std': g_std})
                n = spk_info['n']
                m = spk_info['mean']
                s = spk_info['std']
                
                mu_shrunken = (n * m + K_m * g_mean) / (n + K_m)
                if n > 1:
                    s_raw_var = (n / (n - 1)) * (s ** 2)
                    var_shrunken = ((n - 1) * s_raw_var + K_v * (g_std ** 2)) / ((n - 1) + K_v)
                    std_shrunken = np.sqrt(var_shrunken)
                else:
                    std_shrunken = g_std
                std_shrunken = max(std_shrunken, 0.05)
                
                new_log_rate = np.log(new_rate)
                new_z = (new_log_rate - mu_shrunken) / std_shrunken
                
                is_ok_speaker = (abs(new_z) <= z_thresh)
                
                if is_ok_global or is_ok_speaker:
                    df.loc[idx, 'audio_duration'] = trimmed_dur
                    df.loc[idx, 'speech_rate'] = new_rate
                    df.loc[idx, 'outlier_type'] = 'Normal'
                    df.loc[idx, 'final_status'] = 'PASS'
                    recovered_count += 1
                else:
                    # Still fail, but update rate to the trimmed one
                    df.loc[idx, 'audio_duration'] = trimmed_dur
                    df.loc[idx, 'speech_rate'] = new_rate
            elif trimmed_dur is not None and trimmed_dur <= 0.5:
                df.loc[idx, 'outlier_type'] = 'Too Short'
                df.loc[idx, 'final_status'] = 'FAIL'
                
        print(f"✅ Active Speech Verification completed: Recovered {recovered_count} false outliers!", flush=True)
        # Recalculate outlier_count
        outlier_count = len(df[(df['final_status'] == 'FAIL') & (~df['outlier_type'].isin(['No Script']))])


    # 4. AI-Assisted Transcription for Outliers
    if not enable_ai_recovery:
        if outlier_count > 0:
            print(f"\nAI Recovery (Whisper fallback) is disabled. {outlier_count} outliers will remain FAIL.", flush=True)
    elif outlier_count > 0:
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
    
    # New options
    parser.add_argument("--no-ai", action="store_true", help="Bypass AI/Whisper recovery phase")
    parser.add_argument("--bucket-iqr", type=float, default=2.0, help="Global bucket IQR multiplier (default: 2.0)")
    parser.add_argument("--speaker-iqr", type=float, default=2.5, help="Per-speaker Z-score threshold (default: 2.5)")
    
    args = parser.parse_args()
    
    # Note: If testing mode is enabled, we could modify run_pipeline to accept it
    run_pipeline(
        args.input, 
        max_workers=args.workers, 
        is_test_mode=args.test, 
        sample=args.sample,
        enable_ai_recovery=not args.no_ai,
        bucket_iqr_multiplier=args.bucket_iqr,
        speaker_iqr_multiplier=args.speaker_iqr
    )
