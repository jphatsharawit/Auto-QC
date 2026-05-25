import os
import sys
import re
import subprocess
import shutil
import requests
import pandas as pd

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


def download_audio(url: str, dest: str) -> bool:
    """Download audio from a URL."""
    try:
        if not os.path.exists("temp_audio"):
            os.makedirs("temp_audio")
        print(f"📥 Downloading audio from: {url}...")
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            with open(dest, 'wb') as f:
                f.write(resp.content)
            print(f"✅ Download completed: {dest}")
            return True
        else:
            print(f"❌ Download failed (Status {resp.status_code})")
            return False
    except Exception as e:
        print(f"❌ Error downloading audio: {e}")
        return False


def get_row_data_from_excel(excel_path: str, audio_idx: int):
    """Extract audio_url and text from Excel based on the audio index."""
    try:
        df = pd.read_excel(excel_path)

        if 'audio_index' in df.columns:
            row = df[df['audio_index'] == audio_idx]
        elif 'tid' in df.columns:
            row = df[df['tid'] == audio_idx]
        elif 'id' in df.columns:
            row = df[df['id'] == audio_idx]
        else:
            print("⚠️ ไม่พบคอลัมน์ 'audio_index', 'tid' หรือ 'id' ในไฟล์ Excel")
            return None, None

        if not row.empty:
            text = row.iloc[0]['text']
            url = row.iloc[0].get('audio_url', row.iloc[0].get('audiourl'))
            print(f"📖 ดึงข้อมูลจาก Excel (Index {audio_idx}):")
            print(f"   - Script: '{text}'")
            print(f"   - URL: {url}")
            return str(url), str(text)
        else:
            print(f"⚠️ ไม่พบข้อมูล Index {audio_idx} ในไฟล์ Excel")
            return None, None

    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดในการอ่าน Excel: {e}")
        return None, None


def get_script_from_excel(excel_path: str, audio_file_path: str) -> str:
    """Extract reference script text from Excel based on the audio file name (Legacy lookup)."""
    try:
        filename = os.path.basename(audio_file_path)
        match = re.search(r'(\d+)', filename)

        if not match:
            return None

        audio_idx = int(match.group(1))
        _, text = get_row_data_from_excel(excel_path, audio_idx)
        return text
    except Exception:
        return None


def trim_audio_silence(
    input_path: str,
    output_path: str,
    silence_thresh_db: float = -40.0,
    min_silence_len: float = 0.3,
    normalize_volume: bool = True,
) -> bool:
    """
    Prepare audio for Whisper in one ffmpeg pass:
      1. Remove leading/trailing silence (silenceremove)
      2. EBU R128 loudness normalization — optional (loudnorm)
      3. Resample to 16 kHz mono (Whisper's native format)

    Args:
        input_path:         Path to input audio file
        output_path:        Destination for processed audio
        silence_thresh_db:  Silence cutoff in dBFS (default -40 dB)
        min_silence_len:    Minimum silence duration to remove in seconds
        normalize_volume:   Apply loudnorm EBU R128 (improves quiet recordings)

    Returns:
        True on success, False if fallback copy was used.
    """
    try:
        # Double-pass silenceremove: forward (trim head) -> areverse -> forward (trim tail)
        silence_filter = (
            f"silenceremove=start_periods=1:start_silence={min_silence_len}"
            f":start_threshold={silence_thresh_db}dB"
            f",areverse"
            f",silenceremove=start_periods=1:start_silence={min_silence_len}"
            f":start_threshold={silence_thresh_db}dB"
            f",areverse"
        )
        # highpass=f=80: cut low-frequency rumble common on phone/crowd recordings
        # afftdn=nf=-25: neural FFT denoiser built into ffmpeg (no extra install needed)
        enhance_filters = "highpass=f=80,afftdn=nf=-25"
        filter_chain = (
            f"{silence_filter},{enhance_filters},loudnorm=I=-16:TP=-1.5:LRA=11"
            if normalize_volume
            else f"{silence_filter},{enhance_filters}"
        )
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-af", filter_chain,
            "-ar", "16000",  # Whisper's native sample rate
            "-ac", "1",      # Mono
            output_path,
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60
        )
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True
    except Exception:
        pass
    # Fallback: copy original so caller always has a file
    try:
        shutil.copy2(input_path, output_path)
    except Exception:
        pass
    return False


def denoise_audio(input_path: str, output_path: str) -> bool:
    """
    Neural speech enhancement — DeepFilterNet (primary) with noisereduce fallback.

    DeepFilterNet is a real-time neural model trained on noisy real-world speech.
    It handles non-stationary noise (crowd, wind, phone mic hiss) far better than
    spectral gating, which is why it's the right choice for crowdsourced recordings.

    Install:
        pip install deepfilternet          # primary  (neural, CPU/GPU)
        pip install noisereduce soundfile  # fallback (spectral gating)

    Returns:
        True  — audio successfully enhanced and written to output_path
        False — both methods unavailable; caller should use the untouched file
    """
    # --- Primary: DeepFilterNet (neural, state-of-the-art) ---
    try:
        from df.enhance import enhance, init_df, load_audio, save_audio

        # init_df() caches the model globally after first load — fast on repeat calls
        model, df_state, _ = init_df()
        audio, _ = load_audio(input_path, sr=df_state.sr())
        enhanced = enhance(model, df_state, audio)
        save_audio(output_path, enhanced, df_state.sr())
        return True
    except ImportError:
        pass  # deepfilternet not installed → try fallback
    except Exception:
        pass

    # --- Fallback: noisereduce (spectral gating, CPU-only) ---
    try:
        import noisereduce as nr
        import soundfile as sf

        data, rate = sf.read(input_path)
        # Use first 0.5 s as the noise profile (assumes near-silent recording head)
        noise_clip = data[: int(rate * 0.5)] if len(data) > rate * 0.5 else data
        reduced = nr.reduce_noise(y=data, sr=rate, y_noise=noise_clip, stationary=False)
        sf.write(output_path, reduced, rate)
        return True
    except ImportError:
        return False
    except Exception:
        return False
