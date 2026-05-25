import sys
import os

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

import numpy as np
import stable_whisper
from faster_whisper import WhisperModel
import torch
import json
import argparse
import difflib

from text_normalizer import convert_nan_to_none, normalize_script, normalize_whisper, normalize_phonetic
from scorer import semantic_score, compare_with_script
from audio_processor import download_audio, get_row_data_from_excel, get_script_from_excel, trim_audio_silence, denoise_audio
from report_generator import generate_single_html_report, generate_batch_html_report

# ==================== Settings ====================
INPUT_FILE_PATH = "test.mp4"
OUTPUT_JSON_PATH = os.path.join(os.path.dirname(__file__), "result.json")
REFERENCE_SCRIPT_PATH = "script.txt"
EXCEL_DATA_PATH = None
AUDIO_INDEX = None

OUTPUT_AUDIO_FILE = False
OUTPUT_AUDIO_PATH = os.path.join(os.path.dirname(__file__), "test.mp3")

LANGUAGE = "th"

# Set model type:
# - "large-v3" / "large" / "medium" / "small" / "base" / "tiny"
MODEL_TYPE = "large-v3"

USE_AUTO_TIMESTAMPS = False
USE_262B_PARAMETERS = False
# ==================================================


def _get_device():
    """Return (device, compute_type, label) for faster-whisper.
    CUDA → float16 | CPU → int8 (4-6x faster than PyTorch fp32)
    """
    if torch.cuda.is_available():
        return "cuda", "float16", "cuda (float16)"
    return "cpu", "int8", "cpu (int8)"


def load_whisper_model():
    """Load and return faster-whisper model. Call once and reuse across all files."""
    device, compute_type, device_label = _get_device()
    print(f"Loading {MODEL_TYPE} model on {device_label}...", flush=True)
    model = WhisperModel(MODEL_TYPE, device=device, compute_type=compute_type)
    print("Whisper model loaded.", flush=True)
    return model


def _extract_hotwords(ref_text: str) -> list:
    """
    Normalize the reference script and return its tokens as Whisper hotwords.

    This biases beam search toward the expected email vocabulary (brand names,
    domain tokens) without feeding the full sentence as initial_prompt — which
    would cause Whisper to hallucinate the script verbatim and inflate scores.

    Only tokens longer than 1 char are kept (filters out noise like 'a', 'i').
    """
    tokens = [t for t in normalize_phonetic(ref_text).split() if len(t) > 1]
    # faster-whisper expects hotwords as a single whitespace-separated string, not a list
    return " ".join(tokens) if tokens else None


def transcribe_with_model(model, audio_path: str, ref_text: str) -> dict:
    """
    Transcribe a single audio file with a pre-loaded Whisper model.
    No file I/O side-effects — returns comparison dict directly.
    Call load_whisper_model() once and pass the model here for each file.

    Uses only a generic email-context initial_prompt (no script injection,
    no hotwords). Injecting the actual script caused Whisper to hallucinate
    the reference verbatim regardless of what was spoken, inflating scores
    to 100% for mismatched recordings.  Discrimination is handled entirely
    by the normalize+hybrid-fuzzy scorer instead.
    """
    initial_prompt = (
        "ลูกค้าพูดที่อยู่อีเมลของตัวเอง เช่น มาร์เก็ตติ้ง ดอท บรานช์ แอท เคแบงก์ ดอท คอม "
        "หรือ ซัพพอร์ต แอท ยาฮู ดอท คอม หรือ เฮช อาร์ แอท ทีโอที ดอท ซีโอ ดอท ทีเอช"
    )

    segments, info = model.transcribe(
        audio_path,
        language=LANGUAGE,
        beam_size=5,
        temperature=0,
        condition_on_previous_text=False,
        initial_prompt=initial_prompt,
        # Cut silence before decoding → fewer hallucinations on quiet recordings
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
        # Enable per-word confidence so we can flag low-confidence transcriptions
        word_timestamps=True,
    )
    text_parts = []
    all_word_probs = []
    with __import__('tqdm').tqdm(
        total=round(info.duration, 2), unit="sec",
        desc="Transcribe", file=sys.stdout
    ) as pbar:
        for seg in segments:
            text_parts.append(seg.text)
            pbar.update(seg.end - pbar.n)
            if seg.words:
                for w in seg.words:
                    all_word_probs.append(w.probability)
    full_text = "".join(text_parts)

    # Average word-level confidence — low value suggests hallucination
    avg_word_prob = round(sum(all_word_probs) / len(all_word_probs), 3) if all_word_probs else 1.0

    if not ref_text or not ref_text.strip():
        return {
            "minilm_score": 0.0,
            "diff_score": 0.0,
            "diffs": [],
            "norm_s": "",
            "norm_a": full_text,
            "transcribed_text": full_text,
            "reference_text": "",
            "avg_word_prob": avg_word_prob,
            "low_confidence": avg_word_prob < 0.5,
        }

    norm_s = normalize_script(ref_text)
    norm_a = normalize_whisper(full_text)
    ref_tokens = norm_s.split()
    trans_tokens = norm_a.split()

    matcher = difflib.SequenceMatcher(None, ref_tokens, trans_tokens)
    diffs = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            diffs.append({
                "type": "replace",
                "ref": " ".join(ref_tokens[i1:i2]),
                "asr": " ".join(trans_tokens[j1:j2]),
            })
        elif tag == "delete":
            diffs.append({"type": "delete", "ref": " ".join(ref_tokens[i1:i2]), "asr": ""})
        elif tag == "insert":
            diffs.append({"type": "insert", "ref": "", "asr": " ".join(trans_tokens[j1:j2])})

    minilm_score, norm_s_out, norm_a_out = semantic_score(full_text, ref_text)

    return {
        "minilm_score": minilm_score,
        "diff_score": matcher.ratio() * 100,
        "diffs": diffs,
        "norm_s": norm_s_out,
        "norm_a": norm_a_out,
        "transcribed_text": full_text,
        "reference_text": ref_text,
        "avg_word_prob": avg_word_prob,
        "low_confidence": avg_word_prob < 0.5,
    }


def stable_whisper_transcribe(input_file_path: str, output_json_path: str, script_path: str = None, excel_path: str = None):
    """
    Run Stable Whisper to transcribe audio from video and output result to JSON file.

    Args:
        input_file_path: Path to input video file
        output_json_path: Path to output JSON file
        script_path: Path to reference script for comparison
        excel_path: Path to Excel data file
    """
    use_262b_parameters = USE_262B_PARAMETERS and MODEL_TYPE == "large-v3"

    device, compute_type, device_label = _get_device()
    print(f"Loading {MODEL_TYPE} model with {device_label} device...")
    print(f"262B parameters enabled: {use_262b_parameters}")

    model = stable_whisper.load_model(
        name=MODEL_TYPE,
        device=device
    )

    print("Loading model completed.")
    print(f"Starting transcription of {input_file_path}...")
    print(f"Timestamp precision: {'auto_timestamps=False' if not USE_AUTO_TIMESTAMPS else 'auto_timestamps=True'}")
    print("="*70)

    result = model.transcribe(
        input_file_path,
        language=LANGUAGE
    )

    print("="*70)
    print("Transcription completed!")
    print(f"Output saved to: {output_json_path}")

    result_dict = convert_nan_to_none(result if isinstance(result, dict) else result.to_dict())

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)

    print(f"\nTranscription Summary:")
    print(f"Total segments: {len(result_dict.get('segments', []))}")
    print(f"Total duration: {result_dict.get('duration', 0):.2f} seconds")

    full_text = result_dict.get('text', '')

    ref_text = None
    if excel_path and os.path.exists(excel_path):
        ref_text = get_script_from_excel(excel_path, input_file_path)
    elif script_path and os.path.exists(script_path):
        with open(script_path, 'r', encoding='utf-8') as f:
            ref_text = f.read()

    comparison_results = None
    if ref_text:
        temp_script_path = "temp_script_for_comparison.txt"
        with open(temp_script_path, 'w', encoding='utf-8') as f:
            f.write(ref_text)
        comparison_results = compare_with_script(full_text, temp_script_path)
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)
    elif script_path:
        comparison_results = compare_with_script(full_text, script_path)

    html_output_path = output_json_path.replace(".json", "_report.html")
    generate_single_html_report(input_file_path, result_dict, comparison_results, html_output_path)

    if OUTPUT_AUDIO_FILE:
        try:
            print(f"\nExtracting audio to {OUTPUT_AUDIO_PATH}...")
            audio_path = model.extract_audio(
                input_file_path=input_file_path,
                output_file_path=OUTPUT_AUDIO_PATH
            )
            print(f"Audio saved to: {audio_path}")
        except Exception as e:
            print(f"Error extracting audio: {e}")

    print("="*70)

    return {
        "result": result_dict,
        "comparison": comparison_results,
        "audio_path": input_file_path
    }


# ==================== Main Execution ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stable Whisper for audio transcription and script comparison")
    parser.add_argument("--video", type=str, default=INPUT_FILE_PATH, help="Path to input video/audio file")
    parser.add_argument("--output", type=str, default=OUTPUT_JSON_PATH, help="Path to output JSON file")
    parser.add_argument("--script", type=str, default=REFERENCE_SCRIPT_PATH, help="Path to reference script text file for comparison")
    parser.add_argument("--excel", type=str, default=EXCEL_DATA_PATH, help="Path to Excel data file (e.g. ../data/Email1_Cleaned.xlsx)")
    parser.add_argument("--index", type=int, default=AUDIO_INDEX, help="Audio index to process from Excel (pulls URL and Script)")
    parser.add_argument("--language", type=str, default=LANGUAGE, help="Language code (default: th)")
    parser.add_argument("--model", type=str, default=MODEL_TYPE, help="Model type (default: large-v3)")
    parser.add_argument("--no-timestamps", action="store_true", help="Use auto_timestamps=False (faster)")

    args = parser.parse_args()

    INPUT_FILE_PATH = args.video
    OUTPUT_JSON_PATH = args.output
    REFERENCE_SCRIPT_PATH = args.script
    EXCEL_DATA_PATH = args.excel
    AUDIO_INDEX = args.index
    LANGUAGE = args.language
    MODEL_TYPE = args.model
    USE_AUTO_TIMESTAMPS = not args.no_timestamps

    # --- Case 1: Pulling from Excel Index ---
    if EXCEL_DATA_PATH and AUDIO_INDEX is not None:
        url, ref_text = get_row_data_from_excel(EXCEL_DATA_PATH, AUDIO_INDEX)
        if not url:
            print("❌ ไม่สามารถดึง URL จาก Excel ได้")
            sys.exit(1)

        audio_filename = f"audio_{AUDIO_INDEX}.wav"
        audio_local_path = os.path.join("temp_audio", audio_filename)
        if not download_audio(url, audio_local_path):
            print("❌ ดาวน์โหลดไฟล์เสียงไม่สำเร็จ")
            sys.exit(1)

        temp_script_path = f"temp_script_{AUDIO_INDEX}.txt"
        with open(temp_script_path, 'w', encoding='utf-8') as f:
            f.write(ref_text)

        stable_whisper_transcribe(audio_local_path, OUTPUT_JSON_PATH, temp_script_path, EXCEL_DATA_PATH)

        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)

    # --- Case 2: Batch Processing from Excel ---
    elif EXCEL_DATA_PATH and AUDIO_INDEX is None:
        import pandas as pd
        from tqdm import tqdm

        try:
            df = pd.read_excel(EXCEL_DATA_PATH)
        except Exception as e:
            print(f"❌ ไม่สามารถเปิดไฟล์ Excel ได้: {e}")
            sys.exit(1)

        print(f"🚀 เริ่มการตรวจสอบแบบ Batch จำนวน {len(df)} รายการ...")
        batch_results = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="QC Batch"):
            audio_idx = row.get('audio_index', row.get('tid', row.get('id', idx)))
            url = row.get('audio_url', row.get('audiourl'))
            ref_text = str(row.get('text', ''))

            if not url or pd.isna(url):
                print(f"⚠️ แถวที่ {idx} ไม่มี URL ข้ามการทำงาน...")
                continue

            audio_filename = f"audio_{audio_idx}.wav"
            audio_local_path = os.path.join("temp_audio", audio_filename)
            if not download_audio(url, audio_local_path):
                continue

            temp_script_path = f"temp_script_batch_{idx}.txt"
            with open(temp_script_path, 'w', encoding='utf-8') as f:
                f.write(ref_text)

            try:
                res = stable_whisper_transcribe(audio_local_path, f"temp_result_{idx}.json", temp_script_path, EXCEL_DATA_PATH)
                batch_results.append(res)
            except Exception as e:
                print(f"❌ Error processing row {idx}: {e}")
            finally:
                if os.path.exists(temp_script_path):
                    os.remove(temp_script_path)
                if os.path.exists(f"temp_result_{idx}.json"):
                    os.remove(f"temp_result_{idx}.json")

        batch_report_path = EXCEL_DATA_PATH.replace(".xlsx", "_batch_report.html")
        generate_batch_html_report(batch_results, batch_report_path)
        print(f"\n✅ การตรวจสอบ Batch เสร็จสิ้น! ดูรายงานได้ที่: {batch_report_path}")

    # --- Case 3: Standard Single File Processing ---
    else:
        stable_whisper_transcribe(INPUT_FILE_PATH, OUTPUT_JSON_PATH, REFERENCE_SCRIPT_PATH, EXCEL_DATA_PATH)
