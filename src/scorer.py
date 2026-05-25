import os
import re
import difflib
from rapidfuzz.distance import Levenshtein
from text_normalizer import normalize_phonetic


def simplify_phonetic_token(token: str) -> str:
    """Simplify phonetic spelling of a token to align English and Thai-transliterated sounds."""
    if not token:
        return ""
    
    t = token.lower().replace('y', 'i')
    t = re.sub(r'ce$', 's', t)
    t = re.sub(r'ge$', 's', t)
    t = re.sub(r'se$', 's', t)
    t = re.sub(r'(?<=.[bcdfghjklmnpqrstvwxyz])e$', r'', t)  # Silent e (min 3 chars)
    t = re.sub(r'v$', 'p', t)  # final v → p (e.g. gov → gop, live → lip)
    
    t = t.replace("group", "grup").replace("soup", "sup").replace("uou", "u") # you -> uou -> u

    vowels_map = [
        ("oo", "u"), ("ee", "i"), ("ea", "i"), ("ou", "ao"), ("ow", "ao"),
        ("oi", "oi"), ("ai", "a"), ("ay", "a"), ("ae", "a"), ("ui", "ui"),
        ("ei", "i"), ("ey", "a"), ("ie", "i"), ("oa", "o"), ("oe", "a"),
        ("er", "a"), ("ir", "a"), ("ur", "a"), ("or", "a"), ("ar", "a"),
    ]
    
    prev = ""
    while prev != t:
        prev = t
        for src, dst in vowels_map:
            t = t.replace(src, dst)
        
    consonants_map = [
        ("ph", "p"),
        ("th", "t"),
        ("kh", "k"),
        ("ch", "s"),
        ("sh", "s"),
        ("sch", "s"),
        ("ck", "k"),
        ("gh", "g"),
        ("dg", "g"),
        ("wh", "w"),
        ("wr", "r"),
        ("kn", "n"),
        ("gn", "n"),
        ("mb", "m"),
    ]
    for src, dst in consonants_map:
        t = t.replace(src, dst)

    res = []
    chars = list(t)
    for i, c in enumerate(chars):
        if c == 'c':
            if i + 1 < len(chars) and chars[i+1] in ['e', 'i', 'y']:
                res.append('s')
            else:
                res.append('k')
        elif c == 'g':
            if i + 1 < len(chars) and chars[i+1] in ['e', 'i', 'y']:
                res.append('s')
            else:
                res.append('k')
        elif c in ['j', 'z']:
            res.append('s')
        elif c == 'q':
            res.append('k')
        elif c == 'x':
            res.append('k')
            res.append('s')
        elif c == 'v':
            res.append('w')
        elif c == 'r':
            res.append('l')
        elif c == 'y':
            res.append('i')
        elif c in ['e', 'i', 'ai', 'ae', 'ay', 'oe']:
            res.append('a')
        else:
            res.append(c)
            
    t = "".join(res)
    
    collapsed = []
    for c in t:
        if not collapsed or collapsed[-1] != c:
            collapsed.append(c)
    t = "".join(collapsed)
    
    if len(t) > 1:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        last_char = t[-1]
        body = t[:-1]
        
        if last_char not in vowels:
            if len(t) >= 2 and t[-2:] == 'ng':
                pass
            elif len(t) >= 2 and t[-2:] == 'nk':
                t = body[:-1] + 'ng'
            else:
                if last_char in ['d', 't', 's']:
                    last_char = 't'
                elif last_char in ['l', 'r']:
                    last_char = 'n'
                elif last_char in ['b', 'p', 'f', 'v']:
                    last_char = 'p'
                elif last_char in ['g', 'k']:
                    last_char = 'k'
                t = body + last_char
            
    return t


COMMON_SIMPLIFIED_SET = {
    "dot", "kom", "ko", "t", "at", "ak", "a", "net", "adu", "an", "ok", "bas", "anfo", "as", "kl", "no", "kow", "au"
}
POLITE_FILTERS = {
    "krap", "ka", "na", "naka", "krappom", "krab", "nakha"
}


def semantic_score(asr_text: str, script: str) -> tuple:
    """
    Phonetic score based on Levenshtein distance of simplified phonetic tokens.
    Filters out common TLDs and polite particles to compute a core score.
    Returns (score, norm_s, norm_a).
    """
    norm_s = normalize_phonetic(script)
    norm_a = normalize_phonetic(asr_text)
    
    tokens_s = norm_s.split()
    tokens_a = norm_a.split()
    
    sim_tokens_s = [simplify_phonetic_token(t) for t in tokens_s if t]
    sim_tokens_a = [simplify_phonetic_token(t) for t in tokens_a if t]
    
    sim_s = "".join(sim_tokens_s)
    sim_a = "".join(sim_tokens_a)
    
    dist_full = Levenshtein.distance(sim_s, sim_a)
    max_len_full = max(1, len(sim_s), len(sim_a))
    score_full = (1.0 - dist_full / max_len_full) * 100.0
    
    core_tokens_s = [t for t in sim_tokens_s if t not in COMMON_SIMPLIFIED_SET and t not in POLITE_FILTERS]
    core_tokens_a = [t for t in sim_tokens_a if t not in COMMON_SIMPLIFIED_SET and t not in POLITE_FILTERS]
    
    sim_core_s = "".join(core_tokens_s)
    sim_core_a = "".join(core_tokens_a)
    
    if not sim_core_s or not sim_core_a:
        score_core = score_full
    else:
        dist_core = Levenshtein.distance(sim_core_s, sim_core_a)
        max_len_core = max(1, len(sim_core_s), len(sim_core_a))
        score_core = (1.0 - dist_core / max_len_core) * 100.0
        
    final_score = min(score_full, score_core)
    
    return final_score, norm_s, norm_a


def compare_with_script(transcribed_text: str, script_path: str):
    """Compare transcribed text with reference script and print differences."""
    if not script_path or not os.path.exists(script_path):
        print(f"Skipping comparison: Script file not found at '{script_path}'")
        return

    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            reference_text = f.read()
    except Exception as e:
        print(f"Error reading script file: {e}")
        return

    print("\n" + "="*70)
    print("Script Comparison Result:")
    print("="*70)

    ref_clean = normalize_phonetic(reference_text)
    trans_clean = normalize_phonetic(transcribed_text)

    ref_tokens = ref_clean.split()
    trans_tokens = trans_clean.split()

    matcher = difflib.SequenceMatcher(None, ref_tokens, trans_tokens)

    diff_found = False
    diffs = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            ref_words = ' '.join(ref_tokens[i1:i2])
            trans_words = ' '.join(trans_tokens[j1:j2])
            print(f"❌ [พูดผิด/เปลี่ยนคำ] สคริปต์: '{ref_words}' -> เสียงจริง: '{trans_words}'")
            diffs.append({"type": "replace", "ref": ref_words, "asr": trans_words})
            diff_found = True
        elif tag == 'delete':
            ref_words = ' '.join(ref_tokens[i1:i2])
            print(f"➖ [พูดตกหล่น/หายไป] สคริปต์: '{ref_words}' (ไม่ได้พูด)")
            diffs.append({"type": "delete", "ref": ref_words, "asr": ""})
            diff_found = True
        elif tag == 'insert':
            trans_words = ' '.join(trans_tokens[j1:j2])
            print(f"➕ [พูดเกิน/เพิ่มมา] เสียงจริง: '{trans_words}' (ไม่มีในสคริปต์)")
            diffs.append({"type": "insert", "ref": "", "asr": trans_words})
            diff_found = True

    if not diff_found:
        print("✅ ปกติ: คำพูดตรงกับสคริปต์ทั้งหมด (หลัง Normalize)!")

    diff_score = matcher.ratio() * 100
    print(f"\nExact Word Match Accuracy (Normalized): {diff_score:.2f}%")

    fuzzy_score, norm_s, norm_a = semantic_score(transcribed_text, reference_text)
    print("\n" + "-"*70)
    print("🔤 Phonetic Fuzzy Score:")
    print(f"Norm Script  : '{norm_s}'")
    print(f"Norm Whisper : '{norm_a}'")

    status_icon = "✅" if fuzzy_score >= 80 else "❌"
    print(f"Fuzzy Score: {fuzzy_score:.1f}% {status_icon}")
    print("="*70)

    return {
        "diffs": diffs,
        "diff_score": diff_score,
        "minilm_score": fuzzy_score,
        "norm_s": norm_s,
        "norm_a": norm_a,
        "reference_text": reference_text,
        "transcribed_text": transcribed_text
    }
