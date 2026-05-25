import re
import numpy as np
from pythainlp.tokenize import syllable_tokenize

# Multi-syllable closed-set words (replaced first)
MULTI_SYLLABLE_MAP = [
    # Non-letter symbols only — single letters are handled by SINGLE_SYLLABLE_MAP automatically
    ("ดับเบิ้ลยู", "w"),
    ("ดับเบิลยู", "w"),
    ("ดับบลิว", "w"),
    ("ขีดล่าง", "underscore"),
    ("ไฮเฟน", "dash"),
]

# Single-syllable closed-set words
SINGLE_SYLLABLE_MAP = {
    "ดอท": "dot",
    "จุด": "dot",
    "ดอต": "dot",
    "ด๊อต": "dot",
    "แอท": "at",
    "ขีด": "dash",
    "ศูนย์": "0",
    "หนึ่ง": "1",
    "สอง": "2",
    "สาม": "3",
    "สี่": "4",
    "ห้า": "5",
    "หก": "6",
    "เจ็ด": "7",
    "แปด": "8",
    "เก้า": "9",
    "เอ": "a",
    "บี": "b",
    "ซี": "c",
    "ดี": "d",
    "อี": "e",
    "เอฟ": "f",
    "เอ็ฟ": "f",
    "จี": "g",
    "เอช": "h",
    "เฮช": "h",
    "เจ": "j",
    "เค": "k",
    "แอล": "l",
    "เอ็ล": "l",
    "เอ็ม": "m",
    "เอ็น": "n",
    "โอ": "o",
    "พี": "p",
    "คิว": "q",
    "อาร์": "r",
    "เอส": "s",
    "ที": "t",
    "ยู": "u",
    "วี": "v",
    "ดับเบิ้ล": "w",
    "เอ็กซ์": "x",
    "วาย": "y",
    "ไอ": "i",
    "ไว": "y",
    "แซด": "z",
}

_MULTI_SYLLABLE_SORTED = sorted(MULTI_SYLLABLE_MAP, key=lambda x: -len(x[0]))


def convert_nan_to_none(obj):
    """Recursively convert all NaN values to None in the object."""
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: convert_nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_nan_to_none(elem) for elem in obj]
    return obj


def custom_thai_to_latin(text: str) -> str:
    """Romanize a Thai word to standard simple Latin phonetics."""
    if not text:
        return ""
    
    # Swap leading vowels (เแโใไ) with the following consonant/cluster
    t = re.sub(r'([เแโใไ])(?:([กขคตปผพบฟดท])([รลว])|([ก-ฮ]))', 
               lambda m: m.group(2) + m.group(3) + m.group(1) if m.group(2) else m.group(4) + m.group(1), 
               text)
    
    consonants = {
        'ก': 'k', 'ข': 'k', 'ค': 'k', 'ฆ': 'k', 'ง': 'ng',
        'จ': 's', 'ฉ': 's', 'ช': 's', 'ฌ': 's', 'ซ': 's', 'ศ': 's', 'ษ': 's', 'ส': 's',
        'ด': 'd', 'ต': 't', 'ถ': 't', 'ท': 't', 'ธ': 't',
        'ฎ': 'd', 'ฏ': 't', 'ฐ': 't', 'ฑ': 't', 'ฒ': 't',
        'บ': 'b', 'ป': 'p', 'ผ': 'p', 'พ': 'p', 'ภ': 'p', 'ฟ': 'f',
        'ม': 'm', 'ย': 'y', 'ร': 'r', 'ล': 'l', 'ฬ': 'l', 'ว': 'w', 'ห': 'h', 'ฮ': 'h',
        'น': 'n', 'ณ': 'n', 'ญ': 'y', 'ฝ': 'f', 'ฃ': 'k', 'ฅ': 'k'
    }
    vowels = {
        'ะ': 'a', 'า': 'a', 'ั': 'a', 'ิ': 'i', 'ี': 'i', 'ึ': 'u', 'ื': 'u', 'ุ': 'u', 'ู': 'u',
        'เ': 'e', 'แ': 'e', '็': 'e', 'โ': 'o', 'ใ': 'ai', 'ไ': 'ai', 'ำ': 'am',
    }
    strip_marks = {'่', '้', '๊', '๋', '์', 'ํ'}
    
    res = []
    chars = list(t)
    for i, char in enumerate(chars):
        if char in strip_marks:
            continue
        if char == 'อ':
            has_next_vowel = False
            for j in range(i + 1, len(chars)):
                next_c = chars[j]
                if next_c in strip_marks:
                    continue
                if next_c in vowels:
                    has_next_vowel = True
                break
            if has_next_vowel:
                continue
            else:
                res.append('o')
        elif char in consonants:
            res.append(consonants[char])
        elif char in vowels:
            res.append(vowels[char])
        elif char.isalnum() or char.isspace():
            res.append(char)
            
    return "".join(res)


def normalize_phonetic(text: str) -> str:
    """
    Normalize both Thai phonetic spelling and Latin email text into a
    uniform sequence of lowercase words (a-z, 0-9, space only) without brand-specific maps.
    """
    if not text:
        return ""
    t = str(text).lower().strip()
    
    # Step 1: Split "at" from merged email patterns (e.g. "johnatgmail.com" → "john at gmail . com")
    # Generic: any word ending in 'at' followed by a domain (contains a dot)
    t = re.sub(r'\b([a-z0-9]+)at([a-z0-9\-]+(?:\.[a-z0-9\-]+)+)\b', r'\1 at \2', t)

    # Step 2: Pad symbols with spaces
    t = re.sub(r'([.@_+\-])', r' \1 ', t)

    # Step 3: Replace multi-syllable closed-set words
    for thai, latin in _MULTI_SYLLABLE_SORTED:
        t = t.replace(thai, f" {latin} ")

    # Step 4: Tokenize remaining Thai words into syllables and translate them.
    # Group consecutive non-closed-set syllables together and romanize them as single words.
    def _translate_thai_part(m: re.Match) -> str:
        thai_part = m.group(0)
        if not thai_part.strip():
            return " "
        syllables = syllable_tokenize(thai_part)
        translated_tokens = []
        accum = []
        
        def flush_accum():
            if accum:
                word = "".join(accum)
                translated_tokens.append(custom_thai_to_latin(word))
                accum.clear()
                
        for s in syllables:
            s_clean = s.strip()
            if not s_clean:
                continue
            if s_clean in SINGLE_SYLLABLE_MAP:
                flush_accum()
                translated_tokens.append(SINGLE_SYLLABLE_MAP[s_clean])
            else:
                accum.append(s_clean)
        flush_accum()
        return " " + " ".join(translated_tokens) + " "

    t = re.sub(r'[\u0E00-\u0E7F]+', _translate_thai_part, t)

    # Step 5: Symbols to standard words
    t = t.replace('.', ' dot ').replace('@', ' at ')
    t = re.sub(r'[^a-z0-9\s]', '', t)
    t = re.sub(r'\s+', ' ', t).strip()

    # Step 6: Join single letters/digits
    prev = ""
    while prev != t:
        prev = t
        t = re.sub(r'\b([a-z0-9])\s+(?=[a-z0-9]\b)', r'\1', t)
    
    t = re.sub(r'\b([a-z0-9])\s+', r'\1', t)
    t = re.sub(r'\s+([a-z0-9])\b', r'\1', t)

    # Map sa cluster at start of words (e.g. sataim -> staim)
    t = re.sub(r'\bsa([tpklmn])', r's\1', t)

    return t


# Backward-compatible aliases
normalize_script = normalize_phonetic
normalize_whisper = normalize_phonetic
