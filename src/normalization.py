"""
normalization.py
----------------
High-performance, reusable normalization routines for business names and addresses.
Supports multi-language content (English, Indian transliterations, French).
Optimized for rapid streaming over tens of millions of records.
"""

import string
import unicodedata

PUNCT_NAME_TABLE = str.maketrans(string.punctuation, " " * len(string.punctuation))
PUNCT_ADDR_TABLE = str.maketrans(
    "".join(c for c in string.punctuation if c not in "-,/#"),
    " " * len([c for c in string.punctuation if c not in "-,/#"])
)

# Common legal suffix word replacements
LEGAL_WORD_MAP = {
    "incorporated": "inc",
    "corporation": "corp",
    "company": "co",
    "limited": "ltd",
    "private": "pvt",
}


def strip_accents(text: str) -> str:
    """Strips accents/diacritics from text using fast ASCII short-circuit."""
    if not text or text.isascii():
        return text
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_business_name(name: str) -> str:
    """
    Normalizes a business name:
    - Lowercase
    - Fast Unicode accent stripping
    - Normalizes '&' to 'and'
    - Punctuation removal
    - Standardizes legal words
    - Whitespace collapse
    """
    if not name:
        return ""
    
    s = name.lower()
    if not s.isascii():
        s = strip_accents(s)
        
    if "&" in s:
        s = s.replace("&", " and ")
        
    s = s.translate(PUNCT_NAME_TABLE)
    words = s.split()
    
    # Standardize individual legal tokens
    normalized_words = [LEGAL_WORD_MAP.get(w, w) for w in words]
    return " ".join(normalized_words)


def normalize_address(address: str) -> str:
    """
    Normalizes an address:
    - Lowercase
    - Fast Unicode accent stripping
    - Standardizes common abbreviations
    - Cleans punctuation while preserving numbers and structural delimiters
    - Whitespace collapse
    """
    if not address:
        return ""
        
    s = address.lower()
    if not s.isascii():
        s = strip_accents(s)
        
    s = s.translate(PUNCT_ADDR_TABLE)
    words = s.split()
    
    addr_map = {
        "street": "st",
        "road": "rd",
        "avenue": "ave",
        "boulevard": "blvd",
        "drive": "dr",
        "lane": "ln",
        "suite": "ste",
        "apartment": "apt",
        "floor": "fl",
        "building": "bldg",
        "opposite": "opp",
    }
    
    normalized_words = [addr_map.get(w, w) for w in words]
    return " ".join(normalized_words)


def extract_tokens(text: str, min_len: int = 2) -> list[str]:
    """Extracts non-empty tokens of length >= min_len."""
    if not text:
        return []
    return [tok for tok in text.split() if len(tok) >= min_len]


def extract_char_ngrams(text: str, n: int = 3) -> list[str]:
    """Extracts character n-grams from text."""
    clean = "".join(text.split())
    if len(clean) < n:
        return [clean] if clean else []
    return [clean[i:i+n] for i in range(len(clean) - n + 1)]
