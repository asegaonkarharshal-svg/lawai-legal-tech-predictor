"""
fir_parser.py
-------------
Phase 1 – Step 1.3: FIR Text Parser  (spaCy + Regex Engine)

Extracts structured case features from raw First Information Report (FIR)
text using a two-layer approach:
    Layer 1 — Regex pattern matching  (fast, deterministic, primary)
    Layer 2 — spaCy NER fallback      (MONEY/CARDINAL entities, secondary)

Each extracted field records which method was used via `extraction_confidence`,
enabling the frontend to surface data-quality indicators.

Public API:
    parse_fir_text(raw_text: str) -> ParsedFIR

Usage:
    from src.fir_parser import parse_fir_text
    result = parse_fir_text("Complainant reported theft of articles valued at INR 23963...")
    features = result.to_feature_dict()   # ready for model inference
"""

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ─── spaCy lazy singleton ─────────────────────────────────────────────────────
_NLP = None


def _get_nlp():
    """
    Lazily loads the spaCy 'en_core_web_sm' model on first call.
    Returns None (gracefully) if the model is not installed, allowing
    the Regex layer to still function independently.
    """
    global _NLP
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load("en_core_web_sm")
            logger.debug("spaCy 'en_core_web_sm' loaded successfully.")
        except OSError:
            logger.warning(
                "spaCy model 'en_core_web_sm' not found — NER fallback disabled. "
                "Install with: python -m spacy download en_core_web_sm"
            )
            _NLP = None
        except ImportError:
            logger.warning("spaCy not installed — NER fallback disabled.")
            _NLP = None
    return _NLP


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ParsedFIR:
    """
    Structured output of parse_fir_text().

    Attributes:
        stolen_value_inr       : Extracted INR amount (None if not found)
        weapon_present         : 1 if weapon keyword detected, else 0
        forced_entry           : 1 if forced-entry / dwelling keyword detected, else 0
        prior_convictions      : Integer count of prior offenses (0 if not found)
        property_recovered_pct : Recovery percentage (None if not found)
        confession_recorded    : 1 if confession keyword detected, else 0
        counsel_type           : "Public Defender" | "Private Counsel" | None
        extraction_confidence  : Per-field dict showing extraction method used:
                                   "regex"          — primary Regex matched
                                   "regex_fallback" — secondary Regex matched
                                   "spacy_ner"      — spaCy NER entity used
                                   "keyword"        — keyword list matched
                                   "default"        — no signal found, default used
                                   "not_found"      — no signal found, value is None
    """
    stolen_value_inr:       Optional[float] = None
    weapon_present:         int             = 0
    forced_entry:           int             = 0
    prior_convictions:      int             = 0
    property_recovered_pct: Optional[float] = None
    confession_recorded:    int             = 0
    counsel_type:           Optional[str]   = None
    extraction_confidence:  dict            = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to plain Python dict."""
        return asdict(self)

    def to_feature_dict(self, defaults: Optional[dict] = None) -> dict:
        """
        Returns a flat dict of the 7 model input features, applying fallback
        defaults for any field that returned None.

        Default fallbacks:
            stolen_value_inr       → 0.0
            property_recovered_pct → 0.0
            counsel_type           → "Public Defender"

        Args:
            defaults : Optional override dict for fallback values.

        Returns:
            dict — ready for StandardScaler + model.predict_proba()
        """
        _defaults = {
            "stolen_value_inr":       0.0,
            "property_recovered_pct": 0.0,
            "counsel_type":           "Public Defender",
        }
        if defaults:
            _defaults.update(defaults)

        return {
            "stolen_value_inr":       self.stolen_value_inr       or _defaults["stolen_value_inr"],
            "counsel_type":           self.counsel_type            or _defaults["counsel_type"],
            "weapon_present":         self.weapon_present,
            "prior_convictions":      self.prior_convictions,
            "property_recovered_pct": self.property_recovered_pct or _defaults["property_recovered_pct"],
            "confession_recorded":    self.confession_recorded,
            "forced_entry":           self.forced_entry,
        }

    def missing_fields(self) -> list[str]:
        """Returns list of field names where extraction returned None or default."""
        missing = []
        if self.stolen_value_inr       is None: missing.append("stolen_value_inr")
        if self.property_recovered_pct is None: missing.append("property_recovered_pct")
        if self.counsel_type           is None: missing.append("counsel_type")
        return missing


# ══════════════════════════════════════════════════════════════════════════════
# REGEX PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

_RE = {
    # ── Monetary value ──────────────────────────────────────────────────────
    # Matches: "INR 23963", "INR 23,963.4", "Rs. 45,000", "Rs 1,50,000", "₹ 46,454"
    "inr_primary": re.compile(
        r"(?:INR|Rs\.?|₹)\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
        re.IGNORECASE,
    ),
    # Matches: "valued at INR 23963", "valued at 23963"
    "inr_valued_at": re.compile(
        r"valued?\s+at\s+(?:INR|Rs\.?|₹)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
        re.IGNORECASE,
    ),
    # Matches: "worth INR 50000", "worth Rs 1,20,000"
    "inr_worth": re.compile(
        r"worth\s+(?:INR|Rs\.?|₹)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
        re.IGNORECASE,
    ),

    # ── Recovery percentage ──────────────────────────────────────────────────
    # Matches: "Recovered property stands at 71.9%", "recovery stands at 50%"
    "recovery_primary": re.compile(
        r"(?:recovered?\s+property|recovery)\s+(?:stands?\s+at\s+)?([0-9]+(?:\.[0-9]+)?)\s*%",
        re.IGNORECASE,
    ),
    # Matches: "property stands at 71.9%"
    "recovery_stands": re.compile(
        r"(?:property\s+)?stands?\s+at\s+([0-9]+(?:\.[0-9]+)?)\s*%",
        re.IGNORECASE,
    ),
    # Matches: "71.9% recovered", "50% of property recovered"
    "recovery_suffix": re.compile(
        r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:of\s+(?:stolen\s+)?property\s+)?recover",
        re.IGNORECASE,
    ),

    # ── Prior convictions ───────────────────────────────────────────────────
    # Matches: "2 past offenses", "1 past offense", "3 prior convictions"
    "prior_primary": re.compile(
        r"(\d+)\s+(?:past|prior|previous)\s+(?:offense[s]?|offence[s]?|conviction[s]?|record[s]?|case[s]?)",
        re.IGNORECASE,
    ),
    # Matches: "Prior records show 2", "shows 3 past"
    "prior_records_show": re.compile(
        r"(?:prior\s+records?\s+show|records?\s+show)\s+(\d+)",
        re.IGNORECASE,
    ),
    # Matches: "convicted 2 times", "convicted twice" (word→int handled separately)
    "prior_convicted": re.compile(
        r"convicted\s+(\d+)\s+time",
        re.IGNORECASE,
    ),
}

# ── Word-to-integer mapping for written numbers ──────────────────────────────
_WORD_NUMS: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "twice": 2, "thrice": 3,
}

# ══════════════════════════════════════════════════════════════════════════════
# KEYWORD SETS
# ══════════════════════════════════════════════════════════════════════════════

_WEAPON_KEYWORDS: frozenset[str] = frozenset({
    "knife", "gun", "pistol", "revolver", "rifle", "firearm",
    "rod", "iron rod", "blade", "dagger", "sword", "axe", "hammer",
    "sharp object", "blunt object", "dangerous weapon", "lethal weapon",
    "armed", "threatened with",
})

# Negation phrases: if found near "weapon", suppress positive detection
_WEAPON_NEGATION_RE = re.compile(
    r"\b(?:without|no|unarmed|not\s+armed|apprehended\s+without)\s+weapon",
    re.IGNORECASE,
)

# Positive weapon phrase (explicit "with weapon" / "with a dangerous weapon")
_WEAPON_POSITIVE_RE = re.compile(
    r"\bwith\s+(?:a\s+)?(?:dangerous\s+)?weapon\b|"
    r"\b(?:knife|gun|pistol|revolver|rifle|firearm|rod|iron\s+rod|blade|"
    r"dagger|sword|axe|hammer|sharp\s+object|blunt\s+object|"
    r"lethal\s+weapon|dangerous\s+weapon)\b",
    re.IGNORECASE,
)

_FORCED_ENTRY_KEYWORDS: frozenset[str] = frozenset({
    "forced entry", "forced open", "forced his way",
    "broke open", "break open", "breaking open", "broken open",
    "broke in", "broken into", "broke into",
    "entrance lock", "door lock", "window lock",
    "smashed the lock", "pried open", "picked the lock",
    "trespassed", "unlawfully entered",
    "dwelling house", "residential premises",
})

_CONFESSION_KEYWORDS: frozenset[str] = frozenset({
    "confessed", "confession", "admitted guilt", "admitted to the crime",
    "pleaded guilty", "voluntary statement", "accepted responsibility",
    "own statement", "self-incriminating",
})

_PUBLIC_DEFENDER_KEYWORDS: frozenset[str] = frozenset({
    "public defender", "legal aid", "state-appointed", "state appointed",
    "duty counsel", "appointed counsel", "legal services authority",
    "free legal aid",
})

_PRIVATE_COUNSEL_KEYWORDS: frozenset[str] = frozenset({
    "private counsel", "private advocate", "private lawyer",
    "retained counsel", "hired advocate", "own counsel",
    "engaged an advocate", "his counsel",
})


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _clean_amount(raw: str) -> float:
    """Strip commas and convert to float."""
    return float(raw.replace(",", ""))


def _extract_stolen_value(
    text: str, doc
) -> tuple[Optional[float], str]:
    """
    Attempts to extract stolen value in INR using Regex (primary) then
    spaCy MONEY entities (fallback).

    Returns:
        (value_inr, method)  where method ∈ {'regex','regex_fallback','spacy_ner','not_found'}
    """
    # Primary regex: "INR 23963.4"
    m = _RE["inr_primary"].search(text)
    if m:
        return _clean_amount(m.group(1)), "regex"

    # Fallback 1: "valued at 23963"
    m = _RE["inr_valued_at"].search(text)
    if m:
        return _clean_amount(m.group(1)), "regex_fallback"

    # Fallback 2: "worth 50000"
    m = _RE["inr_worth"].search(text)
    if m:
        return _clean_amount(m.group(1)), "regex_fallback"

    # spaCy MONEY entity fallback
    if doc:
        for ent in doc.ents:
            if ent.label_ == "MONEY":
                raw = re.sub(r"[^\d.]", "", ent.text)
                if raw:
                    try:
                        return float(raw), "spacy_ner"
                    except ValueError:
                        continue

    return None, "not_found"


def _extract_recovery_pct(text: str) -> tuple[Optional[float], str]:
    """
    Extracts property recovery percentage from FIR text.

    Returns:
        (pct_float, method)  where method ∈ {'regex','regex_fallback','not_found'}
    """
    m = _RE["recovery_primary"].search(text)
    if m:
        return float(m.group(1)), "regex"

    m = _RE["recovery_stands"].search(text)
    if m:
        return float(m.group(1)), "regex_fallback"

    m = _RE["recovery_suffix"].search(text)
    if m:
        return float(m.group(1)), "regex_fallback"

    return None, "not_found"


def _extract_prior_convictions(
    text: str, doc
) -> tuple[int, str]:
    """
    Extracts the number of prior convictions from FIR text.

    Returns:
        (count, method)  where method ∈ {'regex','regex_fallback','spacy_ner','default'}
    """
    m = _RE["prior_primary"].search(text)
    if m:
        return int(m.group(1)), "regex"

    m = _RE["prior_records_show"].search(text)
    if m:
        return int(m.group(1)), "regex_fallback"

    m = _RE["prior_convicted"].search(text)
    if m:
        return int(m.group(1)), "regex_fallback"

    # Word-number check ("Prior records show two past offenses")
    text_lower = text.lower()
    for word, num in _WORD_NUMS.items():
        pattern = rf"\b{word}\b\s+(?:past|prior|previous)\s+(?:offense|conviction|record)"
        if re.search(pattern, text_lower):
            return num, "regex_fallback"

    # spaCy CARDINAL near conviction context
    if doc:
        for ent in doc.ents:
            if ent.label_ == "CARDINAL":
                start   = max(0, ent.start_char - 40)
                end     = min(len(text), ent.end_char + 40)
                context = text[start:end].lower()
                if any(
                    kw in context
                    for kw in ("offense", "offence", "conviction", "record", "prior", "past")
                ):
                    try:
                        val = int(ent.text)
                        return val, "spacy_ner"
                    except ValueError:
                        pass

    return 0, "default"


def _keyword_present(text_lower: str, keywords: frozenset) -> bool:
    """Returns True if any keyword from the set appears in the lowercased text."""
    return any(kw in text_lower for kw in keywords)


def _extract_counsel_type(text_lower: str) -> tuple[Optional[str], str]:
    """
    Detects counsel type from keyword matching.

    Returns:
        ("Public Defender" | "Private Counsel" | None, method)
    """
    if _keyword_present(text_lower, _PUBLIC_DEFENDER_KEYWORDS):
        return "Public Defender", "keyword"
    if _keyword_present(text_lower, _PRIVATE_COUNSEL_KEYWORDS):
        return "Private Counsel", "keyword"
    return None, "not_found"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_fir_text(raw_text: str) -> ParsedFIR:
    """
    Parses a raw FIR narrative and extracts the 7 structured case features
    needed for ML model inference.

    Extraction strategy (per field):
        stolen_value_inr       : Regex (INR/Rs/₹ patterns) → spaCy MONEY
        weapon_present         : Weapon keyword set
        forced_entry           : Forced-entry / dwelling keyword set
        prior_convictions      : Regex (digit patterns) → word numbers → spaCy CARDINAL
        property_recovered_pct : Regex (% patterns)
        confession_recorded    : Confession keyword set
        counsel_type           : Public/private counsel keyword sets

    Args:
        raw_text : Raw FIR narrative string (English)

    Returns:
        ParsedFIR  — structured fields + extraction_confidence dict

    Example:
        >>> result = parse_fir_text(
        ...     "Theft of articles valued at INR 23963. Accused without weapon. "
        ...     "2 past offenses. Recovered property stands at 71.9%."
        ... )
        >>> result.stolen_value_inr
        23963.0
        >>> result.prior_convictions
        2
        >>> result.property_recovered_pct
        71.9
    """
    if not raw_text or not raw_text.strip():
        logger.warning("parse_fir_text received empty input.")
        return ParsedFIR(extraction_confidence={
            f: "not_found" for f in [
                "stolen_value_inr", "weapon_present", "forced_entry",
                "prior_convictions", "property_recovered_pct",
                "confession_recorded", "counsel_type",
            ]
        })

    text_lower = raw_text.lower()
    confidence: dict[str, str] = {}

    # ── Run spaCy NER (lazy, used as fallback for numeric fields) ──
    nlp = _get_nlp()
    doc = nlp(raw_text) if nlp else None

    # ── 1. Stolen value ──
    stolen_val, method = _extract_stolen_value(raw_text, doc)
    confidence["stolen_value_inr"] = method

    # ── 2. Recovery percentage ──
    recovery_pct, method = _extract_recovery_pct(raw_text)
    confidence["property_recovered_pct"] = method

    # ── 3. Prior convictions ──
    prior_conv, method = _extract_prior_convictions(raw_text, doc)
    confidence["prior_convictions"] = method

    # ── 4. Weapon present (negation-aware) ──
    # Step 1: check for explicit negation ("without weapon", "no weapon")
    if _WEAPON_NEGATION_RE.search(raw_text):
        weapon = 0
    # Step 2: check for positive weapon mention (specific weapon names or "with weapon")
    elif _WEAPON_POSITIVE_RE.search(raw_text):
        weapon = int(bool(_WEAPON_POSITIVE_RE.search(raw_text)))
    # Step 3: fallback to keyword set (catches "armed", "threatened with")
    else:
        weapon = int(_keyword_present(text_lower, _WEAPON_KEYWORDS))
    confidence["weapon_present"] = "keyword" if weapon else "default"

    # ── 5. Forced entry (keyword match) ──
    forced = int(_keyword_present(text_lower, _FORCED_ENTRY_KEYWORDS))
    confidence["forced_entry"] = "keyword" if forced else "default"

    # ── 6. Confession recorded (keyword match) ──
    confession = int(_keyword_present(text_lower, _CONFESSION_KEYWORDS))
    confidence["confession_recorded"] = "keyword" if confession else "default"

    # ── 7. Counsel type ──
    counsel, method = _extract_counsel_type(text_lower)
    confidence["counsel_type"] = method

    return ParsedFIR(
        stolen_value_inr       = stolen_val,
        weapon_present         = weapon,
        forced_entry           = forced,
        prior_convictions      = prior_conv,
        property_recovered_pct = recovery_pct,
        confession_recorded    = confession,
        counsel_type           = counsel,
        extraction_confidence  = confidence,
    )


# ══════════════════════════════════════════════════════════════════════════════
# CLI SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Self-test against 6 real FIR samples from criminal_theft_dataset_v2_1500.csv.
    Prints extracted values alongside ground-truth columns for comparison.
    """
    # (case_id, fir_text, GT_stolen, GT_weapon, GT_forced, GT_prior, GT_recovery, GT_confession)
    samples = [
        (
            "CASE-2026-1000",
            "Complainant reported theft of articles valued at INR 23963 unnoticed. "
            "Accused was apprehended without weapon. Prior records show 2 past offenses. "
            "Recovered property stands at 71.9%.",
            23963.4, 0, 0, 2, 71.9, 0,
        ),
        (
            "CASE-2026-1001",
            "Complainant reported theft of articles valued at INR 151006 unnoticed. "
            "Accused was apprehended with dangerous weapon. Prior records show 0 past offenses. "
            "Recovered property stands at 50.4%.",
            151006.07, 1, 0, 0, 50.4, 0,
        ),
        (
            "CASE-2026-1003",
            "Complainant reported theft of articles valued at INR 46147 unnoticed. "
            "Accused was apprehended without weapon. Prior records show 1 past offenses. "
            "Recovered property stands at 24.9%.",
            46147.13, 0, 0, 1, 24.9, 1,  # confession_recorded=1 in dataset
        ),
        (
            "CASE-2026-1008",
            "Complainant reported theft of articles valued at INR 46454 by breaking open "
            "the entrance lock. Accused was apprehended without weapon. Prior records show "
            "0 past offenses. Recovered property stands at 80.3%.",
            46454.11, 0, 1, 0, 80.3, 0,
        ),
        (
            "CASE-2026-1005",
            "Complainant reported theft of articles valued at INR 8980 unnoticed. "
            "Accused was apprehended with dangerous weapon. Prior records show 1 past offenses. "
            "Recovered property stands at 33.4%.",
            8979.81, 1, 0, 1, 33.4, 1,
        ),
        (
            "CASE-2026-1007",
            "Complainant reported theft of articles valued at INR 101062 unnoticed. "
            "Accused was apprehended without weapon. Prior records show 2 past offenses. "
            "Recovered property stands at 34.1%.",
            101061.54, 0, 0, 2, 34.1, 0,
        ),
    ]

    sep = "=" * 72
    print(f"\n{sep}")
    print("FIR Parser  —  Self-Test (6 real dataset samples)")
    print(sep)

    for (case_id, fir, gt_val, gt_wp, gt_fe, gt_pr, gt_rec, gt_conf) in samples:
        result = parse_fir_text(fir)
        print(f"\n  [{case_id}]")
        print(f"  FIR  : {fir[:75]}...")

        checks = [
            ("stolen_value_inr",       result.stolen_value_inr,       gt_val,  lambda a, b: abs((a or 0) - b) < 1),
            ("weapon_present",         result.weapon_present,          gt_wp,   lambda a, b: a == b),
            ("forced_entry",           result.forced_entry,            gt_fe,   lambda a, b: a == b),
            ("prior_convictions",      result.prior_convictions,       gt_pr,   lambda a, b: a == b),
            ("property_recovered_pct", result.property_recovered_pct,  gt_rec,  lambda a, b: abs((a or 0) - b) < 0.1),
        ]

        for fname, extracted, gt, match_fn in checks:
            ok  = match_fn(extracted, gt)
            sym = "✓" if ok else "✗"
            conf = result.extraction_confidence.get(fname, "?")
            print(f"    {sym}  {fname:<28}  extracted={str(extracted):<12}  gt={gt:<12}  [{conf}]")
