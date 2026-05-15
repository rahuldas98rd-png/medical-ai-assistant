"""
Rule-based extraction of structured fields from raw OCR text.

This version uses STRUCTURAL anchoring rather than line-based merging.
Each Rx item is identified by the pattern:
    <num>[.\\)]  <form>  <Name>  <dose><unit>
We find every such anchor in the raw text, then extract frequency/duration
from a bounded window after each one. This is robust to OCR line-break
weirdness that confused the line-merging approach.
"""

from __future__ import annotations

import re
from typing import Optional

from backend.modules.prescription_ocr.schemas.prescription import (
    Medicine,
    PrescriptionExtraction,
)
from difflib import get_close_matches

# -------------------------------------------------------------------------
# Frequency patterns. ORDER MATTERS — specific before general.
# In particular, HS/Morning/Evening must be checked BEFORE "OD (once daily)"
# because "Take 1 tablet ... once daily after food at bedtime" would
# otherwise match OD first.
# -------------------------------------------------------------------------
# PRIMARY = how often. Exactly one of these should apply per medicine.
PRIMARY_FREQUENCY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bQID\b|\bq\.?i\.?d\.?\b|\b4\s*times", re.I), "QID (four times daily)"),
    (re.compile(r"\bTDS\b|\bt\.?d\.?s\.?\b|thrice\s+(a|per)?\s*day|\b3\s*times", re.I), "TDS (thrice daily)"),
    (re.compile(r"\bBD\b|\bb\.?d\.?\b|twice\s+(a|per)?\s*day|\b2\s*times", re.I), "BD (twice daily)"),
    (re.compile(r"at\s+bedtime|at\s+night|before\s+sleep|\bHS\b|\bh\.?s\.?\b", re.I), "HS (at bedtime)"),
    (re.compile(r"in\s+the\s+morning|every\s+morning|each\s+morning", re.I), "Morning"),
    (re.compile(r"in\s+the\s+evening|every\s+evening|each\s+evening", re.I), "Evening"),
    (re.compile(r"\bSOS\b|\bs\.?o\.?s\.?\b|as\s+(needed|required)|\bp\.?r\.?n\.?\b", re.I), "SOS (as needed)"),
    (re.compile(r"once\s+(a|per)?\s*day|once\s+daily|every\s+day|\b1\s*time\s+(a|per)?\s*day|\bOD\b|\bo\.?d\.?\b", re.I), "OD (once daily)"),
]

# MODIFIER = WHEN within the day (relative to meals). Combined with primary.
MODIFIER_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bAC\b|before\s+(food|meal)", re.I), "before meals"),
    (re.compile(r"\bPC\b|after\s+(food|meal)", re.I), "after meals"),
]

# -------------------------------------------------------------------------
# Anchor: <num>) <form> <Name> <dose><unit>
# Catches all six medicines in the heart prescription regardless of how
# Tesseract chops up the lines.
# -------------------------------------------------------------------------
RX_ANCHOR = re.compile(
    r"\b(?P<num>\d+)[\.\)]\s*"
    r"(?:Tab|Cap|Inj|Syp|Syrup|Drops?|Oint|Cream|Gel|Sol)\.?\s+"
    r"(?P<name>[A-Z][a-zA-Z]+(?:[\s\-]+[A-Z][a-zA-Z]+)*(?:\s*\([A-Z]+\))?)\s+"
    r"(?P<dose>\d+(?:\.\d+)?)\s*(?P<unit>mg|mcg|g|ml|iu|units?)\b",
    re.IGNORECASE,
)

# Section boundaries — stop scanning a medicine's description window
# once we hit one of these.
SECTION_BOUNDARY = re.compile(
    r"\b(advice|notes?|recommendations?|follow[\s\-]*up|signature|signed|"
    r"next\s+visit|reg\.?\s*no|consultant)\b",
    re.IGNORECASE,
)

# Duration pattern (unchanged from prior version)
DURATION_PATTERN = re.compile(
    r"(?:for|x|×)\s*(\d+)\s*(day|days|week|weeks|month|months)",
    re.IGNORECASE,
)

# -------------------------------------------------------------------------
# Date — handle BOTH slash format AND word format ("May 24, 2024").
# Slash-format requires the prefix not to look like a Rx number.
# -------------------------------------------------------------------------
DATE_WORD = re.compile(
    r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+"
    r"(\d{1,2})[,\s]+\s*(\d{4})\b",
    re.IGNORECASE,
)
DATE_SLASH = re.compile(r"\b(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2,4})\b")
DATE_LABEL_WINDOW = re.compile(r"\bDate\s*[:\-]?\s*([^\n]{0,40})", re.IGNORECASE)

# -------------------------------------------------------------------------
# Patient name — bounded by common header labels so it doesn't run on.
# -------------------------------------------------------------------------
STOP_LABELS = r"Age|Gender|Sex|Address|Diagnosis|DOB|Visit|Date|Phone|Tel|Rx|Prescription"

NAME_LABEL = re.compile(
    rf"\b(?:patient(?:\s+name)?|name|pt\.?\s*name)\s*[:\-]\s*"
    rf"((?:(?!{STOP_LABELS}\b)[A-Z][a-zA-Z\.]+\s*){{1,4}})",
    re.IGNORECASE,
)

AGE_LABEL = re.compile(r"\b(?:age)\s*[:\-/]?\s*(\d{1,3})", re.IGNORECASE)

# Doctor — capture 1-3 words after "Dr.", stop at comma, paren, MD, MBBS, etc.
DR_LABEL = re.compile(
    r"\bDr\.?\s+"
    rf"((?:(?!{STOP_LABELS}\b)[A-Z][a-zA-Z\.]+\s*){{1,3}}?)"
    r"(?:\s*[,\(]|\s+M\.?D\.?|\s+MBBS|\s*$|\s*\n)",
)


# =========================================================================
# Public API
# =========================================================================
def _fuzzy_correct_first_word(name: str, medicine_dict: set[str], cutoff: float = 0.85) -> str:
    """
    Snap OCR-mangled medicine names to dictionary entries.

    Operates on the FIRST word only (the primary medicine name), preserving
    any "<First> Mononitrate (SR)" style suffix. Catches common Tesseract
    confusions:
      Lsosorbide  -> Isosorbide   (I/L)
      Metformln   -> Metformin    (i/l)
      Atorvastat1n -> Atorvastatin (i/1)

    Only applies a correction if the first word is NOT already a dictionary
    entry and a close match (similarity >= cutoff) exists.
    """
    if not medicine_dict or not name:
        return name
    words = name.split()
    if not words:
        return name
    first = words[0].lower()
    if first in medicine_dict:
        return name  # already correct
    matches = get_close_matches(first, medicine_dict, n=1, cutoff=cutoff)
    if matches:
        words[0] = matches[0].title()
        return " ".join(words)
    return name


def extract_structured(
    raw_text: str,
    medicine_dict: set[str],
) -> PrescriptionExtraction:
    medicines = _extract_medicines_structural(raw_text, medicine_dict)

    # Fallback: dictionary-based scan if structural pass found nothing
    if not medicines and medicine_dict:
        medicines = _extract_medicines_dictionary(raw_text, medicine_dict)

    return PrescriptionExtraction(
        patient_name=_extract_patient_name(raw_text),
        patient_age=_first_group(AGE_LABEL, raw_text),
        doctor_name=_extract_doctor(raw_text),
        prescription_date=_extract_date(raw_text),
        medicines=medicines,
        general_instructions=_extract_instructions(raw_text),
        confidence_warnings=_build_warnings(raw_text, medicines),
    )


# -------------------------------------------------------------------------
# Medicine extraction — structural (preferred)
# -------------------------------------------------------------------------
def _extract_medicines_structural(text: str, medicine_dict: set[str]) -> list[Medicine]:
    matches = list(RX_ANCHOR.finditer(text))
    medicines: list[Medicine] = []

    for i, m in enumerate(matches):
        # Description window: from end of this match to whichever comes first:
        #   (a) start of next anchor, (b) section boundary, (c) +200 chars.
        win_start = m.end()
        win_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        boundary = SECTION_BOUNDARY.search(text, win_start, win_end)
        if boundary:
            win_end = boundary.start()
        win_end = min(win_end, win_start + 200)
        description = text[win_start:win_end]

        # Clean trailing "(ER)" / "(SR)" / "(EC)" form suffix off the name
        name = re.sub(r"\s*\([A-Z]+\)\s*$", "", m.group("name")).strip()
        name = _fuzzy_correct_first_word(name, medicine_dict)
        dosage = f"{m.group('dose')} {m.group('unit').lower()}"

        primary_freq: Optional[str] = None
        for pat, label in PRIMARY_FREQUENCY_PATTERNS:
            if pat.search(description):
                primary_freq = label
                break

        modifier: Optional[str] = None
        for pat, label in MODIFIER_PATTERNS:
            if pat.search(description):
                modifier = label
                break

        if primary_freq and modifier:
            frequency = f"{primary_freq} — {modifier}"
        else:
            frequency = primary_freq or modifier

        duration_m = DURATION_PATTERN.search(description)
        duration = (
            f"{duration_m.group(1)} {duration_m.group(2)}" if duration_m else None
        )

        raw_line = re.sub(r"\s+", " ", text[m.start():win_end]).strip()[:200]

        medicines.append(
            Medicine(
                name=name.title(),
                dosage=dosage,
                frequency=frequency,
                duration=duration,
                raw_line=raw_line,
            )
        )

    return medicines


# -------------------------------------------------------------------------
# Medicine extraction — dictionary fallback (used only when structural finds nothing)
# -------------------------------------------------------------------------
def _extract_medicines_dictionary(text: str, medicine_dict: set[str]) -> list[Medicine]:
    found: list[Medicine] = []
    seen: set[str] = set()
    text_lower = text.lower()

    for med in medicine_dict:
        if med in seen:
            continue
        m = re.search(rf"\b{re.escape(med)}\b", text_lower)
        if not m:
            continue

        ctx_start = max(0, m.start() - 50)
        ctx_end = min(len(text), m.end() + 150)
        context = text[ctx_start:ctx_end]

        dose_m = re.search(
            r"(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|iu|units?)", context, re.IGNORECASE
        )
        dosage = f"{dose_m.group(1)} {dose_m.group(2)}" if dose_m else None

        frequency = None
        for pat, label in PRIMARY_FREQUENCY_PATTERNS:
            if pat.search(context):
                frequency = label
                break

        dur_m = DURATION_PATTERN.search(context)
        duration = (
            f"{dur_m.group(1)} {dur_m.group(2)}" if dur_m else None
        )

        found.append(
            Medicine(
                name=med.title(),
                dosage=dosage,
                frequency=frequency,
                duration=duration,
                raw_line=re.sub(r"\s+", " ", context).strip()[:200],
            )
        )
        seen.add(med)

    return found


# -------------------------------------------------------------------------
# Header field extraction
# -------------------------------------------------------------------------
def _extract_patient_name(text: str) -> Optional[str]:
    m = NAME_LABEL.search(text)
    if not m:
        return None
    name = m.group(1).strip().rstrip(",.")
    # Strip salutations
    name = re.sub(r"^(Mr|Mrs|Ms|Dr)\.?\s+", "", name, flags=re.IGNORECASE).strip()
    parts = name.split()
    return " ".join(parts[:4]) if parts else None


def _extract_doctor(text: str) -> Optional[str]:
    m = DR_LABEL.search(text)
    if not m:
        return None
    return m.group(1).strip().rstrip(",.")


def _extract_date(text: str) -> Optional[str]:
    """
    Resolution order:
      1. Word-format date in the 40 chars after a "Date:" label
      2. Slash-format date in that same window
      3. Word-format date anywhere
      4. Slash-format date anywhere, IF not preceded by "No.", "ID", or "Prescription"
    """
    label_m = DATE_LABEL_WINDOW.search(text)
    if label_m:
        snippet = label_m.group(1)
        word_m = DATE_WORD.search(snippet)
        if word_m:
            return word_m.group(0)
        slash_m = DATE_SLASH.search(snippet)
        if slash_m:
            return slash_m.group(0)

    word_m = DATE_WORD.search(text)
    if word_m:
        return word_m.group(0)

    for m in DATE_SLASH.finditer(text):
        before = text[max(0, m.start() - 30):m.start()]
        if re.search(r"\b(no|id|prescription)\b", before, re.IGNORECASE):
            continue
        return m.group(0)

    return None


def _extract_instructions(text: str) -> list[str]:
    advice_m = re.search(
        r"\b(?:advice|notes?|recommendations?)\s*[:\-]?\s*", text, re.IGNORECASE
    )
    if not advice_m:
        return []
    start = advice_m.end()
    end_m = re.search(
        r"\b(?:signature|follow[\s\-]*up|signed|reg\.?\s*no|next\s+follow)\b",
        text[start:],
        re.IGNORECASE,
    )
    block = text[start:start + end_m.start()] if end_m else text[start:start + 800]
    lines = re.split(r"[\n•·]|(?:^|\s)[\*\-]\s+", block)
    return [
        ln.strip().lstrip("•·-* ").rstrip(".")
        for ln in lines
        if len(ln.strip()) > 10
    ][:8]


def _build_warnings(raw_text: str, medicines: list[Medicine]) -> list[str]:
    warnings: list[str] = []
    if not medicines:
        warnings.append(
            "No medicines detected. The OCR text is shown below — try a clearer "
            "image, or check whether medicine names appear in the dictionary."
        )
    if len(raw_text.strip()) < 30:
        warnings.append(
            "Very little text was extracted. The image may be low-resolution, "
            "blurry, or poorly lit."
        )
    no_freq = [m.name for m in medicines if not m.frequency]
    if no_freq:
        warnings.append(
            f"Frequency could not be determined for: {', '.join(no_freq)}. "
            "Verify dosing against the original prescription."
        )
    return warnings


def _first_group(pat: re.Pattern, text: str) -> Optional[str]:
    m = pat.search(text)
    return m.group(1).strip() if m else None