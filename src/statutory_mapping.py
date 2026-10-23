"""
statutory_mapping.py
--------------------
Phase 1 – Step 1.2: Statutory Code Mapping Module

Deterministic, rule-based lookup engine mapping case feature inputs
(forced_entry, stolen_value_inr, weapon_present) to applicable legal
sections under the Indian Penal Code (IPC) and the Bharatiya Nyaya
Sanhita (BNS 2023, effective 1 July 2024).

Returns full statutory metadata — offense definitions, punishment caps,
bailability — alongside verdict predictions from the ML models.

No ML involved: this module is purely rule-based and always deterministic.

Usage:
    from src.statutory_mapping import map_to_sections, get_section_by_label
    result = map_to_sections(forced_entry=1, stolen_value_inr=46454, weapon_present=0)
    print(result.bns_section)  # "BNS 305"
"""

from dataclasses import asdict, dataclass
from typing import Optional

# ─── Configurable thresholds ─────────────────────────────────────────────────
# Derived from dataset labeling patterns (IPC 379 Aggravated vs Simple boundary).
# NOTE: IPC 378 (basic petty theft) is NOT present in the dataset; only 3 sections
# exist: IPC 379 Simple, IPC 379 Aggravated, IPC 380. The mapping reflects this.
AGGRAVATED_VALUE_THRESHOLD_INR: float = 50_000.0


# ─── Statutory result dataclass ───────────────────────────────────────────────
@dataclass(frozen=True)
class StatutoryResult:
    """
    Immutable container for a matched IPC / BNS statutory section.

    Attributes:
        ipc_section          : IPC section string, e.g. "IPC 379 (Simple Theft)"
        ipc_description      : Full legal text of the IPC section
        bns_section          : BNS section string, e.g. "BNS 303(1)"
        bns_description      : Full legal text of the BNS section
        max_punishment_ipc   : Maximum sentence under IPC (string)
        max_punishment_bns   : Maximum sentence under BNS (string)
        offense_type         : "Petty" | "Simple" | "Aggravated" | "Dwelling"
        is_bailable          : True if offense is bailable
        severity_range       : (min, max) expected sentence_severity from ML model
    """
    ipc_section:        str
    ipc_description:    str
    bns_section:        str
    bns_description:    str
    max_punishment_ipc: str
    max_punishment_bns: str
    offense_type:       str
    is_bailable:        bool
    severity_range:     tuple   # (min_severity, max_severity)

    def to_dict(self) -> dict:
        """Serialise to plain dict (severity_range as list for JSON compat)."""
        d = asdict(self)
        d["severity_range"] = list(self.severity_range)
        return d

    def summary(self) -> str:
        """One-line human-readable summary for display."""
        bail = "Bailable" if self.is_bailable else "Non-Bailable"
        return (
            f"{self.offense_type} Theft  |  "
            f"{self.ipc_section} → {self.bns_section}  |  "
            f"Max: {self.max_punishment_bns}  |  {bail}"
        )


# ─── Section reference table ──────────────────────────────────────────────────
_SECTION_TABLE: dict[str, StatutoryResult] = {

    # ── IPC 378 / BNS 303 ── Basic/Petty Theft ────────────────────────────
    "IPC_378": StatutoryResult(
        ipc_section        = "IPC 378",
        ipc_description    = (
            "Whoever, intending to take dishonestly any moveable property out of "
            "the possession of any person without that person's consent, moves that "
            "property in order to such taking, is said to commit theft."
        ),
        bns_section        = "BNS 303",
        bns_description    = (
            "Whoever, intending to take dishonestly any moveable property out of the "
            "possession of any person without that person's consent, moves that "
            "property, is said to commit theft. (Bharatiya Nyaya Sanhita, 2023)"
        ),
        max_punishment_ipc = "3 years imprisonment and/or fine",
        max_punishment_bns = "3 years imprisonment and/or fine",
        offense_type       = "Petty",
        is_bailable        = True,
        severity_range     = (0, 1),
    ),

    # ── IPC 379 Simple / BNS 303(1) ── Petty/Simple Theft ────────────────
    "IPC_379_SIMPLE": StatutoryResult(
        ipc_section        = "IPC 379 (Simple Theft)",
        ipc_description    = (
            "Whoever commits theft shall be punished with imprisonment of either "
            "description for a term which may extend to three years, or with fine, "
            "or with both. (Punishment for theft, simple form)"
        ),
        bns_section        = "BNS 303(1)",
        bns_description    = (
            "Whoever commits theft shall be punished with imprisonment of either "
            "description for a term which may extend to three years, or with fine, "
            "or with both. (BNS 303(1) — Petty/Simple Theft, Bharatiya Nyaya Sanhita 2023)"
        ),
        max_punishment_ipc = "3 years imprisonment and/or fine",
        max_punishment_bns = "3 years imprisonment and/or fine",
        offense_type       = "Simple",
        is_bailable        = True,
        severity_range     = (1, 2),
    ),

    # ── IPC 379 Aggravated / BNS 303(2) ── Aggravated Theft ──────────────
    "IPC_379_AGGRAVATED": StatutoryResult(
        ipc_section        = "IPC 379 (Aggravated Theft)",
        ipc_description    = (
            "Aggravated form of theft involving use of a dangerous weapon or theft "
            "of property of high value (exceeding prescribed threshold). Enhanced "
            "punishment applies under aggravating circumstances."
        ),
        bns_section        = "BNS 303(2)",
        bns_description    = (
            "Whoever commits theft in aggravated circumstances — including use of a "
            "weapon, imminent threat to any person, or theft of high-value property — "
            "shall be punished with rigorous imprisonment which may extend to seven "
            "years, and shall also be liable to fine. "
            "(BNS 303(2) — Aggravated Theft, Bharatiya Nyaya Sanhita 2023)"
        ),
        max_punishment_ipc = "7 years rigorous imprisonment and fine",
        max_punishment_bns = "7 years rigorous imprisonment and fine",
        offense_type       = "Aggravated",
        is_bailable        = False,
        severity_range     = (2, 3),
    ),

    # ── IPC 380 / BNS 305 ── Dwelling House Theft ─────────────────────────
    "IPC_380": StatutoryResult(
        ipc_section        = "IPC 380",
        ipc_description    = (
            "Whoever commits theft in any building, tent or vessel, which building, "
            "tent or vessel is used as a human dwelling, or used for the custody of "
            "property, shall be punished with imprisonment of either description for "
            "a term which may extend to seven years, and shall also be liable to fine."
        ),
        bns_section        = "BNS 305",
        bns_description    = (
            "Whoever commits theft in any building used as a human dwelling, or used "
            "for the custody of property, or in any vehicle used for transportation "
            "of persons or goods, shall be punished with imprisonment of either "
            "description for a term which may extend to seven years, and shall also "
            "be liable to fine. "
            "(BNS 305 — Theft in Dwelling House, Bharatiya Nyaya Sanhita 2023)"
        ),
        max_punishment_ipc = "7 years imprisonment and fine",
        max_punishment_bns = "7 years imprisonment and fine",
        offense_type       = "Dwelling",
        is_bailable        = False,
        severity_range     = (0, 3),
    ),
}


# ─── Primary lookup function ──────────────────────────────────────────────────

def map_to_sections(
    forced_entry:      int,
    stolen_value_inr:  float,
    weapon_present:    int,
    prior_convictions: int = 0,
) -> StatutoryResult:
    """
    Maps case feature inputs to the applicable IPC and BNS legal sections.

    Decision logic (mirrors the dataset's labeling rules exactly):

        Rule 1 — Forced entry takes priority:
            forced_entry == 1
                → IPC 380 / BNS 305  (Theft in Dwelling House)

        Rule 2 — Aggravating factors:
            weapon_present == 1  OR  stolen_value_inr > 50,000
                → IPC 379 Aggravated / BNS 303(2)

        Rule 3 — Default (simple theft):
                → IPC 379 Simple / BNS 303(1)

    Note: IPC 378 (basic/petty theft) is not present in the dataset and is
    therefore not returned by this function. The 3 sections above cover all
    1,500 cases in criminal_theft_dataset_v2_1500.csv.

    Args:
        forced_entry      : 1 if accused forced entry into dwelling, else 0
        stolen_value_inr  : Monetary value of stolen property in INR
        weapon_present    : 1 if a dangerous weapon was used/present, else 0
        prior_convictions : Number of prior offenses (reserved for future
                            enhancement — does not affect section mapping yet)

    Returns:
        StatutoryResult  — frozen dataclass with full IPC/BNS metadata
    """
    # Rule 1: Forced entry into dwelling → highest priority
    if forced_entry == 1:
        return _SECTION_TABLE["IPC_380"]

    # Rule 2: Aggravating factors (weapon OR high value)
    if weapon_present == 1 or stolen_value_inr > AGGRAVATED_VALUE_THRESHOLD_INR:
        return _SECTION_TABLE["IPC_379_AGGRAVATED"]

    # Rule 3: Default simple theft
    return _SECTION_TABLE["IPC_379_SIMPLE"]


# ─── Reference table accessors ────────────────────────────────────────────────

def get_all_sections() -> dict[str, StatutoryResult]:
    """Returns the complete reference table of all statutory sections."""
    return dict(_SECTION_TABLE)


def get_section_by_label(label: str) -> Optional[StatutoryResult]:
    """
    Lookup a statutory section by its IPC or BNS string label.

    Accepts labels in any format present in the dataset:
        'IPC 378', 'IPC 379 (Simple Theft)', 'IPC 379 (Aggravated Theft)',
        'IPC 380', 'IPC 380 (Theft in dwelling house)',
        'BNS 303', 'BNS 303(1)', 'BNS 303(1) (Petty/Simple Theft)',
        'BNS 303(2)', 'BNS 303(2) (Aggravated Theft)',
        'BNS 305',  'BNS 305 (Theft in dwelling house)'

    Returns None if the label is not recognised.
    """
    _LABEL_MAP: dict[str, str] = {
        # IPC labels
        "IPC 378":                            "IPC_378",
        "IPC 379":                            "IPC_379_SIMPLE",
        "IPC 379 (Simple Theft)":             "IPC_379_SIMPLE",
        "IPC 379 (Aggravated Theft)":         "IPC_379_AGGRAVATED",
        "IPC 380":                            "IPC_380",
        "IPC 380 (Theft in dwelling house)":  "IPC_380",
        # BNS labels
        "BNS 303":                                  "IPC_378",
        "BNS 303(1)":                               "IPC_379_SIMPLE",
        "BNS 303(1) (Petty/Simple Theft)":          "IPC_379_SIMPLE",
        "BNS 303(2)":                               "IPC_379_AGGRAVATED",
        "BNS 303(2) (Aggravated Theft)":            "IPC_379_AGGRAVATED",
        "BNS 305":                                  "IPC_380",
        "BNS 305 (Theft in dwelling house)":        "IPC_380",
    }
    key = _LABEL_MAP.get(label.strip())
    return _SECTION_TABLE.get(key) if key else None


# ─── CLI self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Self-test using real cases from criminal_theft_dataset_v2_1500.csv
    to verify the mapping logic matches dataset ground-truth labels.
    """
    test_cases = [
        # (case_id, forced, value, weapon, expected_bns)
        ("CASE-2026-1000", 0, 23_963.40,  0, "BNS 303(1)"),  # Simple
        ("CASE-2026-1001", 0, 151_006.07, 1, "BNS 303(2)"),  # Aggravated (weapon+value)
        ("CASE-2026-1002", 0, 66_337.28,  0, "BNS 303(2)"),  # Aggravated (value > 50k)
        ("CASE-2026-1003", 0, 46_147.13,  0, "BNS 303(1)"),  # Simple
        ("CASE-2026-1005", 0, 8_979.81,   1, "BNS 303(2)"),  # Aggravated (weapon)
        ("CASE-2026-1008", 1, 46_454.11,  0, "BNS 305"),     # Dwelling (forced entry)
        ("CASE-2026-1004", 0, 8_981.24,   0, "BNS 303(1)"),  # Simple (8981 > petty threshold)
    ]

    sep = "=" * 70
    print(f"\n{sep}")
    print("Statutory Mapping  —  Self-Test (matching dataset ground truth)")
    print(sep)

    all_pass = True
    for case_id, fe, val, wp, expected in test_cases:
        result   = map_to_sections(fe, val, wp)
        matched  = result.bns_section.startswith(expected.split(" ")[0] + " " +
                   expected.split(" ")[1] if len(expected.split()) > 1 else expected)
        # simpler equality check
        matched  = expected in result.bns_section or result.bns_section.startswith(expected)
        status   = "✓ PASS" if matched else "✗ FAIL"
        if not matched:
            all_pass = False
        print(
            f"  {status}  {case_id}  forced={fe} val={val:>10,.0f} weapon={wp}"
            f"  →  {result.bns_section:<35}  (expected: {expected})"
        )

    print(f"\n{'All tests PASSED ✓' if all_pass else 'Some tests FAILED ✗'}")
    print(sep)

    print("\n--- Full Section Reference Table ---")
    for key, sec in get_all_sections().items():
        print(f"\n  [{key}]  {sec.summary()}")
