"""
backend/schemas.py
------------------
Phase 2 - Step 2.1: Pydantic Data Schemas

Defines request and response validation models for the LawAI FastAPI service:
- FIRParseRequest: Unstructured input text from FIR reports.
- CaseInputSchema: 7 structured numerical and categorical features.
- PredictionResponse: Multi-class verdict probabilities, labels, and statutory mappings.
- PrecedentCaseSchema: Historical case precedent details with similarity scores.
- ExplanationResponse: SHAP feature attributions and LIME natural language cards.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# 1. FIR Parsing Request & Case Input Schema
# ---------------------------------------------------------------------------

class FIRParseRequest(BaseModel):
    raw_text: str = Field(
        ...,
        min_length=5,
        description="Raw narrative text of the First Information Report (FIR).",
        examples=[
            "Complainant reported theft of articles valued at INR 46454 by breaking open the entrance lock. "
            "Accused was apprehended without weapon. Prior records show 0 past offenses. "
            "Recovered property stands at 80.3%."
        ],
    )


class CaseInputSchema(BaseModel):
    stolen_value_inr: float = Field(
        ...,
        ge=0.0,
        description="Estimated value of stolen property in Indian Rupees (INR).",
        examples=[46454.0],
    )
    counsel_type: str = Field(
        ...,
        pattern=r"^(Public Defender|Private Counsel)$",
        description="Type of legal representation for the accused ('Public Defender' or 'Private Counsel').",
        examples=["Public Defender"],
    )
    weapon_present: int = Field(
        ...,
        ge=0,
        le=1,
        description="Binary indicator: 1 if weapon was present/used, 0 otherwise.",
        examples=[0],
    )
    prior_convictions: int = Field(
        ...,
        ge=0,
        description="Count of previous criminal offenses / convictions.",
        examples=[0],
    )
    property_recovered_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of stolen property recovered by authorities (0.0 to 100.0).",
        examples=[80.3],
    )
    confession_recorded: int = Field(
        ...,
        ge=0,
        le=1,
        description="Binary indicator: 1 if confession was formally recorded, 0 otherwise.",
        examples=[0],
    )
    forced_entry: int = Field(
        ...,
        ge=0,
        le=1,
        description="Binary indicator: 1 if forced entry or lock breaking into dwelling occurred, 0 otherwise.",
        examples=[1],
    )
    extraction_confidence: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Provenance / extraction method for fields ('regex', 'keyword', 'spacy_ner', etc.).",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "stolen_value_inr": 46454.0,
                "counsel_type": "Public Defender",
                "weapon_present": 0,
                "prior_convictions": 0,
                "property_recovered_pct": 80.3,
                "confession_recorded": 0,
                "forced_entry": 1,
                "extraction_confidence": {
                    "stolen_value_inr": "regex",
                    "weapon_present": "keyword",
                    "forced_entry": "keyword",
                    "prior_convictions": "regex",
                    "property_recovered_pct": "regex",
                },
            }
        }
    )


# ---------------------------------------------------------------------------
# 2. Prediction Response Schemas
# ---------------------------------------------------------------------------

class StatutoryMetadata(BaseModel):
    ipc_section: str
    ipc_description: str
    bns_section: str
    bns_description: str
    max_punishment_ipc: str
    max_punishment_bns: str
    offense_type: str
    is_bailable: bool
    summary: str


class ModelClassProbability(BaseModel):
    severity_0: float = Field(..., description="Probability of Class 0 (Acquittal / Minimal)")
    severity_1: float = Field(..., description="Probability of Class 1 (Light Sentence)")
    severity_2: float = Field(..., description="Probability of Class 2 (Moderate Sentence)")
    severity_3: float = Field(..., description="Probability of Class 3 (Severe Sentence)")


class SingleModelVerdict(BaseModel):
    model_name: str
    predicted_severity: int
    confidence: float
    probabilities: ModelClassProbability


class PredictionResponse(BaseModel):
    predicted_severity: int = Field(..., description="Ensemble predicted sentence severity (0-3).")
    severity_label: str = Field(..., description="Human-readable severity level name.")
    confidence: float = Field(..., description="Max calibrated confidence probability.")
    ensemble_probabilities: ModelClassProbability
    statutory_info: StatutoryMetadata
    model_breakdown: List[SingleModelVerdict]
    processing_time_ms: float


# ---------------------------------------------------------------------------
# 3. Precedent Case Schema (K-NN Retrieval)
# ---------------------------------------------------------------------------

class PrecedentCaseSchema(BaseModel):
    case_id: str
    similarity_score: float = Field(..., description="Cosine similarity score (0.0 to 1.0).")
    stolen_value_inr: float
    counsel_type: str
    weapon_present: int
    prior_convictions: int
    property_recovered_pct: float
    confession_recorded: int
    forced_entry: int
    ipc_section: str
    bns_section: str
    sentence_severity: int
    severity_label: str
    fir_text: str


class SimilarCasesResponse(BaseModel):
    query_case: CaseInputSchema
    total_precedents_found: int
    precedents: List[PrecedentCaseSchema]
    processing_time_ms: float


# ---------------------------------------------------------------------------
# 4. Explainability Schemas (SHAP & LIME)
# ---------------------------------------------------------------------------

class SHAPFeatureContribution(BaseModel):
    feature_name: str
    unscaled_value: float
    shap_value: float
    direction: str = Field(..., description="'positive' if pushing toward severity, 'negative' if mitigating.")
    rank: int


class LIMECard(BaseModel):
    feature_name: str
    condition: str
    weight: float
    direction: str
    natural_language: str


class ExplanationResponse(BaseModel):
    predicted_severity: int
    severity_label: str
    shap_base_value: float
    shap_contributions: List[SHAPFeatureContribution]
    lime_cards: List[LIMECard]
    explanation_model: str
    processing_time_ms: float
