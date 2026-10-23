"""
backend/services.py
-------------------
Phase 2 - Step 2.2: Async Backend Services & Model Handler

Singleton ModelRegistry for loading artifacts:
- Calibrated classifiers: XGBoost, Random Forest, LightGBM
- K-NN precedent model and corpus reference data
- StandardScaler and LabelEncoder
- Pre-initialized SHAP TreeExplainer and LIME TabularExplainer

Implements core inference functions:
- preprocess_case_input()
- run_prediction()
- run_similar_cases()
- run_explanation()
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

# Add project root to sys.path so src imports work reliably
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.statutory_mapping import map_to_sections
from backend.schemas import (
    CaseInputSchema,
    ExplanationResponse,
    LIMECard,
    ModelClassProbability,
    PrecedentCaseSchema,
    PredictionResponse,
    SHAPFeatureContribution,
    SimilarCasesResponse,
    SingleModelVerdict,
    StatutoryMetadata,
)

logger = logging.getLogger(__name__)

SEVERITY_LABELS: Dict[int, str] = {
    0: "Acquittal / Minimal",
    1: "Light Sentence",
    2: "Moderate Sentence",
    3: "Severe Sentence",
}

FEATURE_ORDER: List[str] = [
    "stolen_value_inr",
    "counsel_type_enc",
    "weapon_present",
    "prior_convictions",
    "property_recovered_pct",
    "confession_recorded",
    "forced_entry",
]


class ModelRegistry:
    """
    In-memory singleton loaded during FastAPI lifespan startup.
    Holds all models, preprocessing transformers, datasets, and explainers.
    """
    def __init__(self, models_dir: Path, data_path: Path):
        self.models_dir = models_dir
        self.data_path = data_path

        logger.info("Loading model artifacts from %s...", models_dir)
        self.xgboost = joblib.load(models_dir / "xgboost_calibrated.pkl")
        self.random_forest = joblib.load(models_dir / "random_forest_calibrated.pkl")
        self.scaler = joblib.load(models_dir / "feature_scaler.pkl")
        self.label_encoder = joblib.load(models_dir / "label_encoder.pkl")
        self.feature_names = joblib.load(models_dir / "feature_names.pkl")

        # K-NN retrieval model and dataset
        self.knn = joblib.load(models_dir / "knn_precedent_model.pkl")
        self.knn_ref = joblib.load(models_dir / "knn_reference_data.pkl")

        logger.info("Loading reference historical dataset from %s...", data_path)
        self.dataset_df = pd.read_csv(data_path)
        # Index dataframe by case_id for quick lookups
        self.dataset_by_id = self.dataset_df.set_index("case_id")

        # Initialize Explainers
        logger.info("Initializing SHAP TreeExplainer and LIME TabularExplainer...")
        # Extract underlying base XGBoost estimator from calibrated classifier fold
        try:
            self.base_xgb = self.xgboost.calibrated_classifiers_[0].estimator
        except Exception:
            self.base_xgb = self.xgboost

        self.shap_explainer = shap.TreeExplainer(self.base_xgb)

        # LIME Tabular Explainer on the scaled feature corpus
        X_background = self.knn_ref["X_scaled"]
        self.lime_explainer = LimeTabularExplainer(
            training_data=X_background,
            feature_names=self.feature_names,
            class_names=[SEVERITY_LABELS[i] for i in range(4)],
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )
        logger.info("ModelRegistry successfully initialized.")


# Global singleton holder
_REGISTRY: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Returns the loaded global ModelRegistry singleton."""
    global _REGISTRY
    if _REGISTRY is None:
        models_dir = BASE_DIR / "models"
        data_path = BASE_DIR / "data" / "criminal_theft_dataset_v2_1500.csv"
        _REGISTRY = ModelRegistry(models_dir=models_dir, data_path=data_path)
    return _REGISTRY


def set_model_registry(registry: ModelRegistry) -> None:
    """Explicitly sets the singleton registry (e.g., during app startup)."""
    global _REGISTRY
    _REGISTRY = registry


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_case_input(case: CaseInputSchema, registry: ModelRegistry) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocesses raw case inputs into:
    1. Unscaled numeric array [1, 7]
    2. Scaled feature vector [1, 7] ready for model inference and KNN.
    """
    # Encode counsel_type
    counsel_enc = 1 if case.counsel_type == "Public Defender" else 0

    # Unscaled representation (raw stolen_value)
    unscaled_vec = np.array([
        case.stolen_value_inr,
        float(counsel_enc),
        float(case.weapon_present),
        float(case.prior_convictions),
        case.property_recovered_pct,
        float(case.confession_recorded),
        float(case.forced_entry),
    ], dtype=float).reshape(1, -1)

    # Transform for model: log1p on stolen_value_inr
    transformed_vec = unscaled_vec.copy()
    transformed_vec[0, 0] = np.log1p(transformed_vec[0, 0])

    # Convert to DataFrame with feature names to suppress scikit-learn warnings
    transformed_df = pd.DataFrame(transformed_vec, columns=registry.feature_names)

    # Scale with fitted StandardScaler
    scaled_vec = registry.scaler.transform(transformed_df)
    return unscaled_vec, scaled_vec


# ---------------------------------------------------------------------------
# Prediction Logic
# ---------------------------------------------------------------------------

def run_prediction(case: CaseInputSchema, registry: ModelRegistry) -> PredictionResponse:
    """
    Executes inference across calibrated models, calculates ensemble probabilities,
    maps to statutory sections, and builds a comprehensive PredictionResponse.
    """
    start_t = time.perf_counter()
    _, scaled_vec = preprocess_case_input(case, registry)

    # 1. XGBoost Calibrated Probabilities
    xgb_probs = registry.xgboost.predict_proba(scaled_vec)[0]
    xgb_pred = int(np.argmax(xgb_probs))

    # 2. Random Forest Calibrated Probabilities
    rf_probs = registry.random_forest.predict_proba(scaled_vec)[0]
    rf_pred = int(np.argmax(rf_probs))

    # 3. Soft-Voting Ensemble Probabilities (Average)
    ensemble_probs = (xgb_probs + rf_probs) / 2.0
    ensemble_pred = int(np.argmax(ensemble_probs))
    ensemble_conf = float(np.max(ensemble_probs))

    model_breakdown = [
        SingleModelVerdict(
            model_name="XGBoost (Calibrated)",
            predicted_severity=xgb_pred,
            confidence=float(np.max(xgb_probs)),
            probabilities=ModelClassProbability(
                severity_0=float(xgb_probs[0]),
                severity_1=float(xgb_probs[1]),
                severity_2=float(xgb_probs[2]),
                severity_3=float(xgb_probs[3]),
            ),
        ),
        SingleModelVerdict(
            model_name="Random Forest (Calibrated)",
            predicted_severity=rf_pred,
            confidence=float(np.max(rf_probs)),
            probabilities=ModelClassProbability(
                severity_0=float(rf_probs[0]),
                severity_1=float(rf_probs[1]),
                severity_2=float(rf_probs[2]),
                severity_3=float(rf_probs[3]),
            ),
        ),
    ]

    # Statutory Mapping
    stat_res = map_to_sections(
        forced_entry=case.forced_entry,
        stolen_value_inr=case.stolen_value_inr,
        weapon_present=case.weapon_present,
        prior_convictions=case.prior_convictions,
    )

    statutory_info = StatutoryMetadata(
        ipc_section=stat_res.ipc_section,
        ipc_description=stat_res.ipc_description,
        bns_section=stat_res.bns_section,
        bns_description=stat_res.bns_description,
        max_punishment_ipc=stat_res.max_punishment_ipc,
        max_punishment_bns=stat_res.max_punishment_bns,
        offense_type=stat_res.offense_type,
        is_bailable=stat_res.is_bailable,
        summary=stat_res.summary(),
    )

    elapsed_ms = (time.perf_counter() - start_t) * 1000.0

    return PredictionResponse(
        predicted_severity=ensemble_pred,
        severity_label=SEVERITY_LABELS[ensemble_pred],
        confidence=round(ensemble_conf, 4),
        ensemble_probabilities=ModelClassProbability(
            severity_0=round(float(ensemble_probs[0]), 4),
            severity_1=round(float(ensemble_probs[1]), 4),
            severity_2=round(float(ensemble_probs[2]), 4),
            severity_3=round(float(ensemble_probs[3]), 4),
        ),
        statutory_info=statutory_info,
        model_breakdown=model_breakdown,
        processing_time_ms=round(elapsed_ms, 2),
    )


# ---------------------------------------------------------------------------
# Precedent Retrieval (K-NN)
# ---------------------------------------------------------------------------

def run_similar_cases(case: CaseInputSchema, registry: ModelRegistry, top_k: int = 3) -> SimilarCasesResponse:
    """
    Retrieves the top-k nearest matching precedent cases using cosine similarity.
    """
    start_t = time.perf_counter()
    _, scaled_vec = preprocess_case_input(case, registry)

    distances, indices = registry.knn.kneighbors(scaled_vec, n_neighbors=top_k)
    distances = distances[0]
    indices = indices[0]

    precedents: List[PrecedentCaseSchema] = []
    case_ids = registry.knn_ref["case_ids"]

    for dist, idx in zip(distances, indices):
        case_id = case_ids[idx]
        row = registry.dataset_by_id.loc[case_id]
        # Cosine similarity is 1.0 - cosine distance
        sim_score = max(0.0, min(1.0, 1.0 - float(dist)))

        sev = int(row["sentence_severity"])
        precedent = PrecedentCaseSchema(
            case_id=case_id,
            similarity_score=round(sim_score, 4),
            stolen_value_inr=float(row["stolen_value_inr"]),
            counsel_type=str(row["counsel_type"]),
            weapon_present=int(row["weapon_present"]),
            prior_convictions=int(row["prior_convictions"]),
            property_recovered_pct=float(row["property_recovered_pct"]),
            confession_recorded=int(row["confession_recorded"]),
            forced_entry=int(row["forced_entry"]),
            ipc_section=str(row["ipc_section"]),
            bns_section=str(row["bns_section"]),
            sentence_severity=sev,
            severity_label=SEVERITY_LABELS.get(sev, f"Class {sev}"),
            fir_text=str(row["fir_text"]),
        )
        precedents.append(precedent)

    elapsed_ms = (time.perf_counter() - start_t) * 1000.0

    return SimilarCasesResponse(
        query_case=case,
        total_precedents_found=len(precedents),
        precedents=precedents,
        processing_time_ms=round(elapsed_ms, 2),
    )


# ---------------------------------------------------------------------------
# Explainability (SHAP & LIME)
# ---------------------------------------------------------------------------

def _build_lime_card(feature_condition: str, weight: float, case: CaseInputSchema) -> LIMECard:
    """Converts a LIME condition string and weight into a structured natural language card."""
    direction = "positive" if weight > 0 else "negative"

    # Friendly human-readable phrasing for legal interpretation
    nl_desc = ""
    cond_clean = feature_condition.replace("counsel_type_enc", "Counsel Type")
    cond_clean = cond_clean.replace("stolen_value_inr", "Stolen Value")
    cond_clean = cond_clean.replace("weapon_present", "Weapon Present")
    cond_clean = cond_clean.replace("forced_entry", "Forced Entry")
    cond_clean = cond_clean.replace("prior_convictions", "Prior Convictions")
    cond_clean = cond_clean.replace("property_recovered_pct", "Property Recovery %")
    cond_clean = cond_clean.replace("confession_recorded", "Confession Recorded")

    if "weapon_present" in feature_condition:
        if case.weapon_present == 1:
            nl_desc = "Presence of dangerous weapon acts as an aggravating factor, increasing severity."
        else:
            nl_desc = "Absence of weapon acts as a mitigating factor."
    elif "forced_entry" in feature_condition:
        if case.forced_entry == 1:
            nl_desc = "Forced entry into dwelling house directly elevates statutory severity under BNS 305."
        else:
            nl_desc = "No forced entry detected, mitigating severity."
    elif "prior_convictions" in feature_condition:
        if case.prior_convictions > 0:
            nl_desc = f"History of {case.prior_convictions} prior offenses significantly increases sentencing severity."
        else:
            nl_desc = "Clean criminal record serves as a mitigating factor in sentencing."
    elif "stolen_value_inr" in feature_condition:
        if case.stolen_value_inr > 50000:
            nl_desc = f"High stolen value (INR {case.stolen_value_inr:,.0f}) increases sentencing threshold."
        else:
            nl_desc = f"Modest stolen property value (INR {case.stolen_value_inr:,.0f}) limits maximum sentencing."
    elif "property_recovered_pct" in feature_condition:
        if case.property_recovered_pct >= 50.0:
            nl_desc = f"Substantial property recovery ({case.property_recovered_pct:.1f}%) acts favorably for mitigation."
        else:
            nl_desc = f"Low property recovery ({case.property_recovered_pct:.1f}%) weakens mitigation arguments."
    elif "confession_recorded" in feature_condition:
        if case.confession_recorded == 1:
            nl_desc = "Recorded confession provides judicial admission but may support cooperation mitigation."
        else:
            nl_desc = "Absence of recorded confession leaves standard burden on prosecution evidence."
    else:
        nl_desc = f"Feature condition '{cond_clean}' has a {direction} influence ({weight:+.3f}) on the severity verdict."

    # Identify primary feature name
    primary_feat = "unknown"
    for f in FEATURE_ORDER:
        if f in feature_condition:
            primary_feat = f
            break

    return LIMECard(
        feature_name=primary_feat,
        condition=cond_clean,
        weight=round(weight, 4),
        direction=direction,
        natural_language=nl_desc,
    )


def run_explanation(case: CaseInputSchema, registry: ModelRegistry) -> ExplanationResponse:
    """
    Generates quantitative SHAP feature contributions and interpretable LIME cards.
    """
    start_t = time.perf_counter()
    unscaled_vec, scaled_vec = preprocess_case_input(case, registry)

    # 1. Determine target class
    xgb_probs = registry.xgboost.predict_proba(scaled_vec)[0]
    predicted_class = int(np.argmax(xgb_probs))

    # 2. Compute SHAP values via TreeExplainer on base XGBoost
    shap_vals = registry.shap_explainer.shap_values(scaled_vec)

    # Check shap structure (multi-class tree returns list of arrays or 3D array)
    if isinstance(shap_vals, list):
        class_shap = shap_vals[predicted_class][0]
    elif len(shap_vals.shape) == 3:
        class_shap = shap_vals[0, :, predicted_class]
    else:
        class_shap = shap_vals[0]

    # Expected value (baseline)
    expected_val = registry.shap_explainer.expected_value
    if isinstance(expected_val, (list, np.ndarray)):
        base_val = float(expected_val[predicted_class])
    else:
        base_val = float(expected_val)

    # Build SHAP contributions sorted by absolute magnitude
    shap_items: List[SHAPFeatureContribution] = []
    raw_vals = [
        case.stolen_value_inr,
        1.0 if case.counsel_type == "Public Defender" else 0.0,
        float(case.weapon_present),
        float(case.prior_convictions),
        case.property_recovered_pct,
        float(case.confession_recorded),
        float(case.forced_entry),
    ]

    for idx, f_name in enumerate(registry.feature_names):
        sv = float(class_shap[idx])
        direction = "positive" if sv >= 0 else "negative"
        shap_items.append(
            SHAPFeatureContribution(
                feature_name=f_name,
                unscaled_value=raw_vals[idx],
                shap_value=round(sv, 4),
                direction=direction,
                rank=0,  # will assign after sort
            )
        )

    # Sort descending by absolute SHAP impact
    shap_items.sort(key=lambda item: abs(item.shap_value), reverse=True)
    for rank, item in enumerate(shap_items, 1):
        item.rank = rank

    # 3. Compute LIME tabular local explanation
    def predict_fn(x):
        # Base model prediction probabilities
        return registry.base_xgb.predict_proba(x)

    lime_exp = registry.lime_explainer.explain_instance(
        data_row=scaled_vec[0],
        predict_fn=predict_fn,
        labels=[predicted_class],
        num_features=5,
    )

    lime_cards: List[LIMECard] = []
    exp_list = lime_exp.as_list(label=predicted_class)
    for cond_str, weight in exp_list:
        card = _build_lime_card(cond_str, weight, case)
        lime_cards.append(card)

    elapsed_ms = (time.perf_counter() - start_t) * 1000.0

    return ExplanationResponse(
        predicted_severity=predicted_class,
        severity_label=SEVERITY_LABELS[predicted_class],
        shap_base_value=round(base_val, 4),
        shap_contributions=shap_items,
        lime_cards=lime_cards,
        explanation_model="XGBoost Base Tree (SHAP) + LIME Tabular",
        processing_time_ms=round(elapsed_ms, 2),
    )
