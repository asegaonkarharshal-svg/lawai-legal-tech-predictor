"""
backend/main.py
---------------
Phase 2 - Step 2.3: REST API Endpoints

FastAPI application serving:
- POST /api/v1/parse-fir: NLP parser using spaCy + Regex
- POST /api/v1/predict: Multi-class verdict probabilities + IPC/BNS statutory mapping
- POST /api/v1/explain: SHAP feature attributions and LIME natural language cards
- POST /api/v1/similar-cases: Top-3 matching historical precedents via K-NN
- GET /health: Health check and model registry status
- GET /api/v1/sections: Full statutory reference mapping table
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Ensure repository root is in search path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.fir_parser import parse_fir_text
from src.statutory_mapping import get_all_sections
from backend.schemas import (
    CaseInputSchema,
    ExplanationResponse,
    FIRParseRequest,
    PredictionResponse,
    SimilarCasesResponse,
    StatutoryMetadata,
)
from backend.services import (
    ModelRegistry,
    get_model_registry,
    run_explanation,
    run_prediction,
    run_similar_cases,
    set_model_registry,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lawai-backend")


# ---------------------------------------------------------------------------
# Lifespan Management: Preload all models into memory at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads all ML models, encoders, datasets, and explainers into memory on startup
    and handles graceful cleanup upon application termination.
    """
    logger.info("Initializing LawAI Backend Service...")
    models_dir = BASE_DIR / "models"
    data_path = BASE_DIR / "data" / "criminal_theft_dataset_v2_1500.csv"

    try:
        registry = ModelRegistry(models_dir=models_dir, data_path=data_path)
        set_model_registry(registry)
        app.state.registry = registry
        logger.info("All model artifacts and explainers ready for inference.")
    except Exception as exc:
        logger.exception("Failed to initialize ModelRegistry: %s", exc)
        raise exc

    yield

    logger.info("Shutting down LawAI Backend Service...")


# ---------------------------------------------------------------------------
# FastAPI App Initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LawAI — Legal Tech Predictor API",
    description=(
        "Production-grade RESTful microservice providing criminal theft outcome predictions, "
        "calibrated probability assessments, statutory IPC ↔ BNS mappings, "
        "SHAP/LIME explainability, and precedent case retrieval."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local Streamlit frontend and web integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["System"])
async def root():
    """Root endpoint welcoming users and directing to interactive documentation."""
    return {
        "service": "LawAI Legal Tech Predictor API",
        "status": "online",
        "docs_url": "/docs",
        "health_check": "/health",
        "frontend_instructions": "Run 'streamlit run frontend/app.py' in a separate terminal to view the UI."
    }


@app.get("/health", tags=["System"])
async def health_check():
    """Returns application health and verifies loaded models."""
    try:
        registry = get_model_registry()
        models_ready = bool(registry and registry.xgboost and registry.knn)
    except Exception:
        models_ready = False

    return {
        "status": "online" if models_ready else "degraded",
        "models_loaded": models_ready,
        "api_version": "1.0.0",
        "service": "LawAI Legal Tech Predictor",
    }


@app.post(
    "/api/v1/parse-fir",
    response_model=CaseInputSchema,
    tags=["NLP Extraction"],
    summary="Extract structured case features from raw FIR text",
)
async def api_parse_fir(request: FIRParseRequest):
    """
    Accepts raw First Information Report (FIR) narrative and executes dual-layer
    Regex + spaCy Named Entity Recognition (NER) to extract structured features.
    """
    if not request.raw_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="FIR narrative text cannot be empty or blank.",
        )

    try:
        parsed_fir = parse_fir_text(request.raw_text)
        feature_dict = parsed_fir.to_feature_dict()
        feature_dict["extraction_confidence"] = parsed_fir.extraction_confidence
        return CaseInputSchema(**feature_dict)
    except Exception as exc:
        logger.exception("Error parsing FIR text: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"FIR parsing failed: {str(exc)}",
        )


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    tags=["Legal Prediction"],
    summary="Generate calibrated severity probabilities and statutory sections",
)
async def api_predict(case: CaseInputSchema):
    """
    Evaluates input case features against calibrated XGBoost and Random Forest
    classifiers, computes soft-voting probabilities, and provides deterministic
    IPC and BNS statutory mapping.
    """
    try:
        registry = get_model_registry()
        return run_prediction(case, registry)
    except Exception as exc:
        logger.exception("Error during prediction: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(exc)}",
        )


@app.post(
    "/api/v1/explain",
    response_model=ExplanationResponse,
    tags=["Explainability"],
    summary="Compute SHAP waterfall attributions and LIME natural language cards",
)
async def api_explain(case: CaseInputSchema):
    """
    Computes local model explainability:
    1. SHAP TreeExplainer feature attributions against the base tree model.
    2. LIME Tabular local explanations formatted into human-interpretable legal cards.
    """
    try:
        registry = get_model_registry()
        return run_explanation(case, registry)
    except Exception as exc:
        logger.exception("Error generating explanations: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation generation failed: {str(exc)}",
        )


@app.post(
    "/api/v1/similar-cases",
    response_model=SimilarCasesResponse,
    tags=["Precedent Retrieval"],
    summary="Retrieve top-3 most similar historical case precedents",
)
async def api_similar_cases(case: CaseInputSchema):
    """
    Runs K-Nearest Neighbors (K-NN) cosine distance query on the full 1,500
    precedent database and returns top-3 matching case records.
    """
    try:
        registry = get_model_registry()
        return run_similar_cases(case, registry, top_k=3)
    except Exception as exc:
        logger.exception("Error retrieving similar cases: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Precedent retrieval failed: {str(exc)}",
        )


@app.get(
    "/api/v1/sections",
    response_model=Dict[str, StatutoryMetadata],
    tags=["Legal Reference"],
    summary="Get full IPC ↔ BNS statutory reference table",
)
async def api_get_sections():
    """
    Returns the complete statutory mapping table detailing all theft provisions
    under both the Indian Penal Code (IPC) and Bharatiya Nyaya Sanhita (BNS).
    """
    try:
        sections = get_all_sections()
        response_dict = {}
        for key, sec in sections.items():
            response_dict[key] = StatutoryMetadata(
                ipc_section=sec.ipc_section,
                ipc_description=sec.ipc_description,
                bns_section=sec.bns_section,
                bns_description=sec.bns_description,
                max_punishment_ipc=sec.max_punishment_ipc,
                max_punishment_bns=sec.max_punishment_bns,
                offense_type=sec.offense_type,
                is_bailable=sec.is_bailable,
                summary=sec.summary(),
            )
        return response_dict
    except Exception as exc:
        logger.exception("Error retrieving statutory sections: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch statutory sections: {str(exc)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
