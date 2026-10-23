"""
train_models.py
---------------
Phase 1 – Step 1.1: ML Training, Isotonic Calibration & K-NN Search Engine

Trains XGBoost, Random Forest, and LightGBM multi-class classifiers on the
criminal theft dataset (sentence_severity: 0–3). Each classifier is wrapped
with CalibratedClassifierCV(method='isotonic', cv=5) for reliable probability
output. Trains a NearestNeighbors engine on the full scaled dataset for
precedent-case retrieval. Saves all artifacts to models/.

Usage:
    python src/train_models.py
"""

import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_PATH  = BASE_DIR / "data" / "criminal_theft_dataset_v2_1500.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
TARGET_COL    = "sentence_severity"
RANDOM_STATE  = 42
TEST_SIZE     = 0.15          # 15% held-out test split

# Ordered feature names after encoding (used by backend services for inference)
FEATURE_NAMES = [
    "stolen_value_inr",       # log1p-transformed before scaling
    "counsel_type_enc",       # LabelEncoded: Public Defender=1, Private Counsel=0
    "weapon_present",
    "prior_convictions",
    "property_recovered_pct",
    "confession_recorded",
    "forced_entry",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD & PREPROCESS
# ══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess(
    csv_path: Path = DATA_PATH,
) -> tuple:
    """
    Loads the dataset, encodes categoricals, applies log1p to stolen_value_inr,
    and fits a StandardScaler on the feature matrix.

    Returns:
        df             : Raw DataFrame (1 500 rows)
        X_scaled       : Scaled feature matrix  (ndarray, shape [1500, 7])
        y              : Target array            (ndarray, shape [1500,])
        scaler         : Fitted StandardScaler
        label_encoder  : Fitted LabelEncoder (counsel_type)
    """
    logger.info(f"Loading dataset → {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"  Shape : {df.shape}")

    # ── Encode counsel_type ──
    le = LabelEncoder()
    df["counsel_type_enc"] = le.fit_transform(df["counsel_type"])
    # LabelEncoder classes: alphabetical → 'Private Counsel'=0, 'Public Defender'=1
    logger.info(f"  counsel_type classes : {le.classes_.tolist()}")

    X = df[FEATURE_NAMES].copy()

    # ── log1p on stolen_value_inr (right-skewed: ₹732 – ₹4,09,122, mean ₹50,923) ──
    X["stolen_value_inr"] = np.log1p(X["stolen_value_inr"])

    y = df[TARGET_COL].values
    unique, counts = np.unique(y, return_counts=True)
    logger.info(f"  Class distribution : { dict(zip(unique.tolist(), counts.tolist())) }")

    # ── Fit scaler on full dataset (KNN uses full-dataset scaling) ──
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled, y, scaler, le


# ══════════════════════════════════════════════════════════════════════════════
# 2.  TRAIN CLASSIFIERS  (XGBoost · Random Forest · LightGBM)
# ══════════════════════════════════════════════════════════════════════════════

def train_all_models(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """
    Trains three calibrated multi-class classifiers on the training split.

    Each base estimator is wrapped with:
        CalibratedClassifierCV(method='isotonic', cv=5)

    Why isotonic over Platt (sigmoid)?
        - Isotonic is non-parametric → handles asymmetric calibration curves
        - With 4 classes and ~1050 training samples, it outperforms sigmoid
          calibration on Log Loss and Reliability Diagrams.

    Returns:
        dict with keys: 'xgboost', 'random_forest', 'lgbm'
    """
    n_classes = len(np.unique(y_train))
    models    = {}

    # ── XGBoost ────────────────────────────────────────────────────────────
    logger.info("Training XGBoost  ...")
    xgb_base = XGBClassifier(
        n_estimators      = 300,
        max_depth         = 5,
        learning_rate     = 0.05,
        objective         = "multi:softprob",
        num_class         = n_classes,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        eval_metric       = "mlogloss",
        random_state      = RANDOM_STATE,
        verbosity         = 0,
    )
    xgb_cal = CalibratedClassifierCV(xgb_base, method="isotonic", cv=5)
    xgb_cal.fit(X_train, y_train)
    models["xgboost"] = xgb_cal
    logger.info("  XGBoost calibrated ✓")

    # ── Random Forest ──────────────────────────────────────────────────────
    logger.info("Training Random Forest ...")
    rf_base = RandomForestClassifier(
        n_estimators    = 300,
        max_depth       = None,
        min_samples_split = 4,
        class_weight    = "balanced",
        random_state    = RANDOM_STATE,
        n_jobs          = -1,
    )
    rf_cal = CalibratedClassifierCV(rf_base, method="isotonic", cv=5)
    rf_cal.fit(X_train, y_train)
    models["random_forest"] = rf_cal
    logger.info("  Random Forest calibrated ✓")

    # ── LightGBM ───────────────────────────────────────────────────────────
    logger.info("Training LightGBM ...")
    lgbm_base = LGBMClassifier(
        n_estimators  = 300,
        num_leaves    = 31,
        learning_rate = 0.05,
        objective     = "multiclass",
        num_class     = n_classes,
        class_weight  = "balanced",
        random_state  = RANDOM_STATE,
        n_jobs        = -1,
        verbose       = -1,
    )
    lgbm_cal = CalibratedClassifierCV(lgbm_base, method="isotonic", cv=5)
    lgbm_cal.fit(X_train, y_train)
    models["lgbm"] = lgbm_cal
    logger.info("  LightGBM calibrated ✓")

    return models


# ══════════════════════════════════════════════════════════════════════════════
# 3.  TRAIN KNN PRECEDENT ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def train_knn(X_all: np.ndarray, df: pd.DataFrame) -> tuple:
    """
    Fits a NearestNeighbors model on the FULL scaled dataset (all 1,500 rows).

    Design choice: KNN is a retrieval system, not a classifier being evaluated
    on held-out data, so it uses the full corpus for maximum coverage.

    Args:
        X_all : Full scaled feature matrix  [1500, 7]
        df    : Raw DataFrame (needed for case_id index)

    Returns:
        knn      : Fitted NearestNeighbors(n_neighbors=3, metric='cosine')
        case_ids : List of 1,500 case_id strings (index for lookup)
    """
    logger.info("Training KNN precedent engine  (n=3, cosine, brute) ...")
    knn = NearestNeighbors(
        n_neighbors = 3,
        metric      = "cosine",
        algorithm   = "brute",
        n_jobs      = -1,
    )
    knn.fit(X_all)
    case_ids = df["case_id"].tolist()
    logger.info("  KNN fitted on full dataset ✓")
    return knn, case_ids


# ══════════════════════════════════════════════════════════════════════════════
# 4.  EVALUATE
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_models(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluates all classifiers on the held-out test split.

    Metrics:
        - Accuracy            : Baseline correctness
        - Log Loss            : Calibration quality (lower = better-calibrated)
        - ROC-AUC (OvR macro) : Discrimination ability across all 4 classes
        - Confusion Matrix    : Per-class breakdown
        - Classification Report: Precision/Recall/F1 per severity class

    Returns:
        Nested dict: { model_name: { metric: value, ... } }
    """
    report = {}
    sep    = "=" * 62
    logger.info(f"\n{sep}\nEVALUATION  —  {len(y_test)} held-out test samples\n{sep}")

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        ll  = log_loss(y_test, y_prob)
        roc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        cm  = confusion_matrix(y_test, y_pred).tolist()
        cls = classification_report(y_test, y_pred, output_dict=True)

        report[name] = {
            "accuracy":            round(acc, 4),
            "log_loss":            round(ll,  4),
            "roc_auc_macro_ovr":   round(roc, 4),
            "confusion_matrix":    cm,
            "classification_report": cls,
        }

        logger.info(
            f"\n[ {name.upper()} ]\n"
            f"  Accuracy        : {acc:.4f}\n"
            f"  Log Loss        : {ll:.4f}\n"
            f"  ROC-AUC (macro) : {roc:.4f}\n"
            f"  Confusion Matrix:\n{np.array(cm)}"
        )

    return report


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SAVE ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════

def save_artifacts(
    models:        dict,
    knn:           NearestNeighbors,
    case_ids:      list,
    X_all:         np.ndarray,
    scaler:        StandardScaler,
    label_encoder: LabelEncoder,
    eval_report:   dict,
) -> None:
    """
    Pickles all model artifacts and metadata to models/.

    Artifacts written:
        xgboost_calibrated.pkl        — XGBoost CalibratedClassifierCV
        random_forest_calibrated.pkl  — Random Forest CalibratedClassifierCV
        lgbm_calibrated.pkl           — LightGBM CalibratedClassifierCV
        knn_precedent_model.pkl       — NearestNeighbors (retrieval)
        knn_reference_data.pkl        — {'case_ids': [...], 'X_scaled': ndarray}
        feature_scaler.pkl            — Fitted StandardScaler
        label_encoder.pkl             — Fitted LabelEncoder (counsel_type)
        feature_names.pkl             — Ordered feature name list
        evaluation_report.json        — Metrics dict (accuracy/logloss/roc-auc)
    """
    # Classifiers
    joblib.dump(models["xgboost"],       MODELS_DIR / "xgboost_calibrated.pkl")
    joblib.dump(models["random_forest"], MODELS_DIR / "random_forest_calibrated.pkl")
    joblib.dump(models["lgbm"],          MODELS_DIR / "lgbm_calibrated.pkl")
    logger.info("  Classifier .pkl files saved ✓")

    # KNN + reference corpus
    joblib.dump(knn, MODELS_DIR / "knn_precedent_model.pkl")
    joblib.dump(
        {"case_ids": case_ids, "X_scaled": X_all},
        MODELS_DIR / "knn_reference_data.pkl",
    )
    logger.info("  KNN artifacts saved ✓")

    # Pre-processing artifacts
    joblib.dump(scaler,        MODELS_DIR / "feature_scaler.pkl")
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")
    joblib.dump(FEATURE_NAMES, MODELS_DIR / "feature_names.pkl")
    logger.info("  Scaler & encoder saved ✓")

    # Evaluation JSON
    report_path = MODELS_DIR / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(eval_report, fh, indent=2)
    logger.info(f"  Evaluation report → {report_path} ✓")

    logger.info(f"\n{'-'*62}\nAll artifacts saved to: {MODELS_DIR}\n{'-'*62}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    logger.info("=" * 62)
    logger.info("LawAI  —  Phase 1 · Step 1.1 · Model Training Pipeline")
    logger.info("=" * 62)

    # ── Load & preprocess ──
    df, X_scaled, y, scaler, le = load_and_preprocess()

    # ── Stratified train/test split (85 / 15) ──
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y,
    )
    logger.info(f"Split → Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ── Train calibrated classifiers ──
    models = train_all_models(X_train, y_train)

    # ── Train KNN on FULL dataset ──
    knn, case_ids = train_knn(X_scaled, df)

    # ── Evaluate on held-out test set ──
    eval_report = evaluate_models(models, X_test, y_test)

    # ── Persist everything ──
    save_artifacts(models, knn, case_ids, X_scaled, scaler, le, eval_report)

    logger.info("Phase 1 · Step 1.1  COMPLETE ✓")


if __name__ == "__main__":
    main()
