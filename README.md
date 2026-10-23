# ⚖️ LawAI — Legal Tech Predictor

An Explainable AI (XAI) and Machine Learning platform for Indian criminal theft case assessment under both the **Indian Penal Code (IPC, 1860)** and the **Bharatiya Nyaya Sanhita (BNS, 2023)**.

---

## 🌟 Key Features

- **NLP Case Triage & Extraction**: Auto-extracts legal features (stolen valuation, weapon presence, dwelling entry, recovery %, priors, confession) from raw FIR narratives using a hybrid Regex + spaCy NER parser.
- **Calibrated Multi-Class Severity Prediction**: Classifies cases into 4 severity tiers (Acquittal/Minimal, Light Sentence, Moderate Sentence, Severe Sentence) using soft-voting ensemble across calibrated XGBoost, Random Forest, and LightGBM models.
- **Statutory Cross-Mapping**: Deterministic mapping between legacy IPC sections (379 Simple, 379 Aggravated, 380 Dwelling) and modern BNS sections (303(1), 303(2), 305).
- **Explainable AI (XAI)**:
  - **SHAP Waterfall Plots**: Quantitative positive/negative attribution per feature.
  - **LIME Natural Language Cards**: Human-interpretable legal explanations.
- **Precedent Case Matcher**: K-NN cosine-similarity retrieval of top 3 matching historical court precedents.
- **"What-If" Sensitivity Simulator**: Interactive baseline vs. modified scenario comparator with probability shift deltas and radar charts.
- **In-Memory PDF Report Generator**: Generates formal, publication-ready Case Evaluation reports with embedded charts via ReportLab in zero-disk memory buffer.
- **Dual-Engine Architecture**: Operates both as a decoupled FastAPI microservice + Streamlit frontend AND as a standalone in-process app on Streamlit Community Cloud.

---

## 🚀 Live Demo & Quickstart

### Streamlit Community Cloud (Standalone Mode)
Deploy directly on Streamlit Community Cloud by pointing to `streamlit_app.py`. The app automatically activates its embedded in-process ML engine.

### Local Development

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. **Run FastAPI Backend**:
```bash
uvicorn backend.main:app --reload --port 8000
```

3. **Run Streamlit Frontend**:
```bash
streamlit run frontend/app.py
```

---

## 🏛️ Project Structure

```
├── backend/
│   ├── main.py              # FastAPI REST API endpoints
│   ├── schemas.py           # Pydantic v2 data models
│   └── services.py          # Model inference, SHAP/LIME, KNN handlers
├── data/
│   └── criminal_theft_dataset_v2_1500.csv
├── frontend/
│   └── app.py               # 4-tab Streamlit web application
├── models/                  # Calibrated models, scalers, eval report
├── src/
│   ├── fir_parser.py        # spaCy + Regex NLP parser
│   ├── pdf_generator.py     # In-memory ReportLab PDF engine
│   ├── statutory_mapping.py # IPC ↔ BNS statutory rules
│   └── train_models.py      # ML training pipeline
├── .streamlit/
│   └── config.toml          # Streamlit UI theme and server settings
├── streamlit_app.py         # Root entrypoint for Streamlit Cloud
└── requirements.txt
```

---

## ⚖️ Disclaimer

*LawAI is designed for research, academic, and decision-support purposes only. It does not constitute formal legal counsel. All judicial interpretations should be evaluated by qualified legal practitioners.*
