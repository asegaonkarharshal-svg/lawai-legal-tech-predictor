"""
src/pdf_generator.py
--------------------
Phase 4 - Step 4.1: ReportLab PDF Template Engine

Generates a formal Legal Case Evaluation Report in-memory using ReportLab.
No disk I/O — outputs raw PDF bytes via io.BytesIO.

Report sections:
  1. Header Block            — Case ID, timestamp, disclaimer notice
  2. Input Feature Matrix    — 7 case features + IPC/BNS statutory mapping
  3. Verdict Summary         — Color-coded severity, confidence, offense metadata
  4. Probability Chart       — Matplotlib bar chart embedded as PNG
  5. SHAP Risk Factor Tables — Aggravating (red) vs Mitigating (green)
  6. Top 3 Precedent Cases   — Similarity, BNS section, key case facts
  7. Footer                  — Page numbers, LawAI version

Usage:
    from src.pdf_generator import generate_case_report
    pdf_bytes = generate_case_report(case_features, prediction, explanation, similar_cases)
"""

import io
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — required in server contexts
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from reportlab.lib import colors
from reportlab.lib.colors import HexColor, white, black, Color
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Table,
    TableStyle,
    Spacer,
    Image,
    HRFlowable,
    KeepTogether,
)
from reportlab.platypus.flowables import HRFlowable

# ── Design Palette ────────────────────────────────────────────────────────────
C_NAVY        = HexColor("#2C3E50")   # header, section titles
C_BLUE        = HexColor("#2980B9")   # table headers
C_BLUE_LIGHT  = HexColor("#D6EAF8")   # alternate row
C_GRAY_LIGHT  = HexColor("#F2F3F4")   # alternate row
C_WHITE       = white
C_BLACK       = black
C_RED_LIGHT   = HexColor("#FADBD8")   # aggravating highlight
C_GREEN_LIGHT = HexColor("#D5F5E3")   # mitigating highlight
C_BORDER      = HexColor("#BDC3C7")   # grid borders

SEVERITY_HEX = {
    0: "#2ECC71",   # green
    1: "#F39C12",   # amber
    2: "#E67E22",   # orange
    3: "#E74C3C",   # red
}
SEVERITY_LABELS = {
    0: "Acquittal / Minimal",
    1: "Light Sentence",
    2: "Moderate Sentence",
    3: "Severe Sentence",
}

PAGE_W, PAGE_H = A4
LEFT_MARGIN = RIGHT_MARGIN = 2.0 * cm
TOP_MARGIN  = BOTTOM_MARGIN = 2.0 * cm
CONTENT_W   = PAGE_W - LEFT_MARGIN - RIGHT_MARGIN


# ── Style Registry ────────────────────────────────────────────────────────────

def _build_styles() -> dict:
    base = getSampleStyleSheet()
    styles = {}

    def add(name, parent_name="Normal", **kwargs):
        styles[name] = ParagraphStyle(name=name, parent=base[parent_name], **kwargs)

    add("ReportTitle",  "Title",  fontSize=20, textColor=C_NAVY,
        alignment=TA_CENTER, spaceAfter=4)
    add("ReportSubtitle", "Normal", fontSize=10, textColor=C_NAVY,
        alignment=TA_CENTER, spaceAfter=2)
    add("SectionHeader", "Heading2", fontSize=12, textColor=C_WHITE,
        backColor=C_NAVY, spaceBefore=14, spaceAfter=4,
        leftIndent=-4, rightIndent=-4)
    add("BodyText",    "Normal",  fontSize=8.5, spaceAfter=3, leading=13)
    add("SmallText",   "Normal",  fontSize=7.5, textColor=HexColor("#666666"), spaceAfter=2)
    add("Disclaimer",  "Normal",  fontSize=7.0, textColor=HexColor("#888888"),
        alignment=TA_JUSTIFY, spaceAfter=2)
    add("TableHeader", "Normal",  fontSize=8.5, textColor=C_WHITE, alignment=TA_CENTER,
        fontName="Helvetica-Bold")
    add("TableCell",   "Normal",  fontSize=8.0, alignment=TA_LEFT, leading=11)
    add("TableCellC",  "Normal",  fontSize=8.0, alignment=TA_CENTER, leading=11)
    add("VerdictText", "Normal",  fontSize=14, textColor=C_WHITE,
        alignment=TA_CENTER, fontName="Helvetica-Bold", leading=20)
    add("VerdictSub",  "Normal",  fontSize=9.0, textColor=C_WHITE,
        alignment=TA_CENTER, leading=13)
    add("FooterText",  "Normal",  fontSize=7.0, textColor=HexColor("#AAAAAA"),
        alignment=TA_CENTER)

    return styles


# ── Chart Builder (matplotlib → PNG bytes) ────────────────────────────────────

def _probability_chart_png(ensemble_probs: dict, predicted_severity: int) -> bytes:
    """Renders a horizontal bar chart of 4-class probabilities, returns PNG bytes."""
    labels = [f"Sev {i}: {SEVERITY_LABELS[i]}" for i in range(4)]
    values = [ensemble_probs.get(f"severity_{i}", 0) * 100 for i in range(4)]
    bar_colors = [
        SEVERITY_HEX[i] if i == predicted_severity else "#BDC3C7"
        for i in range(4)
    ]

    fig, ax = plt.subplots(figsize=(7.5, 2.4))
    bars = ax.barh(labels, values, color=bar_colors, edgecolor="white",
                   height=0.55)

    # Annotate bars
    for bar, val in zip(bars, values):
        ax.text(val + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left",
                fontsize=9, color="#333333")

    ax.set_xlabel("Probability (%)", fontsize=9)
    ax.set_xlim(0, max(values) * 1.25 + 2)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("#FAFAFA")
    plt.tight_layout(pad=0.4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── Table Style Helpers ───────────────────────────────────────────────────────

def _base_table_style(header_bg=C_BLUE) -> TableStyle:
    return TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),   header_bg),
        ("TEXTCOLOR",    (0, 0), (-1, 0),   C_WHITE),
        ("FONTNAME",     (0, 0), (-1, 0),   "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),   8.5),
        ("FONTNAME",     (0, 1), (-1, -1),  "Helvetica"),
        ("FONTSIZE",     (0, 1), (-1, -1),  8.0),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_GRAY_LIGHT]),
        ("GRID",         (0, 0), (-1, -1),  0.4, C_BORDER),
        ("ALIGN",        (0, 0), (-1, -1),  "LEFT"),
        ("VALIGN",       (0, 0), (-1, -1),  "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1),  6),
        ("RIGHTPADDING", (0, 0), (-1, -1),  6),
        ("TOPPADDING",   (0, 0), (-1, -1),  4),
        ("BOTTOMPADDING",(0, 0), (-1, -1),  4),
    ])


def _section_header(title: str, styles: dict) -> list:
    """Returns spacer + blue banner paragraph for section heading."""
    return [
        Spacer(1, 0.25 * cm),
        HRFlowable(width="100%", thickness=0.5, color=C_NAVY, spaceAfter=4),
        Paragraph(f"  {title}", styles["SectionHeader"]),
        Spacer(1, 0.15 * cm),
    ]


# ── Page Numbering ────────────────────────────────────────────────────────────

class _NumberedCanvas:
    """Adds page numbers and footer to each page via canvas callback."""
    def __init__(self, canvas, doc):
        self._canvas = canvas
        self._doc    = doc

    def afterPage(self):
        c = self._canvas
        c.saveState()
        c.setFont("Helvetica", 7)
        c.setFillColor(HexColor("#AAAAAA"))
        page_num = f"Page {c.getPageNumber()}"
        c.drawString(LEFT_MARGIN, 12 * mm, "LawAI Legal Tech Predictor v1.0  |  "
                     "For educational use only. Not legal advice.")
        c.drawRightString(PAGE_W - RIGHT_MARGIN, 12 * mm, page_num)
        c.restoreState()


def _on_first_page(canvas, doc):
    _NumberedCanvas(canvas, doc).afterPage()


def _on_later_pages(canvas, doc):
    _NumberedCanvas(canvas, doc).afterPage()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _section_1_header(case_id: str, styles: dict) -> list:
    """Section 1: Title block, case metadata, generated timestamp."""
    ts = datetime.now().strftime("%d %B %Y, %H:%M IST")
    elements = [
        Paragraph("⚖  LAWAI LEGAL TECH PREDICTOR", styles["ReportTitle"]),
        Paragraph("Formal Case Evaluation Report", styles["ReportSubtitle"]),
        Spacer(1, 0.1 * cm),
        HRFlowable(width="100%", thickness=2, color=C_NAVY, spaceAfter=6),
        Paragraph(f"<b>Case Reference:</b> {case_id} &nbsp;&nbsp;&nbsp; "
                  f"<b>Generated:</b> {ts}", styles["SmallText"]),
        Spacer(1, 0.15 * cm),
        Paragraph(
            "<b>Legal Disclaimer:</b> This report is generated by an AI system "
            "for research and educational purposes only. Predictions are based on "
            "historical data patterns and do not constitute legal advice. "
            "All outputs must be reviewed by a qualified legal professional before "
            "any judicial or advisory use.",
            styles["Disclaimer"],
        ),
        Spacer(1, 0.2 * cm),
    ]
    return elements


def _section_2_features(case_features: dict, statutory_info: dict,
                         styles: dict) -> list:
    """Section 2: 7-feature input matrix + IPC/BNS statutory mapping table."""
    elements = _section_header("2. INPUT FEATURE MATRIX & STATUTORY MAPPING", styles)

    # Feature display names and formatted values
    feat_rows = [
        [Paragraph("<b>Feature</b>",          styles["TableHeader"]),
         Paragraph("<b>Value</b>",             styles["TableHeader"]),
         Paragraph("<b>Notes</b>",             styles["TableHeader"])],
        ["Stolen Value (INR)",
         f"₹ {case_features.get('stolen_value_inr', 0):,.2f}",
         "log1p-transformed before model inference"],
        ["Counsel Type",
         case_features.get("counsel_type", "—"),
         "Legal representation category"],
        ["Weapon Present",
         "Yes 🔫" if case_features.get("weapon_present") else "No",
         "Dangerous weapon used or found at scene"],
        ["Forced Entry",
         "Yes 🚪" if case_features.get("forced_entry") else "No",
         "Dwelling house / lock breaking detected"],
        ["Prior Convictions",
         str(case_features.get("prior_convictions", 0)),
         "Number of past criminal offenses"],
        ["Property Recovered",
         f"{case_features.get('property_recovered_pct', 0):.1f}%",
         "Percentage of stolen items recovered"],
        ["Confession Recorded",
         "Yes 📝" if case_features.get("confession_recorded") else "No",
         "Formal confession on record"],
    ]

    feat_table = Table(feat_rows, colWidths=[4.5 * cm, 4.0 * cm, None])
    ts = _base_table_style()
    ts.add("ALIGN", (1, 1), (1, -1), "CENTER")
    feat_table.setStyle(ts)
    elements.append(feat_table)
    elements.append(Spacer(1, 0.3 * cm))

    # Statutory Mapping sub-table
    stat_rows = [
        [Paragraph("<b>Law</b>",          styles["TableHeader"]),
         Paragraph("<b>Section</b>",       styles["TableHeader"]),
         Paragraph("<b>Offense Type</b>",  styles["TableHeader"]),
         Paragraph("<b>Max Punishment</b>",styles["TableHeader"]),
         Paragraph("<b>Bailable</b>",      styles["TableHeader"])],
        ["IPC (1860)",
         statutory_info.get("ipc_section", "—"),
         statutory_info.get("offense_type", "—"),
         statutory_info.get("max_punishment_ipc", "—"),
         "Yes" if statutory_info.get("is_bailable") else "No"],
        ["BNS (2023)",
         statutory_info.get("bns_section", "—"),
         statutory_info.get("offense_type", "—"),
         statutory_info.get("max_punishment_bns", "—"),
         "Yes" if statutory_info.get("is_bailable") else "No"],
    ]
    stat_table = Table(stat_rows, colWidths=[2.5*cm, 4.0*cm, 3.0*cm, None, 2.0*cm])
    stat_ts = _base_table_style(header_bg=HexColor("#1A5276"))
    stat_table.setStyle(stat_ts)
    elements.append(Paragraph("<b>Statutory Mapping (IPC ↔ BNS)</b>", styles["BodyText"]))
    elements.append(stat_table)

    return elements


def _section_3_verdict(prediction: dict, styles: dict) -> list:
    """Section 3: Color-coded verdict summary box."""
    elements = _section_header("3. VERDICT SUMMARY", styles)

    sev      = prediction["predicted_severity"]
    sev_hex  = SEVERITY_HEX[sev]
    sev_col  = HexColor(sev_hex)
    stat     = prediction["statutory_info"]

    # Verdict box implemented as a single-cell colored Table
    verdict_para = Paragraph(
        f"<b>SEVERITY CLASS {sev}: {SEVERITY_LABELS[sev].upper()}</b>",
        styles["VerdictText"],
    )
    sub_para = Paragraph(
        f"Ensemble Confidence: <b>{prediction['confidence']*100:.1f}%</b> &nbsp;|&nbsp; "
        f"Section: <b>{stat['bns_section']}</b> &nbsp;|&nbsp; "
        f"{'Non-Bailable ❌' if not stat['is_bailable'] else 'Bailable ✅'}",
        styles["VerdictSub"],
    )

    verdict_table = Table([[verdict_para], [sub_para]],
                          colWidths=[CONTENT_W])
    verdict_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), sev_col),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 14),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 14),
        ("ROUNDEDCORNERS", [6]),
    ]))
    elements.append(verdict_table)
    elements.append(Spacer(1, 0.25 * cm))

    # Max punishment description
    elements.append(Paragraph(
        f"<b>Maximum Punishment (BNS):</b> {stat['max_punishment_bns']}",
        styles["BodyText"],
    ))
    elements.append(Paragraph(
        f"<b>Statutory Description:</b> {stat['bns_description'][:220]}...",
        styles["SmallText"],
    ))

    # Per-model breakdown table
    elements.append(Spacer(1, 0.2 * cm))
    model_rows = [
        [Paragraph("<b>Model</b>",             styles["TableHeader"]),
         Paragraph("<b>Predicted Severity</b>", styles["TableHeader"]),
         Paragraph("<b>Confidence</b>",          styles["TableHeader"])],
    ]
    for mv in prediction.get("model_breakdown", []):
        model_rows.append([
            mv["model_name"],
            f"{mv['predicted_severity']} — {SEVERITY_LABELS[mv['predicted_severity']]}",
            f"{mv['confidence']*100:.1f}%",
        ])
    model_table = Table(model_rows, colWidths=[6.0*cm, 6.5*cm, None])
    model_table.setStyle(_base_table_style())
    elements.append(Paragraph("<b>Individual Model Breakdown</b>", styles["BodyText"]))
    elements.append(model_table)

    return elements


def _section_4_probabilities(prediction: dict, styles: dict) -> list:
    """Section 4: Probability bar chart (matplotlib PNG) + raw probability table."""
    elements = _section_header("4. MULTI-CLASS PROBABILITY OUTCOMES", styles)

    # Generate and embed chart
    chart_png = _probability_chart_png(
        prediction["ensemble_probabilities"],
        prediction["predicted_severity"],
    )
    img = Image(io.BytesIO(chart_png), width=CONTENT_W, height=6.5 * cm)
    elements.append(img)
    elements.append(Spacer(1, 0.2 * cm))

    # Raw probability table
    probs = prediction["ensemble_probabilities"]
    prob_rows = [
        [Paragraph("<b>Severity Class</b>", styles["TableHeader"]),
         Paragraph("<b>Label</b>",           styles["TableHeader"]),
         Paragraph("<b>Ensemble Prob.</b>",  styles["TableHeader"]),
         Paragraph("<b>Status</b>",          styles["TableHeader"])],
    ]
    for i in range(4):
        pval  = probs.get(f"severity_{i}", 0)
        is_predicted = (i == prediction["predicted_severity"])
        prob_rows.append([
            f"Class {i}",
            SEVERITY_LABELS[i],
            f"{pval*100:.1f}%",
            "◀ PREDICTED" if is_predicted else "",
        ])

    prob_table = Table(prob_rows, colWidths=[2.5*cm, 5.5*cm, 3.5*cm, None])
    pts = _base_table_style()
    # Highlight predicted row
    pred_row_idx = prediction["predicted_severity"] + 1  # +1 for header
    pts.add("BACKGROUND", (0, pred_row_idx), (-1, pred_row_idx),
            HexColor(SEVERITY_HEX[prediction["predicted_severity"]] + "55"))
    pts.add("FONTNAME", (0, pred_row_idx), (-1, pred_row_idx), "Helvetica-Bold")
    prob_table.setStyle(pts)
    elements.append(prob_table)

    return elements


def _section_5_shap(explanation: dict, styles: dict) -> list:
    """Section 5: Aggravating vs Mitigating risk factor tables from SHAP values."""
    elements = _section_header("5. AGGRAVATING & MITIGATING RISK FACTORS (SHAP)", styles)

    elements.append(Paragraph(
        f"SHAP base value (E[f(X)]): <b>{explanation.get('shap_base_value', 0):.4f}</b> &nbsp;|&nbsp; "
        f"Explanation model: {explanation.get('explanation_model', 'XGBoost')}",
        styles["SmallText"],
    ))
    elements.append(Spacer(1, 0.15 * cm))

    contribs = sorted(explanation.get("shap_contributions", []),
                      key=lambda x: abs(x["shap_value"]), reverse=True)

    aggravating = [c for c in contribs if c["shap_value"] >= 0][:4]
    mitigating  = [c for c in contribs if c["shap_value"] <  0][:4]

    def _shap_table(rows_data, bg_color, header_bg):
        header = [
            Paragraph("<b>Rank</b>",    styles["TableHeader"]),
            Paragraph("<b>Feature</b>", styles["TableHeader"]),
            Paragraph("<b>Value</b>",   styles["TableHeader"]),
            Paragraph("<b>SHAP</b>",    styles["TableHeader"]),
        ]
        rows = [header]
        for c in rows_data:
            fname = c["feature_name"].replace("_enc", "").replace("_", " ").title()
            val   = c["unscaled_value"]
            fmt   = f"₹{val:,.0f}" if "inr" in c["feature_name"] else (
                    f"{val:.1f}%" if "pct" in c["feature_name"] else f"{int(val)}")
            rows.append([
                str(c["rank"]),
                fname,
                fmt,
                f"{c['shap_value']:+.4f}",
            ])
        tbl = Table(rows, colWidths=[1.0*cm, 3.8*cm, 2.5*cm, 2.2*cm])
        ts  = _base_table_style(header_bg=header_bg)
        ts.add("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, bg_color])
        tbl.setStyle(ts)
        return tbl

    agg_tbl = _shap_table(aggravating, C_RED_LIGHT,   HexColor("#922B21"))
    mit_tbl = _shap_table(mitigating,  C_GREEN_LIGHT, HexColor("#1D6A39"))

    # Place side-by-side in outer table
    half = (CONTENT_W - 0.4 * cm) / 2
    side_by_side = Table(
        [[Paragraph("<b>🔴 Aggravating Factors</b>", styles["BodyText"]),
          Paragraph("<b>🟢 Mitigating Factors</b>",  styles["BodyText"])],
         [agg_tbl, mit_tbl]],
        colWidths=[half, half],
        spaceBefore=0,
    )
    side_by_side.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(side_by_side)

    # LIME cards (top 3 as bullet points)
    lime_cards = explanation.get("lime_cards", [])
    if lime_cards:
        elements.append(Spacer(1, 0.3 * cm))
        elements.append(Paragraph("<b>LIME Local Interpretation (Top Factors):</b>",
                                  styles["BodyText"]))
        for card in lime_cards[:3]:
            icon = "📈" if card["direction"] == "positive" else "📉"
            elements.append(Paragraph(
                f"{icon} <b>{card['condition']}</b> (weight: {card['weight']:+.3f}) — "
                f"{card['natural_language']}",
                styles["SmallText"],
            ))

    return elements


def _section_6_precedents(similar_cases: dict, styles: dict) -> list:
    """Section 6: Top 3 K-NN matched precedent cases."""
    elements = _section_header("6. TOP MATCHING HISTORICAL PRECEDENTS (K-NN)", styles)

    precedents = similar_cases.get("precedents", [])
    if not precedents:
        elements.append(Paragraph("No precedent data available.", styles["BodyText"]))
        return elements

    for i, prec in enumerate(precedents, 1):
        sev     = prec["sentence_severity"]
        sev_hex = SEVERITY_HEX.get(sev, "#BDC3C7")

        prec_rows = [
            [Paragraph(f"<b>Precedent #{i}: {prec['case_id']}</b>",
                       styles["TableHeader"]),
             Paragraph(f"Similarity: <b>{prec['similarity_score']*100:.1f}%</b>",
                       styles["TableHeader"])],
            ["BNS Section",   prec["bns_section"]],
            ["IPC Section",   prec["ipc_section"]],
            ["Severity",
             f"Class {sev} — {SEVERITY_LABELS.get(sev, '?')}"],
            ["Stolen Value",  f"₹ {prec['stolen_value_inr']:,.0f}"],
            ["Weapon / Forced Entry",
             f"{'Yes' if prec['weapon_present'] else 'No'} / "
             f"{'Yes' if prec['forced_entry'] else 'No'}"],
            ["Recovery",      f"{prec['property_recovered_pct']:.1f}%"],
            ["Prior Convictions", str(prec["prior_convictions"])],
        ]

        prec_tbl = Table(prec_rows, colWidths=[4.5 * cm, None])
        pts = TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),   HexColor(sev_hex)),
            ("TEXTCOLOR",     (0, 0), (-1, 0),   C_WHITE),
            ("FONTNAME",      (0, 0), (-1, 0),   "Helvetica-Bold"),
            ("FONTNAME",      (0, 1), (0, -1),   "Helvetica-Bold"),
            ("FONTNAME",      (1, 1), (1, -1),   "Helvetica"),
            ("FONTSIZE",      (0, 0), (-1, -1),  8.0),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1),  [C_WHITE, C_GRAY_LIGHT]),
            ("GRID",          (0, 0), (-1, -1),  0.4, C_BORDER),
            ("LEFTPADDING",   (0, 0), (-1, -1),  6),
            ("RIGHTPADDING",  (0, 0), (-1, -1),  6),
            ("TOPPADDING",    (0, 0), (-1, -1),  4),
            ("BOTTOMPADDING", (0, 0), (-1, -1),  4),
            ("VALIGN",        (0, 0), (-1, -1),  "MIDDLE"),
        ])
        prec_tbl.setStyle(pts)
        elements.append(prec_tbl)
        elements.append(Spacer(1, 0.2 * cm))

    return elements


def _section_7_footer(styles: dict) -> list:
    """Section 7: Closing disclaimer block."""
    elements = [
        Spacer(1, 0.4 * cm),
        HRFlowable(width="100%", thickness=1, color=C_NAVY),
        Spacer(1, 0.15 * cm),
        Paragraph(
            "This report was automatically generated by <b>LawAI Legal Tech Predictor v1.0</b>. "
            "AI-generated content should always be reviewed by a qualified legal professional. "
            "The prediction models are trained on historical criminal theft case data and may "
            "not account for all relevant legal, factual, or procedural considerations. "
            "Statutory references reflect Indian criminal law under both the Indian Penal Code "
            "(IPC, 1860) and Bharatiya Nyaya Sanhita (BNS, 2023 — effective 1 July 2024).",
            styles["Disclaimer"],
        ),
    ]
    return elements


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def generate_case_report(
    case_features:  dict,
    prediction:     dict,
    explanation:    Optional[dict] = None,
    similar_cases:  Optional[dict] = None,
    report_title:   str = "Legal Case Evaluation Report",
) -> bytes:
    """
    Generates a formal Legal Case Evaluation PDF report in-memory.

    Args:
        case_features : CaseInputSchema dict (7 feature fields).
        prediction    : PredictionResponse dict from /api/v1/predict.
        explanation   : ExplanationResponse dict from /api/v1/explain (optional).
        similar_cases : SimilarCasesResponse dict from /api/v1/similar-cases (optional).
        report_title  : PDF document title string.

    Returns:
        Raw PDF bytes — pass directly to st.download_button(data=...).

    Raises:
        ValueError: if prediction dict is None or missing required keys.
    """
    if not prediction or "predicted_severity" not in prediction:
        raise ValueError("prediction dict is required and must contain 'predicted_severity'.")

    # Auto case ID
    ts_str  = datetime.now().strftime("%Y%m%d-%H%M%S")
    case_id = f"LAWAI-{ts_str}"

    styles = _build_styles()
    buf    = io.BytesIO()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=LEFT_MARGIN, rightMargin=RIGHT_MARGIN,
        topMargin=TOP_MARGIN,   bottomMargin=BOTTOM_MARGIN + 1.2 * cm,
        title=report_title,
        author="LawAI Legal Tech Predictor v1.0",
        subject="Criminal Theft Case Evaluation",
    )

    # Assemble all sections
    story = []
    story += _section_1_header(case_id, styles)
    story += _section_2_features(case_features,
                                  prediction["statutory_info"], styles)
    story += _section_3_verdict(prediction, styles)
    story += _section_4_probabilities(prediction, styles)

    if explanation:
        story += _section_5_shap(explanation, styles)

    if similar_cases:
        story += _section_6_precedents(similar_cases, styles)

    story += _section_7_footer(styles)

    doc.build(
        story,
        onFirstPage=_on_first_page,
        onLaterPages=_on_later_pages,
    )

    buf.seek(0)
    return buf.read()
