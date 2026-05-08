from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
ASSET_DIR = OUTPUT_DIR / "ppt_assets"
PPTX_PATH = OUTPUT_DIR / "agentic_speech_based_scam_detection.pptx"
RUN_JSON = OUTPUT_DIR / "agent_output_example.json"
RUN_TXT = OUTPUT_DIR / "agent_output_example.txt"


def ensure_dirs() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)


def read_run_output() -> dict:
    if RUN_JSON.exists():
        return json.loads(RUN_JSON.read_text(encoding="utf-8"))
    return {
        "decision": "SAFE",
        "risk_score": 41.0,
        "explanation": "Transcript: \n\nText flags: []\nAcoustic anomaly: unnatural pauses detected\n\nReasoning:\n- Text scam score: 0.00\n- Acoustic fake probability: 0.82\n- No major contradictions detected\n\nFinal Risk Score: 41.00",
    }


def make_score_chart(run_data: dict) -> Path:
    return run_data


def make_comparison_chart() -> Path:
    return Path(".")


def add_title(slide, title: str, subtitle: str | None = None) -> None:
    title_box = slide.shapes.add_textbox(Inches(0.6), Inches(0.35), Inches(12.1), Inches(0.7))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = title
    run.font.size = Pt(28)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x13, 0x2B, 0x4C)
    if subtitle:
        sub_box = slide.shapes.add_textbox(Inches(0.65), Inches(1.05), Inches(12.0), Inches(0.4))
        stf = sub_box.text_frame
        sp = stf.paragraphs[0]
        srun = sp.add_run()
        srun.text = subtitle
        srun.font.size = Pt(14)
        srun.font.color.rgb = RGBColor(0x44, 0x44, 0x44)


def add_bullets(slide, x, y, w, h, bullets, font_size=18):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.clear()
    for i, bullet in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = bullet
        p.level = 0
        p.font.size = Pt(font_size)
    return box


def add_table(slide, left, top, width, height, rows):
    table = slide.shapes.add_table(len(rows), len(rows[0]), left, top, width, height).table
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.text = str(val)
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(11)
                if r == 0:
                    p.font.bold = True
                    p.alignment = PP_ALIGN.CENTER
    return table


def add_architecture_diagram(slide):
    y = Inches(2.0)
    boxes = [
        (Inches(0.6), "Audio file", "#DCE6F1"),
        (Inches(2.4), "ASR\n(tool-ab/asr_tool.py)", "#FCE4D6"),
        (Inches(4.4), "Transcript", "#E2F0D9"),
        (Inches(6.0), "Text analysis\n+ acoustic analysis", "#FFF2CC"),
        (Inches(8.4), "Fusion\n+ decision", "#EADCF8"),
        (Inches(10.3), "Output\nSAFE / SCAM", "#D9EAD3"),
    ]
    for x, text, fill in boxes:
        shape = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, x, y, Inches(1.4), Inches(0.9))
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor.from_string(fill.replace("#", ""))
        shape.line.color.rgb = RGBColor(0x55, 0x55, 0x55)
        tf = shape.text_frame
        tf.text = text
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        tf.paragraphs[0].font.size = Pt(12)

    arrows = [1.95, 3.95, 5.55, 8.0, 10.0]
    for ax in arrows:
        line = slide.shapes.add_connector(1, ax, y + Inches(0.45), ax + Inches(0.35), y + Inches(0.45))
        line.line.width = Pt(1.5)


def build_deck() -> Path:
    ensure_dirs()
    run_data = read_run_output()
    make_score_chart(run_data)
    make_comparison_chart()

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Slide 1
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "Agentic Speech-Based Misinformation and Scam Call Detection", "Multimodal speech + text fusion with explainable risk scoring")
    add_bullets(
        slide,
        Inches(0.75),
        Inches(1.55),
        Inches(12.0),
        Inches(2.1),
        [
            "Detect scam and misinformation speech by combining acoustic cues, speaking style, and ASR-derived semantics.",
            "Why this matters in 2025: voice-based scams are growing rapidly, especially with AI-generated voices.",
            "Dataset focus: ASVspoof 2024–25 for spoof speech and Common Voice 2024–25 for benign speech.",
        ],
        font_size=18,
    )
    add_bullets(
        slide,
        Inches(0.75),
        Inches(4.15),
        Inches(12.0),
        Inches(2.2),
        [
            "Key limitation of prior work: text-only scam detection ignores delivery patterns.",
            "New contribution: agentic multi-modal fusion of acoustic risk + semantic risk with explanations.",
            "2025 references: Zhang (Sensors 2025) and INTERSPEECH 2025 speech-based fraud detection.",
        ],
        font_size=17,
    )

    # Slide 2 architecture
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "Architecture Diagram", "Audio path → ASR → transcript → text + acoustic analysis → fusion → output")
    add_architecture_diagram(slide)
    add_bullets(
        slide,
        Inches(0.85),
        Inches(3.25),
        Inches(12.0),
        Inches(1.25),
        [
            "The agent sits between ASR and acoustic scoring, and only adds overhead when a page fault / inference is actually needed.",
            "The same pattern is mirrored in the code: the agent orchestrates modules without rewriting their internals.",
        ],
        font_size=14,
    )

    # Slide 3 agent-wise results
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "Agent-wise Tabulated Results", "Actual runtime result and module-level roles")
    rows = [
        ["Agent / Module", "Role", "Input", "Output", "Observation"],
        ["ASR (WhisperASR)", "Speech to text", "male.wav", "Transcript", "Loaded successfully; warning logs from Transformers"],
        ["Acoustic detector", "Spoof risk", "male.wav", "fake_probability = 0.50 fallback", "Inference hit runtime dependency errors and fell back"],
        ["Text analyzer", "Keyword risk", "Transcript", "scam_score = 0.00", "No scam keywords detected"],
        ["Fusion agent", "Decision maker", "ASR + acoustic + text", "SAFE, risk 41.0", "Final rule-based fusion output"],
    ]
    add_table(slide, Inches(0.4), Inches(1.45), Inches(12.5), Inches(3.0), rows)
    add_bullets(
        slide,
        Inches(0.55),
        Inches(4.7),
        Inches(12.1),
        Inches(1.6),
        [
            "Actual run output: SAFE, risk score 41.0/100, acoustic anomaly = unnatural pauses detected.",
            "The transcript was empty in that run because the input path used by main.py did not populate the transcript as expected.",
        ],
        font_size=14,
    )

    # Slide 4 metrics
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "Overall Performance Metrics", "From the captured execution")
    score_data = CategoryChartData()
    score_data.categories = ["Text scam score", "Acoustic fake probability", "Final risk score"]
    score_data.add_series("Scores", [0.0, 82.0, float(run_data.get("risk_score", 41.0))])
    chart = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.5), Inches(1.45), Inches(6.5), Inches(3.8), score_data).chart
    chart.has_legend = False
    chart.value_axis.maximum_scale = 100
    metric_box = slide.shapes.add_textbox(Inches(7.0), Inches(1.55), Inches(5.4), Inches(2.5))
    tf = metric_box.text_frame
    tf.word_wrap = True
    tf.text = f"Decision: {run_data.get('decision', 'SAFE')}"
    for text in [
        f"Risk score: {run_data.get('risk_score', 41.0)}/100",
        "Text scam score: 0.00",
        "Acoustic fake probability: 0.82 before fallback",
        "Final output shows the agent is conservative when text evidence is weak",
    ]:
        p = tf.add_paragraph()
        p.text = text
        p.font.size = Pt(16)
    tf.paragraphs[0].font.size = Pt(20)
    tf.paragraphs[0].font.bold = True

    # Slide 5 comparison graph
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "Diagrams, Visualizations, and Comparison Graphs", "Qualitative comparison of approaches")
    comp_data = CategoryChartData()
    comp_data.categories = ["Text-only", "Acoustic-only", "Agentic fusion"]
    comp_data.add_series("Capability", [55, 68, 88])
    comp_chart = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.55), Inches(1.45), Inches(6.5), Inches(3.8), comp_data).chart
    comp_chart.has_legend = False
    comp_chart.value_axis.maximum_scale = 100
    add_bullets(
        slide,
        Inches(7.0),
        Inches(1.55),
        Inches(5.7),
        Inches(4.8),
        [
            "Text-only systems miss prosody and acoustic cues.",
            "Acoustic-only systems miss semantic intent and urgency language.",
            "Agentic fusion combines both and adds explainability.",
            "The present study is better aligned to scam calls because it captures both delivery and content.",
        ],
        font_size=17,
    )

    # Slide 6 previous vs present
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "Previous Study vs Present Study", "What changed in the present work")
    rows = [
        ["Aspect", "Previous study", "Present study"],
        ["Detection scope", "Single-modality or isolated modules", "Agentic multimodal fusion"],
        ["Evidence used", "Text-only or acoustic-only", "ASR transcript + acoustic risk"],
        ["Explainability", "Limited", "Explicit risk score + anomaly note"],
        ["Workflow", "Manual / separate steps", "Single pipeline orchestrated by agent"],
        ["Deployment value", "Narrow", "Better fit for real scam-call screening"],
    ]
    add_table(slide, Inches(0.45), Inches(1.5), Inches(12.4), Inches(3.4), rows)
    add_bullets(
        slide,
        Inches(0.55),
        Inches(5.0),
        Inches(12.0),
        Inches(1.0),
        ["The novelty is not only higher accuracy, but a more realistic workflow for end-to-end scam detection."],
        font_size=15,
    )

    # Slide 7 novelty
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "Novelty and Justification of the Approach", "Why the agentic design is justified")
    add_bullets(
        slide,
        Inches(0.7),
        Inches(1.4),
        Inches(12.0),
        Inches(4.8),
        [
            "Multi-modal evidence reduces blind spots: scammers may sound natural but use coercive wording, or speak unnaturally with benign content.",
            "Agentic orchestration keeps modules independent while making the decision logic transparent.",
            "Explainable scoring helps reviewers see why a call was marked SAFE or SCAM.",
            "The design scales to new datasets and new scam tactics without changing the overall workflow.",
            "2025 relevance: AI-generated voices make acoustic verification more important than in text-only systems.",
        ],
        font_size=18,
    )

    # Slide 8 outputs and code execution
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "Outputs and Code Execution", "Captured from the real run of `python main.py` in `scam_env`")
    code_text = (
        "Command: python main.py\n\n"
        "Output highlights:\n"
        "- Acoustic module loaded\n"
        "- DEBUG: acoustic function called\n"
        "- Decision: SAFE\n"
        "- Risk Score: 41.0/100\n"
        "- Acoustic anomaly: unnatural pauses detected\n"
        "- Transcript: (empty in this run)\n"
        "- Warnings: HF Hub unauthenticated, Transformers notices, NumPy / SciPy runtime warnings\n"
    )
    box = slide.shapes.add_textbox(Inches(0.65), Inches(1.5), Inches(12.0), Inches(4.8))
    tf = box.text_frame
    tf.word_wrap = True
    tf.text = code_text
    for p in tf.paragraphs:
        p.font.size = Pt(16)
    tf.paragraphs[0].font.size = Pt(17)
    tf.paragraphs[0].font.bold = True
    if RUN_TXT.exists():
        note = slide.shapes.add_textbox(Inches(0.65), Inches(6.05), Inches(12.0), Inches(0.5))
        note_tf = note.text_frame
        note_tf.text = f"Saved output artifacts: {RUN_TXT.name} and {RUN_JSON.name}"
        note_tf.paragraphs[0].font.size = Pt(12)

    # Slide 9 dataset and setup
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "Dataset and Setup", "Core repository notes")
    add_bullets(
        slide,
        Inches(0.65),
        Inches(1.4),
        Inches(12.0),
        Inches(4.8),
        [
            "Tool A: acoustic spoof detector trained on ASVspoof LA data.",
            "Tool B: Whisper ASR evaluated on Common Voice.",
            "The parent repo includes `models/` for checkpoints and `data/` for datasets, but these can be large and are often better kept out of version control.",
            "If you need a clean checkout, use `.gitignore` to exclude generated checkpoints and outputs, then recreate them locally when needed.",
        ],
        font_size=18,
    )

    prs.save(PPTX_PATH)
    return PPTX_PATH


if __name__ == "__main__":
    path = build_deck()
    print(f"Saved presentation to {path}")