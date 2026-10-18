"""Constants shared by every module.

Anything that could plausibly need editing (data paths, thresholds,
feature names, the GoEmotions->VAD lookup, the rule bank, the style
prefixes) lives here so other modules import it instead of duplicating.
"""

from __future__ import annotations
import os

# ── Paths ────────────────────────────────────────────────────────────────────
# Override by setting EXPR_AI_DATA_DIR before importing this module.
DATA_DIR     = os.environ.get("EXPR_AI_DATA_DIR", "/content/expression_ai")
DB_PATH      = f"{DATA_DIR}/wildchat.db"
PARQUET_PATH = f"{DATA_DIR}/emotion_vectors.parquet"
FEATURES_PATH= f"{DATA_DIR}/features.csv"

# Trained model artefacts
SAVED_MODELS_DIR = f"{DATA_DIR}/saved_models"
XGB_PATH         = f"{SAVED_MODELS_DIR}/xgb_escalation.json"
BILSTM_PATH      = f"{SAVED_MODELS_DIR}/bilstm_escalation.pt"

# Evaluation artefacts the Streamlit Evaluation tab reads
EVAL_DIR                = f"{DATA_DIR}/evaluation"
EVAL_METRICS_JSON       = f"{EVAL_DIR}/metrics.json"
EVAL_SHAP_CSV           = f"{EVAL_DIR}/shap_importance.csv"
EVAL_ROC_CSV            = f"{EVAL_DIR}/roc_curve.csv"
EVAL_CLF_REPORT_CSV     = f"{EVAL_DIR}/classification_report.csv"
EVAL_INTERVENTION_CSV   = f"{EVAL_DIR}/intervention_log.csv"
CALIB_IMG               = f"{EVAL_DIR}/calibration_pr.png"

# ── Modelling hyper-params ───────────────────────────────────────────────────
WINDOW_SIZE        = 4       # rolling-window length, in user turns
ESCALATION_THRESH  = 0.15    # label rule: drop > 0.15 in valence -> escalating
RISK_THRESHOLD     = 0.30    # operating point for the intervention engine
N_CONVERSATIONS    = 50_000  # WildChat sample size
BATCH              = 128     # GoEmotions inference batch size (256 on A100)

MISTRAL_MODEL_ID   = "mistralai/Mistral-7B-Instruct-v0.3"
GOEMO_MODEL_ID     = "monologg/bert-base-cased-goemotions-original"

# ── Feature columns ──────────────────────────────────────────────────────────
FEATURE_COLS = [
    "u_val_mean", "u_val_last", "u_val_delta", "u_val_std", "u_val_slope",
    "u_aro_mean", "u_aro_last", "u_dom_mean", "n_turns",
    "a_len_mean", "a_val_mean", "a_hedge_rate", "a_apology_rate", "a_refusal_rate",
]

# ── GoEmotions (28 labels) -> approximate VAD triples ────────────────────────
# V in [-1, 1], A in [0, 1], D in [0, 1].
EMOTION_VAD = {
    "admiration":     ( 0.7,  0.5,  0.4),
    "amusement":      ( 0.8,  0.5,  0.5),
    "anger":          (-0.7,  0.8,  0.6),
    "annoyance":      (-0.5,  0.6,  0.5),
    "approval":       ( 0.6,  0.3,  0.5),
    "caring":         ( 0.7,  0.3,  0.4),
    "confusion":      (-0.2,  0.5,  0.3),
    "curiosity":      ( 0.5,  0.6,  0.4),
    "desire":         ( 0.6,  0.7,  0.4),
    "disappointment": (-0.6,  0.4,  0.3),
    "disapproval":    (-0.6,  0.5,  0.5),
    "disgust":        (-0.7,  0.5,  0.5),
    "embarrassment":  (-0.5,  0.5,  0.3),
    "excitement":     ( 0.8,  0.8,  0.5),
    "fear":           (-0.8,  0.8,  0.2),
    "gratitude":      ( 0.8,  0.3,  0.4),
    "grief":          (-0.8,  0.3,  0.2),
    "joy":            ( 0.9,  0.7,  0.6),
    "love":           ( 0.9,  0.5,  0.5),
    "nervousness":    (-0.5,  0.7,  0.2),
    "neutral":        ( 0.0,  0.1,  0.5),
    "optimism":       ( 0.7,  0.5,  0.5),
    "pride":          ( 0.7,  0.5,  0.7),
    "realization":    ( 0.3,  0.4,  0.4),
    "relief":         ( 0.7,  0.2,  0.5),
    "remorse":        (-0.6,  0.3,  0.2),
    "sadness":        (-0.7,  0.3,  0.3),
    "surprise":       ( 0.2,  0.7,  0.3),
}

# ── SHAP feature -> intervention directive ───────────────────────────────────
INTERVENTION_RULES = {
    "u_val_delta":    "Valence is dropping fast -- acknowledge the frustration explicitly.",
    "u_val_last":     "User sentiment very negative -- lead with empathy, not information.",
    "u_aro_last":     "High arousal -- keep reply short (3 sentences max).",
    "u_val_std":      "Volatile pattern -- stay consistent in tone.",
    "u_val_slope":    "Valence trend declining -- reframe positively.",
    "a_refusal_rate": "Refusals driving frustration -- offer concrete alternative.",
    "a_hedge_rate":   "Excessive hedging -- be more direct.",
    "a_len_mean":     "Length mismatch -- consider shorter reply.",
    "a_apology_rate": "Apologies insufficient -- follow with action.",
    "n_turns":        "Long conversation, no resolution -- summarise and ask what user needs.",
    "u_aro_mean":     "Sustained high arousal -- validate feelings first.",
    "u_dom_mean":     "User feels low control -- offer clear choice.",
    "a_val_mean":     "AI responses flat -- add warmth.",
    "u_val_mean":     "Negative conversation valence -- check response framing.",
}

STYLE_PREFIXES = {
    "empathy_first": "I completely understand your frustration. ",
    "direct":        "",
    "apologetic":    "I sincerely apologise for the confusion. ",
}

COLOR_MAP = {
    "calm":          "#4CAF50",
    "escalating":    "#F44336",
    "de-escalating": "#2196F3",
    "volatile":      "#FF9800",
}

ARCHETYPE_NAMES = {
    0: "Calm Queries",
    1: "Frustrated Refusals",
    2: "Emotional Support Breakdown",
    3: "Volatile Task Switching",
    4: "De-escalated by Apology",
}


def ensure_dirs() -> None:
    """Create all output dirs if they don't exist."""
    for d in (DATA_DIR, SAVED_MODELS_DIR, EVAL_DIR):
        os.makedirs(d, exist_ok=True)
