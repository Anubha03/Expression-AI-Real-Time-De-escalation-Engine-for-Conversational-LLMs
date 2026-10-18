"""Closed-loop intervention engine.

The runtime pipeline for one user turn is:

    annotate -> build_feature_vector -> predict_risk
             -> if risk >= threshold: SHAP -> directives + style
             -> build_mistral_prompt with directives
             -> generate reply

Each function in this module corresponds to one of those steps so that both
the Streamlit app and the notebook chat loop can call them.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import (
    FEATURE_COLS,
    INTERVENTION_RULES,
    STYLE_PREFIXES,
    MISTRAL_MODEL_ID,
    RISK_THRESHOLD,
)
from .features import build_feature_vector, feature_vector_to_X  # re-exported


# ── Risk scoring ────────────────────────────────────────────────────────────

def hard_rule_risk(history: Sequence[Dict]) -> float:
    """Lightweight rule-based boost layered on top of the XGBoost score.

    Captures abrupt, obvious escalations that the rolling-window model may
    under-weight (single very-negative turn, large valence drop). Scaled by
    conversation length so it doesn't fire on the second turn.
    """
    user_turns = [t for t in history if t["role"] == "user"]
    if len(user_turns) < 2:
        return 0.0
    vals     = [t["valence"] for t in user_turns]
    delta    = vals[-1] - vals[-2]
    mean_val = sum(vals) / len(vals)
    boost = 0.0
    if vals[-1] < -0.4: boost += 0.35
    if delta     < -0.3: boost += 0.25
    if mean_val  < -0.2: boost += 0.15
    scale = min(len(user_turns) / 5.0, 1.0)
    return min(boost * scale, 0.40)


def predict_risk_heuristic(fv: Dict[str, float]) -> float:
    """Fallback heuristic used only when the XGBoost model isn't loaded."""
    score = 0.0
    if fv["u_val_last"]   < -0.5:  score += 0.20
    elif fv["u_val_last"] < -0.3:  score += 0.10
    if fv["u_val_delta"]  < -0.4:  score += 0.15
    elif fv["u_val_delta"]< -0.2:  score += 0.08
    if fv["u_val_slope"]  < -0.15: score += 0.10
    if fv["u_aro_last"]   >  0.75: score += 0.07
    if fv["a_refusal_rate"] > 0.3: score += 0.08
    if fv["n_turns"]      > 10:    score += 0.05
    return min(float(score), 0.60)


def predict_risk(
    fv: Dict[str, float],
    clf=None,
    explainer=None,
    history: Optional[Sequence[Dict]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Return (escalation_prob, shap_dict).

    If ``clf`` is None we fall back to the heuristic risk and approximate
    SHAP values from the raw feature magnitudes — this keeps the Streamlit
    app usable when no trained artefact is on disk.
    """
    X = feature_vector_to_X(fv)
    if clf is not None:
        prob = float(clf.predict_proba(X)[0, 1])
        if explainer is not None:
            sv = explainer.shap_values(X)[0]
            shap_dict = dict(zip(FEATURE_COLS, sv))
        else:
            shap_dict = {k: abs(fv[k]) for k in FEATURE_COLS}
    else:
        prob = predict_risk_heuristic(fv)
        shap_dict = {k: abs(fv[k]) * 0.2 for k in FEATURE_COLS}

    if history:
        prob = min(prob + hard_rule_risk(history), 0.95)
    return prob, shap_dict


# ── SHAP -> intervention directives ─────────────────────────────────────────

def generate_interventions(shap_dict: Dict[str, float], top_k: int = 3) -> List[str]:
    """Translate the top-|SHAP| features into natural-language directives."""
    ranked = sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)
    out: List[str] = []
    for feat, _ in ranked:
        if feat in INTERVENTION_RULES:
            out.append(INTERVENTION_RULES[feat])
        if len(out) >= top_k:
            break
    return out


def pick_style(shap_dict: Dict[str, float]) -> str:
    """Pick a style prefix from the dominant SHAP feature."""
    top = max(shap_dict, key=lambda k: abs(shap_dict[k]))
    if top in ("u_val_last", "u_val_delta", "u_val_mean"):
        return "empathy_first"
    if top in ("a_refusal_rate", "a_apology_rate"):
        return "apologetic"
    return "direct"


# ── SHAP explainer ──────────────────────────────────────────────────────────

def load_shap_explainer(clf):
    """Build a SHAP TreeExplainer for the XGBoost classifier."""
    if clf is None:
        return None
    import shap
    return shap.TreeExplainer(clf)


# ── Mistral-7B ──────────────────────────────────────────────────────────────

_MISTRAL = None  # cached (tokenizer, model)


def load_mistral(model_id_or_path: str = MISTRAL_MODEL_ID):
    """Load Mistral-7B-Instruct in 4-bit NF4. Cached after first call."""
    global _MISTRAL
    if _MISTRAL is not None:
        return _MISTRAL

    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(model_id_or_path)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    mdl.eval()
    _MISTRAL = (tok, mdl)
    return _MISTRAL


def build_mistral_prompt(
    history: Sequence[Dict],
    style: str = "direct",
    interventions: Optional[List[str]] = None,
) -> str:
    """Construct a Mistral-7B [INST] prompt with optional intervention directives.

    The system prompt is embedded in the *first* user [INST] block via the
    Mistral 3-style ``<<SYS>>...<</SYS>>`` wrapper. If `interventions` is
    non-empty, the top two are appended to the system prompt as a "STYLE
    DIRECTIVE:" line.
    """
    prefix = STYLE_PREFIXES.get(style, "")
    base   = "You are a helpful, empathetic AI assistant. Keep replies concise (2-4 sentences)."
    system = (f"{base} STYLE DIRECTIVE: {'. '.join(interventions[:2])}."
              if interventions else base)

    parts: List[str] = []
    history = list(history)[-6:]
    for idx, t in enumerate(history):
        if t["role"] == "user":
            text = (f"<<SYS>>\n{system}\n<</SYS>>\n\n{t['content']}"
                    if idx == 0 else t["content"])
            parts.append(f"<s>[INST] {text} [/INST]")
        else:
            parts.append(f" {t['content']} </s>")

    prompt = "".join(parts)
    if prefix:
        prompt = prompt.rstrip() + f"\n{prefix}"
    return prompt


def mistral_generate(
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
) -> str:
    """Generate one continuation. Strips the prompt from the decoded output."""
    import torch
    tok, mdl = load_mistral()
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tok.eos_token_id,
        )
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_ids, skip_special_tokens=True).strip()


def generate_reply(
    history: Sequence[Dict],
    style: str = "direct",
    interventions: Optional[List[str]] = None,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
) -> str:
    """One-call wrapper: build prompt + generate."""
    prompt = build_mistral_prompt(history, style=style, interventions=interventions)
    return mistral_generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)


# ── End-to-end one-turn step (used by the chat loop) ────────────────────────

def chat_step(
    user_message: str,
    history: List[Dict],
    *,
    emotion_pipe=None,
    clf=None,
    explainer=None,
    risk_threshold: float = RISK_THRESHOLD,
    use_mistral: bool = True,
) -> Dict:
    """Process one user turn end-to-end. Mutates `history` in place.

    Returns a dict with the keys the UI cares about:
    reply, risk_prob, alert_fired, interventions, style, user_emotion.
    """
    from .emotion import annotate_text  # local to avoid heavy import at module load

    # 1. annotate the user message
    v, a, d, em = annotate_text(user_message, emotion_pipe)
    history.append({
        "role": "user", "content": user_message,
        "valence": v, "arousal": a, "dominance": d, "top_emotion": em,
    })

    # 2. risk + SHAP
    fv = build_feature_vector(history)
    risk, shap_dict = predict_risk(fv, clf=clf, explainer=explainer, history=history)

    alert_fired   = risk >= risk_threshold
    interventions = []
    style         = "direct"
    if alert_fired:
        interventions = generate_interventions(shap_dict)
        style         = pick_style(shap_dict)

    # 3. generate reply
    if use_mistral:
        reply = generate_reply(history, style=style,
                                interventions=interventions if alert_fired else None)
    else:
        reply = STYLE_PREFIXES.get(style, "") + "I hear you. Let me address your concern."

    # 4. re-annotate AI reply
    rv, ra, rd, rem = annotate_text(reply, emotion_pipe)
    history.append({
        "role": "assistant", "content": reply,
        "valence": rv, "arousal": ra, "dominance": rd, "top_emotion": rem,
    })

    return {
        "reply":          reply,
        "risk_prob":      risk,
        "alert_fired":    alert_fired,
        "interventions":  interventions,
        "style":          style,
        "user_emotion":   {"valence": v, "arousal": a, "dominance": d, "label": em},
        "ai_emotion":     {"valence": rv, "arousal": ra, "dominance": rd, "label": rem},
        "shap_dict":      shap_dict,
    }
