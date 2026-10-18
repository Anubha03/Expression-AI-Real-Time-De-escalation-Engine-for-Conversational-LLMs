"""
Expression AI — Real-Time De-escalation & Adaptive Response Engine
=================================================================
Streamlit dashboard. All heavy logic is delegated to the `src/` package;
this file is the UI only.

Run:
    streamlit run streamlit_app.py

The app reads:
- $EXPR_AI_DATA_DIR/emotion_vectors.parquet   (annotated turns)
- $EXPR_AI_DATA_DIR/features.csv              (rolling-window features + UMAP)
- $EXPR_AI_DATA_DIR/saved_models/xgb_escalation.json  (XGBoost classifier)
- $EXPR_AI_DATA_DIR/evaluation/*              (computed by src.evaluation)

When any of those files are missing it falls back to DEMO MODE (synthetic
data, heuristic risk scoring) so the UI is still demo-able without a GPU.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Make `src/` importable when running from the repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config
from src.emotion       import load_pipeline as load_emotion_pipe_real, annotate_text as annotate_text_real
from src.features      import build_feature_vector
from src.models        import load_xgboost
from src.intervention  import (
    load_shap_explainer,
    predict_risk,
    generate_interventions,
    pick_style,
    build_mistral_prompt,
    mistral_generate,
    load_mistral as load_mistral_real,
)
from src.evaluation    import (
    load_metrics,
    load_shap_importance,
    load_roc_curve,
    load_classification_report,
    load_intervention_log,
)

warnings.filterwarnings("ignore")

# ── Config (paths come from src.config so notebook + app stay in sync) ───────
PARQUET_PATH   = config.PARQUET_PATH
FEATURES_PATH  = config.FEATURES_PATH
CALIB_IMG      = config.CALIB_IMG
RISK_THRESHOLD = config.RISK_THRESHOLD


# ═══════════════════════════════════════════════════════════════════════════
# CACHED LOADERS (wrap the src/ functions so Streamlit caches them)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading GoEmotions BERT…")
def load_emotion_pipe():
    try:
        return load_emotion_pipe_real()
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading XGBoost classifier…")
def load_classifier():
    return load_xgboost()


@st.cache_resource(show_spinner="Loading SHAP explainer…")
def _load_shap(_clf):
    if _clf is None:
        return None
    try:
        return load_shap_explainer(_clf)
    except Exception:
        return None


@st.cache_resource(show_spinner="Loading Mistral-7B-Instruct…")
def load_mistral():
    try:
        return load_mistral_real()
    except Exception:
        return None


@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    if os.path.exists(PARQUET_PATH):
        df   = pd.read_parquet(PARQUET_PATH)
        feat = pd.read_csv(FEATURES_PATH) if os.path.exists(FEATURES_PATH) else None
        return df, feat, False
    # ── DEMO MODE: synthetic data ────────────────────────────────────────────
    rng = np.random.default_rng(42)
    n   = 800
    df  = pd.DataFrame({
        "conv_id":         np.repeat(np.arange(200), 4),
        "turn_idx":        np.tile(np.arange(4), 200),
        "role":            np.tile(["user", "assistant", "user", "assistant"], 200),
        "top_emotion":     rng.choice(list(config.EMOTION_VAD.keys()), n),
        "valence":         rng.uniform(-1, 1, n),
        "arousal":         rng.uniform(0, 1, n),
        "dominance":       rng.uniform(0, 1, n),
        "language":        rng.choice(["English", "Spanish", "Chinese", "French", "German"], n),
        "escalation_type": rng.choice(
            ["calm", "escalating", "de-escalating", "volatile"], n,
            p=[0.55, 0.22, 0.13, 0.10],
        ),
        "content":         ["[demo content]"] * n,
    })
    feat = pd.DataFrame({
        "conv_id":         np.arange(200),
        "umap_x":          rng.uniform(-5, 5, 200),
        "umap_y":          rng.uniform(-5, 5, 200),
        "escalation_type": rng.choice(["calm", "escalating", "de-escalating", "volatile"], 200),
        "n_turns":         rng.integers(2, 20, 200),
        "u_val_delta":     rng.uniform(-0.5, 0.3, 200),
        "cluster":         rng.integers(0, 5, 200),
    })
    return df, feat, True


# ── Heuristic annotation for demo mode (BERT not loaded) ────────────────────

def annotate_text(text: str, pipe):
    if pipe is None:
        neg_words = ["frustrat", "useless", "angry", "terrible", "hate",
                     "can't", "stupid", "awful"]
        score = sum(1 for w in neg_words if w in text.lower())
        v = max(-0.9, min(0.9, -0.15 * score + 0.1))
        a = 0.3 + 0.1 * score
        d = 0.5 - 0.05 * score
        em = "annoyance" if score > 1 else ("anger" if score > 3 else "neutral")
        return v, a, d, em
    return annotate_text_real(text, pipe)


def generate_reply_mistral(history, style, interventions):
    """Wrap the src/ generate-reply call with a friendly error message."""
    bundle = load_mistral()
    if bundle is None:
        return "[Mistral not loaded — install bitsandbytes and a GPU runtime]"
    prompt = build_mistral_prompt(
        history, style=style,
        interventions=interventions if interventions else None,
    )
    return mistral_generate(prompt, max_new_tokens=200, temperature=0.7)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE SETUP
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Expression AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #0D1B2A; }
  [data-testid="stSidebar"] * { color: #E0E0E0 !important; }
  .metric-card {
      background: #1E293B; border-radius: 10px; padding: 16px 20px;
      text-align: center; color: white;
  }
  .metric-card h2 { font-size: 2.2rem; margin: 0; color: #38BDF8; }
  .metric-card p  { margin: 4px 0 0; font-size: 0.85rem; color: #94A3B8; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/80/brain.png", width=60)
    st.markdown("## 🧠 Expression AI")
    st.markdown("*Real-Time De-escalation Engine*")
    st.divider()

    st.markdown("### ⚙️ Settings")
    risk_thresh = st.slider("Risk threshold", 0.2, 0.8, RISK_THRESHOLD, 0.05)
    use_mistral = st.toggle("Use Mistral-7B for replies", value=True)
    st.divider()

    st.markdown("### 📊 Pipeline")
    for num, name, desc in [
        ("1", "GoEmotions BERT", "Emotion → VAD"),
        ("2", "Feature Engineering", "14 rolling features"),
        ("3", "XGBoost",          "2-turn-ahead risk"),
        ("4", "SHAP",             "Top-3 explanations"),
        ("5", "Intervention Rules","14 rule bank"),
        ("6", "Mistral-7B",       "Style-conditioned reply"),
    ]:
        st.markdown(
            f"**{num}. {name}**  \n<small style='color:#94A3B8'>{desc}</small>",
            unsafe_allow_html=True,
        )

# ── Load data + models ───────────────────────────────────────────────────────
df, feat, demo_mode = load_data()
emotion_pipe = load_emotion_pipe()
clf          = load_classifier()
explainer    = _load_shap(clf)

if demo_mode:
    st.warning(
        "📌 **Demo Mode** — running on synthetic data. "
        f"Set `EXPR_AI_DATA_DIR` to a folder containing `emotion_vectors.parquet`, "
        f"`features.csv`, and `saved_models/xgb_escalation.json` and reload.",
        icon="ℹ️",
    )


# ═══════════════════════════════════════════════════════════════════════════
# HEADER METRICS  (live from the model when available)
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("# 🧠 Expression AI Dashboard")

metrics_live = load_metrics()
auc_live     = metrics_live.get("auc_xgb")

n_conv   = df["conv_id"].nunique()
n_turns  = len(df)
esc_rate = (df["escalation_type"] == "escalating").mean() if "escalation_type" in df.columns else 0.22
auc_disp = f"{auc_live:.3f}" if auc_live is not None else "—"

c1, c2, c3, c4 = st.columns(4)
for col, val, label in [
    (c1, f"{n_conv:,}",      "Conversations"),
    (c2, f"{n_turns:,}",     "Turns Annotated"),
    (c3, f"{esc_rate:.1%}",  "Escalation Rate"),
    (c4, auc_disp,           "XGBoost AUC-ROC"),
]:
    col.markdown(
        f"<div class='metric-card'><h2>{val}</h2><p>{label}</p></div>",
        unsafe_allow_html=True,
    )

st.divider()


# ═══════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════

tabs = st.tabs([
    "💬 Live Intervention Chat",
    "🗺️ UMAP Clusters",
    "📈 Emotion Timeline",
    "🌡️ Escalation Heatmap",
    "📐 Evaluation",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Live Intervention Chat
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:

    st.markdown("""
    <style>
    .imsg-wrap { display:flex; flex-direction:column; gap:6px; padding:12px 8px; }
    .imsg-user-row  { display:flex; justify-content:flex-end; align-items:flex-start; gap:8px; }
    .imsg-user-bub  { background:#0A84FF; color:#fff; border-radius:18px 18px 4px 18px;
                       padding:10px 14px; max-width:68%; font-size:0.95rem; line-height:1.45; word-break:break-word; }
    .imsg-user-meta { font-size:0.72rem; color:#8E8E93; text-align:right; margin-top:2px; margin-right:4px; }
    .imsg-ai-row    { display:flex; justify-content:flex-start; align-items:flex-start; gap:8px; }
    .imsg-ai-bub    { background:#2C2C2E; color:#F2F2F7; border-radius:18px 18px 18px 4px;
                       padding:10px 14px; max-width:68%; font-size:0.95rem; line-height:1.45; word-break:break-word; }
    .imsg-ai-meta   { font-size:0.72rem; color:#8E8E93; margin-top:2px; margin-left:4px; }
    .imsg-avatar    { width:32px; height:32px; border-radius:50%; display:flex;
                       align-items:center; justify-content:center; font-size:1rem; flex-shrink:0; }
    .imsg-avatar-ai  { background:#3A3A3C; }
    .imsg-avatar-you { background:#0A84FF; }
    .imsg-alert {
        background:rgba(255,149,0,0.12); border:1px solid rgba(255,149,0,0.5);
        border-radius:12px; padding:10px 14px; margin:4px auto; max-width:90%;
        font-size:0.82rem; color:#1a1a1a;
    }
    .imsg-alert b { color:#c05a00; }
    .imsg-alert span { color:#333333; }
    .typing-indicator { display:flex; align-items:center; gap:4px; padding:10px 14px;
        background:#2C2C2E; border-radius:18px 18px 18px 4px; width:fit-content; }
    .typing-dot { width:8px; height:8px; border-radius:50%; background:#8E8E93;
        animation:bounce 1.2s infinite; }
    .typing-dot:nth-child(2) { animation-delay:0.2s; }
    .typing-dot:nth-child(3) { animation-delay:0.4s; }
    @keyframes bounce {
        0%,80%,100% { transform:translateY(0); opacity:0.4; }
        40%         { transform:translateY(-6px); opacity:1; }
    }
    .risk-bar-track { background:#2C2C2E; border-radius:6px; height:8px; width:100%; margin-top:4px; }
    .risk-bar-fill  { border-radius:6px; height:8px; transition:width 0.4s ease; }
    </style>
    """, unsafe_allow_html=True)

    # Session state defaults
    for k, v in [
        ("chat_history", []), ("conv_history", []), ("risk_arc", []),
        ("turn_idx", 0), ("last_risk", 0.0), ("last_emotion", "—"),
        ("last_valence", 0.0), ("last_arousal", 0.0), ("last_dominance", 0.0),
        ("processing", False),
    ]:
        if k not in st.session_state:
            st.session_state[k] = v

    # Top status bar
    top_l, top_r = st.columns([3, 1])
    with top_l:
        risk      = st.session_state.last_risk
        bar_color = "#FF3B30" if risk >= risk_thresh else ("#FF9F0A" if risk >= 0.2 else "#30D158")
        risk_lbl  = "HIGH" if risk >= risk_thresh else ("MED" if risk >= 0.2 else "LOW")
        st.markdown(
            f"<div style='font-size:0.85rem;color:#8E8E93;margin-bottom:2px'>"
            f"Escalation Risk &nbsp;"
            f"<span style='color:{bar_color};font-weight:700'>{risk:.0%} {risk_lbl}</span>"
            f" &nbsp;·&nbsp; {st.session_state.last_emotion}"
            f" &nbsp;·&nbsp; V<span style='color:#0A84FF'>{st.session_state.last_valence:+.2f}</span>"
            f" &nbsp; A<span style='color:#FF9F0A'>{st.session_state.last_arousal:+.2f}</span>"
            f" &nbsp; D<span style='color:#30D158'>{st.session_state.last_dominance:+.2f}</span>"
            f"</div>"
            f"<div class='risk-bar-track'>"
            f"<div class='risk-bar-fill' style='width:{int(risk*100)}%;background:{bar_color}'></div></div>",
            unsafe_allow_html=True,
        )
    with top_r:
        if st.button("↺ New Session", use_container_width=True):
            for key in ["chat_history", "conv_history", "risk_arc",
                        "turn_idx", "last_risk", "last_emotion",
                        "last_valence", "last_arousal", "last_dominance", "processing"]:
                if key in ("chat_history", "conv_history", "risk_arc"):
                    st.session_state[key] = []
                elif key == "turn_idx":
                    st.session_state[key] = 0
                elif key == "last_emotion":
                    st.session_state[key] = "—"
                elif key == "processing":
                    st.session_state[key] = False
                else:
                    st.session_state[key] = 0.0
            st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Chat window
    chat_box = st.container(height=500)
    with chat_box:
        st.markdown("<div class='imsg-wrap'>", unsafe_allow_html=True)
        for entry in st.session_state.chat_history:
            if entry["type"] == "user":
                st.markdown(
                    f"<div class='imsg-user-row'><div>"
                    f"<div class='imsg-user-bub'>{entry['content']}</div>"
                    f"<div class='imsg-user-meta'>{entry['emotion']} · {entry['valence']:+.2f}</div>"
                    f"</div><div class='imsg-avatar imsg-avatar-you'>🙂</div></div>",
                    unsafe_allow_html=True,
                )
            elif entry["type"] == "alert":
                bullets = " &nbsp;·&nbsp; ".join(entry["interventions"])
                st.markdown(
                    f"<div class='imsg-alert'>⚠️ <b>INTERVENTION</b> · "
                    f"risk {entry['risk']:.0%} · {entry['style']}<br>"
                    f"<span>{bullets}</span></div>",
                    unsafe_allow_html=True,
                )
            elif entry["type"] == "assistant":
                st.markdown(
                    f"<div class='imsg-ai-row'><div class='imsg-avatar imsg-avatar-ai'>🤖</div>"
                    f"<div><div class='imsg-ai-bub'>{entry['content']}</div>"
                    f"<div class='imsg-ai-meta'>valence {entry['valence']:+.2f}</div>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )
        if st.session_state.processing:
            st.markdown(
                "<div class='imsg-ai-row'><div class='imsg-avatar imsg-avatar-ai'>🤖</div>"
                "<div class='typing-indicator'>"
                "<div class='typing-dot'></div><div class='typing-dot'></div><div class='typing-dot'></div>"
                "</div></div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # Input bar
    with st.form("chat_form", clear_on_submit=True):
        col_inp, col_btn = st.columns([5, 1])
        user_msg  = col_inp.text_input("msg", placeholder="iMessage…",
                                       label_visibility="collapsed")
        submitted = col_btn.form_submit_button("Send ↑", use_container_width=True)

    # Quick scenario pills
    st.markdown(
        "<div style='font-size:0.78rem;color:#8E8E93;margin:4px 0 6px'>Quick scenario:</div>",
        unsafe_allow_html=True,
    )
    qc1, qc2, qc3, qc4 = st.columns(4)
    SCENARIOS = {
        "😤 Frustrated": "This is the third time you've ignored what I said. This is really frustrating.",
        "😡 Escalated":  "You clearly don't understand my problem at all. Why can't you just listen?",
        "😊 Positive":   "Thanks so much! That worked perfectly. You're really helpful.",
        "😕 Confused":   "I already told you the issue. Can you please re-read what I wrote?",
    }
    quick_msg = None
    for col, (label, msg) in zip([qc1, qc2, qc3, qc4], SCENARIOS.items()):
        if col.button(label, use_container_width=True):
            quick_msg = msg

    if quick_msg:
        user_msg  = quick_msg
        submitted = True

    # Process message
    if submitted and user_msg and user_msg.strip() and not st.session_state.processing:
        st.session_state.turn_idx += 1
        v, a, d, em = annotate_text(user_msg, emotion_pipe)
        st.session_state.conv_history.append(
            {"role": "user", "content": user_msg,
             "valence": v, "arousal": a, "dominance": d, "top_emotion": em}
        )
        st.session_state.chat_history.append(
            {"type": "user", "content": user_msg, "emotion": em, "valence": v}
        )
        st.session_state.last_emotion   = em
        st.session_state.last_valence   = v
        st.session_state.last_arousal   = a
        st.session_state.last_dominance = d
        st.session_state.processing     = True
        st.rerun()

    if st.session_state.processing:
        fv = build_feature_vector(st.session_state.conv_history)
        risk, shap_dict = predict_risk(fv, clf, explainer,
                                       history=st.session_state.conv_history)
        st.session_state.last_risk = risk
        st.session_state.risk_arc.append((st.session_state.turn_idx, risk))

        alert_fired   = risk >= risk_thresh
        interventions = []
        style         = "direct"
        if alert_fired:
            interventions = generate_interventions(shap_dict)
            style         = pick_style(shap_dict)
            st.session_state.chat_history.append(
                {"type": "alert", "interventions": interventions,
                 "risk": risk, "style": style}
            )

        with st.spinner("Mistral is thinking…"):
            if use_mistral:
                reply = generate_reply_mistral(
                    st.session_state.conv_history,
                    style,
                    interventions if alert_fired else [],
                )
            else:
                reply = config.STYLE_PREFIXES.get(style, "") + \
                        "I hear you. Let me address your concern carefully."

        rv, ra, rd, rem = annotate_text(reply, emotion_pipe)
        st.session_state.conv_history.append(
            {"role": "assistant", "content": reply,
             "valence": rv, "arousal": ra, "dominance": rd, "top_emotion": rem}
        )
        st.session_state.chat_history.append(
            {"type": "assistant", "content": reply, "valence": rv}
        )
        st.session_state.processing = False
        st.rerun()

    # Live risk arc
    if st.session_state.risk_arc:
        turns, risks = zip(*st.session_state.risk_arc)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(turns), y=list(risks),
            mode="lines+markers",
            line=dict(color="#F44336", width=2),
            fill="tozeroy", fillcolor="rgba(244,67,54,0.1)",
            name="Escalation risk",
        ))
        fig.add_hline(y=risk_thresh, line_dash="dash", line_color="#FF9800",
                      annotation_text=f"Threshold ({risk_thresh:.0%})")
        fig.update_layout(
            title="Live Risk Arc",
            xaxis_title="User turn", yaxis_title="P(escalation)",
            yaxis=dict(range=[0, 1]),
            height=240, margin=dict(t=40, b=40),
            paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
            font=dict(color="#E2E8F0"),
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — UMAP Clusters
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("🗺️ UMAP — Conversation Emotional Trajectory Space")
    col_l, col_r = st.columns([3, 1])
    with col_r:
        show_clusters = st.toggle("Show KMeans clusters", value=False)
        types_shown   = st.multiselect(
            "Escalation types",
            ["calm", "escalating", "de-escalating", "volatile"],
            default=["calm", "escalating", "de-escalating", "volatile"],
        )

    if feat is not None and {"umap_x", "umap_y"}.issubset(feat.columns):
        plot_feat = (
            feat[feat["escalation_type"].isin(types_shown)]
            if "escalation_type" in feat.columns else feat
        )
        color_col = "cluster" if show_clusters and "cluster" in plot_feat.columns else "escalation_type"
        cmap = ({str(i): c for i, c in enumerate(px.colors.qualitative.Bold)}
                if show_clusters else config.COLOR_MAP)
        fig = px.scatter(
            plot_feat,
            x="umap_x", y="umap_y",
            color=color_col if not show_clusters else plot_feat[color_col].astype(str),
            color_discrete_map=cmap,
            hover_data=[c for c in ["conv_id", "n_turns", "u_val_delta"] if c in plot_feat.columns],
            opacity=0.65, height=560,
            title="UMAP: One dot = one conversation window",
        )
        fig.update_layout(paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
                          font=dict(color="#E2E8F0"))
        col_l.plotly_chart(fig, use_container_width=True)
    else:
        st.info("UMAP coordinates not found. Run the notebook's UMAP cell to populate.")

    st.divider()
    st.markdown("**Escalation Archetypes (KMeans on UMAP)**")
    st.dataframe(pd.DataFrame({
        "Cluster":    [0, 1, 2, 3, 4],
        "Archetype":  list(config.ARCHETYPE_NAMES.values()),
        "Key Signal": [
            "Stable positive valence, low arousal, short conversations",
            "High refusal rate → valence drops sharply after AI declines",
            "User seeks support but AI responds with information, not empathy",
            "High valence std, user pivots topic frequently under stress",
            "Apology + concrete fix restores valence — success case",
        ],
    }), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Emotion Timeline
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("📈 Per-Conversation Emotion Arc")
    conv_ids = df["conv_id"].unique().tolist()
    sel_id   = st.selectbox("Select conversation ID", conv_ids)
    grp      = df[df.conv_id == sel_id].sort_values("turn_idx")
    esc_type = grp["escalation_type"].iloc[0] if "escalation_type" in grp.columns else "unknown"

    fig = go.Figure()
    for role, color in [("user", "#E53935"), ("assistant", "#1E88E5")]:
        sub = grp[grp.role == role]
        if len(sub):
            hover = sub["top_emotion"] if "top_emotion" in sub.columns else sub["role"]
            fig.add_trace(go.Scatter(
                x=sub["turn_idx"], y=sub["valence"],
                mode="lines+markers", name=f"{role} valence",
                line=dict(color=color, width=2),
                text=hover,
                hovertemplate="%{text}<br>Valence: %{y:.2f}",
            ))
    fig.add_hline(y=0, line_dash="dot", line_color="#64748B")
    fig.update_layout(
        title=f"Conversation {sel_id} — Archetype: {esc_type}",
        xaxis_title="Turn index", yaxis_title="Valence",
        yaxis=dict(range=[-1, 1]),
        height=400, paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
        font=dict(color="#E2E8F0"), hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    if "top_emotion" in grp.columns:
        st.dataframe(
            grp[["turn_idx", "role", "top_emotion", "valence", "arousal", "dominance", "content"]]
               .reset_index(drop=True),
            use_container_width=True, height=280,
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Heatmap
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("🌡️ Escalation Heatmap by Language (top 10)")
    if "language" in df.columns and "escalation_type" in df.columns:
        heat = df.groupby(["language", "escalation_type"]).size().unstack(fill_value=0)
        top_langs = df["language"].value_counts().head(10).index
        heat = heat.loc[heat.index.isin(top_langs)]
        fig = px.imshow(
            heat, text_auto=True, color_continuous_scale="YlOrRd",
            aspect="auto", height=450,
            title="Rows = language | Columns = escalation type | Value = windows",
        )
        fig.update_layout(paper_bgcolor="#0F172A", font=dict(color="#E2E8F0"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Language column not found in the dataset.")

    if "top_emotion" in df.columns:
        st.subheader("Top Emotions Across All Turns")
        em_counts = df["top_emotion"].value_counts().head(15).reset_index()
        em_counts.columns = ["emotion", "count"]
        fig2 = px.bar(
            em_counts, x="count", y="emotion", orientation="h",
            color="count", color_continuous_scale="Blues", height=420,
        )
        fig2.update_layout(
            yaxis=dict(categoryorder="total ascending"),
            paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
            font=dict(color="#E2E8F0"),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Evaluation (READS LIVE METRICS COMPUTED BY src/evaluation.py)
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("📐 Model Evaluation")

    metrics      = load_metrics()
    shap_df      = load_shap_importance()
    roc_df       = load_roc_curve()
    clf_report   = load_classification_report()
    intervention = load_intervention_log()

    have_live = bool(metrics)

    if not have_live:
        st.info(
            "No evaluation artefacts found at "
            f"`{config.EVAL_DIR}`. Run the **Evaluation** section of the "
            "notebook to populate this tab with live values."
        )

    # ── Headline metric cards (live when available) ───────────────────────
    auc_xgb      = metrics.get("auc_xgb")
    auc_bilstm   = metrics.get("auc_bilstm")
    f1_esc       = metrics.get("f1_escalating")
    avg_prec     = metrics.get("avg_precision")
    thresh_used  = metrics.get("risk_threshold", RISK_THRESHOLD)

    def fmt(v, p=3):
        return f"{v:.{p}f}" if isinstance(v, (int, float)) else "—"

    m1, m2, m3, m4 = st.columns(4)
    for col, val, label in [
        (m1, fmt(auc_xgb),     "AUC-ROC (XGBoost)"),
        (m2, fmt(f1_esc, 2),   "F1 — Escalating"),
        (m3, fmt(avg_prec),    "Avg Precision (PR)"),
        (m4, fmt(thresh_used, 2), "Risk Threshold"),
    ]:
        col.markdown(
            f"<div class='metric-card'><h2>{val}</h2><p>{label}</p></div>",
            unsafe_allow_html=True,
        )
    if auc_bilstm is not None:
        st.caption(f"For comparison, BiLSTM AUC = {auc_bilstm:.3f}.")

    st.divider()

    col_a, col_b = st.columns(2)

    # SHAP feature importance — live from the trained model
    if len(shap_df):
        fig_shap = px.bar(
            shap_df.head(10).iloc[::-1],
            x="mean_abs_shap", y="feature", orientation="h",
            color="mean_abs_shap", color_continuous_scale="Reds",
            title="SHAP Feature Importance (live from trained XGBoost)",
            height=380,
        )
        fig_shap.update_layout(
            yaxis=dict(categoryorder="total ascending"),
            paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
            font=dict(color="#E2E8F0"),
        )
        col_a.plotly_chart(fig_shap, use_container_width=True)
    else:
        col_a.info("SHAP importance not available — run training + evaluation.")

    # ROC curve — live points written by src.evaluation
    if len(roc_df):
        fig_roc = go.Figure()
        auc_lbl = f"XGBoost (AUC={auc_xgb:.3f})" if auc_xgb is not None else "XGBoost"
        fig_roc.add_trace(go.Scatter(
            x=roc_df["fpr"], y=roc_df["tpr"],
            mode="lines", name=auc_lbl,
            line=dict(color="#F87171", width=2),
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random baseline",
            line=dict(dash="dash", color="#64748B"),
        ))
        fig_roc.update_layout(
            title="ROC Curve — 2-Turn-Ahead Escalation (live)",
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            height=380,
            paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
            font=dict(color="#E2E8F0"),
        )
        col_b.plotly_chart(fig_roc, use_container_width=True)
    else:
        col_b.info("ROC curve not available — run training + evaluation.")

    # Calibration & PR PNG (rendered by src.evaluation)
    if os.path.exists(CALIB_IMG):
        st.image(CALIB_IMG,
                 caption="Calibration Curve + Precision-Recall Curve",
                 use_container_width=True)
    else:
        st.info("Calibration plot not found — run the evaluation cell in the notebook.")

    st.divider()
    st.subheader("Intervention Effectiveness (Closed-Loop)")
    if len(intervention):
        # Per-alert delta-valence bar chart
        fig_eff = go.Figure(go.Bar(
            x=intervention.get("turn", intervention.index),
            y=intervention["delta_val"] if "delta_val" in intervention.columns else [],
            marker_color=[
                "#4CAF50" if v > 0 else "#F44336"
                for v in (intervention["delta_val"] if "delta_val" in intervention.columns else [])
            ],
        ))
        fig_eff.add_hline(y=0, line_color="white", line_width=0.8)
        fig_eff.update_layout(
            title="Delta AI Valence after each intervention alert (live)",
            xaxis_title="Turn index", yaxis_title="Delta AI Valence",
            height=300,
            paper_bgcolor="#0F172A", plot_bgcolor="#0F172A",
            font=dict(color="#E2E8F0"),
        )
        st.plotly_chart(fig_eff, use_container_width=True)

        # Summary stats
        if "delta_val" in intervention.columns:
            mean_d = intervention["delta_val"].mean()
            pos_pct = (intervention["delta_val"] > 0).mean()
            st.caption(
                f"**Summary:** {len(intervention)} interventions fired · "
                f"mean Δ-valence = **{mean_d:+.3f}** · "
                f"{pos_pct:.0%} of alerts produced a warmer reply."
            )
    else:
        st.info(
            "Intervention log empty — run the scripted demo / chat loop in "
            "the notebook (Section 9), then re-run evaluation."
        )

    # Classification report — live, not hardcoded
    st.subheader("Classification Report (live)")
    if len(clf_report):
        st.dataframe(clf_report, use_container_width=True, hide_index=True)
    else:
        st.info("Classification report not available — run training + evaluation.")
