"""Feature engineering for escalation prediction.

Two surfaces:
- `engineer_rolling_features(df_turns)` builds the training matrix from the
  annotated turn dataframe. Each row is one 4-turn rolling window and the
  label is "did the user's valence drop by >0.15 two turns later".
- `build_feature_vector(history)` builds the same feature vector at runtime
  from an in-memory list of turn dicts (used by the chat loop / Streamlit).

Plus `label_escalation_per_conversation()` which assigns one of
{calm, escalating, de-escalating, volatile} to each conversation as a whole.
"""

from __future__ import annotations
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .config import WINDOW_SIZE, ESCALATION_THRESH, FEATURE_COLS


# ── Per-conversation 4-class label ──────────────────────────────────────────

def label_escalation(group: pd.DataFrame) -> str:
    """Classify one conversation as calm / escalating / de-escalating / volatile."""
    user_vals = group[group.role == "user"]["valence"].values
    if len(user_vals) < 2:
        return "calm"
    delta  = np.diff(user_vals)
    mean_d = delta.mean()
    std_d  = delta.std()
    if mean_d < -0.15:
        return "escalating"
    elif mean_d > 0.15:
        return "de-escalating"
    elif std_d > 0.25:
        return "volatile"
    return "calm"


def label_escalation_per_conversation(df_turns: pd.DataFrame) -> pd.DataFrame:
    """Return a 2-col DataFrame: (conv_id, escalation_type)."""
    out = (
        df_turns
        .groupby("conv_id", group_keys=False)
        .apply(label_escalation)
        .reset_index()
    )
    out.columns = ["conv_id", "escalation_type"]
    return out


# ── Rolling-window feature matrix (training time) ───────────────────────────

_HEDGE_RE   = r"\b(?:perhaps|maybe|might|could|it depends)\b"
_APOL_RE    = r"\b(?:sorry|apologize|apologies)\b"
_REFUSAL_RE = r"\b(?:i can't|i cannot|i'm unable)\b"


def engineer_rolling_features(
    df_turns: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    label_threshold: float = ESCALATION_THRESH,
) -> pd.DataFrame:
    """Build the training matrix from an annotated turn dataframe.

    For each user turn at window-end position ``w_end`` in a conversation
    with at least ``window_size + 2`` user turns, emit one row containing
    14 features and the binary label ``is_escalating`` =
    1[V_user(t+1) < V_user(t) - label_threshold].
    """
    rows: List[Dict] = []
    for conv_id, grp in df_turns.groupby("conv_id"):
        grp  = grp.sort_values("turn_idx").reset_index(drop=True)
        user = grp[grp.role == "user"].reset_index(drop=True)
        asst = grp[grp.role == "assistant"].reset_index(drop=True)
        if len(user) < window_size + 2:
            continue
        u_vals = user["valence"].values
        u_aros = user["arousal"].values
        u_doms = user["dominance"].values

        for w_start in range(len(u_vals) - window_size - 1):
            w_end      = w_start + window_size
            target_idx = w_end + 1
            w_val = u_vals[w_start:w_end]
            w_aro = u_aros[w_start:w_end]
            w_dom = u_doms[w_start:w_end]
            is_esc = int(u_vals[target_idx] < u_vals[w_end] - label_threshold)

            w_tmin = user.iloc[w_start]["turn_idx"]
            w_tmax = user.iloc[w_end]["turn_idx"]
            asst_w = asst[(asst["turn_idx"] >= w_tmin) & (asst["turn_idx"] <= w_tmax)]

            a_len     = asst_w["content"].str.len().mean()                           if len(asst_w) else 100
            a_val_m   = asst_w["valence"].mean()                                     if len(asst_w) else 0
            a_hedge   = asst_w["content"].str.lower().str.count(_HEDGE_RE).mean()    if len(asst_w) else 0
            a_apology = asst_w["content"].str.lower().str.contains(_APOL_RE).mean()  if len(asst_w) else 0
            a_refusal = asst_w["content"].str.lower().str.contains(_REFUSAL_RE).mean() if len(asst_w) else 0

            rows.append({
                "conv_id":        conv_id,
                "window_start":   w_start,
                "corpus":         grp["corpus"].iloc[0] if "corpus" in grp else "unknown",
                "u_val_mean":     w_val.mean(),
                "u_val_last":     w_val[-1],
                "u_val_delta":    w_val[-1] - w_val[0],
                "u_val_std":      w_val.std() if len(w_val) > 1 else 0,
                "u_val_slope":    float(np.polyfit(range(len(w_val)), w_val, 1)[0]),
                "u_aro_mean":     w_aro.mean(),
                "u_aro_last":     w_aro[-1],
                "u_dom_mean":     w_dom.mean(),
                "n_turns":        len(grp),
                "a_len_mean":     a_len,
                "a_val_mean":     a_val_m,
                "a_hedge_rate":   a_hedge,
                "a_apology_rate": a_apology,
                "a_refusal_rate": a_refusal,
                "is_escalating":  is_esc,
            })
    return pd.DataFrame(rows)


# ── Live feature vector (inference time) ────────────────────────────────────

def build_feature_vector(history: Sequence[Dict]) -> Dict[str, float]:
    """Build a single 14-element feature dict from a live chat history.

    Each element of ``history`` must be a dict with the keys
    role, content, valence, arousal, dominance, top_emotion.
    """
    df_c   = pd.DataFrame(list(history))
    user_c = df_c[df_c.role == "user"]
    asst_c = df_c[df_c.role == "assistant"]

    u_val = user_c["valence"].values  if len(user_c) else np.array([0.0])
    u_aro = user_c["arousal"].values  if len(user_c) else np.array([0.5])
    u_dom = user_c["dominance"].values if len(user_c) else np.array([0.5])

    a_len     = asst_c["content"].str.len().mean()                              if len(asst_c) else 100
    a_val     = asst_c["valence"].mean()                                        if len(asst_c) else 0.0
    a_hedge   = asst_c["content"].str.lower().str.count(_HEDGE_RE).mean()       if len(asst_c) else 0
    a_apology = asst_c["content"].str.lower().str.contains(_APOL_RE).mean()     if len(asst_c) else 0
    a_refusal = asst_c["content"].str.lower().str.contains(_REFUSAL_RE).mean() if len(asst_c) else 0
    slope     = float(np.polyfit(range(len(u_val)), u_val, 1)[0]) if len(u_val) > 1 else 0.0

    return {
        "u_val_mean":     float(u_val.mean()),
        "u_val_last":     float(u_val[-1]),
        "u_val_delta":    float(u_val[-1] - u_val[0]) if len(u_val) > 1 else 0.0,
        "u_val_std":      float(u_val.std()) if len(u_val) > 1 else 0.0,
        "u_val_slope":    slope,
        "u_aro_mean":     float(u_aro.mean()),
        "u_aro_last":     float(u_aro[-1]),
        "u_dom_mean":     float(u_dom.mean()),
        "n_turns":        len(df_c),
        "a_len_mean":     float(a_len),
        "a_val_mean":     float(a_val),
        "a_hedge_rate":   float(a_hedge),
        "a_apology_rate": float(a_apology),
        "a_refusal_rate": float(a_refusal),
    }


def feature_vector_to_X(fv: Dict[str, float]) -> pd.DataFrame:
    """Order a feature dict into a 1-row DataFrame with `FEATURE_COLS`."""
    return pd.DataFrame([fv])[FEATURE_COLS].fillna(0)
