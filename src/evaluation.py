"""Evaluation artefacts the Streamlit dashboard reads.

The notebook calls `run_evaluation()` once after training. That function
computes everything we want to show in the Evaluation tab (ROC, SHAP, PR,
calibration, classification report) and writes the results to CSV/JSON/PNG
under ``config.EVAL_DIR``. The Streamlit app loads those files instead of
hardcoded values.
"""

from __future__ import annotations
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import (
    FEATURE_COLS,
    EVAL_DIR,
    EVAL_METRICS_JSON,
    EVAL_SHAP_CSV,
    EVAL_ROC_CSV,
    EVAL_CLF_REPORT_CSV,
    EVAL_INTERVENTION_CSV,
    CALIB_IMG,
    RISK_THRESHOLD,
)


def _ensure_eval_dir() -> None:
    os.makedirs(EVAL_DIR, exist_ok=True)


# ── Main entry point ────────────────────────────────────────────────────────

def run_evaluation(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_prob: np.ndarray,
    *,
    bilstm_auc: Optional[float] = None,
    intervention_log: Optional[List[Dict]] = None,
    decision_threshold: float = 0.40,
) -> Dict:
    """Compute every metric the dashboard reports and persist it to disk.

    Parameters
    ----------
    clf : trained XGBoost classifier (used for SHAP TreeExplainer)
    X_test, y_test, y_prob : held-out test arrays from training
    bilstm_auc : optional BiLSTM AUC for the comparison row in metrics.json
    intervention_log : list of dicts produced by the chat loop with at least
        the keys ``turn``, ``risk_before``, ``ai_valence``, ``delta_val``,
        ``style``. Persisted verbatim to ``EVAL_INTERVENTION_CSV``.
    """
    from sklearn.metrics import (
        roc_curve, roc_auc_score, classification_report,
        precision_recall_curve, average_precision_score, f1_score,
    )

    _ensure_eval_dir()
    out: Dict[str, object] = {}

    # ── 1. Core metrics ─────────────────────────────────────────────────────
    auc = float(roc_auc_score(y_test, y_prob))
    ap  = float(average_precision_score(y_test, y_prob))
    y_pred = (y_prob >= decision_threshold).astype(int)
    f1_esc = float(f1_score(y_test, y_pred))

    out["auc_xgb"]        = auc
    out["avg_precision"]  = ap
    out["f1_escalating"]  = f1_esc
    out["risk_threshold"] = RISK_THRESHOLD
    out["decision_threshold"] = decision_threshold
    out["n_test"]         = int(len(y_test))
    out["positive_rate"]  = float(y_test.mean())
    if bilstm_auc is not None:
        out["auc_bilstm"] = float(bilstm_auc)

    # ── 2. ROC curve points ─────────────────────────────────────────────────
    fpr, tpr, thr = roc_curve(y_test, y_prob)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}) \
      .to_csv(EVAL_ROC_CSV, index=False)

    # ── 3. SHAP feature importance ──────────────────────────────────────────
    try:
        import shap
        explainer   = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test)
        mean_abs    = np.abs(shap_values).mean(axis=0)
        shap_df = (
            pd.DataFrame({"feature": FEATURE_COLS, "mean_abs_shap": mean_abs})
              .sort_values("mean_abs_shap", ascending=False)
              .reset_index(drop=True)
        )
        shap_df.to_csv(EVAL_SHAP_CSV, index=False)
    except Exception as e:
        print(f"SHAP step skipped: {e}")

    # ── 4. Classification report ────────────────────────────────────────────
    rep_dict = classification_report(
        y_test, y_pred,
        target_names=["not-escalating", "escalating"],
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for cls in ["not-escalating", "escalating", "macro avg", "weighted avg"]:
        r = rep_dict[cls]
        rows.append({
            "Class":     cls,
            "Precision": round(float(r["precision"]), 3),
            "Recall":    round(float(r["recall"]),    3),
            "F1-Score":  round(float(r["f1-score"]),  3),
            "Support":   int(r["support"]),
        })
    pd.DataFrame(rows).to_csv(EVAL_CLF_REPORT_CSV, index=False)

    # ── 5. Calibration + PR curve (matplotlib PNG for the UI) ───────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.calibration import CalibrationDisplay

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        CalibrationDisplay.from_predictions(
            y_test, y_prob, n_bins=10, ax=axes[0], name="XGBoost"
        )
        axes[0].set_title("Calibration Curve")
        axes[0].plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        axes[0].legend()

        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        axes[1].plot(rec, prec, color="crimson", lw=2, label=f"XGBoost AP={ap:.3f}")
        axes[1].axhline(y=y_test.mean(), linestyle="--", color="gray", label="Chance")
        axes[1].set(xlabel="Recall", ylabel="Precision",
                    title="Precision-Recall — 2-Turn-Ahead Escalation")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(CALIB_IMG, dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"Calibration plot skipped: {e}")

    # ── 6. Intervention log ─────────────────────────────────────────────────
    if intervention_log:
        df_int = pd.DataFrame(intervention_log)
        df_int.to_csv(EVAL_INTERVENTION_CSV, index=False)
        out["n_interventions"]    = len(df_int)
        if "delta_val" in df_int.columns:
            out["mean_delta_val"]     = float(df_int["delta_val"].mean())
            out["pct_delta_positive"] = float((df_int["delta_val"] > 0).mean())
        if "ai_valence" in df_int.columns:
            out["mean_ai_valence"]    = float(df_int["ai_valence"].mean())

    # ── 7. Persist the top-level metrics JSON ───────────────────────────────
    with open(EVAL_METRICS_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Evaluation artefacts written to {EVAL_DIR}")
    return out


# ── Load helpers (used by the Streamlit app) ────────────────────────────────

def load_metrics() -> Dict:
    if os.path.exists(EVAL_METRICS_JSON):
        with open(EVAL_METRICS_JSON) as f:
            return json.load(f)
    return {}


def load_shap_importance() -> pd.DataFrame:
    if os.path.exists(EVAL_SHAP_CSV):
        return pd.read_csv(EVAL_SHAP_CSV)
    return pd.DataFrame(columns=["feature", "mean_abs_shap"])


def load_roc_curve() -> pd.DataFrame:
    if os.path.exists(EVAL_ROC_CSV):
        return pd.read_csv(EVAL_ROC_CSV)
    return pd.DataFrame(columns=["fpr", "tpr", "threshold"])


def load_classification_report() -> pd.DataFrame:
    if os.path.exists(EVAL_CLF_REPORT_CSV):
        return pd.read_csv(EVAL_CLF_REPORT_CSV)
    return pd.DataFrame(columns=["Class", "Precision", "Recall", "F1-Score", "Support"])


def load_intervention_log() -> pd.DataFrame:
    if os.path.exists(EVAL_INTERVENTION_CSV):
        return pd.read_csv(EVAL_INTERVENTION_CSV)
    return pd.DataFrame(columns=["turn", "risk_before", "ai_valence", "delta_val", "style"])
