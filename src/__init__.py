"""Expression AI — Real-Time De-escalation & Adaptive Response Engine.

This package contains the full pipeline used by both the training notebook
(`notebooks/Expression_AI.ipynb`) and the live Streamlit dashboard
(`streamlit_app.py`). It is deliberately broken up so the notebook can train
artefacts once and the Streamlit app can load them cheaply.

Modules
-------
config        : constants (EMOTION_VAD lookup, FEATURE_COLS, RULES, paths)
emotion       : GoEmotions BERT pipeline + VAD projection
features      : rolling-window feature engineering + live feature vector
models        : XGBoost (primary) + BiLSTM (sequence baseline) train/load
intervention  : SHAP -> directives, risk scoring, Mistral generation
evaluation    : computes and persists evaluation artefacts used by the UI
"""

from . import config, emotion, features, models, intervention, evaluation  # noqa: F401

__all__ = ["config", "emotion", "features", "models", "intervention", "evaluation"]
