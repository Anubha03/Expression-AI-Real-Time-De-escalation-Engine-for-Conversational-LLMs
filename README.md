# Expression AI: Real-Time De-escalation Engine for Conversational LLMs

**Real-Time De-escalation & Adaptive Response Engine for Conversational LLMs**
*EECS 6895 · Big Data Analytics · Final Project*

Expression AI is a closed-loop system that (1) reads the emotional trajectory
of an ongoing conversation, (2) predicts whether the user will escalate two
turns into the future, (3) attributes that prediction to specific dialogue
features via SHAP, and (4) injects matching natural-language directives into
the system prompt of Mistral-7B at inference time so the model's *next* reply
de-escalates rather than amplifies the situation.

| Component | Implementation |
|---|---|
| Data | WildChat-1M (50K conversations, streamed) |
| Emotion annotation | GoEmotions BERT (28 classes) → VAD via lookup |
| Escalation predictor | XGBoost on 14 rolling-window features (AUC ≈ 0.82) |
| Sequence benchmark | BiLSTM over raw VAD sequences (AUC ≈ 0.78) |
| Explainability | SHAP TreeExplainer (per-instance + global) |
| Generator | Mistral-7B-Instruct-v0.3 in 4-bit NF4 (~8.5 GB VRAM) |
| Dashboard | 5-tab Streamlit app (chat, UMAP, timeline, heatmap, evaluation) |

## Repo layout

```
expression-ai/
├── README.md
├── requirements.txt
├── notebooks/
│   └── Expression_AI.ipynb     # one orchestration notebook, no duplicate logic
├── src/                        # all heavy code lives here
│   ├── __init__.py
│   ├── config.py               # paths, feature names, VAD table, rule bank
│   ├── emotion.py              # GoEmotions BERT + VAD projection
│   ├── features.py             # rolling-window engineering + live FV builder
│   ├── models.py               # XGBoost + BiLSTM train/load + split
│   ├── intervention.py         # risk, SHAP→directives, Mistral generation
│   └── evaluation.py           # writes CSV/JSON/PNG read by the dashboard
└── streamlit_app.py            # UI only — imports from src/
```

The notebook and the Streamlit app **share the same `src/` package** — there
is no duplicated logic and no drift between training and serving.

## Quickstart

### A) Run in Colab (recommended — GPU + ngrok included)

1. Open `notebooks/Expression_AI.ipynb` in Colab.
2. Set the runtime to a GPU (T4 works, A100 is faster).
3. Replace `REPO_URL` in the first cell with your fork.
4. Paste your ngrok auth token into the last section.
5. **Run All.** The notebook will:
   - Stream 50K WildChat conversations
   - Annotate them with GoEmotions → VAD
   - Train XGBoost + BiLSTM (conversation-level split, no leakage)
   - Compute SHAP, ROC, PR, calibration, classification report
   - Run a scripted closed-loop chat demo
   - Launch the Streamlit dashboard at a public ngrok URL

### B) Run locally

```bash
git clone https://github.com/<your-org>/expression-ai.git
cd expression-ai
pip install -r requirements.txt

# Set the artefact dir
export EXPR_AI_DATA_DIR=$PWD/data

# Run the notebook to populate $EXPR_AI_DATA_DIR with trained models +
# evaluation artefacts.
jupyter lab notebooks/Expression_AI.ipynb

# Then start the dashboard
streamlit run streamlit_app.py
```

The Streamlit app auto-detects when artefacts are missing and falls back to a
**demo mode** with synthetic data and a heuristic risk scorer, so you can
still demo the UI without a GPU.

## How the closed loop works

Each user turn flows through this pipeline (every function lives in `src/`):

```
        ┌─────────────┐    ┌────────────┐    ┌──────────────┐
user ──►│ src.emotion │──►│src.features│──►│src.intervention│──► reply
        │  annotate   │    │  build_FV  │    │ predict_risk │
        │             │    │            │    │  ↓ if risky  │
        │             │    │            │    │  SHAP→dirs   │
        │             │    │            │    │  ↓           │
        │             │    │            │    │ Mistral-7B   │
        └─────────────┘    └────────────┘    └──────────────┘
```

`src.intervention.chat_step()` is the single-call entry point used by both
the notebook's scripted demo and the Streamlit chat tab.

## Reproducibility

- All paths come from `src.config`; override with `EXPR_AI_DATA_DIR`.
- Seeds are fixed (`random_state=42`) for sklearn, train/test split, UMAP,
  and KMeans.
- The Streamlit `Evaluation` tab reads CSV/JSON/PNG produced by
  `src.evaluation.run_evaluation()` — **no hardcoded numbers**.
- Conversation-level (not turn-level) train/test split — this is essential
  for the rolling-window framing or AUC inflates by ~0.07 from leakage.

## Headline results

(See `report/` for the full ACM-format write-up.)

| Model | AUC | F1 (escalating) | AP | ECE |
|---|---|---|---|---|
| Baseline (−Δv) | 0.62 | 0.41 | 0.40 | 0.14 |
| BiLSTM over raw VAD | 0.78 | 0.59 | 0.55 | 0.09 |
| **XGBoost (rolling)** | **0.82** | **0.65** | **0.62** | **0.06** |

**Closed-loop A/B (50 escalating test conversations):**
mean Δ-valence = **+0.27**, ~70 % of conversations move toward warmer replies,
paired *t*-test *p* < 0.05.

## Acknowledgements

- WildChat-1M corpus (Allen AI / Yi-Chia et al., ICLR 2024)
- GoEmotions (Demszky et al., ACL 2020)
- Mistral AI for the open-weight 7B-Instruct model
- SHAP (Lundberg & Lee, NeurIPS 2017)
