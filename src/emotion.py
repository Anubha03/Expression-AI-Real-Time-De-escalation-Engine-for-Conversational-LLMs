"""GoEmotions BERT + VAD projection.

Two responsibilities:
1. Load the GoEmotions text-classification pipeline (cached).
2. Convert the 28-class probability vector to a continuous (V, A, D) triple
   via the lookup table in `config.EMOTION_VAD`.

The same VAD projection is used at training time (offline annotation of all
WildChat turns) and at inference time (one turn at a time in the chat loop).
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any, Iterable

import numpy as np
import pandas as pd

from .config import EMOTION_VAD, GOEMO_MODEL_ID, BATCH

_PIPE = None


def load_pipeline(device: str | int = "auto"):
    """Return (and cache) the HuggingFace text-classification pipeline.

    Parameters
    ----------
    device : "auto", "cuda", "cpu", or an int device index. "auto" picks 0
        when CUDA is available, -1 otherwise.
    """
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    from transformers import pipeline  # local import: heavy
    import torch

    if device == "auto":
        device = 0 if torch.cuda.is_available() else -1
    elif device == "cuda":
        device = 0
    elif device == "cpu":
        device = -1

    _PIPE = pipeline(
        "text-classification",
        model=GOEMO_MODEL_ID,
        tokenizer=GOEMO_MODEL_ID,
        top_k=None,
        device=device,
        truncation=True,
        max_length=512,
    )
    return _PIPE


def scores_to_vad(label_scores: List[Dict[str, Any]]) -> Tuple[float, float, float]:
    """Probability-weighted average VAD over GoEmotions labels.

    Each item in `label_scores` must be `{"label": str, "score": float}`.
    Labels not present in `EMOTION_VAD` are ignored. If the total of
    matched probability mass is 0, returns a neutral (0, 0.5, 0.5).
    """
    V = A = D = total = 0.0
    for item in label_scores:
        lbl = item["label"].lower()
        sc  = item["score"]
        if lbl in EMOTION_VAD:
            v, a, d = EMOTION_VAD[lbl]
            V += v * sc; A += a * sc; D += d * sc; total += sc
    if total == 0:
        return 0.0, 0.5, 0.5
    return V / total, A / total, D / total


def top_emotion(label_scores: List[Dict[str, Any]]) -> str:
    """Return the highest-scoring GoEmotions label."""
    return max(label_scores, key=lambda x: x["score"])["label"]

def annotate_text(text: str, pipe=None) -> Tuple[float, float, float, str]:
    """Annotate one piece of text. Loads the pipeline lazily if not passed."""
    pipe = pipe or load_pipeline()
    result = pipe([text if text and text.strip() else "."])[0]
    v, a, d = scores_to_vad(result)
    return v, a, d, top_emotion(result)

class _TurnDataset:
    """Tiny Dataset wrapper for the HuggingFace pipeline iterator."""
    def __init__(self, texts: Iterable[str]):
        self.texts = [t if (t and t.strip()) else "." for t in texts]
    def __len__(self):  return len(self.texts)
    def __getitem__(self, i): return self.texts[i]


def annotate_dataframe(
    df_turns: pd.DataFrame,
    text_col: str = "content",
    batch_size: int = BATCH,
    pipe=None,
    progress: bool = True,
) -> pd.DataFrame:
    pipe = pipe or load_pipeline()
    raw_texts = df_turns[text_col].fillna("").astype(str).tolist()
    texts = [t if (t and t.strip()) else "." for t in raw_texts]

    iterator = texts
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(texts, desc="Emotion inference")
        except ImportError:
            pass

    Vs, As, Ds, Tops = [], [], [], []
    for text in iterator:
        result = pipe(text)[0]
        v, a, d = scores_to_vad(result)
        Vs.append(v); As.append(a); Ds.append(d); Tops.append(top_emotion(result))

    out = df_turns.copy()
    out["valence"]     = Vs
    out["arousal"]     = As
    out["dominance"]   = Ds
    out["top_emotion"] = Tops
    return out
