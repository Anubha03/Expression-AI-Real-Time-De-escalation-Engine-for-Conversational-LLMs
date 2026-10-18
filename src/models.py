"""XGBoost (primary) + BiLSTM (sequence baseline) for escalation prediction.

Both models are trained with a conversation-level train/test split to avoid
leakage between overlapping rolling windows. The XGBoost classifier is the
production model — fast, SHAP-explainable. The BiLSTM is kept around as a
benchmark that the BiLSTM doesn't beat the engineered features at
window_size=4 (the result the report cites).
"""

from __future__ import annotations
import os
import pickle
from typing import Tuple, Sequence

import numpy as np
import pandas as pd

from .config import FEATURE_COLS, WINDOW_SIZE, XGB_PATH, BILSTM_PATH


# ── XGBoost ─────────────────────────────────────────────────────────────────

def train_xgboost(
    df_feat: pd.DataFrame,
    train_convs: Sequence,
    test_convs: Sequence,
    n_estimators: int = 300,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    use_cuda: bool = True,
):
    """Train and return the XGBoost classifier on the rolling-window matrix.

    Returns
    -------
    clf : xgboost.XGBClassifier
    X_test, y_test, y_prob : numpy arrays, for downstream evaluation
    """
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, classification_report

    X = df_feat[FEATURE_COLS].fillna(0)
    y = df_feat["is_escalating"]
    train_mask = df_feat["conv_id"].isin(train_convs)
    test_mask  = df_feat["conv_id"].isin(test_convs)
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        tree_method="hist",
        device="cuda" if use_cuda else "cpu",
    )
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.4).astype(int)
    auc    = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, target_names=["not-escalating", "escalating"])

    print(f"XGBoost AUC (rolling 2-turn-ahead): {auc:.4f}")
    print(report)
    return clf, X_test.values, y_test.values, y_prob


def save_xgboost(clf, path: str = XGB_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clf.save_model(path)
    return path


def load_xgboost(path: str = XGB_PATH):
    """Try JSON model first; fall back to a pickle next to it."""
    import xgboost as xgb
    if os.path.exists(path):
        clf = xgb.XGBClassifier()
        clf.load_model(path)
        return clf
    pkl_path = os.path.splitext(path)[0] + ".pkl"
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    return None


# ── BiLSTM over raw VAD sequences ───────────────────────────────────────────

def _build_vad_dataset(df_turns: pd.DataFrame, df_feat_subset: pd.DataFrame, window_size: int):
    """Build a list of (sequence, label) pairs from raw VAD per turn."""
    import torch
    samples = []
    for _, row in df_feat_subset.iterrows():
        cid = row["conv_id"]
        grp = df_turns[df_turns.conv_id == cid]
        usr = grp[grp.role == "user"].sort_values("turn_idx").reset_index(drop=True)
        w   = int(row["window_start"])
        seq = usr.iloc[w:w + window_size][["valence", "arousal", "dominance"]].values.astype(np.float32)
        if len(seq) < window_size:
            pad = np.zeros((window_size - len(seq), 3), dtype=np.float32)
            seq = np.vstack([pad, seq])
        samples.append((seq, int(row["is_escalating"])))
    return samples


def train_bilstm(
    df_turns: pd.DataFrame,
    df_feat: pd.DataFrame,
    train_convs: Sequence,
    test_convs: Sequence,
    window_size: int = WINDOW_SIZE,
    sample_n: int = 8000,
    epochs: int = 12,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    lr: float = 1e-3,
):
    """Train the BiLSTM benchmark and return (model, test_auc)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from sklearn.metrics import roc_auc_score

    class _Set(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.samples = samples
        def __len__(self): return len(self.samples)
        def __getitem__(self, i):
            s, l = self.samples[i]
            return torch.tensor(s), torch.tensor(l, dtype=torch.float32)

    class BiLSTMEscalation(nn.Module):
        def __init__(self, input_dim=3, hidden=hidden_dim, layers=num_layers, drop=dropout):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers,
                                batch_first=True, bidirectional=True, dropout=drop)
            self.fc   = nn.Sequential(
                nn.Linear(hidden * 2, 32), nn.ReLU(),
                nn.Dropout(drop), nn.Linear(32, 1), nn.Sigmoid(),
            )
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    df_samp = df_feat.sample(min(sample_n, len(df_feat)), random_state=42).reset_index(drop=True)
    tr_s = df_samp[df_samp.conv_id.isin(train_convs)]
    te_s = df_samp[df_samp.conv_id.isin(test_convs)]
    tr_dl = DataLoader(_Set(_build_vad_dataset(df_turns, tr_s, window_size)), batch_size=64, shuffle=True)
    te_dl = DataLoader(_Set(_build_vad_dataset(df_turns, te_s, window_size)), batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = BiLSTMEscalation().to(device)
    optim  = torch.optim.Adam(model.parameters(), lr=lr)
    crit   = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        tot = 0
        for seqs, labels in tr_dl:
            seqs, labels = seqs.to(device), labels.to(device)
            optim.zero_grad()
            loss = crit(model(seqs), labels)
            loss.backward()
            optim.step()
            tot += loss.item()
        if (epoch + 1) % 3 == 0:
            print(f"  BiLSTM epoch {epoch+1}/{epochs}  loss={tot/len(tr_dl):.4f}")

    model.eval()
    probs, labels_all = [], []
    with torch.no_grad():
        for seqs, labels in te_dl:
            probs.extend(model(seqs.to(device)).cpu().numpy())
            labels_all.extend(labels.numpy())

    auc = roc_auc_score(labels_all, probs)
    print(f"BiLSTM AUC: {auc:.4f}")
    return model, auc


def save_bilstm(model, path: str = BILSTM_PATH) -> str:
    import torch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def conversation_level_split(df_feat: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Conversation-level 80/20 split — required to avoid leakage between
    overlapping rolling windows of the same conversation.
    """
    from sklearn.model_selection import train_test_split
    conv_ids = df_feat["conv_id"].unique()
    train_convs, test_convs = train_test_split(
        conv_ids, test_size=test_size, random_state=random_state
    )
    return train_convs, test_convs
