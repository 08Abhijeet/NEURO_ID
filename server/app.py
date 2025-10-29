import os
import io
import json
from typing import List, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load


# ---------------------------
# Config
# ---------------------------
SAMPLES_PER_EPOCH = 400  # 2 seconds of data (200 Hz * 2s)
NUM_CHANNELS = 4

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
USERS_DB = os.path.join(BASE_DIR, "users.json")
os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------
# Utils - data loading & features
# ---------------------------
def read_csv_eeg(file_bytes: bytes) -> np.ndarray:
    """Read EEG CSV/TXT and return np.ndarray of shape (N, C) using last 4 columns as EEG.
    Falls back to all columns if exactly 4.
    """
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception:
        # try tab/space delimited
        df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python")

    if df.shape[1] >= NUM_CHANNELS + 1:
        data = df.iloc[:, -NUM_CHANNELS:].values
    elif df.shape[1] == NUM_CHANNELS:
        data = df.values
    else:
        raise ValueError("Uploaded EEG must have at least 4 channels or include them among last columns")
    return data.astype(np.float64, copy=False)


def slice_epochs(eeg: np.ndarray, samples_per_epoch: int = SAMPLES_PER_EPOCH) -> np.ndarray:
    epochs: List[np.ndarray] = []
    n = eeg.shape[0]
    for start in range(0, n - samples_per_epoch + 1, samples_per_epoch):
        epochs.append(eeg[start : start + samples_per_epoch, :])
    if not epochs:
        raise ValueError("EEG file too short for one epoch")
    return np.stack(epochs, axis=0)


def extract_features(epochs: np.ndarray) -> np.ndarray:
    """Replicates the statistical + simple spectral features used in the notebook."""
    features: List[List[float]] = []
    for epoch in epochs:
        epoch_feats: List[float] = []
        for ch in range(epoch.shape[1]):
            x = epoch[:, ch]
            epoch_feats.extend([
                float(np.mean(x)),
                float(np.std(x)),
                float(np.var(x)),
                float(np.min(x)),
                float(np.max(x)),
                float(np.median(x)),
                float(np.percentile(x, 25)),
                float(np.percentile(x, 75)),
                float(np.sum(np.abs(x))),
                float(np.sum(x**2)),
            ])
            # frequency domain (rough bins)
            fft = np.fft.fft(x)
            ps = np.abs(fft) ** 2
            delta = float(np.sum(ps[1:9]))
            theta = float(np.sum(ps[9:17]))
            alpha = float(np.sum(ps[17:33]))
            beta = float(np.sum(ps[33:81]))
            epoch_feats.extend([
                delta,
                theta,
                alpha,
                beta,
                float(alpha / (delta + 1e-8)),
                float(beta / (alpha + 1e-8)),
            ])
        features.append(epoch_feats)
    return np.asarray(features, dtype=np.float64)


# ---------------------------
# Model I/O
# ---------------------------
def model_paths(user_id: str) -> Tuple[str, str]:
    model_path = os.path.join(MODELS_DIR, f"{user_id}.model.joblib")
    scaler_path = os.path.join(MODELS_DIR, f"{user_id}.scaler.joblib")
    return model_path, scaler_path


def save_user(user_id: str, name: str, email: str):
    users = {}
    if os.path.exists(USERS_DB):
        try:
            users = json.load(open(USERS_DB, "r", encoding="utf-8"))
        except Exception:
            users = {}
    users[user_id] = {"name": name, "email": email}
    json.dump(users, open(USERS_DB, "w", encoding="utf-8"), indent=2)


# ---------------------------
# Training (Signup)
# ---------------------------
def train_user_model(user_id: str, eeg_files: List[UploadFile]) -> dict:
    # collect epochs from all files
    all_epochs: List[np.ndarray] = []
    for f in eeg_files:
        file_bytes = f.file.read()
        eeg = read_csv_eeg(file_bytes)
        epochs = slice_epochs(eeg)
        all_epochs.append(epochs)
    X_epochs = np.concatenate(all_epochs, axis=0)
    X_features = extract_features(X_epochs)

    # Use One-Class SVM (learn user's cluster) for robust auth without impostor data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    clf = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)  # ~5% expected outliers
    clf.fit(X_scaled)

    # Persist
    model_path, scaler_path = model_paths(user_id)
    dump(clf, model_path)
    dump(scaler, scaler_path)

    return {"epochs": int(X_epochs.shape[0]), "features": int(X_features.shape[1])}


# ---------------------------
# Authentication (Signin)
# ---------------------------
def authenticate(user_id: str, eeg_file: UploadFile) -> dict:
    model_path, scaler_path = model_paths(user_id)
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        raise HTTPException(status_code=404, detail="Model not found for user. Please sign up first.")

    clf: OneClassSVM = load(model_path)
    scaler: StandardScaler = load(scaler_path)

    eeg = read_csv_eeg(eeg_file.file.read())
    epochs = slice_epochs(eeg)
    X = extract_features(epochs)
    Xs = scaler.transform(X)

    # One-Class SVM: predict {1: inlier/authenticated, -1: outlier}
    pred = clf.predict(Xs)
    inlier_rate = float((pred == 1).mean())
    authenticated = bool(inlier_rate >= 0.5)

    return {
        "authenticated": authenticated,
        "inlier_rate": inlier_rate,
        "epochs_evaluated": int(X.shape[0])
    }


# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="NeuroID Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "null"],  # include file:// origin (null)
    allow_origin_regex=".*",       # permissive for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/auth/signup")
async def api_signup(
    name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),  # not stored here; add real auth in production
    eeg_files: List[UploadFile] = File(...),
):
    user_id = email.strip().lower()
    if not eeg_files:
        raise HTTPException(status_code=400, detail="Please upload at least one EEG file")
    stats = train_user_model(user_id, eeg_files)
    save_user(user_id, name=name, email=email)
    return JSONResponse({"message": "Model trained successfully", "user": user_id, **stats})


@app.post("/api/auth/signin")
async def api_signin(
    email: str = Form(...),
    eeg_file: UploadFile = File(...),
):
    user_id = email.strip().lower()
    result = authenticate(user_id, eeg_file)
    return JSONResponse({"user": user_id, **result})


# Local dev entry
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


