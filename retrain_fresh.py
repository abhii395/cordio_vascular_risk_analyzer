"""
retrain_fresh.py
Retrain the ensemble model from heart.csv (UCI format) using the current
scikit-learn version so the .pkl files are compatible with Vercel.

The UCI CSV has columns:
  age, sex(0/1), cp(0-3), trestbps, chol, fbs(0/1),
  restecg(0-2), thalach, exang(0/1), oldpeak, slope(0-2), ca, thal, target

app.py expects one-hot features like:
  Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak,
  Sex_M, Sex_F,
  ChestPainType_ASY, ChestPainType_ATA, ChestPainType_NAP, ChestPainType_TA,
  RestingECG_Normal, RestingECG_ST, RestingECG_LVH,
  ExerciseAngina_N, ExerciseAngina_Y,
  ST_Slope_Up, ST_Slope_Flat, ST_Slope_Down
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               VotingClassifier, ExtraTreesClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

ROOT = Path(__file__).resolve().parent

# ── 1. Load & map UCI → app.py feature space ──────────────────────────────
df = pd.read_csv(ROOT / "heart.csv")
print(f"Loaded {len(df)} rows.  Columns: {list(df.columns)}")

# Map sex: 1=M, 0=F
df["Sex_M"] = (df["sex"] == 1).astype(int)
df["Sex_F"] = (df["sex"] == 0).astype(int)

# Map chest pain type: UCI 0=typical angina,1=atypical,2=non-anginal,3=asymptomatic
cp_map = {0: "TA", 1: "ATA", 2: "NAP", 3: "ASY"}
df["cp_str"] = df["cp"].map(cp_map)
for code in ["ASY", "ATA", "NAP", "TA"]:
    df[f"ChestPainType_{code}"] = (df["cp_str"] == code).astype(int)

# Map restecg: 0=Normal, 1=ST, 2=LVH
ecg_map = {0: "Normal", 1: "ST", 2: "LVH"}
df["ecg_str"] = df["restecg"].map(ecg_map)
for code in ["Normal", "ST", "LVH"]:
    df[f"RestingECG_{code}"] = (df["ecg_str"] == code).astype(int)

# Map exercise angina: 1=Y, 0=N
df["ExerciseAngina_Y"] = (df["exang"] == 1).astype(int)
df["ExerciseAngina_N"] = (df["exang"] == 0).astype(int)

# Map slope: UCI 0=downsloping,1=flat,2=upsloping
slope_map = {0: "Down", 1: "Flat", 2: "Up"}
df["slope_str"] = df["slope"].map(slope_map)
for code in ["Up", "Flat", "Down"]:
    df[f"ST_Slope_{code}"] = (df["slope_str"] == code).astype(int)

# Rename numeric columns
df["Age"]        = df["age"]
df["RestingBP"]  = df["trestbps"]
df["Cholesterol"]= df["chol"]
df["FastingBS"]  = df["fbs"]
df["MaxHR"]      = df["thalach"]
df["Oldpeak"]    = df["oldpeak"]

# ── 2. Build feature matrix in the EXACT order app.py will reconstruct ────
# This must match what app.py puts in `raw` dict keys
FEATURE_COLS = [
    "Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak",
    "Sex_M", "Sex_F",
    "ChestPainType_ASY", "ChestPainType_ATA", "ChestPainType_NAP", "ChestPainType_TA",
    "RestingECG_Normal", "RestingECG_ST", "RestingECG_LVH",
    "ExerciseAngina_N", "ExerciseAngina_Y",
    "ST_Slope_Down", "ST_Slope_Flat", "ST_Slope_Up",
]

X_raw = df[FEATURE_COLS].values.astype(float)
y     = df["target"].values
print(f"Feature matrix: {X_raw.shape}  |  Positive labels: {y.sum()}/{len(y)}")

# ── 3. Scale ──────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
scaler.feature_names_in_ = np.array(FEATURE_COLS, dtype=object)

# ── 4. Cross-validate ─────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

ensemble = VotingClassifier(
    estimators=[
        ("rf",  RandomForestClassifier(n_estimators=250, max_depth=6,
                                        min_samples_split=3, random_state=42)),
        ("gb",  GradientBoostingClassifier(n_estimators=130, learning_rate=0.04,
                                            max_depth=3, random_state=42)),
        ("et",  ExtraTreesClassifier(n_estimators=200, max_depth=6, random_state=42)),
        ("lr",  LogisticRegression(C=0.8, max_iter=1000, random_state=42)),
        ("knn", KNeighborsClassifier(n_neighbors=15, weights="distance",
                                      metric="manhattan")),
    ],
    voting="soft",
    weights=[3, 3, 2, 2, 1],
)

print("\nRunning 5-fold cross-validation...")
acc = cross_val_score(ensemble, X_scaled, y, cv=cv, scoring="accuracy").mean()
auc = cross_val_score(ensemble, X_scaled, y, cv=cv, scoring="roc_auc").mean()
print(f"  Accuracy : {acc*100:.2f}%")
print(f"  ROC-AUC  : {auc*100:.2f}%")

# ── 5. Train on full dataset & save ───────────────────────────────────────
print("\nFitting on full dataset and saving...")
ensemble.fit(X_scaled, y)

joblib.dump(ensemble, ROOT / "knn_heart_model.pkl")
joblib.dump(scaler,   ROOT / "heart_scaler.pkl")

api_dir = ROOT / "api"
api_dir.mkdir(exist_ok=True)
joblib.dump(ensemble, api_dir / "knn_heart_model.pkl")
joblib.dump(scaler,   api_dir / "heart_scaler.pkl")

sz = (ROOT / "knn_heart_model.pkl").stat().st_size // 1024
print(f"\n✅  knn_heart_model.pkl  saved  ({sz} KB)")
print(f"✅  heart_scaler.pkl     saved")
print(f"✅  Copies written to api/")
print(f"\nFeatures ({len(FEATURE_COLS)}): {FEATURE_COLS}")
