import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

ROOT = Path(__file__).resolve().parent

# 1. Load legacy model to extract the authentic 734 training observations
old_model = joblib.load(ROOT / "knn_heart_model.pkl")
old_scaler = joblib.load(ROOT / "heart_scaler.pkl")

X_scaled = old_model._fit_X
y = np.array(old_model._y)
features = list(old_scaler.feature_names_in_)

print(f"Loaded {X_scaled.shape[0]} training samples across {len(features)} features.")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 2. Benchmark Candidates
models = {
    "Original Baseline KNN": KNeighborsClassifier(n_neighbors=5),
    "Optimized KNN": KNeighborsClassifier(n_neighbors=15, weights="distance", metric="manhattan"),
    "Logistic Regression": LogisticRegression(C=0.8, max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_split=4, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.04, max_depth=3, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=150, max_depth=6, min_samples_split=3, random_state=42)
}

print("\n--- 5-Fold Cross Validation Benchmark ---")
for name, clf in models.items():
    acc = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy").mean()
    auc = cross_val_score(clf, X_scaled, y, cv=cv, scoring="roc_auc").mean()
    print(f"{name:25}: Accuracy = {acc*100:.2f}% | ROC-AUC = {auc*100:.2f}%")

# 3. Create Weighted Soft-Voting Ensemble (State-of-the-Art)
ensemble = VotingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(n_estimators=250, max_depth=6, min_samples_split=3, random_state=42)),
        ("gb", GradientBoostingClassifier(n_estimators=130, learning_rate=0.04, max_depth=3, random_state=42)),
        ("et", ExtraTreesClassifier(n_estimators=200, max_depth=6, random_state=42)),
        ("lr", LogisticRegression(C=0.8, max_iter=1000, random_state=42)),
        ("knn", KNeighborsClassifier(n_neighbors=15, weights="distance", metric="manhattan"))
    ],
    voting="soft",
    weights=[3, 3, 2, 2, 1]
)

ens_acc = cross_val_score(ensemble, X_scaled, y, cv=cv, scoring="accuracy").mean()
ens_auc = cross_val_score(ensemble, X_scaled, y, cv=cv, scoring="roc_auc").mean()
print(f"\nFinal Weighted Ensemble: Accuracy = {ens_acc*100:.2f}% | ROC-AUC = {ens_auc*100:.2f}%")

# 4. Train final models on entire dataset
ensemble.fit(X_scaled, y)

# We also keep a clean StandardScaler re-fitted on the unscaled data for consistency
X_raw = old_scaler.inverse_transform(X_scaled)
new_scaler = StandardScaler()
new_scaler.fit(X_raw)
new_scaler.feature_names_in_ = np.array(features, dtype=object)

# 5. Save the upgraded models
joblib.dump(ensemble, ROOT / "knn_heart_model.pkl")
joblib.dump(new_scaler, ROOT / "heart_scaler.pkl")

print("\nSuccessfully upgraded and saved knn_heart_model.pkl and heart_scaler.pkl!")
