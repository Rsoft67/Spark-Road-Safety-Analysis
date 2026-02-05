import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib

print("Chargement du dataset propre...")
df = pd.read_csv("output/clean_dataset.csv", sep=";")

# 1. Target = grav_bin
X = df.drop("grav_bin", axis=1)
y = df["grav_bin"]

# 2. Train/Test split
print("Séparation train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. SMOTE binaire
print("Application de SMOTE binaire...")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("Avant SMOTE :", np.bincount(y_train))
print("Après SMOTE :", np.bincount(y_train_res))

# 4. Modèles
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        learning_rate=0.1,
        max_depth=8,
        n_estimators=300,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    ),

    "LightGBM": LGBMClassifier(
        objective="binary",
        learning_rate=0.1,
        n_estimators=300,
        num_leaves=31,
        random_state=42,
        force_row_wise=True
    )
}

results = {}

# 5. Entraînement & Évaluation
for name, model in models.items():
    print(f"\n=== Entraînement du modèle : {name} ===")

    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred))

    results[name] = {
        "model": model,
        "accuracy": acc,
        "f1": f1
    }


# 6. Sélection du meilleur modèle
best_model_name = max(results, key=lambda m: results[m]["f1"])
best_model = results[best_model_name]["model"]

print(f"\nMeilleur modèle : {best_model_name}")
print(f"F1-score : {results[best_model_name]['f1']:.4f}")

# 7. Sauvegarde
os.makedirs("output/models", exist_ok=True)
joblib.dump(best_model, f"output/models/best_model_{best_model_name}.pkl")

print(f"Modèle sauvegardé dans : output/models/best_model_{best_model_name}.pkl")
