import pandas as pd
import numpy as np
import optuna
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

print("Chargement du dataset propre (binaire)...")
df = pd.read_csv("output/clean_dataset.csv", sep=";")

X = df.drop("grav_bin", axis=1)
y = df["grav_bin"]

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# SMOTE binaire
print("Application SMOTE binaire...")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("Après SMOTE :", np.bincount(y_train_res))

# OBJECTIF OPTUNA
def objective(trial):

    model_name = trial.suggest_categorical("model", ["RF", "XGB", "LGBM"])

    if model_name == "RF":
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 150, 400),
            max_depth=trial.suggest_int("max_depth", 5, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            n_jobs=1,
            random_state=42
        )

    elif model_name == "XGB":
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            n_estimators=trial.suggest_int("n_estimators", 300, 800),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            n_jobs=1,
            random_state=42
        )

    else:  # LGBM
        model = LGBMClassifier(
            objective="binary",
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2),
            n_estimators=trial.suggest_int("n_estimators", 300, 800),
            num_leaves=trial.suggest_int("num_leaves", 20, 120),
            max_depth=trial.suggest_int("max_depth", 5, 20),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            random_state=42
        )

    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_val)

    return f1_score(y_val, y_pred)

# LANCEMENT OPTUNA
print("Lancement Optuna...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=12)

print("\nMeilleur score F1 :", study.best_value)
print("Meilleurs paramètres :", study.best_params)

best_params = study.best_params
best_model_name = best_params["model"]

print(f"\nReconstruction du meilleur modèle : {best_model_name}")

# Reconstruction finale
if best_model_name == "RF":
    best_model = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        n_jobs=-1,
        random_state=42
    )

elif best_model_name == "XGB":
    best_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        learning_rate=best_params["learning_rate"],
        max_depth=best_params["max_depth"],
        n_estimators=best_params["n_estimators"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
        n_jobs=-1,
        random_state=42
    )

else:
    best_model = LGBMClassifier(
        objective="binary",
        learning_rate=best_params["learning_rate"],
        n_estimators=best_params["n_estimators"],
        num_leaves=best_params["num_leaves"],
        max_depth=best_params["max_depth"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
        random_state=42
    )

best_model.fit(X_train_res, y_train_res)

# Sauvegarde finale
os.makedirs("output/models", exist_ok=True)
joblib.dump(best_model, "output/models/best_model_optuna.pkl")

print("\nModèle final Optuna sauvegardé.")
