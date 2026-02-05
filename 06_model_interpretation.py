import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

import shap

print("Chargement du modèle optimisé...")
best_model = joblib.load("output/models/best_model_optuna.pkl")

print("Chargement du dataset propre...")
df = pd.read_csv("output/clean_dataset.csv", sep=";")

# Utiliser grav_bin (colonne correcte)
X = df.drop("grav_bin", axis=1)
y = df["grav_bin"]

print("Split train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Prédiction du modèle final...")
y_pred = best_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

print("\nAccuracy :", round(acc, 4))
print("F1-score :", round(f1, 4))

#MATRICE DE CONFUSION
print("Création de la matrice de confusion...")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Non grave", "Grave"],
    yticklabels=["Non grave", "Grave"]
)
plt.xlabel("Prédictions")
plt.ylabel("Valeurs réelles")
plt.title("Matrice de confusion - Modèle optimisé")
os.makedirs("output/plots", exist_ok=True)
plt.savefig("output/plots/confusion_matrix_optuna.png", dpi=300, bbox_inches="tight")
plt.close()

#FEATURE IMPORTANCES
print("Calcul des importances des variables...")

try:
    importances = best_model.feature_importances_
    feature_importances = pd.DataFrame({
        "feature": X.columns,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    feature_importances.to_csv("output/feature_importances_optuna.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importances.head(20), x="importance", y="feature")
    plt.title("Top 20 feature importances (modèle optimisé)")
    plt.tight_layout()
    plt.savefig("output/plots/feature_importances_optuna.png", dpi=300)
    plt.close()

    print("Feature importances sauvegardées.")

except:
    print("Le modèle ne fournit pas feature_importances_.")

#SHAP
print("Calcul des SHAP values (échantillon réduit)...")

X_small = X_test.sample(3000, random_state=42)

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_small)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_small, show=False)
plt.tight_layout()
plt.savefig("output/plots/shap_summary.png", dpi=300)
plt.close()

print("SHAP summary sauvegardé dans output/plots/shap_summary.png")

print("\nANALYSE TERMINÉE")
