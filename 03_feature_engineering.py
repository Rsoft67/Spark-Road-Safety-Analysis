import pandas as pd
import numpy as np
import os

# Chargement du dataset fusionné
df = pd.read_csv("output/merged_dataset.csv", sep=";")
print("Dataset chargé :", df.shape)

# 1. Nettoyage et conversions simples
# Heure (HHMM -> HH)
df["heure"] = df["hrmn"].astype(str).str[:2]
df["heure"] = pd.to_numeric(df["heure"], errors="coerce").fillna(-1).astype(int)

# Mois
df["mois"] = pd.to_numeric(df["mois"], errors="coerce").fillna(0).astype(int)

# Weekend (samedi=6, dimanche=7)
df["weekend"] = df["jour"].isin([6, 7]).astype(int)

# Nuit
df["is_night"] = df["lum"].isin([5, 6]).astype(int)

# Âge
df["age"] = df["annee"] - df["an_nais"]
df["age"] = df["age"].clip(lower=0, upper=100).fillna(df["age"].median())

# 2. Agrégations par accident
# Nombre d'usagers
df["nb_usagers"] = df.groupby("Num_Acc")["id_usager"].transform("count")

# Nombre de véhicules
df["nb_vehicules"] = df.groupby("Num_Acc")["id_vehicule_x"].transform("nunique")

# 3. Variables importantes
cat_vars = ["choc", "manv", "catv", "place", "secu1", "col", "atm", "catr", "catu", "sexe", "trajet"]

for col in cat_vars:
    df[col] = df[col].astype("category").cat.codes

# Vitesse limite
df["vma"] = pd.to_numeric(df["vma"], errors="coerce").fillna(df["vma"].median())

# 4. Interactions pertinentes
conditions_humides = [2, 3, 4]

df["nuit_pluie"] = ((df["is_night"] == 1) &
                    (df["atm"].isin(conditions_humides))).astype(int)

df["rapide_pluie"] = ((df["catr"].isin([1, 2])) &
                      (df["atm"].isin(conditions_humides))).astype(int)

df["moto"] = (df["catv"] == 2).astype(int)
df["vehicule_lourd"] = (df["catv"].isin([7, 8])).astype(int)
df["choc_frontal"] = (df["col"] == 1).astype(int)
df["sans_secu"] = (df["secu1"] == 0).astype(int)

# 5. Features finales
features = [
    "mois", "heure", "weekend", "is_night",
    "age", "catu", "sexe", "trajet",
    "catv", "manv", "choc", "place", "col",
    "catr", "atm", "vma",
    "nb_usagers", "nb_vehicules",
    "nuit_pluie", "rapide_pluie", "moto",
    "vehicule_lourd", "choc_frontal", "sans_secu"
]

# 6. Création de la cible BINAIRE
df = df[df["grav"].notna()]
df = df[df["grav"].between(1, 4)]
df["grav"] = df["grav"].astype(int)

# Non grave = 1 (indemne + blessé léger)
# Grave = 2 ou 3 (hospitalisé + tué)
df["grav_bin"] = df["grav"].apply(lambda g: 1 if g >= 3 else 0)

# jeu final
df_final = df[features + ["grav_bin"]]

print("Dataset final (model-ready) :", df_final.shape)

# Sauvegarde
os.makedirs("output", exist_ok=True)
df_final.to_csv("output/clean_dataset.csv", index=False, sep=";")
print("clean_dataset.csv sauvegardé.")
