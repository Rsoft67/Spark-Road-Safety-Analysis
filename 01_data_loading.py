import pandas as pd
import os

def load_year_data(year, base_path="data"):
    """
    Charge les fichiers d'une année.
    """
    print(f"\nChargement des fichiers pour {year}")

    year_path = os.path.join(base_path, str(year))

    # Dictionnaire des vrais noms pour CHAQUE année
    filenames = {
        2021: {
            "carac": "carcteristiques-2021.csv",
            "lieux": "lieux-2021.csv",
            "veh": "vehicules-2021.csv",
            "usagers": "usagers-2021.csv"
        },
        2022: {
            "carac": "carcteristiques-2022.csv",
            "lieux": "lieux-2022.csv",
            "veh": "vehicules-2022.csv",
            "usagers": "usagers-2022.csv"
        },
        2023: {
            "carac": "caract-2023.csv",
            "lieux": "lieux-2023.csv",
            "veh": "vehicules-2023.csv",
            "usagers": "usagers-2023.csv"
        }
    }

    f = filenames[year]

    # Lecture des 4 fichiers
    df_carac = pd.read_csv(os.path.join(year_path, f["carac"]), sep=';', low_memory=False)
    df_lieux = pd.read_csv(os.path.join(year_path, f["lieux"]), sep=';', low_memory=False)
    df_veh = pd.read_csv(os.path.join(year_path, f["veh"]), sep=';', low_memory=False)
    df_usagers = pd.read_csv(os.path.join(year_path, f["usagers"]), sep=';', low_memory=False)

    # Cas particulier : 2022 caracteristiques → colonne Accident_Id
    if year == 2022 and "Accident_Id" in df_carac.columns:
        df_carac = df_carac.rename(columns={"Accident_Id": "Num_Acc"})
        print("Renommage : Accident_Id → Num_Acc (2022)")

    # Fusion standard BAAC
    df = (df_carac
          .merge(df_lieux, on="Num_Acc", how="left")
          .merge(df_veh, on="Num_Acc", how="left")
          .merge(df_usagers, on="Num_Acc", how="left"))

    df["annee"] = year
    return df


def load_multiple_years(years, base_path="data"):
    df_total = pd.DataFrame()

    for year in years:
        df_year = load_year_data(year, base_path)
        df_total = pd.concat([df_total, df_year], ignore_index=True)

    return df_total


if __name__ == "__main__":
    years = [2021, 2022, 2023]

    df = load_multiple_years(years)

    print("\nAperçu du dataset fusionné :")
    print(df.head())

    os.makedirs("output", exist_ok=True)
    df.to_csv("output/merged_dataset.csv", index=False, sep=";")

    print("\nDataset final sauvegardé dans output/merged_dataset.csv")
