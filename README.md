# Road Accident Severity Prediction in France

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](TON_LIEN_COLAB_ICI)

## 📂 Documentation
For a detailed analysis of the methodology, feature engineering, and final results, please refer to the full technical report:
👉 **[View Project Report (PDF)](./Rapport_Projet_Accident.pdf)**

## 📌 Project Overview
This project aims to predict the severity of road accidents in France using historical data from the **BAAC (Bulletins d'Analyse des Accidents Corporels)**. By merging and analyzing multiple datasets (characteristics, locations, vehicles, and users), we built a machine learning pipeline capable of distinguishing between "Non-Serious" and "Serious" accidents.

## 🛠 Methodology & Tech Stack
- **Data Engineering:** Merging 4 relational datasets and handling massive class imbalance.
- **Problem Reformulation:** Transitioned from a 4-class problem to a **Binary Classification** (Non-Serious vs. Serious) to achieve higher model robustness.
- **Algorithms:** Implementation of **LightGBM** (Light Gradient Boosting Machine) for its efficiency with tabular data.
- **Optimization:** Automated hyperparameter tuning and feature selection.
- **Interpretability:** Use of **SHAP Values** and Global Feature Importance to explain the biological and environmental factors influencing accident outcomes.

## 📊 Key Results
- **Final Model:** LightGBM
- **Accuracy:** 0.738
- **F1-Score:** 0.737
- **Key Risk Factors:** Absence of safety equipment (seatbelts/helmets), user category (pedestrians/motorcyclists), gender, and impact type.

## 📂 Project Structure
* **src/**: Python scripts for data cleaning, merging, and feature engineering.
* **notebooks/**: Jupyter notebooks containing the exploratory data analysis (EDA) and model training.
* **Rapport_Projet_1.pdf**: Detailed technical report including methodology and results.
* **README.md**: Project documentation.

## 🚀 Getting Started
1. **View the Full Report:** For detailed insights, check the [Full PDF Report](./Rapport_Projet_1_NathanWOHL_RobinKHATIB_AlexandruNITESCUpdf.pdf).
2. **Execution:** Open the notebook via the Colab badge or run locally:
   ```bash
   pip install lightgbm shap pandas scikit-learn matplotlib
