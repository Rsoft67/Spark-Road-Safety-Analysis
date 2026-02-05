# French Real Estate Market Segmentation

## Overview
This project analyzes the French real estate market using official **DVF (Demandes de Valeurs Foncières)** data. The goal is to identify homogeneous property groups and detect pricing anomalies across different geographical clusters using unsupervised learning.

## Key Features
- **Data Engineering:** Comprehensive cleaning of raw DVF data, handling of multi-lot transactions, and removal of non-residential outliers.
- **Market Segmentation:** Implementation of **K-Means Clustering** to stratify the market into 5 distinct categories (from rural areas to premium Parisian districts).
- **Anomaly Detection:** Used **Isolation Forest** to identify extreme price deviations and potential real estate opportunities/errors.
- **Exploratory Data Analysis (EDA):** Visualization of price distributions, surface areas, and geographical density.

## Tech Stack
- **Language:** Python
- **Libraries:** Pandas, Scikit-Learn, Matplotlib, Seaborn
- **Algorithms:** K-Means, Isolation Forest, PCA (Dimensionality Reduction)

## Results
- Identified 5 clear market clusters based on price per m² and location.
- Detected significant outliers in high-demand areas (Paris) with price discounts exceeding 95% compared to cluster averages.

## Project Structure
├── data/               # DVF datasets (raw and processed)
├── src/                # Python scripts for cleaning and modeling
├── notebooks/          # Analysis and visualization notebooks
├── reports/            # Project documentation and final report
└── README.md