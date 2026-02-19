1. `README.md`
*Save this file as `README.md` in the root of your repository.*

# Sepsis Deterioration Trajectory Prediction

This repository contains the complete source code for the manuscript:
**"Machine Learning Predicts Sepsis Deterioration Trajectories"** .

## ⚠️ Data Privacy & Reproducibility Note

Due to strict patient privacy regulations (HIPAA and institutional review board requirements), the raw clinical datasets used in this study cannot be publicly shared.

To ensure the full reproducibility of our methods and results, **we have provided a synthetic dataset (`sample_data_full.xlsx`)** within this repository.
*   **Synthetic Data:** This dataset mimics the statistical properties (distributions, correlations, missing data patterns) of the original cohorts but contains no Protected Health Information (PHI).
*   **End-to-End Pipeline:** The provided `run_analysis.py` script is pre-configured to load this synthetic data automatically. It executes the entire pipeline—from data preprocessing and feature engineering to model training and figure generation.

## 📊 Expected Outputs

Running the analysis script will generate the following figures in the `results_paper_final/figures/dev/` directory. These match the main and supplementary figures presented in the manuscript:

### Main Figures
*   **`Figure_2_Trajectory_Patterns.pdf`**: 
    *   (a) PCA visualization of trajectory groups.
    *   (b) Longitudinal SOFA score trajectories (Mean ± 95% CI).
    *   (c) Temporal trends of key physiological parameters (Lactate, MAP, HR, Respiratory Rate).
*   **`Figure_3_Model_Performance.pdf`**: 
    *   (a) Receiver Operating Characteristic (ROC) curve.
    *   (b) Precision-Recall (PR) curve.
    *   (c) SHAP summary plot ranking top feature importance.

### Supplementary Figures
*   **`Supp_Figure_1_Polynomial_Fit.pdf`**: Polynomial curve fitting for trajectory modeling.
*   **`Supp_Figure_2_Individual_Trajectories.pdf`**: Spaghetti plots showing individual patient evolution.
*   **`Supp_Figure_3_Physiological_Variability.pdf`**: Boxplots comparing physiological variability (HR, MAP, RR, SpO2) across groups.
*   **`Supp_Figure_4_Calibration_DCA.pdf`**: Model calibration curves and Decision Curve Analysis (DCA).
*   **`Supp_Figure_5_Trajectory_Calibration.pdf`**: Calibration assessment for trajectory classification.
*   **`Supp_Figure_6_HRV_Analysis.pdf`**: Analysis of Heart Rate Variability (HRV) trends and associated mortality risk.
*   **`Supp_Figure_7_SHAP_Interactions.pdf`**: Top synergistic feature interactions identified by the model.
*   **`Supp_Figure_8_SHAP_Dependence.pdf`**: Feature dependence plots for top predictors.

*Note: As these figures are generated from synthetic data, specific numeric values (e.g., AUC scores) will differ slightly from the published manuscript, but the methodological workflow and visual structure are identical.*

## 🚀 Getting Started

### Prerequisites
*   Python 3.8 or higher

### Installation
1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Analysis
Execute the main script from the terminal:

```bash
python run_analysis.py
```

The script will:
1. Detect and load `sample_data_full.xlsx`.
2. Preprocess the time-series data.
3. Train the XGBoost machine learning model.
4. Generate and save all PDF figures to the output directory.

## 📁 Repository Structure

```text
.
├── run_analysis.py          # Main analysis script (End-to-End pipeline)
├── sample_data_full.xlsx    # Synthetic dataset for reproducibility
├── requirements.txt         # Python dependencies
├── README.md                # Documentation
└── results_paper_final/     # Output directory (created automatically)
    └── figures/dev/         # Generated figures
```

## 🔍 Contact

For questions regarding the code or methodology, please contact the corresponding author:
**Rui Zhang** – ccmzhangrui@foxmail.com
```

---

### 2. `requirements.txt`
*Save this file as `requirements.txt`.*

```text
numpy>=1.21.0
pandas>=1.3.5
matplotlib>=3.5.0
seaborn>=0.11.2
scikit-learn>=1.0.2
xgboost>=1.6.0
shap>=0.40.0
openpyxl>=3.0.9
tqdm>=4.64.0
```

---

### 3. `run_analysis.py`
*Save this file as `run_analysis.py`.*
