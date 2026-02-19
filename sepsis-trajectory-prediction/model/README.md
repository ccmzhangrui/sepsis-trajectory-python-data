# Sepsis Trajectory Analysis and Outcome Prediction – Code Usage Guide

This repository contains the complete code for clustering sepsis patient trajectories, predicting clinical outcomes, and generating publication‑ready figures as described in the manuscript:
"Machine Learning Predicts Sepsis Deterioration Trajectories"

⚠ Important Note on Data & Reproducibility

Due to strict patient privacy regulations (HIPAA and institutional review board requirements), the original clinical datasets used in the study cannot be publicly shared.

To ensure full reproducibility, the main analysis script (run_analysis.py) includes a built‑in synthetic data generator. If the script does not find the private data files, it automatically creates a synthetic dataset that mimics the statistical properties (distributions, correlations, missing patterns) of the original cohorts. This allows any user to execute the entire pipeline – from preprocessing and model training to figure generation – without accessing real patient data.

Important: Because the synthetic data are randomly generated, the exact numerical results (e.g., AUC values, SHAP importance magnitudes) will differ from those reported in the published paper. However, the methodological workflow, figure structures, and analytical steps are preserved exactly, enabling reviewers and readers to verify the correctness of our implementation.            
📋 Table of Contents                                                                                                               
· 1. Prerequisites
· 2. Installation
· 3. Data Preparation (Optional)
· 4. Running the Analysis
· 5. Outputs
· 6. Why Do Results Differ from the Paper?
· 7. Customization
· 8. Troubleshooting                                                                                                              
1. Prerequisites

· Python version: 3.8 – 3.12 (newer versions may cause compatibility issues with some libraries)
· Operating system: Windows / Linux / macOS (on Windows, ensure correct file path formats)

---

2. Installation

Open a terminal (Command Prompt/PowerShell on Windows, or Terminal on Mac/Linux) and install the required packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn lifelines shap scipy openpyxl
```

If any installation fails, try upgrading pip first:

```bash
python -m pip install --upgrade pip
```

All dependencies are also listed in requirements.txt (included in the repository) – you can install them with:

```bash
pip install -r requirements.txt
```

---

3. Data Preparation (Optional)

Using synthetic data (default)

The script automatically generates synthetic data when no real data file is found. No manual data preparation is required to run the code.

Using your own data (if available)

If you have access to the original or a similar dataset, you can use it by following these steps:

1. Place your Excel file (e.g., sample_data_full.xlsx) in a known location.
2. Ensure the file contains at least the following essential columns (column names must match those used in the code):
   · Trajectory group: traj_true (0, 1, 2 for the three trajectory classes)
   · Outcome indicators: event_28d (1 = event occurred, 0 = censored) and time_28d_days (time to event or censoring)
   · Clinical variables: t0_sofa (baseline SOFA), t0_lactate (baseline lactate), age, etc. (see the feature_candidates list in the code for a complete set)
3. Modify the DATA_PATHS variable in the script to point to your file. For example:
   ```python
   DATA_PATHS = {
       "dev": r"path/to/your/data.xlsx",
   }
   ```

---

4. Running the Analysis

1. Save the script as a .py file (e.g., sepsis_analysis.py).
2. Open a terminal and navigate to the folder containing the script:
   ```bash
   cd /path/to/your/folder
   ```
3. Run the script:
   ```bash
   python sepsis_analysis.py
   ```
4. The first run may take 1–3 minutes. When finished, the terminal will display the final model metrics (AUROC, sensitivity, specificity, etc.).

---

5. Outputs

All generated figures and results are saved in the folder results_paper_final/figures/dev/. The following files are produced:

File Name Description
Main_Figure_2.pdf PCA of trajectory separation, anchored SOFA trajectories, physiological parameter trends
Main_Figure_3.pdf ROC curve, calibration curve, SHAP feature importance
Clinical_Outcomes.pdf Comparison of clinical outcomes (ICU length of stay, ventilation duration, 28‑day mortality) before and after intervention implementation
KM_Survival_28d.pdf Kaplan–Meier survival curves for the three trajectory groups
BIC_Cluster_Selection.pdf BIC curve for optimal number of trajectory clusters (KMeans)
Supp_Figure_1.pdf Polynomial curve fitting, model enhancement, mortality risk ratios
Supp_Figure_2.pdf Individual patient SOFA trajectories by group
Supp_Figure_3.pdf Physiological variability boxplots and correlation heatmap
Supp_Figure_4.pdf Calibration, performance metrics, feature importance, decision curve analysis
Supp_Figure_5.pdf Trajectory‑specific calibration curves
Supp_Figure_6.pdf Heart rate (HR) trends, HR variability (HRV) evolution, HRV mortality risk, consistency across cohorts
Supp_Figure_7.pdf Top SHAP interaction feature pairs
Supp_Figure_8.pdf SHAP interaction matrix and dependence plots

All figures are saved in both PDF (vector) and PNG (raster) formats.

---

6. Why Do Results Differ from the Paper?

Primary reason: synthetic data randomness

· The script defaults to randomly generated synthetic data (500 simulated patients) when real data are not present.
· The synthetic data are designed only to mimic the format of the original dataset – they do not reflect real medical patterns.
· Therefore, all numerical results (AUC, hazard ratios, SHAP values, etc.) will differ substantially from those in the published manuscript.

Other factors

· Fixed random seed: The code uses SEED = 2024 for reproducibility, but some libraries (e.g., SHAP, KMeans) may still introduce minor variations across different environments.
· Environment differences: Different Python versions or package versions (e.g., scikit‑learn, matplotlib) can slightly alter outputs.
· Missing columns: If the provided data lack certain columns (e.g., t6_sofa, hr_sd_0_48), the script fills them with random values, affecting results.

To obtain results close to the paper, you must use the original (or a similarly structured) real dataset. The synthetic mode is intended only for testing the code workflow.

---

7. Customization

Modifying figure appearance

You can adjust colors, fonts, and sizes by editing the following parameters in the script:

· Colors: PAL_TRAJ (trajectory group colors), DATA_COLORS (dataset colors)
· Font size / resolution: plt.rcParams['font.size'], plt.rcParams['figure.dpi']
· Figure dimensions: plt.figure(figsize=(width, height)) in each plotting function

Adjusting the prediction model

The stacking model is defined in train_stacking_model(). You can modify:

· Base estimators: e.g., change n_estimators or min_samples_leaf in RandomForestClassifier
· Stacking parameters: cv (number of cross‑validation folds) or final_estimator (e.g., replace LogisticRegression with another classifier)

Using your own synthetic data generator

If you prefer to provide a static synthetic dataset instead of relying on the built‑in generator, you can place an Excel file in the data/ folder and update DATA_PATHS accordingly. An example synthetic file (sample_data_synthetic.xlsx) is included in the repository for demonstration.

---

8. Troubleshooting

Problem Possible Solution
ModuleNotFoundError Install missing packages using pip install ... (see Section 2).
Figures appear blank or incomplete The synthetic data may lack some columns; the script fills missing values with random numbers – this is normal.
Permission error when saving figures Run the script in a directory where you have write access (e.g., your home folder or a non‑system drive).
Results differ from expected Verify that you are using the correct dataset and that all required columns exist. If using synthetic data, differences are expected.

---

📬 Contact

For questions or issues, please contact the corresponding author:

Rui Zhang
Department of Critical Care Medicine, Ruijin Hospital, Shanghai Jiao Tong University School of Medicine
Email: ccmzhangrui@foxmail.com
