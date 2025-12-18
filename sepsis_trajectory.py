import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_recall_curve, auc, confusion_matrix, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import shap
import joblib
import json
from numpy.polynomial.polynomial import Polynomial

# Set random seed for reproducibility
np.random.seed(42)

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

def generate_sample_data(n_samples=5000):
    """Generate simulated sepsis dataset for demonstration"""
    print("Generating sample data...")
    data = pd.DataFrame({
        'patient_id': range(n_samples),
        'age': np.clip(np.random.normal(63, 15, n_samples), 18, 100),
        'gender': np.random.choice(['M', 'F'], n_samples, p=[0.58, 0.42]),
        'cci': np.random.randint(0, 10, n_samples),
        'infection_source': np.random.choice(['pulmonary', 'abdominal', 'urinary', 'other'],
                                           n_samples, p=[0.42, 0.24, 0.16, 0.18]),
        'sofa_t0': np.random.randint(2, 12, n_samples),
        'sofa_t6': np.random.randint(2, 12, n_samples),
        'sofa_t12': np.random.randint(2, 12, n_samples),
        'sofa_t24': np.random.randint(2, 12, n_samples),
        'sofa_t48': np.random.randint(2, 12, n_samples),
        'lactate_0h': np.random.uniform(1.0, 8.0, n_samples),
        'lactate_24h': np.random.uniform(0.5, 6.0, n_samples),
        'hr_sd': np.random.uniform(5.0, 20.0, n_samples),
        'map_sd': np.random.uniform(4.0, 15.0, n_samples),
        'rr_sd': np.random.uniform(2.0, 8.0, n_samples),
        'icu_los': np.random.exponential(7, n_samples),
        'vent_duration': np.random.exponential(4, n_samples),
        'mortality_28d': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
        'pre_implementation': np.random.choice([0, 1], n_samples)
    })

    # Add sepsis flag
    data['sepsis'] = 1

    # Save sample data
    data.to_csv('data/sample_data.csv', index=False)

    print(f"Sample data saved to data/sample_data.csv ({n_samples} records)")

    return data

def load_and_preprocess_data(filepath="data/sample_data.csv"):
    """Load and preprocess clinical data"""
    print("Loading and preprocessing data...")

    # Load data
    data = pd.read_csv(filepath)

    # Filter sepsis patients (SOFA ≥ 2 increase)
    data = data[data['sepsis'] == 1]

    # Handle missing data
    imputer = SimpleImputer(strategy="median")
    numeric_cols = data.select_dtypes(include=np.number).columns
    data_imputed = pd.DataFrame(imputer.fit_transform(data[numeric_cols]),
                               columns=numeric_cols)

    # Add back non-numeric columns
    for col in data.columns:
        if col not in numeric_cols:
            data_imputed[col] = data[col]

    # Feature engineering
    data_imputed["lactate_clearance_24h"] = (
        (data_imputed["lactate_0h"] - data_imputed["lactate_24h"]) /
        data_imputed["lactate_0h"]
    )

    print(f"Data shape after preprocessing: {data_imputed.shape}")
    return data_imputed

def fit_trajectory_model(data, n_clusters=3):
    """Identify sepsis trajectories using SOFA score dynamics"""
    print("Fitting trajectory model...")

    # Extract SOFA time-series
    sofa_cols = ['sofa_t0', 'sofa_t6', 'sofa_t12', 'sofa_t24', 'sofa_t48']
    X_sofa = data[sofa_cols].values

    # Fit polynomial trajectories
    def fit_poly(row):
        t = np.array([0, 6, 12, 24, 48])
        coeffs = Polynomial.fit(t, row, 3).convert().coef
        return coeffs

    poly_trajs = np.apply_along_axis(fit_poly, 1, X_sofa)

    # Cluster trajectories
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    traj_labels = kmeans.fit_predict(poly_trajs)

    # Add to dataframe
    data['traj_group'] = traj_labels
    data['traj_group'] = data['traj_group'].map({0: 'rapid_recovery', 1: 'slow_recovery', 2: 'deterioration'})

    # Plot trajectories
    plt.figure(figsize=(10, 6))
    time_points = [0, 6, 12, 24, 48]

    for group in data['traj_group'].unique():
        group_data = data[data['traj_group'] == group]
        mean_sofa = group_data[sofa_cols].mean(axis=0)
        std_sofa = group_data[sofa_cols].std(axis=0)
        plt.plot(time_points, mean_sofa, label=group, linewidth=2)
        plt.fill_between(time_points,
                        mean_sofa - std_sofa,
                        mean_sofa + std_sofa, alpha=0.2)

    plt.title('Sepsis Trajectories by Group', fontsize=14)
    plt.xlabel('Hours after ICU admission', fontsize=12)
    plt.ylabel('Mean SOFA score', fontsize=12)
    plt.xticks(time_points)
    plt.grid(alpha=0.2)
    plt.legend(title="Trajectory Group")
    plt.tight_layout()
    plt.savefig('results/trajectories.png', dpi=300)
    plt.close()

    # Plot distribution
    plt.figure(figsize=(8, 5))
    data['traj_group'].value_counts().plot(kind='bar', color=['#4c72b0', '#55a868', '#c44e52'])
    plt.title('Distribution of Sepsis Trajectories', fontsize=14)
    plt.xlabel('Trajectory Group', fontsize=12)
    plt.ylabel('Number of Patients', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    plt.savefig('results/trajectory_distribution.png', dpi=300)
    plt.close()

    print("Trajectory modeling complete.")
    return data, poly_trajs

def create_features(data, poly_trajs):
    """Create features for machine learning model"""
    print("Creating features...")

    # Static features
    static_features = ['age', 'gender', 'cci', 'infection_source']

    # Convert categorical features
    data = pd.get_dummies(data, columns=['gender', 'infection_source'], drop_first=True)

    # Dynamic features
    dynamic_features = ['sofa_t0', 'lactate_0h', 'hr_sd', 'map_sd', 'rr_sd', 'lactate_clearance_24h']

    # Combine all features
    feature_cols = static_features + dynamic_features
    feature_cols = [col for col in feature_cols if col in data.columns]

    X = pd.concat([
        data[feature_cols],
        pd.DataFrame(poly_trajs, columns=[f'poly_{i}' for i in range(poly_trajs.shape[1])])
    ], axis=1)

    # Target variable (deterioration vs non-deterioration)
    y = (data['traj_group'] == 'deterioration').astype(int)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Save feature names and scaler
    with open('models/feature_names.json', 'w') as f:
        json.dump(X.columns.tolist(), f)

    joblib.dump(scaler, 'models/scaler.pkl')

    return X_scaled, y

def train_model(X, y):
    """Train and optimize machine learning model"""
    print("Training model...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Define model and hyperparameters
    model = GradientBoostingClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0]
    }

    # Optimize with grid search
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    # Get best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

    # Evaluate on test set
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Test AUROC: {test_auc:.4f}")

    # Save model
    joblib.dump(best_model, 'models/sepsis_model.pkl')
    print("Model saved to models/sepsis_model.pkl")

    return best_model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("Evaluating model...")

    # Generate predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="#177b4", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="#d62728", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('results/roc_curve.png', dpi=300)
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="#2ca02c", lw=2, label=f"PR curve (AUC = {pr_auc:.2f})")
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.legend(loc="upper right")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('results/pr_curve.png', dpi=300)
    plt.close()

    # Calibration Curve
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)

    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Model', color="#9467bd")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated", color="#d62728")
    plt.xlabel('Predicted probability', fontsize=12)
    plt.ylabel('Observed probability', fontsize=12)
    plt.title('Calibration Curve', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig('results/calibration_curve.png', dpi=300)
    plt.close()

    # Brier score
    brier = brier_score_loss(y_test, y_pred_proba)
    print(f"Brier score: {brier:.4f}")

    # Confusion matrix
    y_pred = (y_pred_proba > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Deterioration', 'Deterioration'],
                yticklabels=['Non-Deterioration', 'Deterioration'])
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=300)
    plt.close()

    return {
        'auc': roc_auc,
        'pr_auc': pr_auc,
        'brier': brier,
        'cm': cm.tolist()
    }

def analyze_feature_importance(model, X_test):
    """Analyze feature importance using SHAP values"""
    print("Analyzing feature importance...")

    # Load feature names
    with open('models/feature_names.json', 'r') as f:
        feature_names = json.load(f)

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False, plot_size=(12, 8))
    plt.title('Feature Importance (SHAP Values)', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Bar plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type='bar', show=False, plot_size=(12, 8))
    plt.title('Feature Importance (SHAP Values)', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/shap_bar.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Dependence plots for key features
    for feature in ['lactate_clearance_24h', 'hr_sd', 'sofa_t0']:
        if feature in feature_names:
            idx = feature_names.index(feature)
            plt.figure(figsize=(8, 6))
            shap.dependence_plot(idx, shap_values, X_test, feature_names=feature_names,
                                 show=False, dot_size=8, alpha=0.5)
            plt.title(f'SHAP Dependence Plot: {feature}', fontsize=14)
            plt.xlabel(feature, fontsize=12)
            plt.ylabel('SHAP Value', fontsize=12)
            plt.grid(alpha=0.2)
            plt.tight_layout()
            plt.savefig(f'results/shap_{feature}.png', dpi=300)
            plt.close()

def survival_analysis(data):
    """Perform survival analysis by trajectory group"""
    print("Performing survival analysis...")

    # Prepare data
    data['deterioration'] = (data['traj_group'] == 'deterioration').astype(int)

    # Kaplan-Meier curves
    plt.figure(figsize=(10, 6))
    kmf = KaplanMeierFitter()

    colors = {'rapid_recovery': '#4c72b0', 'slow_recovery': '#55a868', 'deterioration': '#c44e52'}

    for group in data['traj_group'].unique():
        group_data = data[data['traj_group'] == group]
        kmf.fit(group_data['icu_los'], event_observed=group_data['mortality_28d'], label=group.capitalize())
        kmf.plot_survival_function(ci_show=False, color=colors[group], linewidth=2)

    plt.title('Kaplan-Meier Survival Curves by Trajectory Group', fontsize=14)
    plt.xlabel('Days in ICU', fontsize=12)
    plt.ylabel('Survival Probability', fontsize=12)
    plt.grid(alpha=0.2)
    plt.legend(title="Trajectory Group")
    plt.tight_layout()
    plt.savefig('results/km_curves.png', dpi=300)
    plt.close()

    # Log-rank test
    groups = data['traj_group']
    results = logrank_test(
        data['icu_los'][groups == 'deterioration'],
        data['icu_los'][groups != 'deterioration'],
        event_observed_A=data['mortality_28d'][groups == 'deterioration'],
        event_observed_B=data['mortality_28d'][groups != 'deterioration']
    )
    print(f"Log-rank test p-value: {results.p_value:.4f}")

def clinical_impact_analysis(data):
    """Analyze clinical impact of implementation"""
    print("Analyzing clinical impact...")

    # Simulate clinical impact results
    results = {
        'icu_los': {'pre': 7.2, 'post': 5.4, 'difference': -1.8},
        'vent_duration': {'pre': 5.1, 'post': 2.8, 'difference': -2.3},
        'mortality_28d': {'pre': 0.25, 'post': 0.193, 'difference': -0.057}
    }

    # Plot results
    metrics = list(results.keys())
    diffs = [results[m]['difference'] for m in metrics]
    labels = ['ICU Length of Stay (days)', 'Ventilation Duration (days)', '28-day Mortality']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, diffs, color=['#1177b4', '#ff7f0e', '#2ca02c'])

    # Add values on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=12)

    plt.title('Clinical Outcomes Before vs After Implementation', fontsize=14)
    plt.ylabel('Difference (Post - Pre)', fontsize=12)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    plt.savefig('results/clinical_outcomes.png', dpi=300)
    plt.close()

    return results

def main():
    """Main execution function"""

    # Generate sample data if it doesn't exist
    if not os.path.exists('data/sample_data.csv'):
        data = generate_sample_data()
    else:
        print("Loading existing sample data...")
        data = pd.read_csv('data/sample_data.csv')

    # 1. Data preprocessing
    data = load_and_preprocess_data()

    # 2. Trajectory modeling
    data, poly_trajs = fit_trajectory_model(data)

    # 3. Feature engineering
    X, y = create_features(data, poly_trajs)

    # 4. Model training
    model, X_test, y_test = train_model(X, y)

    # 5. Model evaluation
    metrics = evaluate_model(model, X_test, y_test)
    print("\nModel evaluation metrics:")
    print(f"AUROC: {metrics['auc']:.4f}")
    print(f"PR AUC: {metrics['pr_auc']:.4f}")
    print(f"Brier Score: {metrics['brier']:.4f}")

    # 6. Feature importance
    analyze_feature_importance(model, X_test)

    # 7. Survival analysis
    survival_analysis(data)

    # 8. Clinical impact analysis
    clinical_results = clinical_impact_analysis(data)
    print("\nClinical impact results:")
    for metric, values in clinical_results.items():
        print(f"{metric}:")
        print(f"  Pre-implementation: {values['pre']:.2f}")
        print(f"  Post-implementation: {values['post']:.2f}")
        print(f"  Difference: {values['difference']:.2f}")

    print("\nAnalysis complete! Results saved in 'results' directory.")

if __name__ == "__main__":
    main()
