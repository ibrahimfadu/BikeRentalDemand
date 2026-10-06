"""
Runs all 5 model experiments from the paper's Discussion section.

  1. Linear Regression (baseline)
  2. Principal Component Regression (PCR)
  3. Random Forest
  4. Conditional Inference Tree (CTree approximation via DecisionTree)
  5. Split regression on registered + casual
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from utils import rmsle, engineer_features


# ---------------------------------------------------------------
# Helper: run 10-fold CV for any sklearn-compatible model
# ---------------------------------------------------------------
def cross_validate(model, X, y, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for tr, val in kf.split(X):
        model.fit(X.iloc[tr], y.iloc[tr])
        preds = model.predict(X.iloc[val])
        scores.append(rmsle(y.iloc[val], preds))
    return np.mean(scores), np.std(scores)


# ---------------------------------------------------------------
# Load & prep data
# ---------------------------------------------------------------
print("Loading data...")
train = pd.read_csv('../data/train.csv')

# Version with targets kept (needed for split regression)
train_full = engineer_features(train, keep_targets=True)

# Version with targets dropped (needed for regular models)
train_fe = train_full.drop(columns=['casual', 'registered', 'count'])
feature_cols = list(train_fe.columns)
X = train_fe
y = train_full['count']

print(f"Features ({len(feature_cols)}): {feature_cols}\n")

results = {}


# ===============================================================
# MODEL 1: Linear Regression
# ===============================================================
print("=" * 55)
print("MODEL 1: Linear Regression")
print("=" * 55)
lr = LinearRegression()
score, std = cross_validate(lr, X, y)
results['Linear Regression'] = score
print(f"CV RMSLE: {score:.4f} (+/- {std:.4f})\n")


# ===============================================================
# MODEL 2: Principal Component Regression (PCR)
# ===============================================================
print("=" * 55)
print("MODEL 2: Principal Component Regression (PCR)")
print("=" * 55)
# Standardize → PCA → Linear Regression
# Paper found using ALL components gave lowest error
pcr = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),   # keep 95% variance
    ('lr', LinearRegression())
])
score, std = cross_validate(pcr, X, y)
results['PCR'] = score
print(f"CV RMSLE: {score:.4f} (+/- {std:.4f})")

# Report how many components were kept
pcr.fit(X, y)
n_comp = pcr.named_steps['pca'].n_components_
print(f"Components kept (95% variance): {n_comp}\n")


# ===============================================================
# MODEL 3: Random Forest
# ===============================================================
print("=" * 55)
print("MODEL 3: Random Forest")
print("=" * 55)
rf = RandomForestRegressor(
    n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
)
score, std = cross_validate(rf, X, y)
results['Random Forest'] = score
print(f"CV RMSLE: {score:.4f} (+/- {std:.4f})\n")


# ===============================================================
# MODEL 4: Conditional Inference Tree (CTree substitute)
# ===============================================================
# NOTE: True CTree is in R (party::ctree). In Python, the closest
# substitute is a DecisionTreeRegressor. We note this limitation
# in the write-up. Alternatively, if you want real CTree, install R
# + rpy2 (see bottom of this file).
# ===============================================================
print("=" * 55)
print("MODEL 4: Conditional Inference Tree (DecisionTree substitute)")
print("=" * 55)
ctree = DecisionTreeRegressor(
    max_depth=12, min_samples_leaf=5, random_state=42
)
score, std = cross_validate(ctree, X, y)
results['CTree (DecisionTree sub.)'] = score
print(f"CV RMSLE: {score:.4f} (+/- {std:.4f})\n")


# ===============================================================
# MODEL 5: Split regression on registered + casual
# ===============================================================
print("=" * 55)
print("MODEL 5: Split regression (registered + casual, summed)")
print("=" * 55)

y_reg = train_full['registered']
y_cas = train_full['casual']

kf = KFold(n_splits=10, shuffle=True, random_state=42)
split_scores = []

for tr, val in kf.split(X):
    # Train separate models on each component
    rf_reg = RandomForestRegressor(n_estimators=200, max_depth=15,
                                   random_state=42, n_jobs=-1)
    rf_cas = RandomForestRegressor(n_estimators=200, max_depth=15,
                                   random_state=42, n_jobs=-1)
    rf_reg.fit(X.iloc[tr], y_reg.iloc[tr])
    rf_cas.fit(X.iloc[tr], y_cas.iloc[tr])

    # Predict and sum
    pred = rf_reg.predict(X.iloc[val]) + rf_cas.predict(X.iloc[val])
    split_scores.append(rmsle(y.iloc[val], pred))

results['Split (reg+cas)'] = np.mean(split_scores)
print(f"CV RMSLE: {np.mean(split_scores):.4f} "
      f"(+/- {np.std(split_scores):.4f})\n")


# ===============================================================
# FINAL RESULTS TABLE
# ===============================================================
print("=" * 55)
print(f"{'Model':<32}{'CV RMSLE':>15}")
print("=" * 55)
for name, s in sorted(results.items(), key=lambda x: x[1]):
    print(f"{name:<32}{s:>15.4f}")
print("=" * 55)

# Save results to CSV for the write-up
pd.DataFrame(
    list(results.items()), columns=['Model', 'CV_RMSLE']
).sort_values('CV_RMSLE').to_csv('../results.csv', index=False)
print("\nSaved results to ../results.csv")