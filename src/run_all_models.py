"""
Runs all 8 models from the paper + the split-regression experiment.

  1. Linear Regression
  2. GLMNet (Elastic Net, Poisson)
  3. PCR (Principal Component Regression)
  4. SVR (Support Vector Regression)
  5. Random Forest
  6. GBM (Gradient Boosting)
  7. CTree (DecisionTree substitute)
  8. Stacking Ensemble (CTree + RF -> Linear meta)
  +  Split regression (registered + casual)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from utils import rmsle, engineer_features


# ---------------------------------------------------------------
# CV helper
# ---------------------------------------------------------------
def cross_validate(model, X, y, n_splits=10, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr, val in kf.split(X):
        model.fit(X.iloc[tr], y.iloc[tr])
        preds = model.predict(X.iloc[val])
        scores.append(rmsle(y.iloc[val], preds))
    return float(np.mean(scores)), float(np.std(scores))


# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------
print("Loading data...")
train = pd.read_csv('../data/train.csv')
train_full = engineer_features(train, keep_targets=True)
train_fe   = train_full.drop(columns=['casual', 'registered', 'count'])
X = train_fe
y = train_full['count']
feature_cols = list(X.columns)
print(f"Features ({len(feature_cols)}): {feature_cols}\n")

results = {}

# ===============================================================
# 1. Linear Regression
# ===============================================================
print("=" * 60)
print("1. Linear Regression")
print("=" * 60)
score, std = cross_validate(LinearRegression(), X, y)
results['Linear'] = score
print(f"CV RMSLE: {score:.4f} (+/- {std:.4f})\n")


# ===============================================================
# 2. GLMNet (Elastic Net)
# ===============================================================
print("=" * 60)
print("2. GLMNet (Elastic Net)")
print("=" * 60)
# l1_ratio=0.5 -> 50/50 mix of L1/L2
glmnet = Pipeline([
    ('scaler', StandardScaler()),
    ('enet', ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000))
])
score, std = cross_validate(glmnet, X, y)
results['GLMNet'] = score
print(f"CV RMSLE: {score:.4f} (+/- {std:.4f})\n")


# ===============================================================
# 3. PCR (Principal Component Regression)
# ===============================================================
print("=" * 60)
print("3. PCR (Principal Component Regression)")
print("=" * 60)
pcr = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('lr', LinearRegression())
])
score, std = cross_validate(pcr, X, y)
results['PCR'] = score
pcr.fit(X, y)
print(f"CV RMSLE: {score:.4f} (+/- {std:.4f}) "
      f"| components kept: {pcr.named_steps['pca'].n_components_}\n")


# ===============================================================
# 4. SVR (Support Vector Regression)
# ===============================================================
print("=" * 60)
print("4. SVR (RBF kernel)")
print("=" * 60)
svr = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf', C=10, epsilon=0.5, gamma='scale'))
])
score, std = cross_validate(svr, X, y)
results['SVR'] = score
print(f"CV RMSLE: {score:.4f} (+/- {std:.4f})\n")


# ===============================================================
# 5. Random Forest
# ===============================================================
print("=" * 60)
print("5. Random Forest")
print("=" * 60)
rf = RandomForestRegressor(n_estimators=200, max_depth=15,
                           random_state=42, n_jobs=-1)
score, std = cross_validate(rf, X, y)
results['Random Forest'] = score
print(f"CV RMSLE: {score:.4f} (+/- {std:.4f})\n")


# ===============================================================
# 6. GBM (Gradient Boosting)
# ===============================================================
print("=" * 60)
print("6. GBM (Gradient Boosting)")
print("=" * 60)
gbm = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                max_depth=5, random_state=42)
score, std = cross_validate(gbm, X, y)
results['GBM'] = score
print(f"CV RMSLE: {score:.4f} (+/- {std:.4f})\n")


# ===============================================================
# 7. CTree (DecisionTree substitute)
# ===============================================================
print("=" * 60)
print("7. CTree (DecisionTree substitute)")
print("=" * 60)
ctree = DecisionTreeRegressor(max_depth=12, min_samples_leaf=5,
                              random_state=42)
score, std = cross_validate(ctree, X, y)
results['CTree (DT sub.)'] = score
print(f"CV RMSLE: {score:.4f} (+/- {std:.4f})\n")


# ===============================================================
# 8. Stacking Ensemble (RF + CTree -> Linear meta)
# ===============================================================
print("=" * 60)
print("8. Stacking Ensemble (RF + CTree -> Linear meta)")
print("=" * 60)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
stack_scores = []

for tr, val in kf.split(X):
    # Split training portion further to train meta-model
    inner_split = int(0.8 * len(tr))
    tr_main, tr_meta = tr[:inner_split], tr[inner_split:]

    X_tr_main, y_tr_main = X.iloc[tr_main], y.iloc[tr_main]
    X_tr_meta, y_tr_meta = X.iloc[tr_meta], y.iloc[tr_meta]
    X_val, y_val         = X.iloc[val], y.iloc[val]

    # Train base learners on main
    m_rf = RandomForestRegressor(n_estimators=200, max_depth=15,
                                 random_state=42, n_jobs=-1)
    m_ct = DecisionTreeRegressor(max_depth=12, min_samples_leaf=5,
                                 random_state=42)
    m_rf.fit(X_tr_main, y_tr_main)
    m_ct.fit(X_tr_main, y_tr_main)

    # Meta-features on the meta portion
    meta_X = np.column_stack([m_rf.predict(X_tr_meta),
                              m_ct.predict(X_tr_meta)])
    meta_model = LinearRegression()
    meta_model.fit(meta_X, y_tr_meta)

    # Predict on validation fold
    val_X = np.column_stack([m_rf.predict(X_val), m_ct.predict(X_val)])
    preds = meta_model.predict(val_X)
    stack_scores.append(rmsle(y_val, preds))

results['Stacking'] = float(np.mean(stack_scores))
print(f"CV RMSLE: {np.mean(stack_scores):.4f} "
      f"(+/- {np.std(stack_scores):.4f})\n")


# ===============================================================
# BONUS: Split regression (registered + casual)
# ===============================================================
print("=" * 60)
print("BONUS: Split regression (registered + casual)")
print("=" * 60)
y_reg = train_full['registered']
y_cas = train_full['casual']

split_scores = []
for tr, val in kf.split(X):
    rf_reg = RandomForestRegressor(n_estimators=200, max_depth=15,
                                   random_state=42, n_jobs=-1)
    rf_cas = RandomForestRegressor(n_estimators=200, max_depth=15,
                                   random_state=42, n_jobs=-1)
    rf_reg.fit(X.iloc[tr], y_reg.iloc[tr])
    rf_cas.fit(X.iloc[tr], y_cas.iloc[tr])
    pred = rf_reg.predict(X.iloc[val]) + rf_cas.predict(X.iloc[val])
    split_scores.append(rmsle(y.iloc[val], pred))

results['Split (reg+cas)'] = float(np.mean(split_scores))
print(f"CV RMSLE: {np.mean(split_scores):.4f} "
      f"(+/- {np.std(split_scores):.4f})\n")


# ===============================================================
# FINAL TABLE
# ===============================================================
print("=" * 60)
print(f"{'Model':<28}{'CV RMSLE':>16}")
print("=" * 60)
for name, s in sorted(results.items(), key=lambda x: x[1]):
    print(f"{name:<28}{s:>16.4f}")
print("=" * 60)

pd.DataFrame(list(results.items()),
             columns=['Model', 'CV_RMSLE']
            ).sort_values('CV_RMSLE').to_csv('../results.csv', index=False)
print("\nSaved results to ../results.csv")
