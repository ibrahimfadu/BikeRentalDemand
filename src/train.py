import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from utils import rmsle, engineer_features


def load_data():
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    return train, test


def cross_validate(model, X, y, n_splits=10):
    """10-fold CV, returns mean RMSLE."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        scores.append(rmsle(y_val, preds))
    return np.mean(scores)


def main():
    print("Loading data...")
    train, test = load_data()

    print("Engineering features...")
    train_fe = engineer_features(train)
    test_fe = engineer_features(test)

    # Separate features (X) from target (y)
    feature_cols = [c for c in train_fe.columns if c != 'count']
    X = train_fe[feature_cols]
    y = train_fe['count']
    X_test = test_fe[feature_cols]

    print(f"Training shape: {X.shape}, Test shape: {X_test.shape}")
    print(f"Features: {list(X.columns)}\n")

    # ---- Define models ----
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
        ),
    }

    # ---- Cross-validate each model ----
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        score = cross_validate(model, X, y)
        results[name] = score
        print(f"  -> CV RMSLE: {score:.4f}\n")

    # ---- Print results table ----
    print("=" * 45)
    print(f"{'Model':<22}{'CV RMSLE':>12}")
    print("=" * 45)
    for name, score in sorted(results.items(), key=lambda x: x[1]):
        print(f"{name:<22}{score:>12.4f}")
    print("=" * 45)

    # ---- Train best model on full data, predict on test ----
    best_name = min(results, key=results.get)
    print(f"\nBest model: {best_name}")
    best_model = models[best_name]
    best_model.fit(X, y)

    test_preds = best_model.predict(X_test)
    test_preds = np.maximum(test_preds, 0)  # no negative counts

    submission = pd.DataFrame({
        'datetime': test['datetime'],
        'count': test_preds
    })
    submission.to_csv('../submission.csv', index=False)
    print("Saved predictions to submission.csv")

    # ---- Feature importance (if tree-based) ----
    if hasattr(best_model, 'feature_importances_'):
        print("\nFeature importances:")
        importances = pd.Series(
            best_model.feature_importances_, index=feature_cols
        ).sort_values(ascending=False)
        print(importances)


if __name__ == '__main__':
    main()