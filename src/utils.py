"""
Utility functions: RMSLE metric and feature engineering.
"""
import numpy as np
import pandas as pd


def rmsle(y_true, y_pred):
    """Root Mean Squared Logarithmic Error."""
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def engineer_features(df, keep_targets=False):
    """
    Engineer features.

    keep_targets: if True, keep 'casual' and 'registered' columns
                  (needed for the split-regression experiment).
    """
    df = df.copy()

    # 1. Parse datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['month'] = df['datetime'].dt.month
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['hour'] = df['datetime'].dt.hour

    # 2. Peak-hour flag
    is_weekend = df['dayofweek'] >= 5
    weekday_peak = (~is_weekend) & (df['hour'].isin([7, 8, 9, 17, 18, 19]))
    weekend_peak = (is_weekend) & (df['hour'].between(10, 18))
    df['peak_hour'] = (weekday_peak | weekend_peak).astype(int)

    # 3. Drop columns
    drop_cols = ['datetime', 'holiday', 'temp']
    if not keep_targets:
        for c in ['casual', 'registered', 'count']:
            if c in df.columns:
                drop_cols.append(c)
    else:
        # Keep casual/registered/count; only drop 'datetime', 'holiday', 'temp'
        pass

    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df