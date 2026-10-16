import time
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb

from .config import MODELS, RANDOM_STATE
from .features import FEATURES, CAT_FEATURES


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def prepare_xy(df, features=FEATURES, cat_features=CAT_FEATURES):
    df = df.dropna(subset=["trips"]).copy()
    for c in cat_features:
        df[c] = df[c].astype(str).astype("category")
    return df[features], df["trips"]


def train_lgbm(train_df):
    X, y = prepare_xy(train_df)
    model = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X, y, categorical_feature=CAT_FEATURES)
    return model


def evaluate(model, df):
    X, y = prepare_xy(df)
    pred = np.clip(model.predict(X), 0, None)
    return rmse(y, pred), mae(y, pred)


def save_model(model, path=MODELS / "final_model.joblib"):
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "cat_features": CAT_FEATURES}, path)