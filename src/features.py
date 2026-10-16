import numpy as np
import pandas as pd

from .config import TIMEZONE


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["pickup_hour"].dt.hour
    df["dow"] = df["pickup_hour"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["month"] = df["pickup_hour"].dt.month
    df["day"] = df["pickup_hour"].dt.day
    df["days_since_start"] = (
        df["pickup_hour"].dt.normalize()
        - pd.Timestamp("2025-01-01", tz=TIMEZONE)
    ).dt.days
    df["is_payday"] = df["day"].isin([1, 2, 3, 28, 29, 30, 31]).astype(int)
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rain_3h"] = (
        df.sort_values("pickup_hour")
        .groupby("zone")["rain_mm"]
        .transform(lambda s: s.rolling(3, min_periods=1).sum())
    )
    df["is_rain"] = (df["rain_mm"] > 0).astype(int)
    df["rain_class"] = pd.cut(
        df["rain_mm"],
        bins=[-0.01, 0, 2, 10, 1000],
        labels=["none", "light", "moderate", "heavy"],
    ).astype(str)
    return df


def add_hist_features(train: pd.DataFrame, other: pd.DataFrame, cutoff=None) -> pd.DataFrame:
    """Add zone-dow-hour mean from train history only."""
    if cutoff is None:
        cutoff = train["pickup_hour"].max() + pd.Timedelta(hours=1)

    hist = train[train["pickup_hour"] < cutoff]

    agg = (
        hist.groupby(["zone", "dow", "hour"])["trips"]
        .agg(["mean", "median"])
        .reset_index()
    )
    agg.columns = ["zone", "dow", "hour", "zone_dow_hour_mean", "zone_dow_hour_median"]

    other = other.merge(agg, on=["zone", "dow", "hour"], how="left")

    zone_mean = hist.groupby("zone")["trips"].mean().rename("zone_mean").reset_index()
    other = other.merge(zone_mean, on="zone", how="left")
    other["zone_dow_hour_mean"] = other["zone_dow_hour_mean"].fillna(other["zone_mean"])

    return other


FEATURES = [
    "zone",
    "hour",
    "dow",
    "is_weekend",
    "month",
    "day",
    "days_since_start",
    "is_payday",
    "temp_c",
    "rain_mm",
    "rain_3h",
    "rain_class",
    "humidity_pct",
    "wind_kmh",
    "is_rain",
    "event_count",
    "in_event",
    "hours_to_event_start",
    "hours_since_event_end",
    "event_attendance",
    "event_type_primary",
    "zone_dow_hour_mean",
]

CAT_FEATURES = ["zone", "rain_class", "event_type_primary"]