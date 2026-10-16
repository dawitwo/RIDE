import re
import numpy as np
import pandas as pd

from .config import (
    RAW_FILES, DATA_PROCESSED, TIMEZONE, CANONICAL_ZONES,
    ZONE_MAP, EVENT_TYPE_MAP
)


def parse_dt(s: pd.Series) -> pd.Series:
    """Parse multiple date formats. Day-first fallback."""
    s = s.astype(str).str.strip()
    s = s.replace({"nan": np.nan, "": np.nan, "None": np.nan})

    out = pd.Series(pd.NaT, index=s.index)
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%m/%d/%Y %H:%M",
        "%d-%m-%Y %H:%M",
        "%Y/%m/%d %H:%M",
        "%d.%m.%Y %H:%M",
    ]

    for fmt in formats:
        mask = out.isna() & s.notna()
        out.loc[mask] = pd.to_datetime(s[mask], format=fmt, errors="coerce")

    mask = out.isna() & s.notna()
    out.loc[mask] = pd.to_datetime(s[mask], errors="coerce", dayfirst=True)

    return out


def ensure_tz(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    if getattr(s.dt, "tz", None) is None:
        return s.dt.tz_localize(TIMEZONE, ambiguous="NaT", nonexistent="NaT")
    return s.dt.tz_convert(TIMEZONE)


def normalize_zone_text(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower().str.strip()
    s = s.str.replace(r"[^a-z0-9]+", " ", regex=True)
    s = s.str.replace(
        r"\b(subcity|sub city|zone|area|addis ababa|citywide|city wide|all)\b",
        " ",
        regex=True,
    )
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    s = s.replace(ZONE_MAP)
    return s


def clean_zone(s: pd.Series) -> pd.Series:
    s = normalize_zone_text(s)
    if CANONICAL_ZONES:
        def match(x):
            for c in CANONICAL_ZONES:
                cl = c.lower()
                if cl in x or x in cl:
                    return c
            return x
        s = s.apply(match)
    return s.str.title()


def clean_event_type(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower().str.strip()
    s = s.replace(EVENT_TYPE_MAP)
    s = s.str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_")
    return s


def clean_trips(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pickup_hour"] = ensure_tz(parse_dt(df["pickup_hour"]))
    df["zone"] = clean_zone(df["zone"])

    for c in ["trips", "avg_fare_birr", "avg_wait_min", "active_drivers"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "trips" in df.columns:
        df.loc[df["trips"] < 0, "trips"] = np.nan
    if "avg_fare_birr" in df.columns:
        df.loc[df["avg_fare_birr"] <= 0, "avg_fare_birr"] = np.nan
    if "avg_wait_min" in df.columns:
        df.loc[df["avg_wait_min"] < 0, "avg_wait_min"] = np.nan
    if "active_drivers" in df.columns:
        df.loc[df["active_drivers"] < 0, "active_drivers"] = np.nan

    df = df.sort_values("pickup_hour")
    df = df.drop_duplicates(["zone", "pickup_hour"], keep="first")
    return df


def clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = ensure_tz(parse_dt(df["timestamp"]))

    for c in ["temp_c", "rain_mm", "humidity_pct", "wind_kmh"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c].isin([-999, -99, 999, 9999]), c] = np.nan

    # Fahrenheit fix if temp > 60
    mask = df["temp_c"] > 60
    df.loc[mask, "temp_c"] = (df.loc[mask, "temp_c"] - 32) * 5 / 9

    df.loc[(df["humidity_pct"] < 0) | (df["humidity_pct"] > 100), "humidity_pct"] = np.nan
    df.loc[df["rain_mm"] < 0, "rain_mm"] = np.nan
    df.loc[df["wind_kmh"] < 0, "wind_kmh"] = np.nan

    # Duplicate hours: median
    df = (
        df.groupby("timestamp", as_index=False)
        .agg({
            "temp_c": "median",
            "rain_mm": "median",
            "humidity_pct": "median",
            "wind_kmh": "median",
            "data_type": lambda x: x.mode().iloc[0] if len(x.mode()) else np.nan,
        })
    )

    # Full hourly index
    full = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h", tz=TIMEZONE)
    df = df.set_index("timestamp").reindex(full)

    for c in ["temp_c", "rain_mm", "humidity_pct", "wind_kmh"]:
        df[c] = df[c].interpolate(limit=3).ffill().bfill()
    df["data_type"] = df["data_type"].ffill().bfill()

    df = df.reset_index().rename(columns={"index": "timestamp"})
    return df


def parse_event_zones(z) -> list:
    if pd.isna(z):
        return []
    z = str(z).lower()
    if any(x in z for x in ["all", "citywide", "city wide", "addis ababa"]):
        return CANONICAL_ZONES
    parts = re.split(r"[,;/&]| and ", z)
    return [normalize_zone_text(pd.Series([p])).iloc[0] for p in parts if p.strip()]


def clean_events(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["event_type"] = clean_event_type(df["event_type"])
    df["start"] = ensure_tz(parse_dt(df["start_datetime"]))
    df["end"] = ensure_tz(parse_dt(df["end_datetime"]))

    # Fix missing / bad ends
    df["duration_h"] = (df["end"] - df["start"]).dt.total_seconds() / 3600
    df.loc[df["end"].isna(), "end"] = df["start"] + pd.Timedelta(hours=2)
    df.loc[df["end"] < df["start"], "end"] = df["start"] + pd.Timedelta(hours=2)

    # Attendance
    att = df["expected_attendance"].astype(str).str.extract(r"(\d[\d,]*)")[0]
    df["attendance"] = pd.to_numeric(att.str.replace(",", ""), errors="coerce")

    df["status"] = df["status"].astype(str).str.lower().str.strip().fillna("confirmed")
    df["zones"] = df["zone"].apply(parse_event_zones)

    return df


def add_event_features(trips: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    trips = trips.copy()
    trips["event_count"] = 0
    trips["event_type_primary"] = ""
    trips["in_event"] = 0
    trips["hours_to_event_start"] = np.nan
    trips["hours_since_event_end"] = np.nan
    trips["event_attendance"] = 0.0

    for _, ev in events.iterrows():
        if ev["status"] == "cancelled":
            continue

        zones = ev["zones"] if ev["zones"] else CANONICAL_ZONES
        start = ev["start"] - pd.Timedelta(hours=2)
        end = ev["end"] + pd.Timedelta(hours=2)

        mask = (
            trips["zone"].isin(zones)
            & (trips["pickup_hour"] >= start)
            & (trips["pickup_hour"] <= end)
        )

        if not mask.any():
            continue

        trips.loc[mask, "event_count"] += 1
        trips.loc[mask, "event_type_primary"] = trips.loc[mask, "event_type_primary"].apply(
            lambda x: x if x else ev["event_type"]
        )
        trips.loc[mask, "event_attendance"] = np.maximum(
            trips.loc[mask, "event_attendance"], ev["attendance"] or 0
        )

        hrs_to = (ev["start"] - trips.loc[mask, "pickup_hour"]).dt.total_seconds() / 3600
        hrs_since = (trips.loc[mask, "pickup_hour"] - ev["end"]).dt.total_seconds() / 3600

        trips.loc[mask, "hours_to_event_start"] = np.minimum(
            trips.loc[mask, "hours_to_event_start"].fillna(hrs_to), hrs_to
        )
        trips.loc[mask, "hours_since_event_end"] = np.minimum(
            trips.loc[mask, "hours_since_event_end"].fillna(hrs_since), hrs_since
        )

        in_mask = mask & (trips["pickup_hour"] >= ev["start"]) & (trips["pickup_hour"] <= ev["end"])
        trips.loc[in_mask, "in_event"] = 1

    return trips


def build_master():
    train = clean_trips(pd.read_csv(RAW_FILES["train"]))
    test = clean_trips(pd.read_csv(RAW_FILES["test"]))
    weather = clean_weather(pd.read_csv(RAW_FILES["weather"]))
    events = clean_events(pd.read_csv(RAW_FILES["events"]))

    # Weather join
    train = train.merge(weather, left_on="pickup_hour", right_on="timestamp", how="left", validate="many_to_one")
    test = test.merge(weather, left_on="pickup_hour", right_on="timestamp", how="left", validate="many_to_one")

    # Events
    train = add_event_features(train, events)
    test = add_event_features(test, events)

    # Save
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    train.to_csv(DATA_PROCESSED / "master_train_raw.csv", index=False)
    test.to_csv(DATA_PROCESSED / "master_test_raw.csv", index=False)
    return train, test, weather, events


if __name__ == "__main__":
    build_master()