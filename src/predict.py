import numpy as np
import pandas as pd
import joblib

from .config import RAW_FILES, SUBMISSION, MODELS
from .features import FEATURES, CAT_FEATURES


def make_submission(master_test: pd.DataFrame, model_path=MODELS / "final_model.joblib"):
    bundle = joblib.load(model_path)
    model = bundle["model"]
    features = bundle["features"]
    cat_features = bundle["cat_features"]

    df = master_test.copy()
    for c in cat_features:
        df[c] = df[c].astype(str).astype("category")

    preds = np.clip(model.predict(df[features]), 0, None)

    template = pd.read_csv(RAW_FILES["submission_template"])
    pred_map = dict(zip(df["row_id"], preds))
    template["predicted_trips"] = template["row_id"].map(pred_map)

    assert template["predicted_trips"].notna().all()
    assert (template["predicted_trips"] >= 0).all()

    SUBMISSION.mkdir(parents=True, exist_ok=True)
    out = SUBMISSION / "team_qiyas_ride_submission.csv"
    template.to_csv(out, index=False)
    print(f"Saved {out} with {len(template)} rows")
    return template