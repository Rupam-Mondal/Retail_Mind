
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf


def _find_project_dir():
    current_dir = Path.cwd().resolve()
    return next(
        (folder for folder in [current_dir, *current_dir.parents]
         if (folder / "models" / "forecasting").is_dir()),
        current_dir.parent if current_dir.name == "src" else current_dir
    )


def forecast_demand(steps=None, project_dir=None, save_csv=True):
    root = Path(project_dir).resolve() if project_dir else _find_project_dir()
    model_dir = root / "models" / "forecasting"
    model_path = model_dir / "retailmind_seq2seq_attention.keras"
    metadata_path = model_dir / "retailmind_seq2seq_metadata.json"

    if not model_path.is_file():
        raise FileNotFoundError(f"Trained Seq2Seq model not found: {model_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Model metadata not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    horizon = int(metadata["forecast_horizon"])
    if steps is None:
        steps = horizon
    if int(steps) != horizon:
        raise ValueError(
            f"This model was trained for {horizon} steps. "
            f"Use steps={horizon} or train a model with a different horizon."
        )

    model = tf.keras.models.load_model(model_path, compile=False)
    look_back = int(metadata["look_back"])
    window = np.asarray(
        metadata["last_window_scaled"], dtype=np.float32
    ).reshape(1, look_back, 1)

    scale_min = float(metadata["scale_min"])
    scale_range = float(metadata["scale_max"]) - scale_min
    if scale_range == 0:
        scale_range = 1.0

    decoder = np.zeros((1, horizon, 1), dtype=np.float32)
    decoder[:, 0, 0] = window[:, -1, 0]
    predictions = []

    for step in range(horizon):
        output = model.predict([window, decoder], verbose=0)
        scaled_value = float(output[0, step, 0])
        actual_value = max(0.0, scaled_value * scale_range + scale_min)
        predictions.append(actual_value)
        if step + 1 < horizon:
            decoder[0, step + 1, 0] = (actual_value - scale_min) / scale_range

    start_date = pd.Timestamp(metadata["last_date"]) + pd.Timedelta(days=1)
    dates = pd.date_range(start=start_date, periods=horizon, freq="D")
    result = pd.DataFrame({
        "date": dates,
        "store": metadata["store_id"],
        "item": metadata["item_id"],
        "forecast_sales": predictions
    })

    if save_csv:
        report_dir = root / "reports" / "day10_seq2seq_attention"
        report_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(report_dir / "multi_week_demand_forecast.csv", index=False)

    return result
