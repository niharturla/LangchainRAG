from __future__ import annotations

import json
import math
import os

import xgboost as xgb
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players

from features import FEATURE_COLUMNS, build_live_feature_row

MODEL_DIR = "models"


def _season_string() -> str:
    from datetime import datetime

    now = datetime.now()
    return f"{now.year - 1}-{str(now.year)[-2:]}"


def _model_paths(market: str) -> tuple[str, str]:
    model_path = os.path.join(MODEL_DIR, f"xgb_{market.lower()}.json")
    meta_path = os.path.join(MODEL_DIR, f"xgb_{market.lower()}_meta.json")
    return model_path, meta_path


def _load_model_and_meta(market: str):
    model_path, meta_path = _model_paths(market)
    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        return None, None

    model = xgb.XGBRegressor()
    model.load_model(model_path)
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    return model, meta


def predict_player_market(player_name: str, market: str, opponent: str | None, line_value: float | None) -> dict | None:
    model, meta = _load_model_and_meta(market)
    if model is None or meta is None:
        return {
            "prediction": None,
            "residual_std": None,
            "proba_over": None,
            "signal": "PASS_SIGNAL",
            "features": {},
            "status": "model_missing",
            "message": f"No trained XGBoost model found for {market}. Run ./venv/bin/python train_xgb.py",
        }

    matches = players.find_players_by_full_name(player_name)
    if not matches:
        return {
            "prediction": None,
            "residual_std": None,
            "proba_over": None,
            "signal": "PASS_SIGNAL",
            "features": {},
            "status": "player_missing",
            "message": f"Player '{player_name}' not found for XGBoost inference.",
        }

    player_id = matches[0]["id"]
    logs = playergamelog.PlayerGameLog(player_id=player_id, season=_season_string()).get_data_frames()[0]
    if logs.empty:
        return {
            "prediction": None,
            "residual_std": None,
            "proba_over": None,
            "signal": "PASS_SIGNAL",
            "features": {},
            "status": "logs_missing",
            "message": f"No recent logs for {player_name} to run XGBoost inference.",
        }

    features = build_live_feature_row(logs, market, opponent, line_value)
    if features.empty:
        return {
            "prediction": None,
            "residual_std": None,
            "proba_over": None,
            "signal": "PASS_SIGNAL",
            "features": {},
            "status": "feature_missing",
            "message": f"Insufficient features for XGBoost inference on {player_name}.",
        }

    prediction = float(model.predict(features[FEATURE_COLUMNS])[0])
    residual_std = float(meta.get("residual_std", 4.0))
    if line_value is None:
        signal = "PASS_SIGNAL"
        proba_over = None
    else:
        z_score = (prediction - line_value) / max(residual_std, 1.0)
        proba_over = 1 / (1 + math.exp(-z_score))
        if proba_over >= 0.58:
            signal = "OVER_SIGNAL"
        elif proba_over <= 0.42:
            signal = "UNDER_SIGNAL"
        else:
            signal = "PASS_SIGNAL"

    return {
        "prediction": prediction,
        "residual_std": residual_std,
        "proba_over": proba_over,
        "signal": signal,
        "features": features.iloc[0].to_dict(),
        "status": "ok",
        "message": "XGBoost inference successful.",
    }
