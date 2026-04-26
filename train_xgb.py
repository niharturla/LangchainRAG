from __future__ import annotations

import json
import os

import pandas as pd
import xgboost as xgb
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
from sklearn.metrics import mean_absolute_error

from features import FEATURE_COLUMNS, build_training_frame

MODEL_DIR = "models"
SUPPORTED_MARKETS = ("PTS", "REB", "AST")


def _season_strings(seasons_back: int = 2) -> list[str]:
    from datetime import datetime

    now = datetime.now()
    seasons = []
    for start_year in range(now.year - seasons_back, now.year):
        seasons.append(f"{start_year}-{str(start_year + 1)[-2:]}")
    return seasons


def _collect_player_logs(player_id: int, seasons: list[str]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        try:
            frame = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
        except Exception:
            continue
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def train_market_model(market: str = "PTS", player_limit: int = 25, seasons_back: int = 2) -> dict:
    os.makedirs(MODEL_DIR, exist_ok=True)
    seasons = _season_strings(seasons_back=seasons_back)
    active_players = players.get_active_players()[:player_limit]

    training_frames = []
    for player in active_players:
        logs = _collect_player_logs(player["id"], seasons)
        if logs.empty:
            continue
        frame = build_training_frame(logs, market)
        if not frame.empty:
            training_frames.append(frame)

    if not training_frames:
        raise RuntimeError("No training data collected. Try increasing player_limit.")

    dataset = pd.concat(training_frames, ignore_index=True)
    dataset = dataset.sort_values("GAME_DATE").reset_index(drop=True)

    split_idx = int(len(dataset) * 0.8)
    train_df = dataset.iloc[:split_idx]
    test_df = dataset.iloc[split_idx:]

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(train_df[FEATURE_COLUMNS], train_df["target"])

    preds = model.predict(test_df[FEATURE_COLUMNS])
    mae = float(mean_absolute_error(test_df["target"], preds))
    residual_std = float((test_df["target"] - preds).std(ddof=0))

    model_path = os.path.join(MODEL_DIR, f"xgb_{market.lower()}.json")
    meta_path = os.path.join(MODEL_DIR, f"xgb_{market.lower()}_meta.json")
    model.save_model(model_path)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "market": market,
                "mae": mae,
                "residual_std": residual_std,
                "player_limit": player_limit,
                "seasons": seasons,
                "rows": int(len(dataset)),
            },
            handle,
            indent=2,
        )

    return {
        "market": market,
        "rows": int(len(dataset)),
        "mae": mae,
        "residual_std": residual_std,
        "model_path": model_path,
    }


if __name__ == "__main__":
    for market in SUPPORTED_MARKETS:
        result = train_market_model(market=market)
        print(f"Trained {market} model:", result)
