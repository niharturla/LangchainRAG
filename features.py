from __future__ import annotations

import re

import pandas as pd
from nba_api.stats.static import teams

TEAM_DATA = teams.get_teams()
TEAM_ID_BY_ABBR = {team["abbreviation"].upper(): int(team["id"]) for team in TEAM_DATA}

FEATURE_COLUMNS = [
    "last_3_avg",
    "last_5_avg",
    "last_10_avg",
    "last_10_std",
    "last_game_value",
    "minutes_last_5",
    "fga_last_5",
    "fta_last_5",
    "tov_last_5",
    "home_flag",
    "days_rest",
    "is_b2b",
    "opponent_id",
    "line_value",
]


def parse_opponent_from_matchup(matchup: str) -> str | None:
    match = re.search(r"\bvs\.\s+([A-Z]{2,3})|\b@\s+([A-Z]{2,3})", matchup)
    if not match:
        return None
    return (match.group(1) or match.group(2) or "").upper() or None


def is_home_from_matchup(matchup: str) -> int:
    return 1 if "vs." in matchup else 0


def build_training_frame(logs_df: pd.DataFrame, market: str) -> pd.DataFrame:
    if logs_df.empty:
        return pd.DataFrame()

    df = logs_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    df["target"] = df[market].astype(float)
    df["opponent_abbr"] = df["MATCHUP"].apply(parse_opponent_from_matchup)
    df["opponent_id"] = df["opponent_abbr"].map(TEAM_ID_BY_ABBR).fillna(0).astype(int)
    df["home_flag"] = df["MATCHUP"].apply(is_home_from_matchup).astype(int)
    df["days_rest"] = df["GAME_DATE"].diff().dt.days.fillna(3).clip(lower=0, upper=7)
    df["is_b2b"] = (df["days_rest"] <= 1).astype(int)

    shifted_target = df["target"].shift(1)
    shifted_min = df["MIN"].astype(float).shift(1)
    shifted_fga = df["FGA"].astype(float).shift(1)
    shifted_fta = df["FTA"].astype(float).shift(1)
    shifted_tov = df["TOV"].astype(float).shift(1)

    df["last_3_avg"] = shifted_target.rolling(3).mean()
    df["last_5_avg"] = shifted_target.rolling(5).mean()
    df["last_10_avg"] = shifted_target.rolling(10).mean()
    df["last_10_std"] = shifted_target.rolling(10).std(ddof=0)
    df["last_game_value"] = shifted_target
    df["minutes_last_5"] = shifted_min.rolling(5).mean()
    df["fga_last_5"] = shifted_fga.rolling(5).mean()
    df["fta_last_5"] = shifted_fta.rolling(5).mean()
    df["tov_last_5"] = shifted_tov.rolling(5).mean()
    df["line_value"] = df["last_10_avg"]

    df = df.dropna(subset=["last_10_avg", "last_10_std"]).reset_index(drop=True)
    return df


def build_live_feature_row(logs_df: pd.DataFrame, market: str, opponent_abbr: str | None, line_value: float | None) -> pd.DataFrame:
    if logs_df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    df = logs_df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    if len(df) < 10:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    recent = df.tail(10).copy()
    last_five = df.tail(5).copy()
    last_game = df.tail(1).iloc[0]
    prev_game = df.tail(2).iloc[0] if len(df) >= 2 else last_game

    days_rest = (pd.to_datetime(last_game["GAME_DATE"]) - pd.to_datetime(prev_game["GAME_DATE"])).days
    row = {
        "last_3_avg": float(recent.tail(3)[market].mean()),
        "last_5_avg": float(last_five[market].mean()),
        "last_10_avg": float(recent[market].mean()),
        "last_10_std": float(recent[market].std(ddof=0) or 0.0),
        "last_game_value": float(last_game[market]),
        "minutes_last_5": float(last_five["MIN"].astype(float).mean()),
        "fga_last_5": float(last_five["FGA"].astype(float).mean()),
        "fta_last_5": float(last_five["FTA"].astype(float).mean()),
        "tov_last_5": float(last_five["TOV"].astype(float).mean()),
        "home_flag": 1,
        "days_rest": float(max(0, min(days_rest, 7))),
        "is_b2b": int(days_rest <= 1),
        "opponent_id": int(TEAM_ID_BY_ABBR.get((opponent_abbr or "").upper(), 0)),
        "line_value": float(line_value if line_value is not None else recent[market].mean()),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
