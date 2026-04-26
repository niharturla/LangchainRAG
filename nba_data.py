import re
from datetime import datetime

from nba_api.stats.endpoints import commonplayerinfo, leaguegamefinder, playergamelog, scoreboardv2
from nba_api.stats.static import players, teams

from predict_xgb import predict_player_market

TEAM_DATA = teams.get_teams()
TEAM_BY_ABBR = {t["abbreviation"].upper(): t for t in TEAM_DATA}
TEAM_BY_NICKNAME = {t["nickname"].lower(): t for t in TEAM_DATA}
TEAM_BY_FULL = {t["full_name"].lower(): t for t in TEAM_DATA}

PLAYER_DATA = players.get_players()
PLAYER_BY_FULL = {p["full_name"].lower(): p for p in PLAYER_DATA}

MARKET_MAP = {
    "points": "PTS",
    "point": "PTS",
    "pts": "PTS",
    "rebounds": "REB",
    "rebound": "REB",
    "rebs": "REB",
    "assists": "AST",
    "assist": "AST",
    "asts": "AST",
    "threes": "FG3M",
    "3pm": "FG3M",
}


def parse_betting_question(question: str) -> dict:
    normalized = question.lower()
    market = "PTS"
    for token, stat in MARKET_MAP.items():
        if re.search(rf"\b{re.escape(token)}\b", normalized):
            market = stat
            break

    line_match = re.search(r"(?<!\d)(\d{1,2}(?:\.\d)?)(?!\d)", normalized)
    line = float(line_match.group(1)) if line_match else None

    player_name = _infer_player_name(question)
    player_team = _get_player_team_abbr(player_name) if player_name else None
    teams_found = _infer_teams(normalized)

    team = player_team or (teams_found[0] if teams_found else None)
    opponent = None
    if player_team:
        for abbr in teams_found:
            if abbr != player_team:
                opponent = abbr
                break
    elif len(teams_found) > 1:
        team, opponent = teams_found[0], teams_found[1]

    if team and not opponent:
        opponent = _infer_opponent_from_today_scoreboard(team)

    return {
        "player_name": player_name,
        "team": team,
        "opponent": opponent,
        "market": market,
        "line": line,
    }


def _infer_opponent_from_today_scoreboard(team_abbr: str) -> str | None:
    today = datetime.now().strftime("%m/%d/%Y")
    board = scoreboardv2.ScoreboardV2(game_date=today)
    game_header = board.game_header.get_data_frame()
    line_score = board.line_score.get_data_frame()
    if game_header.empty or line_score.empty:
        return None

    for _, game in game_header.iterrows():
        game_id = game["GAME_ID"]
        teams_in_game = line_score[line_score["GAME_ID"] == game_id]["TEAM_ABBREVIATION"].tolist()
        if team_abbr in teams_in_game and len(teams_in_game) == 2:
            return teams_in_game[0] if teams_in_game[1] == team_abbr else teams_in_game[1]
    return None


def _infer_player_name(question: str) -> str | None:
    normalized = question.lower()

    # Highest confidence: direct full-name containment, case-insensitive.
    full_matches = [name for name in PLAYER_BY_FULL if f" {name} " in f" {normalized} "]
    if full_matches:
        full_matches.sort(key=len, reverse=True)
        return PLAYER_BY_FULL[full_matches[0]]["full_name"]

    # Fallback: title-case phrase extraction for prompts with capitalization.
    name_candidates = re.findall(r"\b([A-Z][A-Za-z'\.-]+(?:\s+[A-Z][A-Za-z'\.-]+){1,2})\b", question)
    for candidate in name_candidates:
        matches = players.find_players_by_full_name(candidate)
        if matches:
            return matches[0]["full_name"]

    return None


def _get_player_team_abbr(player_name: str) -> str | None:
    matches = players.find_players_by_full_name(player_name)
    if not matches:
        return None
    player_id = matches[0]["id"]
    info = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
    if info.empty:
        return None
    return info.iloc[0]["TEAM_ABBREVIATION"]


def _infer_teams(normalized_question: str) -> list[str]:
    found: list[str] = []
    for full_name, team in TEAM_BY_FULL.items():
        if full_name in normalized_question:
            found.append(team["abbreviation"])
    for nickname, team in TEAM_BY_NICKNAME.items():
        if re.search(rf"\b{re.escape(nickname)}\b", normalized_question):
            found.append(team["abbreviation"])
    for abbr in TEAM_BY_ABBR:
        if re.search(rf"\b{re.escape(abbr.lower())}\b", normalized_question):
            found.append(abbr)

    deduped: list[str] = []
    seen = set()
    for abbr in found:
        if abbr in seen:
            continue
        seen.add(abbr)
        deduped.append(abbr)
    return deduped[:2]


def _today_scoreboard_text() -> str:
    today = datetime.now().strftime("%m/%d/%Y")
    board = scoreboardv2.ScoreboardV2(game_date=today)
    games = board.game_header.get_data_frame()
    if games.empty:
        return "No games on today's NBA scoreboard."

    team_lines = board.line_score.get_data_frame()
    lines = []
    for _, game in games.iterrows():
        game_id = game["GAME_ID"]
        game_teams = team_lines[team_lines["GAME_ID"] == game_id]
        matchup = " vs ".join(game_teams["TEAM_ABBREVIATION"].tolist())
        lines.append(f"{matchup} - {game['GAME_STATUS_TEXT']}")
    return "\n".join(lines)


def _today_matchup_for_team(team_abbr: str | None) -> str:
    if not team_abbr:
        return "No team inferred; cannot resolve today's specific matchup."

    today = datetime.now().strftime("%m/%d/%Y")
    board = scoreboardv2.ScoreboardV2(game_date=today)
    game_header = board.game_header.get_data_frame()
    line_score = board.line_score.get_data_frame()
    if game_header.empty or line_score.empty:
        return "No games on today's NBA scoreboard."

    for _, game in game_header.iterrows():
        game_id = game["GAME_ID"]
        game_teams = line_score[line_score["GAME_ID"] == game_id]["TEAM_ABBREVIATION"].tolist()
        if team_abbr in game_teams and len(game_teams) == 2:
            opp = game_teams[0] if game_teams[1] == team_abbr else game_teams[1]
            return f"Today's matchup for {team_abbr}: {team_abbr} vs {opp} ({game['GAME_STATUS_TEXT']})"

    return f"No scheduled game found today for {team_abbr}."


def _team_recent_form(team_abbr: str, games: int = 8) -> str:
    team = TEAM_BY_ABBR.get(team_abbr.upper())
    if not team:
        return f"No team data found for {team_abbr}."

    logs = leaguegamefinder.LeagueGameFinder(team_id_nullable=team["id"]).get_data_frames()[0]
    if logs.empty:
        return f"No recent games found for {team_abbr}."

    recent = logs.head(games).copy()
    wins = int((recent["WL"] == "W").sum())
    losses = games - wins
    ppg = recent["PTS"].mean()
    opp_ppg = (recent["PTS"] - recent["PLUS_MINUS"]).mean()
    pace_proxy = (recent["FGA"] + recent["FTA"] * 0.44 + recent["TOV"]).mean()

    return (
        f"{team_abbr} last {games}: {wins}-{losses}, "
        f"PPG {ppg:.1f}, Opp PPG {opp_ppg:.1f}, Pace proxy {pace_proxy:.1f}"
    )


def _recent_head_to_head(team_abbr: str | None, opponent_abbr: str | None, games: int = 3) -> str:
    if not team_abbr or not opponent_abbr:
        return "Head-to-head: unavailable (missing inferred team or opponent)."

    team = TEAM_BY_ABBR.get(team_abbr.upper())
    if not team:
        return f"Head-to-head: unavailable (unknown team {team_abbr})."

    logs = leaguegamefinder.LeagueGameFinder(team_id_nullable=team["id"]).get_data_frames()[0]
    if logs.empty:
        return f"Head-to-head: no recent games for {team_abbr}."

    h2h = logs[logs["MATCHUP"].str.contains(opponent_abbr, case=False, na=False)].head(games)
    if h2h.empty:
        return f"Head-to-head: no recent {team_abbr} vs {opponent_abbr} games found."

    lines = []
    for _, row in h2h.iterrows():
        lines.append(
            f"{row['GAME_DATE']} {row['MATCHUP']} -> {int(row['PTS'])} pts, {row['WL']}, +/- {int(row['PLUS_MINUS'])}"
        )
    return "Head-to-head recent games:\n" + "\n".join(lines)


def _player_market_snapshot(
    player_name: str,
    market: str,
    line: float | None,
    opponent: str | None,
    games: int = 10,
) -> tuple[str, dict]:
    matches = players.find_players_by_full_name(player_name)
    if not matches:
        return f"Could not find player '{player_name}' in nba_api.", {}

    player_id = matches[0]["id"]
    season = f"{datetime.now().year - 1}-{str(datetime.now().year)[-2:]}"
    logs = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
    if logs.empty:
        return f"No recent logs found for {player_name} in season {season}.", {}

    recent = logs.head(games).copy()
    avg = recent[market].mean()
    median = recent[market].median()
    std_dev = recent[market].std(ddof=0)
    text = f"{market} last {games}: avg {avg:.2f}, median {median:.2f}, std {std_dev:.2f}"
    metrics = {"avg": float(avg), "median": float(median), "std": float(std_dev), "sample_size": games}

    if line is not None:
        over_hits = int((recent[market] > line).sum())
        under_hits = int((recent[market] < line).sum())
        push_hits = int((recent[market] == line).sum())
        text += f"; vs line {line}: over {over_hits}/{games}, under {under_hits}/{games}, push {push_hits}/{games}"
        metrics.update(
            {
                "line": float(line),
                "over_hits": over_hits,
                "under_hits": under_hits,
                "push_hits": push_hits,
                "over_rate": float(over_hits / games),
                "under_rate": float(under_hits / games),
            }
        )

    if opponent:
        opp_games = recent[recent["MATCHUP"].str.contains(opponent, case=False, na=False)]
        if not opp_games.empty:
            opp_avg = opp_games[market].mean()
            text += f"; recent vs {opponent}: avg {opp_avg:.2f} across {len(opp_games)} games"
            metrics["opponent_avg"] = float(opp_avg)
            metrics["opponent_sample_size"] = int(len(opp_games))
        else:
            text += f"; recent vs {opponent}: no head-to-head sample in last {games}"
            metrics["opponent_sample_size"] = 0

    return text, metrics


def _model_signal(metrics: dict, low_confidence: bool) -> str:
    if low_confidence or not metrics:
        return "PASS_SIGNAL"
    xgb_signal = metrics.get("xgb_signal")
    if xgb_signal in {"OVER_SIGNAL", "UNDER_SIGNAL", "PASS_SIGNAL"}:
        return xgb_signal
    over_rate = metrics.get("over_rate")
    under_rate = metrics.get("under_rate")
    if over_rate is None or under_rate is None:
        return "PASS_SIGNAL"
    if over_rate >= 0.6:
        return "OVER_SIGNAL"
    if under_rate >= 0.6:
        return "UNDER_SIGNAL"
    return "PASS_SIGNAL"


def build_betting_context(question: str) -> dict:
    parsed = parse_betting_question(question)
    player_name = parsed["player_name"]
    team = parsed["team"]
    opponent = parsed["opponent"]
    market = parsed["market"]
    line = parsed["line"]

    scoreboard_text = _today_scoreboard_text()
    matchup_text = _today_matchup_for_team(team)
    team_text = _team_recent_form(team) if team else "No team inferred from question."
    opponent_text = _team_recent_form(opponent) if opponent else "No opponent inferred from question."
    h2h_text = _recent_head_to_head(team, opponent, games=3)
    metrics: dict = {}
    if player_name:
        player_text, metrics = _player_market_snapshot(player_name, market, line, opponent)
    else:
        player_text = "No player inferred from question. Lean should be Pass."

    xgb_result = None
    xgb_pred = None
    xgb_proba_over = None
    xgb_signal = "PASS_SIGNAL"
    xgb_status = "not_run"
    xgb_message = "XGBoost was not run."
    if player_name:
        xgb_result = predict_player_market(player_name, market, opponent, line)
        if xgb_result:
            xgb_status = xgb_result.get("status", "unknown")
            xgb_message = xgb_result.get("message", "")
            if xgb_result["prediction"] is not None:
                xgb_pred = round(float(xgb_result["prediction"]), 2)
            xgb_signal = xgb_result["signal"]
            metrics["xgb_prediction"] = xgb_pred
            metrics["xgb_signal"] = xgb_signal
            if xgb_result["proba_over"] is not None:
                xgb_proba_over = round(float(xgb_result["proba_over"]), 3)
                metrics["xgb_proba_over"] = xgb_proba_over
            if xgb_result["prediction"] is not None:
                player_text += (
                    f"; XGBoost expected {market}: {xgb_result['prediction']:.2f}"
                )
            if xgb_result["proba_over"] is not None and line is not None:
                player_text += f"; XGBoost P(Over {line})={xgb_result['proba_over']:.3f}"
        else:
            metrics["xgb_signal"] = "PASS_SIGNAL"

    context = (
        f"Question parsed:\n"
        f"- Player: {player_name or 'N/A'}\n"
        f"- Team: {team or 'N/A'}\n"
        f"- Opponent: {opponent or 'N/A'}\n"
        f"- Market: {market}\n"
        f"- Line: {line if line is not None else 'N/A'}\n\n"
        f"Today's scoreboard:\n{scoreboard_text}\n\n"
        f"Relevant matchup:\n{matchup_text}\n\n"
        f"Team trend:\n{team_text}\n\n"
        f"Opponent trend:\n{opponent_text}\n\n"
        f"{h2h_text}\n\n"
        f"Player trend:\n{player_text}"
    )
    low_confidence = (player_name is None) or (line is None)
    parsed["context"] = context
    parsed["low_confidence"] = low_confidence
    parsed["model_signal"] = _model_signal(metrics, low_confidence)
    parsed["metrics"] = metrics
    parsed["xgb_pred"] = xgb_pred
    parsed["xgb_proba_over"] = xgb_proba_over
    parsed["xgb_signal"] = xgb_signal
    parsed["xgb_status"] = xgb_status
    parsed["xgb_message"] = xgb_message
    return parsed
