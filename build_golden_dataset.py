"""Generate ``golden_dataset.jsonl`` for the Project 1 soccer-facts eval harness.

DESIGN PRINCIPLE (read this before editing):
Every "answer" in the output file is computed directly from your local Soccer
Data API responses -- never from the model's memory or from an external source.

Your wrapper API (openapi.json) exposes::

    GET /health
    GET /understat/schedule?league=&season=
    GET /understat/player-season-stats?league=&season=
    GET /statsbomb/matches?competition_id=&season_id=
    GET /statsbomb/matches/{match_id}/events

Understat only covers the "big five" European leagues. Fill in
``STATSBOMB_TARGETS`` below with valid ``(competition_id, season_id)`` pairs
from StatsBomb's open data. Every actual match/event fact comes from your
wrapper at ``BASE_URL``.

Reproducibility:
    Given the same ``--seed`` and the same API responses, this script emits a
    byte-identical dataset. Fetching happens concurrently, but every sampling
    and ordering decision runs in a deterministic second phase.

Ambiguity:
    Questions whose answer is a tie (two joint top scorers, two teams level on
    points, etc.) are dropped -- a ground-truth file must have one right answer.

Usage::

    python build_golden_dataset.py --output golden_dataset.jsonl --target 50

Also writes ``golden_dataset.provenance.jsonl`` so you can spot-check every
answer against its source before trusting it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp

# ---------------------------------------------------------------------------
# Config -- edit before running a full generation pass (step 3 above).
# ---------------------------------------------------------------------------

BASE_URL = "http://0.0.0.0:8000"

# Understat covers only: "ENG-Premier League", "ESP-La Liga", "ITA-Serie A",
# "GER-Bundesliga", "FRA-Ligue 1". Season format per the API's own docstring
# is like "2023-2024".
UNDERSTAT_TARGETS: list[tuple[str, str]] = [
    ("ENG-Premier League", "2021-2022"),
    ("ENG-Premier League", "2022-2023"),
    ("ESP-La Liga", "2021-2022"),
    ("ITA-Serie A", "2021-2022"),
    ("GER-Bundesliga", "2021-2022"),
    ("FRA-Ligue 1", "2021-2022"),
]

# StatsBomb open data only publishes specific competition/season pairs. Fill
# these in with valid IDs. Left empty by default so the script doesn't silently
# generate zero StatsBomb questions using unverified IDs.
STATSBOMB_TARGETS: list[tuple[int, int]] = [
    # (competition_id, season_id),
]

# How many StatsBomb matches (per target) to pull play-by-play events for, to
# generate first-goal questions. Event calls are heavy -- keep this small.
STATSBOMB_EVENTS_SAMPLE_PER_TARGET = 3

# How many scoreline questions to draw per league-season / competition-season.
SCORELINES_PER_TARGET = 3

RANDOM_SEED = 7

HTTP_TIMEOUT_SECONDS = 30.0
HTTP_MAX_RETRIES = 3

JSONObj = dict[str, Any]

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


class FetchError(Exception):
    """A request to the wrapper API failed or returned something unusable.

    Attributes:
        path (str): The request path that failed.
        status (int): The HTTP status code, or 0 for a transport-level failure.
        body (str): The (truncated) response body or error description.
    """

    def __init__(self, path: str, status: int, body: str) -> None:
        """Store the failing request's context.

        Args:
            path (str): The request path that failed.
            status (int): The HTTP status code, or 0 for a transport failure.
            body (str): The response body or error description.
        """
        super().__init__(f"{path} -> HTTP {status}: {body[:300]}")
        self.path = path
        self.status = status
        self.body = body


async def fetch_json(
    session: aiohttp.ClientSession,
    path: str,
    params: dict[str, Any] | None = None,
) -> Any:
    """GET ``path`` and parse the JSON body, retrying transient failures.

    4xx/5xx responses raise immediately (no retry); connection errors and
    timeouts are retried with linear backoff up to ``HTTP_MAX_RETRIES``.

    Args:
        session (aiohttp.ClientSession): Session bound to the wrapper base URL.
        path (str): Request path, e.g. ``"/understat/schedule"``.
        params (dict[str, Any] | None): Query parameters.

    Returns:
        Any: The decoded JSON payload.

    Raises:
        FetchError: On an HTTP error status, an unparseable body, or repeated
            transport failures.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, HTTP_MAX_RETRIES + 1):
        try:
            async with session.get(path, params=params) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    raise FetchError(path, resp.status, text)
                try:
                    return json.loads(text)
                except json.JSONDecodeError as exc:
                    raise FetchError(path, resp.status, f"invalid JSON: {exc}")
        except (aiohttp.ClientError, TimeoutError) as exc:
            last_exc = exc
            if attempt < HTTP_MAX_RETRIES:
                await asyncio.sleep(0.5 * attempt)
    raise FetchError(
        path, 0, f"transport failure after {HTTP_MAX_RETRIES} attempts: {last_exc}"
    )


def pick(record: JSONObj, *candidates: str, default: Any = None) -> Any:
    """Return the first present, non-None value among ``candidates``.

    Field names from soccerdata/statsbombpy wrappers vary by version, so every
    extraction tries a short list of plausible names.

    Args:
        record (JSONObj): The record to read from.
        *candidates (str): Keys to try, in priority order.
        default (Any): Value to return if no candidate is present.

    Returns:
        Any: The first matching value, or ``default``.
    """
    for candidate in candidates:
        if record.get(candidate) is not None:
            return record[candidate]
    return default


def unwrap_name(value: Any) -> Any:
    """Flatten a nested ``{"id": ..., "name": ...}`` dict to its name.

    Args:
        value (Any): A plain value or a nested id/name dict.

    Returns:
        Any: ``value["name"]`` if ``value`` is such a dict, else ``value``.
    """
    if isinstance(value, dict) and "name" in value:
        return value["name"]
    return value


def format_season(season: str) -> str:
    """Normalise ``"2021-2022"`` to ``"2021-22"`` for question text.

    Args:
        season (str): A season string, possibly ``YYYY-YYYY`` or ``YYYY/YYYY``.

    Returns:
        str: The shortened form, or ``season`` unchanged if it doesn't match.
    """
    parts = season.replace("/", "-").split("-")
    if len(parts) == 2 and len(parts[0]) == 4 and len(parts[1]) == 4:
        return f"{parts[0]}-{parts[1][2:]}"
    return season


def league_label(league: str) -> str:
    """Strip the country prefix: ``"ENG-Premier League"`` -> ``"Premier League"``.

    Args:
        league (str): A soccerdata league identifier.

    Returns:
        str: The human-facing league name.
    """
    return league.split("-", 1)[-1]


def to_int(value: Any) -> int | None:
    """Best-effort int conversion.

    Args:
        value (Any): Anything that might be an integer.

    Returns:
        int | None: The integer value, or None if it can't be converted.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Normalised records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchRow:
    """A single played match, normalised across the two data sources.

    Attributes:
        home (str): Home team name.
        away (str): Away team name.
        home_goals (int): Full-time home goals.
        away_goals (int): Full-time away goals.
        date (str): Match date as returned by the API.
        competition (str | None): Competition name, when the source provides it.
    """

    home: str
    away: str
    home_goals: int
    away_goals: int
    date: str
    competition: str | None = None


def parse_understat_match(record: JSONObj) -> MatchRow | None:
    """Extract a :class:`MatchRow` from an Understat schedule record.

    Args:
        record (JSONObj): One entry from ``/understat/schedule``.

    Returns:
        MatchRow | None: The parsed row, or None for an unplayed fixture or
        unrecognised field names.
    """
    home = unwrap_name(pick(record, "home_team", "home"))
    away = unwrap_name(pick(record, "away_team", "away"))
    home_goals = to_int(pick(record, "home_goals", "home_score", "hgoal", "goals_home"))
    away_goals = to_int(pick(record, "away_goals", "away_score", "agoal", "goals_away"))
    date = pick(record, "date", "match_date", "game_date")
    if home is None or away is None or home_goals is None or away_goals is None:
        return None
    if date is None:
        return None
    return MatchRow(str(home), str(away), home_goals, away_goals, str(date))


def parse_statsbomb_match(record: JSONObj) -> MatchRow | None:
    """Extract a :class:`MatchRow` from a StatsBomb matches record.

    Args:
        record (JSONObj): One entry from ``/statsbomb/matches``.

    Returns:
        MatchRow | None: The parsed row, or None if required fields are missing.
    """
    home = unwrap_name(pick(record, "home_team", "home_team_name"))
    away = unwrap_name(pick(record, "away_team", "away_team_name"))
    home_goals = to_int(pick(record, "home_score"))
    away_goals = to_int(pick(record, "away_score"))
    date = pick(record, "match_date", "date")
    competition = unwrap_name(pick(record, "competition", "competition_name"))
    if home is None or away is None or home_goals is None or away_goals is None:
        return None
    if date is None:
        return None
    return MatchRow(
        str(home),
        str(away),
        home_goals,
        away_goals,
        str(date),
        str(competition) if competition is not None else None,
    )


def statsbomb_match_id(record: JSONObj) -> int | None:
    """Return the numeric match id from a StatsBomb matches record.

    Args:
        record (JSONObj): One entry from ``/statsbomb/matches``.

    Returns:
        int | None: The match id, or None if absent/non-numeric.
    """
    return to_int(pick(record, "match_id", "id"))


# ---------------------------------------------------------------------------
# Fetch phase -- everything comes from BASE_URL (your wrapper) only
# ---------------------------------------------------------------------------


@dataclass
class UnderstatPayload:
    """Raw Understat responses for one league-season.

    Attributes:
        league (str): The league identifier requested.
        season (str): The season string requested.
        schedule (list[JSONObj]): Response from ``/understat/schedule``.
        players (list[JSONObj]): Response from ``/understat/player-season-stats``.
    """

    league: str
    season: str
    schedule: list[JSONObj]
    players: list[JSONObj]


@dataclass
class StatsbombPayload:
    """Raw StatsBomb responses for one competition-season.

    Attributes:
        competition_id (int): The competition id requested.
        season_id (int): The season id requested.
        matches (list[JSONObj]): Response from ``/statsbomb/matches``.
        events (dict[int, list[JSONObj]]): match id -> event list, for a sample.
    """

    competition_id: int
    season_id: int
    matches: list[JSONObj]
    events: dict[int, list[JSONObj]]


@dataclass
class FetchResult:
    """Everything the generation phase needs, plus non-fatal warnings.

    Attributes:
        understat (list[UnderstatPayload]): One entry per Understat target that
            responded successfully.
        statsbomb (list[StatsbombPayload]): One entry per StatsBomb target that
            responded successfully.
        warnings (list[str]): Human-readable notes about targets that failed.
    """

    understat: list[UnderstatPayload]
    statsbomb: list[StatsbombPayload]
    warnings: list[str]


async def fetch_understat_schedule(
    session: aiohttp.ClientSession, league: str, season: str
) -> list[JSONObj]:
    """Fetch ``/understat/schedule`` for one league-season.

    Args:
        session (aiohttp.ClientSession): Session bound to the wrapper base URL.
        league (str): League identifier.
        season (str): Season string.

    Returns:
        list[JSONObj]: The schedule records.
    """
    return await fetch_json(
        session, "/understat/schedule", {"league": league, "season": season}
    )


async def fetch_understat_player_stats(
    session: aiohttp.ClientSession, league: str, season: str
) -> list[JSONObj]:
    """Fetch ``/understat/player-season-stats`` for one league-season.

    Args:
        session (aiohttp.ClientSession): Session bound to the wrapper base URL.
        league (str): League identifier.
        season (str): Season string.

    Returns:
        list[JSONObj]: The player-season records.
    """
    return await fetch_json(
        session,
        "/understat/player-season-stats",
        {"league": league, "season": season},
    )


async def fetch_statsbomb_matches(
    session: aiohttp.ClientSession, competition_id: int, season_id: int
) -> list[JSONObj]:
    """Fetch ``/statsbomb/matches`` for one competition-season.

    Args:
        session (aiohttp.ClientSession): Session bound to the wrapper base URL.
        competition_id (int): Competition id.
        season_id (int): Season id.

    Returns:
        list[JSONObj]: The match records.
    """
    return await fetch_json(
        session,
        "/statsbomb/matches",
        {"competition_id": competition_id, "season_id": season_id},
    )


async def fetch_statsbomb_events(
    session: aiohttp.ClientSession, match_id: int
) -> list[JSONObj]:
    """Fetch play-by-play events for one StatsBomb match.

    Args:
        session (aiohttp.ClientSession): Session bound to the wrapper base URL.
        match_id (int): The match id.

    Returns:
        list[JSONObj]: The event records.
    """
    return await fetch_json(session, f"/statsbomb/matches/{match_id}/events")


async def _fetch_understat_target(
    session: aiohttp.ClientSession, league: str, season: str
) -> tuple[UnderstatPayload | None, list[str]]:
    """Fetch both Understat endpoints for one target.

    Args:
        session (aiohttp.ClientSession): Session bound to the wrapper base URL.
        league (str): League identifier.
        season (str): Season string.

    Returns:
        tuple[UnderstatPayload | None, list[str]]: The payload (or None on
        failure) and any warning messages.
    """
    try:
        schedule, players = await asyncio.gather(
            fetch_understat_schedule(session, league, season),
            fetch_understat_player_stats(session, league, season),
        )
    except FetchError as exc:
        return None, [f"understat {league}/{season}: {exc}"]
    return UnderstatPayload(league, season, schedule, players), []


async def _fetch_statsbomb_target(
    session: aiohttp.ClientSession,
    competition_id: int,
    season_id: int,
    seed: int,
) -> tuple[StatsbombPayload | None, list[str]]:
    """Fetch matches and a deterministic sample of event lists for one target.

    The event sample is chosen with an RNG seeded from ``seed`` and the target
    ids, so it does not depend on network timing.

    Args:
        session (aiohttp.ClientSession): Session bound to the wrapper base URL.
        competition_id (int): Competition id.
        season_id (int): Season id.
        seed (int): Base random seed.

    Returns:
        tuple[StatsbombPayload | None, list[str]]: The payload (or None on a
        failed matches call) and any warning messages.
    """
    try:
        matches = await fetch_statsbomb_matches(session, competition_id, season_id)
    except FetchError as exc:
        return None, [f"statsbomb matches {competition_id}/{season_id}: {exc}"]

    played = sorted(
        (m for m in matches if parse_statsbomb_match(m) is not None),
        key=lambda m: statsbomb_match_id(m) or 0,
    )
    rng = random.Random(f"{seed}-{competition_id}-{season_id}")
    k = min(STATSBOMB_EVENTS_SAMPLE_PER_TARGET, len(played))
    sample = rng.sample(played, k) if k else []

    events: dict[int, list[JSONObj]] = {}
    warnings: list[str] = []

    async def one(record: JSONObj) -> None:
        match_id = statsbomb_match_id(record)
        if match_id is None:
            return
        try:
            events[match_id] = await fetch_statsbomb_events(session, match_id)
        except FetchError as exc:
            warnings.append(f"statsbomb events {match_id}: {exc}")

    await asyncio.gather(*(one(record) for record in sample))
    payload = StatsbombPayload(competition_id, season_id, matches, events)
    return payload, warnings


async def fetch_all(base_url: str, seed: int) -> FetchResult:
    """Fetch every configured target concurrently.

    Args:
        base_url (str): The wrapper API base URL.
        seed (int): Base random seed (used only for the event-sample choice).

    Returns:
        FetchResult: Payloads for targets that responded, plus warnings.
    """
    result = FetchResult(understat=[], statsbomb=[], warnings=[])
    timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(base_url=base_url, timeout=timeout) as session:
        understat_coros = [
            _fetch_understat_target(session, league, season)
            for league, season in UNDERSTAT_TARGETS
        ]
        statsbomb_coros = [
            _fetch_statsbomb_target(session, comp_id, season_id, seed)
            for comp_id, season_id in STATSBOMB_TARGETS
        ]
        understat_out = await asyncio.gather(*understat_coros)
        statsbomb_out = await asyncio.gather(*statsbomb_coros)

    for understat_payload, warnings in understat_out:
        result.warnings.extend(warnings)
        if understat_payload is not None:
            result.understat.append(understat_payload)
    for statsbomb_payload, warnings in statsbomb_out:
        result.warnings.extend(warnings)
        if statsbomb_payload is not None:
            result.statsbomb.append(statsbomb_payload)
    return result


# ---------------------------------------------------------------------------
# Standings (computed from real results -- not an API field)
# ---------------------------------------------------------------------------


@dataclass
class TeamRecord:
    """A team's aggregate record over a set of matches.

    Attributes:
        points (int): League points (3 win / 1 draw / 0 loss).
        goals_for (int): Goals scored.
        goals_against (int): Goals conceded.
        played (int): Matches played.
    """

    points: int = 0
    goals_for: int = 0
    goals_against: int = 0
    played: int = 0

    @property
    def goal_difference(self) -> int:
        """int: Goals scored minus goals conceded."""
        return self.goals_for - self.goals_against


def compute_standings(matches: list[MatchRow]) -> list[tuple[str, TeamRecord]]:
    """Build a league table from played matches, sorted best-first.

    Sort key is points, then goal difference, then goals for, then team name
    (the last purely to keep the ordering deterministic).

    Args:
        matches (list[MatchRow]): Played matches for one league-season.

    Returns:
        list[tuple[str, TeamRecord]]: ``(team, record)`` pairs, best-first.
    """
    table: dict[str, TeamRecord] = defaultdict(TeamRecord)
    for match in matches:
        home, away = table[match.home], table[match.away]
        home.played += 1
        away.played += 1
        home.goals_for += match.home_goals
        home.goals_against += match.away_goals
        away.goals_for += match.away_goals
        away.goals_against += match.home_goals
        if match.home_goals > match.away_goals:
            home.points += 3
        elif match.home_goals < match.away_goals:
            away.points += 3
        else:
            home.points += 1
            away.points += 1
    return sorted(
        table.items(),
        key=lambda kv: (
            -kv[1].points,
            -kv[1].goal_difference,
            -kv[1].goals_for,
            kv[0],
        ),
    )


# ---------------------------------------------------------------------------
# Question generators
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QAItem:
    """One golden question with its verified answer and provenance.

    Attributes:
        question (str): The natural-language question.
        answer (str): The ground-truth answer, computed from the API.
        category (str): A tag used to keep the trimmed dataset balanced.
        provenance (JSONObj): Source endpoint, params and derivation notes;
            written to the side file only, never to ``golden_dataset.jsonl``.
    """

    question: str
    answer: str
    category: str
    provenance: JSONObj


def _leader(ranked: list[tuple[str, int]]) -> tuple[str, int] | None:
    """Return the unambiguous leader of a ``(name, value)`` ranking.

    Args:
        ranked (list[tuple[str, int]]): Descending-by-value ranking.

    Returns:
        tuple[str, int] | None: The top ``(name, value)``, or None when the
        list is empty or the top value is tied.
    """
    if not ranked:
        return None
    if len(ranked) >= 2 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0]


def _rank_players(players: list[JSONObj], *stat_keys: str) -> list[tuple[str, int]]:
    """Rank players by a stat, descending, tie-broken by name for determinism.

    Args:
        players (list[JSONObj]): Player-season records.
        *stat_keys (str): Candidate keys holding the stat (first present wins).

    Returns:
        list[tuple[str, int]]: ``(player, value)`` pairs, best-first.
    """
    ranked: list[tuple[str, int]] = []
    for player in players:
        name = pick(player, "player", "player_name", "name")
        value = to_int(pick(player, *stat_keys))
        if name is None or value is None:
            continue
        ranked.append((str(name), value))
    ranked.sort(key=lambda t: (-t[1], t[0]))
    return ranked


def gen_league_champion(payload: UnderstatPayload) -> list[QAItem]:
    """Generate the "who won league X" question for one Understat target.

    Args:
        payload (UnderstatPayload): Raw responses for one league-season.

    Returns:
        list[QAItem]: Zero or one item (zero if the season is partial or the
        title race is level on the chosen tie-breakers).
    """
    matches = [
        row for row in map(parse_understat_match, payload.schedule) if row is not None
    ]
    standings = compute_standings(matches)
    if len(standings) < 2 or standings[0][1].played < 10:
        return []
    if standings[0][1].points == standings[1][1].points:
        return []
    winner = standings[0][0]
    label = league_label(payload.league)
    question = f"Who won the {label} in the {format_season(payload.season)} season?"
    provenance: JSONObj = {
        "endpoint": "/understat/schedule",
        "params": {"league": payload.league, "season": payload.season},
        "derivation": "standings (3/1/0 pts) computed from all played matches",
        "top_of_table": [
            {
                "team": team,
                "points": rec.points,
                "goal_difference": rec.goal_difference,
                "played": rec.played,
            }
            for team, rec in standings[:3]
        ],
    }
    return [QAItem(question, winner, "league_champion", provenance)]


def gen_scorelines(payload: UnderstatPayload, count: int, rng: random.Random) -> list[QAItem]:
    """Generate final-score questions for a sample of Understat matches.

    Args:
        payload (UnderstatPayload): Raw responses for one league-season.
        count (int): How many matches to sample.
        rng (random.Random): Seeded RNG for the sample.

    Returns:
        list[QAItem]: One item per sampled match.
    """
    played = [
        row for row in map(parse_understat_match, payload.schedule) if row is not None
    ]
    played.sort(key=lambda m: (m.date, m.home, m.away))
    sample = rng.sample(played, min(count, len(played)))
    label = league_label(payload.league)
    season = format_season(payload.season)
    out: list[QAItem] = []
    for match in sample:
        question = (
            f"What was the final score when {match.home} played {match.away} "
            f"on {match.date} ({label}, {season})?"
        )
        answer = f"{match.home} {match.home_goals}-{match.away_goals} {match.away}"
        provenance: JSONObj = {
            "endpoint": "/understat/schedule",
            "params": {"league": payload.league, "season": payload.season},
            "matched_record": {
                "home_team": match.home,
                "away_team": match.away,
                "date": match.date,
            },
        }
        out.append(QAItem(question, answer, "scoreline", provenance))
    return out


def gen_top_scorer(payload: UnderstatPayload) -> list[QAItem]:
    """Generate top-scorer name and goal-count questions for one target.

    Args:
        payload (UnderstatPayload): Raw responses for one league-season.

    Returns:
        list[QAItem]: Two items, or none if the goal lead is tied.
    """
    ranked = _rank_players(payload.players, "goals", "G")
    leader = _leader(ranked)
    if leader is None:
        return []
    name, goals = leader
    label = league_label(payload.league)
    season = format_season(payload.season)
    provenance: JSONObj = {
        "endpoint": "/understat/player-season-stats",
        "params": {"league": payload.league, "season": payload.season},
        "top_5_by_goals": ranked[:5],
    }
    return [
        QAItem(
            f"Who was the top scorer in the {label} {season} season?",
            name,
            "top_scorer",
            provenance,
        ),
        QAItem(
            f"How many goals did {name} score in the {label} {season} season?",
            str(goals),
            "top_scorer_count",
            provenance,
        ),
    ]


def gen_assist_leader(payload: UnderstatPayload) -> list[QAItem]:
    """Generate the assist-leader question for one Understat target.

    Args:
        payload (UnderstatPayload): Raw responses for one league-season.

    Returns:
        list[QAItem]: One item, or none if the assist lead is tied.
    """
    ranked = _rank_players(payload.players, "assists", "A")
    leader = _leader(ranked)
    if leader is None:
        return []
    name, _ = leader
    label = league_label(payload.league)
    season = format_season(payload.season)
    provenance: JSONObj = {
        "endpoint": "/understat/player-season-stats",
        "params": {"league": payload.league, "season": payload.season},
        "top_5_by_assists": ranked[:5],
    }
    question = f"Who had the most assists in the {label} {season} season?"
    return [QAItem(question, name, "assist_leader", provenance)]


def gen_statsbomb_scorelines(
    payload: StatsbombPayload, count: int, rng: random.Random
) -> list[QAItem]:
    """Generate score questions for a sample of StatsBomb matches.

    Args:
        payload (StatsbombPayload): Raw responses for one competition-season.
        count (int): How many matches to sample.
        rng (random.Random): Seeded RNG for the sample.

    Returns:
        list[QAItem]: One item per sampled match.
    """
    played = [
        row for row in map(parse_statsbomb_match, payload.matches) if row is not None
    ]
    played.sort(key=lambda m: (m.date, m.home, m.away))
    sample = rng.sample(played, min(count, len(played)))
    out: list[QAItem] = []
    for match in sample:
        comp_txt = f" in the {match.competition}" if match.competition else ""
        question = (
            f"What was the score of the match between {match.home} and "
            f"{match.away} on {match.date}{comp_txt}?"
        )
        answer = f"{match.home} {match.home_goals}-{match.away_goals} {match.away}"
        provenance: JSONObj = {
            "endpoint": "/statsbomb/matches",
            "params": {
                "competition_id": payload.competition_id,
                "season_id": payload.season_id,
            },
            "matched_record": {
                "home_team": match.home,
                "away_team": match.away,
                "date": match.date,
            },
        }
        out.append(QAItem(question, answer, "sb_scoreline", provenance))
    return out


def gen_statsbomb_first_goal(
    payload: StatsbombPayload, match: MatchRow, match_id: int, events: list[JSONObj]
) -> list[QAItem]:
    """Generate first-goal-scorer and first-goal-minute questions for a match.

    Args:
        payload (StatsbombPayload): The owning competition-season payload.
        match (MatchRow): The parsed match record.
        match_id (int): The StatsBomb match id.
        events (list[JSONObj]): The match's event list.

    Returns:
        list[QAItem]: Two items, or none if no goal is found or the opening
        goal's timestamp is shared by another goal.
    """
    goals: list[tuple[int, int, str, str]] = []
    for event in events:
        if unwrap_name(pick(event, "type")) != "Shot":
            continue
        if unwrap_name(pick(event, "shot_outcome")) != "Goal":
            continue
        minute = to_int(pick(event, "minute"))
        player = unwrap_name(pick(event, "player"))
        if minute is None or player is None:
            continue
        second = to_int(pick(event, "second")) or 0
        team = unwrap_name(pick(event, "team"))
        goals.append((minute, second, str(player), str(team) if team else ""))
    if not goals:
        return []
    goals.sort(key=lambda g: (g[0], g[1]))
    if len(goals) >= 2 and goals[0][:2] == goals[1][:2]:
        return []  # ambiguous opening goal (identical timestamp)
    minute, _, scorer, team = goals[0]
    provenance: JSONObj = {
        "endpoint": "/statsbomb/matches/{match_id}/events",
        "params": {"match_id": match_id, "competition_id": payload.competition_id},
        "matched_event": {"minute": minute, "player": scorer, "team": team},
    }
    prefix = f"the {match.home} vs {match.away} match on {match.date}"
    return [
        QAItem(
            f"Who scored the first goal in {prefix}?",
            scorer,
            "sb_first_goal",
            provenance,
        ),
        QAItem(
            f"In which minute was the opening goal scored in {prefix}?",
            str(minute),
            "sb_first_goal_minute",
            provenance,
        ),
    ]


def generate_items(fetched: FetchResult, seed: int) -> list[QAItem]:
    """Run every generator over the fetched payloads, deterministically.

    Args:
        fetched (FetchResult): Payloads from the fetch phase.
        seed (int): Random seed for all sampling in this phase.

    Returns:
        list[QAItem]: Every generated item, before de-duplication and trimming.
    """
    rng = random.Random(seed)
    items: list[QAItem] = []

    for payload in sorted(fetched.understat, key=lambda p: (p.league, p.season)):
        items.extend(gen_league_champion(payload))
        items.extend(gen_scorelines(payload, SCORELINES_PER_TARGET, rng))
        items.extend(gen_top_scorer(payload))
        items.extend(gen_assist_leader(payload))

    for payload in sorted(
        fetched.statsbomb, key=lambda p: (p.competition_id, p.season_id)
    ):
        items.extend(gen_statsbomb_scorelines(payload, SCORELINES_PER_TARGET, rng))
        by_id = {
            statsbomb_match_id(m): m
            for m in payload.matches
            if statsbomb_match_id(m) is not None
        }
        for match_id in sorted(payload.events):
            record = by_id.get(match_id)
            if record is None:
                continue
            match = parse_statsbomb_match(record)
            if match is None:
                continue
            items.extend(
                gen_statsbomb_first_goal(
                    payload, match, match_id, payload.events[match_id]
                )
            )
    return items


def select_balanced(items: list[QAItem], target: int, seed: int) -> list[QAItem]:
    """De-duplicate, then round-robin across categories up to ``target``.

    Round-robin keeps the trimmed set from being dominated by whichever
    category happened to produce the most questions.

    Args:
        items (list[QAItem]): Candidate items.
        target (int): Desired item count.
        seed (int): Seed for the per-category shuffle.

    Returns:
        list[QAItem]: The selected items, in category-cycled order.
    """
    rng = random.Random(seed)
    seen: set[str] = set()
    by_category: dict[str, list[QAItem]] = defaultdict(list)
    for item in items:
        if item.question in seen:
            continue
        seen.add(item.question)
        by_category[item.category].append(item)
    for bucket in by_category.values():
        rng.shuffle(bucket)

    categories = sorted(by_category)
    out: list[QAItem] = []
    while len(out) < target and any(by_category[cat] for cat in categories):
        for cat in categories:
            if by_category[cat]:
                out.append(by_category[cat].pop())
                if len(out) >= target:
                    break
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_outputs(items: list[QAItem], output_path: Path) -> tuple[Path, Path]:
    """Write the eval file and its side-car provenance file.

    Args:
        items (list[QAItem]): The selected items.
        output_path (Path): Destination for ``golden_dataset.jsonl``.

    Returns:
        tuple[Path, Path]: ``(dataset_path, provenance_path)``.
    """
    with output_path.open("w") as handle:
        for item in items:
            handle.write(
                json.dumps({"question": item.question, "answer": item.answer}) + "\n"
            )

    prov_path = output_path.with_suffix("").with_suffix(".provenance.jsonl")
    with prov_path.open("w") as handle:
        for item in items:
            handle.write(
                json.dumps(
                    {
                        "question": item.question,
                        "answer": item.answer,
                        "category": item.category,
                        "provenance": item.provenance,
                    },
                    default=str,
                )
                + "\n"
            )
    return output_path, prov_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv (list[str] | None): Argument vector, defaulting to ``sys.argv``.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", default=BASE_URL, help="wrapper API base URL")
    parser.add_argument("--output", default="golden_dataset.jsonl", help="output path")
    parser.add_argument(
        "--target", type=int, default=50, help="target pair count (plan calls for 40-60)"
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="random seed")
    return parser.parse_args(argv)


def run_generation(base_url: str, target: int, seed: int, output: Path) -> int:
    """Fetch, generate, select, and write the dataset.

    Args:
        base_url (str): The wrapper API base URL.
        target (int): Target item count.
        seed (int): Random seed.
        output (Path): Output path for the dataset.

    Returns:
        int: A process exit code (0 on success).
    """
    if not UNDERSTAT_TARGETS and not STATSBOMB_TARGETS:
        print(
            "No targets configured -- edit UNDERSTAT_TARGETS / STATSBOMB_TARGETS "
            "at the top of this script."
        )
        return 1

    fetched = asyncio.run(fetch_all(base_url, seed))
    for warning in fetched.warnings:
        print(f"  warning: {warning}")

    items = generate_items(fetched, seed)
    selected = select_balanced(items, target, seed)

    if not selected:
        print("No Q&A pairs generated -- check the API is running and reachable.")
        return 1
    if len(selected) < 40:
        print(
            f"Only {len(selected)} pairs generated (plan target: 40-60). "
            "Add more UNDERSTAT_TARGETS / STATSBOMB_TARGETS and rerun."
        )

    dataset_path, prov_path = write_outputs(selected, output)
    by_category: dict[str, int] = defaultdict(int)
    for item in selected:
        by_category[item.category] += 1
    print(f"Wrote {len(selected)} pairs to {dataset_path}")
    print(f"  by category: {dict(sorted(by_category.items()))}")
    print(f"Wrote matching provenance to {prov_path}")
    print("Spot-check a sample of provenance entries against the live API first.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv (list[str] | None): Argument vector, defaulting to ``sys.argv``.

    Returns:
        int: A process exit code.
    """
    args = parse_args(argv)
    return run_generation(args.base_url, args.target, args.seed, Path(args.output))


if __name__ == "__main__":
    sys.exit(main())
