"""
CFB Score Prediction, Rankings & Playoff Projection -- single-file pipeline
============================================================================
Input:  /mnt/user-data/uploads/cfb_box-scores_2002-2025.csv  (2002-2025 box scores)

Pipeline:
  1. build_game_features()   -> engineered, leakage-free, pre-game features
                                 (rolling scoring form, Elo ratings, rest days, etc.)
  2. train_models()           -> two XGBoost regressors: score_home, score_away
  3. top25()                  -> Elo-based Top 25 for any season
  4. project_conference_championships() -> top-2-by-conference-record matchup,
                                            predicted by the model
  5. project_playoff()        -> seeds a 12-team CFP field and simulates every
                                  round with the model

Run it end to end:
    python3 cfb_pipeline.py

To reuse on a real, unplayed 2026 schedule: build a DataFrame of games with
columns [season, week, date, away, home, conf_away, conf_home, neutral] (no
scores yet), append it to the historical games before running
build_game_features(), and step through it week by week -- predicting each
game, then feeding the PREDICTED score back into the rolling-form/Elo update
before predicting the next week. See the `simulate_future_week()` helper
at the bottom for exactly that pattern.
"""
import pickle
import math
import statistics
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

def round_half_up(x):
    """Standard rounding (0.5 always rounds up), unlike numpy/Python's
    round-half-to-even. Returns an int."""
    return int(math.floor(float(x) + 0.5))

def norm_cdf(x):
    """Standard normal CDF, no scipy dependency."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

RANDOM_SEED = 7  # test seed
CHAOS_DAMPENING = 0.3  # scales down the overall randomness in simulated outcomes (1.0 = full, empirically-calibrated variance)

def compute_team_chaos(games, clip_range=(0.9, 1.1)):
    """How much a team's own scoring margin swings around, relative to the
    league average -- a rough 'boom or bust' factor. Team with a history of
    both blowout wins and blowout losses gets a HIGHER multiplier (more
    chaos in their games); a team that grinds out similar-margin results
    every week gets a lower one. Clipped so no team's chaos gets wild
    enough to swamp everything else."""
    away = games[["away", "score_away", "score_home"]].rename(
        columns={"away": "team", "score_away": "team_score", "score_home": "opp_score"})
    home = games[["home", "score_home", "score_away"]].rename(
        columns={"home": "team", "score_home": "team_score", "score_away": "opp_score"})
    long = pd.concat([away, home], ignore_index=True).dropna()
    long["margin"] = long["team_score"] - long["opp_score"]
    team_std = long.groupby("team")["margin"].std()
    league_std = long["margin"].std()
    chaos = (team_std / league_std).fillna(1.0).clip(*clip_range)
    return chaos.to_dict()

def game_sigma_for(models, team_a, team_b):
    """Per-game uncertainty, scaled by how chaotic these two specific teams
    have historically been. Returns (prob_sigma, noise_sigma):
    prob_sigma is the FULL, empirically-calibrated value -- this is what
    win probabilities must be computed from, or they stop matching the
    model's real historical accuracy. noise_sigma is prob_sigma scaled
    down by CHAOS_DAMPENING -- used ONLY for the actual random draw that
    picks a realized score/winner, so a single simulated season doesn't
    compound implausible upset chains. Conflating the two (an earlier bug)
    made every win probability look far more extreme than the model
    actually supports -- a 10-point favorite showing 89% instead of the
    correctly-calibrated 73%."""
    base_sigma = models.get("margin_sigma", 14.0)
    chaos = models.get("team_chaos", {})
    chaos_a, chaos_b = chaos.get(team_a, 1.0), chaos.get(team_b, 1.0)
    prob_sigma = base_sigma * np.sqrt((chaos_a ** 2 + chaos_b ** 2) / 2)
    noise_sigma = prob_sigma * CHAOS_DAMPENING
    return prob_sigma, noise_sigma

def realize_chaotic_score(pred_a, pred_b, sigma, floor=3.0, rng=None):
    """Draws ONE random, plausible outcome consistent with the model's own
    uncertainty -- not just the safest point estimate. This is what makes
    occasional upsets and lopsided blowouts possible in the simulation,
    the same way they happen in a real season, instead of the higher-rated
    team winning by a modest margin literally every single time."""
    rng = rng or np.random
    noise = rng.normal(0, sigma, size=np.shape(pred_a) if hasattr(pred_a, "__len__") else None)
    total = pred_a + pred_b
    realized_margin = (pred_a - pred_b) + noise
    real_a = np.maximum((total + realized_margin) / 2, floor)
    real_b = np.maximum((total - realized_margin) / 2, floor)
    return real_a, real_b, realized_margin

RAW_PATH = "/mnt/user-data/uploads/cfb_box-scores_2002-2025.csv"

# Overtime simulation -- rounding two independently-noised scores can
# occasionally land on an exact tie (e.g. 27-27), which isn't a legal final
# score in college football. When that happens, simulate real NCAA
# overtime rules rather than just leaving/breaking the tie arbitrarily:
#   OT1: each team gets one possession, TD try is a normal PAT kick
#   OT2: each team gets one possession, TD try MUST be a 2-pt conversion
#   OT3+: no possessions, teams alternate 2-pt conversion attempts only
# First team to lead after a complete round wins.
OT_P_TOUCHDOWN = 0.55
OT_P_FIELD_GOAL = 0.25
OT_P_PAT_GOOD = 0.98
OT_P_TWO_PT_GOOD = 0.50

def simulate_overtime(rng=None, max_periods=15):
    """Returns (home_ot_points, away_ot_points), the ADDITIONAL points from
    overtime periods to add to a tied regulation score. Guaranteed to
    return an unequal pair (a game cannot legally end in a tie)."""
    rng = rng or np.random

    def possession(two_pt_required):
        r = rng.random()
        if r < OT_P_TOUCHDOWN:
            if two_pt_required:
                return 6 + (2 if rng.random() < OT_P_TWO_PT_GOOD else 0)
            return 6 + (1 if rng.random() < OT_P_PAT_GOOD else 0)
        if r < OT_P_TOUCHDOWN + OT_P_FIELD_GOAL:
            return 3
        return 0

    home_pts, away_pts = 0, 0
    period = 1
    while period <= max_periods:
        if period <= 2:
            h = possession(two_pt_required=(period == 2))
            a = possession(two_pt_required=(period == 2))
        else:
            h = 2 if rng.random() < OT_P_TWO_PT_GOOD else 0
            a = 2 if rng.random() < OT_P_TWO_PT_GOOD else 0
        home_pts += h
        away_pts += a
        if home_pts != away_pts:
            return home_pts, away_pts
        period += 1
    # Vanishingly unlikely fallback so this can never infinite-loop.
    return (home_pts + 1, away_pts) if rng.random() < 0.5 else (home_pts, away_pts + 1)

def break_ties_scalar(home_score, away_score, rng=None):
    """Scalar version, for single hypothetical matchups (championships,
    playoff rounds)."""
    if home_score == away_score:
        h_ot, a_ot = simulate_overtime(rng=rng)
        return home_score + h_ot, away_score + a_ot
    return home_score, away_score

def break_ties_vectorized(home_scores, away_scores, rng=None):
    """Array version, for a full week of games at once. Loops only over
    the (rare) tied games rather than every game."""
    rng = rng or np.random
    home_scores = np.array(home_scores, dtype=float)
    away_scores = np.array(away_scores, dtype=float)
    tied = np.where(home_scores == away_scores)[0]
    for i in tied:
        h_ot, a_ot = simulate_overtime(rng=rng)
        home_scores[i] += h_ot
        away_scores[i] += a_ot
    return home_scores, away_scores

K_ELO = 22
HOME_FIELD_ELO = 55
ELO_START = 1500
TALENT_ELO_SCALE = 120  # Elo points per 1 std dev of combined recruiting+transfer+returning-production talent
COACHING_GRADE_ELO_SCALE = 280  # max Elo bonus for an A+ coaching hire; scales down to 0 at a "C" grade or worse (never negative)

FEATURES = [
    "elo_away_pre", "elo_home_pre", "elo_diff",
    "pts_for_L4_away", "pts_ag_L4_away", "pts_for_L8_away", "pts_ag_L8_away",
    "pts_for_L4_home", "pts_ag_L4_home", "pts_for_L8_home", "pts_ag_L8_home",
    "season_win_pct_away", "season_win_pct_home",
    "sos_away_pre", "sos_home_pre",
    "games_played_season_away", "games_played_season_home",
    "rest_days_away", "rest_days_home",
    "is_neutral", "is_postseason", "same_conf", "week",
]

# ---------------------------------------------------------------------------
# 1. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def load_raw():
    df = pd.read_csv(RAW_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "season", "week"]).reset_index(drop=True)
    df["game_id"] = df.index
    return df

def build_long(df):
    """One row per team per game (so rolling form can be computed per team)."""
    away = df.copy()
    away["team"], away["opp"], away["is_home"] = away["away"], away["home"], 0
    away["team_score"], away["opp_score"] = away["score_away"], away["score_home"]

    home = df.copy()
    home["team"], home["opp"], home["is_home"] = home["home"], home["away"], 1
    home["team_score"], home["opp_score"] = home["score_home"], home["score_away"]

    keep = ["game_id", "season", "week", "date", "team", "opp", "is_home",
            "team_score", "opp_score"]
    long_df = pd.concat([away[keep], home[keep]], ignore_index=True)
    return long_df.sort_values(["team", "date"]).reset_index(drop=True)

def add_rolling_form(long_df, windows=(4, 8)):
    """Pre-game rolling averages per team -- shifted by 1 so no leakage.

    pts_for_L4/L8 and pts_ag_L4/L8 are OPPONENT-ADJUSTED, not raw scoring
    margin. Raw points don't distinguish "blew out a bad team" from "lost
    close to a great team" -- a 47-point win over a team that normally
    gives up 40 barely means anything, while a 2-point loss to a team
    that normally blows everyone out is actually a strong showing. Instead,
    each game's performance is measured relative to the opponent's own
    season-to-date scoring profile (how many more/fewer points did I score
    than this opponent's defense usually allows; how many more/fewer
    points did I allow than this opponent's offense usually scores),
    and THAT adjusted number is what gets rolled up over the trailing
    window."""
    long_df = long_df.sort_values(["team", "date"]).copy()
    grp = long_df.groupby("team", group_keys=False)

    # Each team's own season-to-date scoring profile (pre-game, no leakage) --
    # this is the baseline opponents get adjusted against.
    long_df["team_season_avg_for"] = long_df.groupby(["team", "season"])["team_score"].transform(
        lambda s: s.shift(1).expanding().mean())
    long_df["team_season_avg_ag"] = long_df.groupby(["team", "season"])["opp_score"].transform(
        lambda s: s.shift(1).expanding().mean())

    # Attach each row's OPPONENT's own baseline (self-join on game_id).
    opp_baseline = long_df[["game_id", "team", "team_season_avg_for", "team_season_avg_ag"]].rename(
        columns={"team": "opp", "team_season_avg_for": "opp_season_avg_for", "team_season_avg_ag": "opp_season_avg_ag"})
    long_df = long_df.merge(opp_baseline, on=["game_id", "opp"], how="left")
    long_df = long_df.sort_values(["team", "date"]).reset_index(drop=True)

    # This game's performance relative to what this specific opponent
    # normally allows/scores.
    long_df["adj_pts_for"] = long_df["team_score"] - long_df["opp_season_avg_ag"]
    long_df["adj_pts_ag"] = long_df["opp_score"] - long_df["opp_season_avg_for"]

    grp = long_df.groupby("team", group_keys=False)
    for w in windows:
        long_df[f"pts_for_L{w}"] = grp["adj_pts_for"].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        long_df[f"pts_ag_L{w}"] = grp["adj_pts_ag"].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())

    long_df["margin"] = long_df["team_score"] - long_df["opp_score"]
    long_df["win"] = (long_df["margin"] > 0).astype(float)
    long_df["season_win_pct"] = (
        long_df.groupby(["team", "season"])["win"].transform(lambda s: s.shift(1).expanding().mean())
    )
    long_df["games_played_season"] = long_df.groupby(["team", "season"]).cumcount()
    long_df["prev_game_date"] = long_df.groupby("team")["date"].shift(1)
    long_df["rest_days"] = (long_df["date"] - long_df["prev_game_date"]).dt.days.fillna(120).clip(upper=120)
    return long_df

def standard_elo_delta(elo_home, elo_away, score_home, score_away, hf):
    """The normal Elo update: home team's rating change (away's is the
    negative of this). hf is the home-field Elo bump to apply (0 for
    neutral-site games)."""
    margin = score_home - score_away
    exp_home = 1 / (1 + 10 ** (-((elo_home + hf) - elo_away) / 400))
    result_home = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
    mov_mult = np.log(abs(margin) + 1) * (2.2 / (abs(elo_home - elo_away) * 0.001 + 2.2))
    return K_ELO * mov_mult * (result_home - exp_home)

# Conference championship games get a DIFFERENT, asymmetric rule for how
# they move a team's Elo for PLAYOFF SEEDING specifically (not for the
# season-long training Elo used everywhere else): winning still gains the
# normal, opponent-strength-dependent amount, but losing only costs
# anything if the loss is by more than 15 points -- a close championship
# loss shouldn't tank a team's seeding, only a real blowout loss should.
CHAMP_LOSS_NO_PENALTY_MAX_MARGIN = 15   # lose by this much or less: 0 change, stay stagnant
CHAMP_LOSS_MINOR_MAX_MARGIN = 24        # 16-24 pt loss: minor penalty
CHAMP_LOSS_MINOR_PENALTY = -15
CHAMP_LOSS_MAJOR_PENALTY = -30          # 25+ pt loss: bigger, but still capped, penalty

def championship_seeding_adjustment(elo_dict, champs_df):
    """Returns a COPY of elo_dict with conference-championship-game results
    applied using the asymmetric win/loss rule above -- used only to build
    the playoff seeding, never fed back into the persisted season Elo."""
    elo_dict = dict(elo_dict)
    for _, g in champs_df.iterrows():
        winner = g["predicted_champion"]
        team1, team2 = g["team1"], g["team2"]
        loser = team2 if winner == team1 else team1
        winner_score = g["pred_score_team1"] if winner == team1 else g["pred_score_team2"]
        loser_score = g["pred_score_team2"] if winner == team1 else g["pred_score_team1"]

        elo_winner = elo_dict.get(winner, ELO_START)
        elo_loser = elo_dict.get(loser, ELO_START)
        winner_gain = standard_elo_delta(elo_winner, elo_loser, winner_score, loser_score, hf=0)

        margin = winner_score - loser_score
        if margin <= CHAMP_LOSS_NO_PENALTY_MAX_MARGIN:
            loser_change = 0
        elif margin <= CHAMP_LOSS_MINOR_MAX_MARGIN:
            loser_change = CHAMP_LOSS_MINOR_PENALTY
        else:
            loser_change = CHAMP_LOSS_MAJOR_PENALTY

        elo_dict[winner] = elo_winner + winner_gain
        elo_dict[loser] = elo_loser + loser_change
    return elo_dict

def sos_tier_points(rank):
    """Points awarded for playing a team currently ranked this well (by
    live Elo, among teams with an established rating). Escalates as the
    opponent's rank tightens -- a top-10 win/game is worth much more than
    a bottom-of-top-50 one, and anything outside the top 50 is worth 0."""
    if rank <= 10:
        return 5
    if rank <= 20:
        return 4
    if rank <= 30:
        return 3
    if rank <= 40:
        return 2
    if rank <= 50:
        return 1
    return 0

def compute_elo(df, preseason_adjustments=None, extra_regression=None):
    """Sequential Elo update with margin-of-victory scaling and home-field edge.
    Returns df with pre-game Elo (and strength-of-schedule) columns, plus a
    dict of season -> {team: elo}.

    preseason_adjustments: optional {season: {team: elo_delta}}. Applied once,
    exactly when transitioning INTO that season -- e.g. {2026: {"Georgia": +80}}
    bumps Georgia's rating by 80 points on top of its normal season-to-season
    carryover, right before any 2026 games are played. Used to inject
    recruiting/transfer talent signal for a season that has no in-season
    results yet to build an Elo rating from.

    extra_regression: optional {season: (set_of_teams, retention)}. For a
    team with a new head coach, last season's results are much less
    predictive of this season -- the standard 0.85 carryover retention
    keeps too much of a trailing record that a coaching change makes
    obsolete. This pulls JUST those teams further toward 1500 (using
    `retention` instead of the standard 0.85) before the preseason talent
    bump is added, so a team like a coaching-change hire doesn't stay
    artificially anchored to two years of results under a different staff.

    Strength of schedule: each team accumulates points (see sos_tier_points)
    for every opponent played so far THIS season, based on that opponent's
    LIVE Elo rank at the moment the game was played -- naturally starts at
    0 for everyone's first game and grows week by week as the season goes,
    using only information that would have been known at the time (no
    leakage), and resets to 0 at the start of each new season."""
    preseason_adjustments = preseason_adjustments or {}
    extra_regression = extra_regression or {}
    elo, season_end_elo = {}, {}
    sos_points = {}
    pre_away, pre_home = np.zeros(len(df)), np.zeros(len(df))
    sos_pre_away, sos_pre_home = np.zeros(len(df)), np.zeros(len(df))
    current_season = None

    for i, row in df.iterrows():
        season = row["season"]
        if current_season is not None and season != current_season:
            season_end_elo[current_season] = elo.copy()
            elo = {t: ELO_START + 0.85 * (r - ELO_START) for t, r in elo.items()}  # regress toward mean
            if season in extra_regression:
                cc_teams, cc_retention = extra_regression[season]
                for t in cc_teams:
                    if t in elo:
                        elo[t] = ELO_START + cc_retention * (elo[t] - ELO_START)
            for team, delta in preseason_adjustments.get(season, {}).items():
                elo[team] = elo.get(team, ELO_START) + delta
            sos_points = {}  # strength of schedule resets fully every season
        current_season = season

        a, h = row["away"], row["home"]
        ea, eh = elo.get(a, ELO_START), elo.get(h, ELO_START)
        pre_away[i], pre_home[i] = ea, eh
        sos_pre_away[i] = sos_points.get(a, 0)
        sos_pre_home[i] = sos_points.get(h, 0)

        if pd.isna(row["score_home"]) or pd.isna(row["score_away"]):
            continue

        # Rank both teams among the live population BEFORE this game's own
        # Elo update, then award each side points for the OTHER team's
        # current tier -- this is what "SOS so far" grows by.
        ranked = sorted(elo.items(), key=lambda kv: -kv[1])
        rank_lookup = {t: idx + 1 for idx, (t, _) in enumerate(ranked)}
        rank_a = rank_lookup.get(a, len(ranked) + 1)
        rank_h = rank_lookup.get(h, len(ranked) + 1)
        sos_points[a] = sos_points.get(a, 0) + sos_tier_points(rank_h)
        sos_points[h] = sos_points.get(h, 0) + sos_tier_points(rank_a)

        margin = row["score_home"] - row["score_away"]
        hf = 0 if row["neutral"] else HOME_FIELD_ELO
        delta = standard_elo_delta(eh, ea, row["score_home"], row["score_away"], hf)
        elo[h], elo[a] = eh + delta, ea - delta

    season_end_elo[current_season] = elo.copy()
    df = df.copy()
    df["elo_away_pre"], df["elo_home_pre"] = pre_away, pre_home
    df["sos_away_pre"], df["sos_home_pre"] = sos_pre_away, sos_pre_home
    return df, season_end_elo

def build_game_features():
    raw = load_raw()
    raw, season_end_elo = compute_elo(raw)
    long_df = add_rolling_form(build_long(raw))

    feat_cols = ["pts_for_L4", "pts_ag_L4", "pts_for_L8", "pts_ag_L8",
                 "season_win_pct", "games_played_season", "rest_days"]
    away_feat = long_df[long_df["is_home"] == 0][["game_id"] + feat_cols].add_suffix("_away").rename(columns={"game_id_away": "game_id"})
    home_feat = long_df[long_df["is_home"] == 1][["game_id"] + feat_cols].add_suffix("_home").rename(columns={"game_id_home": "game_id"})

    games = raw.merge(away_feat, on="game_id").merge(home_feat, on="game_id")
    games["elo_diff"] = (games["elo_home_pre"] + np.where(games["neutral"], 0, HOME_FIELD_ELO)) - games["elo_away_pre"]
    games["is_neutral"] = games["neutral"].astype(int)
    games["is_postseason"] = (games["game_type"] == "post").astype(int)
    games["same_conf"] = (games["conf_away"] == games["conf_home"]).astype(int)
    return games, season_end_elo

# ---------------------------------------------------------------------------
# 2. MODEL TRAINING
# ---------------------------------------------------------------------------

def train_models(games):
    games = games.dropna(subset=["score_home", "score_away"]).copy()
    for c in FEATURES:
        games[c] = games[c].fillna(games[c].median())

    train_df = games[games["season"] <= 2023]
    val_df = games[games["season"] == 2024]
    test_df = games[games["season"] == 2025]

    # Single model, every season weighted equally -- no upweighting of
    # 2024/2025. Trained through 2023, validated on 2024, tested on 2025,
    # and this is the exact model used everywhere downstream (2025 backtest
    # and 2026 projections alike) -- no separate "production" retrain.
    models, mae_by_target = {}, {}
    for target in ["score_home", "score_away"]:
        model = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.03,
                              subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                              random_state=42, eval_metric="mae")
        model.fit(train_df[FEATURES], train_df[target],
                  eval_set=[(val_df[FEATURES], val_df[target])], verbose=False)
        models[target] = model
        mae_test = np.mean(np.abs(model.predict(test_df[FEATURES]) - test_df[target]))
        mae_by_target[target] = round(float(mae_test), 2)
        print(f"  {target}: 2025 holdout MAE = {mae_test:.2f} pts")

    pred_home = models["score_home"].predict(test_df[FEATURES])
    pred_away = models["score_away"].predict(test_df[FEATURES])
    acc = ((pred_home > pred_away) == (test_df["score_home"] > test_df["score_away"]).values).mean()
    print(f"  2025 holdout winner-pick accuracy = {acc:.1%}")

    pred_margin = pred_home - pred_away
    actual_margin = (test_df["score_home"] - test_df["score_away"]).values
    margin_sigma = float(np.std(actual_margin - pred_margin))
    print(f"  margin residual std (used for win probabilities AND chaos draws) = {margin_sigma:.2f} pts")
    print(f"  (real margin std {np.std(actual_margin):.1f} vs. raw predicted std {np.std(pred_margin):.1f} -- "
          f"adding {margin_sigma:.1f}-pt random noise per game restores realistic variance instead of "
          f"every score being a safe, hedged point estimate)")

    team_chaos = compute_team_chaos(games)
    chaos_ranked = sorted(team_chaos.items(), key=lambda kv: -kv[1])
    print(f"  Most 'boom or bust' teams (highest chaos multiplier): "
          f"{[f'{t} ({c:.2f}x)' for t, c in chaos_ranked[:3]]}")
    print(f"  Most consistent teams (lowest chaos multiplier): "
          f"{[f'{t} ({c:.2f}x)' for t, c in chaos_ranked[-3:]]}")

    models["winner_accuracy"] = round(float(acc), 4)
    models["mae_home"] = mae_by_target["score_home"]
    models["mae_away"] = mae_by_target["score_away"]
    models["margin_sigma"] = round(margin_sigma, 2)
    models["team_chaos"] = team_chaos
    return models

# ---------------------------------------------------------------------------
# 3. RATINGS / TOP 25
# ---------------------------------------------------------------------------

def top25(season_end_elo, season, n=25):
    ranked = sorted(season_end_elo[season].items(), key=lambda kv: -kv[1])[:n]
    return pd.DataFrame(ranked, columns=["team", "elo_rating"]).assign(rank=range(1, n + 1))[["rank", "team", "elo_rating"]]

def full_rank_lookup(elo_dict):
    """team -> rank (1 = best), for EVERY team in elo_dict, not just the
    Top 25 -- used to tag games with each team's current rank and to know
    who's ranked vs. unranked."""
    ranked = sorted(elo_dict.items(), key=lambda kv: -kv[1])
    return {t: i + 1 for i, (t, _) in enumerate(ranked)}

# ---------------------------------------------------------------------------
# 4 & 5. CONFERENCE CHAMPIONSHIPS + 12-TEAM CFP PROJECTION
# ---------------------------------------------------------------------------

def conference_standings(games, season):
    reg = games[(games.season == season) & (games.game_type == "regular") &
                (games.conf_away == games.conf_home) & games.conf_home.notna()]
    rows = []
    for _, g in reg.iterrows():
        rows.append({"conf": g.conf_home, "team": g.home, "win": int(g.score_home > g.score_away)})
        rows.append({"conf": g.conf_home, "team": g.away, "win": int(g.score_away > g.score_home)})
    df = pd.DataFrame(rows)
    standings = df.groupby(["conf", "team"]).agg(wins=("win", "sum"), games=("win", "count")).reset_index()
    standings["pct"] = standings["wins"] / standings["games"]
    return standings

def top2_by_conf(standings, elo):
    picks = {}
    for conf, sub in standings.groupby("conf"):
        sub = sub.copy()
        sub["elo"] = sub["team"].map(lambda t: elo.get(t, 1500))
        sub = sub.sort_values(["pct", "elo"], ascending=False)
        if len(sub) >= 2:
            picks[conf] = (sub.iloc[0]["team"], sub.iloc[1]["team"])
    return picks

def latest_team_row(games, season, team):
    """Most recent pre-game feature snapshot for a team, used to project a
    hypothetical future game (championship / playoff round)."""
    as_away = games[(games.season == season) & (games.away == team)]
    as_home = games[(games.season == season) & (games.home == team)]
    cand = []
    if len(as_away): cand.append((as_away.iloc[-1]["date"], "away", as_away.iloc[-1]))
    if len(as_home): cand.append((as_home.iloc[-1]["date"], "home", as_home.iloc[-1]))
    cand.sort(key=lambda x: x[0])
    _, side, row = cand[-1]
    p = "_away" if side == "away" else "_home"
    return {
        "elo_pre": row["elo_away_pre"] if side == "away" else row["elo_home_pre"],
        "sos_pre": row["sos_away_pre"] if side == "away" else row["sos_home_pre"],
        "pts_for_L4": row[f"pts_for_L4{p}"], "pts_ag_L4": row[f"pts_ag_L4{p}"],
        "pts_for_L8": row[f"pts_for_L8{p}"], "pts_ag_L8": row[f"pts_ag_L8{p}"],
        "season_win_pct": row[f"season_win_pct{p}"],
        "games_played_season": row[f"games_played_season{p}"] + 1,
        "rest_days": 10,
    }

def predict_matchup(models, team_a, team_b, feat_a, feat_b, neutral=True,
                     postseason=1, same_conf=0, week=16):
    """team_a plays the 'away' slot, team_b the 'home' slot (arbitrary at a
    neutral site -- doesn't affect the home-field bump when neutral=True)."""
    hf = 0 if neutral else HOME_FIELD_ELO
    row = {
        "elo_away_pre": feat_a["elo_pre"], "elo_home_pre": feat_b["elo_pre"],
        "elo_diff": (feat_b["elo_pre"] + hf) - feat_a["elo_pre"],
        "pts_for_L4_away": feat_a["pts_for_L4"], "pts_ag_L4_away": feat_a["pts_ag_L4"],
        "pts_for_L8_away": feat_a["pts_for_L8"], "pts_ag_L8_away": feat_a["pts_ag_L8"],
        "pts_for_L4_home": feat_b["pts_for_L4"], "pts_ag_L4_home": feat_b["pts_ag_L4"],
        "pts_for_L8_home": feat_b["pts_for_L8"], "pts_ag_L8_home": feat_b["pts_ag_L8"],
        "season_win_pct_away": feat_a["season_win_pct"], "season_win_pct_home": feat_b["season_win_pct"],
        "sos_away_pre": feat_a["sos_pre"], "sos_home_pre": feat_b["sos_pre"],
        "games_played_season_away": feat_a["games_played_season"], "games_played_season_home": feat_b["games_played_season"],
        "rest_days_away": feat_a["rest_days"], "rest_days_home": feat_b["rest_days"],
        "is_neutral": int(neutral), "is_postseason": postseason, "same_conf": same_conf, "week": week,
    }
    X = pd.DataFrame([row])[FEATURES]
    pred_a = models["score_away"].predict(X)[0]
    pred_b = models["score_home"].predict(X)[0]
    prob_sigma, noise_sigma = game_sigma_for(models, team_a, team_b)
    prob_a = norm_cdf((pred_a - pred_b) / prob_sigma)  # team_a's win probability, correctly calibrated
    disp_a, disp_b, realized_margin = realize_chaotic_score(pred_a, pred_b, noise_sigma, floor=3.0)
    score_a, score_b = round_half_up(disp_a), round_half_up(disp_b)
    score_a, score_b = break_ties_scalar(score_a, score_b)
    winner = team_a if score_a > score_b else team_b  # consistent with the final (post-OT) score
    return winner, score_a, score_b, prob_a

import hashlib

def _deterministic_coinflip(team):
    """A fixed, reproducible 'coin flip' per team name -- used only as the
    very last tiebreaker, when every other criterion is exactly equal."""
    h = hashlib.md5(team.encode()).hexdigest()
    return int(h, 16) % 100_000 / 100_000.0

def compute_conference_standings(games_df):
    """Full conference standings: conference and overall records (each with
    home/away splits), ranked by conference win % (handles teams playing a
    different number of conference games), with tiebreakers applied in
    order: 1) head-to-head among the tied teams, 2) overall record,
    3) overall point differential, 4) points per game, 5) points allowed
    per game, 6) a fixed, reproducible coin flip."""
    df = games_df.dropna(subset=["score_home", "score_away"])
    away = df[["away", "home", "score_away", "score_home", "conf_away", "same_conf"]].rename(
        columns={"away": "team", "home": "opp", "score_away": "team_score", "score_home": "opp_score", "conf_away": "conf"})
    away["is_home"] = False
    home = df[["home", "away", "score_home", "score_away", "conf_home", "same_conf"]].rename(
        columns={"home": "team", "away": "opp", "score_home": "team_score", "score_away": "opp_score", "conf_home": "conf"})
    home["is_home"] = True
    long = pd.concat([away, home], ignore_index=True)
    long["win"] = long["team_score"] > long["opp_score"]

    def record(d):
        return int(d["win"].sum()), int((~d["win"]).sum())

    rows = []
    for team, grp in long.groupby("team"):
        conf = grp["conf"].mode()
        if conf.empty or pd.isna(conf.iloc[0]):
            continue  # independents (no conference championship to speak of)
        conf = conf.iloc[0]
        conf_games = grp[grp["same_conf"] == 1]
        ow, ol = record(grp)
        cw, cl = record(conf_games)
        oh_w, oh_l = record(grp[grp.is_home])
        oa_w, oa_l = record(grp[~grp.is_home])
        ch_w, ch_l = record(conf_games[conf_games.is_home])
        ca_w, ca_l = record(conf_games[~conf_games.is_home])
        n = len(grp)
        division = None
        if conf == "sun-belt":
            if team in SUN_BELT_DIVISIONS["east"]:
                division = "east"
            elif team in SUN_BELT_DIVISIONS["west"]:
                division = "west"
        rows.append({
            "team": team, "conference": conf, "division": division,
            "conf_wins": cw, "conf_losses": cl, "conf_pct": round(cw / (cw + cl), 4) if (cw + cl) else 0.0,
            "conf_home_wins": ch_w, "conf_home_losses": ch_l,
            "conf_away_wins": ca_w, "conf_away_losses": ca_l,
            "overall_wins": ow, "overall_losses": ol, "overall_pct": round(ow / (ow + ol), 4) if (ow + ol) else 0.0,
            "overall_home_wins": oh_w, "overall_home_losses": oh_l,
            "overall_away_wins": oa_w, "overall_away_losses": oa_l,
            "point_diff": int(grp["team_score"].sum() - grp["opp_score"].sum()),
            "ppg": round(grp["team_score"].sum() / n, 2) if n else 0.0,
            "papg": round(grp["opp_score"].sum() / n, 2) if n else 0.0,
        })
    standings = pd.DataFrame(rows)
    if standings.empty:
        return standings

    h2h = {}
    for _, r in long.iterrows():
        key = (r["team"], r["opp"])
        h2h.setdefault(key, [0, 0])
        h2h[key][1] += 1
        h2h[key][0] += int(r["win"])

    def group_h2h_pct(team, group_teams):
        w = g = 0
        for other in group_teams:
            if other == team:
                continue
            if (team, other) in h2h:
                w += h2h[(team, other)][0]
                g += h2h[(team, other)][1]
        return w / g if g > 0 else 0.5  # no head-to-head data -> neutral, falls through to next tiebreaker

    out = []
    for conf, grp in standings.groupby("conference"):
        grp = grp.copy()
        grp["h2h_pct"] = [
            group_h2h_pct(row["team"], grp.loc[grp["conf_pct"] == row["conf_pct"], "team"].tolist())
            if (grp["conf_pct"] == row["conf_pct"]).sum() > 1 else 0.5
            for _, row in grp.iterrows()
        ]
        grp["coinflip"] = grp["team"].map(_deterministic_coinflip)
        grp = grp.sort_values(
            ["conf_pct", "h2h_pct", "overall_pct", "point_diff", "ppg", "papg", "coinflip"],
            ascending=[False, False, False, False, False, True, False]
        ).reset_index(drop=True)
        grp["conf_rank"] = range(1, len(grp) + 1)
        out.append(grp)
    return pd.concat(out, ignore_index=True)

# Sun Belt is split into divisions (unlike most other conferences, which
# have moved to a top-2-overall format); the championship game is the
# winner of each division, not simply the top 2 teams in the conference.
SUN_BELT_DIVISIONS = {
    "east": {"Georgia Southern", "Georgia State", "Coastal Carolina", "Marshall",
             "James Madison", "Old Dominion", "Appalachian State"},
    "west": {"UL-Lafayette", "Louisiana Tech", "UL-Monroe", "Troy",
             "Southern Miss", "Arkansas State", "South Alabama"},
}

def project_conference_championships(games, season_end_elo, models, season):
    standings = compute_conference_standings(games[games.season == season])
    rows = []
    for conf, grp in standings.groupby("conference"):
        if conf == "ind":
            continue  # independents don't have a conference championship
        grp = grp.sort_values("conf_rank")
        if conf == "sun-belt":
            east = grp[grp["team"].isin(SUN_BELT_DIVISIONS["east"])].sort_values("conf_rank")
            west = grp[grp["team"].isin(SUN_BELT_DIVISIONS["west"])].sort_values("conf_rank")
            if east.empty or west.empty:
                continue
            t1, t2 = east.iloc[0]["team"], west.iloc[0]["team"]
        else:
            if len(grp) < 2:
                continue
            t1, t2 = grp.iloc[0]["team"], grp.iloc[1]["team"]
        f1, f2 = latest_team_row(games, season, t1), latest_team_row(games, season, t2)
        winner, s1, s2, prob1 = predict_matchup(models, t1, t2, f1, f2, neutral=True, postseason=1, same_conf=1)
        rows.append({"conference": conf, "team1": t1, "team2": t2,
                     "pred_score_team1": s1, "pred_score_team2": s2,
                     "win_prob_team1": round(prob1, 3), "win_prob_team2": round(1 - prob1, 3),
                     "predicted_champion": winner})
    return pd.DataFrame(rows).sort_values("conference")

def project_playoff(games, season_end_elo, models, season, champ_df):
    elo = season_end_elo[season]
    champs = list(champ_df["predicted_champion"])
    ranked_all = sorted(elo.items(), key=lambda kv: -kv[1])
    rank_lookup = {t: i + 1 for i, (t, _) in enumerate(ranked_all)}

    champs_ranked = sorted(champs, key=lambda t: rank_lookup.get(t, 999))
    auto_bids = champs_ranked[:5]
    at_large = [t for t, _ in ranked_all if t not in champs][:12 - len(auto_bids)]
    field = sorted(auto_bids + at_large, key=lambda t: rank_lookup.get(t, 999))[:12]

    # Notre Dame rule: as an independent, they can never win a conference
    # championship, but if they're ranked 12th or better they get a
    # guaranteed at-large bid regardless -- bumping the LOWEST-ranked
    # current at-large team (never one of the 5 conference-champion auto
    # bids, those are protected). No-op if Notre Dame is already in the
    # field on merit, or if they're ranked worse than 12th.
    nd_rank = rank_lookup.get("Notre Dame", 999)
    if nd_rank <= 12 and "Notre Dame" not in field:
        at_large_in_field = [t for t in field if t not in auto_bids]
        if at_large_in_field:
            bumped = max(at_large_in_field, key=lambda t: rank_lookup.get(t, 999))
            field = [t for t in field if t != bumped] + ["Notre Dame"]
            field = sorted(field, key=lambda t: rank_lookup.get(t, 999))

    seeds = {i + 1: t for i, t in enumerate(field)}

    def play(team_a, team_b):
        f1, f2 = latest_team_row(games, season, team_a), latest_team_row(games, season, team_b)
        return predict_matchup(models, team_a, team_b, f1, f2, neutral=True, postseason=1, same_conf=0, week=17)

    def fmt(team_a, s_a, team_b, s_b, prob_a, winner):
        prob_winner = prob_a if winner == team_a else 1 - prob_a
        return f"{team_a} {s_a} - {team_b} {s_b}  ({winner} {prob_winner:.0%} win prob.)"

    log = []
    struct = []  # JSON-friendly: seed numbers + explicit team_a/team_b/scores/winner
    r1_winners = {}
    for hi, lo in [(5, 12), (6, 11), (7, 10), (8, 9)]:
        w, s1, s2, p = play(seeds[lo], seeds[hi])
        log.append(("Round 1", f"({hi}) {seeds[hi]} vs ({lo}) {seeds[lo]}",
                     fmt(seeds[lo], s1, seeds[hi], s2, p, w), w))
        prob_winner = p if w == seeds[lo] else 1 - p
        struct.append({"round": "Round 1", "seed_a": lo, "team_a": seeds[lo], "score_a": s1,
                        "seed_b": hi, "team_b": seeds[hi], "score_b": s2,
                        "winner": w, "win_prob_winner": round(prob_winner, 3)})
        r1_winners[hi] = w

    qf_winners = []
    for seed, opp in [(1, r1_winners[8]), (2, r1_winners[7]), (3, r1_winners[6]), (4, r1_winners[5])]:
        w, s1, s2, p = play(opp, seeds[seed])
        log.append(("Quarterfinal", f"({seed}) {seeds[seed]} vs {opp}",
                     fmt(opp, s1, seeds[seed], s2, p, w), w))
        prob_winner = p if w == opp else 1 - p
        struct.append({"round": "Quarterfinal", "seed_a": None, "team_a": opp, "score_a": s1,
                        "seed_b": seed, "team_b": seeds[seed], "score_b": s2,
                        "winner": w, "win_prob_winner": round(prob_winner, 3)})
        qf_winners.append(w)

    sf1, s1a, s1b, p1 = play(qf_winners[0], qf_winners[3])
    log.append(("Semifinal", f"{qf_winners[0]} vs {qf_winners[3]}",
                fmt(qf_winners[0], s1a, qf_winners[3], s1b, p1, sf1), sf1))
    prob_winner1 = p1 if sf1 == qf_winners[0] else 1 - p1
    struct.append({"round": "Semifinal", "seed_a": None, "team_a": qf_winners[0], "score_a": s1a,
                    "seed_b": None, "team_b": qf_winners[3], "score_b": s1b,
                    "winner": sf1, "win_prob_winner": round(prob_winner1, 3)})

    sf2, s2a, s2b, p2 = play(qf_winners[1], qf_winners[2])
    log.append(("Semifinal", f"{qf_winners[1]} vs {qf_winners[2]}",
                fmt(qf_winners[1], s2a, qf_winners[2], s2b, p2, sf2), sf2))
    prob_winner2 = p2 if sf2 == qf_winners[1] else 1 - p2
    struct.append({"round": "Semifinal", "seed_a": None, "team_a": qf_winners[1], "score_a": s2a,
                    "seed_b": None, "team_b": qf_winners[2], "score_b": s2b,
                    "winner": sf2, "win_prob_winner": round(prob_winner2, 3)})

    champ, sfa, sfb, pf = play(sf1, sf2)
    log.append(("National Championship", f"{sf1} vs {sf2}",
                fmt(sf1, sfa, sf2, sfb, pf, champ), champ))
    prob_winnerf = pf if champ == sf1 else 1 - pf
    struct.append({"round": "National Championship", "seed_a": None, "team_a": sf1, "score_a": sfa,
                    "seed_b": None, "team_b": sf2, "score_b": sfb,
                    "winner": champ, "win_prob_winner": round(prob_winnerf, 3)})

    return seeds, pd.DataFrame(log, columns=["round", "matchup", "predicted_score", "predicted_winner"]), champ, struct

# ---------------------------------------------------------------------------
# FUTURE-SEASON HELPER (for when you have a real, unplayed schedule)
# ---------------------------------------------------------------------------

def simulate_future_week(games, season_end_elo, models, season, week_schedule):
    """week_schedule: list of dicts with keys away, home, conf_away, conf_home,
    neutral, week. Predicts each game with the model, since results don't
    exist yet -- returns predictions. To carry state into the next week,
    append these predicted scores onto `games` as if they were real results
    and re-run build_game_features()/compute_elo() before predicting again."""
    elo = season_end_elo[season]
    out = []
    for g in week_schedule:
        fa = latest_team_row(games, season, g["away"])
        fb = latest_team_row(games, season, g["home"])
        winner, sa, sb, prob_away = predict_matchup(models, g["away"], g["home"], fa, fb,
                                          neutral=g.get("neutral", False), postseason=0,
                                          same_conf=int(g["conf_away"] == g["conf_home"]), week=g["week"])
        out.append({**g, "pred_score_away": sa, "pred_score_home": sb,
                     "win_prob_away": round(prob_away, 3), "win_prob_home": round(1 - prob_away, 3),
                     "predicted_winner": winner})
    return pd.DataFrame(out)

SCHEDULE_TEXT_2026 = """
WEEK 0
Saturday, August 29
North Carolina vs TCU (in Dublin, Ireland)
San Jose State at USC
NC State at Virginia
Jacksonville State at North Dakota State
Sacramento State at Eastern Michigan
Hawaii at Stanford
New Mexico State at Florida State
Memphis at UNLV

WEEK 1
Thursday, September 3
UMass at Rutgers
Akron at Wake Forest
Bethune-Cookman at UCF
Merrimack at Delaware
UAlbany at Buffalo
West Georgia at Kennesaw State
UAPB at Missouri
Colorado at Georgia Tech
Eastern Illinois at Minnesota
Idaho at 18 Utah
UAB at Illinois
Friday, September 4
San Jose State at Eastern Michigan
Indiana State at Purdue
North Carolina A&T at Georgia State
LIU at Kansas
Toledo at Michigan State
UTEP at 13 Oklahoma
Fresno State at 20 USC
7 Miami (FL) at Stanford
Saturday, September 5
Liberty at James Madison
New Hampshire at Syracuse
Tarleton State at Bowling Green
Oregon State at Houston
Ohio at Nebraska
North Texas at 6 Indiana
Bryant at Army
Coastal Carolina at West Virginia
East Carolina at 16 Alabama
Lafayette at UConn
Ball State at 1 Ohio State
Miami (Ohio) at Pitt
Kent State at South Carolina
Southeast Missouri at Iowa State
Duquesne at Air Force
Youngstown State at Kentucky
Rhode Island at Temple
Tennessee State at 3 Georgia
Furman at 25 Tennessee
Texas State at 5 Texas
The Citadel at Charlotte
Towson at Navy
Tulane at Duke
Fordham at North Dakota State
Marshall at 15 Penn State
Maine at Appalachian State
UTRGV at UTSA
Boston College at Cincinnati
Boise State at 2 Oregon
Baylor vs Auburn (in Atlanta, GA)
Oklahoma State at Tulsa
North Alabama at Arkansas
NIU at 19 Iowa
Alcorn State at Southern Miss
Norfolk State at Old Dominion
Wyoming at Colorado State
Southeastern La. at South Alabama
Sam Houston at Troy
Nicholls at Kansas State
Murray State at Middle Tennessee
Missouri State at 10 Texas A&M
Mercyhurst at New Mexico State
Eastern Kentucky at Jacksonville State
Idaho State at Utah State
HCU at Rice
FIU at USF
Charleston So. at Georgia Southern
Arkansas State at Memphis
Abilene Christian at 8 Texas Tech
Austin Peay at Vanderbilt
Clemson at 11 LSU
Western Michigan at 14 Michigan
VMI at Virginia Tech
ULM at Mississippi State
Northwestern State at Louisiana Tech
Florida Atlantic at Florida
Hampton at Maryland
Lamar at Louisiana
South Dakota State at Northwestern
Utah Tech at 12 BYU
Northern Arizona at Arizona
Portland State at San Diego State
Morgan State at Arizona State
MVSU at Sacramento State
Central Michigan at New Mexico
UNLV at Hawaii
WKU at Nevada
UCLA at California
Sunday, September 6
Washington State at 17 Washington
21 Louisville vs 9 Ole Miss (in Nashville, TN)
Wisconsin vs 4 Notre Dame (in Green Bay, WI)
Monday, September 7
22 SMU at Florida State

WEEK 2
Thursday, September 10
Florida A&M at 7 Miami (FL)
Friday, September 11
Villanova at 21 Louisville
Norfolk State at Virginia
Richmond at NC State
Rutgers at Boston College
Missouri at Kansas
Saturday, September 12
Wake Forest at Purdue
USF at Army
Wofford at Kent State
Howard at 6 Indiana
Washington State at Kansas State
Appalachian State at East Carolina
Arizona State at 10 Texas A&M
ETSU at North Carolina
13 Oklahoma at 14 Michigan
Old Dominion at Virginia Tech
2 Oregon at Oklahoma State
15 Penn State at Temple
WKU at 3 Georgia
Holy Cross at Miami (Ohio)
Colgate at Central Michigan
UT Martin at West Virginia
Stony Brook at Ball State
Southern Miss at Auburn
Wagner at James Madison
ULM at UAB
CCSU at Toledo
Rice at 4 Notre Dame
Sacred Heart at UMass
Weber State at Colorado
Arizona at 12 BYU
UTSA at Texas State
Robert Morris at Akron
UCF at Pitt
16 Alabama at Kentucky
Mississippi State at Minnesota
Maryland at UConn
Eastern Michigan at Michigan State
Duke at Illinois
California at Syracuse
Utah State at 17 Washington
UNLV at North Texas
Alabama State at Troy
Northern Colorado at Wyoming
UC Davis at 22 SMU
Delaware at Vanderbilt
Campbell at Florida
Jacksonville State at Ohio
Memphis at Boise State
Gardner-Webb at Liberty
Buffalo at FIU
Monmouth at Western Michigan
Tulsa at Sam Houston
Southern at Houston
25 Tennessee at Georgia Tech
Bowling Green at Nebraska
Georgia State at Kennesaw State
Western Carolina at Cincinnati
West Georgia at Arkansas State
Lindenwood at Missouri State
Towson at South Carolina
Southern Utah at Colorado State
South Alabama at Tulane
Middle Tennessee at Marshall
Illinois State at NIU
San Diego State at UCLA
Western Illinois at Wisconsin
Navy at Florida Atlantic
Louisiana Tech at 11 LSU
8 Texas Tech at Oregon State
1 Ohio State at 5 Texas
Iowa State at 19 Iowa
Georgia Southern at Clemson
Fordham at Coastal Carolina
Charlotte at 9 Ole Miss
Prairie View A&M at Baylor
Grambling State at 23 TCU
Texas Southern at UTEP
North Dakota State at Air Force
Arkansas at 18 Utah
Sacramento State at Fresno State
Montana State at Nevada
Louisiana at 20 USC
New Mexico State at Hawaii
Mercyhurst at New Mexico
Cal Poly at San Jose State

WEEK 3
Thursday, September 17
Syracuse at Pitt
Friday, September 18
7 Miami (FL) at Wake Forest
Houston at 8 Texas Tech
Portland State at 2 Oregon
Saturday, September 19
Coastal Carolina at Delaware
Mercer at Georgia Tech
Tulane at Kansas State
North Carolina at Clemson
3 Georgia at Arkansas
Arizona State vs Kansas (in London, England)
Akron at Minnesota
Bowling Green at Iowa State
Buffalo at 15 Penn State
Kent State at 1 Ohio State
North Texas at Texas State
Eastern Michigan at Wisconsin
Troy at Missouri
NC State at Vanderbilt
Wyoming at Central Michigan
Southern Illinois at Illinois
Maine at Boston College
Temple at Toledo
Wagner at California
Duquesne at Washington State
UTEP at 14 Michigan
Utah State at 18 Utah
Miami (Ohio) vs Cincinnati (in Cincinnati, OH)
Florida State at 16 Alabama
22 SMU at 21 Louisville
Stonehill at UMass
20 USC at Rutgers
Kentucky at 10 Texas A&M
Stanford at Duke
Louisiana Tech at Baylor
WKU at 6 Indiana
Northern Iowa at 19 Iowa
Ball State at Liberty
Mississippi State at South Carolina
Southeastern La. at ULM
Charlotte at Appalachian State
FIU at Florida Atlantic
East Carolina at Old Dominion
Marshall at Missouri State
Georgia State at UCF
Western Michigan at Rice
Nevada at Middle Tennessee
UConn at Southern Miss
UT Martin at Memphis
Florida at Auburn
Delaware State at USF
Nicholls at Sam Houston
Ohio at South Alabama
Murray State at Oklahoma State
Georgia Southern at Jacksonville State
Eastern Washington at 17 Washington
North Dakota at Nebraska
Virginia Tech at Maryland
11 LSU at 9 Ole Miss
Virginia vs West Virginia (in Charlotte, NC)
12 BYU at Colorado State
Colorado at Northwestern
Michigan State at 4 Notre Dame
New Mexico at 13 Oklahoma
Kennesaw State at 25 Tennessee
Arkansas State at 23 TCU
UAB at Louisiana
UTSA at 5 Texas
East Texas A&M at Tulsa
James Madison at San Diego State
South Dakota at Boise State
NIU at Arizona
North Dakota State at Sacramento State
Montana at Oregon State
Fresno State at San Jose State
Purdue at UCLA

WEEK 4
Thursday, September 24
Liberty at Coastal Carolina
Friday, September 25
Army at Temple
Navy at UAB
Howard at Rutgers
Northwestern at 6 Indiana
Clemson at California
Saturday, September 26
Vanderbilt at Auburn
9 Ole Miss at Florida
10 Texas A&M at 11 LSU
13 Oklahoma at 3 Georgia
5 Texas at 25 Tennessee
Bucknell at Pitt
South Alabama at Kentucky
Lindenwood at Eastern Michigan
South Carolina at 16 Alabama
Missouri at Mississippi State
Hawaii at Wyoming
New Mexico at New Mexico State
Robert Morris at Buffalo
William & Mary at Duke
Stonehill at Ohio
NC Central at East Carolina
USF at Bowling Green
UIW at Texas State
LIU at FIU
Central Michigan at 7 Miami (FL)
Kennesaw State at Arkansas State
Mercyhurst at WKU
Middle Tennessee at Jacksonville State
HCU at North Texas
Arizona at Washington State
Troy at Utah State
Tulsa at Arkansas
Oregon State at UTEP
Rice at Fresno State
Georgia Tech at Stanford
UMass at Sacramento State
Kansas State at Cincinnati
4 Notre Dame at Purdue
San Diego State at Toledo
Southern Miss at Tulane
UNLV at Akron
Sam Houston at 8 Texas Tech
Boise State at Western Michigan
UConn at Miami (Ohio)
Florida Atlantic at ULM
Central Arkansas at Florida State
Louisiana at Charlotte
NIU at Georgia State
Gardner-Webb at Marshall
Colorado at Baylor
Oklahoma State at West Virginia
Houston at Georgia Southern
23 TCU at UCF
18 Utah at Iowa State
Virginia Tech at Boston College
Wake Forest at 21 Louisville
2 Oregon at 20 USC
Illinois at 1 Ohio State
19 Iowa at 14 Michigan
Minnesota at 17 Washington
Wisconsin at 15 Penn State
UCLA at Maryland
Nebraska at Michigan State
Air Force at Nevada
James Madison at Old Dominion
Ball State at Kent State
Missouri State at 22 SMU
Delaware at Virginia
Colorado State at UTSA
Appalachian State at NC State

WEEK 5
Thursday, October 1
WKU at New Mexico State
North Texas at Tulsa
Friday, October 2
Pitt at Virginia Tech
Liberty at Delaware
15 Penn State at Northwestern
Saturday, October 3
Kentucky at South Carolina
Florida at Missouri
Auburn at 25 Tennessee
Arkansas at 10 Texas A&M
Navy at Air Force
Syracuse at UConn
16 Alabama at Mississippi State
Vanderbilt at 3 Georgia
Wyoming at North Dakota State
Samford at UAB
California at UNLV
Oregon State at Colorado State
Texas Southern at Florida Atlantic
Utah State at Boise State
McNeese at 11 LSU
6 Indiana at Rutgers
Fresno State at Washington State
Texas State at San Diego State
San Jose State at Hawaii
Western Michigan at Buffalo
Temple at USF
Stanford at Wake Forest
Virginia at Florida State
7 Miami (FL) at Clemson
1 Ohio State at 19 Iowa
14 Michigan at Minnesota
17 Washington at 20 USC
Purdue at Illinois
Maryland at Nebraska
Michigan State at Wisconsin
Memphis at Charlotte
UTSA at Rice
Boston College at 22 SMU
UTEP at New Mexico
Georgia Southern at Coastal Carolina
Old Dominion at Georgia State
Marshall at James Madison
Arkansas State at Louisiana
ULM at South Alabama
Akron at Central Michigan
Bowling Green at Miami (Ohio)
Eastern Michigan at UMass
Ohio at Kent State
Toledo at Ball State
21 Louisville at NC State
Middle Tennessee at Kansas
West Virginia at Iowa State
UCF at Houston
8 Texas Tech at Colorado
Cincinnati at Arizona
12 BYU at 23 TCU
Baylor at Arizona State
Army at Louisiana Tech
4 Notre Dame at North Carolina

WEEK 6
Tuesday, October 6
Southern Miss at Troy
Wednesday, October 7
Jacksonville State at Kennesaw State
New Mexico State at FIU
Thursday, October 8
Missouri State at WKU
Sam Houston at Liberty
USF at UTSA
South Alabama at Arkansas State
Friday, October 9
Florida State at 21 Louisville
19 Iowa at 17 Washington
Wyoming at San Jose State
Washington State at Utah State
Iowa State at 12 BYU
Saturday, October 10
9 Ole Miss at Vanderbilt
25 Tennessee at Arkansas
Tulane at Army
South Carolina at Florida
10 Texas A&M at Missouri
11 LSU at Kentucky
3 Georgia at 16 Alabama
Tulsa at Navy
Stanford at 4 Notre Dame
5 Texas vs 13 Oklahoma (in Dallas, TX)
San Diego State at Oregon State
North Dakota State at UNLV
Air Force at NIU
Boise State at Fresno State
Sacramento State at Bowling Green
UAB at Memphis
6 Indiana at Nebraska
UCLA at 2 Oregon
Maryland at 1 Ohio State
20 USC at 15 Penn State
Illinois at Michigan State
Minnesota at Purdue
Charlotte at North Texas
Rice at East Carolina
Nevada at UTEP
Virginia Tech at California
Old Dominion at Appalachian State
James Madison at Georgia Southern
Louisiana at Louisiana Tech
Coastal Carolina at Marshall
Buffalo at Toledo
Central Michigan at Ohio
Eastern Michigan at Akron
Kent State at Western Michigan
Miami (Ohio) at UMass
Wake Forest at NC State
Syracuse at Virginia
Duke at Georgia Tech
UCF at Oklahoma State
Kansas at 18 Utah
Houston at Kansas State
Hawaii at Arizona State
Arizona at West Virginia
UConn at Temple
Ball State at Northwestern
North Carolina at Pitt

WEEK 7
Tuesday, October 13
Delaware at Middle Tennessee
FIU at Jacksonville State
Wednesday, October 14
Kennesaw State at Missouri State
WKU at Sam Houston
Thursday, October 15
East Carolina at UAB
Georgia Southern at Old Dominion
Colorado State at Texas State
Friday, October 16
Memphis at Tulane
Appalachian State at Coastal Carolina
17 Washington at Purdue
Saturday, October 17
Auburn at 3 Georgia
Missouri at 9 Ole Miss
Kentucky at 13 Oklahoma
16 Alabama at 25 Tennessee
Arkansas at Vanderbilt
Florida Atlantic at Army
The Citadel at 10 Texas A&M
Mississippi State at 11 LSU
Florida at 5 Texas
NIU at Wyoming
UNLV at Air Force
Washington State at Oregon State
Nevada at North Dakota State
Elon at Stanford
San Jose State at UTEP
Fresno State at San Diego State
Western Michigan at Central Michigan
Navy at UTSA
Virginia at 22 SMU
Wake Forest at California
Florida State at 7 Miami (FL)
Nebraska at 2 Oregon
15 Penn State at 14 Michigan
Northwestern at Michigan State
Wisconsin at UCLA
Rutgers at Maryland
Kent State at USF
Charlotte at Temple
Tulsa at Rice
North Carolina at Duke
New Mexico at Hawaii
Georgia State at James Madison
Troy at Louisiana
Louisiana Tech at ULM
Arkansas State at Southern Miss
Akron at Miami (Ohio)
Ball State at Bowling Green
Ohio at Sacramento State
Toledo at Eastern Michigan
UMass at Buffalo
Pitt at Boston College
21 Louisville at Syracuse
18 Utah at Colorado
23 TCU at Baylor
Oklahoma State at Houston
4 Notre Dame at 12 BYU
Kansas at Kansas State
Cincinnati at West Virginia
Arizona State at 8 Texas Tech
1 Ohio State at 6 Indiana
Charleston So. at Clemson
Georgia Tech at Virginia Tech

WEEK 8
Tuesday, October 20
Middle Tennessee at FIU
Missouri State at Delaware
South Alabama at Marshall
Wednesday, October 21
Liberty at Kennesaw State
New Mexico State at Sam Houston
Thursday, October 22
James Madison at Appalachian State
East Carolina at Memphis
Friday, October 23
UMass at UConn
Duke at Virginia
Army at Tulsa
Air Force at Wyoming
NC State at Stanford
Saturday, October 24
25 Tennessee at South Carolina
19 Iowa at Minnesota
10 Texas A&M at 16 Alabama
9 Ole Miss at 5 Texas
13 Oklahoma at Mississippi State
11 LSU at Auburn
Vanderbilt at Kentucky
Hawaii at NIU
North Texas at Navy
San Jose State at Nevada
Boise State at Washington State
San Diego State at Colorado State
Utah State at Texas State
North Dakota State at New Mexico
Western Michigan at Toledo
UTSA at Tulane
Virginia Tech at Clemson
Pitt at 7 Miami (FL)
6 Indiana at 14 Michigan
2 Oregon at Illinois
20 USC at Wisconsin
Rutgers at Northwestern
Michigan State at UCLA
Rice at Florida Atlantic
Georgia State at Arkansas State
California at 22 SMU
Old Dominion at Louisiana Tech
Louisiana at Southern Miss
ULM at Troy
Akron at Kent State
Bowling Green at Buffalo
Eastern Michigan at Ohio
Miami (Ohio) at Central Michigan
Sacramento State at Ball State
Syracuse at North Carolina
Boston College at Georgia Tech
West Virginia at 23 TCU
8 Texas Tech at Cincinnati
Kansas State at Arizona State
Iowa State at Arizona
Houston at 18 Utah
Colorado at Oklahoma State
12 BYU at UCF
Baylor at Kansas

WEEK 9
Tuesday, October 27
Delaware at WKU
Sam Houston at Missouri State
Wednesday, October 28
Kennesaw State at Middle Tennessee
Jacksonville State at New Mexico State
Thursday, October 29
Florida Atlantic at North Texas
Troy at James Madison
Friday, October 30
Tulane at Charlotte
Baylor at UCF
Kent State at Sacramento State
Saturday, October 31
South Carolina at 13 Oklahoma
Missouri at Arkansas
4 Notre Dame vs Navy (in Foxborough, MA)
Bowling Green at Western Michigan
Auburn at 9 Ole Miss
Mississippi State at 5 Texas
UConn at Air Force
Colorado State at Utah State
Florida vs 3 Georgia (in Atlanta, GA)
UTEP at North Dakota State
FIU at Liberty
Oregon State at Fresno State
Washington State at San Diego State
Texas State at Boise State
NIU at UNLV
Louisiana Tech at South Alabama
17 Washington at Nebraska
Stanford at 21 Louisville
Virginia at Wake Forest
7 Miami (FL) at North Carolina
Minnesota at 6 Indiana
Northwestern at 2 Oregon
1 Ohio State at 20 USC
14 Michigan at Rutgers
Wisconsin at 19 Iowa
Illinois at Maryland
Purdue at 15 Penn State
Georgia Tech at Pitt
Nevada at UCLA
Army at Memphis
Temple at East Carolina
UAB at USF
New Mexico at San Jose State
Appalachian State at Georgia Southern
Coastal Carolina at Georgia State
Southern Miss at ULM
Marshall at Old Dominion
22 SMU at Syracuse
Clemson at Florida State
Boston College at Duke
18 Utah at Cincinnati
Oklahoma State at Iowa State
Kansas State at Colorado
Kansas at 23 TCU
Arizona State at 12 BYU
Arizona at 8 Texas Tech
California at NC State

WEEK 10
Tuesday, November 3
Buffalo at Miami (Ohio)
Ohio at Akron
Wednesday, November 4
Ball State at UMass
Central Michigan vs Eastern Michigan (in Detroit, MI)
Toledo at Sacramento State
Thursday, November 5
UTSA at Florida Atlantic
James Madison at Southern Miss
Friday, November 6
Virginia Tech at 22 SMU
USF at East Carolina
Nebraska at Illinois
New Mexico at Nevada
23 TCU at Arizona
Saturday, November 7
10 Texas A&M at South Carolina
13 Oklahoma at Florida
16 Alabama at 11 LSU
3 Georgia at 9 Ole Miss
Vanderbilt at Mississippi State
North Carolina at UConn
5 Texas at Missouri
Arkansas at Auburn
Kentucky at 25 Tennessee
Delaware at Kennesaw State
WKU at Middle Tennessee
Temple at Navy
Missouri State at FIU
Liberty at New Mexico State
Boise State at Colorado State
Sam Houston at Jacksonville State
Air Force at Army
7 Miami (FL) at 4 Notre Dame
Fresno State at Utah State
Wyoming at UNLV
Texas State at Oregon State
Louisiana Tech at Troy
Rutgers at Wisconsin
West Virginia at 8 Texas Tech
Clemson at Syracuse
Duke at NC State
Florida State at Boston College
21 Louisville at Georgia Tech
2 Oregon at 1 Ohio State
Michigan State at 14 Michigan
19 Iowa at Northwestern
UCLA at Minnesota
15 Penn State at 17 Washington
Maryland at Purdue
Oklahoma State at Kansas State
Charlotte at UAB
Rice at North Texas
Tulsa at Tulane
Hawaii at UTEP
NIU at San Jose State
Georgia State at Appalachian State
ULM at Arkansas State
Old Dominion at Coastal Carolina
Marshall at Georgia Southern
South Alabama at Louisiana
UCF at Kansas
Iowa State at Baylor
Cincinnati at Houston
12 BYU at 18 Utah
Merrimack at Wake Forest
Colorado at Arizona State

WEEK 11
Tuesday, November 10
Kent State at Bowling Green
Ohio at Miami (Ohio)
Western Michigan at Akron
Wednesday, November 11
Buffalo at Ball State
Sacramento State at Central Michigan
UMass at Toledo
Thursday, November 12
Memphis at USF
Louisiana at ULM
Friday, November 13
Florida State at Pitt
Illinois at UCLA
Houston at Colorado
Saturday, November 14
9 Ole Miss at 13 Oklahoma
5 Texas at 11 LSU
Missouri at 3 Georgia
South Carolina at Arkansas
25 Tennessee at 10 Texas A&M
James Madison at UConn
16 Alabama at Vanderbilt
Auburn at Mississippi State
Florida at Kentucky
Kennesaw State at Sam Houston
New Mexico State at Missouri State
FIU at Delaware
Nevada at NIU
Boston College at 4 Notre Dame
Middle Tennessee at Liberty
Jacksonville State at WKU
Fresno State at Texas State
Oregon State at Boise State
San Jose State at Air Force
UNLV at New Mexico
Utah State at San Diego State
Colorado State at Washington State
North Dakota State at Hawaii
Troy at South Alabama
17 Washington at Michigan State
Georgia Tech at Clemson
21 Louisville at North Carolina
Stanford at Virginia Tech
Syracuse at NC State
Wake Forest at 22 SMU
Duke at 7 Miami (FL)
20 USC at 6 Indiana
14 Michigan at 2 Oregon
Northwestern at 1 Ohio State
Purdue at 19 Iowa
Minnesota at 15 Penn State
Nebraska at Rutgers
18 Utah at Arizona
Wisconsin at Maryland
East Carolina at Charlotte
Florida Atlantic at Tulsa
North Texas at UTSA
Tulane at Rice
UAB at Temple
Wyoming at UTEP
Arkansas State at Coastal Carolina
Georgia Southern at Georgia State
Southern Miss at Louisiana Tech
Appalachian State at Marshall
California at Virginia
8 Texas Tech at Oklahoma State
Kansas at West Virginia
Cincinnati at Iowa State
Baylor at 12 BYU
Arizona State at UCF
Kansas State at 23 TCU

WEEK 12
Tuesday, November 17
Ball State at Ohio
Eastern Michigan at Western Michigan
Miami (Ohio) at Kent State
Wednesday, November 18
Akron at UMass
Central Michigan at Buffalo
Thursday, November 19
Rice at Temple
Friday, November 20
Iowa State at UCF
Clemson at Duke
Bowling Green at Toledo
2 Oregon at Michigan State
UTEP at Air Force
New Mexico at Wyoming
Saturday, November 21
Arkansas at 5 Texas
11 LSU at 25 Tennessee
3 Georgia at South Carolina
Wofford at 9 Ole Miss
East Carolina at Army
Memphis at Navy
10 Texas A&M at 13 Oklahoma
Vanderbilt at Florida
WKU at Liberty
Tennessee Tech at Mississippi State
Missouri State at Jacksonville State
Kentucky at Missouri
Middle Tennessee at Sam Houston
Chattanooga at 16 Alabama
FIU at Kennesaw State
Delaware at New Mexico State
Samford at Auburn
Old Dominion at UConn
NIU at North Dakota State
Washington State at Texas State
UNLV at San Jose State
22 SMU at 4 Notre Dame
Colorado State at Fresno State
San Diego State at Boise State
Utah State at Oregon State
Arkansas State at Louisiana Tech
Northwestern at Minnesota
NC State at Florida State
North Carolina at Virginia
Pitt at 21 Louisville
Stanford at California
Syracuse at Boston College
Wake Forest at Georgia Tech
6 Indiana at 17 Washington
1 Ohio State at Nebraska
UCLA at 14 Michigan
Maryland at 20 USC
19 Iowa at Illinois
Rutgers at 15 Penn State
8 Texas Tech at Baylor
Wisconsin at Purdue
Charlotte at Tulsa
North Texas at Tulane
USF at Florida Atlantic
UTSA at UAB
Hawaii at Nevada
ULM at Appalachian State
Coastal Carolina at Louisiana
Georgia State at Marshall
South Alabama at Southern Miss
Georgia Southern at Troy
Virginia Tech at 7 Miami (FL)
18 Utah at 23 TCU
Oklahoma State at Arizona State
Houston at West Virginia
Colorado at Cincinnati
12 BYU at Kansas
Arizona at Kansas State

WEEK 13
Tuesday, November 24
Kent State at Eastern Michigan
Miami (Ohio) at Western Michigan
Thursday, November 26
23 TCU at 8 Texas Tech
Friday, November 27
Mississippi State at 9 Ole Miss
Nebraska at 19 Iowa
Buffalo at Akron
Toledo at Ohio
Appalachian State at South Alabama
Florida at Florida State
North Dakota State at San Jose State
Air Force at New Mexico
5 Texas at 10 Texas A&M
Minnesota at Wisconsin
West Virginia at 18 Utah
Saturday, November 28
13 Oklahoma at Missouri
Georgia Tech at 3 Georgia
Auburn at 16 Alabama
Central Michigan at Ball State
14 Michigan at 1 Ohio State
UMass at Bowling Green
Kennesaw State at WKU
11 LSU at Arkansas
21 Louisville at Kentucky
Sam Houston at FIU
New Mexico State at Middle Tennessee
Liberty at Missouri State
25 Tennessee at Vanderbilt
Jacksonville State at Delaware
UTEP at NIU
Nevada at UNLV
Sacramento State at Hawaii
Purdue at 6 Indiana
Army at Rice
22 SMU at Stanford
Boston College at 7 Miami (FL)
17 Washington at 2 Oregon
20 USC at UCLA
Illinois at Northwestern
15 Penn State at Maryland
Michigan State at Rutgers
Boise State at Utah State
Oregon State at Washington State
San Diego State at Fresno State
Texas State at Colorado State
Florida Atlantic at East Carolina
NC State at North Carolina
Navy at Charlotte
Temple at Memphis
Tulane at USF
Tulsa at UTSA
UAB at North Texas
Troy at Arkansas State
Louisiana Tech at Georgia Southern
Louisiana at Georgia State
Coastal Carolina at James Madison
Marshall at ULM
Southern Miss at Old Dominion
Pitt at California
Duke at Wake Forest
UCF at Colorado
Kansas State at Iowa State
Kansas at Oklahoma State
Cincinnati at 12 BYU
Baylor at Houston
Arizona State at Arizona
Virginia at Virginia Tech
UConn at Wyoming
South Carolina at Clemson
4 Notre Dame at Syracuse

WEEK 15
Saturday, December 12
Navy vs Army (in East Rutherford, NJ)

"""
# ---------------------------------------------------------------------------
# SCHEDULE PARSER -- turns the plain-text weekly schedule (the format you get
# pasting a "fbs Schedule - Week N" page) into structured game rows.
# ---------------------------------------------------------------------------
import re
import io

_DAYS = "Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday"
_MONTHS = {"August": 8, "September": 9, "October": 10, "November": 11, "December": 12, "January": 1}
_game_re = re.compile(r"^(?:(\d{1,2})\s+)?(.+?)\s+(at|vs)\s+(?:(\d{1,2})\s+)?(.+?)(?:\s*\(in\s+([^)]+)\))?$")
_week_re = re.compile(r"^WEEK\s+(\d+)$")
_date_re = re.compile(rf"^({_DAYS}),\s+([A-Za-z]+)\s+(\d{{1,2}})$")

# Team-name fixes so the parsed schedule matches the training data's naming.
NAME_FIX = {
    "Pitt": "Pittsburgh", "UConn": "Connecticut", "UMass": "Massachusetts",
    "WKU": "Western Kentucky", "NIU": "Northern Illinois", "ULM": "UL-Monroe",
    "Louisiana": "UL-Lafayette", "Sam Houston": "Sam Houston State",
    "Miami (Ohio)": "Miami (OH)", "UT Martin": "Tennessee-Martin",
    "Southeastern La.": "Southeastern Louisiana", "Charleston So.": "Charleston Southern",
    "The Citadel": "Citadel", "UC Davis": "UC-Davis", "HCU": "Houston Christian",
    "Nicholls": "Nicholls State", "Elon": "Elon University",
}

def parse_schedule_text(text, season=2026):
    """Parse the plain-text weekly schedule format into a DataFrame with
    columns: season, week, date, away, home, rank_away, rank_home, neutral."""
    rows = []
    week, cur_date = None, None
    for raw in io.StringIO(text):
        line = raw.strip()
        if not line:
            continue
        m = _week_re.match(line)
        if m:
            week = int(m.group(1))
            continue
        m = _date_re.match(line)
        if m:
            month, day = _MONTHS[m.group(2)], int(m.group(3))
            yr = season + 1 if month == 1 else season  # bowl-season Jan games roll to next year
            cur_date = f"{yr}-{month:02d}-{day:02d}"
            continue
        m = _game_re.match(line)
        if m and week is not None:
            rank_a, team_a, sep, rank_b, team_b, site = m.groups()
            rows.append({
                "season": season, "week": week, "date": cur_date,
                "away": NAME_FIX.get(team_a.strip(), team_a.strip()),
                "home": NAME_FIX.get(team_b.strip(), team_b.strip()),
                "rank_away": int(rank_a) if rank_a else None,
                "rank_home": int(rank_b) if rank_b else None,
                "neutral": sep == "vs",
            })
    return pd.DataFrame(rows)

RECRUITING_TEXT_2026 = """
1
1
USC
USC
35 Commits
91.97

    3
    19
    13

310.67
2
3
Alabama
Alabama
27 Commits
91.94

    4
    11
    12

303.79
3
2
Oregon
Oregon
24 Commits
92.86

    4
    14
    6

303.22
4
5
Ohio State
Ohio State
29 Commits
91.90

    3
    15
    11

301.23
5
6
Notre Dame
Notre Dame
30 Commits
91.91

    4
    20
    6

299.28
6
8
Texas
Texas
26 Commits
91.27

    4
    13
    9

298.37
7
9
Georgia
Georgia
32 Commits
91.32

    1
    22
    7

291.98
8
4
Tennessee
Tennessee
32 Commits
90.66

    2
    14
    15

290.71
9
7
Miami
Miami
31 Commits
91.03

    1
    21
    9

284.61
10
10
Texas A&M
Texas A&M
26 Commits
91.94

    1
    21
    4

283.13
11
13
LSU
LSU
19 Commits
91.67

    2
    10
    7

272.69
12
11
Michigan
Michigan
24 Commits
90.35

    2
    11
    11

270.12
13
12
Washington
Washington
26 Commits
89.56

    1
    10
    15

262.32
14
14
South Carolina
South Carolina
19 Commits
90.27

    0
    10
    9

256.43
15
16
Oklahoma
Oklahoma
25 Commits
89.43

    0
    11
    14

252.55
16
19
Florida State
Florida State
34 Commits
88.68

    0
    12
    22

251.54
17
20
Florida
Florida
19 Commits
90.79

    0
    13
    6

250.98
18
15
Texas Tech
Texas Tech
22 Commits
89.34

    1
    8
    13

250.26
19
17
North Carolina
North Carolina
41 Commits
88.23

    0
    12
    29

247.70
20
21
Clemson
Clemson
23 Commits
89.00

    0
    10
    12

243.28
21
18
BYU
BYU
22 Commits
88.98

    0
    7
    14

237.33
22
22
Ole Miss
Ole Miss
22 Commits
88.80

    0
    8
    14

236.72
23
24
Mississippi State
Mississippi State
34 Commits
87.10

    0
    4
    30

230.56
24
26
Illinois
Illinois
38 Commits
87.63

    0
    5
    30

230.18
25
25
West Virginia
West Virginia
49 Commits
86.96

    0
    5
    44

229.51
26
27
Iowa
Iowa
19 Commits
89.04

    0
    8
    11

228.57
27
32
Utah
Utah
20 Commits
88.49

    1
    2
    17

226.89
28
28
Minnesota
Minnesota
31 Commits
87.66

    0
    6
    25

226.67
29
38
Indiana
Indiana
22 Commits
88.42

    0
    7
    15

226.46
30
34
Virginia Tech
Virginia Tech
22 Commits
88.37

    0
    8
    14

226.24
31
39
Auburn
Auburn
21 Commits
87.81

    0
    6
    15

225.99
32
35
SMU
SMU
23 Commits
88.26

    0
    7
    16

225.82
33
29
Vanderbilt
Vanderbilt
22 Commits
88.47

    1
    4
    16

225.43
34
23
Missouri
Missouri
23 Commits
88.24

    0
    5
    18

225.09
35
30
Syracuse
Syracuse
27 Commits
87.43

    0
    4
    23

224.78
36
31
Rutgers
Rutgers
25 Commits
87.86

    0
    5
    20

222.03
37
45
Stanford
Stanford
23 Commits
87.68

    0
    4
    19

218.95
38
33
Houston
Houston
18 Commits
88.50

    1
    3
    14

218.85
39
42
Wake Forest
Wake Forest
30 Commits
87.07

    0
    3
    27

217.86
40
41
Arizona State
Arizona State
22 Commits
87.68

    0
    5
    16

217.26
41
37
Georgia Tech
Georgia Tech
24 Commits
87.48

    0
    3
    21

216.70
42
43
Arizona
Arizona
20 Commits
87.60

    0
    6
    14

216.67
43
40
Maryland
Maryland
20 Commits
87.94

    1
    1
    17

215.87
44
36
Pittsburgh
Pittsburgh
21 Commits
87.67

    0
    6
    15

214.09
45
51
Michigan State
Michigan State
21 Commits
87.23

    0
    5
    16

213.73
46
46
Arkansas
Arkansas
25 Commits
86.71

    0
    3
    21

212.68
47
44
TCU
TCU
22 Commits
87.53

    0
    4
    17

212.60
48
48
Louisville
Louisville
21 Commits
87.46

    0
    3
    18

211.65
49
47
NC State
NC State
28 Commits
86.91

    0
    2
    26

210.28
50
52
Kansas State
Kansas State
24 Commits
86.85

    0
    0
    24

207.83
51
49
Boise State
Boise State
34 Commits
85.57

    0
    1
    33

205.71
52
53
Northwestern
Northwestern
22 Commits
86.74

    0
    2
    20

204.95
53
54
Boston College
Boston College
24 Commits
86.60

    0
    1
    23

204.06
54
56
Iowa State
Iowa State
27 Commits
86.15

    0
    1
    26

203.31
55
57
Purdue
Purdue
25 Commits
86.44

    0
    0
    25

203.29
56
50
California
California
18 Commits
87.43

    0
    2
    16

202.81
57
55
Baylor
Baylor
17 Commits
87.47

    0
    4
    13

200.60
58
58
Cincinnati
Cincinnati
25 Commits
86.43

    0
    1
    23

199.89
59
62
Appalachian State
Appalachian State
27 Commits
85.75

    0
    1
    26

198.58
60
74
Kentucky
Kentucky
16 Commits
87.97

    0
    3
    12

197.21
61
60
Kansas
Kansas
19 Commits
87.02

    0
    1
    17

196.96
62
63
UCLA
UCLA
20 Commits
86.21

    0
    1
    19

196.57
63
61
UNLV
UNLV
28 Commits
85.41

    0
    0
    28

196.45
64
59
Tulane
Tulane
20 Commits
86.25

    0
    2
    18

195.14
65
65
Penn State
Penn State
15 Commits
87.75

    0
    4
    11

194.39
66
75
UCF
UCF
16 Commits
87.35

    0
    2
    14

194.30
67
66
Colorado
Colorado
19 Commits
86.78

    0
    2
    15

193.36
68
64
Temple
Temple
34 Commits
85.23

    0
    0
    34

192.17
69
81
Oklahoma State
Oklahoma State
20 Commits
85.92

    0
    0
    20

190.27
70
80
Fresno State
Fresno State
24 Commits
85.04

    0
    0
    23

189.36
71
68
Duke
Duke
15 Commits
87.39

    0
    2
    13

189.26
72
77
Memphis
Memphis
22 Commits
85.76

    0
    0
    21

189.20
73
72
Colorado State
Colorado State
29 Commits
85.14

    0
    0
    28

188.82
74
71
Texas State
Texas State
22 Commits
85.26

    0
    0
    22

188.31
75
70
Bowling Green
Bowling Green
27 Commits
84.97

    0
    0
    27

188.03
76
73
Troy
Troy
36 Commits
84.64

    0
    0
    35

187.25
77
76
South Alabama
South Alabama
35 Commits
84.65

    0
    0
    33

187.23
78
67
Wisconsin
Wisconsin
15 Commits
87.26

    0
    1
    14

186.75
79
78
UTSA
UTSA
20 Commits
85.21

    0
    0
    19

185.81
80
85
James Madison
James Madison
21 Commits
85.46

    0
    0
    21

185.60
81
88
Western Kentucky
Western Kentucky
24 Commits
84.99

    0
    0
    24

185.21
82
86
Northern Illinois
Northern Illinois
32 Commits
84.37

    0
    0
    32

184.90
83
83
San Diego State
San Diego State
24 Commits
85.08

    0
    0
    23

184.15
84
82
Georgia State
Georgia State
23 Commits
84.56

    0
    1
    22

184.01
85
69
Arkansas State
Arkansas State
28 Commits
84.67

    0
    0
    28

183.85
86
84
Jacksonville State
Jacksonville State
30 Commits
84.55

    0
    0
    30

183.36
87
91
New Mexico
New Mexico
29 Commits
84.56

    0
    0
    28

182.61
88
79
Nebraska
Nebraska
12 Commits
88.41

    0
    3
    9

182.28
89
100
Georgia Southern
Georgia Southern
22 Commits
84.81

    0
    1
    21

181.78
90
90
South Florida
South Florida
18 Commits
85.56

    0
    0
    18

180.12
91
87
Louisiana
Louisiana
23 Commits
84.48

    0
    0
    23

179.80
92
92
UTEP
UTEP
29 Commits
84.22

    0
    0
    28

179.56
93
89
Tulsa
Tulsa
20 Commits
85.16

    0
    0
    19

179.39
94
94
East Carolina
East Carolina
18 Commits
85.23

    0
    0
    18

178.99
95
96
Central Michigan
Central Michigan
38 Commits
83.53

    0
    0
    35

178.86
96
98
Old Dominion
Old Dominion
30 Commits
83.78

    0
    0
    29

178.55
97
97
Utah State
Utah State
20 Commits
84.83

    0
    0
    20

177.56
98
103
Nevada
Nevada
27 Commits
83.84

    0
    0
    26

177.11
99
101
Louisiana Tech
Louisiana Tech
24 Commits
84.07

    0
    0
    22

177.00
100
93
Oregon State
Oregon State
19 Commits
85.37

    0
    0
    17

176.78
101
95
Buffalo
Buffalo
20 Commits
84.60

    0
    0
    20

176.60
102
109
FIU
FIU
21 Commits
84.55

    0
    0
    21

176.58
103
102
Charlotte
Charlotte
27 Commits
84.26

    0
    0
    27

176.53
104
105
Rice
Rice
25 Commits
84.07

    0
    0
    24

176.01
105
107
Miami (OH)
Miami (OH)
19 Commits
85.01

    0
    0
    19

175.90
106
106
Eastern Michigan
Eastern Michigan
38 Commits
83.66

    0
    0
    32

175.41
107
104
Ohio
Ohio
18 Commits
85.06

    0
    0
    18

175.40
108
112
Southern Miss
Southern Miss
25 Commits
83.96

    0
    0
    22

175.33
109
110
Liberty
Liberty
19 Commits
84.47

    0
    0
    19

172.64
110
108
UConn
UConn
16 Commits
85.45

    0
    0
    16

172.49
111
99
Virginia
Virginia
14 Commits
86.38

    0
    0
    14

171.82
112
113
Western Michigan
Western Michigan
23 Commits
83.78

    0
    0
    23

171.38
113
114
Wyoming
Wyoming
23 Commits
83.75

    0
    0
    23

170.23
114
111
Ball State
Ball State
22 Commits
83.89

    0
    0
    21

170.02
115
121
Middle Tennessee
Middle Tennessee
19 Commits
84.22

    0
    0
    18

167.21
116
116
Delaware
Delaware
19 Commits
84.11

    0
    0
    19

167.01
117
119
UMass
UMass
20 Commits
83.94

    0
    0
    18

166.97
118
115
North Dakota State
North Dakota State
21 Commits
84.08

    0
    0
    19

166.58
119
117
Coastal Carolina
Coastal Carolina
17 Commits
84.47

    0
    0
    17

165.03
120
118
Kennesaw State
Kennesaw State
18 Commits
84.10

    0
    0
    18

163.41
121
122
San Jose State
San Jose State
16 Commits
84.73

    0
    0
    16

162.84
122
120
Toledo
Toledo
16 Commits
84.55

    0
    0
    16

162.52
123
125
Kent State
Kent State
18 Commits
83.79

    0
    0
    18

162.20
124
124
Marshall
Marshall
19 Commits
83.99

    0
    0
    17

162.09
125
126
Hawaii
Hawaii
19 Commits
83.60

    0
    0
    19

161.47
126
128
Sacramento State
Sacramento State
30 Commits
82.68

    0
    0
    23

161.23
127
123
Sam Houston
Sam Houston
19 Commits
83.78

    0
    0
    18

161.18
128
132
Louisiana-Monroe
Louisiana-Monroe
25 Commits
82.61

    0
    0
    22

160.28
129
127
Washington State
Washington State
16 Commits
84.90

    0
    0
    13

159.25
130
130
Florida Atlantic
Florida Atlantic
14 Commits
85.15

    0
    0
    14

157.50
131
131
Missouri State
Missouri State
18 Commits
82.79

    0
    0
    16

151.56
132
129
Idaho
Idaho
21 Commits
82.70

    0
    0
    17

150.00
133
134
Akron
Akron
14 Commits
83.76

    0
    0
    14

147.36
134
133
Montana State
Montana State
18 Commits
83.36

    0
    0
    14

142.34
135
135
UAB
UAB
12 Commits
83.96

    0
    0
    12

135.31
136
138
New Mexico State
New Mexico State
11 Commits
83.98

    0
    0
    11

129.05
137
137
North Texas
North Texas
11 Commits
83.91

    0
    0
    11

128.04
138
136
Northern Arizona
Northern Arizona
21 Commits
82.54

    0
    0
    13

127.93
139
139
Cal Poly
Cal Poly
29 Commits
82.50

    0
    0
    12

120.60
140
140
UC Davis
UC Davis
16 Commits
82.73

    0
    0
    11

117.36
141
141
Air Force
Air Force
49 Commits
83.50

    0
    0
    10

116.49
0
0
Chicago State
Chicago State
10 Commits
84.88

    0
    0
    8

108.35
142
143
Harvard
Harvard
11 Commits
84.69

    0
    0
    8

106.82
143
142
Eastern Washington
Eastern Washington
19 Commits
83.28

    0
    0
    9

106.39
144
145
Navy
Navy
51 Commits
83.62

    0
    0
    8

99.40
145
144
Illinois State
Illinois State
17 Commits
85.00

    0
    0
    7

97.52
146
147
West Georgia
West Georgia
15 Commits
84.74

    0
    0
    7

95.93
147
146
Utah Tech
Utah Tech
10 Commits
82.93

    0
    0
    8

94.83
148
148
Portland State
Portland State
19 Commits
84.13

    0
    0
    7

92.18
149
149
Stephen F. Austin
Stephen F. Austin
14 Commits
83.93

    0
    0
    7

90.81
150
150
McNeese
McNeese
13 Commits
85.08

    0
    0
    6

86.12
151
151
Weber State
Weber State
18 Commits
81.68

    0
    0
    8

85.27
152
152
Montana
Montana
19 Commits
82.43

    0
    0
    7

80.74
153
153
Southern
Southern
19 Commits
80.44

    0
    0
    4

77.80
154
157
Murray State
Murray State
10 Commits
85.38

    0
    0
    5

74.36
155
155
North Dakota
North Dakota
20 Commits
82.96

    0
    0
    6

74.17
156
154
Gardner-Webb
Gardner-Webb
12 Commits
82.92

    0
    0
    5

74.17
157
156
South Dakota State
South Dakota State
27 Commits
84.87

    0
    0
    5

71.90
158
159
Jackson State
Jackson State
7 Commits
84.60

    0
    0
    5

70.68
159
158
Villanova
Villanova
10 Commits
83.60

    0
    0
    5

65.77
160
162
Wofford
Wofford
9 Commits
85.44

    0
    0
    4

60.53
161
160
Nicholls
Nicholls
8 Commits
84.67

    0
    0
    4

57.53
162
163
Rhode Island
Rhode Island
5 Commits
84.39

    0
    0
    4

56.44
163
161
Furman
Furman
9 Commits
84.25

    0
    0
    4

55.92
164
165
Howard
Howard
6 Commits
83.63

    0
    0
    4

53.45
165
164
Pennsylvania
Pennsylvania
6 Commits
83.50

    0
    0
    3

53.11
166
166
Columbia
Columbia
9 Commits
83.25

    0
    0
    4

51.95
167
168
Albany
Albany
8 Commits
83.13

    0
    0
    4

51.51
168
171
Alabama State
Alabama State
7 Commits
82.94

    0
    0
    3

51.00
169
169
Mercer
Mercer
5 Commits
82.63

    0
    0
    4

49.48
170
167
Idaho State
Idaho State
13 Commits
82.36

    0
    0
    4

48.56
171
170
Tarleton State
Tarleton State
8 Commits
86.31

    0
    0
    3

48.47
172
172
South Dakota
South Dakota
19 Commits
82.25

    0
    0
    4

48.03
173
173
Prairie View A&M
Prairie View A&M
9 Commits
82.13

    0
    0
    3

47.66
174
176
Western Carolina
Western Carolina
5 Commits
85.30

    0
    0
    3

45.43
175
183
Towson
Towson
6 Commits
85.09

    0
    0
    3

44.84
0
0
Central State
Central State
7 Commits
85.00

    0
    0
    3

44.59
176
174
Chattanooga
Chattanooga
5 Commits
85.00

    0
    0
    3

44.57
177
175
Samford
Samford
4 Commits
84.67

    0
    0
    3

43.62
178
176
Central Arkansas
Central Arkansas
7 Commits
84.67

    0
    0
    3

43.57
179
178
Ferris State
Ferris State
23 Commits
84.17

    0
    0
    3

42.08
180
179
Lamar
Lamar
6 Commits
83.67

    0
    0
    3

40.62
181
186
Army
Army
68 Commits
83.41

    0
    0
    3

39.85
182
181
Abilene Christian
Abilene Christian
3 Commits
83.33

    0
    0
    3

39.63
183
182
Lehigh
Lehigh
6 Commits
83.33

    0
    0
    3

39.62
184
183
Brown
Brown
6 Commits
83.33

    0
    0
    3

39.60
185
180
Southeastern Louisiana
Southeastern Louisiana
8 Commits
83.26

    0
    0
    3

39.43
186
185
Lafayette
Lafayette
7 Commits
83.00

    0
    0
    3

38.63
187
187
Princeton
Princeton
12 Commits
82.00

    0
    0
    2

35.70
188
188
UT Martin
UT Martin
6 Commits
87.00

    0
    0
    2

33.90
189
196
Northern Colorado
Northern Colorado
7 Commits
85.67

    0
    0
    2

31.24
190
190
Savannah State
Savannah State
2 Commits
85.50

    0
    0
    2

30.91
191
189
Pittsburg State
Pittsburg State
3 Commits
85.33

    0
    0
    2

30.59
192
191
Colgate
Colgate
7 Commits
85.17

    0
    0
    2

30.24
193
192
Lindenwood
Lindenwood
9 Commits
85.00

    0
    0
    2

29.92
194
193
Eastern Kentucky
Eastern Kentucky
4 Commits
84.00

    0
    0
    2

27.92
195
198
West Florida
West Florida
4 Commits
83.50

    0
    0
    2

26.92
196
197
East Tennessee State
East Tennessee State
21 Commits
83.11

    0
    0
    1

26.17
197
194
Western New Mexico
Western New Mexico
5 Commits
83.00

    0
    0
    2

25.94
198
195
Florida A&M
Florida A&M
5 Commits
83.00

    0
    0
    2

25.93
0
0
Western Oregon
Western Oregon
5 Commits
83.00

    0
    0
    2

25.93
0
0
Southern Arkansas
Southern Arkansas
2 Commits
83.00

    0
    0
    2

25.92
199
199
Austin Peay
Austin Peay
8 Commits
82.50

    0
    0
    2

24.94
200
200
Cornell
Cornell
8 Commits
82.00

    0
    0
    2

23.93
201
201
North Alabama
North Alabama
5 Commits
82.00

    0
    0
    2

23.93
202
202
Northwestern State
Northwestern State
6 Commits
81.75

    0
    0
    2

23.43
203
203
North Carolina A&T
North Carolina A&T
1 Commits
88.00

    0
    0
    1

18.00
204
207
Texas Southern
Texas Southern
3 Commits
86.89

    0
    0
    1

16.89
205
207
Shorter
Shorter
3 Commits
86.50

    0
    0
    1

16.50
206
204
West Texas A&M
West Texas A&M
1 Commits
86.33

    0
    0
    1

16.33
207
207
Alabama A&M
Alabama A&M
5 Commits
86.11

    0
    0
    1

16.11
208
205
CSU-Pueblo
CSU-Pueblo
2 Commits
86.00

    0
    0
    1

16.00
0
0
Eastern New Mexico
Eastern New Mexico
2 Commits
86.00

    0
    0
    1

16.00
208
207
Youngstown State
Youngstown State
17 Commits
86.00

    0
    0
    1

16.00
210
205
Richmond
Richmond
10 Commits
85.78

    0
    0
    1

15.78
211
207
Mississippi Valley State
Mississippi Valley State
2 Commits
85.67

    0
    0
    1

15.67
212
207
Delaware State
Delaware State
2 Commits
85.00

    0
    0
    1

15.00
213
220
Bryant
Bryant
6 Commits
84.78

    0
    0
    1

14.78
214
213
Bethune-Cookman
Bethune-Cookman
2 Commits
84.00

    0
    0
    1

14.00
214
213
Dartmouth
Dartmouth
10 Commits
84.00

    0
    0
    1

14.00
214
213
Hampton
Hampton
5 Commits
84.00

    0
    0
    1

14.00
214
213
Newberry
Newberry
1 Commits
84.00

    0
    0
    1

14.00
214
213
Western Illinois
Western Illinois
15 Commits
84.00

    0
    0
    1

14.00
214
213
William & Mary
William & Mary
7 Commits
84.00

    0
    0
    1

14.00
220
220
Fordham
Fordham
2 Commits
83.50

    0
    0
    1

13.50
220
213
Grand Valley State
Grand Valley State
22 Commits
83.50

    0
    0
    1

13.50
220
226
UTRGV
UTRGV
1 Commits
83.50

    0
    0
    1

13.50
0
0
Black Hills State
Black Hills State
1 Commits
83.00

    0
    0
    1

13.00
223
220
Central Washington
Central Washington
2 Commits
83.00

    0
    0
    1

13.00
223
220
Colorado School of Mines
Colorado School of Mines
1 Commits
83.00

    0
    0
    1

13.00
223
220
Grambling State
Grambling State
2 Commits
83.00

    0
    0
    1

13.00
0
0
Hampden-Sydney
Hampden-Sydney
1 Commits
83.00

    0
    0
    1

13.00
0
0
Missouri Western State
Missouri Western State
1 Commits
83.00

    0
    0
    1

13.00
0
0
Southern Virginia
Southern Virginia
1 Commits
83.00

    0
    0
    1

13.00
223
220
Yale
Yale
5 Commits
83.00

    0
    0
    1

13.00
227
226
Tennessee State
Tennessee State
5 Commits
82.50

    0
    0
    1

12.50
0
0
Fort Lewis College
Fort Lewis College
1 Commits
82.00

    0
    0
    1

12.00
0
0
Montana Tech
Montana Tech
2 Commits
82.00

    0
    0
    1

12.00
0
0
Western Colorado
Western Colorado
1 Commits
81.00

    0
    0
    1

11.00
228
228
Merrimack
Merrimack
2 Commits
80.00

    0
    0
    1

10.00
228
228
Presbyterian
Presbyterian
4 Commits
80.00

    0
    0
    1

10.00
228
228
San Diego
San Diego
8 Commits
80.00

    0
    0
    1

10.00
231
231
Northern State
Northern State
1 Commits
77.00

    0
    0
    0

7.00
0
0
Southern Oregon
Southern Oregon
1 Commits
76.00

    0
    0
    0

6.00

"""

TRANSFER_TEXT_2026 = """
1
LSU
LSU
41 Commits
88.45

    3
    14
    23

81.77
2
Ole Miss
Ole Miss
29 Commits
88.69

    0
    12
    17

60.95
3
Texas
Texas
22 Commits
88.59

    1
    9
    12

60.31
4
Penn State
Penn State
39 Commits
87.21

    0
    9
    29

53.94
5
Miami
Miami
13 Commits
89.54

    0
    7
    6

52.28
6
Texas Tech
Texas Tech
22 Commits
88.05

    1
    6
    13

51.53
7
Ohio State
Ohio State
17 Commits
88.88

    0
    7
    10

50.43
8
Indiana
Indiana
17 Commits
88.41

    0
    6
    11

48.31
9
Auburn
Auburn
39 Commits
87.41

    0
    9
    30

47.93
10
Kentucky
Kentucky
29 Commits
87.14

    0
    6
    23

47.73
11
Notre Dame
Notre Dame
7 Commits
92.71

    0
    7
    0

44.64
12
Texas A&M
Texas A&M
19 Commits
87.84

    0
    5
    14

44.26
13
Louisville
Louisville
34 Commits
86.59

    0
    7
    25

43.32
14
California
California
32 Commits
87.20

    0
    7
    23

42.97
15
Oklahoma State
Oklahoma State
55 Commits
85.85

    1
    4
    50

42.59
16
Georgia
Georgia
9 Commits
90.56

    0
    6
    3

41.64
17
Oklahoma
Oklahoma
16 Commits
87.94

    0
    6
    10

41.47
18
Alabama
Alabama
17 Commits
88.12

    0
    5
    12

40.70
19
South Carolina
South Carolina
25 Commits
86.96

    1
    2
    22

40.26
20
Missouri
Missouri
30 Commits
86.67

    0
    6
    24

39.69
21
Oregon
Oregon
13 Commits
88.15

    0
    4
    9

38.06
22
Arizona State
Arizona State
25 Commits
86.70

    0
    3
    20

37.70
23
Colorado
Colorado
43 Commits
86.62

    0
    3
    39

36.90
24
Michigan
Michigan
17 Commits
87.50

    0
    4
    12

36.17
25
UCLA
UCLA
42 Commits
86.17

    0
    5
    36

34.74
26
Tennessee
Tennessee
21 Commits
86.81

    0
    4
    17

34.39
27
Virginia Tech
Virginia Tech
27 Commits
86.48

    0
    3
    24

32.75
28
Florida State
Florida State
23 Commits
86.50

    0
    3
    19

32.50
29
USC
USC
10 Commits
88.13

    0
    3
    5

31.96
30
Florida
Florida
29 Commits
86.28

    0
    2
    23

30.97
31
Nebraska
Nebraska
17 Commits
87.19

    0
    2
    14

28.74
32
SMU
SMU
17 Commits
87.13

    0
    3
    12

28.59
33
Arkansas
Arkansas
42 Commits
85.80

    0
    3
    38

27.89
34
Vanderbilt
Vanderbilt
18 Commits
86.94

    0
    3
    13

27.40
35
Georgia Tech
Georgia Tech
19 Commits
86.53

    0
    3
    16

27.36
36
Baylor
Baylor
30 Commits
86.03

    0
    2
    28

27.34
37
Virginia
Virginia
31 Commits
86.21

    0
    2
    27

26.91
38
Wisconsin
Wisconsin
33 Commits
86.06

    0
    3
    30

25.94
39
BYU
BYU
9 Commits
87.78

    0
    3
    6

25.42
40
Mississippi State
Mississippi State
27 Commits
86.15

    0
    3
    24

24.72
41
Cincinnati
Cincinnati
22 Commits
86.18

    0
    2
    20

24.04
42
Northwestern
Northwestern
17 Commits
85.94

    0
    3
    14

22.64
43
Illinois
Illinois
19 Commits
85.68

    0
    2
    17

22.50
44
Utah
Utah
17 Commits
86.38

    0
    2
    14

22.48
45
Kansas State
Kansas State
27 Commits
85.69

    0
    1
    25

21.57
46
NC State
NC State
20 Commits
86.30

    0
    1
    19

21.40
47
TCU
TCU
12 Commits
86.42

    0
    2
    10

20.29
48
Purdue
Purdue
29 Commits
85.76

    0
    1
    28

20.25
49
Arizona
Arizona
23 Commits
85.86

    0
    1
    20

19.58
50
Houston
Houston
19 Commits
85.89

    0
    0
    18

19.42
51
Minnesota
Minnesota
19 Commits
85.53

    0
    0
    19

18.82
52
North Carolina
North Carolina
20 Commits
86.00

    0
    0
    20

18.55
53
Kansas
Kansas
31 Commits
85.48

    0
    0
    31

18.31
54
Syracuse
Syracuse
18 Commits
86.06

    0
    1
    17

17.61
55
Michigan State
Michigan State
29 Commits
85.44

    0
    0
    27

16.93
56
Clemson
Clemson
11 Commits
86.09

    0
    1
    10

16.45
57
West Virginia
West Virginia
34 Commits
85.24

    0
    0
    34

16.09
58
South Florida
South Florida
44 Commits
84.86

    0
    0
    42

15.69
59
Washington
Washington
14 Commits
85.71

    0
    0
    14

15.08
60
Boston College
Boston College
26 Commits
85.31

    0
    0
    26

14.32
61
James Madison
James Madison
41 Commits
84.49

    0
    0
    36

14.22
62
Maryland
Maryland
15 Commits
85.27

    0
    0
    15

14.15
63
Duke
Duke
19 Commits
85.58

    0
    0
    19

14.04
64
UCF
UCF
32 Commits
84.94

    0
    0
    31

13.95
65
Memphis
Memphis
53 Commits
84.21

    0
    0
    52

13.64
66
Wake Forest
Wake Forest
24 Commits
85.38

    0
    0
    24

13.38
67
Iowa State
Iowa State
48 Commits
84.94

    0
    0
    48

13.36
68
Tulsa
Tulsa
24 Commits
84.50

    0
    0
    24

13.06
69
North Texas
North Texas
49 Commits
84.41

    0
    0
    46

12.80
70
UConn
UConn
56 Commits
84.38

    0
    0
    53

12.65
71
Colorado State
Colorado State
35 Commits
84.50

    0
    0
    34

12.57
72
Washington State
Washington State
28 Commits
85.04

    0
    0
    27

12.46
73
East Carolina
East Carolina
24 Commits
84.43

    0
    0
    23

12.46
74
San Diego State
San Diego State
26 Commits
84.52

    0
    0
    25

12.24
75
Oregon State
Oregon State
21 Commits
84.52

    0
    0
    21

12.17
76
UAB
UAB
39 Commits
84.13

    0
    0
    38

12.16
77
Iowa
Iowa
15 Commits
85.14

    0
    0
    14

12.14
78
UNLV
UNLV
19 Commits
85.06

    0
    0
    17

12.00
79
Appalachian State
Appalachian State
38 Commits
84.41

    0
    0
    34

12.00
80
Arkansas State
Arkansas State
31 Commits
84.03

    0
    0
    28

11.66
81
Rutgers
Rutgers
15 Commits
85.13

    0
    0
    15

11.65
82
Coastal Carolina
Coastal Carolina
42 Commits
84.51

    0
    0
    37

11.42
83
Tulane
Tulane
21 Commits
84.42

    0
    0
    19

11.08
84
Miami (OH)
Miami (OH)
17 Commits
84.73

    0
    0
    15

10.96
85
Southern Miss
Southern Miss
38 Commits
83.68

    0
    0
    37

10.95
86
Utah State
Utah State
32 Commits
83.74

    0
    0
    31

10.90
87
Florida Atlantic
Florida Atlantic
30 Commits
84.10

    0
    0
    29

10.87
88
Georgia Southern
Georgia Southern
22 Commits
84.24

    0
    0
    21

10.82
89
Nevada
Nevada
18 Commits
84.65

    0
    0
    17

10.81
90
Pittsburgh
Pittsburgh
16 Commits
84.64

    0
    0
    14

10.79
91
Louisiana-Monroe
Louisiana-Monroe
20 Commits
83.70

    0
    0
    19

10.79
92
Texas State
Texas State
18 Commits
84.76

    0
    0
    17

10.78
93
Sacramento State
Sacramento State
27 Commits
83.50

    0
    0
    26

10.75
94
Toledo
Toledo
37 Commits
84.03

    0
    0
    34

10.64
95
Sam Houston
Sam Houston
22 Commits
84.00

    0
    0
    20

10.58
96
Charlotte
Charlotte
21 Commits
83.67

    0
    0
    21

10.58
97
San Jose State
San Jose State
21 Commits
83.74

    0
    0
    19

10.50
98
Liberty
Liberty
29 Commits
84.11

    0
    0
    27

10.48
99
Boise State
Boise State
12 Commits
84.67

    0
    0
    12

10.42
100
Temple
Temple
23 Commits
83.96

    0
    0
    23

10.39
101
Kennesaw State
Kennesaw State
29 Commits
83.78

    0
    0
    27

10.34
102
UTEP
UTEP
23 Commits
83.90

    0
    0
    20

10.34
103
Jacksonville State
Jacksonville State
21 Commits
83.65

    0
    0
    20

10.26
104
New Mexico
New Mexico
16 Commits
83.73

    0
    0
    15

10.19
105
Hawaii
Hawaii
18 Commits
84.11

    0
    0
    18

10.15
106
UTSA
UTSA
22 Commits
84.00

    0
    0
    20

10.10
107
Georgia State
Georgia State
33 Commits
83.69

    0
    0
    32

10.06
108
Marshall
Marshall
27 Commits
84.04

    0
    0
    26

10.00
109
UMass
UMass
25 Commits
83.44

    0
    0
    25

9.82
110
Western Michigan
Western Michigan
21 Commits
83.40

    0
    0
    19

9.80
111
Ball State
Ball State
26 Commits
83.32

    0
    0
    25

9.68
112
Louisiana Tech
Louisiana Tech
11 Commits
84.55

    0
    0
    11

9.62
113
Western Kentucky
Western Kentucky
20 Commits
84.00

    0
    0
    20

9.56
114
Missouri State
Missouri State
21 Commits
83.90

    0
    0
    20

9.54
115
New Mexico State
New Mexico State
26 Commits
83.96

    0
    0
    24

9.54
116
Buffalo
Buffalo
19 Commits
83.44

    0
    0
    18

9.53
117
FIU
FIU
19 Commits
83.68

    0
    0
    19

9.48
118
Northern Illinois
Northern Illinois
19 Commits
83.71

    0
    0
    17

9.39
119
Bowling Green
Bowling Green
16 Commits
83.44

    0
    0
    15

9.38
120
Old Dominion
Old Dominion
16 Commits
83.93

    0
    0
    15

9.36
121
Middle Tennessee
Middle Tennessee
18 Commits
83.59

    0
    0
    17

9.17
122
Delaware
Delaware
13 Commits
83.38

    0
    0
    12

9.07
123
Rice
Rice
20 Commits
83.50

    0
    0
    20

8.98
124
Akron
Akron
19 Commits
82.89

    0
    0
    19

8.95
125
Wyoming
Wyoming
19 Commits
83.37

    0
    0
    19

8.87
126
Kent State
Kent State
17 Commits
83.25

    0
    0
    15

8.55
127
Ohio
Ohio
20 Commits
83.22

    0
    0
    18

8.42
128
Fresno State
Fresno State
13 Commits
83.23

    0
    0
    13

8.37
129
Troy
Troy
15 Commits
83.85

    0
    0
    13

8.21
130
Stanford
Stanford
6 Commits
85.17

    0
    0
    6

7.92
131
Central Michigan
Central Michigan
15 Commits
82.67

    0
    0
    14

7.86
132
Eastern Michigan
Eastern Michigan
12 Commits
83.90

    0
    0
    10

7.79
133
Tarleton State
Tarleton State
15 Commits
84.38

    0
    0
    8

7.62
134
Alabama A&M
Alabama A&M
7 Commits
85.40

    0
    0
    5

7.37
135
South Alabama
South Alabama
12 Commits
83.00

    0
    0
    9

7.15
136
Tennessee Tech
Tennessee Tech
12 Commits
83.38

    0
    0
    8

6.35
137
Northern Arizona
Northern Arizona
10 Commits
83.40

    0
    0
    5

5.52
138
Louisiana
Louisiana
5 Commits
83.20

    0
    0
    5

4.99
139
Cal Poly
Cal Poly
11 Commits
84.25

    0
    0
    4

4.95
140
East Tennessee State
East Tennessee State
23 Commits
85.00

    0
    0
    3

4.82
141
Abilene Christian
Abilene Christian
7 Commits
84.67

    0
    0
    3

4.68
142
Stephen F. Austin
Stephen F. Austin
15 Commits
85.00

    0
    0
    3

4.59
143
Austin Peay
Austin Peay
28 Commits
83.50

    0
    0
    4

4.53
144
Prairie View A&M
Prairie View A&M
4 Commits
83.33

    0
    0
    3

3.99
145
Mercer
Mercer
6 Commits
82.25

    0
    0
    4

3.85
146
Eastern Kentucky
Eastern Kentucky
8 Commits
83.67

    0
    0
    3

3.80
147
Jackson State
Jackson State
3 Commits
85.00

    0
    0
    2

3.33
147
South Dakota
South Dakota
6 Commits
85.00

    0
    0
    2

3.33
147
UT Martin
UT Martin
2 Commits
85.00

    0
    0
    2

3.33
150
Western Carolina
Western Carolina
5 Commits
82.33

    0
    0
    3

3.27
151
Holy Cross
Holy Cross
4 Commits
82.00

    0
    0
    3

3.16
152
Lamar
Lamar
8 Commits
84.50

    0
    0
    2

3.07
152
Tennessee State
Tennessee State
4 Commits
84.50

    0
    0
    2

3.07
152
Utah Tech
Utah Tech
2 Commits
84.50

    0
    0
    2

3.07
0
Kilgore College
Kilgore College
2 Commits
84.00

    0
    0
    2

2.91
0
Professional
Professional
1 Commits
87.00

    0
    0
    1

2.85
155
Bryant
Bryant
6 Commits
83.50

    0
    0
    2

2.79
155
North Carolina Central
North Carolina Central
3 Commits
83.50

    0
    0
    2

2.79
155
Youngstown State
Youngstown State
5 Commits
83.50

    0
    0
    2

2.79
158
Morgan State
Morgan State
2 Commits
83.50

    0
    0
    2

2.61
158
South Dakota State
South Dakota State
3 Commits
83.50

    0
    0
    2

2.61
160
Campbell
Campbell
5 Commits
83.00

    0
    0
    2

2.42
160
Idaho State
Idaho State
7 Commits
83.00

    0
    0
    2

2.42
162
West Georgia
West Georgia
6 Commits
82.00

    0
    0
    2

2.32
163
Furman
Furman
6 Commits
82.50

    0
    0
    2

2.31
163
Portland State
Portland State
5 Commits
82.50

    0
    0
    2

2.31
165
Central Arkansas
Central Arkansas
4 Commits
86.00

    0
    0
    1

2.26
0
Long Beach City College
Long Beach City College
1 Commits
86.00

    0
    0
    1

2.26
165
Montana
Montana
13 Commits
86.00

    0
    0
    1

2.26
167
Robert Morris
Robert Morris
3 Commits
81.50

    0
    0
    2

2.13
0
Blinn College
Blinn College
1 Commits
85.00

    0
    0
    1

1.81
0
City College of San Francisco
City College of San Francisco
1 Commits
85.00

    0
    0
    1

1.81
168
Delaware State
Delaware State
2 Commits
85.00

    0
    0
    1

1.81
168
Elon
Elon
1 Commits
85.00

    0
    0
    1

1.81
168
Emporia State
Emporia State
1 Commits
85.00

    0
    0
    1

1.81
168
Grand Valley State
Grand Valley State
1 Commits
85.00

    0
    0
    1

1.81
168
Houston Christian
Houston Christian
1 Commits
85.00

    0
    0
    1

1.81
168
Incarnate Word
Incarnate Word
6 Commits
85.00

    0
    0
    1

1.81
168
Indiana State
Indiana State
5 Commits
85.00

    0
    0
    1

1.81
168
Maine
Maine
2 Commits
85.00

    0
    0
    1

1.81
0
Missouri S&T
Missouri S&T
1 Commits
85.00

    0
    0
    1

1.81
168
North Dakota State
North Dakota State
7 Commits
85.00

    0
    0
    1

1.81
168
Southern
Southern
3 Commits
85.00

    0
    0
    1

1.81
0
Southwest Mississippi C.C.
Southwest Mississippi C.C.
2 Commits
85.00

    0
    0
    1

1.81
168
Texas Southern
Texas Southern
1 Commits
85.00

    0
    0
    1

1.81
168
The Citadel
The Citadel
2 Commits
85.00

    0
    0
    1

1.81
168
UC Davis
UC Davis
5 Commits
85.00

    0
    0
    1

1.81
168
West Florida
West Florida
3 Commits
85.00

    0
    0
    1

1.81
182
Bethune-Cookman
Bethune-Cookman
4 Commits
84.00

    0
    0
    1

1.52
182
Chattanooga
Chattanooga
3 Commits
84.00

    0
    0
    1

1.52
0
Copiah-Lincoln C.C.
Copiah-Lincoln C.C.
1 Commits
84.00

    0
    0
    1

1.52
0
Hutchinson C.C.
Hutchinson C.C.
1 Commits
84.00

    0
    0
    1

1.52
182
North Alabama
North Alabama
7 Commits
84.00

    0
    0
    1

1.52
182
Southeastern Louisiana
Southeastern Louisiana
1 Commits
84.00

    0
    0
    1

1.52
182
UTRGV
UTRGV
4 Commits
84.00

    0
    0
    1

1.52
182
Wagner
Wagner
1 Commits
84.00

    0
    0
    1

1.52
188
McNeese
McNeese
2 Commits
83.00

    0
    0
    1

1.33
188
Nicholls
Nicholls
3 Commits
83.00

    0
    0
    1

1.33
188
Norfolk State
Norfolk State
2 Commits
83.00

    0
    0
    1

1.33
188
Wofford
Wofford
1 Commits
83.00

    0
    0
    1

1.33
192
Harding
Harding
1 Commits
82.00

    0
    0
    1

1.19
0
Johnson C. Smith
Johnson C. Smith
1 Commits
82.00

    0
    0
    1

1.19
192
North Dakota
North Dakota
3 Commits
82.00

    0
    0
    1

1.19
192
Stony Brook
Stony Brook
3 Commits
82.00

    0
    0
    1

1.19
195
Idaho
Idaho
3 Commits
80.00

    0
    0
    1

0.98
195
Mercyhurst
Mercyhurst
1 Commits
80.00

    0
    0
    1

0.98
195
New Hampshire
New Hampshire
3 Commits
80.00

    0
    0
    1

0.98
195
South Carolina State
South Carolina State
4 Commits
80.00

    0
    0
    1

0.98

"""

RETURNING_PRODUCTION_TEXT_2026 = """
1. Notre Dame
66%
57% (17)
73% (2)
2. Maryland
65%
59% (14)
70% (3)
3. BYU
63%
60% (11)
65% (4)
4. Virginia Tech
62%
64% (4)
60% (9)
5. Georgia
61%
58% (16)
65% (5)
6. Stanford
59%
55% (22)
64% (7)
7. New Mexico
58%
61% (10)
56% (18)
8. Delaware
57%
59% (13)
54% (19)
9. Air Force
57%
41% (63)
75% (1)
10. USC
56%
58% (15)
53% (23)
11. Ohio State
56%
67% (2)
43% (55)
12. Nebraska
56%
54% (23)
57% (13)
13. Oklahoma
55%
54% (24)
57% (14)
14. Minnesota
54%
56% (18)
52% (27)
15. Texas
54%
50% (37)
58% (11)
16. Oregon
54%
48% (44)
60% (10)
17. Army
54%
72% (1)
34% (93)
18. Tennessee
53%
61% (9)
45% (54)
19. Eastern Michigan
53%
41% (61)
64% (6)
20. Navy
53%
49% (39)
56% (16)
21. Texas Tech
52%
55% (20)
49% (42)
22. Boise State
52%
54% (25)
50% (39)
23. Florida Atlantic
52%
46% (49)
57% (12)
24. Fresno State
51%
56% (27)
50% (37)
25. Washington
51%
56% (19)
47% (48)
26. Pittsburgh
51%
51% (32)
51% (35)
27. Florida
51%
38% (72)
63% (8)
28. Michigan
51%
62% (7)
40% (66)
29. North Dakota State
51%
49% (43)
53% (26)
30. Houston
50%
50% (35)
51% (34)
31. Temple
50%
65% (3)
36% (81)
32. Northwestern
50%
46% (50)
54% (21)
33. South Carolina
49%
48% (45)
50% (36)
34. Miami (Ohio)
49%
50% (36)
49% (46)
35. Clemson
49%
49% (41)
49% (45)
36. Arkansas State
49%
61% (8)
35% (84)
37. Louisiana
48%
59% (12)
38% (74)
38. Tulsa
48%
45% (52)
51% (33)
39. SMU
46%
51% (31)
43% (56)
40. San Diego State
46%
63% (6)
31% (108)
41. Miami
46%
36% (79)
57% (15)
42. Texas A&M
46%
43% (57)
50% (41)
43. Vanderbilt
46%
39% (69)
52% (28)
44. Utah State
45%
40% (65)
50% (38)
45. UTSA
45%
55% (21)
35% (83)
46. Texas State
45%
51% (34)
40% (68)
47. TCU
45%
39% (67)
51% (31)
48. Jacksonville State
45%
41% (64)
49% (44)
49. Louisiana Tech
45%
52% (28)
38% (76)
50. Liberty
45%
51% (33)
38% (72)
51. Arizona
44%
48% (46)
40% (65)
52. Western Michigan
44%
52% (29)
36% (82)
53. Ole Miss
44%
49% (42)
39% (70)
54. Syracuse
43%
36% (81)
51% (32)
55. Marshall
43%
51% (30)
35% (89)
56. UCF
42%
32% (92)
53% (25)
57. Kansas
42%
31% (94)
52% (29)
58. NC State
42%
43% (55)
40% (69)
59. California
41%
47% (48)
35% (85)
60. Central Michigan
41%
63% (5)
20% (123)
61. Akron
41%
45% (53)
37% (78)
62. Indiana
41%
34% (84)
48% (47)
63. Alabama
41%
30% (96)
53% (24)
64. Wake Forest
41%
25% (111)
56% (17)
65. Missouri
41%
53% (26)
27% (115)
66. Duke
41%
35% (83)
46% (51)
67. Hawaii
40%
47% (47)
33% (98)
68. Kansas State
40%
49% (38)
33% (100)
69. Louisiana Monroe
40%
43% (56)
37% (77)
70. Kent State
40%
49% (40)
32% (103)
71. UCLA
40%
42% (60)
38% (75)
72. Virginia
39%
39% (68)
40% (67)
73. Boston College
39%
25% (107)
54% (20)
74. Utah
39%
37% (74)
42% (58)
75. Iowa
39%
45% (51)
33% (97)
76. South Alabama
38%
42% (59)
34% (92)
77. Georgia Tech
38%
24% (114)
52% (30)
78. New Mexico State
38%
26% (103)
50% (40)
79. FIU
37%
36% (78)
38% (73)
80. Purdue
37%
33% (87)
41% (62)
81. Mississippi State
37%
33% (90)
41% (60)
82. Sam Houston
36%
32% (93)
41% (63)
83. Rutgers
36%
41% (62)
30% (110)
84. Oregon State
36%
31% (95)
41% (61)
85. Georgia Southern
35%
17% (126)
54% (22)
86. Tulane
35%
24% (117)
47% (49)
87. Illinois
35%
36% (76)
33% (96)
88. Rice
34%
36% (77)
33% (102)
89. LSU
34%
25% (104)
42% (57)
90. Missouri State
34%
33% (88)
35% (87)
91. UNLV
34%
33% (89)
35% (86)
92. Kennesaw State
34%
22% (121)
46% (52)
93. Nevada
34%
38% (70)
29% (112)
94. Cincinnati
33%
36% (82)
32% (105)
95. Georgia State
33%
40% (66)
26% (116)
96. Wisconsin
33%
37% (75)
30% (109)
97. Troy
32%
38% (71)
27% (114)
98. Florida State
32%
24% (116)
41% (59)
99. Wyoming
32%
34% (85)
29% (113)
100. Old Dominion
31%
12% (132)
49% (43)
101. Arizona State
31%
27% (98)
34% (90)
102. UMass
30%
29% (97)
31% (107)
103. Bowling Green
30%
25% (108)
35% (88)
104. Middle Tennessee
30%
25% (106)
34% (91)
105. Louisville
30%
27% (102)
33% (99)
106. Baylor
30%
24% (118)
36% (80)
107. Kentucky
29%
18% (125)
41% (64)
108. Charlotte
29%
25% (110)
33% (101)
109. Arkansas
29%
37% (73)
21% (120)
110. Ohio
29%
13% (131)
46% (53)
111. Auburn
29%
11% (133)
46% (50)
112. North Carolina
29%
25% (109)
32% (104)
113. Ball State
28%
36% (80)
20% (122)
114. Sacramento State
27%
20% (122)
34% (94)
115. Coastal Carolina
27%
24% (112)
30% (111)
116. Northern Illinois
27%
42% (58)
12% (130)
117. Washington State
27%
44% (54)
8% (135)
118. Buffalo
27%
22% (119)
31% (106)
119. Michigan State
27%
33% (91)
20% (121)
120. East Carolina
26%
16% (127)
36% (79)
121. App State
25%
27% (99)
24% (119)
122. Colorado State
25%
16% (128)
34% (95)
123. UTEP
25%
10% (135)
39% (71)
124. Western Kentucky
23%
22% (120)
24% (118)
125. South Florida
22%
25% (105)
19% (124)
126. Penn State
22%
19% (124)
25% (117)
127. Colorado
21%
34% (86)
8% (134)
128. UAB
20%
27% (101)
13% (129)
129. San Jose State
20%
24% (115)
15% (127)
130. West Virginia
19%
27% (100)
11% (131)
131. James Madison
19%
20% (123)
18% (125)
132. Toledo
17%
24% (113)
10% (133)
133. Memphis
10%
15% (129)
6% (136)
134. Oklahoma State
10%
4% (137)
15% (128)
135. Iowa State
10%
4% (138)
16% (126)
136. Southern Miss
10%
15% (130)
5% (137)
137. North Texas
8%
5% (136)
11% (132)
138. UConn
7%
11% (134)
4% (138)

"""

COACHING_HIRES_TEXT_2026 = """
30. Ryan Silverfield, Arkansas
Career College Record: 50-25 (67%)
Previous Stop: Memphis head coach
Expectation: Turn Arkansas into a regular bowl team
CFN Grade: C-

29. Rob Harley, Northern Illinois
Career College Record: 0-0
Previous Stop: Northern Illinois defensive coordinator
Expectation: Try to make Northern Illinois competitive enough to give him a shot at the full-time job.
CFN Grade: C

28. Will Hall, Tulane
Career College Record: 14-30 (32%)
Previous Stop: Tulane passing game coordinator
Expectation: Keep the program at a College Football Playoff level.
CFN Grade: C

27. Kirby Moore, Washington State
Career College Record: 0-0
Previous Stop: Missouri offensive coordinator
Expectation: Make the Wazzu offense crank in the new Pac-12.
CFN Grade: C+

26. Kyle Whittingham, Michigan
Career College Record: 177-88 (69%)
Previous Stop: Utah head coach
Expectation: Be in the national title hunt, win the Big Ten, beat Ohio State (not necessarily in that order)
CFN Grade: B-

25. Alonzo Carter, Sacramento State
Career College Record: 0-0
Previous Stop: Arizona running backs coach
Expectation: Make the Hornets competitive in the MAC
CFN Grade: B

24. JaMarcus Shephard, Oregon State
Career College Record: 0-0
Previous Stop: Alabama assistant head coach
Expectation: Rebuild Oregon State back up in the new Pac-12
CFN Grade: B

23. Casey Woods, Missouri State
Career College Record: 0-0
Previous Stop: SMU offensive coordinator
Expectation: Make the Bears a fun team that goes to bowls
CFN Grade: B

22. Will Stein, Kentucky
Career College Record: 0-0
Previous Stop: Oregon offensive coordinator
Expectation: Turn Kentucky into a dangerous SEC threat.
CFN Grade: B

21. Jimmy Rogers, Iowa State
Career College Record: 6-6 (50%)
Previous Stop: Washington State head coach
Expectation: Finally get Iowa State that outright Big conference championship within a few years.
CFN Grade: B

20. Neal Brown, North Texas
Career College Record: 72-51 (59%)
Previous Stop: Texas assistant coach
Expectation: Keep North Texas relevant.
CFN Grade: B

19. Brian Hartline, USF
Career College Record: 0-0
Previous Stop: Ohio State offensive coordinator
Expectation: Be in the American title mix every season.
CFN Grade: B

18. Morgan Scalley, Utah
Career College Record: 1-0 (100%)
Previous Stop: Utah defensive coordinator
Expectation: Win Big 12 titles and finally get the Utes to the College Football Playoff.
CFN Grade: B

17. Alex Golesh, Auburn
Career College Record: 23-15 (61%)
Previous Stop: USF head coach
Expectation: Beat Bama, make Auburn a College Football Playoff program.
CFN Grade: B

16. Matt Campbell, Penn State
Career College Record: 107-70 (61%)
Previous Stop: Iowa State head coach
Expectation: Win a national title
CFN Grade: B

15. Tavita Pritchard, Stanford
Career College Record: 0-0
Previous Stop: Washington Commanders quarterback coach
Expectation: Build Stanford back up, and make noise in the ACC.
CFN Grade: B+

14. Tosh Lupoi, Cal
Career College Record: 0-0
Previous Stop: Oregon defensive coordinator
Expectation: Make Cal into an annual factor in the ACC title chase.
CFN Grade: B+

13. Pat Fitzgerald, Michigan State
Career College Record: 110-101 (52%)
Previous Stop: Northwestern head coach
Expectation: Get Michigan State back up to speed with a toughness and fight like the Mark Dantonio era.
CFN Grade: B+

12. Eric Morris, Oklahoma State
Career College Record: 22-16 (58%)
Previous Stop: North Texas head coach
Expectation: Shake up the Big 12 and be in the conference title chase every year.
CFN Grade: A-

11. Jason Candle, UConn
Career College Record: 81-44 (65%)
Previous Stop: Toledo head coach
Expectation: Keep being a bother in everyone's schedule - a bowl game every year.
CFN Grade: A-

10. Charles Huff, Memphis
Career College Record: 39-25 (61%)
Previous Stop: Southern Miss head coach
Expectation: Win American Conference titles, plural.
CFN Grade: A-

9. Ryan Beard, Coastal Carolina
Career College Record: 7-5 (58%)
Previous Stop: Missouri State head coach
Expectation: Get the Chanticleers back in the Sun Belt title chase.
CFN Grade: A

8. Billy Napier, James Madison
Career College Record: 62-35 (64%)
Previous Stop: Florida head coach
Expectation: Win the Sun Belt title right now.
CFN Grade: A

7. Mike Jacobs, Toledo
Career College Record: 0-0
Previous Stop: Mercer head coach
Expectation: Be in the MAC title chase every season.
CFN Grade: A

6. Collin Klein, Kansas State
Career College Record: 0-0
Previous Stop: Texas A&M offensive coordinator
Expectation: Big 12 championship runs and College Football Playoff appearances.
CFN Grade: A

5. Jim Mora, Colorado State
Career College Record: 73-53 (58%)
Previous Stop: UConn head coach
Expectation: Make Colorado State an instant Pac-12 player.
CFN Grade: A

4. Jon Sumrall, Florida
Career College Record: 43-12 (78%)
Previous Stop: Tulane head coach
Expectation: Win the national championship after a year or so of rebuilding
CFN Grade: A

3. Bob Chesney, UCLA
Career College Record: 21-6 (78%)
Previous Stop: James Madison head coach
Expectation: Turn UCLA into a regular bowl team fast, and then make some Big Ten noise in a few years.
CFN Grade: A

2. James Franklin, Virginia Tech
Career College Record: 128-60 (68%)
Previous Stop: Penn State head coach
Expectation: ACC championship and College Football Playoff appearances.
CFN Grade: A+

1. Lane Kiffin, LSU
Career College Record: 116-52 (69%)
Previous Stop: Ole Miss head coach
Expectation: Win national championships - plural.
CFN Grade: A+

"""

SP_PLUS_TEXT_2026 = """
1. Ohio State	31.8	40.6 (2)	8.8 (1)	0.1 (67)
2. Oregon	28.3	40.7 (1)	12.6 (3)	0.3 (50)
3. Notre Dame	25.8	40.2 (3)	14.6 (9)	0.3 (47)
4. Georgia	25.5	38.2 (5)	13.3 (5)	0.6 (3)
5. Indiana	24.5	37.4 (9)	13.5 (6)	0.5 (22)
6. Texas	23.7	37.6 (7)	14.3 (8)	0.4 (31)
7. Texas Tech	23.1	37.6 (8)	14.7 (11)	0.2 (54)
8. Miami	21.0	34.4 (12)	13.7 (7)	0.3 (42)
9. Texas A&M	20.3	37.3 (10)	16.7 (14)	-0.3 (99)
10. LSU	20.2	32.5 (21)	12.5 (2)	0.2 (53)
11. Alabama	18.2	31.3 (32)	12.6 (4)	-0.4 (109)
12. Oklahoma	17.2	31.6 (27)	14.7 (10)	0.4 (33)
13. USC	16.8	37.7 (6)	20.5 (29)	-0.3 (100)
14. Michigan	16.1	32.8 (20)	16.2 (13)	-0.5 (113)
15. Tennessee	16.0	38.7 (4)	23.0 (50)	0.4 (34)
16. Ole Miss	15.9	35.0 (11)	19.7 (25)	0.7 (1)
17. Penn St.	15.7	33.6 (14)	18.6 (23)	0.6 (6)
18. BYU	15.5	33.3 (16)	18.1 (20)	0.3 (45)
19. Florida	14.9	30.2 (39)	15.9 (12)	0.5 (17)
20. Missouri	14.8	32.2 (24)	16.9 (15)	-0.5 (118)
21. Washington	14.5	32.4 (22)	17.6 (19)	-0.3 (96)
22. Iowa	13.6	30.3 (38)	17.3 (17)	0.6 (10)
23. Clemson	12.8	29.5 (48)	17.0 (16)	0.3 (41)
24. S. Carolina	12.1	29.9 (44)	18.4 (21)	0.5 (19)
25. Utah	11.9	33.0 (18)	21.1 (34)	0.0 (74)
26. Auburn	11.2	28.4 (55)	17.5 (18)	0.3 (51)
27. Louisville	11.0	30.6 (36)	19.7 (26)	0.2 (55)
28. SMU	10.9	31.9 (25)	20.7 (32)	-0.4 (104)
29. Kansas St.	10.4	33.1 (17)	23.0 (49)	0.3 (49)
30. Arizona	10.2	31.5 (29)	20.8 (33)	-0.5 (119)
31. Vanderbilt	10.0	33.4 (15)	24.1 (56)	0.6 (7)
32. Va. Tech	9.4	31.1 (34)	21.5 (38)	-0.2 (89)
33. Illinois	9.3	31.5 (28)	22.7 (44)	0.4 (30)
34. TCU	9.1	31.3 (31)	21.9 (40)	-0.2 (87)
35. Florida St.	8.8	29.7 (45)	20.6 (30)	-0.3 (94)
36. Houston	8.2	31.3 (30)	22.8 (45)	-0.4 (101)
37. Nebraska	7.7	29.2 (49)	21.5 (39)	0.1 (69)
38. Oklahoma St.	7.1	30.0 (42)	23.4 (52)	0.5 (26)
39. Boise State	6.8	29.9 (43)	22.8 (46)	-0.3 (93)
40. Virginia	6.6	26.3 (63)	19.9 (27)	0.2 (58)
41. Pittsburgh	6.5	30.2 (40)	23.8 (53)	0.1 (60)
42. Arizona St.	6.4	28.2 (57)	21.2 (37)	-0.6 (129)
43. Ga. Tech	6.0	28.8 (53)	23.4 (51)	0.6 (11)
44. Duke	5.7	32.8 (19)	27.5 (81)	0.4 (36)
45. Minnesota	5.2	25.3 (71)	19.5 (24)	-0.5 (125)
46. UCLA	5.1	29.0 (52)	23.9 (54)	0.0 (71)
47. Arkansas	5.0	34.1 (13)	29.3 (91)	0.3 (44)
48. NC State	4.9	30.7 (35)	25.3 (64)	-0.4 (103)
49. Northwestern	4.6	24.5 (78)	20.4 (28)	0.5 (29)
50. Cincinnati	4.5	29.6 (46)	25.4 (65)	0.3 (46)
51. Baylor	4.5	32.2 (23)	28.3 (87)	0.6 (2)
52. Miss. St.	3.9	30.1 (41)	26.7 (76)	0.5 (21)
53. Kentucky	3.8	25.0 (75)	21.2 (36)	0.0 (75)
54. N. Carolina	3.8	24.5 (80)	21.1 (35)	0.4 (32)
55. Maryland	3.8	26.1 (65)	22.7 (43)	0.3 (39)
56. California	3.7	29.2 (50)	25.2 (62)	-0.3 (95)
57. Kansas	3.7	29.0 (51)	25.9 (70)	0.6 (9)
58. Wake Forest	3.6	24.2 (83)	20.7 (31)	0.0 (78)
59. UNLV	2.8	31.2 (33)	28.4 (89)	0.1 (66)
60. UCF	2.3	24.5 (79)	22.1 (41)	-0.1 (83)
61. Wisconsin	1.8	20.6 (106)	18.5 (22)	-0.3 (97)
62. Rutgers	1.8	30.4 (37)	28.7 (90)	0.1 (68)
63. Navy	1.1	27.8 (58)	26.9 (77)	0.2 (56)
64. Iowa State	1.0	22.9 (94)	22.4 (42)	0.5 (27)
65. Colorado	0.9	26.3 (64)	24.9 (57)	-0.5 (116)
66. W. Virginia	0.8	26.3 (62)	25.6 (66)	0.1 (63)
67. Michigan St.	0.4	26.7 (61)	25.9 (69)	-0.4 (105)
68. New Mexico	-0.5	25.3 (68)	26.4 (73)	0.6 (13)
69. Syracuse	-0.7	23.9 (85)	25.0 (59)	0.4 (35)
70. Memphis	-1.1	28.3 (56)	30.0 (95)	0.5 (18)
71. SDSU	-1.3	23.1 (93)	25.0 (60)	0.6 (5)
72. NDSU	-1.4	24.4 (82)	25.8 (67)	0.0 (76)
73. UTSA	-1.5	29.5 (47)	31.2 (102)	0.1 (59)
74. Boston Coll.	-1.5	25.4 (67)	27.4 (80)	0.5 (28)
75. Stanford	-1.9	22.7 (97)	24.0 (55)	-0.6 (128)
76. ECU	-2.0	24.2 (84)	26.0 (71)	-0.2 (84)
77. JMU	-2.1	25.0 (76)	26.7 (75)	-0.4 (102)
78. Fresno St.	-2.3	20.6 (105)	22.9 (47)	0.0 (79)
79. Air Force	-2.4	25.3 (70)	27.7 (83)	0.1 (65)
80. USF	-2.8	27.5 (59)	30.1 (98)	-0.3 (91)
81. Miami-OH	-2.9	19.5 (111)	22.9 (48)	0.5 (16)
82. Purdue	-2.9	23.8 (86)	27.4 (79)	0.6 (4)
83. Army	-3.0	24.5 (81)	27.6 (82)	0.1 (64)
84. Hawaii	-3.9	25.3 (69)	29.8 (93)	0.6 (8)
85. Wash. St.	-5.3	19.9 (110)	25.1 (61)	-0.1 (81)
86. WKU	-5.3	25.2 (73)	31.0 (101)	0.5 (24)
87. Tulane	-5.5	22.2 (99)	28.3 (86)	0.6 (14)
88. ODU	-5.8	20.2 (109)	25.3 (63)	-0.7 (134)
89. Texas St.	-5.9	31.8 (26)	37.8 (129)	0.1 (61)
90. Troy	-6.0	23.6 (90)	29.8 (94)	0.2 (52)
91. Oregon St.	-6.3	20.9 (104)	26.5 (74)	-0.7 (137)
92. Marshall	-6.4	28.5 (54)	35.2 (124)	0.3 (48)
93. Liberty	-6.4	25.5 (66)	31.7 (104)	-0.2 (88)
94. FAU	-7.1	27.2 (60)	34.9 (120)	0.5 (20)
95. WMU	-7.2	19.0 (112)	25.8 (68)	-0.4 (106)
96. Tulsa	-7.6	23.5 (91)	31.7 (103)	0.5 (23)
97. Utah St.	-7.7	23.7 (87)	30.9 (100)	-0.5 (117)
98. J'ville State	-7.7	24.8 (77)	32.0 (107)	-0.5 (126)
99. Colorado St.	-8.3	18.8 (113)	27.0 (78)	-0.1 (82)
100. La. Tech	-8.3	22.3 (98)	30.6 (99)	0.0 (70)
101. Arkansas St.	-8.5	23.7 (88)	32.3 (111)	0.1 (62)
102. Temple	-8.7	25.2 (74)	34.3 (117)	0.3 (38)
103. Ga. Southern	-8.9	23.2 (92)	32.1 (109)	0.0 (72)
104. Louisiana	-9.1	25.2 (72)	33.9 (116)	-0.5 (115)
105. Kennesaw St.	-9.3	21.3 (101)	30.1 (97)	-0.5 (127)
106. Wyoming	-9.6	16.0 (124)	25.0 (58)	-0.6 (130)
107. UConn	-11.2	20.3 (107)	32.1 (108)	0.6 (15)
108. Toledo	-11.5	16.7 (121)	27.9 (84)	-0.4 (108)
109. North Texas	-11.8	23.6 (89)	35.2 (123)	-0.2 (90)
110. Buffalo	-11.9	15.7 (126)	28.1 (85)	0.6 (12)
111. App. St.	-12.1	21.2 (102)	33.6 (115)	0.3 (40)
112. Nevada	-12.2	17.4 (117)	30.0 (96)	0.4 (37)
113. CMU	-12.4	17.4 (116)	29.6 (92)	-0.2 (86)
114. Delaware	-13.0	22.7 (96)	35.4 (126)	-0.3 (98)
115. BGSU	-13.3	14.8 (128)	28.3 (88)	0.3 (43)
116. S. Alabama	-13.3	22.7 (95)	35.4 (125)	-0.7 (136)
117. Ohio	-13.6	13.3 (134)	26.2 (72)	-0.6 (132)
118. FIU	-13.7	21.5 (100)	35.0 (122)	-0.3 (92)
119. Coastal Caro.	-13.8	21.1 (103)	34.4 (118)	-0.5 (112)
120. Rice	-14.7	17.0 (119)	32.2 (110)	0.5 (25)
121. EMU	-15.0	17.8 (115)	32.8 (112)	0.0 (73)
122. SJSU	-15.5	18.6 (114)	33.6 (114)	-0.5 (120)
123. NMSU	-16.4	17.0 (118)	33.4 (113)	-0.1 (80)
124. UAB	-18.1	20.3 (108)	37.9 (132)	-0.5 (123)
125. NIU	-18.2	13.8 (132)	31.8 (105)	-0.2 (85)
126. Missouri St.	-18.7	16.7 (122)	34.9 (121)	-0.5 (122)
127. Akron	-19.5	13.0 (135)	31.9 (106)	-0.6 (131)
128. Kent St.	-20.1	15.9 (125)	35.5 (127)	-0.5 (121)
129. UTEP	-20.5	14.7 (129)	34.7 (119)	-0.5 (114)
130. Sac State	-22.7	15.1 (127)	37.9 (131)	0.0 (76)
131. So. Miss	-23.3	16.4 (123)	39.2 (133)	-0.4 (110)
132. UL-Monroe	-24.3	14.3 (130)	37.9 (130)	-0.7 (138)
133. Georgia St.	-25.1	17.0 (120)	41.4 (137)	-0.7 (133)
134. Ball State	-25.2	12.2 (136)	36.7 (128)	-0.7 (135)
135. MTSU	-26.0	13.7 (133)	39.9 (134)	0.2 (57)
136. Sam Houston	-26.3	14.1 (131)	40.0 (135)	-0.4 (107)
137. UMass	-30.9	9.6 (138)	40.1 (136)	-0.5 (124)
138. Charlotte	-32.4	10.4 (137)	42.3 (138)	-0.5 (111)

"""

# ---------------------------------------------------------------------------
# RECRUITING + TRANSFER PORTAL + RETURNING PRODUCTION + SP+ -- used to nudge
# a team's STARTING 2026 Elo rating. There's no way to use these as normal
# trained model features (we have no historical version of these paired with
# 2002-2025 results to learn "how much does this predict winning" from), so
# instead they're applied once, directly, as a one-time preseason Elo
# adjustment right at the 2025->2026 season transition -- exactly where they
# matter most, before any actual 2026 games have been played to build a real
# in-season rating from. SP+ in particular already incorporates recent
# recruiting, program history, returning production, and coaching-change
# effects in its own methodology, which is exactly the kind of context a
# pure results-based Elo carryover misses (see: Oklahoma State, a team whose
# last two seasons were disastrous but that projects far better than its
# trailing record alone would suggest).
# ---------------------------------------------------------------------------

# Team-name reconciliation: each source's naming vs. this pipeline's internal
# team names (which follow the training data's convention).
RECRUITING_NAME_FIX = {
    "Miami": "Miami (FL)", "UConn": "Connecticut", "UMass": "Massachusetts", "Cal": "California",
    "Sam Houston": "Sam Houston State", "Louisiana": "UL-Lafayette",
    "Louisiana Monroe": "UL-Monroe", "Louisiana-Monroe": "UL-Monroe", "South Florida": "USF",
    "UT Martin": "Tennessee-Martin", "UC Davis": "UC-Davis",
    "App State": "Appalachian State", "App. St.": "Appalachian State",
    "Miami (Ohio)": "Miami (OH)", "Miami-OH": "Miami (OH)",
    "Arizona St.": "Arizona State", "Arkansas St.": "Arkansas State",
    "BGSU": "Bowling Green", "Boston Coll.": "Boston College", "CMU": "Central Michigan",
    "Coastal Caro.": "Coastal Carolina", "Colorado St.": "Colorado State", "ECU": "East Carolina",
    "EMU": "Eastern Michigan", "FAU": "Florida Atlantic", "Florida St.": "Florida State",
    "Fresno St.": "Fresno State", "Ga. Southern": "Georgia Southern", "Ga. Tech": "Georgia Tech",
    "Georgia St.": "Georgia State", "J'ville State": "Jacksonville State", "JMU": "James Madison",
    "Kansas St.": "Kansas State", "Kennesaw St.": "Kennesaw State", "Kent St.": "Kent State",
    "La. Tech": "Louisiana Tech", "MTSU": "Middle Tennessee", "Michigan St.": "Michigan State",
    "Miss. St.": "Mississippi State", "Missouri St.": "Missouri State", "N. Carolina": "North Carolina",
    "NDSU": "North Dakota State", "NIU": "Northern Illinois", "NMSU": "New Mexico State",
    "ODU": "Old Dominion", "Oklahoma St.": "Oklahoma State", "Oregon St.": "Oregon State",
    "Penn St.": "Penn State", "S. Alabama": "South Alabama", "S. Carolina": "South Carolina",
    "SDSU": "San Diego State", "SJSU": "San Jose State", "Sac State": "Sacramento State",
    "So. Miss": "Southern Miss", "Texas St.": "Texas State", "Utah St.": "Utah State",
    "Va. Tech": "Virginia Tech", "W. Virginia": "West Virginia", "WKU": "Western Kentucky",
    "WMU": "Western Michigan", "Wash. St.": "Washington State",
}

def parse_recruiting_text(text, chunk_size, fields):
    """chunk_size: how many non-blank lines make up one team's record.
    fields: names for each position in the chunk (must include 'team' and
    'points')."""
    lines = [l.strip() for l in io.StringIO(text) if l.strip()]
    assert len(lines) % chunk_size == 0, f"{len(lines)} lines not divisible by chunk size {chunk_size}"
    rows = []
    for i in range(0, len(lines), chunk_size):
        chunk = lines[i:i + chunk_size]
        rows.append(dict(zip(fields, chunk)))
    df = pd.DataFrame(rows)
    df["team"] = df["team"].str.strip().replace(RECRUITING_NAME_FIX)
    df["points"] = df["points"].astype(float)
    return df[["team", "points"]]

def parse_returning_production_text(text):
    """CBS Sports-style: 'N. Team' then overall%, off% (rank), def% (rank)
    each on their own line."""
    lines = [l.strip() for l in io.StringIO(text) if l.strip()]
    rows = []
    rank_re = re.compile(r"^\d+\.\s+(.+)$")
    i = 0
    while i < len(lines):
        m = rank_re.match(lines[i])
        if not m:
            i += 1
            continue
        team = RECRUITING_NAME_FIX.get(m.group(1).strip(), m.group(1).strip())
        overall_pct = float(lines[i + 1].replace("%", ""))
        rows.append({"team": team, "points": overall_pct})
        i += 4
    return pd.DataFrame(rows)

def parse_sp_plus_text(text):
    """ESPN SP+ style: 'N. Team<tab>SP+<tab>Off (rank)<tab>Def (rank)<tab>ST (rank)'."""
    rows = []
    rank_re = re.compile(r"^(\d+)\.\s+(.+?)\t([\-\d.]+)\t([\-\d.]+)\s*\((\d+)\)\t([\-\d.]+)\s*\((\d+)\)\t([\-\d.]+)\s*\((\d+)\)$")
    for line in io.StringIO(text):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        m = rank_re.match(line)
        if not m:
            continue
        _, team, sp, *_ = m.groups()
        team = RECRUITING_NAME_FIX.get(team.strip(), team.strip())
        rows.append({"team": team, "points": float(sp)})
    return pd.DataFrame(rows)

def parse_coaching_hires_text(text):
    """CFN-style: 'N. Coach Name, Team' header block, with a 'CFN Grade: X'
    line somewhere in the block. Only teams that actually hired a new head
    coach appear here -- everyone else gets no entry (handled as neutral/0
    downstream), and separately, this list is used to identify which teams
    should get extra season-to-season Elo regression (see COACHING_CHANGE
    handling in the main block) since a new coach makes a team's trailing
    record much less predictive of what's coming."""
    grade_map = {"A+": 4.3, "A": 4.0, "A-": 3.7, "B+": 3.3, "B": 3.0, "B-": 2.7,
                 "C+": 2.3, "C": 2.0, "C-": 1.7, "D+": 1.3, "D": 1.0, "D-": 0.7, "F": 0.0}
    records = re.split(r"\n(?=\d+\.\s)", text.strip())
    header_re = re.compile(r"^\d+\.\s+([^,]+),\s+(.+)$")
    grade_re = re.compile(r"CFN Grade:\s*(\S+)")
    rows = []
    for rec in records:
        rec = rec.strip()
        if not rec:
            continue
        lines = rec.split("\n")
        m = header_re.match(lines[0])
        if not m:
            continue
        team = RECRUITING_NAME_FIX.get(m.group(2).strip(), m.group(2).strip())
        gm = grade_re.search(rec)
        if not gm or gm.group(1) not in grade_map:
            continue
        rows.append({"team": team, "points": grade_map[gm.group(1)]})
    return pd.DataFrame(rows)

def build_talent_scores(recruiting_text, transfer_text, returning_production_text, coaching_hires_text,
                         sp_plus_text, base_weights=None):
    """Returns (base_talent, coaching_grade_z, coaching_change_teams).

    base_talent: {team: z}, a blend of recruiting class, transfer portal,
    returning production, and SP+ (weighted 15/12/33/40 respectively) --
    SP+ gets the largest single share since it's the most comprehensive
    external power rating available (already synthesizes recruiting,
    returning production, and recent history on its own), with the other
    three keeping the same relative proportions as before to preserve
    Bill Connelly's documented emphasis on returning production. Applies
    to every team.

    coaching_grade_z: {team: z}, a SEPARATE, standalone score for just the
    ~30 teams with a new 2026 head coach, based on that hire's CFN grade.
    This is intentionally NOT folded into base_talent as one more diluted
    ingredient -- a coaching change is a distinct kind of event from
    "how good is the roster," and its effect should be able to swing a
    team's rating on its own, from a significant boost (great hire) to
    quite a drop (bad hire), independent of how the roster itself grades
    out. The caller applies this on top of base_talent specifically for
    teams in coaching_change_teams."""
    base_weights = base_weights or {"recruiting": 0.15, "transfer": 0.12, "returning_production": 0.33, "sp_plus": 0.40}

    recruiting = parse_recruiting_text(
        recruiting_text, 10,
        ["rank", "prev_rank", "team", "team_dup", "commits_str", "avg_rating", "five_star", "four_star", "three_star", "points"]
    )
    transfer = parse_recruiting_text(
        transfer_text, 9,
        ["rank", "team", "team_dup", "commits_str", "avg_rating", "five_star", "four_star", "three_star", "points"]
    )
    returning = parse_returning_production_text(returning_production_text)
    sp_plus = parse_sp_plus_text(sp_plus_text)
    coaching = parse_coaching_hires_text(coaching_hires_text)
    coaching_change_teams = set(coaching["team"])

    # Rank-based (not raw z-score) standardization: 247's transfer-portal
    # points in particular are heavily right-skewed (most teams cluster
    # near 0, a handful of huge portal hauls sit way out on the tail), and
    # a plain z-score blows up for those outliers -- LSU's raw transfer
    # z-score was +4.6 std devs, which alone would have been worth more
    # Elo than a full season of games. Percentile rank keeps every source's
    # influence bounded to roughly [-2, +2] regardless of how skewed the
    # underlying values are.
    def rank_scaled(series):
        pct = series.rank(pct=True)  # (0, 1]
        return (pct - 0.5) * 4

    base_scored = {
        "recruiting": rank_scaled(recruiting.set_index("team")["points"]),
        "transfer": rank_scaled(transfer.set_index("team")["points"]),
        "returning_production": rank_scaled(returning.set_index("team")["points"]),
        "sp_plus": rank_scaled(sp_plus.set_index("team")["points"]),
    }
    all_teams = set()
    for s in base_scored.values():
        all_teams |= set(s.index)
    base_talent = {
        team: sum(base_weights[src] * base_scored[src].get(team, 0.0) for src in base_weights)
        for team in all_teams
    }

    # Coaching grade is now NON-NEGATIVE: a new coach can only help a team's
    # rating, never hurt it below what the roster-talent blend above already
    # says. A "C" grade (replacement-level hire) or worse contributes
    # exactly 0 -- no bonus, no penalty. Only above-average grades add a
    # boost, scaling up to the maximum for an A+ hire. This avoids absurd
    # outcomes like a mediocre coaching grade dragging a team's rating so
    # low it projects to lose to an outmatched non-conference opponent.
    COACHING_GRADE_FLOOR = 2.0   # "C" -- at or below this, zero bonus
    COACHING_GRADE_CEILING = 4.3  # "A+" -- the best possible bonus
    if len(coaching):
        raw_points = coaching.set_index("team")["points"]
        coaching_grade_z = {
            team: max(0.0, (pts - COACHING_GRADE_FLOOR) / (COACHING_GRADE_CEILING - COACHING_GRADE_FLOOR))
            for team, pts in raw_points.items()
        }
    else:
        coaching_grade_z = {}

    return base_talent, coaching_grade_z, coaching_change_teams

# ---------------------------------------------------------------------------
# RUN IT
# ---------------------------------------------------------------------------

def run_setup():
    """Everything that does NOT depend on the random seed: load box scores,
    build historical Elo/features, train the model, parse 2026 recruiting/
    transfer/returning-production/SP+/coaching data into preseason Elo
    adjustments, parse the 2026 schedule (with conference realignment
    overrides applied). Returns a dict consumed by simulate_one_season()
    and finalize_median_season() -- this work happens ONCE regardless of
    how many seasons get simulated."""
    np.random.seed(RANDOM_SEED)
    print(f"Random seed set to {RANDOM_SEED} -- change RANDOM_SEED at the top of this file "
          f"to get a different random realization of the chaotic games below.\n")

    print("Building features from box scores...")
    games, season_end_elo = build_game_features()
    print(f"  {len(games)} games processed.\n")

    print("Training XGBoost score models (train<=2023, val=2024, test=2025)...")
    models = train_models(games)

    SEASON = 2025
    print(f"\n=== Elo Top 25, end of {SEASON} ===")
    print(top25(season_end_elo, SEASON).to_string(index=False))

    print(f"\n=== Projected conference championships, {SEASON} ===")
    champs = project_conference_championships(games, season_end_elo, models, SEASON)
    print(champs.to_string(index=False))

    seeds, bracket, champion, bracket_struct = project_playoff(games, season_end_elo, models, SEASON, champs)
    print(f"\n=== CFP field (seeded 1-12), {SEASON} ===")
    for s, t in seeds.items():
        print(f"  {s}. {t}")
    print(f"\n=== Playoff bracket (model-simulated) ===")
    print(bracket.to_string(index=False))
    print(f"\nProjected national champion: {champion}")

    print("\n\n############ 2026 SEASON SETUP ############")
    sched = parse_schedule_text(SCHEDULE_TEXT_2026)

    last_conf = {}
    for _, r in games.sort_values("date").iterrows():
        last_conf[r["away"]] = r["conf_away"]
        last_conf[r["home"]] = r["conf_home"]
    sched["conf_away"] = sched["away"].map(last_conf)
    sched["conf_home"] = sched["home"].map(last_conf)

    # 2026 conference realignment: some teams' historical conference (last
    # time they appeared in 2002-2025 box scores) no longer matches their
    # actual 2026 home. Override those specifically rather than trust the
    # historical lookup for them.
    CONF_OVERRIDES_2026 = {
        "Sacramento State": "mac", "North Dakota State": "mwc", "Northern Illinois": "mwc", "UTEP": "mwc",
        "Colorado State": "pac12", "Utah State": "pac12", "Fresno State": "pac12",
        "San Diego State": "pac12", "Boise State": "pac12", "Texas State": "pac12",
        "Louisiana Tech": "sun-belt",
    }
    for team, conf in CONF_OVERRIDES_2026.items():
        sched.loc[sched["away"] == team, "conf_away"] = conf
        sched.loc[sched["home"] == team, "conf_home"] = conf

    print("\nBuilding 2026 preseason talent adjustments from recruiting, transfer portal, "
          "returning production, SP+, and coaching hire grades...")
    base_talent, coaching_grade_z, coaching_change_teams = build_talent_scores(
        RECRUITING_TEXT_2026, TRANSFER_TEXT_2026, RETURNING_PRODUCTION_TEXT_2026, COACHING_HIRES_TEXT_2026, SP_PLUS_TEXT_2026)

    preseason_2026 = {team: round(z * TALENT_ELO_SCALE, 1) for team, z in base_talent.items()}
    for team, cz in coaching_grade_z.items():
        preseason_2026[team] = preseason_2026.get(team, 0.0) + round(cz * COACHING_GRADE_ELO_SCALE, 1)
    preseason_adjustments = {2026: preseason_2026}

    talent_ranked = sorted(preseason_adjustments[2026].items(), key=lambda kv: -kv[1])
    print(f"  Parsed talent scores for {len(talent_ranked)} teams. Biggest 2026 preseason Elo boosts:")
    for team, delta in talent_ranked[:5]:
        print(f"    {team}: +{delta}")
    print("  Biggest preseason Elo penalties:")
    for team, delta in talent_ranked[-5:]:
        print(f"    {team}: {delta}")

    COACHING_CHANGE_RETENTION = 0.35
    extra_regression = {2026: (coaching_change_teams, COACHING_CHANGE_RETENTION)}
    print(f"\n  {len(coaching_change_teams)} teams flagged with a 2026 coaching change "
          f"(extra Elo regression, retention={COACHING_CHANGE_RETENTION}):")
    print(f"    {sorted(coaching_change_teams)}")
    coaching_ranked = sorted(coaching_grade_z.items(), key=lambda kv: -kv[1])
    print(f"  Best coaching-grade swings: {[(t, round(z*COACHING_GRADE_ELO_SCALE,1)) for t, z in coaching_ranked[:3]]}")
    print(f"  Worst coaching-grade swings: {[(t, round(z*COACHING_GRADE_ELO_SCALE,1)) for t, z in coaching_ranked[-3:]]}")

    league_avg_pts_for = float(games["pts_for_L4_away"].mean())
    league_avg_pts_ag = float(games["pts_ag_L4_away"].mean())
    league_avg_pts_for_L8 = float(games["pts_for_L8_away"].mean())
    league_avg_pts_ag_L8 = float(games["pts_ag_L8_away"].mean())
    print(f"  Coaching-change teams' rolling scoring form will be blended toward the league "
          f"average ({league_avg_pts_for:.1f} ppg for / {league_avg_pts_ag:.1f} ppg against) "
          f"early in the season, fading out by week 4 (L4 stats) / week 8 (L8 stats).")

    raw_2026_base = load_raw()

    return {
        "games": games, "season_end_elo": season_end_elo, "models": models,
        "sched": sched, "preseason_adjustments": preseason_adjustments,
        "extra_regression": extra_regression, "coaching_change_teams": coaching_change_teams,
        "talent_ranked": talent_ranked,
        "league_avg_pts_for": league_avg_pts_for, "league_avg_pts_ag": league_avg_pts_ag,
        "league_avg_pts_for_L8": league_avg_pts_for_L8, "league_avg_pts_ag_L8": league_avg_pts_ag_L8,
        "raw_2026_base": raw_2026_base,
    }


def _reset_coaching_change_form(wk_full, coaching_change_teams, league_avg_pts_for, league_avg_pts_ag,
                                 league_avg_pts_for_L8, league_avg_pts_ag_L8):
    for side, prefix in [("away", "_away"), ("home", "_home")]:
        is_cc = wk_full[side].isin(coaching_change_teams)
        if not is_cc.any():
            continue
        gp = wk_full.loc[is_cc, f"games_played_season{prefix}"].clip(upper=8)
        w4 = (gp / 4).clip(upper=1.0)
        w8 = (gp / 8).clip(upper=1.0)
        wk_full.loc[is_cc, f"pts_for_L4{prefix}"] = w4 * wk_full.loc[is_cc, f"pts_for_L4{prefix}"] + (1 - w4) * league_avg_pts_for
        wk_full.loc[is_cc, f"pts_ag_L4{prefix}"] = w4 * wk_full.loc[is_cc, f"pts_ag_L4{prefix}"] + (1 - w4) * league_avg_pts_ag
        wk_full.loc[is_cc, f"pts_for_L8{prefix}"] = w8 * wk_full.loc[is_cc, f"pts_for_L8{prefix}"] + (1 - w8) * league_avg_pts_for_L8
        wk_full.loc[is_cc, f"pts_ag_L8{prefix}"] = w8 * wk_full.loc[is_cc, f"pts_ag_L8{prefix}"] + (1 - w8) * league_avg_pts_ag_L8
    return wk_full


def simulate_one_season(setup, seed):
    """Simulates ONE full random 2026 season (week by week, with chaos)
    plus its conference championships and playoff, using the shared
    setup. Returns per-game results and that season's specific
    conference-champion / playoff-seed / national-champion outcomes.
    Call this many times (varying seed) and feed the results into
    finalize_median_season() rather than trusting any single run --
    one seed's luck can make an otherwise-average team look like a
    disaster or a Cinderella story."""
    np.random.seed(seed)
    games = setup["games"]
    models = setup["models"]
    preseason_adjustments = setup["preseason_adjustments"]
    extra_regression = setup["extra_regression"]
    sched = setup["sched"]
    coaching_change_teams = setup["coaching_change_teams"]
    league_avg_pts_for = setup["league_avg_pts_for"]
    league_avg_pts_ag = setup["league_avg_pts_ag"]
    league_avg_pts_for_L8 = setup["league_avg_pts_for_L8"]
    league_avg_pts_ag_L8 = setup["league_avg_pts_ag_L8"]
    raw_2026 = setup["raw_2026_base"].copy()

    all_preds, games_2026_frames = [], []
    available_weeks = sorted(sched["week"].unique())

    _preseason_probe = pd.concat([raw_2026, sched.assign(
        score_home=np.nan, score_away=np.nan, game_type="regular",
        game_id=range(raw_2026["game_id"].max() + 1, raw_2026["game_id"].max() + 1 + len(sched)))],
        ignore_index=True, sort=False)
    _preseason_probe["date"] = pd.to_datetime(_preseason_probe["date"])
    _, _preseason_elo_snapshot = compute_elo(_preseason_probe, preseason_adjustments=preseason_adjustments, extra_regression=extra_regression)
    current_ranks = full_rank_lookup(_preseason_elo_snapshot[2026])

    for wk in available_weeks:
        wk_games = sched[sched["week"] == wk].copy()
        wk_games["game_id"] = range(raw_2026["game_id"].max() + 1, raw_2026["game_id"].max() + 1 + len(wk_games))
        wk_games["score_home"] = np.nan
        wk_games["score_away"] = np.nan
        wk_games["game_type"] = "regular"
        wk_games["neutral"] = wk_games["neutral"].astype(bool)
        wk_games["season"] = 2026
        combined = pd.concat([raw_2026, wk_games], ignore_index=True, sort=False)
        combined["date"] = pd.to_datetime(combined["date"])
        combined, _ = compute_elo(combined, preseason_adjustments=preseason_adjustments, extra_regression=extra_regression)
        long_df = add_rolling_form(build_long(combined))
        feat_cols = ["pts_for_L4", "pts_ag_L4", "pts_for_L8", "pts_ag_L8", "season_win_pct", "games_played_season", "rest_days"]
        away_feat = long_df[long_df["is_home"] == 0][["game_id"] + feat_cols].add_suffix("_away").rename(columns={"game_id_away": "game_id"})
        home_feat = long_df[long_df["is_home"] == 1][["game_id"] + feat_cols].add_suffix("_home").rename(columns={"game_id_home": "game_id"})
        wk_full = combined[combined["game_id"].isin(wk_games["game_id"])].merge(away_feat, on="game_id").merge(home_feat, on="game_id")
        wk_full["elo_diff"] = (wk_full["elo_home_pre"] + np.where(wk_full["neutral"], 0, HOME_FIELD_ELO)) - wk_full["elo_away_pre"]
        wk_full["is_neutral"] = wk_full["neutral"].astype(int)
        wk_full["is_postseason"] = 0
        wk_full["same_conf"] = (wk_full["conf_away"] == wk_full["conf_home"]).astype(int)
        wk_full["rank_away_live"] = wk_full["away"].map(current_ranks)
        wk_full["rank_home_live"] = wk_full["home"].map(current_ranks)
        wk_full = _reset_coaching_change_form(wk_full, coaching_change_teams, league_avg_pts_for, league_avg_pts_ag,
                                               league_avg_pts_for_L8, league_avg_pts_ag_L8)
        for c in FEATURES:
            wk_full[c] = wk_full[c].fillna(games[c].median())

        pred_home = models["score_home"].predict(wk_full[FEATURES])
        pred_away = models["score_away"].predict(wk_full[FEATURES])
        base_sigma = models.get("margin_sigma", 14.0)
        chaos = models.get("team_chaos", {})
        chaos_home = wk_full["home"].map(lambda t: chaos.get(t, 1.0)).values
        chaos_away = wk_full["away"].map(lambda t: chaos.get(t, 1.0)).values
        prob_sigma = base_sigma * np.sqrt((chaos_home ** 2 + chaos_away ** 2) / 2)
        noise_sigma = prob_sigma * CHAOS_DAMPENING
        vec_round_half_up = np.vectorize(round_half_up)
        vec_norm_cdf = np.vectorize(norm_cdf)
        disp_home, disp_away, realized_margin = realize_chaotic_score(pred_home, pred_away, noise_sigma, floor=3.0)
        score_home = vec_round_half_up(disp_home)
        score_away = vec_round_half_up(disp_away)
        score_home, score_away = break_ties_vectorized(score_home, score_away)
        wk_full["score_home"] = score_home.astype(int)
        wk_full["score_away"] = score_away.astype(int)
        wk_full["win_prob_home"] = np.round(vec_norm_cdf((pred_home - pred_away) / prob_sigma), 3)
        wk_full["win_prob_away"] = np.round(1 - wk_full["win_prob_home"], 3)
        wk_full["predicted_winner"] = np.where(score_home > score_away, wk_full["home"], wk_full["away"])
        all_preds.append(wk_full[["week", "date", "away", "home", "score_away", "score_home",
                                   "win_prob_away", "win_prob_home", "predicted_winner", "rank_away_live", "rank_home_live"]])
        games_2026_frames.append(wk_full)

        raw_2026 = pd.concat([raw_2026, wk_full[raw_2026.columns.intersection(wk_full.columns).tolist() +
                                                   ["score_home", "score_away"]].drop(columns=["score_home", "score_away"]).assign(
                                                   score_home=wk_full["score_home"], score_away=wk_full["score_away"])],
                              ignore_index=True, sort=False)
        _, season_end_elo_through_this_week = compute_elo(raw_2026, preseason_adjustments=preseason_adjustments, extra_regression=extra_regression)
        current_ranks = full_rank_lookup(season_end_elo_through_this_week[2026])

    preds_2026 = pd.concat(all_preds, ignore_index=True)
    games_2026 = pd.concat(games_2026_frames, ignore_index=True, sort=False)

    standings_2026 = compute_conference_standings(games_2026[games_2026.season == 2026])
    champs_2026 = project_conference_championships(games_2026, season_end_elo_through_this_week, models, 2026)
    season_end_elo_post_champs = dict(season_end_elo_through_this_week)
    season_end_elo_post_champs[2026] = championship_seeding_adjustment(season_end_elo_through_this_week[2026], champs_2026)
    seeds_2026, bracket_2026, champion_2026, bracket_2026_struct = project_playoff(
        games_2026, season_end_elo_post_champs, models, 2026, champs_2026)

    records = {}
    for _, g in preds_2026.iterrows():
        for team, opp_score, team_score in [(g["away"], g["score_home"], g["score_away"]), (g["home"], g["score_away"], g["score_home"])]:
            records.setdefault(team, [0, 0])
            if team_score > opp_score:
                records[team][0] += 1
            else:
                records[team][1] += 1

    return {
        "seed": seed,
        "records": records,
        "games": preds_2026[["week", "away", "home", "score_away", "score_home", "predicted_winner"]].to_dict("records"),
        "champs": champs_2026[["conference", "predicted_champion"]].set_index("conference")["predicted_champion"].to_dict(),
        "playoff_seeds": seeds_2026,
        "national_champion": champion_2026,
    }


def finalize_median_season(setup, all_results):
    """Takes N independent simulate_one_season() results, takes the MEDIAN
    score for every individual game across all of them (not team-level
    win totals -- literally every game gets its own median), then replays
    that median-based season through the same Elo/rolling-form pipeline
    and applies conference standings, championships, the championship-loss
    seeding rule, and the 12-team playoff (including the Notre Dame rule)
    exactly ONCE on top of it. This is what actually gets exported for the
    website -- a single random seed can make an average team look like a
    disaster or a Cinderella story; the median of many seeds is a far more
    honest "most likely" season."""
    games_hist = setup["games"]
    models = setup["models"]
    preseason_adjustments = setup["preseason_adjustments"]
    extra_regression = setup["extra_regression"]
    sched = setup["sched"]
    raw_2026_base = setup["raw_2026_base"]
    coaching_change_teams = setup["coaching_change_teams"]
    league_avg_pts_for = setup["league_avg_pts_for"]
    league_avg_pts_ag = setup["league_avg_pts_ag"]
    league_avg_pts_for_L8 = setup["league_avg_pts_for_L8"]
    league_avg_pts_ag_L8 = setup["league_avg_pts_ag_L8"]

    print(f"\nComputing median score per game across {len(all_results)} simulated seasons...")
    game_scores = {}
    for r in all_results:
        for g in r["games"]:
            key = (g["week"], g["away"], g["home"])
            game_scores.setdefault(key, {"away": [], "home": []})
            game_scores[key]["away"].append(g["score_away"])
            game_scores[key]["home"].append(g["score_home"])

    median_scores = {}
    for key, scores in game_scores.items():
        med_away = round_half_up(statistics.median(scores["away"]))
        med_home = round_half_up(statistics.median(scores["home"]))
        med_away, med_home = break_ties_scalar(med_away, med_home)
        median_scores[key] = (med_away, med_home)

    print("Replaying median season week-by-week for win probs + live rankings...")
    raw_2026 = raw_2026_base.copy()
    available_weeks = sorted(sched["week"].unique())

    _preseason_probe = pd.concat([raw_2026, sched.assign(
        score_home=np.nan, score_away=np.nan, game_type="regular",
        game_id=range(raw_2026["game_id"].max() + 1, raw_2026["game_id"].max() + 1 + len(sched)))],
        ignore_index=True, sort=False)
    _preseason_probe["date"] = pd.to_datetime(_preseason_probe["date"])
    _, _preseason_elo_snapshot = compute_elo(_preseason_probe, preseason_adjustments=preseason_adjustments, extra_regression=extra_regression)
    current_ranks = full_rank_lookup(_preseason_elo_snapshot[2026])

    all_preds, games_2026_frames = [], []
    top25_by_week = {}

    for wk in available_weeks:
        wk_games = sched[sched["week"] == wk].copy()
        wk_games["game_id"] = range(raw_2026["game_id"].max() + 1, raw_2026["game_id"].max() + 1 + len(wk_games))
        wk_games["score_home"] = np.nan
        wk_games["score_away"] = np.nan
        wk_games["game_type"] = "regular"
        wk_games["neutral"] = wk_games["neutral"].astype(bool)
        wk_games["season"] = 2026
        combined = pd.concat([raw_2026, wk_games], ignore_index=True, sort=False)
        combined["date"] = pd.to_datetime(combined["date"])
        combined, _ = compute_elo(combined, preseason_adjustments=preseason_adjustments, extra_regression=extra_regression)
        long_df = add_rolling_form(build_long(combined))
        feat_cols = ["pts_for_L4", "pts_ag_L4", "pts_for_L8", "pts_ag_L8", "season_win_pct", "games_played_season", "rest_days"]
        away_feat = long_df[long_df["is_home"] == 0][["game_id"] + feat_cols].add_suffix("_away").rename(columns={"game_id_away": "game_id"})
        home_feat = long_df[long_df["is_home"] == 1][["game_id"] + feat_cols].add_suffix("_home").rename(columns={"game_id_home": "game_id"})
        wk_full = combined[combined["game_id"].isin(wk_games["game_id"])].merge(away_feat, on="game_id").merge(home_feat, on="game_id")
        wk_full["elo_diff"] = (wk_full["elo_home_pre"] + np.where(wk_full["neutral"], 0, HOME_FIELD_ELO)) - wk_full["elo_away_pre"]
        wk_full["is_neutral"] = wk_full["neutral"].astype(int)
        wk_full["is_postseason"] = 0
        wk_full["same_conf"] = (wk_full["conf_away"] == wk_full["conf_home"]).astype(int)
        wk_full["rank_away_live"] = wk_full["away"].map(current_ranks)
        wk_full["rank_home_live"] = wk_full["home"].map(current_ranks)
        wk_full = _reset_coaching_change_form(wk_full, coaching_change_teams, league_avg_pts_for, league_avg_pts_ag,
                                               league_avg_pts_for_L8, league_avg_pts_ag_L8)
        for c in FEATURES:
            wk_full[c] = wk_full[c].fillna(games_hist[c].median())

        pred_home = models["score_home"].predict(wk_full[FEATURES])
        pred_away = models["score_away"].predict(wk_full[FEATURES])
        base_sigma = models.get("margin_sigma", 14.0)
        chaos = models.get("team_chaos", {})
        chaos_home = wk_full["home"].map(lambda t: chaos.get(t, 1.0)).values
        chaos_away = wk_full["away"].map(lambda t: chaos.get(t, 1.0)).values
        prob_sigma = base_sigma * np.sqrt((chaos_home ** 2 + chaos_away ** 2) / 2)
        vec_norm_cdf = np.vectorize(norm_cdf)
        wk_full["win_prob_home"] = np.round(vec_norm_cdf((pred_home - pred_away) / prob_sigma), 3)
        wk_full["win_prob_away"] = np.round(1 - wk_full["win_prob_home"], 3)

        med_home_scores, med_away_scores = [], []
        for _, row in wk_full.iterrows():
            key = (row["week"], row["away"], row["home"])
            ma, mh = median_scores[key]
            med_away_scores.append(ma)
            med_home_scores.append(mh)
        wk_full["score_home"] = med_home_scores
        wk_full["score_away"] = med_away_scores
        wk_full["predicted_winner"] = np.where(wk_full["score_home"] > wk_full["score_away"], wk_full["home"], wk_full["away"])

        all_preds.append(wk_full[["week", "date", "away", "home", "score_away", "score_home",
                                   "win_prob_away", "win_prob_home", "predicted_winner", "rank_away_live", "rank_home_live"]])
        games_2026_frames.append(wk_full)

        raw_2026 = pd.concat([raw_2026, wk_full[raw_2026.columns.intersection(wk_full.columns).tolist() +
                                                   ["score_home", "score_away"]].drop(columns=["score_home", "score_away"]).assign(
                                                   score_home=wk_full["score_home"], score_away=wk_full["score_away"])],
                              ignore_index=True, sort=False)
        _, season_end_elo_through_this_week = compute_elo(raw_2026, preseason_adjustments=preseason_adjustments, extra_regression=extra_regression)
        top25_by_week[int(wk)] = top25(season_end_elo_through_this_week, 2026)
        current_ranks = full_rank_lookup(season_end_elo_through_this_week[2026])
        print(f"  Week {int(wk)} done.")

    preds_2026 = pd.concat(all_preds, ignore_index=True)
    games_2026 = pd.concat(games_2026_frames, ignore_index=True, sort=False)
    season_end_elo_2026 = season_end_elo_through_this_week

    print("Computing conference standings + championships...")
    standings_2026 = compute_conference_standings(games_2026[games_2026.season == 2026])
    np.random.seed(RANDOM_SEED)
    champs_2026 = project_conference_championships(games_2026, season_end_elo_2026, models, 2026)
    _pre_champ_ranks = full_rank_lookup(season_end_elo_2026[2026])
    champs_2026["rank_team1"] = champs_2026["team1"].map(_pre_champ_ranks)
    champs_2026["rank_team2"] = champs_2026["team2"].map(_pre_champ_ranks)

    season_end_elo_2026_post_champs = dict(season_end_elo_2026)
    season_end_elo_2026_post_champs[2026] = championship_seeding_adjustment(season_end_elo_2026[2026], champs_2026)
    top25_by_week["post_champs"] = top25(season_end_elo_2026_post_champs, 2026)

    print("Projecting playoff...")
    seeds_2026, bracket_2026, champion_2026, bracket_2026_struct = project_playoff(
        games_2026, season_end_elo_2026_post_champs, models, 2026, champs_2026)
    print(f"National champion: {champion_2026}")

    P5_CONFS = {"acc", "big10", "big12", "sec"}
    G6_CONFS = {"aac", "cusa", "mac", "mwc", "sun-belt", "wac", "big-east", "pac12"}
    IND_P5_TEAMS = {"Notre Dame"}
    IND_G6_TEAMS = {"Army", "Massachusetts", "Connecticut", "UConn"}

    def team_tier(conf, team):
        if team in IND_P5_TEAMS:
            return "P5"
        if team in IND_G6_TEAMS:
            return "G6"
        if pd.isna(conf):
            return "FCS"
        if conf in P5_CONFS:
            return "P5"
        if conf in G6_CONFS:
            return "G6"
        return "FCS"

    df = games_2026.copy()
    df["away_tier"] = [team_tier(c, t) for c, t in zip(df["conf_away"], df["away"])]
    df["home_tier"] = [team_tier(c, t) for c, t in zip(df["conf_home"], df["home"])]
    winner_is_away = df["predicted_winner"] == df["away"]
    df["winner_tier"] = np.where(winner_is_away, df["away_tier"], df["home_tier"])
    df["loser_tier"] = np.where(winner_is_away, df["home_tier"], df["away_tier"])
    df["loser"] = np.where(winner_is_away, df["home"], df["away"])
    df["winner_rank"] = np.where(winner_is_away, df["rank_away_live"], df["rank_home_live"])
    df["loser_rank"] = np.where(winner_is_away, df["rank_home_live"], df["rank_away_live"])

    def categorize_all(w_tier, l_tier, w_rank, l_rank):
        cats = []
        if w_tier == "FCS" and l_tier in ("P5", "G6"):
            cats.append("FCS over FBS")
        if w_tier == "G6" and l_tier == "P5":
            cats.append("G6 over P5")
        l_ranked = pd.notna(l_rank) and l_rank <= 25
        w_ranked = pd.notna(w_rank) and w_rank <= 25
        if l_ranked and not w_ranked:
            cats.append("Unranked over Ranked")
        elif l_ranked and w_ranked and w_rank > l_rank:
            cats.append("Lower-ranked over Higher-ranked")
        return cats

    upset_rows = []
    for _, r in df.iterrows():
        for cat in categorize_all(r["winner_tier"], r["loser_tier"], r["winner_rank"], r["loser_rank"]):
            upset_rows.append({
                "category": cat, "week": r["week"], "date": r["date"],
                "away": r["away"], "home": r["home"],
                "score_away": r["score_away"], "score_home": r["score_home"],
                "predicted_winner": r["predicted_winner"], "loser": r["loser"],
                "winner_rank": int(r["winner_rank"]) if pd.notna(r["winner_rank"]) else None,
                "loser_rank": int(r["loser_rank"]) if pd.notna(r["loser_rank"]) else None,
            })
    upsets_2026 = pd.DataFrame(upset_rows)
    if not upsets_2026.empty:
        upsets_2026 = upsets_2026.sort_values(["week", "date"]).reset_index(drop=True)
    print(f"Found {len(upsets_2026)} upsets.")

    return {
        "preds_2026": preds_2026, "games_2026": games_2026, "season_end_elo_2026": season_end_elo_2026,
        "standings_2026": standings_2026, "champs_2026": champs_2026,
        "seeds_2026": seeds_2026, "bracket_2026": bracket_2026, "champion_2026": champion_2026,
        "bracket_2026_struct": bracket_2026_struct, "upsets_2026": upsets_2026,
        "top25_by_week": top25_by_week, "available_weeks": available_weeks,
    }


def team_report(team, preds):
    """Record, projected W/L, and score for every game a team has in the
    parsed schedule. Raises a helpful error if the name doesn't match
    (case-sensitive, must match the schedule's spelling)."""
    rows = preds[(preds["away"] == team) | (preds["home"] == team)].sort_values(["week", "date"]).copy()
    if rows.empty:
        all_teams = sorted(set(preds["away"]) | set(preds["home"]))
        close = [t for t in all_teams if team.lower() in t.lower()]
        hint = f" Did you mean: {', '.join(close[:5])}?" if close else ""
        raise ValueError(f"No games found for '{team}'.{hint}")
    out, wins, losses = [], 0, 0
    for _, g in rows.iterrows():
        is_home = g["home"] == team
        opp = g["away"] if is_home else g["home"]
        team_score = g["score_home"] if is_home else g["score_away"]
        opp_score = g["score_away"] if is_home else g["score_home"]
        team_win_prob = g["win_prob_home"] if is_home else g["win_prob_away"]
        win = team_score > opp_score
        wins += int(win); losses += int(not win)
        out.append({
            "week": g["week"], "date": g["date"].date() if hasattr(g["date"], "date") else g["date"],
            "opponent": opp, "site": "vs" if is_home else "at",
            "pred_score_team": team_score, "pred_score_opp": opp_score,
            "win_prob": team_win_prob,
            "result": "W" if win else "L", "record_after": f"{wins}-{losses}",
        })
    return pd.DataFrame(out)


if __name__ == "__main__":
    import sys
    import json
    import datetime

    N_SIMULATIONS = 20  # how many independent random seasons to simulate and take the median of

    setup = run_setup()

    print(f"\n\n############ SIMULATING {N_SIMULATIONS} INDEPENDENT SEASONS ############")
    all_results = []
    for seed in range(N_SIMULATIONS):
        print(f"\n--- Season {seed + 1}/{N_SIMULATIONS} (seed={seed}) ---")
        result = simulate_one_season(setup, seed)
        all_results.append(result)
        print(f"  Champion: {result['national_champion']}")

    print(f"\n\n############ BUILDING MEDIAN SEASON FROM {N_SIMULATIONS} RUNS ############")
    final = finalize_median_season(setup, all_results)

    preds_2026 = final["preds_2026"]
    print(f"\n=== 2026 Conference Standings (median season) ===")
    for conf, grp in final["standings_2026"].groupby("conference"):
        print(f"\n  {conf}")
        for _, r in grp.sort_values("conf_rank").iterrows():
            print(f"    {r['conf_rank']}. {r['team']}: conf {r['conf_wins']}-{r['conf_losses']}, "
                  f"overall {r['overall_wins']}-{r['overall_losses']}")
    final["standings_2026"].to_csv("conference_standings_2026.csv", index=False)

    print(f"\n=== Projected 2026 conference championships (median season) ===")
    print(final["champs_2026"].to_string(index=False))
    final["champs_2026"].to_csv("champs_2026.csv", index=False)

    print(f"\n=== 2026 CFP field (seeded 1-12), median season ===")
    for s, t in final["seeds_2026"].items():
        print(f"  {s}. {t}")
    print(f"\n=== 2026 playoff bracket (median season) ===")
    print(final["bracket_2026"].to_string(index=False))
    print(f"\n2026 projected national champion (median season): {final['champion_2026']}")
    final["bracket_2026"].to_csv("bracket_2026.csv", index=False)

    print(f"\n=== Upset Alert (median season), by week ===")
    for wk, grp in final["upsets_2026"].groupby("week"):
        print(f"\n  Week {int(wk)}:")
        print(grp[["category", "away", "home", "score_away", "score_home", "predicted_winner"]].to_string(index=False))
    final["upsets_2026"].to_csv("upsets_2026.csv", index=False)

    requested_teams = sys.argv[1:]
    if not requested_teams:
        example_teams = sorted(set(preds_2026["away"]) | set(preds_2026["home"]))[:5]
        print("\n=== Team report ===")
        print("No team name given. Usage:  python3 cfb_pipeline.py \"Team Name\" [\"Another Team\" ...]")
        print(f"Example valid names in this schedule: {example_teams}")
    else:
        for team in requested_teams:
            print(f"\n=== {team}: projected 2026 season (median of {N_SIMULATIONS} sims) ===")
            try:
                rep = team_report(team, preds_2026)
                print(rep.to_string(index=False))
                rep.to_csv(f"team_report_{team.replace(' ', '_').replace('&', 'and')}.csv", index=False)
            except ValueError as e:
                print(f"  {e}")

    def df_records(d):
        return json.loads(d.to_json(orient="records", date_format="iso"))

    site_data = {
        "generated_at": datetime.datetime.now().isoformat(),
        "weeks_included": [int(w) for w in final["available_weeks"]],
        "model_accuracy": {
            "winner_accuracy": setup["models"]["winner_accuracy"],
            "mae_home": setup["models"]["mae_home"],
            "mae_away": setup["models"]["mae_away"],
            "margin_sigma": round(setup["models"]["margin_sigma"], 2),
            "random_seed": f"median of {N_SIMULATIONS} seeds (0-{N_SIMULATIONS - 1})",
        },
        "top25": df_records(top25(final["season_end_elo_2026"], 2026)),
        "top25_by_week": {str(wk): df_records(d) for wk, d in final["top25_by_week"].items()},
        "conference_standings": df_records(final["standings_2026"]),
        "conference_championships": df_records(final["champs_2026"]),
        "playoff_seeds": {str(k): v for k, v in final["seeds_2026"].items()},
        "playoff_bracket": final["bracket_2026_struct"],
        "national_champion": final["champion_2026"],
        "upsets": df_records(final["upsets_2026"]),
        "games": df_records(preds_2026),
        "talent_2026": [{"team": t, "elo_adjustment": d} for t, d in setup["talent_ranked"]],
    }
    with open("site_data.json", "w") as f:
        json.dump(site_data, f)
    print(f"\nWrote site_data.json ({len(site_data['games'])} games, {len(site_data['top25'])} ranked teams, "
          f"median of {N_SIMULATIONS} simulated seasons).")
