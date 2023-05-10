# %%
# Imports
from data.teams.franchise_history import FranchiseHistory
from tqdm import tqdm
from data.teams.team_data import TeamData
from data.stats import Stats
from data.teams.team_stats import TeamStats
from utils.data import *
import numpy as np
import pandas as pd
from tqdm import tqdm
import requests
import time
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

from typing import Union, Callable
from nba_api.stats.static import teams


# %%
# Load all team game data
folder: str = "../../../data/processed/teams/"
team_data = TeamData(folder=folder, save=False, load=True)
df_team_games = team_data.get_multiple_teams_games().rename(columns={"MIN": "MP"})
df_team_games.head()

# %%
# Add stats to our df
stats = Stats()
print(stats.required_stat_params)
team_stats = TeamStats()
print(team_stats.required_stat_params)
df_team_games_dict = df_team_games.loc[
    :, df_team_games.columns.isin(team_stats.required_stat_params)
].to_dict(
    orient="list"
)  # type: ignore
for stat_name, stat_func in tqdm(team_stats.independent_stat_method_map.items()):
    df_team_games[stat_name] = stat_func(**df_team_games_dict)
df_team_games.head()

# %%
# Append opponent statistics and calculate dependent stats
df_team_games_merged = pd.concat(
    [
        df_team_games.groupby(
            ["SEASON_ID", "GAME_ID", "GAME_DATE", "HOME", "TEAM_ID"], sort=True
        )
        .first()
        .reset_index(),
        df_team_games.groupby(["SEASON_ID", "GAME_ID", "GAME_DATE", "HOME", "TEAM_ID"])
        .first()
        .sort_index(level=[0, 1, 2, 3], ascending=[True, True, True, False])
        .add_prefix("OPP_")
        .reset_index(),
    ],
    axis=1,
)
df_team_games_merged_dict = df_team_games_merged.loc[:, df_team_games_merged.columns.isin(team_stats.required_stat_params)].to_dict(orient="list")  # type: ignore
for stat_name, stat_func in tqdm(team_stats.dependent_stat_method_map.items()):
    df_team_games_merged[stat_name] = stat_func(**df_team_games_merged_dict)
df_team_games_merged.head()

# %%
# Merge data on itself once again to access opponent dependent stats
# TODO: See above
# TODO: manually fill in missing data


# %%
# Get team + opponent stats
from nba_api.stats.endpoints import cumestatsteam

cumestatsteam.CumeStatsTeam(
    league_id="00",
    team_id=team_data.team_ids["CHI"],
    season=21983,
    season_type_all_star="Regular Season",
    game_ids="0022201223",
).get_data_frames()[1]


# %%
team_data.team_ids["CHI"]

# %%
from nba_api.stats.endpoints import gamerotation

gamerotation.GameRotation(league_id="00", game_id="0022000001").get_data_frames()[1]

# %%
df_chi = df_team_games.loc[df_team_games.TEAM_ID == team_data.team_ids["CHI"], :].copy()
df_chi


# %%
# Compute cumulative statistics - rolling + expanding
win_size = 5
cum_cols = ["PTS", "AST", "FTA", "TOV"]
expanding_cols = []
pct_cols = ["TS", "eFG", "FG3_PCT", "OffRtg", "PACE"]


def get_rolling_stat(
    df_games: pd.DataFrame,
    col: str,
    win_size: int,
    agg_func: Union[str, Callable] = "mean",
) -> pd.DataFrame:
    return (
        df_games.groupby(["TEAM_ID", "SEASON_ID"])[col]
        .rolling(win_size)
        .agg(agg_func)
        .values
    )


def get_expanding_stat(
    df_games: pd.DataFrame, col: str, agg_func: Union[str, Callable] = "mean"
) -> pd.DataFrame:
    return (
        df_games.groupby(["TEAM_ID", "SEASON_ID"])[col].expanding().agg(agg_func).values
    )


for col in cum_cols:
    df_team_games[f"{col}_5"] = get_rolling_stat(df_team_games, col, win_size)
    df_team_games[f"{col}_exp"] = get_expanding_stat(df_team_games, col)

df_team_games

# %%
# Compute pct cols

from data.stats import Stats

stats = Stats()

## Identify all columns we need to accumulate
agg_stats = []
for stat in pct_cols:
    try:
        agg_stats.append(stats.basic_stat_requirement_map[stat])
    except Exception:
        try:
            agg_stats.append(stats.advanced_stat_requirement_map[stat])
        except Exception:
            print("Stat not found: ", stat)
agg_stats = np.unique(agg_stats)

temp_df_roll = df_team_games.loc[
    :, df_team_games.columns.str.contains(f"_{win_size}")
].copy(deep=True)
temp_df_exp = df_team_games.loc[:, df_team_games.columns.str.contains(f"_exp")].copy(
    deep=True
)
helper_agg_stats = []
for agg_stat in agg_stats:
    if agg_stat not in temp_df.columns:
        temp_df[f"{agg_stat}_{win_size}_agg"] = get_rolling_stat(
            temp_df, agg_stat, win_size, "sum"
        )
        temp_df[f"{agg_stat}_cum_agg"] = get_expanding_stat(temp_df, agg_stat, "sum")

for pct_col in pct_cols:
    temp_df[f"{pct_col}_{win_size}"] = get_rolling_stat(
        temp_df, pct_col, win_size, stats._advanced_stat_func_map[pct_col]
    )


# %%
# Compute differenced stats: home - away
# REST_DAYS (rest advantage)
# PLUS_MINUS / point differential


# %%
from sklearn.feature_selection import mutual_info_classif, chi2

mi_ftm = mutual_info_classif(df_team_games[["FTM"]], df_team_games["WL"])
mi_fta = mutual_info_classif(df_team_games[["FTA"]], df_team_games["WL"])
print(mi_ftm, mi_fta)

chi2_ftm = chi2(df_team_games[["FTM"]], df_team_games["WL"])
chi2_fta = chi2(df_team_games[["FTA"]], df_team_games["WL"])
print(chi2_ftm, chi2_fta)


# %%
# Add rolling / expanding statistics
win_size = 10
roll_cols = [
    "PTS",
]
df_team_games.groupby(["TEAM_ID", "SEASON_ID"])

# %%


def get_all_nba_games() -> pd.DataFrame:
    drop_cols = ["SEASON_ID", "TEAM_NAME", "MATCHUP"]
    nba_team_ids = np.unique(list(team_data.team_ids.values()))
    home_dfs, away_dfs = [], []
    for team_id in tqdm(nba_team_ids):
        team_games = team_data.get_team_games(
            team_id=team_id
        )  # TODO: Compute rolling statistics; incorporate player stats
        home_games = (
            team_games.loc[team_games.HOME, :]
            .set_index("GAME_ID")
            .drop(columns=drop_cols)
        )
        home_games = home_games.add_suffix("_HOME")
        home_dfs.append(home_games)
        away_games = (
            team_games.loc[~team_games.HOME, :]
            .set_index("GAME_ID")
            .drop(columns=drop_cols)
        )
        away_games = away_games.add_suffix("_AWAY")
        away_dfs.append(away_games)
    return pd.concat([pd.concat(home_dfs), pd.concat(away_dfs)], axis=1)


df_games = get_all_nba_games()
df_games

# %%
df_games.to_parquet("../../../data/interim/teams/games/all_team_games.parquet")


# %%
# Wrangle a given team's stats
team_data = TeamData()
nba_team_ids = np.unique(list(team_data.team_ids.values()))
drop_cols = ["SEASON_ID", "TEAM_NAME", "MATCHUP"]
home_dfs, away_dfs = [], []
for team_id in tqdm(nba_team_ids):
    team_games = team_data.load_team_games(
        team_id=team_id
    )  # TODO: Compute rolling statistics; incorporate player stats
    home_games = (
        team_games.loc[team_games.HOME, :].set_index("GAME_ID").drop(columns=drop_cols)
    )
    home_games = home_games.add_suffix("_HOME")
    home_dfs.append(home_games)
    away_games = (
        team_games.loc[~team_games.HOME, :].set_index("GAME_ID").drop(columns=drop_cols)
    )
    away_games = away_games.add_suffix("_AWAY")
    away_dfs.append(away_games)

# %%
all_home_games = pd.concat(home_dfs)
all_away_games = pd.concat(away_dfs)
all_away_games

# %%
import pandas as pd

pd.concat(home_dfs)


# %%
chi_home_games = chi_games.loc[chi_games.HOME, :].set_index("GAME_ID")
chi_home_games.drop(columns=["SEASON_ID", "TEAM_NAME", "MATCHUP"], inplace=True)
chi_home_games.add_suffix("_HOME")
chi_away_games = chi_games.loc[~chi_games.HOME, :].set_index("GAME_ID")
chi_away_games.drop(columns=["SEASON_ID", "TEAM_NAME", "MATCHUP"], inplace=True)
chi_away_games.add_suffix("_AWAY")


# %%
chi_games.groupby("SEASON_ID")

# %%
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.library.parameters import SeasonType

df_chi = leaguegamefinder.LeagueGameFinder(
    team_id_nullable=chi_id,
    season_type_nullable=SeasonType.regular,
    timeout=10,
    league_id_nullable="00",
).get_data_frames()[0]
df_chi

# %%
from nba_api.stats.endpoints import cumestatsteamgames, cumestatsteam, gamerotation

gamerotation.GameRotation(chi_games.GAME_ID.values[0], league_id="00").get_data_frames()


# %%
# Get regular season schedule
import json
import pandas as pd

team_games = cumestatsteamgames.CumeStatsTeamGames(
    league_id="00",
    season="2021",
    season_type_all_star=SeasonType.regular,
    team_id=chi_id,
).get_normalized_json()

pd.DataFrame(json.loads(team_games)["CumeStatsTeamGames"])

# %%
from nba_api.stats.static import teams, players

pd.DataFrame(teams.get_teams())

# %%
# Goal: Predict whether or not the home team will win

# %%
from nba_api.stats.library.parameters import SeasonType, SeasonAll

d = cumestatsteam.CumeStatsTeam(
    league_id="00",
    season="2022-23",
    season_type_all_star=SeasonType.regular,
    team_id=chi_id,
    game_ids=chi_games.GAME_ID.values[:100],
).get_normalized_dict()
d.keys()
# %%
cumestatsteamgames.CumeStatsTeamGames(
    league_id="00",
    season="2021",
    season_type_all_star=SeasonType.regular,
    team_id=chi_id,
).get_data_frames()

# %%
from nba_api.stats.endpoints import gamerotation

temp = gamerotation.GameRotation(game_id=1520900051, league_id="00").get_data_frames()


# %%
def get_game_player_stats():
    from npa_api.stats.endpoints import gamerotation
    from nba_api.stats.library.parameters import LeagueID

    def get_game_player_stats(game_id: str, league_id: str = "00") -> pd.DataFrame:
        """Retrieves player stats for a given game ID.

        Args:
            game_id (str): game ID to retrieve stats for
            league_id (str, optional): league ID. Defaults to "00".

        Returns:
            pd.DataFrame: DataFrame of player stats for a given game ID
        """
        return gamerotation(
            game_id=game_id, league_id=LeagueID
        ).get_data_frames()  # 0: away; 1: home


# %%
franchise_hist = FranchiseHistory()
franchise_hist.active_teams_franchise_history

# %%
df_franchise_hist = NBAFranchiseHistory().get_data_frames()[0]
df_franchise_hist.columns = df_franchise_hist.columns.str.lower()
# %%
df_franchise_hist.groupby("team_id").agg(
    {
        "team_city": "first",
        "team_name": "first",
        "start_year": "min",
        "end_year": "max",
        "years": "sum",
        "games": "sum",
        "wins": "sum",
        "losses": "sum",
        "win_pct": "first",
        "po_appearances": "sum",
        "div_titles": "sum",
        "conf_titles": "sum",
        "league_titles": "sum",
    }
)
# %%
