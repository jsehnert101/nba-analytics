# %%
# Imports
import time
import pandas as pd
from data.loader import TeamGameDataLoader
from features.team_stats import TeamStats

# %%
# Create data loader
loader = TeamGameDataLoader()
team_id_map = loader.load_team_id_map()
team_ids = loader.load_team_ids()
chi_id = team_id_map["Bulls"]
# %%
start = time.time()
df_all_games = loader.load_all_team_games(add_stats=True, max_workers=12)
end = time.time()
print(f"Time elapsed: {end - start} seconds")
df_all_games
# %%
# Append statistics to team games
from features.team_stats import TeamStats

team_stats = TeamStats()
team_stats

# %%
prev_cols = [
    "SEASON_ID",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "TEAM_NAME",
    "GAME_ID",
    "GAME_DATE",
    "MATCHUP",
    "WL",
    "MP",
    "PTS",
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3M",
    "FG3A",
    "FG3_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "WIN",
    "HOME",
    "FG2A",
    "FG2M",
    "2PAr",
    "3PAr",
    "eFG_PCT",
    "TS_PCT",
    "TOV_PCT",
    "SHOOTING_FACTOR",
    "TOV_FACTOR",
    "FT_FACTOR",
]


# %%
# Retrieve all games for a given team
df_team_games = pd.concat(
    [
        loader.load_team_games(
            team_id=team_id,
            season_type_nullable="Regular Season",
            max_workers=len(team_ids),
        )
        for team_id in team_ids
    ]
)
df_team_games

# %%
from features.team_stats import TeamStats

# %%
# TODO: add team stats during retrieval using multiprocessing

# %%
ids = [team_id_map[team] for team in ["Bulls", "Warriors", "Rockets"]]
ids
# %%
import pandas as pd

dfs = []
for team_id in ids:
    dfs.append(
        loader.load_team_games(team_id=team_id, season_type_nullable="Regular Season")
    )
df = pd.concat(dfs)
df
# %%
import os
import numpy as np

np.unique(os.listdir("../../data/external/processed/teams/games/RegularSeason/"))
# %%
from nba_api.stats.library.parameters import SeasonType, Season

Season
# %%
from data.teams.franchise_history import FranchiseHistory

f = FranchiseHistory()
f.active_teams_franchise_history
# %%
