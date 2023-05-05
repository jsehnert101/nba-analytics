# %%
import numpy as np
import pandas as pd
pd.options.display.max_columns = 50

from data.data_wrangler.teams.nba_teams import NBATeams

from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, playbyplay, leaguegamefinder, HustleStatsBoxScore
from nba_api.stats.library.parameters import SeasonAll, SeasonType

from sklearn.ensemble import RandomForestClassifier

# %%
# Get Team IDs to access game stats
nba_teams = NBATeams()
team_ids = nba_teams.get_all_teams_id_dict(load=False, save=False)
team_id = team_ids["CHI"]
df_team_games = nba_teams.get_team_games(team_id=team_id)
df_team_games = df_team_games.set_index("game_date").sort_index()
df_team_games["win"] = df_team_games.wl.replace({"W": True, "L": False})
df_team_games

# %%
# Save team statistics to raw data folder


# %%
# Get team stats at any point in time

def get_cumulative_team_stats()

#%% Access a team's cumulative season at any point in time


# %% Predict whether or not Chicago Bulls will win a game
# Use only Chicago team statistics


# %%

game_finder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_ids["CHI"], season_type_nullable=SeasonType.regular, timeout=10)
df = game_finder.get_data_frames()[0]
df.columns = df.columns.str.lower()

df

# %%
teams.get_teams()
# %%
