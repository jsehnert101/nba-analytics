# %%
# Imports
from data.teams.franchise_history import FranchiseHistory
from tqdm import tqdm

# from data.teams.team_game_data import TeamGameData
from data.teams.team_data import TeamData
from models.stats import Stats
from features.teams.team_stats import TeamStats

# from data.utils.data import *
import numpy as np
import pandas as pd
from tqdm import tqdm
import requests
import time
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

from typing import Union, Callable, Dict, List, Tuple
import pickle
from nba_api.stats.static import teams

# %%
from data.teams.wrangle_team_data import TeamGameDataWrangler

team_data_wrangler = TeamGameDataWrangler()
team_data_wrangler.wrangle_team_data()
team_data_wrangler.wrangle_team_game_data()  # TODO: pass paths to saver() and decorate

# %%
team_data_wrangler.wrangle_team_game_data()

# %%
from data.teams.team_game_data import TeamGameData
import time

team_game_data = TeamGameData()
start = time.time()
df_team_games = team_game_data.get_all_data()
end = time.time()
print(f"That took {end-start:.6f}s!")
df_team_games

# %%
# Cleaning to be performed as features

"""team_games["REST_DAYS"] = (
            team_games.groupby("SEASON_ID")["GAME_DATE"].diff().dt.days.astype(float)  # type: ignore
        )
if team_games.isna().sum().sum() > 0:
        self._inpute_team_game_data(team_games)


    """
