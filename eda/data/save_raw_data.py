# %%
import time
import numpy as np
import pandas as pd
pd.options.display.max_columns = 50

from data.data_wrangler.teams.nba_teams import NBATeamData

from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, playbyplay, leaguegamefinder, HustleStatsBoxScore, BoxScoreAdvancedV2
from nba_api.stats.library.parameters import SeasonAll, SeasonType, SeasonID

from nba_api.stats.endpoints import FranchiseHistory

from tqdm import tqdm

# %%
df_franchise_hist = FranchiseHistory().get_data_frames()[0]
df_franchise_hist # Need to add all team & city names to team_id_dict
df_franchise_hist

# %%
# Get Team IDs to access game stats
nba_teams = NBATeamData()
team_ids = nba_teams.get_all_teams_id_dict(load=False, save=False)
team_id = team_ids["CHI"]
df_team_games = nba_teams.load_team_games(team_id=team_id, save=False)
df_team_games = df_team_games.set_index("GAME_DATE").sort_index()
df_team_games["WIN"] = df_team_games.WL.replace({"W": True, "L": False})
df_team_games["HOME"] = df_team_games.MATCHUP.str.split(expand=True).iloc[:,1].equals("vs.")
#df_team_games["OPP"] = df_team_games.MATCHUP.str.split(expand=True).iloc[:,-1]
df_team_games

# %%
df_team_games["OPP_ID"] = df_team_games.OPP.map(team_ids).fillna(0).astype(int)
df_team_games

# %%
# Identify potential missing team abbreviations
#missing_abbrevs = [name for name in df_team_games.OPP.unique() if name not in team_ids.keys()]

#abbreviation_map = {"NJN": "New Jersey", "PHL": "Philadelphia", "KCK": "Kansas City", "SEA": "Seattle", "GOS":}

# %%
# Get opponent info
df_opp = nba_teams.load_team_games(team_id=team_ids["MIL"], save=False).sort_values("GAME_DATE").reset_index(drop=True)
df_opp

# %%
# Try merging the two dataframes
df_team_games.merge(df_opp, left_on=["GAME_ID", "SEASON_ID"], right_on=["GAME_ID","SEASON_ID"], how="inner")


# %%
# Add opponent info to compute some advanced stats
df = df_team_games.reset_index().copy()
df["opponent_name"] = df.matchup.str.split(" ", expand=True).iloc[:,-1]
df




# %%
# Collect advanced stats
game_id = df.loc[df.shape[0]-10,"game_id"]
df_advanced_stats = BoxScoreAdvancedV2(game_id=game_id).get_data_frames()[0]


# %%
game_date = "2019-01-11"
season_id = df_team_games.loc[game_date,"season_id"]
df_season = df_team_games.loc[df_team_games.season_id == season_id, :].sort_index().loc[:game_date, :].iloc[:-1,:]
df_season

# %%
# Get season stats at any point in time
def get_cum_stats(df: pd.DataFrame, game_date: str):
    season_id = df.loc[game_date,"season_id"]
    df_season = df.loc[df.season_id == season_id, :].loc[:game_date, :].iloc[:-1,:]
    df.


# %%
nba_teams.team_metadata

# %%
dir = "../../../data/raw/NBA/Teams/GameData/RegularSeason/"
for id in tqdm(np.unique(list(team_ids.values()))):
    team_games = nba_teams.get_team_games(team_id=id)
    team_games["game_date"] = pd.to_datetime(team_games.game_date)
    team_games.to_parquet(f"{dir}{id}.parquet")



# %%
from features.stats import NBATeamStats

class NBADataLoader():
    
    def __init__(self):
        self.dir = "../../../data/raw/NBA/Teams/GameData/"
        self._nba_stats = NBATeamStats()
    
    def save_nba_team_game_data(self, team_id: int, return_df: bool = False) -> None:
        team_games = nba_teams.get_team_games(team_id=id)
        team_games["game_date"] = pd.to_datetime(team_games.game_date)
        team_games = self._nba_stats.add_advanced_stats(df=team_games)
        team_games.to_parquet(f"{self.dir}{id}.parquet")
        if return_df:
            return team_games
        
    def save_multiple_nba_team_game_data(self, team_ids: list) -> None:
        for id in tqdm(team_ids):
            team_games = nba_teams.get_team_games(team_id=id)
            team_games["game_date"] = pd.to_datetime(team_games.game_date)
            team_games = self._nba_stats.add_advanced_stats(df=team_games)
            team_games.to_parquet(f"{self.dir}{id}.parquet")
    
    def load_nba_team_data(self, team_id: int) -> pd.DataFrame:
        return pd.read_parquet(f"{self.dir}{team_id}.parquet")


# %%
nba_teams = NBATeamData()
loader = NBADataLoader()
df_chi = loader.save_nba_team_game_data(team_id=nba_teams.get_team_id("CHI"), return_df = True)
df_chi

# %%


    

        