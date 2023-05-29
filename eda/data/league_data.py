# %%
# Imports
import pandas as pd
from data.loader import TeamGameDataLoader
from features.team_stats import TeamStats

# %%
# Wrangle + merge all team data from 1985 onward
stats = TeamStats()
team_game_data_loader = TeamGameDataLoader()
team_id_map = team_game_data_loader.load_team_id_map()
df_team_games = team_game_data_loader.load_all_team_games()
df_team_games = (
    df_team_games.set_index(["SEASON_ID", "GAME_DATE"])
    .sort_index()
    .loc[pd.IndexSlice["21985":, :], :]
)
df_team_games

# %%
# Isolate a single season of data to make life easier
df_season = (
    df_team_games.loc[pd.IndexSlice["22022", :], :]
    .reset_index()
    .set_index(["TEAM_ID", "GAME_DATE"])
    .drop(columns=["SEASON_ID"])
    .copy()
)
df_season

# %%
# Isolate single team to make life easier
df_team = (
    df_season.loc[pd.IndexSlice[team_id_map["CHI"], :], :]
    .reset_index(level=0, drop=True)
    .copy()
)
df_team

# %%
from utils.env import get_api_key()

# %%
# Determine how to aggregate stats
# Which stats do we need to accumulate? Which stats are required to compute?
# Which stats should we average?

cum_stats = []
avg_stats = stats.basic_box_score_percentages + []

# %%
# Incorporate the above aggregations into data loader



# %%
# Aggregate team stats using expanding window
agg_stats = stats.required_stat_params + stats.basic_box_score_stats
agg_stats

# %%


df_tot = df_team.loc[:, stats.required_stat_params].expanding().sum()
team_game_data_loader.add_independent_team_stats(df_tot)


# %%
# Aggregate team stats by year
df_teams = df_team_games.groupby("SEASON_ID").agg(
    {
        "PTS": "sum",
        "FGA": "sum",
        "FGM": "sum",
        "FTA": "sum",
        "OREB": "sum",
    }
)

# %%
# Aggregate league stats by year
