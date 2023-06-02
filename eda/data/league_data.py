# %%
# Imports
import pandas as pd
from pandarallel import pandarallel
from data.loader import TeamGameDataLoader
from features.stats import TeamStats

pandarallel.initialize(progress_bar=False, verbose=False, nb_workers=5)
loader = TeamGameDataLoader()
stats = TeamStats()


# %%
# Wrangle + merge all team data from 1985 onward
team_id_map = loader.load_team_id_map()
exp_games, exp_cum_stats = loader.load_all_team_games(
    add_stats=True, stat_agg_type="expanding"
)
exp_games

# %%
exp_cum_stats

# %%
roll_games, roll_cum_stats = loader.load_all_team_games(
    add_stats=True, stat_agg_type="rolling"
)
roll_games

# %%
roll_cum_stats

# %%
raw = loader.load_all_team_games(add_stats=False)
merged = (
    raw.set_index(["SEASON_ID", "TEAM_ID"])
    .loc[
        :,
        [
            "GAME_ID",
            "GAME_DATE",
            "TEAM_ABBREVIATION",
            "TEAM_NAME",
            "MATCHUP",
            "WL",
            "WIN",
            "HOME",
            "MP",
        ],
    ]
    .merge(
        df_exp_team_games.set_index(["SEASON_ID", "TEAM_ID"]),
        how="left",
        left_index=True,
        right_index=True,
    )
    .reset_index()
)

# %%
# Track down missing stats
missing_stats = {
    "FG2A",
    "FG2M",
    "MAJOR_POSS",
    "MINOR_POSS",
    "OPP_FG2A",
    "OPP_FG2M",
    "OPP_MAJOR_POSS",
    "OPP_MINOR_POSS",
}


# %%
raw["FG2A"] = stats.two_point_attempts(FGA=raw.FGA, FG3A=raw.FG3A)
raw["FG2M"] = stats.two_point_makes(FGM=raw.FGM, FG3M=raw.FG3M)
e = (
    raw.sort_values("GAME_DATE")
    .groupby(["SEASON_ID", "TEAM_ID"], group_keys=True)[stats.basic_box_score_stats]
    .parallel_apply(lambda x: x.rolling(10).mean().shift())
)

# %%
e_r = (
    raw.sort_values("GAME_DATE")
    .groupby(["SEASON_ID", "TEAM_ID"], group_keys=True)[stats.basic_box_score_stats]
    .parallel_apply(lambda x: x.rolling(window=10, closed="left").mean())
)

# %%
e_r_r = (
    raw.sort_values("GAME_DATE")
    .groupby(["SEASON_ID", "TEAM_ID"], group_keys=True)[stats.basic_box_score_stats]
    .rolling(window=10, closed="left")
    .mean()
)
e_r_r


# %%
import time

times = []
for i in range(1, 11):
    pandarallel.initialize(progress_bar=True, verbose=False, nb_workers=i)
    raw = loader.load_all_team_games(add_stats=False)
    raw["FG2A"] = stats.two_point_attempts(FGA=raw.FGA, FG3A=raw.FG3A)
    raw["FG2M"] = stats.two_point_makes(FGM=raw.FGM, FG3M=raw.FG3M)
    start = time.time()
    e = (
        raw.sort_values("GAME_DATE")
        .groupby(["SEASON_ID", "TEAM_ID"], group_keys=True)[stats.basic_box_score_stats]
        .parallel_apply(lambda x: x.rolling(10).mean().shift())
    )
    end = time.time()
    times.append(end - start)

raw = loader.load_all_team_games(add_stats=False)
raw["FG2A"] = stats.two_point_attempts(FGA=raw.FGA, FG3A=raw.FG3A)
raw["FG2M"] = stats.two_point_makes(FGM=raw.FGM, FG3M=raw.FG3M)
start = time.time()
e = (
    raw.sort_values("GAME_DATE")
    .groupby(["SEASON_ID", "TEAM_ID"], group_keys=True)[stats.basic_box_score_stats]
    .apply(lambda x: x.rolling(10).mean().shift())
)
end = time.time()
times.append(end - start)

# %%
import matplotlib.pyplot as plt

plt.plot(times)
plt.show()

# %%
df_team_games = (
    df_team_games.set_index(["SEASON_ID", "GAME_DATE"])
    .sort_index()
    .loc[pd.IndexSlice["21985":, :], :]
    .reset_index()
)
df_team_games

# %%
# Get basic expanding stats + percentages
df_group = df_team_games.groupby(
    ["TEAM_ID", "SEASON_ID"]
)  # Group df to facilitate operations
df_basic_exp = (
    df_group[stats.basic_box_score_stats].expanding().mean()
)  # Create expanding window df
df_pct_expanding = loader.add_independent_team_stats(
    df_group[stats.required_stat_params].expanding().sum()
).loc[:, stats.basic_box_score_cum_stats]
df_exp = pd.concat(  # Merge results
    [df_basic_exp, df_pct_expanding],
    axis=1,
)
df_exp.index = df_exp.index.droplevel(2)
df_team_games.set_index(["TEAM_ID", "SEASON_ID"], inplace=True)
df_exp = (
    df_team_games.loc[:, ~df_team_games.columns.isin(df_exp.columns)]
    .iloc[:, :9]
    .merge(
        df_exp,
        how="left",
        left_index=True,
        right_index=True,
    )
)
df_team_games.reset_index(inplace=True)
df_exp

# %%
# Merge data with self to get opponent stats
df_exp_all = loader._merge_team_games(df_exp.reset_index())
df_exp_all_merged = loader.add_dependent_team_stats(df_exp_all)
df_exp_all_merged


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
# Determine how to aggregate stats
# Which stats do we simply take the average of?
# Compute pct stats based on cumulative sum of stats
df_basic_expanding = df_team.loc[:, stats.basic_box_score_stats].expanding().mean()
df_pct_expanding = loader.add_independent_team_stats(
    df_team.loc[:, stats.required_stat_params].expanding().sum()
).loc[:, stats.basic_box_score_cum_stats]
df_pct_expanding

# %%


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
