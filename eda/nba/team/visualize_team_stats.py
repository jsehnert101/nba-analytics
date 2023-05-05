# %%
# Imports
from data.teams.team_data import TeamData
from nba_api.stats.library.parameters import SeasonID
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

# %%
# Access all team data from 2016 onward
team_data = TeamData(load_data=True, save_data=False)
df_team_games = team_data.get_multiple_teams_games()
df_team_games = df_team_games.loc[
    df_team_games.SEASON_ID.gt(SeasonID().get_season_id("2015")), :
]
df_team_games

# %%
# Evaluate distribution of FTAs vs FTMs by game outcome
fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

sns.histplot(
    data=df_team_games,
    x="FTA",
    hue="WL",
    bins=100,
    multiple="stack",
    kde=True,
    ax=axs[0],
)
sns.histplot(
    data=df_team_games,
    x="FTM",
    hue="WL",
    bins=100,
    multiple="stack",
    kde=True,
    ax=axs[1],
)

plt.show()
# %%
sns.scatterplot(data=df_team_games, x="FTM", y="FTA", hue="WL")
# %%
