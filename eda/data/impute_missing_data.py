# %%
# Imports
import os
from utils.web import open_url
import numpy as np
import pandas as pd
from data.teams.team_data import TeamData


# %%
from nba_api.stats.library.parameters import (
    SeasonType,
    SeasonTypePlayoffs,
    SeasonTypeAllStar,
    SeasonTypeNullable,
)

SeasonType


# %%
# Organize files
team_data = TeamData()
folder = "/Users/jsehnert101/git_repos/nba-analytics/data/internal/raw/teams/games/RegularSeason"
file_names = os.listdir(folder)
file_paths = [os.path.join(folder, file) for file in file_names]

# %%
# Inpute missing data from NBA API
# Iterate through all files
# Identify missing data
# Prompt user to input missing data
# Store missing data for later use
# Save to external/interim
# Account for missing directories


# %%
# Gather names of all missing columns aside from 3PT%
bball_reference_url = "https://www.basketball-reference.com/boxscores/{}0{}.html"
empty_imputation_map = {}
missing_data_map = {}
for i, file in enumerate(file_paths):
    df = pd.read_parquet(file).drop(
        columns=["PLUS_MINUS"]
    )  # Column often missing - will fill later

    # TODO: Perform data cleaning in separate script
    df["GAME_DATE"] = pd.to_datetime(df.GAME_DATE)
    df.loc[df.MATCHUP.str.contains("GOS"), "MATCHUP"] = df.loc[
        df.MATCHUP.str.contains("GOS"), "MATCHUP"
    ].str.replace("GOS", "GSW")

    if df.isna().sum().sum() > 0:
        missing_cols = df.columns[df.isna().any()].tolist()
        if "FG3_PCT" in missing_cols:
            df.loc[(df.FG3_PCT.isna() & df.FG3A == 0), "FG3_PCT"] = 0
            if not df.FG3_PCT.isna().any():
                missing_cols.remove("FG3_PCT")
        if len(missing_cols) > 0:
            team_id = df.TEAM_ID.values[0]
            empty_imputation_map[team_id] = {}
            temp_df = df.loc[
                df.loc[:, missing_cols].isna().any(axis=1), :
            ]  # Isolate rows with missing data
            for (
                idx,
                row,
            ) in temp_df.iterrows():  # Go through each row and inpute its missing data
                # TODO: Open web browser with box score to input
                open_url(
                    bball_reference_url.format(
                        row.GAME_DATE.strftime("%Y%m%d"),
                        row.MATCHUP.split("@")[0].strip(),
                    ),
                    bball_reference_url.format(
                        row.GAME_DATE.strftime("%Y%m%d"),
                        row.MATCHUP.split("@")[1].strip(),
                    ),
                )
                missing_row = row[row.isna()].copy()
                for col, _ in missing_row.items():
                    fill_val = int(
                        input(
                            f"How many {col} did {row.TEAM_NAME} have in their game {row.MATCHUP} on {row.GAME_DATE.strftime('%b %-d, %Y')}?"
                        )
                    )
                    df.loc[
                        idx, col
                    ] = fill_val  # TODO: save data output + track missing data in dict to store after
            break


df


# %%
# Create imputation mapping to manually fill
imputation_map = {}
for team, missing_cols in missing_data_map.items():
    df = pd.read_parquet(os.path.join(folder, f"{team_data.team_id_map[team]}.parquet"))
    df = df.loc[df.loc[:, missing_cols].isna().any(axis=1), :]
    break
df


# %%
test_df = (
    pd.read_parquet(file).drop(columns=["PLUS_MINUS"]).set_index(["TEAM_ID", "GAME_ID"])
)
test_df

# %%
import tkinter as tk
from tkinter import simpledialog

ROOT = tk.Tk()
ROOT.withdraw()
USER_INP = simpledialog.askinteger(title="Test", prompt="Who are you?")
print(USER_INP)

# %%
tk.messagebox.showinfo("Title", "a Tk MessageBox")

# %%
window = tk.Tk()
greeting = tk.Label(text="Hello, Tkinter")
greeting.pack()
window.mainloop()


# %%
# Store missing data locations for future reference
imputation_map = {"TEAM_ID": {"GAME_ID": {"missing_col_name": "missing_col_value"}}}


file = file_paths[i]
df = pd.read_parquet(file)
df

# %%
df.loc[(df.FG3_PCT.isna() & df.FG3A.eq(0)), "FG3_PCT"] = 0
df


# %%
# Check missing data

# %%
res = input("Gimme something")
print(res)

# %%
team_data.team_id_map["GSW"]
# %%
pd.read_parquet("../../data/internal/raw/teams/games/RegularSeason/1610612744.parquet")
