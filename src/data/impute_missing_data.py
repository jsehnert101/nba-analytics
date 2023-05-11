# %%
import pandas as pd

# %%
df_gs = pd.read_parquet(
    "../../../data/processed/teams/games/RegularSeason/1610612744.parquet"
)
df_gs
