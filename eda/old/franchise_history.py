# %%
import numpy as np
import pandas as pd

pd.options.display.max_columns = 50

from nba_api.stats.endpoints import FranchiseHistory as NBAFranchiseHistory
from nba_api.stats.library.parameters import LeagueID


# %%
class FranchiseHistory:
    def __init__(self):
        self.league_id = LeagueID.nba
        self._franchise_history = NBAFranchiseHistory(league_id=self.league_id)
        self.active_teams_franchise_history = self.get_all_teams_franchise_history(
            active=True
        )
        self.defunct_teams_franchise_history = self.get_all_teams_franchise_history(
            active=False
        )

    def aggregate_franchise_history(
        self, df: pd.DataFrame, agg_col: str = "team_id"
    ) -> pd.DataFrame:
        df_agg = df.groupby(agg_col).agg(
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
        df_agg["win_pct"] = df_agg.wins.divide(df_agg.losses)
        return df_agg

    def get_all_teams_franchise_history(
        self, active: bool = True, aggregate: bool = False
    ) -> pd.DataFrame:
        if active:
            df_active: pd.DataFrame = self._franchise_history.get_data_frames()[0].drop(
                columns=["LEAGUE_ID"]
            )
            df_active.columns = df_active.columns.str.lower()
            if aggregate:
                return self.aggregate_franchise_history(df_active)
            else:
                return df_active
        else:
            df_defunct: pd.DataFrame = self._franchise_history.get_data_frames()[
                1
            ].drop(columns=["LEAGUE_ID"])
            df_defunct.columns = df_defunct.columns.str.lower()
            if aggregate:
                return self.aggregate_franchise_history(df_defunct)
            else:
                return df_defunct
