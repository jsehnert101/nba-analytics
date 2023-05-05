# %%
import numpy as np
import pandas as pd
import webbrowser



# %%
class NBAStats():
    
    def __init__(self):
        pass
    
    def three_point_attempt_rate(self, three_point_attempts: np.array, field_goal_attempts: np.array) -> np.array:
        return three_point_attempts/field_goal_attempts
    
    def effective_field_goal_percentage(self, field_goals_made):
        return

# %%
class NBATeamStats():

    def __init__(self):
        self._advanced_stats = ["3PAr", "eFG", "TS", "POSS", "PPP", "OffRtg", "PACE"]
        self._advanced_stats_url = "https://www.fromtherumbleseat.com/pages/advanced-basketball-statistics-formula-sheet"

    def add_advanced_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        ### Advanced stats only available for recent games, so need to engineer our own
        df["3PAr"] = df.loc[:,"3PA"].divide(df.FGA)
        df["eFG"] = df.los[:,"3PM"].multiply(0.5).add(df.FGM).divide(df.FGA)
        df["TS"] = df.PTS.divide(df.FGA.multiply(2).add(df.FTA.multiply(0.44)))
        df["POSS"] = df.FGA.add(df.FTA.multiply(0.44)).subtract(df.OREB).add(df.TOV)
        df["PPP"] = df.PTS/df.POSS
        df["OffRtg"] = df.PPP.multiply(100)
        df["PACE"] = df.POSS.multiply(48).divide(df.MIN)
        return df
    
    def get_features_we_should_add(self) -> list:
        return ["E_OFF_RATING", "DEF_RATING", "E_DEF_RATING", "NET_RATING", "AST_PCT"]
    
    def get_link_to_advanced_stats_formulas(self) -> None:
        webbrowser.open(self._advanced_stats_url)