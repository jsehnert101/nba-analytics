"""
Data cleaning module for NBA data.
"""
from typing import List, Dict, Any, Union
import numpy as np
import pandas as pd
from data.base import Data
from data.loader import TeamGameDataLoader
from data.saver import DataSaver
from utils.web import open_url


class DataCleaner(Data):
    def __init__(self):
        super().__init__()
        self.bball_reference_url = "https://www.basketball-reference.com"


class TeamDataCleaner(DataCleaner):
    def __init__(self):
        super().__init__()
        self._missing_cols = ["DREB", "OREB"]

    def extract_team_ids(self, metadata: List[Dict[str, Any]]) -> List[int]:
        """Extract team IDs from team metadata.

        Args:
            metadata (List[Dict[str, Any]]): team metadata

        Returns:
            List[int]: list of team IDs
        """
        return [int(team["id"]) for team in metadata]

    def create_team_id_map(
        self, team_metadata: List[Dict[str, Any]]
    ) -> Dict[str, Union[int, str]]:
        """Create mapping from team name, city and abbreviation to team ID.

        Args:
            team_metadata (List[Dict[str, Any]]): team metadata

        Returns:
            Dict[str, int]: Mapping to Team ID.
        """
        team_id_map: dict = {}
        for team in team_metadata:
            team_id_map[team["nickname"]] = int(team["id"])
            team_id_map[team["city"]] = int(team["id"])
            team_id_map[team["abbreviation"]] = int(team["id"])
        return team_id_map

    def create_team_abbreviation_map(
        self, team_id_map: Dict[str, Union[int, str]]
    ) -> Dict[int, str]:
        """Create mapping from team ID to team name (i.e. inverse of team_id_map).

        Args:
            team_id_map (Dict[str, Union[int, str]]): _description_

        Returns:
            Dict[int, str]: Mapping from team ID to team name.
        """
        return {
            id: name for i, (name, id) in enumerate(team_id_map.items()) if i % 3 == 0
        }  # type: ignore


class TeamGameDataCleaner(TeamDataCleaner):
    def __init__(self, team_ids: List[int]):
        super().__init__()
        self.team_ids = team_ids
        self.bball_reference_boxscore_url = (
            self.bball_reference_url + "/boxscores/{}0{}.html"
        )
        self.loader = TeamGameDataLoader()
        self.saver = DataSaver()
        self.imputation_map = self._get_imputation_map(team_ids=team_ids)

    def _get_imputation_map(self, team_ids: List[int]):
        return self.loader.load_inputation_map() or {id: {} for id in team_ids}

    def _extract_team_abbreviations(self, matchup: str) -> List[str]:
        if "@" in matchup:
            return [s.strip() for s in matchup.split("@")]
        elif "vs." in matchup:
            return [s.strip() for s in matchup.split("vs.")]
        else:
            raise ValueError(f"Matchup {matchup} not formatted correctly!")

    def _replace_matchups(self, matchup: pd.Series) -> pd.Series:
        """Ensure consistent team abbreviations."""
        return matchup.str.replace("GOS", "GSW").str.replace("UTH", "UTA")

    def _adjust_three_point_shooting(self, team_games: pd.DataFrame) -> pd.DataFrame:
        """Adjust 3PT% for games prior to 2000-01 season."""
        team_games.loc[:, "FG3_PCT"] = team_games.FG3M.divide(team_games.FG3A)
        team_games.loc[team_games.FG3A.eq(0), "FG3_PCT"] = 0.0
        return team_games

    def clean_team_game_data(self, team_games: pd.DataFrame) -> pd.DataFrame:
        """Clean team game data.\n
        Ensure team abbreviations are consistent, denote home/away games,
        adjust datetime format, correct missing 3PT% data, etc.

        Args:
            team_games (pd.DataFrame): team game dataframe\n

        Returns:
            pd.DataFrame: clean team game data
        """
        team_games = team_games.drop(columns=["PLUS_MINUS"]).rename(
            columns={"MIN": "MP"}
        )
        team_games["WIN"] = team_games.WL.replace({"W": True, "L": False})
        team_games["HOME"] = team_games.MATCHUP.str.contains("vs.")
        team_games.loc[:, "MATCHUP"] = self._replace_matchups(team_games.MATCHUP)
        team_games = self._adjust_three_point_shooting(team_games)
        team_games["GAME_DATE"] = pd.to_datetime(team_games.GAME_DATE)
        team_games = team_games.sort_values("GAME_DATE", ascending=True).reset_index(
            drop=True
        )
        # DREB/OREB data tends to be missing, so make sure we aren't filling faulty values.
        for col in self._missing_cols:
            team_games.loc[team_games.GAME_DATE < "2000-1-1", col] = team_games.loc[
                team_games.GAME_DATE < "2000-1-1", col
            ].replace({0: np.NaN})
        return team_games

    def impute_team_game_data(
        self, team_id: int, team_games: pd.DataFrame
    ) -> pd.DataFrame:
        """Manually replace missing team game data.\n
        For each game with missing data, two things will happen:\n
            1. A web browser will open to the corresponding boxscore on basketball-reference.com
            2. The user will be prompted to enter the missing data
        If the missing data is not available, the user should enter -1 as directed by the prompt.

        Args:
            team_id (int): team id whose missing values we're replacing
            team_games (pd.DataFrame): dataframe of team games

        Returns:
            pd.DataFrame: team game data with missing data filled, if possible.
        """
        team_games.set_index("GAME_ID", inplace=True)
        try:
            team_games = team_games.fillna(
                pd.DataFrame().from_dict(self.imputation_map[team_id], orient="index")
            )
            if team_games.isna().sum().sum() == 0:
                return team_games.reset_index()
            else:
                team_games.reset_index(inplace=True)
        except KeyError:
            pass
        missing_cols = team_games.columns[team_games.isna().any()].tolist()
        if "FG3_PCT" in missing_cols:
            missing_cols.remove("FG3_PCT")
        missing_df = team_games.loc[
            team_games.loc[:, missing_cols].isna().any(axis=1), :
        ]
        for idx, row in missing_df.iterrows():
            self.imputation_map[team_id][row.GAME_ID] = {}
            url_game_date = row.GAME_DATE.strftime("%Y%m%d")
            teams = self._extract_team_abbreviations(row.MATCHUP)
            open_url(
                [
                    self.bball_reference_boxscore_url.format(url_game_date, team)
                    for team in teams
                ]
            )
            missing_stats = row[row.isna()].index.tolist()
            for stat in missing_stats:
                if stat == "FG3_PCT":
                    continue
                fill_val = int(
                    input(
                        f"""
                        How many {stat} did {row.TEAM_NAME} have in their game {row.MATCHUP}
                        on {row.GAME_DATE.strftime('%b %-d, %Y')}? Enter -1 for missing data.
                        """
                    )
                )
                if fill_val == -1:
                    fill_val = np.NaN
                else:
                    team_games.loc[idx, stat] = fill_val  # type: ignore
                self.imputation_map[team_id][row.GAME_ID][stat] = fill_val
        team_games = self._adjust_three_point_shooting(team_games)
        self.saver.save_data(
            data=self.imputation_map,
            outer_folder="internal",
            inner_folder="inputation",
            subdir="teams/games",
            data_name="inputation_map.pickle",
        )
        return team_games
