"""
Data cleaning module for NBA data.
"""
from typing import List, Dict, Any, Union
from data.base import Data
from data.loader import TeamGameDataLoader
from data.saver import DataSaver
import pandas as pd
from utils.web import open_url


class DataCleaner(Data):
    def __init__(self):
        super().__init__()
        self.bball_reference_url = "https://www.basketball-reference.com"


class TeamDataCleaner(DataCleaner):
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

    def clean_team_game_data(self, team_games: pd.DataFrame) -> pd.DataFrame:
        team_games.drop(columns=["PLUS_MINUS"], inplace=True)
        team_games.rename(columns={"MIN": "MP"}, inplace=True)
        team_games.loc[:, "MATCHUP"] = team_games.MATCHUP.str.replace("GOS", "GSW")
        team_games.loc[:, "MATCHUP"] = team_games.MATCHUP.str.replace("UTH", "UTA")
        team_games["WIN"] = team_games.WL.replace({"W": True, "L": False})
        team_games["HOME"] = team_games.MATCHUP.str.contains("vs.")
        team_games.loc[team_games.FG3A == 0, "FG3_PCT"] = 0
        team_games["GAME_DATE"] = pd.to_datetime(team_games.GAME_DATE)
        team_games = team_games.sort_values("GAME_DATE", ascending=True).reset_index(
            drop=True
        )
        return team_games

    def inpute_team_game_data(
        self, team_id: int, team_games: pd.DataFrame
    ) -> pd.DataFrame:
        if team_games.drop(columns=["FG3_PCT"]).isna().sum().sum() == 0:
            return team_games
        else:
            team_games_copy = team_games.set_index("GAME_ID").copy(deep=True)
            try:
                team_games_copy = team_games_copy.fillna(
                    pd.DataFrame().from_dict(
                        self.imputation_map[team_id], orient="index"
                    )
                )
                if team_games_copy.drop(columns=["FG3_PCT"]).isna().sum().sum() == 0:
                    return team_games_copy.reset_index()
            except KeyError:
                pass

        missing_cols = team_games.columns[team_games.isna().any()].tolist()
        if "FG3_PCT" in missing_cols:
            missing_cols.remove("FG3_PCT")
        missing_df = team_games.loc[
            team_games.loc[:, missing_cols].isna().any(axis=1), :
        ].drop(columns=["FG3_PCT"])
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
                fill_val = int(
                    input(
                        f"How many {stat} did {row.TEAM_NAME} have in their game {row.MATCHUP} on {row.GAME_DATE.strftime('%b %-d, %Y')}?"
                    )
                )
                team_games.loc[idx, stat] = fill_val  # type: ignore
                self.imputation_map[team_id][row.GAME_ID][stat] = fill_val
        self.saver.save_data(
            data=self.imputation_map,
            outer_folder="internal",
            inner_folder="inputation",
            subdir="teams/games",
            data_name="inputation_map.pickle",
        )
        return team_games
