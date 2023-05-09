# %% Import necessary libraries
import numpy as np
import pandas as pd

pd.options.display.max_columns = 50
from nba_api.stats.static import teams
from nba_api.stats.endpoints import (
    leaguegamefinder,
    FranchiseHistory as NBAFranchiseHistory,
)
from nba_api.stats.library.parameters import LeagueID, SeasonType
from data.teams.franchise_history import FranchiseHistory
from utils.data import *
from typing import Union, List, Dict, Any
from data.stats import Stats


# %%
# Define class to wrangle + clean NBA team data
class TeamData:
    def __init__(
        self,
        folder: str = "../../../data/processed/teams/",
        load_data: bool = True,
        save_data: bool = False,
    ):
        self._folder = folder
        self._load_data, self._save_data = load_data, save_data
        self.metadata = self.get_team_metadata()
        self.team_ids: dict = self.get_team_id_map()
        print(self.team_ids)
        self.league_id = LeagueID.nba
        self.franchise_history = (
            NBAFranchiseHistory().get_data_frames()[0].drop(columns=["LEAGUE_ID"])
        )
        self.stats = Stats()

    def _retrieve_team_metadata(self) -> List[Dict[str, Any]]:
        return teams.get_teams()

    def _load_team_metadata(self) -> List[Dict[str, Any]]:
        return load_data(pth=self._folder + "team_metadata.pickle")

    def get_team_metadata(self) -> List[Dict[str, Any]]:
        team_metadata = self._load_team_metadata()
        if team_metadata is None:
            team_metadata = self._retrieve_team_metadata()
            save_data(data=team_metadata, pth=self._folder + "team_metadata.pickle")
        return team_metadata

    def _load_team_id_map(self) -> Union[Dict[str, int], None]:
        return load_data(pth=self._folder + "team_id_map.pickle")

    def _create_team_id_map(self) -> Dict[str, int]:
        """Create mapping from team name, city and abbreviation to team ID.

        Returns:
            Dict[str, int]: Mapping to team ID.
        """
        team_id_map: dict = {}
        for team in self.metadata:
            team_id_map[team["nickname"]] = team["id"]
            team_id_map[team["city"]] = team["id"]
            team_id_map[team["abbreviation"]] = team["id"]
        save_data(data=team_id_map, pth=self._folder + "team_id_map.pickle")
        return team_id_map

    def get_team_id_map(self) -> Dict[str, int]:
        """
        Retrieves mapping from team name, abbreviation and city to team ID
        to ease access of Team IDs for future use.

        Searches locally then online if not found.

        Returns:
            Dict[str, int]: Mapping to team ID.
        """
        team_id_map = self._load_team_id_map()
        if team_id_map is None:
            team_id_map = self._create_team_id_map()
        return team_id_map

    def _load_team_games(
        self, team_id: str, season_type: str
    ) -> Union[pd.DataFrame, None]:  # TODO: parallelize
        season_type = "".join(season_type.split())
        try:
            return pd.read_parquet(
                f"{self._folder}/games/{season_type}/{team_id}.parquet"
            )
        except FileNotFoundError:
            return None

    @retry
    def _retrieve_team_games(
        self, team_id: str, season_type: str = SeasonType.regular
    ) -> pd.DataFrame:
        """Retrieves df of all games of a specified type for a given NBA team.

        Args:
            team_id (int): NBA Team ID.
            season_type (SeasonType, optional): Defaults to SeasonType.regular.

        Returns:
            pd.DataFrame: df of all team games and accompanying stats.
        """
        return leaguegamefinder.LeagueGameFinder(
            league_id_nullable=LeagueID.nba,
            team_id_nullable=team_id,
            season_type_nullable=season_type,
            timeout=60,
        ).get_data_frames()[0]

    def _save_team_games(
        self,
        team_games: pd.DataFrame,
        team_id: str,
        season_type: str,
    ) -> bool:
        season_type = "".join(season_type.split())
        try:
            team_games.to_parquet(
                f"{self._folder}/games/{season_type}/{team_id}.parquet"
            )
            return True
        except Exception as e:
            print(e)
            return False

    def add_team_stats(self):
        """Add all basic + team stats to team game statistics"""
        return

    def get_team_games(
        self,
        team_id: str,
        season_type: str = SeasonType.regular,
    ) -> pd.DataFrame:
        """Get all available games for given team and add desired stats.

        Args:
            team_id (int): NBA Team ID.
            season_type (str, optional): Defaults to SeasonType.regular.
            save (bool, optional): Whether or not to save team games. Defaults to False.

        Returns:
            pd.DataFrame: dataframe of all team game statistics.
        """
        df_games = (
            self._load_team_games(team_id=team_id, season_type=season_type)
            if self._load_data
            else None
        )
        if df_games is None:
            df_games = self._retrieve_team_games(
                team_id=team_id, season_type=season_type
            )
        else:
            return df_games

        df_games["GAME_DATE"] = pd.to_datetime(df_games.GAME_DATE)
        df_games = df_games.sort_values("GAME_DATE", ascending=True).reset_index(
            drop=True
        )
        df_games["WIN"] = df_games.WL.replace({"W": True, "L": False})
        df_games["HOME"] = np.NaN
        df_games.loc[:, "HOME"] = df_games.MATCHUP.str.contains("vs.")
        df_games.loc[df_games.FG3A == 0, "FG3_PCT"] = 0
        df_games["REST_DAYS"] = (
            df_games.groupby("SEASON_ID")["GAME_DATE"].diff().dt.days.astype(float)
        )
        df_games.rename(columns={"REB": "RB"})
        # for key in
        if self._save_data:
            self._save_team_games(
                team_games=df_games, team_id=team_id, season_type=season_type
            )
        return df_games

    def get_multiple_teams_games(
        self,
        team_ids: Union[List[str], None] = None,
        season_type: str = SeasonType.regular,
    ) -> pd.DataFrame:
        """Merge team game logs for provided teams. If no teams provided, merge all teams.

        Args:
            team_ids (Union[List[int], None]): list of team IDs to merge.
            season_type (str, optional): Defaults to SeasonType.regular.

        Returns:
            pd.DataFrame: merged dataframe of all team game logs.
        """
        if team_ids is None:
            team_ids = np.unique(list(self.team_ids.values()))
        team_games = []
        for team_id in team_ids:
            team_games.append(
                self.get_team_games(team_id=team_id, season_type=season_type)
            )
        return pd.concat(team_games)
