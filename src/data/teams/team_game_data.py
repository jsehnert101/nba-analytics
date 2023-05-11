# %%
# Imports
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Literal, Union
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.library.parameters import LeagueID, SeasonType
from data.utils.data import save_data, retry
from data.teams.team_data import TeamData
from data.teams.team_stats import TeamStats

# %%
# Write object to extend TeamData specifically for game data.


class TeamGameData(TeamData):
    def __init__(self, load: bool = True, save: bool = False):
        super().__init__()
        self._load_data = load
        self._save_data = save
        self.interim_folder = "../../../data/interim/teams/"
        self.folder_map.update({"interim": self.interim_folder})
        self.team_stats = TeamStats()

    @property
    def interim_folder(self) -> str:
        """Get interim folder path to team games."""
        return self._interim_folder

    @interim_folder.setter
    def interim_folder(self, folder: str) -> None:
        """Set interim folder path to team games."""
        self._interim_folder = os.path.abspath(folder)

    def _load_team_games(
        self,
        team_id: int,
        season_type: str,
        folder_type: Literal["raw", "interim", "processed"] = "processed",
    ) -> Union[pd.DataFrame, None]:  # TODO: parallelize
        season_type = "".join(season_type.split())
        try:
            return pd.read_parquet(
                os.path.join(
                    self.folder_map[folder_type],
                    "games",
                    season_type,
                    f"{team_id}.parquet",
                )
            )
        except FileNotFoundError as e:
            print(e)
            return None

    @retry
    def _retrieve_team_games(
        self, team_id: int, season_type: str = SeasonType.regular
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
            team_id_nullable=str(team_id),
            season_type_nullable=season_type,
            timeout=60,
        ).get_data_frames()[0]

    def _save_team_games(
        self,
        team_games: pd.DataFrame,
        team_id: int,
        season_type: str,
        folder_type: Literal["raw", "interim", "processed"],
    ) -> bool:
        season_type = "".join(season_type.split())
        folder = os.path.join(self.folder_map[folder_type], "games", season_type)
        return save_data(data=team_games, folder=folder, name=f"{team_id}.parquet")

    def add_team_stats(self):
        """Add all basic + team stats to team game statistics"""

    def _clean_team_games(self, team_games: pd.DataFrame) -> pd.DataFrame:
        """Clean team game statistics.

        Args:
            team_games (pd.DataFrame): dataframe of team game statistics.

        Returns:
            pd.DataFrame: cleaned dataframe of team game statistics.
        """
        team_games["GAME_DATE"] = pd.to_datetime(team_games.GAME_DATE)
        team_games = team_games.sort_values("GAME_DATE", ascending=True).reset_index(
            drop=True
        )
        team_games.rename(columns={"MIN": "MP"}, inplace=True)
        team_games["WIN"] = team_games.WL.replace({"W": True, "L": False})
        team_games["HOME"] = np.NaN
        team_games.loc[:, "HOME"] = team_games.MATCHUP.str.contains("vs.")
        team_games.loc[team_games.FG3A == 0, "FG3_PCT"] = 0
        team_games["REST_DAYS"] = (
            team_games.groupby("SEASON_ID")["GAME_DATE"].diff().dt.days.astype(float)  # type: ignore
        )
        return team_games

    def get_team_games(
        self,
        team_id: int,
        season_type: str = SeasonType.regular,
        folder_type: Literal["raw", "interim", "processed"] = "processed",
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
            self._load_team_games(
                team_id=team_id, season_type=season_type, folder_type=folder_type
            )
            if self._load_data
            else None
        )
        if df_games is None:
            df_games = self._retrieve_team_games(
                team_id=team_id, season_type=season_type
            )
            raw_pth = os.path.join(
                self.folder_map[folder_type],
                "games",
                "".join(season_type.split()),
                f"{team_id}.parquet",
            )
            if not os.path.isfile(raw_pth):
                self._save_team_games(
                    team_games=df_games,
                    team_id=team_id,
                    season_type=season_type,
                    folder_type="raw",
                )
            df_games = self._clean_team_games(team_games=df_games)
            interim_pth = os.path.join(
                self._interim_folder,
                "games",
                "".join(season_type.split()),
                f"{team_id}.parquet",
            )
            if not os.path.isfile(interim_pth):
                self._save_team_games(
                    team_games=df_games,
                    team_id=team_id,
                    season_type=season_type,
                    folder_type="interim",
                )
            return df_games
        else:
            return df_games

    def get_multiple_teams_games(
        self,
        team_ids: Union[List[str], None] = None,
        season_type: str = SeasonType.regular,
        folder_type: Literal["raw", "interim", "processed"] = "processed",
        max_workers: int = 6,
    ) -> pd.DataFrame:
        """
        Retrieve + merge all available games for provided teams. If no teams provided, merge all teams.

        Args:
            team_ids (Union[List[str], None], optional): list of Team IDs to retrieve. Defaults to None.
            season_type (str, optional): season type from which to retrieve. Defaults to SeasonType.regular.
            max_workers (int, optional): max workers in multithreading. Defaults to 6.

        Returns:
            pd.DataFrame: merged DataFrame of all team games.
        """
        team_ids = team_ids or self.all_team_ids
        max_workers = min(max_workers, max(int(os.cpu_count() or 0), max_workers) + 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(
                self.get_team_games,
                team_ids,
                np.repeat(season_type, len(team_ids)),
                np.repeat(folder_type, len(team_ids)),
            )
        return pd.concat([res for res in results])
