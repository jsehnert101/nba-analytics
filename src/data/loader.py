# %%
# Imports
from typing import List, Tuple, Dict, Union, Literal
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
import os
import pickle
from tqdm import tqdm
import pandas as pd
from data.base import Data
from features.team_stats import TeamStats
from utils.data import join_str


# %%
class DataLoader(Data):
    def load_data(
        self,
        outer_folder: Literal["external", "internal"],
        inner_folder: Literal["raw", "interim", "processed", "inputation"],
        subdir: str,
        data_name: str,
    ) -> Union[Dict, List, Tuple, None]:
        """Load non-DataFrame data from local file system using pickle.

        Args:
            outer_folder (Literal["external", "internal"]): outer subfolder in which to save data.
            inner_folder (Literal["raw", "interim", "processed", "inputation"]): inner subfolder in which to save data.
            subdir (str): inner folder subdirectory in which to save data.
            data_name (str): name of data to be saved

        Returns:
            Union[Dict, List, Tuple, None]: requested data or None if not found.
        """
        if data_name.split(".")[-1] != "pickle":
            data_name += ".pickle"
        pth = os.path.join(
            self._folder_map[outer_folder][inner_folder], subdir, data_name
        )
        try:
            with open(pth, "rb") as handle:
                data = pickle.load(handle)
            return data
        except FileNotFoundError:
            return None

    def load_dataframe(
        self,
        outer_folder: Literal["external", "internal"],
        inner_folder: Literal["raw", "interim", "processed", "inputation"],
        subdir: str,
        data_name: str,
    ) -> Union[pd.DataFrame, None]:
        """Load dataframe from requested location.

        Args:
            outer_folder (Literal["external", "internal"]): outer subfolder in which to save data.
            inner_folder (Literal["raw", "interim", "processed", "inputation"]): inner subfolder in which to save data.
            subdir (str): inner folder subdirectory in which to save data.
            data_name (str): name of data to be saved

        Returns:
            Union[pd.DataFrame, None]: requested DataFrame or None if not found.
        """
        if data_name.split(".")[-1] != "parquet":
            data_name += ".parquet"
        pth = os.path.join(
            self._folder_map[outer_folder][inner_folder], subdir, data_name
        )
        try:
            return pd.read_parquet(pth)
        except FileNotFoundError:
            return None


class TeamDataLoader(DataLoader):
    def load_team_metadata(self):
        """Load NBA team metadata from local file system.

        Returns:
            Union[List[Dict[str, Any]], None]: list of dictionaries containing NBA team metadata.
        """
        return self.load_data(
            outer_folder="external",
            inner_folder="processed",
            subdir="teams",
            data_name="metadata.pickle",
        )

    def load_team_ids(self):
        """Load NBA team IDs from local file system.

        Returns:
            Union[List[int], None]: list of NBA team IDs.
        """
        return self.load_data(
            outer_folder="external",
            inner_folder="processed",
            subdir="teams",
            data_name="team_ids.pickle",
        )

    def load_team_id_map(self):
        """Load team ID map from local file system.

        Returns:
            Union[Dict[str, Union[int, str]], None]: dictionary mapping team name, city and abbreviation to team ID, or None if not found.
        """
        return self.load_data(
            outer_folder="external",
            inner_folder="processed",
            subdir="teams",
            data_name="team_id_map.pickle",
        )  # type: ignore

    def load_team_abbreviation_map(self):
        """Load team abbreviation map from local file system.

        Returns:
            Dict[str, Union[int, str]]: dictionary mapping team ID to team abbreviation.
        """
        return self.load_data(
            outer_folder="external",
            inner_folder="processed",
            subdir="teams",
            data_name="abbreviation_map.pickle",
        )


class TeamGameDataLoader(TeamDataLoader):
    def __init__(self):
        super().__init__()
        self.team_stats = TeamStats()
        self.regular_season_ids = self._create_regular_season_id_list()
        self.playoff_season_ids = self._create_playoff_season_id_list()

    def _create_regular_season_id_list(self) -> List[str]:
        return sorted(
            list(
                set(
                    os.listdir(
                        os.path.join(
                            self.folder_map["external"]["processed"],
                            "teams",
                            "games",
                            "RegularSeason",
                        )
                    )
                )
            )
        )

    def _create_playoff_season_id_list(self) -> List[str]:
        return sorted(
            list(
                set(
                    os.listdir(
                        os.path.join(
                            self.folder_map["external"]["processed"],
                            "teams",
                            "games",
                            "Playoffs",
                        )
                    )
                )
            )
        )

    def load_inputation_map(self):
        """Load team game inputation map to facilitate inputation of missing data.

        Returns:
            Dict[int, Dict[str, Dict[str, Union[str, int]]]]: team game data inputation map.
                TEAM_ID -> GAME_ID -> STAT: VALUE
        """
        return self.load_data(
            outer_folder="internal",
            inner_folder="inputation",
            subdir="teams/games",
            data_name="inputation_map.pickle",
        )

    def load_team_season_games(  # TODO account for various levels of cleaning
        self,
        team_id: int,
        season: str,
        season_type_nullable: Literal[
            "Pre Season", "Regular Season", "Playoffs"
        ] = "Regular Season",
    ) -> Union[pd.DataFrame, None]:
        return self.load_dataframe(
            outer_folder="external",
            inner_folder="processed",
            subdir=f"teams/games/{join_str(season_type_nullable)}/{season}",
            data_name=str(team_id),
        )

    def load_team_games(
        self,
        team_id: int,
        season_type_nullable: Literal[
            "Pre Season", "Regular Season", "Playoffs"
        ] = "Regular Season",
        max_workers: int = 6,  # TODO: multiprocessing
    ) -> pd.DataFrame:
        """Load team game data from local file system.

        Args:
            team_id (int): team ID
            season_type_nullable (str): season type (i.e. Pre Season, Regular Season, Playoffs)

        Returns:
            pd.DataFrame: team game data
        """
        max_workers = min(max_workers, max(int(os.cpu_count() or 0), max_workers) + 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self.load_team_season_games,
                    team_id=team_id,
                    season_type_nullable=season_type_nullable,
                    season=season,
                )
                for season in (
                    self.regular_season_ids
                    if season_type_nullable == "Regular Season"
                    else self.playoff_season_ids
                )
            ]
            _ = wait(futures)
            team_games = pd.concat([future.result() for future in futures])
        return team_games

    def _load_team_games(
        self,
        team_id: int,
        season_type_nullable: Literal[
            "Pre Season", "Regular Season", "Playoffs"
        ] = "Regular Season",
    ) -> pd.DataFrame:
        dfs = []
        for season in (
            self.regular_season_ids
            if season_type_nullable == "Regular Season"
            else self.playoff_season_ids
        ):
            dfs.append(
                self.load_team_season_games(
                    team_id=team_id,
                    season_type_nullable=season_type_nullable,
                    season=season,
                )
            )
        return pd.concat(dfs, axis=1)

    def add_independent_team_stats(self, team_games: pd.DataFrame) -> pd.DataFrame:
        """Add all basic + team stats to team game statistics"""
        team_games_dict = team_games.loc[
            :, team_games.columns.isin(self.team_stats.required_stat_params)
        ].to_dict(  # type: ignore
            orient="list"
        )
        for (
            stat_name,
            stat_func,
        ) in self.team_stats.independent_stat_method_map.items():
            team_games[stat_name] = stat_func(**team_games_dict)
        return team_games

    def _merge_team_games(self, team_games: pd.DataFrame) -> pd.DataFrame:
        """Merge team game statistics with opponent game statistics, adding 'OPP_' prefix to opponent stats."""
        return pd.concat(
            [
                team_games.groupby(
                    ["SEASON_ID", "GAME_ID", "GAME_DATE", "HOME", "TEAM_ID"], sort=True
                )
                .first()
                .reset_index(),
                team_games.groupby(
                    ["SEASON_ID", "GAME_ID", "GAME_DATE", "HOME", "TEAM_ID"]
                )
                .first()
                .sort_index(level=[0, 1, 2, 3], ascending=[True, True, True, False])
                .reset_index()
                .add_prefix("OPP_"),
            ],
            axis=1,
        )

    def add_dependent_team_stats(self, team_games: pd.DataFrame) -> pd.DataFrame:
        """Add all dependent team stats to team game statistics"""
        team_games_dict = team_games.loc[
            :, team_games.columns.isin(self.team_stats.required_stat_params)
        ].to_dict(  # type: ignore
            orient="list"
        )
        for stat_name, stat_func in tqdm(
            self.team_stats.dependent_stat_method_map.items()
        ):
            team_games[stat_name] = stat_func(**team_games_dict)
        return team_games

    def load_all_team_games(
        self,
        season_type_nullable: Literal[
            "Pre Season", "Regular Season", "Playoffs"
        ] = "Regular Season",
        add_stats: bool = True,
        max_workers: int = 6,  # TODO: multiprocessing
    ) -> pd.DataFrame:
        """Load all team game data from local file system.

        Args:
            season_type_nullable (str): season type (i.e. Pre Season, Regular Season, Playoffs)

        Returns:
            pd.DataFrame: team game data
        """
        max_workers = min(max_workers, max(int(os.cpu_count() or 0), max_workers) + 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self.load_team_games,
                    team_id=team_id,
                    season_type_nullable=season_type_nullable,
                )
                for team_id in self.load_team_ids()
            ]
            _ = wait(futures)
            team_games = pd.concat([future.result() for future in futures])
        team_games = (
            self.add_independent_team_stats(team_games) if add_stats else team_games
        )
        team_games = self._merge_team_games(team_games)
        return self.add_dependent_team_stats(team_games) if add_stats else team_games


# %%
