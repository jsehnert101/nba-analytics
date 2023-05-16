# %%
# Imports
from typing import List, Tuple, Dict, Union, Literal, Any
import os
import pickle
import pandas as pd
from data.base import Data
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
        """_summary_

        Args:
            outer_folder (Literal["external", "internal"]): outer subfolder in which to save data.
            inner_folder (Literal["raw", "interim", "processed", "inputation"]): inner subfolder in which to save data.
            subdir (str): inner folder subdirectory in which to save data.
            data_name (str): name of data to be saved

        Returns:
            Union[pd.DataFrame, None]: requested DataFrame or None if not found.
        """
        pth = os.path.join(
            self._folder_map[outer_folder][inner_folder], subdir, data_name
        )
        try:
            return pd.read_parquet(pth)
        except FileNotFoundError as e:
            print(e)
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
        season_type_nullable: Literal["Pre Season", "Regular Season", "Playoffs"],
        season: str,
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
        season_type_nullable: Literal["Pre Season", "Regular Season", "Playoffs"],
    ) -> Union[pd.DataFrame, None]:
        return self.load_dataframe(
            outer_folder="external",
            inner_folder="processed",
            subdir=f"teams/games/{join_str(season_type_nullable)}/ALL",
            data_name=str(team_id),
        )
