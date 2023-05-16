from typing import Dict, List, Any, Literal
import os
import pickle
from pandas.core.frame import DataFrame
from data.base import Data
from utils.data import join_str


class DataSaver(Data):
    def save_data(  # TODO: create custom decorator to manage path -- pass pth variable
        self,
        data,
        outer_folder: Literal["external", "internal"],
        inner_folder: Literal["raw", "interim", "processed", "inputation"],
        subdir: str,
        data_name: str,
    ) -> None:
        """Save non-DataFrame data to desired location.

        Args:
            data (_type_): data to be saved.\n
            outer_folder (Literal["external", "internal"]): outer subfolder in which to save data.\n
            inner_folder (Literal["raw", "interim", "processed", "inputation"]): inner subfolder in which to save data.\n
            subdir (str): inner folder subdirectory in which to save data.\n
            data_name (str): name of data to be saved.\n
        """
        if data_name.split(".")[-1] != "pickle":
            data_name += ".pickle"
        folder_pth = os.path.join(self._folder_map[outer_folder][inner_folder], subdir)
        try:
            with open(os.path.join(folder_pth, data_name), "wb") as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except FileNotFoundError:
            try:
                print("Making directory...")
                os.makedirs(folder_pth, exist_ok=True)
                with open(os.path.join(folder_pth, data_name), "wb") as handle:
                    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            except FileNotFoundError as e:
                print(e)

    def save_dataframe(
        self,
        df: DataFrame,
        outer_folder: Literal["external", "internal"],
        inner_folder: Literal["raw", "interim", "processed", "inputation"],
        subdir: str,
        data_name: str,
    ) -> None:
        """Save pandas DataFrame to desired location.

        Args:
            df (DataFrame): DataFrame to be saved.\n
            outer_folder (Literal["external", "internal"]): outer subfolder in which to save data.\n
            inner_folder (Literal["raw", "interim", "processed", "inputation"]): inner subfolder in which to save data.\n
            subdir (str): inner folder subdirectory in which to save data.\n
            data_name (str): name of data to be saved
        """
        if data_name.split(".")[-1] != "parquet":
            data_name += ".parquet"
        folder_pth = os.path.join(self._folder_map[outer_folder][inner_folder], subdir)
        print(folder_pth)
        try:
            df.to_parquet(os.path.join(folder_pth, data_name))
        except OSError:
            try:
                os.makedirs(folder_pth, exist_ok=True)
                df.to_parquet(os.path.join(folder_pth, data_name))
            except FileNotFoundError as e:
                print(e)


class TeamDataSaver(DataSaver):
    def save_team_metadata(self, team_metadata: List[Dict[str, Any]]):
        self.save_data(
            data=team_metadata,
            outer_folder="external",
            inner_folder="raw",
            subdir="teams",
            data_name="metadata",
        )

    def save_team_ids(self, team_ids: List[int]):
        self.save_data(
            data=team_ids,
            outer_folder="external",
            inner_folder="processed",
            subdir="teams",
            data_name="team_ids",
        )

    def save_team_id_map(self, team_id_map):
        self.save_data(
            data=team_id_map,
            outer_folder="external",
            inner_folder="processed",
            subdir="teams",
            data_name="team_id_map",
        )

    def save_team_abbreviation_map(self, team_abbreviation_map):
        self.save_data(
            data=team_abbreviation_map,
            outer_folder="external",
            inner_folder="processed",
            subdir="teams",
            data_name="team_abbreviation_map",
        )


class TeamGameDataSaver(TeamDataSaver):
    def save_team_season_game_data(
        self,
        season_games: DataFrame,
        team_id: int,
        season_type: str,
        season: str,
        outer_folder: Literal["external", "internal"],
        inner_folder: Literal["raw", "interim", "processed", "inputation"],
    ) -> None:
        self.save_dataframe(
            season_games,
            outer_folder=outer_folder,
            inner_folder=inner_folder,
            subdir=f"teams/games/{join_str(season_type)}/{season}",
            data_name=str(team_id),
        )

    def save_team_game_data(
        self,
        games: DataFrame,
        team_id: int,
        season_type: str,
        outer_folder: Literal["external", "internal"],
        inner_folder: Literal["raw", "interim", "processed", "inputation"],
    ) -> None:
        self.save_dataframe(
            games,
            outer_folder=outer_folder,
            inner_folder=inner_folder,
            subdir=f"teams/games/{join_str(season_type)}/ALL",
            data_name=str(team_id),
        )
