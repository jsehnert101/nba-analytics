# %% Import necessary libraries
import os
from typing import Union, List, Dict, Any
import numpy as np
import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.library.parameters import LeagueID
from cachetools import cached
from data.utils import load_data, save_data, retry, cache

# from nba_api.stats.endpoints import FranchiseHistory

pd.options.display.max_columns = 50


# %%
# Define class to wrangle + clean NBA team data
class TeamData(object):
    def __init__(self):
        self.raw_folder = "../../../data/internal/"
        self.raw_folder = "../../../data/internal/raw/teams/"
        self.processed_folder = "../../../data/internal/processed/teams/"
        self.folder_map = {
            "raw": self.raw_folder,
            "processed": self.processed_folder,
        }
        self.metadata = self.get_team_metadata()
        self.team_id_map = self.get_team_id_map()
        self.team_name_map = self.get_team_name_map()
        self.all_team_ids = list(np.unique(list(self.team_id_map.values())))
        self.league_id = LeagueID.nba
        """self.franchise_history = (
            FranchiseHistory().get_data_frames()[0].drop(columns=["LEAGUE_ID"])
        )"""

    @property
    def raw_folder(self) -> str:
        """Get raw folder path to team games."""
        return self._raw_folder

    @raw_folder.setter
    def raw_folder(self, folder: str) -> None:
        """Set raw folder path to team games."""
        self._raw_folder = os.path.abspath(folder)

    @property
    def processed_folder(self) -> str:
        """Get processed folder path to team games."""
        return self._processed_folder

    @processed_folder.setter
    def processed_folder(self, folder: str) -> None:
        """Set processed folder path to team games."""
        self._processed_folder = os.path.abspath(folder)

    def _load_team_metadata(self) -> Union[List[Dict[str, Any]], None]:
        """Load NBA Team Metadata from local file.

        Returns:
            Union[List[Dict[str, Any]], None]: team metadata if available else None
        """
        return load_data(pth=os.path.join(self._raw_folder, "team_metadata.pickle"))  # type: ignore

    @cached(cache)
    @retry
    def _retrieve_team_metadata(self) -> List[Dict[str, Any]]:
        """Retrieve NBA team metadata from nba_api.

        Returns:
            List[Dict[str, Any]]: list of dictionaries containing team metadata
        """
        return teams.get_teams()

    def _save_team_metadata(self) -> bool:  # type: ignore
        """Save team metadata to local file.

        Returns:
            bool: whether or not save was successful
        """
        save_data(
            data=self.metadata,
            folder=self._raw_folder,
            name="team_metadata.pickle",
        )

    def get_team_metadata(self) -> List[Dict[str, Any]]:
        """Load team metadata from local file or retrieve from online.

        Returns:
            List[Dict[str, Any]]: list of dictionaries containing team metadata.
        """
        team_metadata = self._load_team_metadata()
        if team_metadata is None:
            team_metadata = self._retrieve_team_metadata()
            save_data(
                data=team_metadata,
                folder=self._raw_folder,
                name="team_metadata.pickle",
            )
        return team_metadata

    def _load_team_id_map(self) -> Union[Dict[str, int], None]:
        """Load Team ID Map from local file.

        Returns:
            Union[Dict[str, int], None]: Mapping to Team ID.
        """
        return load_data(pth=os.path.join(self._processed_folder, "team_id_map.pickle"))  # type: ignore

    def _create_team_id_map(self) -> Dict[str, int]:
        """Create mapping from team name, city and abbreviation to team ID.

        Returns:
            Dict[str, int]: Mapping to Team ID.
        """
        team_id_map: dict = {}
        for team in self.metadata:
            team_id_map[team["nickname"]] = team["id"]
            team_id_map[team["city"]] = team["id"]
            team_id_map[team["abbreviation"]] = team["id"]
        save_data(
            data=team_id_map,
            folder=self._processed_folder,
            name="team_id_map.pickle",
        )
        return team_id_map

    def get_team_id_map(self) -> Dict[str, int]:
        """
        Retrieves mapping from team name, abbreviation and city to team ID
        to ease access of Team IDs for future use.

        Searches locally then retrieves from API if not found.

        Returns:
            Dict[str, int]: Mapping to team ID.
        """
        return self._load_team_id_map() or self._create_team_id_map()

    def _load_team_name_map(self) -> Union[Dict[int, str], None]:
        """Load Team Name Map from local file.

        Returns:
            Union[Dict[int, str], None]: Mapping from Team ID to Team Name.
        """
        return load_data(pth=os.path.join(self._processed_folder, "team_name_map.pickle"))  # type: ignore

    def _create_team_name_map(self) -> Dict[int, str]:
        """Create mapping from team ID to team name (i.e. inverse of team_id_map).

        Returns:
            Dict[int, str]: Mapping from team ID to team name.
        """
        team_name_map = {
            id: name
            for i, (name, id) in enumerate(self.team_id_map.items())
            if i % 3 == 0
        }
        save_data(
            data=team_name_map,
            folder=self._processed_folder,
            name="team_name_map.pickle",
        )
        return team_name_map

    def get_team_name_map(self) -> Dict[int, str]:
        """
        Retrieves mapping from team ID to team name (i.e. inverse of team_id_map).

        Searches locally then retrieves from API if not found.

        Returns:
            Dict[int, str]: Mapping from team ID to team name.
        """
        return self._load_team_name_map() or self._create_team_name_map()
