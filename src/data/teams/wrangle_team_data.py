"""
Wrangle all team data from the NBA API and save it to the external/raw folder for future processing.
"""
from typing import List, Dict, Any, Union, Literal
from tqdm import tqdm
import pandas as pd
from data.retriever import TeamDataRetriever
from data.cleaner import TeamDataCleaner, TeamGameDataCleaner
from data.saver import TeamGameDataSaver
from data.loader import TeamGameDataLoader


class TeamDataWrangler(object):
    def __init__(self):
        self.retriever = TeamDataRetriever()
        self.cleaner = TeamDataCleaner()
        self.saver = TeamGameDataSaver()
        self.metadata = {}
        self.team_ids = []
        self.team_id_map = {}
        self.team_abbreviation_map = {}

    def _retrieve_save_team_metadata(self) -> List[Dict[str, Any]]:
        """Retrieve team metadata from API and save locally.

        Returns:
            List[Dict[str, Any]]: list of dictionaries containing NBA team metadata.
        """
        metadata = self.retriever.retrieve_team_metadata()
        self.saver.save_team_metadata(metadata)
        return metadata

    def _create_save_team_ids(self, team_id_map: List[Dict[str, Any]]) -> List[int]:
        """Create list of team IDs from team metadata and save locally.

        Args:
            team_metadata (List[Dict[str, Any]]): NBA team metadata as is from API.

        Returns:
            List[int]: list of team IDs.
        """
        team_ids = self.cleaner.extract_team_ids(team_id_map)
        self.saver.save_team_ids(team_ids)
        return team_ids

    def _retrieve_save_team_id_map(
        self, team_metadata: List[Dict[str, Any]]
    ) -> Dict[str, Union[int, str]]:
        """Create team ID map from team metadata and save locally.

        Args:
            team_metadata (List[Dict[str, Any]]): NBA team metadata as is from API.
        Returns:
            Dict[str, Union[int, str]]: dictionary mapping team name, city and abbreviation to team ID.
        """
        team_id_map = self.cleaner.create_team_id_map(team_metadata)
        self.saver.save_team_id_map(team_id_map)
        return team_id_map

    def _retrieve_save_team_abbreviation_map(
        self, team_id_map: Dict[str, Union[int, str]]
    ) -> Dict[int, str]:
        """Create team abbreviation map from team ID map and save locally.

        Args:
            team_id_map (Dict[str, Union[int, str]]): dictionary mapping team name, city and abbreviation to team ID.

        Returns:
            Dict[int, str]: dictionary mapping team ID to team abbreviation.
        """
        team_abbreviation_map = self.cleaner.create_team_abbreviation_map(team_id_map)
        self.saver.save_team_abbreviation_map(team_abbreviation_map)
        return team_abbreviation_map

    def wrangle_team_data(self) -> None:
        """
        Retrieve from NBA API and save locally all team metadata and mappings.
        """
        self.metadata = self._retrieve_save_team_metadata()
        self.team_ids = self._create_save_team_ids(self.metadata)
        self.team_id_map = self._retrieve_save_team_id_map(self.metadata)
        self.team_abbreviation_map = self._retrieve_save_team_abbreviation_map(
            self.team_id_map
        )


class TeamGameDataWrangler(TeamDataWrangler):
    def __init__(self):
        super().__init__()
        self.retriever = TeamDataRetriever()
        self.saver = TeamGameDataSaver()
        self.loader = TeamGameDataLoader()
        self.team_ids = self._get_team_ids()
        self.cleaner = TeamGameDataCleaner(team_ids=self.team_ids)

    def _get_team_ids(self):
        team_ids = self.loader.load_team_ids()
        if team_ids is None:
            self.wrangle_team_data()
        return team_ids

    def _retrieve_save_raw_team_game_data(
        self,
        team_id: int,
        season_type_nullable: str,
    ) -> pd.DataFrame:
        team_games = self.retriever.retrieve_team_games(
            team_id=team_id, season_type_nullable=season_type_nullable
        )
        self.saver.save_team_game_data(
            games=team_games,
            outer_folder="external",
            inner_folder="raw",
            team_id=team_id,
            season_type=season_type_nullable,
        )
        return team_games

    def _save_team_season_game_data(
        self,
        team_games: pd.DataFrame,
        team_id: int,
        season_type_nullable: str,
        outer_folder: Literal["external", "internal"],
        inner_folder: Literal["raw", "interim", "processed", "inputation"],
    ) -> None:
        for season in team_games.SEASON_ID.unique():
            season_games = team_games[team_games.SEASON_ID == season].copy()
            self.saver.save_team_season_game_data(
                season_games=season_games,
                outer_folder=outer_folder,
                inner_folder=inner_folder,
                team_id=team_id,
                season_type=season_type_nullable,
                season=season,
            )

    def wrangle_team_game_data(
        self,
        season_types: List[Literal["Pre Season", "Regular Season", "Playoffs"]] = [
            "Regular Season",
            "Playoffs",
        ],
    ) -> None:
        # TODO: make this rely on SeasonTypeNullable attributes
        # TODO: add preseason data
        for season_type in tqdm(season_types):
            for team_id in self.team_ids:
                raw_team_games = self._retrieve_save_raw_team_game_data(
                    team_id=team_id, season_type_nullable=season_type
                )
                self._save_team_season_game_data(
                    team_games=raw_team_games,
                    team_id=team_id,
                    season_type_nullable=season_type,
                    outer_folder="external",
                    inner_folder="raw",
                )
                clean_team_games = self.cleaner.clean_team_game_data(raw_team_games)
                self._save_team_season_game_data(
                    team_games=clean_team_games,
                    team_id=team_id,
                    season_type_nullable=season_type,
                    outer_folder="external",
                    inner_folder="interim",
                )
                inputed_team_games = self.cleaner.inpute_team_game_data(
                    team_id=team_id, team_games=clean_team_games
                )
                self._save_team_season_game_data(
                    team_games=inputed_team_games,
                    team_id=team_id,
                    season_type_nullable=season_type,
                    outer_folder="external",
                    inner_folder="processed",
                )


if __name__ == "__main__":
    team_data_wrangler = TeamGameDataWrangler()
    team_data_wrangler.wrangle_team_data()
    team_data_wrangler.wrangle_team_game_data()
