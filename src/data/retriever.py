""" Data Retrieval Module"""
from typing import List, Dict, Any
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.library.parameters import LeagueID, SeasonTypeNullable
import pandas as pd
from cachetools import cached
from utils.data import retry, cache


class DataRetriever:
    """Base class for data retrieval objects."""

    def __init__(self):
        self.league_id = LeagueID.nba


class TeamDataRetriever(DataRetriever):
    """Retrieve NBA team data from nba-api."""

    @cached(cache)
    @retry
    def retrieve_team_metadata(self) -> List[Dict[str, Any]]:
        """Retrieves team metadata from nba-api.

        Returns:
            List[Dict[str, Any]]: NBA team metadata.
        """
        return teams.get_teams()

    @cached(cache)
    @retry
    def retrieve_team_games(
        self, team_id: int, season_type_nullable: str = SeasonTypeNullable.regular
    ) -> pd.DataFrame:
        """Retrieves df of all games of a specified type for a given NBA team.

        Args:
            team_id (int): NBA Team ID.
            season_type_nullable (SeasonTypeNullable, optional): Defaults to SeasonTypeNullable.regular.

        Returns:
            pd.DataFrame: df of all team games and accompanying stats.
        """
        return leaguegamefinder.LeagueGameFinder(
            league_id_nullable=self.league_id,
            team_id_nullable=str(team_id),
            season_type_nullable=season_type_nullable,
            timeout=60,
        ).get_data_frames()[0]


if __name__ == "__main__":
    pass
