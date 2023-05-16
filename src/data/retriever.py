""" Data Retrieval Module"""
from typing import List, Dict, Any
from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.library.parameters import LeagueID, SeasonTypeNullable
import pandas as pd
from cachetools import cached
from utils.data import retry, cache


# Base class
class DataRetriever(object):
    def __init__(self):
        self.league_id = LeagueID.nba


# Team data
class TeamDataRetriever(DataRetriever):
    @cached(cache)
    @retry
    def retrieve_team_metadata(self) -> List[Dict[str, Any]]:
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


# Player data
