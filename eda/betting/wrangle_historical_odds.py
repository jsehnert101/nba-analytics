# %%
# Imports
from typing import Literal, Dict, Any, Union, Tuple, List
import numpy as np
import pandas as pd
import requests
import json
from datetime import datetime
from utils.env import get_api_key

api_key = get_api_key()


# %%
# Write object to manage retrieval of NBA odds data
class OddsDataRetriever(object):
    def __init__(self):
        self._api_key = get_api_key()
        self.url_schema = (
            "https://api.the-odds-api.com/v4/sports/basketball_nba/odds-history/"
            "?apiKey={api_key}&regions=us&markets={markets}&dateFormat=iso"
            "&bookmakers={bookmakers}&oddsFormat={odds_format}&date={date}"
        )

    def _convert_to_isoformat(self, dt: datetime) -> str:
        """Convert datetime to isoformat.

        Args:
            dt (datetime): datetime object

        Returns:
            str: isoformat string
        """
        return dt.isoformat(timespec="seconds") + "Z"

    def _convert_datetime(
        self, dt: Union[str, pd.Series]
    ) -> Union[datetime, pd.Series]:
        if isinstance(dt, str):
            return pd.to_datetime(dt).tz_convert("US/Eastern").tz_localize(None)
        else:
            return pd.to_datetime(dt).dt.tz_convert("US/Eastern").dt.tz_localize(None)

    def _extract_timestamps(self, json_data: Dict[str, Any]) -> Dict[str, datetime]:
        """Extract timestamps from json data to facilitate requests.

        Args:
            json_data (Dict[str, Any]): json data from request

        Returns:
            Dict: adjacent timestamps to current request
        """
        keys = ["timestamp", "previous_timestamp", "next_timestamp"]
        return {k: self._convert_datetime(json_data[k]) for k in keys}  # type: ignore

    def _convert_json_to_df(self, odds_data: Dict) -> pd.DataFrame:
        df_meta_odds = pd.json_normalize(
            odds_data,
            record_path="bookmakers",
            meta=["id", "commence_time", "home_team", "away_team"],
        )
        df_game_odds = df_meta_odds.markets.explode().apply(pd.Series)
        df_team_odds = df_game_odds.outcomes.explode().apply(pd.Series)
        df_odds = df_meta_odds.drop(columns=["key", "last_update", "markets"]).merge(
            df_game_odds.drop(columns=["outcomes"]).merge(
                df_team_odds, left_index=True, right_index=True
            ),
            left_index=True,
            right_index=True,
        )
        df_odds.loc[:, ["commence_time", "last_update"]] = df_odds.loc[
            :, ["commence_time", "last_update"]
        ].apply(self._convert_datetime, raw=False)
        return df_odds

    def _join_list(self, l: List[str]) -> str:
        """Proper string formatting for request."""
        return "%2C".join(l)

    def _compute_implied_probability(
        self, df_odds: pd.DataFrame, odds_format: Literal["decimal", "american"]
    ) -> pd.DataFrame:
        df_odds["implied_probability"] = np.NaN
        if odds_format == "decimal":
            df_odds.loc[:, "implied_probability"] = 1 / df_odds.odds
        elif odds_format == "american":
            df_odds.loc[df_odds.odds.gt(0), "implied_probability"] = 100 / (
                df_odds.loc[df_odds.odds.gt(0), "odds"].add(100)
            )
            df_odds.loc[df_odds.odds.lt(0), "implied_probability"] = (
                df_odds.loc[df_odds.odds.lt(0), "odds"]
                .multiply(-1)
                .divide(df_odds.loc[df_odds.odds.lt(0), "odds"].multiply(-1).add(100))
            )
        return df_odds

    def _clean_game_odds(
        self, df_odds: pd.DataFrame, odds_format: Literal["decimal", "american"]
    ) -> pd.DataFrame:
        # TODO: remove redundancies in odds info across rows
        # TODO: add TEAM + GAME IDs to align with NBA API
        df_odds.rename(
            columns={"title": "book", "key": "prop", "name": "team", "price": "odds"},
            inplace=True,
        )
        df_odds = self._compute_implied_probability(df_odds, odds_format)
        return df_odds

    def retrieve_game_data(
        self,
        date: datetime,
        markets: List[str] = ["h2h", "spreads", "totals"],
        bookmakers: List[str] = ["draftkings", "fanduel"],
        odds_format: Literal["decimal", "american"] = "american",
    ) -> Tuple[pd.DataFrame, Dict]:
        """Retrieve game odds data for a given datetime.

        Args:
            date (datetime): game date
            markets (List[str], optional): betting options of interest. Defaults to ["h2h", "spreads", "totals"].
            bookmakers (List[str], optional): sportsbook source of odds. Defaults to ["draftkings", "fanduel"].
            odds_format (Literal["decimal", "american"], optional): odds style. Defaults to "american".

        Returns:
            Tuple[pd.DataFrame, Dict]: game odds data and adjacent response timestamps
        """
        date_str = self._convert_to_isoformat(date)
        url = self.url_schema.format(
            api_key=self._api_key,
            markets=self._join_list(markets),
            bookmakers=self._join_list(bookmakers),
            odds_format=odds_format,
            date=date_str,
        )
        response = requests.get(url, timeout=60)
        assert response.status_code == 200
        json_data = response.json()
        response_timestamps = self._extract_timestamps(json_data)
        odds_data = json_data["data"][0]
        df_odds = self._convert_json_to_df(odds_data)
        df_odds = self._clean_game_odds(df_odds, odds_format)
        return (df_odds, response_timestamps)


# %%
# Retrieve given game's odds
odds_retriever = OddsDataRetriever()
date = datetime(2023, 5, 27)


# %%
# Figure out where reduncancies occur
url = odds_retriever.url_schema.format(
    api_key=odds_retriever._api_key,
    markets=odds_retriever._join_list(["h2h", "spreads", "totals"]),
    bookmakers=odds_retriever._join_list(["draftkings", "fanduel"]),
    odds_format="american",
    date=odds_retriever._convert_to_isoformat(date),
)
resp = requests.get(url, timeout=60)
assert resp.status_code == 200
json_data = resp.json()
odds_data = json_data["data"][0]
df_meta_odds = pd.json_normalize(
    odds_data,
    record_path="bookmakers",
    meta=["id", "commence_time", "home_team", "away_team"],
)
df_meta_odds

# %%
df_game_odds = df_meta_odds.markets.explode().apply(pd.Series)
df_game_odds
# %%
df_team_odds = df_game_odds.outcomes.explode().apply(pd.Series)
df_team_odds
# %%
df_odds = df_meta_odds.drop(columns=["key", "last_update", "markets"]).merge(
    df_game_odds.drop(columns=["outcomes"]).merge(
        df_team_odds, left_index=True, right_index=True
    ),
    left_index=True,
    right_index=True,
)
df_odds

# %%
df_odds, response_timestamps = odds_retriever.retrieve_game_data(date)
df_odds

# %%
# TODO: append TEAM + GAME IDs to df
# TODO: remove redundancies in data
df_odds = pd.read_parquet("../../data/internal/interim/odds.parquet")
df_odds
# %%
temp_df = df_meta_odds.markets.apply(lambda x: pd.json_normalize(x[0]))
temp_df
# %%
