# %%
# Imports
import numpy as np
from numpy.typing import ArrayLike
from typing import List
import inspect


# %%
# Create object(s) to compute statistics for NBA Teams & Players.
class Stats(object):
    """
    Class to compute statistics using NBA Team/Player data.

    Your life will be easier if you use the same names as the NBA API.

    Sources:
        - General:
            - https://www.basketball-reference.com/about/glossary.html
            - https://www.nba.com/stats/help/glossary
        - Four Factors: https://www.basketball-reference.com/about/factors.html
    """

    def __init__(
        self,
        free_throw_weight: float = 0.44,
        pythagorean_exponent: float = 13.91,  # Morey: 13.91; Hollinger: 16.5
        four_factor_shooting_weight: float = 0.4,
        four_factor_turnover_weight: float = 0.25,
        four_factor_rebounding_weight: float = 0.2,
        four_factor_free_throw_weight: float = 0.15,
    ) -> None:
        self._ft_weight = free_throw_weight
        self._pythagorean_exp = pythagorean_exponent
        self._four_factor_shooting_weight = four_factor_shooting_weight
        self._four_factor_turnover_weight = four_factor_turnover_weight
        self._four_factor_rebounding_weight = four_factor_rebounding_weight
        self._four_factor_free_throw_weight = four_factor_free_throw_weight
        self.independent_stat_method_map = (
            {  # Maps statistics to function for stats that don't require opponent data
                "FG2A": self.two_point_attempts,
                "FG2M": self.two_point_makes,
                "FG2_PCT": self.two_point_pct,
                "2PAr": self.two_point_attempt_rate,
                "3PAr": self.three_point_attempt_rate,
                "eFG_PCT": self.effective_field_goal_pct,
                "TS_PCT": self.true_shooting_pct,
                "MINOR_POSS": self.minor_possessions,
                "MAJOR_POSS": self.major_possessions,
            }
        )
        self.dependent_stat_method_map = {}
        self.required_stat_params = self._get_required_stat_params()

    def two_point_attempts(self, FGA: ArrayLike, FG3A: ArrayLike, **_) -> ArrayLike:
        """Two point attempts (FG2A).

        Args:
            FGA (ArrayLike): field goal attempts\n
            FG3A (ArrayLike): three point attempts\n

        Returns:
            ArrayLike: two point attempts
        """
        return np.subtract(FGA, FG3A)

    def two_point_makes(self, FGM: ArrayLike, FG3M: ArrayLike, **_) -> ArrayLike:
        """Two point makes (FG2M).

        Args:
            FGM (ArrayLike): field goal makes\n
            FG3M (ArrayLike): three point field goal makes\n

        Returns:
            ArrayLike: two point makes
        """
        return np.subtract(FGM, FG3M)

    def two_point_pct(
        self, FGA: ArrayLike, FGM: ArrayLike, FG3A: ArrayLike, FG3M: ArrayLike, **_
    ) -> ArrayLike:
        """Two point percentage (FG2_PCT).

        Args:
            FGM (ArrayLike): field goal makes\n
            FGA (ArrayLike): field goal attempts\n

        Returns:
            ArrayLike: two point percentage
        """
        return np.divide(
            self.two_point_makes(FGM=FGM, FG3M=FG3M),
            self.two_point_attempts(FGA=FGA, FG3A=FG3A),
        )

    def two_point_attempt_rate(self, FGA: ArrayLike, FG3A: ArrayLike, **_) -> ArrayLike:
        """
        Two Point Attempt Rate (2PAr)
            2PAr = 2PA / FGA

        Percent of field goal attempts that are two point attempts.

        Args:
            FGA (ArrayLike): field goal attempts\n
            FG3A (ArrayLike): three point attempts\n

        Returns:
            ArrayLike: two point attempt rate
        """
        return np.divide(self.two_point_attempts(FGA=FGA, FG3A=FG3A), FGA)

    def three_point_attempt_rate(
        self, FGA: ArrayLike, FG3A: ArrayLike, **_
    ) -> ArrayLike:
        """
        Three Point Attempt Rate (3PAr)
            3PAr = 3PA / FGA

        Percent of field goal attempts that are three point attempts.

        Args:
            FGA (ArrayLike): field goal attempts\n
            FG3A (ArrayLike): three point attempts\n

        Returns:
            ArrayLike: three point attempt rate
        """
        return np.divide(FG3A, FGA)

    def effective_field_goal_pct(
        self, FGA: ArrayLike, FGM: ArrayLike, FG3M: ArrayLike, **_
    ) -> ArrayLike:
        """
        Effective Field Goal Percentage (eFG%)
            eFG% = (FGM + 0.5*3PM) / FGA

        Adjusts for the fact that 3PT shots are worth one more point than 2PT shots.

        Args:
            FGA (ArrayLike): field goal attempts\n
            FGM (ArrayLike): field goal makes\n
            FG3M (ArrayLike): three point field goal makes\n

        Returns:
            ArrayLike: effective field goal percentage
        """
        return np.divide(np.add(np.multiply(FGM, 0.5), FG3M), FGA)

    def minor_possessions(
        self, FGA: ArrayLike, FTA: ArrayLike, TOV: ArrayLike, **_
    ) -> ArrayLike:
        """
        Minor Possession estimate.
            MINOR_POSS = FGA + 0.44 * FTA + TOV

        Plays/chances that end in a shot attempt, free throw attempt, or turnover per Kevin Pelton.
        Source: http://www.sonicscentral.com/warp.html

        Note: Some sources add assists and subtract offensive rebounds.
            Source: https://fansided.com/2015/12/21/nylon-calculus-101-possessions/

        Args:
            FGA (ArrayLike): field goal attempts\n
            FTA (ArrayLike): free throe attempts\n
            TOV (ArrayLike): turnovers\n

        Returns:
            ArrayLike: minor possession estimate
        """
        return np.sum(np.array([FGA, np.multiply(FTA, self._ft_weight), TOV]), axis=0)

    def major_possessions(
        self, FGA: ArrayLike, FTA: ArrayLike, TOV: ArrayLike, OREB: ArrayLike, **_
    ) -> ArrayLike:
        """
        Major Possession estimate.
            MAJOR_POSS = FGA + 0.44 * FTA + TOV - OREB

        Continue until a score, defensive rebound or a turnover gives the ball to the other team.
        Source: http://www.sonicscentral.com/warp.html
        Note: tend to overestimate possessions

        Args:
            FGA (ArrayLike): field goal attempts\n
            FTA (ArrayLike): free throw attempts\n
            TOV (ArrayLike): turnovers\n
            OREB (ArrayLike): offensive rebounds\n

        Returns:
            ArrayLike: major possession estimate
        """
        return np.sum(
            np.array(
                [FGA, np.multiply(self._ft_weight, FTA), np.multiply(-1, OREB), TOV]
            ),
            axis=0,
        )

    def _true_shooting_attempts(self, FGA: ArrayLike, FTA: ArrayLike, **_) -> ArrayLike:
        """
        True Shooting Attempts (TSA)
            TSA = FGA + 0.44*FTA

        Args:
            FGA (ArrayLike): field goal attempts\n
            FTA (ArrayLike): free throw attempts\n

        Returns:
            ArrayLike: true shooting attempts
        """
        return np.add(FGA, np.multiply(FTA, self._ft_weight))

    def true_shooting_pct(
        self, PTS: ArrayLike, FGA: ArrayLike, FTA: ArrayLike, **_
    ) -> ArrayLike:
        """
        True Shooting Percentage (TS%)
            TS% = PTS / (2*TSA).

        Measure of shooting efficiency that takes into account 2PT, 3PT, and FT.

        Args:
            PTS (ArrayLike): points\n
            FGA (ArrayLike): field goal attempts\n
            FTA (ArrayLike): free throw attempts\n

        Returns:
            ArrayLike: true shooting percentage
        """
        return np.divide(PTS, self._true_shooting_attempts(FGA=FGA, FTA=FTA))

    def _get_required_stat_params(self) -> List[str]:
        """
        Aggregate list of required parameters for all statistics
        this class can compute.

        Returns:
            List[str]: list of all possible required parameters.
        """
        all_stat_methods = {
            **self.independent_stat_method_map,
            **self.dependent_stat_method_map,
        }
        all_params = []
        for stat_func in all_stat_methods.values():
            all_params.extend(inspect.getfullargspec(stat_func).args)
        all_params = sorted(np.unique(all_params))
        all_params.remove("self")
        return all_params
