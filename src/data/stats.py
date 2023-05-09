# %%
# Imports
import numpy as np
from numpy.typing import ArrayLike
from typing import Union


# %%
# Create object(s) to compute statistics for NBA Teams & Players.
class Stats:
    """
    Class to compute statistics for NBA Team/Player data.
    Source: https://www.basketball-reference.com/about/glossary.html
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

    def two_point_attempts(
        self, FGA: Union[float, int, ArrayLike], FG3A: Union[float, int, ArrayLike]
    ) -> Union[float, int, ArrayLike]:
        """Two point attempts (FG2A).

        Args:
            FGA (Union[float, int, ArrayLike]): field goal attempts\n
            FG3A (Union[float, int, ArrayLike]): three point attempts\n

        Returns:
            Union[float, int, ArrayLike]: two point attempts
        """
        return np.subtract(FGA, FG3A)

    def two_point_makes(
        self, FGM: Union[float, int, ArrayLike], FG3M: Union[float, int, ArrayLike]
    ) -> Union[float, int, ArrayLike]:
        """Two point makes (FG2M).

        Args:
            FGM (Union[float, int, ArrayLike]): field goal makes\n
            FG3M (Union[float, int, ArrayLike]): three point field goal makes\n

        Returns:
            Union[float, int, ArrayLike]: two point makes
        """
        return np.subtract(FGM, FG3M)

    def two_point_pct(
        self,
        FGA: Union[float, int, ArrayLike],
        FGM: Union[float, int, ArrayLike],
        FG3A: Union[float, int, ArrayLike],
        FG3M: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """Two point percentage (FG2_PCT).

        Args:
            FGA (Union[float, int, ArrayLike]): field goal attempts\n
            FGM (Union[float, int, ArrayLike]): field goal makes\n
            FG3A (Union[float, int, ArrayLike]): three point field goal attempts\n
            FG3M (Union[float, int, ArrayLike]): three point field goal makes\n

        Returns:
            Union[float, int, ArrayLike]: two point percentage
        """
        return np.divide(
            self.two_point_makes(FGM=FGM, FG3M=FG3M),
            self.two_point_attempts(FGA=FGA, FG3A=FG3A),
        )

    def two_point_attempt_rate(
        self, FGA: Union[float, int, ArrayLike], FG2A: Union[float, int, ArrayLike]
    ) -> Union[float, int, ArrayLike]:
        """
        Two Point Attempt Rate (2PAr)
            2PAr = 2PA / FGA

        Percent of field goal attempts that are two point attempts.

        Args:
            FGA (Union[float, int, ArrayLike]): field goal attempts\n
            FG2A (Union[float, int, ArrayLike]): two point attempts\n

        Returns:
            Union[float, int, ArrayLike]: two point attempt rate
        """
        return np.divide(self.two_point_attempts(FGA=FGA, FG3A=FG2A), FGA)

    def three_point_attempt_rate(
        self, FGA: Union[float, int, ArrayLike], FG3A: Union[float, int, ArrayLike]
    ) -> Union[float, int, ArrayLike]:
        """
        Three Point Attempt Rate (3PAr)
            3PAr = 3PA / FGA

        Percent of field goal attempts that are three point attempts.

        Args:
            FGA (Union[float, int, ArrayLike]): field goal attempts\n
            FG3A (Union[float, int, ArrayLike]): three point attempts\n

        Returns:
            Union[float, int, ArrayLike]: three point attempt rate
        """
        return np.divide(FG3A, FGA)

    def effective_field_goal_pct(
        self,
        FGA: Union[float, int, ArrayLike],
        FGM: Union[float, int, ArrayLike],
        FG3M: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Effective Field Goal Percentage (eFG%)
            eFG% = (FGM + 0.5*3PM) / FGA

        Adjusts for the fact that 3PT shots are worth one more point than 2PT shots.

        Args:
            FGA (Union[float, int, ArrayLike]): field goal attempts\n
            FGM (Union[float, int, ArrayLike]): field goal makes\n
            FG3M (Union[float, int, ArrayLike]): three point field goal makes\n

        Returns:
            Union[float, int, ArrayLike]: effective field goal percentage
        """
        return np.divide(np.add(np.multiply(FGM, 0.5), FG3M), FGA)

    def _true_shooting_attempts(
        self, FGA: Union[float, int, ArrayLike], FTA: Union[float, int, ArrayLike]
    ) -> Union[float, int, ArrayLike]:
        """
        True Shooting Attempts (TSA)
            TSA = FGA + 0.44*FTA

        Args:
            FGA (Union[float, int, ArrayLike]): field goal attempts\n
            FTA (Union[float, int, ArrayLike]): free throw attempts\n

        Returns:
            Union[float, int, ArrayLike]: true shooting attempts
        """
        return np.add(FGA, np.multiply(FTA, self._ft_weight))

    def true_shooting_pct(
        self,
        PTS: Union[float, int, ArrayLike],
        FGA: Union[float, int, ArrayLike],
        FTA: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        True Shooting Percentage (TS%)
            TS% = PTS / (2*TSA).

        Measure of shooting efficiency that takes into account 2PT, 3PT, and FT.

        Args:
            PTS (Union[float, int, ArrayLike]): points\n
            FGA (Union[float, int, ArrayLike]): field goal attempts\n
            FTA (Union[float, int, ArrayLike]): free throw attempts\n

        Returns:
            Union[float, int, ArrayLike]: true shooting percentage
        """
        return np.divide(PTS, self._true_shooting_attempts(FGA=FGA, FTA=FTA))

    def minor_possessions(
        self,
        FGA: Union[float, int, ArrayLike],
        FTA: Union[float, int, ArrayLike],
        TOV: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Minor Possession estimate.
            MINOR_POSS = FGA + 0.44 * FTA + TOV

        Plays/chances that end in a shot attempt, free throw attempt, or turnover per Kevin Pelton.
        Source: http://www.sonicscentral.com/warp.html

        Note: Some sources add assists and subtract offensive rebounds.
            Source: https://fansided.com/2015/12/21/nylon-calculus-101-possessions/

        Args:
            FGA (Union[float, int, ArrayLike]): field goal attempts\n
            FTA (Union[float, int, ArrayLike]): free throe attempts\n
            TOV (Union[float, int, ArrayLike]): turnovers\n

        Returns:
            Union[float, int, ArrayLike]: minor possession estimate
        """
        return np.sum(np.array([FGA, np.multiply(FTA, self._ft_weight), TOV]), axis=0)

    def major_possessions(
        self,
        FGA: Union[float, int, ArrayLike],
        FTA: Union[float, int, ArrayLike],
        TOV: Union[float, int, ArrayLike],
        OREB: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Major Possession estimate.
            MAJOR_POSS = FGA + 0.44 * FTA + TOV - OREB

        Continue until a score, defensive rebound or a turnover gives the ball to the other team.
        Source: http://www.sonicscentral.com/warp.html
        Note: tend to overestimate possessions

        Args:
            FGA (Union[float, int, ArrayLike]): field goal attempts\n
            FTA (Union[float, int, ArrayLike]): free throw attempts\n
            TOV (Union[float, int, ArrayLike]): turnovers\n
            OREB (Union[float, int, ArrayLike]): offensive rebounds\n

        Returns:
            Union[float, int, ArrayLike]: major possession estimate
        """
        return np.sum(
            np.array(
                [FGA, np.multiply(self._ft_weight, FTA), np.multiply(-1, OREB), TOV]
            )
        )
