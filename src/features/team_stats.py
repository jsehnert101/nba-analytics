# %%
# Imports
import inspect
from typing import List
import numpy as np
from numpy.typing import ArrayLike
from features.stats import Stats

# %%
# Define object to compute NBA team statistics.


class TeamStats(Stats):
    """Compute statistics for NBA teams.

    Args:
        Stats (_type_): Generic class that computes statistics.
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
        super().__init__(
            free_throw_weight=free_throw_weight,
            pythagorean_exponent=pythagorean_exponent,
            four_factor_free_throw_weight=four_factor_free_throw_weight,
            four_factor_rebounding_weight=four_factor_rebounding_weight,
            four_factor_shooting_weight=four_factor_shooting_weight,
            four_factor_turnover_weight=four_factor_turnover_weight,
        )
        self.independent_stat_method_map.update(
            {
                "TOV_PCT": self.turnover_pct,
                "SHOOTING_FACTOR": self.shooting_factor,
                "TOV_FACTOR": self.turnover_factor,
                "FT_FACTOR": self.free_throw_factor,
            }
        )
        self.dependent_stat_method_map = {  # Track methods which require opponent data
            "PLUS_MINUS": self.plus_minus
            "REB_PCT": self.rebound_pct,
            "DREB_PCT": self.defensive_rebound_pct,
            "OREB_PCT": self.offensive_rebound_pct,
            "PYTHAG_WINS": self.pythagorean_win_pct,
            "REB_FACTOR": self.rebound_factor,
            "FOUR_FACTOR_SCORE": self.four_factor_score,
            "POSS": self.possessions,
            "PACE": self.pace,
            "OFF_RATING": self.offensive_rating,
            "DEF_RATING": self.defensive_rating,
        }
        self.required_stat_params = self._get_required_stat_params()

    def plus_minus(self, PTS: ArrayLike, OPP_PTS: ArrayLike, **_) -> ArrayLike:
        """Plus Minus (PLUS_MINUS)
            PLUS_MINUS = PTS - OPP_PTS
        Args:
            PTS (ArrayLike): points\n
            OPP_PTS (ArrayLike): opponent points against\n

        Returns:
            ArrayLike: team point differential
        """
        return np.subtract(PTS, OPP_PTS)

    def rebound_pct(self, REB: ArrayLike, OPP_REB: ArrayLike, **_) -> ArrayLike:
        """
        Rebounding Percentage (REB%)
            REB% = REB / (REB + OPP_REB)

        Percentage of rebounds a team grabs while on the floor.

        Args:
            REB (ArrayLike): rebounds\n
            OPP_REB (ArrayLike): opponent rebounds against\n

        Returns:
            ArrayLike: percentage of rebounds a team grabbed while on the floor.
        """
        return np.divide(REB, np.add(REB, OPP_REB))

    def defensive_rebound_pct(
        self, DREB: ArrayLike, OPP_OREB: ArrayLike, **_
    ) -> ArrayLike:
        """
        Defensive Rebounding Percentage (DREB%)
            DREB% = DREB / (DREB + OPP_OREB)

        Percentage of defensive rebounds a team grabs while on the floor.

        Args:
            DREB (ArrayLike): defensive rebounds\n
            OPP_OREB (ArrayLike): opponent offensive rebounds against\n

        Returns:
            ArrayLike: defensive rebounding rate
        """
        return np.divide(DREB, np.add(DREB, OPP_OREB))

    def offensive_rebound_pct(
        self, OREB: ArrayLike, OPP_DREB: ArrayLike, **_
    ) -> ArrayLike:
        """
        Offensive Rebounding Percentage (OREB%)
            OREB% = OREB / (OREB + OPP_DREB)

        Percentage of offensive rebounds a team grabs while on the floor.

        Args:
            OREB (ArrayLike): offensive rebounds\n
            OPP_DREB (ArrayLike): opponent defensive rebounds against\n

        Returns:
            ArrayLike: offensive rebounding rate
        """
        return np.divide(OREB, np.add(OREB, OPP_DREB))

    def turnover_pct(
        self, FGA: ArrayLike, FTA: ArrayLike, TOV: ArrayLike, **_
    ) -> ArrayLike:
        """
        Turnover Percentage (TOV%)
            TOV% = TOV / (FGA + 0.44 * FTA + TOV)

        Estimates percentage of team possessions that end in a turnover.

        Args:
            FGA (ArrayLike): field goal attempts\n
            FTA (ArrayLike): free throw attempts\n
            TOV (ArrayLike): turnovers\n

        Returns:
            ArrayLike: turnover rate
        """
        return np.divide(TOV, self.minor_possessions(FGA=FGA, FTA=FTA, TOV=TOV))

    def pythagorean_win_pct(self, PTS: ArrayLike, OPP_PTS: ArrayLike, **_) -> ArrayLike:
        """
        Pythagorean Expected Win Percentage
            = PTS**EXP / (PTS**EXP + PTS_ALLOWED**EXP)

        Estimates what a team's win percentage "should be."
        The formula was obtained by fitting a logistic regression model: WIN_PCT ~ log(PTS / OPP_PTS)
        Source: https://captaincalculator.com/sports/basketball/pythagorean-win-percentage-calculator/
        Paper: https://web.williams.edu/Mathematics/sjmiller/public_html/399/handouts/PythagWonLoss_Paper.pdf

        Args:
            PTS (ArrayLike): points\n
            OPP_PTS (ArrayLike): opponent points against\n

        Returns:
            ArrayLike: team expected win percentage
        """
        return np.divide(
            np.power(PTS, self._pythagorean_exp),
            np.add(
                np.power(PTS, self._pythagorean_exp),
                np.power(OPP_PTS, self._pythagorean_exp),
            ),
        )

    def shooting_factor(
        self, FGA: ArrayLike, FGM: ArrayLike, FG3M: ArrayLike, **_
    ) -> ArrayLike:
        """
        Dean Oliver's Shooting Factor (eFG%).

        Args:
            FGA (ArrayLike): field goal attempts\n
            FGM (ArrayLike): field goal makes\n
            FG3M (ArrayLike): three point field goal makes\n

        Returns:
            ArrayLike: Shooting Factor / eFG%
        """
        return self.effective_field_goal_pct(FGM=FGM, FG3M=FG3M, FGA=FGA)

    def turnover_factor(
        self, TOV: ArrayLike, FGA: ArrayLike, FTA: ArrayLike, **_
    ) -> ArrayLike:
        """
        Dean Oliver's Turnover Factor (TO%).

        Args:
            FGA (ArrayLike): field goal attempts\n
            FTA (ArrayLike): free throw attempts\n
            TOV (ArrayLike): turnovers\n

        Returns:
            ArrayLike: Turnover Factor / TO%
        """
        return self.turnover_pct(TOV=TOV, FGA=FGA, FTA=FTA)

    def rebound_factor(self, REB: ArrayLike, OPP_REB: ArrayLike, **_) -> ArrayLike:
        """
        Dean Oliver's Rebound Factor.
            REB_FACTOR = REB / (REB + OPP_REB) = REB%

        Args:
            REB (ArrayLike): rebounds\n
            OPP_REB (ArrayLike): opponent rebounds against\n

        Returns:
            ArrayLike: Rebound Factor / REB%
        """
        return self.rebound_pct(REB=REB, OPP_REB=OPP_REB)

    def free_throw_factor(self, FTM: ArrayLike, FGA: ArrayLike, **_) -> ArrayLike:
        """
        Dean Oliver's Free Throw Factor.
            FT_FACTOR = FTM / FGA

        Estimates how often team gets to the line and how often they make them.
        Source: https://www.basketball-reference.com/about/factors.html

        Args:
            FT (ArrayLike): free throws\n
            FGA (ArrayLike): field goal attempts\n

        Returns:
            ArrayLike: Free Throw Factor.
        """
        return np.divide(FTM, FGA)

    def four_factor_score(
        self,
        FGA: ArrayLike,
        FGM: ArrayLike,
        FG3M: ArrayLike,
        FTM: ArrayLike,
        FTA: ArrayLike,
        REB: ArrayLike,
        OPP_REB: ArrayLike,
        TOV: ArrayLike,
        **_
    ) -> ArrayLike:
        """
        Compute Dean Oliver's Four Factors, which summarize a team's strengths and weaknesses.\n
        May be applied offensively or defensively.\n
        Source: https://www.basketball-reference.com/about/factors.html

        Args:
            FGA (ArrayLike): field goal attempts.\n
            FGM (ArrayLike): field goal makes.\n
            FG3M (ArrayLike): three point field goal makes.\n
            FTM (ArrayLike): free throw makes\n
            FTA (ArrayLike): free throw attempts\n
            REB (ArrayLike): rebounds\n
            OPP_REB (ArrayLike): opponent rebounds against\n
            TOV (ArrayLike): turnovers\n

        Returns:
            ArrayLike: Four Factor Score.
        """
        shooting_factor = self.shooting_factor(FGM=FGM, FG3M=FG3M, FGA=FGA)
        turnover_factor = self.turnover_factor(TOV=TOV, FGA=FGA, FTA=FTA)
        rebound_factor = self.rebound_factor(REB=REB, OPP_REB=OPP_REB)
        free_throw_factor = self.free_throw_factor(FTM=FTM, FGA=FGA)
        return np.sum(
            [
                np.multiply(shooting_factor, self._four_factor_shooting_weight),
                np.multiply(turnover_factor, self._four_factor_turnover_weight),
                np.multiply(rebound_factor, self._four_factor_rebounding_weight),
                np.multiply(free_throw_factor, self._four_factor_free_throw_weight),
            ],
            axis=0,
        )

    def _team_possessions(
        self,
        FGA: ArrayLike,
        FGM: ArrayLike,
        FTA: ArrayLike,
        OREB: ArrayLike,
        OPP_DREB: ArrayLike,
        TOV: ArrayLike,
        **_
    ) -> ArrayLike:
        """
        Compute team possession estimate to weight in team possession stat.
            TEAM_POSS = FGA - 0.44 * FTA - 1.07 * OREB% * (FGA - FGM) + TOV

        Source: https://www.basketball-reference.com/about/glossary.html#poss

        Args:
            FGA (ArrayLike): field goal attempts
            FGM (ArrayLike): field goal makes
            FTA (ArrayLike): free throw attempts
            OREB (ArrayLike): offensive rebounds
            OPP_DREB (ArrayLike): opponent defensive rebounds
            TOV (ArrayLike): turnovers

        Returns:
            ArrayLike: estimate of team # possessions
        """
        return np.sum(
            np.array(
                [
                    FGA,
                    np.multiply(self._ft_weight, FTA),
                    np.multiply(
                        -1.07,
                        np.multiply(
                            self.offensive_rebound_pct(OREB=OREB, OPP_DREB=OPP_DREB),
                            np.subtract(FGA, FGM),
                        ),
                    ),
                    TOV,
                ]
            ),
            axis=0,
        )

    def possessions(
        self,
        FGA: ArrayLike,
        FGM: ArrayLike,
        FTA: ArrayLike,
        DREB: ArrayLike,
        OREB: ArrayLike,
        TOV: ArrayLike,
        OPP_FGA: ArrayLike,
        OPP_FGM: ArrayLike,
        OPP_FTA: ArrayLike,
        OPP_OREB: ArrayLike,
        OPP_DREB: ArrayLike,
        OPP_TOV: ArrayLike,
        **_
    ) -> ArrayLike:
        """
        Possessions (POSS)
            POSS = 0.5 * (TEAM_POSS + OPP_POSS)

        Estimates possessions based on team + opponent statistics and
        averages each to provide a more stable estimate.
        Source: https://www.basketball-reference.com/about/glossary.html#poss

        Note: NBA.com & ESPN.com use different formulas.
            Formula: (FGA + 0.44*FTA - OREB + TOV) / 2
            Source: https://fansided.com/2015/12/21/nylon-calculus-101-possessions/

        Args:
            FGA (ArrayLike): field goal attempts\n
            FGM (ArrayLike): field goal makes\n
            FTA (ArrayLike): free throw attempts\n
            DREB (ArrayLike): defensive rebounds\n
            OREB (ArrayLike): offensive rebounds\n
            TOV (ArrayLike): turnovers\n
            OPP_FGA (ArrayLike): opponent field goal attempts\n
            OPP_FGM (ArrayLike): opponent field goal makes\n
            OPP_FTA (ArrayLike): opponent free throw attempts\n
            OPP_OREB (ArrayLike): opponent offensive rebounds\n
            OPP_DREB (ArrayLike): opponent defensive rebounds\n
            OPP_TOV (ArrayLike): opponent turnovers\n

        Returns:
            ArrayLike: possessions as
        """
        return np.multiply(
            0.5,
            np.add(
                self._team_possessions(
                    FGA=FGA, FGM=FGM, FTA=FTA, OREB=OREB, OPP_DREB=OPP_DREB, TOV=TOV
                ),
                self._team_possessions(
                    FGA=OPP_FGA,
                    FGM=OPP_FGM,
                    FTA=OPP_FTA,
                    OREB=OPP_OREB,
                    OPP_DREB=DREB,
                    TOV=OPP_TOV,
                ),
            ),
        )

    def espn_possessions(
        self, FGA: ArrayLike, FTA: ArrayLike, OREB: ArrayLike, TOV: ArrayLike, **_
    ) -> ArrayLike:
        """
        Possessions (POSS) using nba.com/espn.com formulas
            POSS = MAJOR POSSESSIONS / 2 = (FGA + 0.44*FTA - OREB + TOV) / 2

        Source: https://fansided.com/2015/12/21/nylon-calculus-101-possessions/
        Note: tend to overestimate possessions

        Args:
            FGA (ArrayLike): field goal attempts\n
            FTA (ArrayLike): free throw attempts\n
            OREB (ArrayLike): offensive rebounds\n
            TOV (ArrayLike): turnovers\n

        Returns:
            ArrayLike: team possessions according to NBA/ESPN.com.
        """
        return np.divide(
            self.major_possessions(FGA=FGA, FTA=FTA, TOV=TOV, OREB=OREB), 2
        )

    def nylon_calculus_possessions(
        self, FGA: ArrayLike, FT_TRIPS: ArrayLike, OREB: ArrayLike, TOV: ArrayLike, **_
    ) -> ArrayLike:
        """
        Possessions (POSS) using Nylon Calculus formula
            POSS = FGA + FT_TRIPS - OREB + TOV  #TODO: verify if this should be divided by 2

        Source: https://fansided.com/2015/12/21/nylon-calculus-101-possessions/
        Note: FT_TRIPS are pairs/triplets extracted from game logs.

        Args:
            FGA (ArrayLike): field goal attempts\n
            FTA (ArrayLike): free throw attempts\n
            OREB (ArrayLike): offensive rebounds\n
            TOV (ArrayLike): turnovers\n

        Returns:
            ArrayLike: team possessions according to nylon calculus.
        """
        return np.divide(
            np.sum(np.array([FGA, FT_TRIPS, np.multiply(-1, OREB), TOV]), axis=0), 2
        )

    def pace(
        self,
        FGA: ArrayLike,
        FGM: ArrayLike,
        FTA: ArrayLike,
        DREB: ArrayLike,
        OREB: ArrayLike,
        TOV: ArrayLike,
        MP: ArrayLike,
        OPP_FGA: ArrayLike,
        OPP_FGM: ArrayLike,
        OPP_FTA: ArrayLike,
        OPP_OREB: ArrayLike,
        OPP_DREB: ArrayLike,
        OPP_TOV: ArrayLike,
        **_  # TODO: Override function to take possessions as input
    ) -> ArrayLike:
        """
        Pace Factor (PACE)
            PACE = 48 * ((POSS + OPP_POSS) / (2 * (MP / 5)))

        Estimates number of possessions per 48 minutes by a team.
        Created by Dean Oliver.
        Source: https://www.basketball-reference.com//about/glossary.html#pace

        Args:
            FGA (ArrayLike): field goal makes\n
            FGM (ArrayLike): field goal attempts\n
            FTA (ArrayLike): free throw attempts\n
            DREB (ArrayLike): defensive rebounds\n
            OREB (ArrayLike): offensive rebounds\n
            TOV (ArrayLike): turnovers\n
            MP (ArrayLike): minutes played\n
            OPP_FGA (ArrayLike): opponent field goal attempts\n
            OPP_FGM (ArrayLike): opponent field goal makes\n
            OPP_FTA (ArrayLike): opponent free throw attempts\n
            OPP_OREB (ArrayLike): opponent offensive rebounds\n
            OPP_DREB (ArrayLike): opponent defensive rebounds\n
            OPP_TOV (ArrayLike): opponent turnovers\n

        Returns:
            ArrayLike: team pace
        """
        poss = self.possessions(
            FGA=FGA,
            FGM=FGM,
            FTA=FTA,
            DREB=DREB,
            OREB=OREB,
            TOV=TOV,
            OPP_FGA=OPP_FGA,
            OPP_FGM=OPP_FGM,
            OPP_FTA=OPP_FTA,
            OPP_OREB=OPP_OREB,
            OPP_DREB=OPP_DREB,
            OPP_TOV=OPP_TOV,
        )
        opp_poss = self.possessions(
            FGA=OPP_FGA,
            FGM=OPP_FGM,
            FTA=OPP_FTA,
            DREB=OPP_DREB,
            OREB=OPP_OREB,
            TOV=OPP_TOV,
            OPP_FGA=FGA,
            OPP_FGM=FGM,
            OPP_FTA=FTA,
            OPP_OREB=OREB,
            OPP_DREB=DREB,
            OPP_TOV=TOV,
        )
        return np.multiply(
            48, np.divide(np.add(poss, opp_poss), np.multiply(2, np.divide(MP, 5)))
        )

    def offensive_rating(  # TODO: Overload function to take possessions as input
        self,
        PTS: ArrayLike,
        FGA: ArrayLike,
        FGM: ArrayLike,
        FTA: ArrayLike,
        DREB: ArrayLike,
        OREB: ArrayLike,
        TOV: ArrayLike,
        OPP_FGA: ArrayLike,
        OPP_FGM: ArrayLike,
        OPP_FTA: ArrayLike,
        OPP_OREB: ArrayLike,
        OPP_DREB: ArrayLike,
        OPP_TOV: ArrayLike,
        **_
    ) -> ArrayLike:
        """
        Team Offensive Rating (OffRtg)
            OffRtg = 100 * (PTS / POSS)

        Estimates how many points a team scores per 100 possessions.
        Source: https://hackastat.eu/en/learn-a-stat-team-offensive-defensive-and-net-rating/

        Args:
            PTS (ArrayLike): points\n
            FGA (ArrayLike): field goal attempts\n
            FGM (ArrayLike): field goal makes\n
            FTA (ArrayLike): free throw attempts\n
            DREB (ArrayLike): defensive rebounds\n
            OREB (ArrayLike): offensive rebounds\n
            TOV (ArrayLike): turnovers\n
            OPP_FGA (ArrayLike): opponent field goal attempts\n
            OPP_FGM (ArrayLike): opponent field goal makes\n
            OPP_FTA (ArrayLike): opponent free throw attempts\n
            OPP_OREB (ArrayLike): opponent offensive rebounds\n
            OPP_DREB (ArrayLike): opponent defensive rebounds\n
            OPP_TOV (ArrayLike): opponent turnovers\n

        Returns:
            ArrayLike: team offensive rating
        """
        poss = self.possessions(
            FGA=FGA,
            FGM=FGM,
            FTA=FTA,
            DREB=DREB,
            OREB=OREB,
            TOV=TOV,
            OPP_FGA=OPP_FGA,
            OPP_FGM=OPP_FGM,
            OPP_FTA=OPP_FTA,
            OPP_OREB=OPP_OREB,
            OPP_DREB=OPP_DREB,
            OPP_TOV=OPP_TOV,
        )
        return np.multiply(100, np.divide(PTS, poss))

    def defensive_rating(
        self,
        FGA: ArrayLike,
        FGM: ArrayLike,
        FTA: ArrayLike,
        DREB: ArrayLike,
        OREB: ArrayLike,
        TOV: ArrayLike,
        OPP_PTS: ArrayLike,
        OPP_FGA: ArrayLike,
        OPP_FGM: ArrayLike,
        OPP_FTA: ArrayLike,
        OPP_OREB: ArrayLike,
        OPP_DREB: ArrayLike,
        OPP_TOV: ArrayLike,
        **_
    ) -> ArrayLike:
        """
        Team Defensive Rating (DefRtg)
            DefRtg = 100 * (PTS Allowed / POSS)

        Estimates how many points a team allows per 100 possessions.
        Source: https://hackastat.eu/en/learn-a-stat-team-offensive-defensive-and-net-rating/

        Args:
            FGA (ArrayLike): field goal attempts\n
            FGM (ArrayLike): field goal makes\n
            FTA (ArrayLike): free throw attempts\n
            DREB (ArrayLike): defensive rebounds\n
            OREB (ArrayLike): offensive rebounds\n
            TOV (ArrayLike): turnovers\n
            OPP_PTS (ArrayLike): opponent points / points allowed\n
            OPP_FGA (ArrayLike): opponent field goal attempts\n
            OPP_FGM (ArrayLike): opponent field goal makes\n
            OPP_FTA (ArrayLike): opponent free throw attempts\n
            OPP_OREB (ArrayLike): opponent offensive rebounds\n
            OPP_DREB (ArrayLike): opponent defensive rebounds\n
            OPP_TOV (ArrayLike): opponent turnovers\n

        Returns:
            ArrayLike: team defensive rating
        """
        poss = self.possessions(
            FGA=FGA,
            FGM=FGM,
            FTA=FTA,
            DREB=DREB,
            OREB=OREB,
            TOV=TOV,
            OPP_FGA=OPP_FGA,
            OPP_FGM=OPP_FGM,
            OPP_FTA=OPP_FTA,
            OPP_OREB=OPP_OREB,
            OPP_DREB=OPP_DREB,
            OPP_TOV=OPP_TOV,
        )
        return np.multiply(100, np.divide(OPP_PTS, poss))

    def strength_of_schedule(self, **_) -> ArrayLike:
        # Source: https://web.archive.org/web/20180531115621/https://www.pro-football-reference.com/blog/index4837.html?p=37
        return []

    def games_behind(
        self,
        TEAM_WINS: ArrayLike,
        TEAM_LOSSES: ArrayLike,
        FIRST_PLACE_WINS: ArrayLike,
        FIRST_PLACE_LOSSES: ArrayLike,
        **_
    ) -> ArrayLike:
        """
        Games Behind (GB)
            GB = ((FIRST_PLACE_WINS - TEAM_WINS) + (TEAM_LOSSES - FIRST_PLACE_LOSSES)) / 2

        Compute how many games behind a team is in the standings.
        Source: https://www.basketball-reference.com//about/glossary.html#gb

        Args:
            TEAM_WINS (ArrayLike): # team wins\n
            TEAM_LOSSES (ArrayLike): # team losses\n
            FIRST_PLACE_WINS (ArrayLike): # first place team wins\n
            FIRST_PLACE_LOSSES (ArrayLike): # first place team losses\n

        Returns:
            ArrayLike: games behind.
        """
        return np.divide(
            np.add(
                np.subtract(FIRST_PLACE_WINS, TEAM_WINS),
                np.subtract(TEAM_LOSSES, FIRST_PLACE_LOSSES),
            ),
            2,
        )

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
