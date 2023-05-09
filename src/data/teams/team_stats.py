# %%
# Imports
from typing import Union
import numpy as np
from numpy.typing import ArrayLike
from data.stats import Stats

# %%
# Define object to compute NBA team statistics.


class TeamStats(Stats):
    """Compute statistics for NBA teams.

    Args:
        Stats (_type_): Generic class that computes statistics.
    """

    def rebound_pct(
        self, REB: Union[float, int, ArrayLike], OPP_REB: Union[float, int, ArrayLike]
    ) -> Union[float, int, ArrayLike]:
        """
        Rebounding Percentage (REB%)
            REB% = REB / (REB + OPP_REB)

        Percentage of rebounds a team grabs while on the floor.

        Args:
            REB (Union[float, int, ArrayLike]): rebounds\n
            OPP_REB (Union[float, int, ArrayLike]): opponent rebounds against\n

        Returns:
            Union[float, int, ArrayLike]: percentage of rebounds a team grabbed while on the floor.
        """
        return np.divide(REB, np.add(REB, OPP_REB))

    def defensive_rebound_pct(
        self, DREB: Union[float, int, ArrayLike], OPP_OREB: Union[float, int, ArrayLike]
    ) -> Union[float, int, ArrayLike]:
        """
        Defensive Rebounding Percentage (DREB%)
            DREB% = DREB / (DREB + OPP_OREB)

        Percentage of defensive rebounds a team grabs while on the floor.

        Args:
            DREB (Union[float, int, ArrayLike]): defensive rebounds\n
            OPP_OREB (Union[float, int, ArrayLike]): opponent offensive rebounds against\n

        Returns:
            Union[float, int, ArrayLike]: defensive rebounding rate
        """
        return np.divide(DREB, np.add(DREB, OPP_OREB))

    def offensive_rebound_pct(
        self, OREB: Union[float, int, ArrayLike], OPP_DREB: Union[float, int, ArrayLike]
    ) -> Union[float, int, ArrayLike]:
        """
        Offensive Rebounding Percentage (OREB%)
            OREB% = OREB / (OREB + OPP_DREB)

        Percentage of offensive rebounds a team grabs while on the floor.

        Args:
            OREB (Union[float, int, ArrayLike]): offensive rebounds\n
            OPP_DREB (Union[float, int, ArrayLike]): opponent defensive rebounds against\n

        Returns:
            Union[float, int, ArrayLike]: offensive rebounding rate
        """
        return np.divide(OREB, np.add(OREB, OPP_DREB))

    def turnover_pct(
        self,
        FGA: Union[float, int, ArrayLike],
        FTA: Union[float, int, ArrayLike],
        TOV: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Turnover Percentage (TOV%)
            TOV% = TOV / (FGA + 0.44*FTA + TOV)

        Estimates percentage of team possessions that end in a turnover.

        Args:
            FGA (Union[float, int, ArrayLike]): field goal attempts\n
            FTA (Union[float, int, ArrayLike]): free throw attempts\n
            TOV (Union[float, int, ArrayLike]): turnovers\n

        Returns:
            Union[float, int, ArrayLike]: turnover rate
        """
        return np.divide(TOV, self.minor_possessions(FGA=FGA, FTA=FTA, TOV=TOV))

    def pythagorean_win_pct(
        self, PTS: Union[float, int, ArrayLike], OPP_PTS: Union[float, int, ArrayLike]
    ) -> Union[float, int, ArrayLike]:
        """
        Pythagorean Expected Win Percentage
            = PTS**EXP / (PTS**EXP + PTS_ALLOWED**EXP)

        Estimates what a team's win percentage "should be."
        The formula was obtained by fitting a logistic regression model: WIN_PCT ~ log(PTS / OPP_PTS)
        Source: https://captaincalculator.com/sports/basketball/pythagorean-win-percentage-calculator/
        Paper: https://web.williams.edu/Mathematics/sjmiller/public_html/399/handouts/PythagWonLoss_Paper.pdf

        Args:
            PTS (Union[float, int, ArrayLike]): points\n
            OPP_PTS (Union[float, int, ArrayLike]): opponent points against\n

        Returns:
            Union[float, int, ArrayLike]: team expected win percentage
        """
        return np.divide(
            np.power(PTS, self._pythagorean_exp),
            np.add(
                np.power(PTS, self._pythagorean_exp),
                np.power(OPP_PTS, self._pythagorean_exp),
            ),
        )

    def shooting_factor(
        self,
        FGA: Union[float, int, ArrayLike],
        FGM: Union[float, int, ArrayLike],
        FG3M: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Dean Oliver's Shooting Factor (eFG%).

        Args:
            FGA (Union[float, int, ArrayLike]): field goal attempts\n
            FGM (Union[float, int, ArrayLike]): field goal makes\n
            FG3M (Union[float, int, ArrayLike]): three point field goal makes\n

        Returns:
            Union[float, int, ArrayLike]: Shooting Factor / eFG%
        """
        return self.effective_field_goal_pct(FGM=FGM, FG3M=FG3M, FGA=FGA)

    def turnover_factor(
        self,
        TOV: Union[float, int, ArrayLike],
        FGA: Union[float, int, ArrayLike],
        FTA: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Dean Oliver's Turnover Factor (TO%).

        Args:
            FGA (Union[float, int, ArrayLike]): field goal attempts\n
            FTA (Union[float, int, ArrayLike]): free throw attempts\n
            TOV (Union[float, int, ArrayLike]): turnovers\n

        Returns:
            Union[float, int, ArrayLike]: Turnover Factor / TO%
        """
        return self.turnover_pct(TOV=TOV, FGA=FGA, FTA=FTA)

    def rebound_factor(
        self, REB: Union[float, int, ArrayLike], OPP_REB: Union[float, int, ArrayLike]
    ) -> Union[float, int, ArrayLike]:
        """
        Dean Oliver's Rebound Factor.
            REB_FACTOR = REB / (REB + OPP_REB) = REB%

        Args:
            REB (Union[float, int, ArrayLike]): rebounds\n
            OPP_REB (Union[float, int, ArrayLike]): opponent rebounds against\n

        Returns:
            Union[float, int, ArrayLike]: Rebound Factor / REB%
        """
        return self.rebound_pct(REB=REB, OPP_REB=OPP_REB)

    def free_throw_factor(
        self, FTM: Union[float, int, ArrayLike], FGA: Union[float, int, ArrayLike]
    ) -> Union[float, int, ArrayLike]:
        """
        Dean Oliver's Free Throw Factor.
            FT_FACTOR = FTM / FGA

        Estimates how often team gets to the line and how often they make them.
        Source: https://www.basketball-reference.com/about/factors.html

        Args:
            FT (Union[float, int, ArrayLike]): free throws\n
            FGA (Union[float, int, ArrayLike]): field goal attempts\n

        Returns:
            Union[float, int, ArrayLike]: Free Throw Factor.
        """
        return np.divide(FTM, FGA)

    def four_factor_score(
        self,
        FGA: Union[float, int, ArrayLike],
        FGM: Union[float, int, ArrayLike],
        FG3M: Union[float, int, ArrayLike],
        FTM: Union[float, int, ArrayLike],
        FTA: Union[float, int, ArrayLike],
        REB: Union[float, int, ArrayLike],
        OPP_REB: Union[float, int, ArrayLike],
        TOV: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Compute Dean Oliver's Four Factors, which summarize a team's strengths and weaknesses.\n
        May be applied offensively or defensively.\n
        Source: https://www.basketball-reference.com/about/factors.html

        Args:
            FGA (Union[float, int, ArrayLike]): field goal attempts.\n
            FGM (Union[float, int, ArrayLike]): field goal makes.\n
            FG3M (Union[float, int, ArrayLike]): three point field goal makes.\n
            FTM (Union[float, int, ArrayLike]): free throw makes\n
            FTA (Union[float, int, ArrayLike]): free throw attempts\n
            REB (Union[float, int, ArrayLike]): rebounds\n
            OPP_REB (Union[float, int, ArrayLike]): opponent rebounds against\n
            TOV (Union[float, int, ArrayLike]): turnovers\n

        Returns:
            Union[float, int, ArrayLike]: Four Factor Score.
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
        FGA: Union[float, int, ArrayLike],
        FGM: Union[float, int, ArrayLike],
        FTA: Union[float, int, ArrayLike],
        OREB: Union[float, int, ArrayLike],
        OPP_DREB: Union[float, int, ArrayLike],
        TOV: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Compute team possession estimate to weight in team possession stat.
            TEAM_POSS = FGA - 0.4 * FTA - 1.07 * OREB% * (FGA - FGM) + TOV

        Source: https://www.basketball-reference.com/about/glossary.html#poss

        Args:
            FGA (Union[float, int, ArrayLike]): field goal attempts
            FGM (Union[float, int, ArrayLike]): field goal makes
            FTA (Union[float, int, ArrayLike]): free throw attempts
            OREB (Union[float, int, ArrayLike]): offensive rebounds
            OPP_DREB (Union[float, int, ArrayLike]): opponent defensive rebounds
            TOV (Union[float, int, ArrayLike]): turnovers

        Returns:
            Union[float, int, ArrayLike]: estimate of team # possessions
        """
        return np.sum(
            np.array(
                [
                    FGA,
                    np.multiply(0.4, FTA),
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
        FGA: Union[float, int, ArrayLike],
        FGM: Union[float, int, ArrayLike],
        FTA: Union[float, int, ArrayLike],
        DREB: Union[float, int, ArrayLike],
        OREB: Union[float, int, ArrayLike],
        TOV: Union[float, int, ArrayLike],
        OPP_FGA: Union[float, int, ArrayLike],
        OPP_FGM: Union[float, int, ArrayLike],
        OPP_FTA: Union[float, int, ArrayLike],
        OPP_OREB: Union[float, int, ArrayLike],
        OPP_DREB: Union[float, int, ArrayLike],
        OPP_TOV: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
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
            FGA (Union[float, int, ArrayLike]): field goal attempts\n
            FGM (Union[float, int, ArrayLike]): field goal makes\n
            FTA (Union[float, int, ArrayLike]): free throw attempts\n
            DREB (Union[float, int, ArrayLike]): defensive rebounds\n
            OREB (Union[float, int, ArrayLike]): offensive rebounds\n
            TOV (Union[float, int, ArrayLike]): turnovers\n
            OPP_FGA (Union[float, int, ArrayLike]): opponent field goal attempts\n
            OPP_FGM (Union[float, int, ArrayLike]): opponent field goal makes\n
            OPP_FTA (Union[float, int, ArrayLike]): opponent free throw attempts\n
            OPP_OREB (Union[float, int, ArrayLike]): opponent offensive rebounds\n
            OPP_DREB (Union[float, int, ArrayLike]): opponent defensive rebounds\n
            OPP_TOV (Union[float, int, ArrayLike]): opponent turnovers\n

        Returns:
            Union[float, int, ArrayLike]: possessions as
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
        self,
        FGA: Union[float, int, ArrayLike],
        FTA: Union[float, int, ArrayLike],
        OREB: Union[float, int, ArrayLike],
        TOV: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Possessions (POSS) using nba.com/espn.com formulas
            POSS = MAJOR POSSESSIONS / 2 = (FGA + 0.44*FTA - OREB + TOV) / 2

        Source: https://fansided.com/2015/12/21/nylon-calculus-101-possessions/
        Note: tend to overestimate possessions

        Args:
            FGA (Union[float, int, ArrayLike]): field goal attempts\n
            FTA (Union[float, int, ArrayLike]): free throw attempts\n
            OREB (Union[float, int, ArrayLike]): offensive rebounds\n
            TOV (Union[float, int, ArrayLike]): turnovers\n

        Returns:
            Union[float, int, ArrayLike]: team possessions according to NBA/ESPN.com.
        """
        return np.divide(
            self.major_possessions(FGA=FGA, FTA=FTA, TOV=TOV, OREB=OREB), 2
        )

    def nylon_calculus_possessions(
        self,
        FGA: Union[float, int, ArrayLike],
        FT_TRIPS: Union[float, int, ArrayLike],
        OREB: Union[float, int, ArrayLike],
        TOV: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Possessions (POSS) using Nylon Calculus formula
            POSS = FGA + FT_TRIPS - OREB + TOV  #TODO: verify if this should be divided by 2

        Source: https://fansided.com/2015/12/21/nylon-calculus-101-possessions/
        Note: FT_TRIPS are pairs/triplets extracted from game logs.

        Args:
            FGA (Union[float, int, ArrayLike]): field goal attempts\n
            FTA (Union[float, int, ArrayLike]): free throw attempts\n
            OREB (Union[float, int, ArrayLike]): offensive rebounds\n
            TOV (Union[float, int, ArrayLike]): turnovers\n

        Returns:
            Union[float, int, ArrayLike]: team possessions according to nylon calculus.
        """
        return np.divide(
            np.sum(np.array([FGA, FT_TRIPS, np.multiply(-1, OREB), TOV]), axis=0), 2
        )

    def pace(
        self,
        POSS: Union[float, int, ArrayLike],
        OPP_POSS: Union[float, int, ArrayLike],
        MP: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Pace Factor (PACE)
            PACE = 48 * ((POSS + OPP_POSS) / (2 * (MP / 5)))

        Estimates number of possessions per 48 minutes by a team.
        Created by Dean Oliver.
        Source: https://www.basketball-reference.com//about/glossary.html#pace

        Args:
            POSS (Union[float, int, ArrayLike]): possessions\n
            OPP_POSS (Union[float, int, ArrayLike]): opponent possessions\n
            MP (Union[float, int, ArrayLike]): minutes played\n

        Returns:
            Union[float, int, ArrayLike]: team pace
        """
        return np.multiply(
            48, np.divide(np.add(POSS, OPP_POSS), np.multiply(2, np.divide(MP, 5)))
        )

    def offensive_rating(
        self, PTS: Union[float, int, ArrayLike], POSS: Union[float, int, ArrayLike]
    ) -> Union[float, int, ArrayLike]:
        """
        Team Offensive Rating (OffRtg)
            OffRtg = 100 * (PTS / POSS)

        Estimates how many points a team scores per 100 possessions.
        Source: https://hackastat.eu/en/learn-a-stat-team-offensive-defensive-and-net-rating/

        Args:
            PTS (Union[float, int, ArrayLike]): points\n
            POSS (Union[float, int, ArrayLike]): possessions\n

        Returns:
            Union[float, int, ArrayLike]: team offensive rating
        """
        return np.multiply(100, np.divide(PTS, POSS))

    def defensive_rating(
        self, OPP_PTS: Union[float, int, ArrayLike], POSS: Union[float, int, ArrayLike]
    ) -> Union[float, int, ArrayLike]:
        """
        Team Defensive Rating (DefRtg)
            DefRtg = 100 * (PTS Allowed / POSS)

        Estimates how many points a team allows per 100 possessions.
        Source: https://hackastat.eu/en/learn-a-stat-team-offensive-defensive-and-net-rating/

        Args:
            OPP_PTS (Union[float, int, ArrayLike]): opponent points / points allowed\n
            POSS (Union[float, int, ArrayLike]): possessions\n

        Returns:
            Union[float, int, ArrayLike]: team defensive rating
        """
        return np.multiply(100, np.divide(OPP_PTS, POSS))

    def strength_of_schedule(self) -> Union[float, int, ArrayLike]:
        # Source: https://web.archive.org/web/20180531115621/https://www.pro-football-reference.com/blog/index4837.html?p=37
        return []

    def games_behind(
        self,
        TEAM_WINS: Union[float, int, ArrayLike],
        TEAM_LOSSES: Union[float, int, ArrayLike],
        FIRST_PLACE_WINS: Union[float, int, ArrayLike],
        FIRST_PLACE_LOSSES: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Games Behind (GB)
            GB = ((FIRST_PLACE_WINS - TEAM_WINS) + (TEAM_LOSSES - FIRST_PLACE_LOSSES)) / 2

        Compute how many games behind a team is in the standings.
        Source: https://www.basketball-reference.com//about/glossary.html#gb

        Args:
            TEAM_WINS (Union[float, int, ArrayLike]): # team wins\n
            TEAM_LOSSES (Union[float, int, ArrayLike]): # team losses\n
            FIRST_PLACE_WINS (Union[float, int, ArrayLike]): # first place team wins\n
            FIRST_PLACE_LOSSES (Union[float, int, ArrayLike]): # first place team losses\n

        Returns:
            Union[float, int, ArrayLike]: games behind.
        """
        return np.divide(
            np.add(
                np.subtract(FIRST_PLACE_WINS, TEAM_WINS),
                np.subtract(TEAM_LOSSES, FIRST_PLACE_LOSSES),
            ),
            2,
        )
