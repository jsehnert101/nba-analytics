# %%
# Imports
import numpy as np
from numpy.typing import ArrayLike
from typing import Union


# %%
# Create object(s) to compute statistics for NBA Teams & Players.
class Stats:
    """Class to compute statistics for NBA Team/Player data.
    Source: https://www.basketball-reference.com/about/glossary.html
    """

    def __init__(
        self,
        true_shooting_ft_weight: float = 0.44,
        pythagorean_exponent: float = 13.91,  # Morey: 13.91; Hollinger: 16.5
        four_factor_shooting_weight: float = 0.4,
        four_factor_turnover_weight: float = 0.25,
        four_factor_rebounding_weight: float = 0.2,
        four_factor_free_throw_weight: float = 0.15,
    ) -> None:
        self._ts_ft_weight = true_shooting_ft_weight
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
        return np.add(FGA, np.multiply(FTA, self._ts_ft_weight))

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

    def _possession_estimate(
        self,
        FGA: Union[float, int, ArrayLike],
        FTA: Union[float, int, ArrayLike],
        TOV: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Calculate possession estimate used in some stats.
            = FGA + 0.44 * FTA + TOV

        Returns:
            Union[float, int, ArrayLike]: possession estimate
        """
        return np.sum(
            np.array([FGA, np.multiply(FTA, self._ts_ft_weight), TOV]), axis=0
        )


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
        return np.divide(TOV, self._possession_estimate(FGA=FGA, FTA=FTA, TOV=TOV))

    def possessions(FGA: Union[float, int, ArrayLike], FTA: Union[float, int, ArrayLike], OREB: Union[float, int, ArrayLike])


    def pythagorean_win_pct(
        self, PTS: Union[float, int, ArrayLike], OPP_PTS: Union[float, int, ArrayLike]
    ) -> Union[float, int, ArrayLike]:
        """
        Pythagorean Expected Win Percentage
            = PTS**EXP / (PTS**EXP + PTS_ALLOWED**EXP)

        Estimates what a team's win percentage "should be."
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
            FT (Union[float, int, ArrayLike]): free throws
            FGA (Union[float, int, ArrayLike]): field goal attempts

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
        Compute Dean Oliver's Four Factors, which summarize a team's strengths and weaknesses.
        May be applied offensively or defensively.
        Source: https://www.basketball-reference.com/about/factors.html

        Args:
            FGA (Union[float, int, ArrayLike]): field goal attempts.
            FGM (Union[float, int, ArrayLike]): field goal makes.
            FG3M (Union[float, int, ArrayLike]): three point field goal makes.
            FTM (Union[float, int, ArrayLike]): free throw makes
            FTA (Union[float, int, ArrayLike]): free throw attempts
            REB (Union[float, int, ArrayLike]): rebounds
            OPP_REB (Union[float, int, ArrayLike]): opponent rebounds against
            TOV (Union[float, int, ArrayLike]): turnovers

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


class PlayerStats(Stats):
    """Compute statistics for NBA Players.

    Args:
        Stats (_type_): Generic class that computes statistics.
    """

    def assist_pct(
        self, AST, FGM_TEAM, FGM_PLAYER, MP_PLAYER, MP_TEAM
    ) -> Union[float, int, ArrayLike]:
        """
        Assist Percentage (AST%)
            AST% = 100 * AST / (((MP_PLAYER / (MP_TEAM / 5)) * FGM_TEAM) - FGM_PLAYER)

        Estimates percentage of teammate field goals a player assisted while on the floor.

        Args:
            AST (_type_): assists\n
            FGM_TEAM (_type_): team field goal makes\n
            FGM_PLAYER (_type_): player field goal makes\n
            MP_PLAYER (_type_): player minutes played\n
            MP_TEAM (_type_): team minutes played\n

        Returns:
            Union[float, int, ArrayLike]: _description_
        """
        return np.multiply(
            100,
            np.divide(
                AST,
                np.subtract(
                    np.multiply(np.divide(MP_PLAYER, np.divide(MP_TEAM, 5)), FGM_TEAM),
                    FGM_PLAYER,
                ),
            ),
        )

    def defensive_rebounding_pct(
        self,
        DREB: Union[float, int, ArrayLike],
        OPP_OREB: Union[float, int, ArrayLike],
        MP_PLAYER: Union[float, int, ArrayLike],
        MP_TEAM: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Defensive Rebounding Percentage / Rate (DREB%).
            DREB% = 100 * (DREB * (MP_TEAM / 5)) / (MP_PLAYER * (DREB + OPP_OREB))

        Estimates percentage of available defensive rebounds a player grabbed while on the floor.

        Args:
            DREB (Union[float, int, ArrayLike]): defensive rebounds\n
            OPP_OREB (Union[float, int, ArrayLike]): opponent offensive rebounds\n
            MP_PLAYER (Union[float, int, ArrayLike]): player minutes played\n
            MP_TEAM (Union[float, int, ArrayLike]): team minutes played\n

        Returns:
            Union[float, int, ArrayLike]: defensive rebounding percentage
        """
        return np.multiply(
            100,
            np.divide(
                np.multiply(DREB, np.divide(MP_TEAM, 5)),
                np.multiply(MP_PLAYER, np.add(DREB, OPP_OREB)),
            ),
        )

    def offensive_rebounding_pct(
        self,
        OREB: Union[float, int, ArrayLike],
        OPP_DREB: Union[float, int, ArrayLike],
        MP_PLAYER: Union[float, int, ArrayLike],
        MP_TEAM: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Compute Offensive Rebounding Percentage (OREB%)
            OREB% = 100 * (OREB * (MP_TEAM / 5)) / (Opp_DREB * (Opp_MP / 5)).

        Estimates percentage of available offensive rebounds a player grabbed while on the floor.
        """
        return np.multiply(
            100,
            np.divide(
                np.multiply(OREB, np.divide(MP_PLAYER, 5)),
                np.multiply(OPP_DREB, np.divide(MP_TEAM, 5)),
            ),
        )

    def block_pct(
        self,
        BLK: Union[float, int, ArrayLike],
        OPP_FGA: Union[float, int, ArrayLike],
        OPP_FG3A: Union[float, int, ArrayLike],
        MP_PLAYER: Union[float, int, ArrayLike],
        MP_TEAM: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Block Percentage (BLK%)
            BLK% = 100 * (BLK * (MP_TEAM / 5)) / (MP_PLAYER * (OPP_FGA - OPP_3PA)).

        Estimates percentage of opponent 2PT FGA blocked by a player while he was on the floor.

        Args:
            BLK (Union[float, int, ArrayLike]): blocks\n
            MP_PLAYER (Union[float, int, ArrayLike]): player minutes played\n
            MP_TEAM (Union[float, int, ArrayLike]): team minutes played\n
            OPP_FGA (Union[float, int, ArrayLike]): opponent field goal attempts\n
            OPP_FG3A (Union[float, int, ArrayLike]): opponent three-point field goal attempts\n

        Returns:
            Union[float, int, ArrayLike]: block percentage
        """
        return np.divide(
            np.multiply(100, np.multiply(BLK, np.divide(MP_TEAM, 5))),
            np.multiply(MP_PLAYER, np.subtract(OPP_FGA, OPP_FG3A)),
        )

    def steal_pct(
        self,
        STL: Union[float, int, ArrayLike],
        MP_PLAYER: Union[float, int, ArrayLike],
        MP_TEAM: Union[float, int, ArrayLike],
        POSS_OPP: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Steal Percentage (STL%)
            STL% = 100 * (STL * (MP_TEAM / 5)) / (MP_PLAYER * POSS_OPP)

        Estimates percentage of opponent possessions that end in a player's steal while he was on the floor.

        Args:
            STL (Union[float, int, ArrayLike]): steals\n
            MP_PLAYER (Union[float, int, ArrayLike]): player minutes played\n
            MP_TEAM (Union[float, int, ArrayLike]): team minutes played\n
            POSS_OPP (Union[float, int, ArrayLike]): opponent possessions\n

        Returns:
            Union[float, int, ArrayLike]: steal percentage
        """
        return np.multiply(
            100,
            np.divide(
                np.multiply(STL, np.divide(MP_TEAM, 5)),
                np.multiply(MP_PLAYER, POSS_OPP),
            ),
        )

    def usage_rate(
        self,
        FGA: Union[float, int, ArrayLike],
        FTA: Union[float, int, ArrayLike],
        TOV: Union[float, int, ArrayLike],
        MP_PLAYER: Union[float, int, ArrayLike],
        FGA_TEAM: Union[float, int, ArrayLike],
        FTA_TEAM: Union[float, int, ArrayLike],
        TOV_TEAM: Union[float, int, ArrayLike],
        MP_TEAM: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Usage Rate (USG%)
            USG% = 100 * ((FGA + 0.44*FTA + TOV) * (MP_TEAM / 5)) / (MP_PLAYER * (FGA_TEAM + 0.44*FTA_TEAM + TOV_TEAM))

        Estimates percentage of team plays used by a player while on the floor.

        Args:
            FGA (Union[float, int, ArrayLike]): field goal attempts\n
            FTA (Union[float, int, ArrayLike]): free throw attempts\n
            TOV (Union[float, int, ArrayLike]): turnovers\n
            MP_PLAYER (Union[float, int, ArrayLike]): player minutes played\n
            MP_TEAM (Union[float, int, ArrayLike]): team minutes played\n
            FGA_TEAM (Union[float, int, ArrayLike]): team field goal attempts\n
            FTA_TEAM (Union[float, int, ArrayLike]): team free throw attempts\n
            TOV_TEAM (Union[float, int, ArrayLike]): team turnovers\n
            MP_TEAM (Union[float, int, ArrayLike]): team minutes played\n

        Returns:
            Union[float, int, ArrayLike]: player usage rate
        """
        return np.multiply(
            100,
            np.divide(
                np.multiply(
                    self._usage_rate_helper(FGA=FGA, FTA=FTA, TOV=TOV),
                    np.divide(MP_TEAM, 5),
                ),
                np.multiply(
                    MP_PLAYER,
                    self._usage_rate_helper(FGA=FGA_TEAM, FTA=FTA_TEAM, TOV=TOV_TEAM),
                ),
            ),
        )

    def win_shares(self) -> Union[float, int, ArrayLike]:
        return

    def off_win_shares(self) -> Union[float, int, ArrayLike]:
        return

    def def_win_shares(self) -> Union[float, int, ArrayLike]:
        return

    def box_plus_minus(self) -> Union[float, int, ArrayLike]:
        """
        Compute Box Plus/Minus (BPM).
        Source: https://www.basketball-reference.com/about/bpm2.html
        """
        return

    def value_over_replacement(self) -> Union[float, int, ArrayLike]:
        return

    def game_score(
        self,
        PTS: Union[float, int, ArrayLike],
        FGM: Union[float, int, ArrayLike],
        FGA: Union[float, int, ArrayLike],
        FTM: Union[float, int, ArrayLike],
        FTA: Union[float, int, ArrayLike],
        OREB: Union[float, int, ArrayLike],
        DREB: Union[float, int, ArrayLike],
        STL: Union[float, int, ArrayLike],
        AST: Union[float, int, ArrayLike],
        BLK: Union[float, int, ArrayLike],
        PF: Union[float, int, ArrayLike],
        TOV: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Game Score (GAME_SCORE)
            = PTS + 0.4*FGM - 0.7*FGA - 0.4*(FTA - FTM) + 0.7*OREB + 0.3*DREB + STL + 0.7*AST + 0.7*BLK - 0.4*PF - TOV

        Estimates a player's performance in a single game. Created by John Hollinger.
        Scale is similar to points scored (40 outstanding, 10 average).
        Source: https://www.basketball-reference.com/about/glossary.html

        Args:
            PTS (Union[float, int, ArrayLike]): points
            FGM (Union[float, int, ArrayLike]): field goal makes
            FGA (Union[float, int, ArrayLike]): field goal attempts
            FTM (Union[float, int, ArrayLike]): free throw makes
            FTA (Union[float, int, ArrayLike]): free throw attempts
            OREB (Union[float, int, ArrayLike]): offensive rebounds
            DREB (Union[float, int, ArrayLike]): defensive rebounds
            STL (Union[float, int, ArrayLike]): steals
            AST (Union[float, int, ArrayLike]): assists
            BLK (Union[float, int, ArrayLike]): blocks
            PF (Union[float, int, ArrayLike]): personal fouls
            TOV (Union[float, int, ArrayLike]): turnovers

        Returns:
            Union[float, int, ArrayLike]: game score
        """
        return np.sum(
            np.array(
                [
                    PTS,
                    np.multiply(FGM, 0.4),
                    np.multiply(FGA, -0.7),
                    np.multiply(np.subtract(FTA, FTM), -0.4),
                    np.multiply(OREB, 0.7),
                    np.multiply(DREB, 0.3),
                    STL,
                    np.multiply(AST, 0.7),
                    np.multiply(BLK, 0.7),
                    np.multiply(PF, -0.4),
                    np.multiply(TOV, -1),
                ]
            ),
            axis=0,
        )
