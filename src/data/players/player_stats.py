# %%
# Imports
from typing import Union
import numpy as np
from numpy.typing import ArrayLike
from data.stats import Stats
from data.teams.team_stats import TeamStats

# %%
# Create object to calculate player statistics.


class PlayerStats(Stats):
    """Compute statistics for NBA Players.
    Assume all method parameters are for players unless their prefix denotes otherwise.

    Args:
        Stats (_type_): Generic class that computes basketball statistics.
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
        self.team_stats = TeamStats(
            free_throw_weight=free_throw_weight,
            pythagorean_exponent=pythagorean_exponent,
            four_factor_free_throw_weight=four_factor_free_throw_weight,
            four_factor_rebounding_weight=four_factor_rebounding_weight,
            four_factor_shooting_weight=four_factor_shooting_weight,
            four_factor_turnover_weight=four_factor_turnover_weight,
        )

    def assist_pct(
        self,
        AST: Union[float, int, ArrayLike],
        FGM: Union[float, int, ArrayLike],
        MP: Union[float, int, ArrayLike],
        TEAM_FGM: Union[float, int, ArrayLike],
        TEAM_MP: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Assist Percentage / Rate (AST%)
            AST% = 100 * AST / (((MP / (TEAM_MP / 5)) * TEAM_FGM) - FGM)

        Estimates percentage of teammate field goals a player assisted while on the floor.
        Source: https://www.basketball-reference.com/about/glossary.html#ast

        Args:
            AST (Union[float, int, ArrayLike]): assists\n
            FGM (Union[float, int, ArrayLike]): player field goal makes\n
            MP (Union[float, int, ArrayLike]): player minutes played\n
            TEAM_FGM (Union[float, int, ArrayLike]): team field goal makes\n
            TEAM_MP (Union[float, int, ArrayLike]: team minutes played\n

        Returns:
            Union[float, int, ArrayLike]: player assist rate/percentage
        """
        return np.multiply(
            100,
            np.divide(
                AST,
                np.subtract(
                    np.multiply(
                        np.divide(MP, np.divide(TEAM_MP, 5)),
                        TEAM_FGM,
                    ),
                    FGM,
                ),
            ),
        )

    def defensive_rebound_pct(
        self,
        DREB: Union[float, int, ArrayLike],
        TEAM_DREB: Union[float, int, ArrayLike],
        OPP_OREB: Union[float, int, ArrayLike],
        PLAYER_MP: Union[float, int, ArrayLike],
        TEAM_MP: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Defensive Rebounding Percentage / Rate (DREB%).
            DREB% = 100 * (DREB * (TEAM_MP / 5)) / (PLAYER_MP * (TEAM_DREB + OPP_OREB))

        Estimates percentage of available defensive rebounds a player grabbed while on the floor.
        Source: https://www.basketball-reference.com//about/glossary.html#drb

        Args:
            DREB (Union[float, int, ArrayLike]): defensive rebounds\n
            OPP_OREB (Union[float, int, ArrayLike]): opponent offensive rebounds\n
            MP_PLAYER (Union[float, int, ArrayLike]): player minutes played\n
            MP_TEAM (Union[float, int, ArrayLike]): team minutes played\n

        Returns:
            Union[float, int, ArrayLike]: defensive rebounding rate/percentage
        """
        return np.multiply(
            100,
            np.divide(
                np.multiply(DREB, np.divide(TEAM_MP, 5)),
                np.multiply(PLAYER_MP, np.add(TEAM_DREB, OPP_OREB)),
            ),
        )

    def offensive_rebound_pct(
        self,
        OREB: Union[float, int, ArrayLike],
        TEAM_OREB: Union[float, int, ArrayLike],
        OPP_DREB: Union[float, int, ArrayLike],
        PLAYER_MP: Union[float, int, ArrayLike],
        TEAM_MP: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Offensive Rebound Percentage (OREB%)
            OREB% = 100 * (OREB * (TEAM_MP / 5)) / (PLAYER_MP * (TEAM_OREB + OPP_DREB)).

        Estimates percentage of available offensive rebounds a player grabbed while on the floor.
        Source: https://www.basketball-reference.com//about/glossary.html#orb

        Args:
            OREB (Union[float, int, ArrayLike]): offensive rebounds\n
            TEAM_OREB (Union[float, int, ArrayLike]): team offensive rebounds\n
            OPP_DREB (Union[float, int, ArrayLike]): opponent defensive rebounds\n
            PLAYER_MP (Union[float, int, ArrayLike]): player minutes played\n
            TEAM_MP (Union[float, int, ArrayLike]): team minutes played\n

        Returns:
            Union[float, int, ArrayLike]: offensive rebound percentage
        """
        return np.multiply(
            100,
            np.divide(
                np.multiply(OREB, np.divide(TEAM_MP, 5)),
                np.multiply(PLAYER_MP, np.add(TEAM_OREB, OPP_DREB)),
            ),
        )

    def block_pct(
        self,
        BLK: Union[float, int, ArrayLike],
        OPP_FGA: Union[float, int, ArrayLike],
        OPP_FG3A: Union[float, int, ArrayLike],
        PLAYER_MP: Union[float, int, ArrayLike],
        TEAM_MP: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Block Percentage (BLK%)
            BLK% = 100 * (BLK * (MP_TEAM / 5)) / (MP_PLAYER * (OPP_FGA - OPP_3PA)).

        Estimates percentage of opponent 2PT FGA blocked by a player while he was on the floor.
        Source: https://www.basketball-reference.com/about/glossary.html#blk

        Args:
            BLK (Union[float, int, ArrayLike]): blocks\n
            PLAYER_MP (Union[float, int, ArrayLike]): player minutes played\n
            TEAM_MP (Union[float, int, ArrayLike]): team minutes played\n
            OPP_FGA (Union[float, int, ArrayLike]): opponent field goal attempts\n
            OPP_FG3A (Union[float, int, ArrayLike]): opponent three-point field goal attempts\n

        Returns:
            Union[float, int, ArrayLike]: player block rate / percentage
        """
        return np.multiply(
            100,
            np.divide(
                np.multiply(
                    BLK,
                    np.divide(TEAM_MP, 5),
                ),
                np.multiply(PLAYER_MP, np.subtract(OPP_FGA, OPP_FG3A)),
            ),
        )

    def steal_pct(
        self,
        STL: Union[float, int, ArrayLike],
        MP: Union[float, int, ArrayLike],
        TEAM_MP: Union[float, int, ArrayLike],
        OPP_POSS: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Steal Percentage (STL%)
            STL% = 100 * (STL * (MP_TEAM / 5)) / (MP_PLAYER * POSS_OPP)

        Estimates percentage of opponent possessions that end in a player's steal while he was on the floor.
        Source: https://www.basketball-reference.com/about/glossary.html#stl

        Args:
            STL (Union[float, int, ArrayLike]): steals\n
            MP (Union[float, int, ArrayLike]): player minutes played\n
            MP_TEAM (Union[float, int, ArrayLike]): team minutes played\n
            POSS_OPP (Union[float, int, ArrayLike]): opponent possessions\n

        Returns:
            Union[float, int, ArrayLike]: steal percentage
        """
        return np.multiply(
            100,
            np.divide(
                np.multiply(STL, np.divide(TEAM_MP, 5)),
                np.multiply(MP, OPP_POSS),
            ),
        )

    def usage_rate(
        self,
        FGA: Union[float, int, ArrayLike],
        FTA: Union[float, int, ArrayLike],
        TOV: Union[float, int, ArrayLike],
        MP: Union[float, int, ArrayLike],
        TEAM_FGA: Union[float, int, ArrayLike],
        TEAM_FTA: Union[float, int, ArrayLike],
        TEAM_TOV: Union[float, int, ArrayLike],
        TEAM_MP: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Usage Rate (USG%)
            USG% = 100 * ((FGA + 0.44*FTA + TOV) * (MP_TEAM / 5)) / (MP * (TEAM_FGA + 0.44 * TEAM_FTA + TEAM_TOV))

        Estimates percentage of team plays used by a player while on the floor.

        Args:
            FGA (Union[float, int, ArrayLike]): field goal attempts\n
            FTA (Union[float, int, ArrayLike]): free throw attempts\n
            TOV (Union[float, int, ArrayLike]): turnovers\n
            MP (Union[float, int, ArrayLike]): player minutes played\n
            TEAM_MP (Union[float, int, ArrayLike]): team minutes played\n
            TEAM_FGA (Union[float, int, ArrayLike]): team field goal attempts\n
            TEAM_FTA (Union[float, int, ArrayLike]): team free throw attempts\n
            TEAM_TOV (Union[float, int, ArrayLike]): team turnovers\n
            TEAM_MP (Union[float, int, ArrayLike]): team minutes played\n

        Returns:
            Union[float, int, ArrayLike]: player usage rate.
        """
        return np.multiply(
            100,
            np.divide(
                np.multiply(
                    self.minor_possessions(FGA=FGA, FTA=FTA, TOV=TOV),
                    np.divide(TEAM_MP, 5),
                ),
                np.multiply(
                    MP,
                    self.minor_possessions(FGA=TEAM_FGA, FTA=TEAM_FTA, TOV=TEAM_TOV),
                ),
            ),
        )

    def _qAST(
        self,
        FGM: Union[float, int, ArrayLike],
        AST: Union[float, int, ArrayLike],
        MP: Union[float, int, ArrayLike],
        TEAM_FGM: Union[float, int, ArrayLike],
        TEAM_AST: Union[float, int, ArrayLike],
        TEAM_MP: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        qAST term in Field Goal Part of Scoring Possessions.
            qAST = ((MP / (TEAM_MP / 5)) * (1.14 * ((TEAM_AST - AST) / TEAM_FGM))) + ((((TEAM_AST / TEAM_MP) * MP * 5) - AST) / (((TEAM_FGM / TEAM_MP) * MP * 5) - FGM)) * (1 - (MP / (TEAM_MP / 5)))

        Coefficient that takes into consideration the shots made after an assist or not, in order to assign a different weight to the two shot types.
        With this formula, the percentage of shots made by a player following an assist from his teammate is estimated;
        it is a complex formula, created by Oliver when there was also a clear division of roles (i.e. the centers were the players with the most shots
        scored with the help of an assist). Today, with the position-less, this formula is a bit outdated, although still valid.
        If possible, it is better to consider the actual percentage of shots made after an assist.
        With this coefficient, Oliver tends to give less importance to the shots scored after an assist.\n
        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            FGM (Union[float, int, ArrayLike]): field goal makes\n
            AST (Union[float, int, ArrayLike]): assists\n
            MP (Union[float, int, ArrayLike]): player minutes played\n
            TEAM_FGM (Union[float, int, ArrayLike]): team field goal makes\n
            TEAM_AST (Union[float, int, ArrayLike]): team assists\n
            TEAM_MP (Union[float, int, ArrayLike]): team minutes played\n

        Returns:
            Union[float, int, ArrayLike]: _qAST term in Field Goal Part of Scoring Possessions component of Individual Total Possessions.
        """
        return np.add(
            np.multiply(
                np.divide(MP, np.divide(TEAM_MP, 5)),
                np.multiply(1.14, np.divide(np.subtract(TEAM_AST, AST), TEAM_FGM)),
            ),
            np.multiply(
                np.divide(
                    np.subtract(
                        np.multiply(np.divide(TEAM_AST, TEAM_MP), np.multiply(MP, 5)),
                        AST,
                    ),
                    np.subtract(
                        np.multiply(np.divide(TEAM_FGM, TEAM_MP), np.multiply(MP, 5)),
                        FGM,
                    ),
                ),
                np.subtract(1, np.divide(MP, np.divide(TEAM_MP, 5))),
            ),
        )

    def _scposs_fg_part(
        self,
        PTS: Union[float, int, ArrayLike],
        FGA: Union[float, int, ArrayLike],
        FGM: Union[float, int, ArrayLike],
        FTM: Union[float, int, ArrayLike],
        AST: Union[float, int, ArrayLike],
        MP: Union[float, int, ArrayLike],
        TEAM_MP: Union[float, int, ArrayLike],
        TEAM_FGM: Union[float, int, ArrayLike],
        TEAM_AST: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Field Goal Part of Scoring Possessions Component of Individual Total Possessions.
            FG_PART = FGM * (1 - 0.5 * ((PTS - FTM) / (2 * FGA)) * _qAST)

        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            PTS (Union[float, int, ArrayLike]): points\n
            FGA (Union[float, int, ArrayLike]): field goal attempts\n
            FGM (Union[float, int, ArrayLike]): field goal makes\n
            FTM (Union[float, int, ArrayLike]): free throw makes\n
            AST (Union[float, int, ArrayLike]): assists\n
            MP (Union[float, int, ArrayLike]): player minutes played\n
            TEAM_FGM (Union[float, int, ArrayLike]): team field goal makes\n
            TEAM_AST (Union[float, int, ArrayLike]): team assists\n
            TEAM_MP (Union[float, int, ArrayLike]): team minutes played\n

        Returns:
            Union[float, int, ArrayLike]: Field Goal Part of Scoring Possessions Component of Individual Total Possessions.
        """
        return np.multiply(
            FGM,
            np.subtract(
                1,
                np.multiply(
                    0.5,
                    np.multiply(
                        np.divide(np.subtract(PTS, FTM), np.multiply(2, FGA)),
                        self._qAST(
                            MP=MP,
                            FGM=FGM,
                            AST=AST,
                            TEAM_MP=TEAM_MP,
                            TEAM_FGM=TEAM_FGM,
                            TEAM_AST=TEAM_AST,
                        ),
                    ),
                ),
            ),
        )

    def _scposs_assist_part(
        self,
        PTS: Union[float, int, ArrayLike],
        FGA: Union[float, int, ArrayLike],
        FTM: Union[float, int, ArrayLike],
        AST: Union[float, int, ArrayLike],
        TEAM_PTS: Union[float, int, ArrayLike],
        TEAM_FGA: Union[float, int, ArrayLike],
        TEAM_FTM: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Assist Component of Scoring Possessions.
            AST_PART = 0.5 * (((TEAM_PTS - TEAM_FTM) - (PTS - FTM)) / (2 * (TEAM_FGA - FGA))) * AST

        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            PTS (Union[float, int, ArrayLike]): points\n
            FGA (Union[float, int, ArrayLike]): field goal attempts\n
            FTM (Union[float, int, ArrayLike]): free throw makes\n
            AST (Union[float, int, ArrayLike]): assists\n
            TEAM_PTS (Union[float, int, ArrayLike]): team points\n
            TEAM_FGS (Union[float, int, ArrayLike]): team field goals\n
            TEAM_FTA (Union[float, int, ArrayLike]): team free throw attempts\n

        Returns:
            Union[float, int, ArrayLike]: Assist Part of Scoring Possessions component of Individual Total Possessions.
        """
        return np.multiply(
            0.5,
            np.multiply(
                np.divide(
                    np.subtract(np.subtract(TEAM_PTS, TEAM_FTM), np.subtract(PTS, FTM)),
                    np.multiply(2, np.subtract(TEAM_FGA, FGA)),
                ),
                AST,
            ),
        )

    def _scposs_ft_part(
        self, FTM: Union[float, int, ArrayLike], FTA: Union[float, int, ArrayLike]
    ) -> Union[float, int, ArrayLike]:
        """
        Free Throw Part of Scoring Possessions Component of Individual Total Possessions.
            FT_PART = (1 - (1 - (FTM / FTA))**2) * 0.4 * FTA

        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            FTM (Union[float, int, ArrayLike]): free throw makes\n
            FTA (Union[float, int, ArrayLike]): free throw attempts\n

        Returns:
            Union[float, int, ArrayLike]: Free Throw Part of Scoring Possessions component of Individual Total Possessions.
        """
        return np.multiply(
            np.subtract(
                1,
                np.power(
                    np.subtract(1, np.divide(FTM, FTA)),
                    2,
                ),
            ),
            np.multiply(
                0.4,
                FTA,
            ),
        )

    def _team_scoring_possessions(
        self,
        TEAM_FGM: Union[float, int, ArrayLike],
        TEAM_FTA: Union[float, int, ArrayLike],
        TEAM_FTM: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Team Scoring Possessions term in Individual Total Possessions.
            TEAM_SCORING_POSS = TEAM_FGM + (1 - (1 - (TEAM_FTM / TEAM_FTA))**2) * 0.4 * TEAM_FTA

        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            TEAM_FGM (Union[float, int, ArrayLike]): team field goal makes\n
            TEAM_FTA (Union[float, int, ArrayLike]): team free throw attempts\n
            TEAM_FTM (Union[float, int, ArrayLike]): team free throw makes\n

        Returns:
            Union[float, int, ArrayLike]: Team Scoring Possessions Term in Individual Total Possessions.
        """
        return np.add(
            TEAM_FGM,
            np.multiply(
                np.subtract(
                    1,
                    np.power(
                        np.subtract(1, np.divide(TEAM_FTM, TEAM_FTA)),
                        2,
                    ),
                ),
                np.multiply(
                    0.4,
                    TEAM_FTA,
                ),
            ),
        )

    def _team_play_pct(
        self,
        TEAM_FGA: Union[float, int, ArrayLike],
        TEAM_FGM: Union[float, int, ArrayLike],
        TEAM_FTA: Union[float, int, ArrayLike],
        TEAM_FTM: Union[float, int, ArrayLike],
        TEAM_TOV: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Team Play Percentage term in OREB Part of Scoring Possessions component of Individual Total Possessions.
            TEAM_PLAY% = TEAM_SCORING_POSS / TEAM_MINOR_POSSESSIONS

        Estimates percentage of possessions in which a team scores.
        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            TEAM_FGA (Union[float, int, ArrayLike]): team field goal attempts\n
            TEAM_FGM (Union[float, int, ArrayLike]): team field goal makes\n
            TEAM_FTA (Union[float, int, ArrayLike]): team free throw attempts\n
            TEAM_FTM (Union[float, int, ArrayLike]): team free throw makes\n
            TEAM_TOV (Union[float, int, ArrayLike]): team turnovers\n

        Returns:
            Union[float, int, ArrayLike]: Team Play Percentage term in OREB Part of Individual Total Possessions.
        """
        return np.divide(
            self._team_scoring_possessions(
                TEAM_FGM=TEAM_FGM, TEAM_FTA=TEAM_FTA, TEAM_FTM=TEAM_FTM
            ),
            self.minor_possessions(FGA=TEAM_FGA, FTA=TEAM_FTA, TOV=TEAM_TOV),
        )

    def _team_oreb_weight(
        self,
        TEAM_FGA: Union[float, int, ArrayLike],
        TEAM_FGM: Union[float, int, ArrayLike],
        TEAM_FTA: Union[float, int, ArrayLike],
        TEAM_FTM: Union[float, int, ArrayLike],
        TEAM_OREB: Union[float, int, ArrayLike],
        TEAM_TOV: Union[float, int, ArrayLike],
        OPP_DREB: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Team Offensive Rebound Weight term of OREB Part of Scoring Possessions Component of in Individual Total Possessions.
            TEAM_OREB_WEIGHT = ((1 - TEAM_OREB%) * TEAM_PLAY%) / ((1 - TEAM_OREB%) * TEAM_PLAY% + TEAM_OREB% * (1 - TEAM_PLAY%))

        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            TEAM_FGA (Union[float, int, ArrayLike]): team field goal attempts\n
            TEAM_FGM (Union[float, int, ArrayLike]): team field goal makes\n
            TEAM_FTA (Union[float, int, ArrayLike]): team free throw attempts\n
            TEAM_FTM (Union[float, int, ArrayLike]): team free throw makes\n
            TEAM_OREB (Union[float, int, ArrayLike]): team offensive rebounds\n
            TEAM_TOV (Union[float, int, ArrayLike]): team turnovers\n
            OPP_DREB (Union[float, int, ArrayLike]): opponent defensive rebounds\n

        Returns:
            Union[float, int, ArrayLike]: Team Offensive Rebound Weight in Individual Total Possessions.
        """
        team_off_reb_pct = self.team_stats.offensive_rebound_pct(
            OREB=TEAM_OREB, OPP_DREB=OPP_DREB
        )
        team_score_rate = self._team_play_pct(
            TEAM_FGA=TEAM_FGA,
            TEAM_FGM=TEAM_FGM,
            TEAM_FTA=TEAM_FTA,
            TEAM_FTM=TEAM_FTM,
            TEAM_TOV=TEAM_TOV,
        )
        return np.divide(
            np.multiply(
                np.subtract(1, team_off_reb_pct),
                team_score_rate,
            ),
            np.add(
                np.multiply(np.subtract(1, team_off_reb_pct), team_score_rate),
                np.multiply(team_off_reb_pct, np.subtract(1, team_score_rate)),
            ),
        )

    def _oreb_part(
        self,
        OREB: Union[float, int, ArrayLike],
        TEAM_FGA: Union[float, int, ArrayLike],
        TEAM_FGM: Union[float, int, ArrayLike],
        TEAM_FTA: Union[float, int, ArrayLike],
        TEAM_FTM: Union[float, int, ArrayLike],
        TEAM_OREB: Union[float, int, ArrayLike],
        TEAM_TOV: Union[float, int, ArrayLike],
        OPP_DREB: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Offensive Rebound Part of Scoring Possessions Component of Individual Total Possessions.
            TEAM_OFF_REB_PART = OREB * TEAM_OREB_WEIGHT * TEAM_PLAY%

        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            OREB (Union[float, int, ArrayLike]): offensive rebounds\n
            TEAM_FGA (Union[float, int, ArrayLike]): team field goal attempts\n
            TEAM_FGM (Union[float, int, ArrayLike]): team field goal makes\n
            TEAM_FTA (Union[float, int, ArrayLike]): team free throw attempts\n
            TEAM_FTM (Union[float, int, ArrayLike]): team free throw makes\n
            TEAM_OREB (Union[float, int, ArrayLike]): team offensive rebounds\n
            TEAM_TOV (Union[float, int, ArrayLike]): team turnovers\n
            OPP_DREB (Union[float, int, ArrayLike]): opponent defensive rebounds\n

        Returns:
            Union[float, int, ArrayLike]: Offensive Rebound Part of Individual Total Possessions.
        """
        return np.multiply(
            OREB,
            np.multiply(
                self._team_oreb_weight(
                    TEAM_FGA=TEAM_FGA,
                    TEAM_FGM=TEAM_FGM,
                    TEAM_FTA=TEAM_FTA,
                    TEAM_FTM=TEAM_FTM,
                    TEAM_OREB=TEAM_OREB,
                    TEAM_TOV=TEAM_TOV,
                    OPP_DREB=OPP_DREB,
                ),
                self._team_play_pct(
                    TEAM_FGA=TEAM_FGA,
                    TEAM_FGM=TEAM_FGM,
                    TEAM_FTA=TEAM_FTA,
                    TEAM_FTM=TEAM_FTM,
                    TEAM_TOV=TEAM_TOV,
                ),
            ),
        )

    def _scoring_possessions(
        self,
        PTS: Union[float, int, ArrayLike],
        FGA: Union[float, int, ArrayLike],
        FGM: Union[float, int, ArrayLike],
        FTA: Union[float, int, ArrayLike],
        FTM: Union[float, int, ArrayLike],
        OREB: Union[float, int, ArrayLike],
        AST: Union[float, int, ArrayLike],
        MP: Union[float, int, ArrayLike],
        TEAM_PTS: Union[float, int, ArrayLike],
        TEAM_FGA: Union[float, int, ArrayLike],
        TEAM_FGM: Union[float, int, ArrayLike],
        TEAM_FTA: Union[float, int, ArrayLike],
        TEAM_FTM: Union[float, int, ArrayLike],
        TEAM_OREB: Union[float, int, ArrayLike],
        TEAM_AST: Union[float, int, ArrayLike],
        TEAM_TOV: Union[float, int, ArrayLike],
        TEAM_MP: Union[float, int, ArrayLike],
        OPP_DREB: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Scoring Possessions (ScPoss) Component of Individual Total Possessions used in Offensive Rating.
            ScPoss = (FG_PART + AST_PART + FT_PART) * (1 - ((TEAM_OREB / TEAM_SCORING_POSS) * TEAM_OREB_WEIGHT * TEAM_PLAY%) + OREB_PART

        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            PTS (Union[float, int, ArrayLike]): points\n
            FGA (Union[float, int, ArrayLike]): field goal attempts\n
            FGM (Union[float, int, ArrayLike]): field goal makes\n
            FTA (Union[float, int, ArrayLike]): free throw attempts\n
            FTM (Union[float, int, ArrayLike]): free throw makes\n
            OREB (Union[float, int, ArrayLike]): offensive rebounds\n
            AST (Union[float, int, ArrayLike]): assists\n
            MP (Union[float, int, ArrayLike]): player minutes played\n
            TEAM_PTS (Union[float, int, ArrayLike]): team points\n
            TEAM_FGA (Union[float, int, ArrayLike]): team field goal attempts\n
            TEAM_FGM (Union[float, int, ArrayLike]): team field goal makes\n
            TEAM_FTA (Union[float, int, ArrayLike]): team free throw attempts\n
            TEAM_FTM (Union[float, int, ArrayLike]): team free throw makes\n
            TEAM_OREB (Union[float, int, ArrayLike]): team offensive rebounds\n
            TEAM_AST (Union[float, int, ArrayLike]): team assists\n
            TEAM_TOV (Union[float, int, ArrayLike]): team turnovers\n
            TEAM_MP (Union[float, int, ArrayLike]): team minutes played\n
            OPP_DREB (Union[float, int, ArrayLike]): opponent defensive rebounds\n

        Returns:
            Union[float, int, ArrayLike]: Scoring Possessions component of Individual Total Possessions.
        """
        return np.add(
            np.multiply(
                np.sum(
                    np.array(
                        [
                            self._scposs_fg_part(
                                PTS=PTS,
                                FGA=FGA,
                                FGM=FGM,
                                FTM=FTM,
                                AST=AST,
                                MP=MP,
                                TEAM_MP=TEAM_MP,
                                TEAM_FGM=TEAM_FGM,
                                TEAM_AST=TEAM_AST,
                            ),
                            self._scposs_assist_part(
                                PTS=PTS,
                                FGA=FGA,
                                FTM=FTM,
                                AST=AST,
                                TEAM_PTS=TEAM_PTS,
                                TEAM_FGA=TEAM_FGA,
                                TEAM_FTM=TEAM_FTM,
                            ),
                            self._scposs_ft_part(FTM=FTM, FTA=FTA),
                        ]
                    ),
                    axis=0,
                ),
                np.multiply(
                    np.subtract(
                        1,
                        np.divide(
                            TEAM_OREB,
                            self._team_scoring_possessions(
                                TEAM_FGM=TEAM_FGM, TEAM_FTA=TEAM_FTA, TEAM_FTM=TEAM_FTM
                            ),
                        ),
                    ),
                    np.multiply(
                        self._team_oreb_weight(
                            TEAM_FGA=TEAM_FGA,
                            TEAM_FGM=TEAM_FGM,
                            TEAM_FTA=TEAM_FTA,
                            TEAM_FTM=TEAM_FTM,
                            TEAM_OREB=TEAM_OREB,
                            TEAM_TOV=TEAM_TOV,
                            OPP_DREB=OPP_DREB,
                        ),
                        self._team_play_pct(
                            TEAM_FGA=TEAM_FGA,
                            TEAM_FGM=TEAM_FGM,
                            TEAM_FTA=TEAM_FTA,
                            TEAM_FTM=TEAM_FTM,
                            TEAM_TOV=TEAM_TOV,
                        ),
                    ),
                ),
            ),
            self._oreb_part(
                OREB=OREB,
                TEAM_FGA=TEAM_FGA,
                TEAM_FGM=TEAM_FGM,
                TEAM_FTA=TEAM_FTA,
                TEAM_FTM=TEAM_FTM,
                TEAM_OREB=TEAM_OREB,
                TEAM_TOV=TEAM_TOV,
                OPP_DREB=OPP_DREB,
            ),
        )

    def missed_fg_possessions(
        self,
        FGA: Union[float, int, ArrayLike],
        FGM: Union[float, int, ArrayLike],
        TEAM_OREB: Union[float, int, ArrayLike],
        OPP_DREB: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Missed Field Goal Possession component of Individual Total Possessions.
            FGxPOSS = (FGA - FGM) * (1 - 1.07 * TEAM_OREB%)

        Estimates number of possessions ending in a missed field goal.
        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            FGA (Union[float, int, ArrayLike]): field goal attempts\n
            FGM (Union[float, int, ArrayLike]): field goal makes\n
            TEAM_OREB (Union[float, int, ArrayLike]): team offensive rebounds\n
            OPP_DREB (Union[float, int, ArrayLike]): opponent defensive rebounds\n

        Returns:
            Union[float, int, ArrayLike]: Missed Field Goal Possessions.
        """
        return np.multiply(
            np.subtract(FGA, FGM),
            np.subtract(
                1,
                np.multiply(
                    1.07,
                    self.team_stats.offensive_rebound_pct(
                        OREB=TEAM_OREB, OPP_DREB=OPP_DREB
                    ),
                ),
            ),
        )

    def missed_ft_possessions(
        self, FTA: Union[float, int, ArrayLike], FTM: Union[float, int, ArrayLike]
    ) -> Union[float, int, ArrayLike]:
        """
        Missed Free Throw Possessions component of Individual Total Possessions.
            FTxPOSS = (1 - (FTM / FTA)**2) * 0.4 * FTA

        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            FTA (Union[float, int, ArrayLike]): free throw attempts\n
            FTM (Union[float, int, ArrayLike]): free throw makes\n

        Returns:
            Union[float, int, ArrayLike]: Missed Free Throw Possessions.
        """
        return np.multiply(
            np.subtract(1, np.power(np.divide(FTM, FTA), 2)),
            np.multiply(self._ft_weight, FTA),
        )

    def total_possessions(
        self,
        PTS: Union[float, int, ArrayLike],
        FGA: Union[float, int, ArrayLike],
        FGM: Union[float, int, ArrayLike],
        FTA: Union[float, int, ArrayLike],
        FTM: Union[float, int, ArrayLike],
        OREB: Union[float, int, ArrayLike],
        AST: Union[float, int, ArrayLike],
        TOV: Union[float, int, ArrayLike],
        MP: Union[float, int, ArrayLike],
        TEAM_PTS: Union[float, int, ArrayLike],
        TEAM_FGA: Union[float, int, ArrayLike],
        TEAM_FGM: Union[float, int, ArrayLike],
        TEAM_FTA: Union[float, int, ArrayLike],
        TEAM_FTM: Union[float, int, ArrayLike],
        TEAM_OREB: Union[float, int, ArrayLike],
        TEAM_AST: Union[float, int, ArrayLike],
        TEAM_TOV: Union[float, int, ArrayLike],
        TEAM_MP: Union[float, int, ArrayLike],
        OPP_DREB: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Total Possessions used to calculate a player's offensive rating.
            = ScPoss + FGxPOSS + FTxPOSS + TOV

        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            PTS (Union[float, int, ArrayLike]): points\n
            FGA (Union[float, int, ArrayLike]): field goal attempts\n
            FGM (Union[float, int, ArrayLike]): field goal makes\n
            FTA (Union[float, int, ArrayLike]): free throw attempts\n
            FTM (Union[float, int, ArrayLike]): free throw makes\n
            OREB (Union[float, int, ArrayLike]): offensive rebounds\n
            AST (Union[float, int, ArrayLike]): assists\n
            TOV (Union[float, int, ArrayLike]): turnovers\n
            MP (Union[float, int, ArrayLike]): player minutes played\n
            TEAM_PTS (Union[float, int, ArrayLike]): team points\n
            TEAM_FGA (Union[float, int, ArrayLike]): team field goal attempts\n
            TEAM_FGM (Union[float, int, ArrayLike]): team field goal makes\n
            TEAM_FTA (Union[float, int, ArrayLike]): team free throw attempts\n
            TEAM_FTM (Union[float, int, ArrayLike]): team free throw makes\n
            TEAM_OREB (Union[float, int, ArrayLike]): team offensive rebounds\n
            TEAM_AST (Union[float, int, ArrayLike]): team assists\n
            TEAM_TOV (Union[float, int, ArrayLike]): team turnovers\n
            TEAM_MP (Union[float, int, ArrayLike]): team minutes played\n
            OPP_DREB (Union[float, int, ArrayLike]): opponent defensive rebounds\n

        Returns:
            Union[float, int, ArrayLike]: Total Possessions.
        """
        return np.sum(
            np.array(
                [
                    self._scoring_possessions(
                        PTS=PTS,
                        FGA=FGA,
                        FGM=FGM,
                        FTA=FTA,
                        FTM=FTM,
                        OREB=OREB,
                        AST=AST,
                        MP=MP,
                        TEAM_PTS=TEAM_PTS,
                        TEAM_FGA=TEAM_FGA,
                        TEAM_FGM=TEAM_FGM,
                        TEAM_FTA=TEAM_FTA,
                        TEAM_FTM=TEAM_FTM,
                        TEAM_OREB=TEAM_OREB,
                        TEAM_AST=TEAM_AST,
                        TEAM_TOV=TEAM_TOV,
                        TEAM_MP=TEAM_MP,
                        OPP_DREB=OPP_DREB,
                    ),
                    self.missed_fg_possessions(
                        FGA=FGA, FGM=FGM, TEAM_OREB=TEAM_OREB, OPP_DREB=OPP_DREB
                    ),
                    self.missed_ft_possessions(FTA=FTA, FTM=FTM),
                    TOV,
                ],
            ),
            axis=0,
        )

    def _pprod_fg_part(
        self,
        PTS: Union[float, int, ArrayLike],
        FGA: Union[float, int, ArrayLike],
        FGM: Union[float, int, ArrayLike],
        FG3M: Union[float, int, ArrayLike],
        FTM: Union[float, int, ArrayLike],
        AST: Union[float, int, ArrayLike],
        MP: Union[float, int, ArrayLike],
        TEAM_FGM: Union[float, int, ArrayLike],
        TEAM_AST: Union[float, int, ArrayLike],
        TEAM_MP: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Field Goal Part of Individual Points Produced.
            PPROD_FG_PART = 2 * (FGM + 0.5 * FG3M) * (1 - 0.5 * ((PTS - FTM) / (2 * FGA)) * _qAST)

        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            PTS (Union[float, int, ArrayLike]): points\n
            FGA (Union[float, int, ArrayLike]): field goal attempts\n
            FGM (Union[float, int, ArrayLike]): field goal makes\n
            FG3M (Union[float, int, ArrayLike]): three point field goal makes\n
            FTM (Union[float, int, ArrayLike]): free throw makes\n
            AST (Union[float, int, ArrayLike]): assists\n
            MP (Union[float, int, ArrayLike]): player minutes played\n
            TEAM_FGM (Union[float, int, ArrayLike]): team field goal makes\n
            TEAM_AST (Union[float, int, ArrayLike]): team assists\n
            TEAM_MP (Union[float, int, ArrayLike]): team minutes played\n

        Returns:
            Union[float, int, ArrayLike]: Field Goal Part of Individual Points Produced.
        """
        return np.multiply(
            np.multiply(2, np.multiply(np.add(FGM, 0.5), FG3M)),
            np.multiply(
                np.subtract(
                    1,
                    np.multiply(
                        0.5, np.divide(np.subtract(PTS, FTM), np.multiply(2, FGA))
                    ),
                ),
                self._qAST(
                    FGM=FGM,
                    AST=AST,
                    MP=MP,
                    TEAM_FGM=TEAM_FGM,
                    TEAM_AST=TEAM_AST,
                    TEAM_MP=TEAM_MP,
                ),
            ),
        )

    def _pprod_ast_part(
        self,
        PTS: Union[float, int, ArrayLike],
        FGA: Union[float, int, ArrayLike],
        FGM: Union[float, int, ArrayLike],
        FTM: Union[float, int, ArrayLike],
        FG3M: Union[float, int, ArrayLike],
        AST: Union[float, int, ArrayLike],
        TEAM_PTS: Union[float, int, ArrayLike],
        TEAM_FGA: Union[float, int, ArrayLike],
        TEAM_FGM: Union[float, int, ArrayLike],
        TEAM_FTM: Union[float, int, ArrayLike],
        TEAM_FG3M: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Assist Part of Individual Points Produced.
            PPROD_AST_PART = 2 * ((TEAM_FGM - FGM + 0.5 * (TEAM_FG3M - FG3M)) / (TEAM_FGM - FGM)) * 0.5 * ((TEAM_PTS - TEAM_FTM - (PTS - FTM)) / (2 * (TEAM_FGA - FGA))) * AST

        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            PTS (Union[float, int, ArrayLike]): points\n
            FGA (Union[float, int, ArrayLike]): field goal attempts\n
            FGM (Union[float, int, ArrayLike]): field goal makes\n
            FTM (Union[float, int, ArrayLike]): free throw makes\n
            FG3M (Union[float, int, ArrayLike]): three point field goal makes\n
            AST (Union[float, int, ArrayLike]): assists\n
            TEAM_PTS (Union[float, int, ArrayLike]): team points\n
            TEAM_FGA (Union[float, int, ArrayLike]): team field goal attempts\n
            TEAM_FGM (Union[float, int, ArrayLike]): team field goal makes\n
            TEAM_FTM (Union[float, int, ArrayLike]): team free throw makes\n
            TEAM_FG3M (Union[float, int, ArrayLike]): team three point field goal makes\n

        Returns:
            Union[float, int, ArrayLike]: Assist Part of Individual Points Produced.
        """
        return np.multiply(
            np.multiply(
                np.divide(
                    np.multiply(
                        np.add(np.subtract(TEAM_FGM, FGM), 0.5),
                        np.subtract(TEAM_FG3M, FG3M),
                    ),
                    np.subtract(TEAM_FGM, FGM),
                ),
                np.divide(
                    np.subtract(np.subtract(TEAM_PTS, TEAM_FTM), np.subtract(PTS, FTM)),
                    np.multiply(2, np.subtract(TEAM_FGA, FGA)),
                ),
            ),
            AST,
        )

    def _pprod_oreb_part(
        self,
        OREB: Union[float, int, ArrayLike],
        TEAM_PTS: Union[float, int, ArrayLike],
        TEAM_FGA: Union[float, int, ArrayLike],
        TEAM_FGM: Union[float, int, ArrayLike],
        TEAM_FTA: Union[float, int, ArrayLike],
        TEAM_FTM: Union[float, int, ArrayLike],
        TEAM_OREB: Union[float, int, ArrayLike],
        TEAM_TOV: Union[float, int, ArrayLike],
        OPP_DREB: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Offensive Rebound Part of Individual Points Produced.
            PPROD_OREB_PART = OREB * TEAM_OREB_WEIGHT * TEAM_PLAY% * (TEAM_PTS / (TEAM_FGM + (1 - 1 - (TEAM_FTM / TEAM_FTA))**2) * 0.44 * TEAM_FTA)))))

        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            OREB (Union[float, int, ArrayLike]): offensive rebounds\n
            TEAM_PTS (Union[float, int, ArrayLike]): team points\n
            TEAM_FGA (Union[float, int, ArrayLike]): team field goal attempts\n
            TEAM_FGM (Union[float, int, ArrayLike]): team field goal makes\n
            TEAM_FTA (Union[float, int, ArrayLike]): team free throw attempts\n
            TEAM_FTM (Union[float, int, ArrayLike]): team free throw makes\n
            TEAM_OREB (Union[float, int, ArrayLike]): team offensive rebounds\n
            TEAM_TOV (Union[float, int, ArrayLike]): team turnovers\n
            OPP_DREB (Union[float, int, ArrayLike]): opponent defensive rebounds\n

        Returns:
            Union[float, int, ArrayLike]: Offensive Rebound Part of Individual Points Produced.
        """
        return np.multiply(
            self._oreb_part(
                OREB=OREB,
                TEAM_FGA=TEAM_FGA,
                TEAM_FGM=TEAM_FGM,
                TEAM_FTA=TEAM_FTA,
                TEAM_FTM=TEAM_FTM,
                TEAM_OREB=TEAM_OREB,
                TEAM_TOV=TEAM_TOV,
                OPP_DREB=OPP_DREB,
            ),
            np.multiply(
                np.divide(
                    TEAM_PTS,
                    np.add(
                        TEAM_FGM,
                        np.subtract(
                            1,
                            np.subtract(1, np.power(np.divide(TEAM_FTM, TEAM_FTA), 2)),
                        ),
                    ),
                ),
                np.multiply(self._ft_weight, TEAM_FTA),
            ),
        )

    def individual_points_produced(
        self,
        PTS: Union[float, int, ArrayLike],
        FGA: Union[float, int, ArrayLike],
        FGM: Union[float, int, ArrayLike],
        FG3M: Union[float, int, ArrayLike],
        FTM: Union[float, int, ArrayLike],
        OREB: Union[float, int, ArrayLike],
        AST: Union[float, int, ArrayLike],
        MP: Union[float, int, ArrayLike],
        TEAM_PTS: Union[float, int, ArrayLike],
        TEAM_FGA: Union[float, int, ArrayLike],
        TEAM_FGM: Union[float, int, ArrayLike],
        TEAM_FG3M: Union[float, int, ArrayLike],
        TEAM_FTA: Union[float, int, ArrayLike],
        TEAM_FTM: Union[float, int, ArrayLike],
        TEAM_OREB: Union[float, int, ArrayLike],
        TEAM_AST: Union[float, int, ArrayLike],
        TEAM_TOV: Union[float, int, ArrayLike],
        TEAM_MP: Union[float, int, ArrayLike],
        OPP_DREB: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Individual Points Produced (PProd) used to calculate a player's offensive rating.
            PProd = (PProd_FG_Part + PProd_AST_Part + FTM) * (1 - TEAM_OREB / TEAM_SCORING_POSS) * TEAM_OREB_WEIGHT * TEAM_PLAY%) + PProd_OREB_Part

        Estimates points produced by a player.
        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            PTS (Union[float, int, ArrayLike]): points\n
            FGA (Union[float, int, ArrayLike]): field goal attempts\n
            FGM (Union[float, int, ArrayLike]): field goal makes\n
            FG3M (Union[float, int, ArrayLike]): three point field goal makes\n
            FTM (Union[float, int, ArrayLike]): free throw makes\n
            OREB (Union[float, int, ArrayLike]): offensive rebounds\n
            AST (Union[float, int, ArrayLike]): assists\n
            MP (Union[float, int, ArrayLike]): player minutes played\n
            TEAM_PTS (Union[float, int, ArrayLike]): team points\n
            TEAM_FGA (Union[float, int, ArrayLike]): team field goal attempts\n
            TEAM_FGM (Union[float, int, ArrayLike]): team field goal makes\n
            TEAM_FG3M (Union[float, int, ArrayLike]): team three point field goal makes\n
            TEAM_FTA (Union[float, int, ArrayLike]): team free throw attempts\n
            TEAM_FTM (Union[float, int, ArrayLike]): team free throw makes\n
            TEAM_OREB (Union[float, int, ArrayLike]): team offensive rebounds\n
            TEAM_AST (Union[float, int, ArrayLike]): team assists\n
            TEAM_TOV (Union[float, int, ArrayLike]): team turnovers\n
            TEAM_MP (Union[float, int, ArrayLike]): team minutes played\n
            OPP_DREB (Union[float, int, ArrayLike]): opponent defensive rebounds\n

        Returns:
            Union[float, int, ArrayLike]: Individual Points Produced.
        """
        return np.add(
            np.multiply(
                np.sum(
                    np.array(
                        [
                            self._pprod_fg_part(
                                PTS=PTS,
                                FGA=FGA,
                                FGM=FGM,
                                FG3M=FG3M,
                                FTM=FTM,
                                AST=AST,
                                MP=MP,
                                TEAM_FGM=TEAM_FGM,
                                TEAM_AST=TEAM_AST,
                                TEAM_MP=TEAM_MP,
                            ),
                            self._pprod_ast_part(
                                PTS=PTS,
                                FGA=FGA,
                                FGM=FGM,
                                FTM=FTM,
                                FG3M=FG3M,
                                AST=AST,
                                TEAM_PTS=TEAM_PTS,
                                TEAM_FGA=TEAM_FGA,
                                TEAM_FGM=TEAM_FGM,
                                TEAM_FTM=TEAM_FTM,
                                TEAM_FG3M=TEAM_FG3M,
                            ),
                            FTM,
                        ]
                    ),
                    axis=0,
                ),
                np.multiply(
                    np.multiply(
                        np.subtract(
                            1,
                            np.divide(
                                TEAM_OREB,
                                self._team_scoring_possessions(
                                    TEAM_FGM=TEAM_FGM,
                                    TEAM_FTA=TEAM_FTA,
                                    TEAM_FTM=TEAM_FTM,
                                ),
                            ),
                        ),
                        self._team_oreb_weight(
                            TEAM_FGA=TEAM_FGA,
                            TEAM_FGM=TEAM_FGM,
                            TEAM_FTA=TEAM_FTA,
                            TEAM_FTM=TEAM_FTM,
                            TEAM_OREB=TEAM_OREB,
                            TEAM_TOV=TEAM_TOV,
                            OPP_DREB=OPP_DREB,
                        ),
                    ),
                    self._team_play_pct(
                        TEAM_FGA=TEAM_FGA,
                        TEAM_FGM=TEAM_FGM,
                        TEAM_FTA=TEAM_FTA,
                        TEAM_FTM=TEAM_FTM,
                        TEAM_TOV=TEAM_TOV,
                    ),
                ),
            ),
            self._pprod_oreb_part(
                OREB=OREB,
                TEAM_PTS=TEAM_PTS,
                TEAM_FGA=TEAM_FGA,
                TEAM_FGM=TEAM_FGM,
                TEAM_FTA=TEAM_FTA,
                TEAM_FTM=TEAM_FTM,
                TEAM_OREB=TEAM_OREB,
                TEAM_TOV=TEAM_TOV,
                OPP_DREB=OPP_DREB,
            ),
        )

    def offensive_rating(
        self,
        PTS: Union[float, int, ArrayLike],
        FGA: Union[float, int, ArrayLike],
        FGM: Union[float, int, ArrayLike],
        FG3M: Union[float, int, ArrayLike],
        FTA: Union[float, int, ArrayLike],
        FTM: Union[float, int, ArrayLike],
        OREB: Union[float, int, ArrayLike],
        AST: Union[float, int, ArrayLike],
        TOV: Union[float, int, ArrayLike],
        MP: Union[float, int, ArrayLike],
        TEAM_PTS: Union[float, int, ArrayLike],
        TEAM_FGA: Union[float, int, ArrayLike],
        TEAM_FGM: Union[float, int, ArrayLike],
        TEAM_FG3M: Union[float, int, ArrayLike],
        TEAM_FTA: Union[float, int, ArrayLike],
        TEAM_FTM: Union[float, int, ArrayLike],
        TEAM_OREB: Union[float, int, ArrayLike],
        TEAM_AST: Union[float, int, ArrayLike],
        TEAM_TOV: Union[float, int, ArrayLike],
        TEAM_MP: Union[float, int, ArrayLike],
        OPP_DREB: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Player Offensive Rating (OffRtg)
            OffRtg = 100 * (Individual Points Produced / Total Possessions)

        Estimates points produced by a player per 100 possessions played.
        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            PTS (Union[float, int, ArrayLike]): points\n
            FGA (Union[float, int, ArrayLike]): field goal attempts\n
            FGM (Union[float, int, ArrayLike]): field goal makes\n
            FG3M (Union[float, int, ArrayLike]): three point field goal makes\n
            FTA (Union[float, int, ArrayLike]): free throw attempts\n
            FTM (Union[float, int, ArrayLike]): free throw makes\n
            OREB (Union[float, int, ArrayLike]): offensive rebounds\n
            AST (Union[float, int, ArrayLike]): assists\n
            TOV (Union[float, int, ArrayLike]): turnovers\n
            MP (Union[float, int, ArrayLike]): player minutes played\n
            TEAM_PTS (Union[float, int, ArrayLike]): team points\n
            TEAM_FGA (Union[float, int, ArrayLike]): team field goal attempts\n
            TEAM_FGM (Union[float, int, ArrayLike]): team field goal makes\n
            TEAM_FG3M (Union[float, int, ArrayLike]): team three point field goal makes\n
            TEAM_FTA (Union[float, int, ArrayLike]): team free throw attempts\n
            TEAM_FTM (Union[float, int, ArrayLike]): team free throw makes\n
            TEAM_OREB (Union[float, int, ArrayLike]): team offensive rebounds\n
            TEAM_AST (Union[float, int, ArrayLike]): team assists\n
            TEAM_TOV (Union[float, int, ArrayLike]): team turnovers\n
            TEAM_MP (Union[float, int, ArrayLike]): team minutes played\n
            OPP_DREB (Union[float, int, ArrayLike]): opponent defensive rebounds\n

        Returns:
            Union[float, int, ArrayLike]: player offensive rating.
        """
        return np.multiply(
            100,
            np.divide(
                self.individual_points_produced(
                    PTS=PTS,
                    FGA=FGA,
                    FGM=FGM,
                    FG3M=FG3M,
                    FTM=FTM,
                    OREB=OREB,
                    AST=AST,
                    MP=MP,
                    TEAM_PTS=TEAM_PTS,
                    TEAM_FGA=TEAM_FGA,
                    TEAM_FGM=TEAM_FGM,
                    TEAM_FG3M=TEAM_FG3M,
                    TEAM_FTA=TEAM_FTA,
                    TEAM_FTM=TEAM_FTM,
                    TEAM_OREB=TEAM_OREB,
                    TEAM_AST=TEAM_AST,
                    TEAM_TOV=TEAM_TOV,
                    TEAM_MP=TEAM_MP,
                    OPP_DREB=OPP_DREB,
                ),
                self.total_possessions(
                    PTS=PTS,
                    FGA=FGA,
                    FGM=FGM,
                    FTA=FTA,
                    FTM=FTM,
                    OREB=OREB,
                    AST=AST,
                    TOV=TOV,
                    MP=MP,
                    TEAM_PTS=TEAM_PTS,
                    TEAM_FGA=TEAM_FGA,
                    TEAM_FGM=TEAM_FGM,
                    TEAM_FTA=TEAM_FTA,
                    TEAM_FTM=TEAM_FTM,
                    TEAM_OREB=TEAM_OREB,
                    TEAM_AST=TEAM_AST,
                    TEAM_TOV=TEAM_TOV,
                    TEAM_MP=TEAM_MP,
                    OPP_DREB=OPP_DREB,
                ),
            ),
        )

    def floor_pct(
        self,
        PTS: Union[float, int, ArrayLike],
        FGA: Union[float, int, ArrayLike],
        FGM: Union[float, int, ArrayLike],
        FTA: Union[float, int, ArrayLike],
        FTM: Union[float, int, ArrayLike],
        OREB: Union[float, int, ArrayLike],
        AST: Union[float, int, ArrayLike],
        TOV: Union[float, int, ArrayLike],
        MP: Union[float, int, ArrayLike],
        TEAM_PTS: Union[float, int, ArrayLike],
        TEAM_FGA: Union[float, int, ArrayLike],
        TEAM_FGM: Union[float, int, ArrayLike],
        TEAM_FTA: Union[float, int, ArrayLike],
        TEAM_FTM: Union[float, int, ArrayLike],
        TEAM_OREB: Union[float, int, ArrayLike],
        TEAM_AST: Union[float, int, ArrayLike],
        TEAM_TOV: Union[float, int, ArrayLike],
        TEAM_MP: Union[float, int, ArrayLike],
        OPP_DREB: Union[float, int, ArrayLike],
    ) -> Union[float, int, ArrayLike]:
        """
        Floor Percentage (Floor%)
            = Scoring Possessions / Total Possessions

        Answers the question, "What percentage of a player's possessions end in a score?"
        Offensive Rating takes into account number of points score per possession.

        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            PTS (Union[float, int, ArrayLike]): points\n
            FGA (Union[float, int, ArrayLike]): field goal attempts\n
            FGM (Union[float, int, ArrayLike]): field goal makes\n
            FTA (Union[float, int, ArrayLike]): free throw attempts\n
            FTM (Union[float, int, ArrayLike]): free throw makes\n
            OREB (Union[float, int, ArrayLike]): offensive rebounds\n
            AST (Union[float, int, ArrayLike]): assists\n
            TOV (Union[float, int, ArrayLike]): turnovers\n
            MP (Union[float, int, ArrayLike]): player minutes played\n
            TEAM_PTS (Union[float, int, ArrayLike]): team points\n
            TEAM_FGA (Union[float, int, ArrayLike]): team field goal attempts\n
            TEAM_FGM (Union[float, int, ArrayLike]): team field goal makes\n
            TEAM_FTA (Union[float, int, ArrayLike]): team free throw attempts\n
            TEAM_FTM (Union[float, int, ArrayLike]): team free throw makes\n
            TEAM_OREB (Union[float, int, ArrayLike]): team offensive rebounds\n
            TEAM_AST (Union[float, int, ArrayLike]): team assists\n
            TEAM_TOV (Union[float, int, ArrayLike]): team turnovers\n
            TEAM_MP (Union[float, int, ArrayLike]): team minutes played\n
            OPP_DREB (Union[float, int, ArrayLike]): opponent defensive rebounds\n

        Returns:
            Union[float, int, ArrayLike]: Floor Percentage.
        """
        return np.divide(
            self._scoring_possessions(
                PTS=PTS,
                FGA=FGA,
                FGM=FGM,
                FTA=FTA,
                FTM=FTM,
                OREB=OREB,
                AST=AST,
                MP=MP,
                TEAM_PTS=TEAM_PTS,
                TEAM_FGA=TEAM_FGA,
                TEAM_FGM=TEAM_FGM,
                TEAM_FTA=TEAM_FTA,
                TEAM_FTM=TEAM_FTM,
                TEAM_OREB=TEAM_OREB,
                TEAM_AST=TEAM_AST,
                TEAM_TOV=TEAM_TOV,
                TEAM_MP=TEAM_MP,
                OPP_DREB=OPP_DREB,
            ),
            self.total_possessions(
                PTS=PTS,
                FGA=FGA,
                FGM=FGM,
                FTA=FTA,
                FTM=FTM,
                OREB=OREB,
                AST=AST,
                TOV=TOV,
                MP=MP,
                TEAM_PTS=TEAM_PTS,
                TEAM_FGA=TEAM_FGA,
                TEAM_FGM=TEAM_FGM,
                TEAM_FTA=TEAM_FTA,
                TEAM_FTM=TEAM_FTM,
                TEAM_OREB=TEAM_OREB,
                TEAM_AST=TEAM_AST,
                TEAM_TOV=TEAM_TOV,
                TEAM_MP=TEAM_MP,
                OPP_DREB=OPP_DREB,
            ),
        )

    def win_shares(self) -> Union[float, int, ArrayLike]:
        return []

    def off_win_shares(self) -> Union[float, int, ArrayLike]:
        return []

    def def_win_shares(self) -> Union[float, int, ArrayLike]:
        return []

    def box_plus_minus(self) -> Union[float, int, ArrayLike]:
        """
        Box Plus/Minus, v2.0 (BPM).
            BPM =

        Estimate's basketball player's contribution to a team while on the court using simple box score statistics.\n
        Source: https://www.basketball-reference.com/about/bpm2.html.
        Created by Daniel Myers.

        BPM uses a player's box score information, position, and the team's overall performance to estimate the
        player's contribution in points above league average per 100 possessions played.\n
        BPM does not take into account playing time -- it is purely a rate stat!

        To give a sense of the scale:
            +10.0 is an all-time season (think peak Jordan or LeBron)\n
            +8.0 is an MVP season (think peak Dirk or peak Shaq)\n
            +6.0 is an all-NBA season\n
            +4.0 is in all-star consideration\n
            +2.0 is a good starter\n
            +0.0 is a decent starter or solid 6th man\n
            -2.0 is a bench player (this is also defined as 'replacement level')\n
            Below -2.0 are many end-of-bench players\n



        """
        return []

    def value_over_replacement(self) -> Union[float, int, ArrayLike]:
        return []

    def wins_above_replacement(self) -> Union[float, int, ArrayLike]:
        """
        Wins Above Replacement Player (WARP)
            WARP =

        Estimates how many wins a player contributes to his team above what a replacement player would.
        Created by Kevin Pelton.
        Source: http://www.sonicscentral.com/warp.html
        """
        return []

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
