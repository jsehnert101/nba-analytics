# %%
# Imports
import inspect
from typing import List
import numpy as np
from numpy.typing import ArrayLike
from features.stats import Stats
from features.team_stats import TeamStats


# %%
# Create object to calculate player statistics.
class PlayerStats(Stats):
    """
    Compute statistics for NBA Players.

    Your life will be easier if you use the same names as the NBA API.

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
        self.independent_stat_method_map.update({"GAME_SCORE": self.game_score})
        self.dependent_stat_method_map = (
            {  # Map functions which require opponent/team data
                "AST%": self.assist_pct,
                "DREB%": self.defensive_rebound_pct,
                "OREB%": self.offensive_rebound_pct,
                "REB%": self.rebound_pct,
                "BLK%": self.block_pct,
                "STL%": self.steal_pct,
                "USG%": self.usage_rate,
                "TOT_POSS": self.total_possessions,
                "PPROD": self.points_produced,
                "OffRtg": self.offensive_rating,
                "FLOOR%": self.floor_pct,
            }
        )
        self.required_stat_params = self._get_required_stat_params()

    def assist_pct(
        self,
        AST: ArrayLike,
        FGM: ArrayLike,
        MP: ArrayLike,
        TEAM_FGM: ArrayLike,
        TEAM_MP: ArrayLike,
        **_
    ) -> ArrayLike:
        """
        Assist Percentage / Rate (AST%)
            AST% = 100 * AST / (((MP / (TEAM_MP / 5)) * TEAM_FGM) - FGM)

        Estimates percentage of teammate field goals a player assisted while on the floor.
        Source: https://www.basketball-reference.com/about/glossary.html#ast

        Args:
            AST (ArrayLike): assists\n
            FGM (ArrayLike): player field goal makes\n
            MP (ArrayLike): player minutes played\n
            TEAM_FGM (ArrayLike): team field goal makes\n
            TEAM_MP (ArrayLike: team minutes played\n

        Returns:
            ArrayLike: player assist rate/percentage
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
        DREB: ArrayLike,
        MP: ArrayLike,
        TEAM_DREB: ArrayLike,
        TEAM_MP: ArrayLike,
        OPP_OREB: ArrayLike,
        **_
    ) -> ArrayLike:
        """
        Defensive Rebounding Percentage / Rate (DREB%).
            DREB% = 100 * (DREB * (TEAM_MP / 5)) / (MP * (TEAM_DREB + OPP_OREB))

        Estimates percentage of available defensive rebounds a player grabbed while on the floor.
        Source: https://www.basketball-reference.com//about/glossary.html#drb

        Args:
            DREB (ArrayLike): defensive rebounds\n
            MP (ArrayLike): player minutes played\n
            TEAM_DREB (ArrayLike): team defensive rebounds\n
            TEAM_MP (ArrayLike): team minutes played\n
            OPP_OREB (ArrayLike): opponent offensive rebounds\n

        Returns:
            ArrayLike: defensive rebounding rate/percentage
        """
        return np.multiply(
            100,
            np.divide(
                np.multiply(DREB, np.divide(TEAM_MP, 5)),
                np.multiply(MP, np.add(TEAM_DREB, OPP_OREB)),
            ),
        )

    def offensive_rebound_pct(
        self,
        OREB: ArrayLike,
        MP: ArrayLike,
        TEAM_OREB: ArrayLike,
        TEAM_MP: ArrayLike,
        OPP_DREB: ArrayLike,
        **_
    ) -> ArrayLike:
        """
        Offensive Rebound Percentage (OREB%)
            OREB% = 100 * (OREB * (TEAM_MP / 5)) / (MP * (TEAM_OREB + OPP_DREB)).

        Estimates percentage of available offensive rebounds a player grabbed while on the floor.
        Source: https://www.basketball-reference.com//about/glossary.html#orb

        Args:
            OREB (ArrayLike): offensive rebounds\n
            PLAYER_MP (ArrayLike): player minutes played\n
            TEAM_OREB (ArrayLike): team offensive rebounds\n
            TEAM_MP (ArrayLike): team minutes played\n
            OPP_DREB (ArrayLike): opponent defensive rebounds\n

        Returns:
            ArrayLike: offensive rebound percentage
        """
        return np.multiply(
            100,
            np.divide(
                np.multiply(OREB, np.divide(TEAM_MP, 5)),
                np.multiply(MP, np.add(TEAM_OREB, OPP_DREB)),
            ),
        )

    def rebound_pct(
        self,
        DREB: ArrayLike,
        OREB: ArrayLike,
        MP: ArrayLike,
        TEAM_DREB: ArrayLike,
        TEAM_OREB: ArrayLike,
        TEAM_MP: ArrayLike,
        OPP_DREB: ArrayLike,
        OPP_OREB: ArrayLike,
    ) -> ArrayLike:
        """
        Rebound Percentage (REB%)
            REB% = DREB% + OREB%

        Args:
            DREB (ArrayLike): defensive rebounds\n
            OREB (ArrayLike): offensive rebounds\n
            MP (ArrayLike): player minutes played\n
            TEAM_DREB (ArrayLike): team defensive rebounds\n
            TEAM_OREB (ArrayLike): team offensive rebounds\n
            TEAM_MP (ArrayLike): team minutes played\n
            OPP_DREB (ArrayLike): opponent defensive rebounds\n
            OPP_OREB (ArrayLike): opponent offensive rebounds\n

        Returns:
            ArrayLike: rebound rate/percentage
        """
        return np.add(
            self.defensive_rebound_pct(
                DREB=DREB,
                MP=MP,
                TEAM_DREB=TEAM_DREB,
                TEAM_MP=TEAM_MP,
                OPP_OREB=OPP_OREB,
            ),
            self.offensive_rebound_pct(
                OREB=OREB,
                MP=MP,
                TEAM_OREB=TEAM_OREB,
                TEAM_MP=TEAM_MP,
                OPP_DREB=OPP_DREB,
            ),
        )

    def block_pct(
        self,
        BLK: ArrayLike,
        OPP_FGA: ArrayLike,
        OPP_FG3A: ArrayLike,
        PLAYER_MP: ArrayLike,
        TEAM_MP: ArrayLike,
        **_
    ) -> ArrayLike:
        """
        Block Percentage (BLK%)
            BLK% = 100 * (BLK * (MP_TEAM / 5)) / (MP_PLAYER * (OPP_FGA - OPP_3PA)).

        Estimates percentage of opponent 2PT FGA blocked by a player while he was on the floor.
        Source: https://www.basketball-reference.com/about/glossary.html#blk

        Args:
            BLK (ArrayLike): blocks\n
            PLAYER_MP (ArrayLike): player minutes played\n
            TEAM_MP (ArrayLike): team minutes played\n
            OPP_FGA (ArrayLike): opponent field goal attempts\n
            OPP_FG3A (ArrayLike): opponent three-point field goal attempts\n

        Returns:
            ArrayLike: player block rate / percentage
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
        STL: ArrayLike,
        MP: ArrayLike,
        TEAM_MP: ArrayLike,
        OPP_POSS: ArrayLike,
        **_
    ) -> ArrayLike:
        """
        Steal Percentage (STL%)
            STL% = 100 * (STL * (MP_TEAM / 5)) / (MP_PLAYER * POSS_OPP)

        Estimates percentage of opponent possessions that end in a player's steal while he was on the floor.
        Source: https://www.basketball-reference.com/about/glossary.html#stl

        Args:
            STL (ArrayLike): steals\n
            MP (ArrayLike): player minutes played\n
            MP_TEAM (ArrayLike): team minutes played\n
            POSS_OPP (ArrayLike): opponent possessions\n

        Returns:
            ArrayLike: steal percentage
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
        FGA: ArrayLike,
        FTA: ArrayLike,
        TOV: ArrayLike,
        MP: ArrayLike,
        TEAM_FGA: ArrayLike,
        TEAM_FTA: ArrayLike,
        TEAM_TOV: ArrayLike,
        TEAM_MP: ArrayLike,
        **_
    ) -> ArrayLike:
        """
        Usage Rate (USG%)
            USG% = 100 * ((FGA + 0.44 * FTA + TOV) * (MP_TEAM / 5)) / (MP * (TEAM_FGA + 0.44 * TEAM_FTA + TEAM_TOV))

        Estimates percentage of team plays used by a player while on the floor.

        Args:
            FGA (ArrayLike): field goal attempts\n
            FTA (ArrayLike): free throw attempts\n
            TOV (ArrayLike): turnovers\n
            MP (ArrayLike): player minutes played\n
            TEAM_MP (ArrayLike): team minutes played\n
            TEAM_FGA (ArrayLike): team field goal attempts\n
            TEAM_FTA (ArrayLike): team free throw attempts\n
            TEAM_TOV (ArrayLike): team turnovers\n
            TEAM_MP (ArrayLike): team minutes played\n

        Returns:
            ArrayLike: player usage rate.
        """
        player_minor_possessions = self.minor_possessions(FGA=FGA, FTA=FTA, TOV=TOV)
        team_minor_possessions = self.minor_possessions(
            FGA=TEAM_FGA, FTA=TEAM_FTA, TOV=TEAM_TOV
        )

        return np.multiply(
            100,
            np.divide(
                np.multiply(
                    player_minor_possessions,
                    np.divide(TEAM_MP, 5),
                ),
                np.multiply(
                    MP,
                    team_minor_possessions,
                ),
            ),
        )

    def _qAST(
        self,
        FGM: ArrayLike,
        AST: ArrayLike,
        MP: ArrayLike,
        TEAM_FGM: ArrayLike,
        TEAM_AST: ArrayLike,
        TEAM_MP: ArrayLike,
        **_
    ) -> ArrayLike:
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
            FGM (ArrayLike): field goal makes\n
            AST (ArrayLike): assists\n
            MP (ArrayLike): player minutes played\n
            TEAM_FGM (ArrayLike): team field goal makes\n
            TEAM_AST (ArrayLike): team assists\n
            TEAM_MP (ArrayLike): team minutes played\n

        Returns:
            ArrayLike: _qAST term in Field Goal Part of Scoring Possessions component of Individual Total Possessions.
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
        PTS: ArrayLike,
        FGA: ArrayLike,
        FGM: ArrayLike,
        FTM: ArrayLike,
        AST: ArrayLike,
        MP: ArrayLike,
        TEAM_MP: ArrayLike,
        TEAM_FGM: ArrayLike,
        TEAM_AST: ArrayLike,
        **_
    ) -> ArrayLike:
        """
        Field Goal Part of Scoring Possessions Component of Individual Total Possessions.
            FG_PART = FGM * (1 - 0.5 * ((PTS - FTM) / (2 * FGA)) * _qAST)

        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            PTS (ArrayLike): points\n
            FGA (ArrayLike): field goal attempts\n
            FGM (ArrayLike): field goal makes\n
            FTM (ArrayLike): free throw makes\n
            AST (ArrayLike): assists\n
            MP (ArrayLike): player minutes played\n
            TEAM_FGM (ArrayLike): team field goal makes\n
            TEAM_AST (ArrayLike): team assists\n
            TEAM_MP (ArrayLike): team minutes played\n

        Returns:
            ArrayLike: Field Goal Part of Scoring Possessions Component of Individual Total Possessions.
        """
        qAST = self._qAST(
            MP=MP,
            FGM=FGM,
            AST=AST,
            TEAM_MP=TEAM_MP,
            TEAM_FGM=TEAM_FGM,
            TEAM_AST=TEAM_AST,
        )
        return np.multiply(
            FGM,
            np.subtract(
                1,
                np.multiply(
                    0.5,
                    np.multiply(
                        np.divide(np.subtract(PTS, FTM), np.multiply(2, FGA)),
                        qAST,
                    ),
                ),
            ),
        )

    def _scposs_assist_part(
        self,
        PTS: ArrayLike,
        FGA: ArrayLike,
        FTM: ArrayLike,
        AST: ArrayLike,
        TEAM_PTS: ArrayLike,
        TEAM_FGA: ArrayLike,
        TEAM_FTM: ArrayLike,
        **_
    ) -> ArrayLike:
        """
        Assist Component of Scoring Possessions.
            AST_PART = 0.5 * (((TEAM_PTS - TEAM_FTM) - (PTS - FTM)) / (2 * (TEAM_FGA - FGA))) * AST

        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            PTS (ArrayLike): points\n
            FGA (ArrayLike): field goal attempts\n
            FTM (ArrayLike): free throw makes\n
            AST (ArrayLike): assists\n
            TEAM_PTS (ArrayLike): team points\n
            TEAM_FGS (ArrayLike): team field goals\n
            TEAM_FTA (ArrayLike): team free throw attempts\n

        Returns:
            ArrayLike: Assist Part of Scoring Possessions component of Individual Total Possessions.
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

    def _scposs_ft_part(self, FTM: ArrayLike, FTA: ArrayLike, **_) -> ArrayLike:
        """
        Free Throw Part of Scoring Possessions Component of Individual Total Possessions.
            FT_PART = (1 - (1 - (FTM / FTA))**2) * 0.4 * FTA

        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            FTM (ArrayLike): free throw makes\n
            FTA (ArrayLike): free throw attempts\n

        Returns:
            ArrayLike: Free Throw Part of Scoring Possessions component of Individual Total Possessions.
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
        self, TEAM_FGM: ArrayLike, TEAM_FTA: ArrayLike, TEAM_FTM: ArrayLike, **_
    ) -> ArrayLike:
        """
        Team Scoring Possessions term in Individual Total Possessions.
            TEAM_SCORING_POSS = TEAM_FGM + (1 - (1 - (TEAM_FTM / TEAM_FTA))**2) * 0.44 * TEAM_FTA

        Estimates number of successful team possessions (i.e. in which they score).

        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            TEAM_FGM (ArrayLike): team field goal makes\n
            TEAM_FTA (ArrayLike): team free throw attempts\n
            TEAM_FTM (ArrayLike): team free throw makes\n

        Returns:
            ArrayLike: Team Scoring Possessions Term in Individual Total Possessions.
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
                    self._ft_weight,
                    TEAM_FTA,
                ),
            ),
        )

    def _team_score_rate(  # TODO: transfer all team stats over to TeamStats object
        self,
        TEAM_FGA: ArrayLike,
        TEAM_FGM: ArrayLike,
        TEAM_FTA: ArrayLike,
        TEAM_FTM: ArrayLike,
        TEAM_TOV: ArrayLike,
        **_
    ) -> ArrayLike:
        """
        Team Score Rate term in OREB Part of Scoring Possessions component of Individual Total Possessions.
            TEAM_SCORE_RATE = TEAM_SCORING_POSS / TEAM_MINOR_POSSESSIONS

        Estimates percentage of possessions in which a team scores.
        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            TEAM_FGA (ArrayLike): team field goal attempts\n
            TEAM_FGM (ArrayLike): team field goal makes\n
            TEAM_FTA (ArrayLike): team free throw attempts\n
            TEAM_FTM (ArrayLike): team free throw makes\n
            TEAM_TOV (ArrayLike): team turnovers\n

        Returns:
            ArrayLike: Team Play Percentage term in OREB Part of Individual Total Possessions.
        """
        team_scoring_possessions = self._team_scoring_possessions(
            TEAM_FGM=TEAM_FGM, TEAM_FTA=TEAM_FTA, TEAM_FTM=TEAM_FTM
        )
        team_minor_possessions = self.minor_possessions(
            FGA=TEAM_FGA, FTA=TEAM_FTA, TOV=TEAM_TOV
        )
        return np.divide(team_scoring_possessions, team_minor_possessions)

    def _team_oreb_weight(
        self,
        TEAM_FGA: ArrayLike,
        TEAM_FGM: ArrayLike,
        TEAM_FTA: ArrayLike,
        TEAM_FTM: ArrayLike,
        TEAM_OREB: ArrayLike,
        TEAM_TOV: ArrayLike,
        OPP_DREB: ArrayLike,
        **_
    ) -> ArrayLike:
        """
        Team Offensive Rebound Weight term of OREB Part of Scoring Possessions Component of in Individual Total Possessions.
            TEAM_OREB_WEIGHT = ((1 - TEAM_OREB%) * TEAM_SCORE_RATE) / ((1 - TEAM_OREB%) * TEAM_SCORE_RATE + TEAM_OREB% * (1 - TEAM_SCORE_RATE))

        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            TEAM_FGA (ArrayLike): team field goal attempts\n
            TEAM_FGM (ArrayLike): team field goal makes\n
            TEAM_FTA (ArrayLike): team free throw attempts\n
            TEAM_FTM (ArrayLike): team free throw makes\n
            TEAM_OREB (ArrayLike): team offensive rebounds\n
            TEAM_TOV (ArrayLike): team turnovers\n
            OPP_DREB (ArrayLike): opponent defensive rebounds\n

        Returns:
            ArrayLike: Team Offensive Rebound Weight in Individual Total Possessions.
        """
        team_off_reb_pct = self.team_stats.offensive_rebound_pct(
            OREB=TEAM_OREB, OPP_DREB=OPP_DREB
        )
        team_score_rate = self._team_score_rate(
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
        OREB: ArrayLike,
        TEAM_FGA: ArrayLike,
        TEAM_FGM: ArrayLike,
        TEAM_FTA: ArrayLike,
        TEAM_FTM: ArrayLike,
        TEAM_OREB: ArrayLike,
        TEAM_TOV: ArrayLike,
        OPP_DREB: ArrayLike,
        **_
    ) -> ArrayLike:
        """
        Offensive Rebound Part of Scoring Possessions Component of Individual Total Possessions.
            TEAM_OFF_REB_PART = OREB * TEAM_OREB_WEIGHT * TEAM_SCORE_RATE

        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            OREB (ArrayLike): offensive rebounds\n
            TEAM_FGA (ArrayLike): team field goal attempts\n
            TEAM_FGM (ArrayLike): team field goal makes\n
            TEAM_FTA (ArrayLike): team free throw attempts\n
            TEAM_FTM (ArrayLike): team free throw makes\n
            TEAM_OREB (ArrayLike): team offensive rebounds\n
            TEAM_TOV (ArrayLike): team turnovers\n
            OPP_DREB (ArrayLike): opponent defensive rebounds\n

        Returns:
            ArrayLike: Offensive Rebound Part of Individual Total Possessions.
        """
        team_oreb_weight = self._team_oreb_weight(
            TEAM_FGA=TEAM_FGA,
            TEAM_FGM=TEAM_FGM,
            TEAM_FTA=TEAM_FTA,
            TEAM_FTM=TEAM_FTM,
            TEAM_OREB=TEAM_OREB,
            TEAM_TOV=TEAM_TOV,
            OPP_DREB=OPP_DREB,
        )
        team_score_rate = self._team_score_rate(
            TEAM_FGA=TEAM_FGA,
            TEAM_FGM=TEAM_FGM,
            TEAM_FTA=TEAM_FTA,
            TEAM_FTM=TEAM_FTM,
            TEAM_TOV=TEAM_TOV,
        )
        return np.multiply(
            OREB,
            np.multiply(
                team_oreb_weight,
                team_score_rate,
            ),
        )

    def _scoring_possessions(
        self,
        PTS: ArrayLike,
        FGA: ArrayLike,
        FGM: ArrayLike,
        FTA: ArrayLike,
        FTM: ArrayLike,
        OREB: ArrayLike,
        AST: ArrayLike,
        MP: ArrayLike,
        TEAM_PTS: ArrayLike,
        TEAM_FGA: ArrayLike,
        TEAM_FGM: ArrayLike,
        TEAM_FTA: ArrayLike,
        TEAM_FTM: ArrayLike,
        TEAM_OREB: ArrayLike,
        TEAM_AST: ArrayLike,
        TEAM_TOV: ArrayLike,
        TEAM_MP: ArrayLike,
        OPP_DREB: ArrayLike,
        **_
    ) -> ArrayLike:
        """
        Scoring Possessions (ScPoss) Component of Individual Total Possessions used in Offensive Rating.
            ScPoss = (FG_PART + AST_PART + FT_PART) * (1 - ((TEAM_OREB / TEAM_SCORING_POSS) * TEAM_OREB_WEIGHT * TEAM_SCORE_RATE) + OREB_PART

        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            PTS (ArrayLike): points\n
            FGA (ArrayLike): field goal attempts\n
            FGM (ArrayLike): field goal makes\n
            FTA (ArrayLike): free throw attempts\n
            FTM (ArrayLike): free throw makes\n
            OREB (ArrayLike): offensive rebounds\n
            AST (ArrayLike): assists\n
            MP (ArrayLike): player minutes played\n
            TEAM_PTS (ArrayLike): team points\n
            TEAM_FGA (ArrayLike): team field goal attempts\n
            TEAM_FGM (ArrayLike): team field goal makes\n
            TEAM_FTA (ArrayLike): team free throw attempts\n
            TEAM_FTM (ArrayLike): team free throw makes\n
            TEAM_OREB (ArrayLike): team offensive rebounds\n
            TEAM_AST (ArrayLike): team assists\n
            TEAM_TOV (ArrayLike): team turnovers\n
            TEAM_MP (ArrayLike): team minutes played\n
            OPP_DREB (ArrayLike): opponent defensive rebounds\n

        Returns:
            ArrayLike: Scoring Possessions component of Individual Total Possessions.
        """
        scposs_fg_part = self._scposs_fg_part(
            PTS=PTS,
            FGA=FGA,
            FGM=FGM,
            FTM=FTM,
            AST=AST,
            MP=MP,
            TEAM_MP=TEAM_MP,
            TEAM_FGM=TEAM_FGM,
            TEAM_AST=TEAM_AST,
        )
        scposs_ast_part = self._scposs_assist_part(
            PTS=PTS,
            FGA=FGA,
            FTM=FTM,
            AST=AST,
            TEAM_PTS=TEAM_PTS,
            TEAM_FGA=TEAM_FGA,
            TEAM_FTM=TEAM_FTM,
        )
        scposs_ft_part = self._scposs_ft_part(FTM=FTM, FTA=FTA)
        team_scoring_possessions = self._team_scoring_possessions(
            TEAM_FGM=TEAM_FGM, TEAM_FTA=TEAM_FTA, TEAM_FTM=TEAM_FTM
        )
        team_oreb_weight = self._team_oreb_weight(
            TEAM_FGA=TEAM_FGA,
            TEAM_FGM=TEAM_FGM,
            TEAM_FTA=TEAM_FTA,
            TEAM_FTM=TEAM_FTM,
            TEAM_OREB=TEAM_OREB,
            TEAM_TOV=TEAM_TOV,
            OPP_DREB=OPP_DREB,
        )
        team_score_rate = self._team_score_rate(
            TEAM_FGA=TEAM_FGA,
            TEAM_FGM=TEAM_FGM,
            TEAM_FTA=TEAM_FTA,
            TEAM_FTM=TEAM_FTM,
            TEAM_TOV=TEAM_TOV,
        )
        oreb_part = self._oreb_part(
            OREB=OREB,
            TEAM_FGA=TEAM_FGA,
            TEAM_FGM=TEAM_FGM,
            TEAM_FTA=TEAM_FTA,
            TEAM_FTM=TEAM_FTM,
            TEAM_OREB=TEAM_OREB,
            TEAM_TOV=TEAM_TOV,
            OPP_DREB=OPP_DREB,
        )
        return np.add(
            np.multiply(
                np.sum(
                    np.array(
                        [
                            scposs_fg_part,
                            scposs_ast_part,
                            scposs_ft_part,
                        ]
                    ),
                    axis=0,
                ),
                np.multiply(
                    np.subtract(
                        1,
                        np.divide(
                            TEAM_OREB,
                            team_scoring_possessions,
                        ),
                    ),
                    np.multiply(
                        team_oreb_weight,
                        team_score_rate,
                    ),
                ),
            ),
            oreb_part,
        )

    def missed_fg_possessions(
        self,
        FGA: ArrayLike,
        FGM: ArrayLike,
        TEAM_OREB: ArrayLike,
        OPP_DREB: ArrayLike,
        **_
    ) -> ArrayLike:
        """
        Missed Field Goal Possession component of Individual Total Possessions.
            FGxPOSS = (FGA - FGM) * (1 - 1.07 * TEAM_OREB%)

        Estimates number of possessions ending in a missed field goal.
        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            FGA (ArrayLike): field goal attempts\n
            FGM (ArrayLike): field goal makes\n
            TEAM_OREB (ArrayLike): team offensive rebounds\n
            OPP_DREB (ArrayLike): opponent defensive rebounds\n

        Returns:
            ArrayLike: Missed Field Goal Possessions.
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

    def missed_ft_possessions(self, FTA: ArrayLike, FTM: ArrayLike, **_) -> ArrayLike:
        """
        Missed Free Throw Possessions component of Individual Total Possessions.
            FTxPOSS = (1 - (FTM / FTA)**2) * 0.4 * FTA

        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            FTA (ArrayLike): free throw attempts\n
            FTM (ArrayLike): free throw makes\n

        Returns:
            ArrayLike: Missed Free Throw Possessions.
        """
        return np.multiply(
            np.subtract(1, np.power(np.divide(FTM, FTA), 2)),
            np.multiply(self._ft_weight, FTA),
        )

    def total_possessions(
        self,
        PTS: ArrayLike,
        FGA: ArrayLike,
        FGM: ArrayLike,
        FTA: ArrayLike,
        FTM: ArrayLike,
        OREB: ArrayLike,
        AST: ArrayLike,
        TOV: ArrayLike,
        MP: ArrayLike,
        TEAM_PTS: ArrayLike,
        TEAM_FGA: ArrayLike,
        TEAM_FGM: ArrayLike,
        TEAM_FTA: ArrayLike,
        TEAM_FTM: ArrayLike,
        TEAM_OREB: ArrayLike,
        TEAM_AST: ArrayLike,
        TEAM_TOV: ArrayLike,
        TEAM_MP: ArrayLike,
        OPP_DREB: ArrayLike,
        **_
    ) -> ArrayLike:
        """
        Total Possessions used to calculate a player's offensive rating.
            = ScPoss + FGxPOSS + FTxPOSS + TOV

        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            PTS (ArrayLike): points\n
            FGA (ArrayLike): field goal attempts\n
            FGM (ArrayLike): field goal makes\n
            FTA (ArrayLike): free throw attempts\n
            FTM (ArrayLike): free throw makes\n
            OREB (ArrayLike): offensive rebounds\n
            AST (ArrayLike): assists\n
            TOV (ArrayLike): turnovers\n
            MP (ArrayLike): player minutes played\n
            TEAM_PTS (ArrayLike): team points\n
            TEAM_FGA (ArrayLike): team field goal attempts\n
            TEAM_FGM (ArrayLike): team field goal makes\n
            TEAM_FTA (ArrayLike): team free throw attempts\n
            TEAM_FTM (ArrayLike): team free throw makes\n
            TEAM_OREB (ArrayLike): team offensive rebounds\n
            TEAM_AST (ArrayLike): team assists\n
            TEAM_TOV (ArrayLike): team turnovers\n
            TEAM_MP (ArrayLike): team minutes played\n
            OPP_DREB (ArrayLike): opponent defensive rebounds\n

        Returns:
            ArrayLike: Total Possessions.
        """
        scoring_possessions = self._scoring_possessions(
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
        )
        missed_fg_possessions = self.missed_fg_possessions(
            FGA=FGA, FGM=FGM, TEAM_OREB=TEAM_OREB, OPP_DREB=OPP_DREB
        )
        missed_ft_possessions = self.missed_ft_possessions(FTA=FTA, FTM=FTM)
        return np.sum(
            np.array(
                [
                    scoring_possessions,
                    missed_fg_possessions,
                    missed_ft_possessions,
                    TOV,
                ],
            ),
            axis=0,
        )

    def _pprod_fg_part(
        self,
        PTS: ArrayLike,
        FGA: ArrayLike,
        FGM: ArrayLike,
        FG3M: ArrayLike,
        FTM: ArrayLike,
        AST: ArrayLike,
        MP: ArrayLike,
        TEAM_FGM: ArrayLike,
        TEAM_AST: ArrayLike,
        TEAM_MP: ArrayLike,
        **_
    ) -> ArrayLike:
        """
        Field Goal Part of Individual Points Produced.
            PPROD_FG_PART = 2 * (FGM + 0.5 * FG3M) * (1 - 0.5 * ((PTS - FTM) / (2 * FGA)) * _qAST)

        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            PTS (ArrayLike): points\n
            FGA (ArrayLike): field goal attempts\n
            FGM (ArrayLike): field goal makes\n
            FG3M (ArrayLike): three point field goal makes\n
            FTM (ArrayLike): free throw makes\n
            AST (ArrayLike): assists\n
            MP (ArrayLike): player minutes played\n
            TEAM_FGM (ArrayLike): team field goal makes\n
            TEAM_AST (ArrayLike): team assists\n
            TEAM_MP (ArrayLike): team minutes played\n

        Returns:
            ArrayLike: Field Goal Part of Individual Points Produced.
        """
        qAST = self._qAST(
            FGM=FGM,
            AST=AST,
            MP=MP,
            TEAM_FGM=TEAM_FGM,
            TEAM_AST=TEAM_AST,
            TEAM_MP=TEAM_MP,
        )
        return np.multiply(
            np.multiply(2, np.multiply(np.add(FGM, 0.5), FG3M)),
            np.multiply(
                np.subtract(
                    1,
                    np.multiply(
                        0.5, np.divide(np.subtract(PTS, FTM), np.multiply(2, FGA))
                    ),
                ),
                qAST,
            ),
        )

    def _pprod_ast_part(
        self,
        PTS: ArrayLike,
        FGA: ArrayLike,
        FGM: ArrayLike,
        FTM: ArrayLike,
        FG3M: ArrayLike,
        AST: ArrayLike,
        TEAM_PTS: ArrayLike,
        TEAM_FGA: ArrayLike,
        TEAM_FGM: ArrayLike,
        TEAM_FTM: ArrayLike,
        TEAM_FG3M: ArrayLike,
        **_
    ) -> ArrayLike:
        """
        Assist Part of Individual Points Produced.
            PPROD_AST_PART = 2 * ((TEAM_FGM - FGM + 0.5 * (TEAM_FG3M - FG3M)) / (TEAM_FGM - FGM)) * 0.5 * ((TEAM_PTS - TEAM_FTM - (PTS - FTM)) / (2 * (TEAM_FGA - FGA))) * AST

        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            PTS (ArrayLike): points\n
            FGA (ArrayLike): field goal attempts\n
            FGM (ArrayLike): field goal makes\n
            FTM (ArrayLike): free throw makes\n
            FG3M (ArrayLike): three point field goal makes\n
            AST (ArrayLike): assists\n
            TEAM_PTS (ArrayLike): team points\n
            TEAM_FGA (ArrayLike): team field goal attempts\n
            TEAM_FGM (ArrayLike): team field goal makes\n
            TEAM_FTM (ArrayLike): team free throw makes\n
            TEAM_FG3M (ArrayLike): team three point field goal makes\n

        Returns:
            ArrayLike: Assist Part of Individual Points Produced.
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
        OREB: ArrayLike,
        TEAM_PTS: ArrayLike,
        TEAM_FGA: ArrayLike,
        TEAM_FGM: ArrayLike,
        TEAM_FTA: ArrayLike,
        TEAM_FTM: ArrayLike,
        TEAM_OREB: ArrayLike,
        TEAM_TOV: ArrayLike,
        OPP_DREB: ArrayLike,
        **_
    ) -> ArrayLike:
        """
        Offensive Rebound Part of Individual Points Produced.
            PPROD_OREB_PART = OREB * TEAM_OREB_WEIGHT * TEAM_SCORE_RATE * (TEAM_PTS / (TEAM_FGM + (1 - 1 - (TEAM_FTM / TEAM_FTA))**2) * 0.44 * TEAM_FTA)))))

        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            OREB (ArrayLike): offensive rebounds\n
            TEAM_PTS (ArrayLike): team points\n
            TEAM_FGA (ArrayLike): team field goal attempts\n
            TEAM_FGM (ArrayLike): team field goal makes\n
            TEAM_FTA (ArrayLike): team free throw attempts\n
            TEAM_FTM (ArrayLike): team free throw makes\n
            TEAM_OREB (ArrayLike): team offensive rebounds\n
            TEAM_TOV (ArrayLike): team turnovers\n
            OPP_DREB (ArrayLike): opponent defensive rebounds\n

        Returns:
            ArrayLike: Offensive Rebound Part of Individual Points Produced.
        """
        oreb_part = self._oreb_part(
            OREB=OREB,
            TEAM_FGA=TEAM_FGA,
            TEAM_FGM=TEAM_FGM,
            TEAM_FTA=TEAM_FTA,
            TEAM_FTM=TEAM_FTM,
            TEAM_OREB=TEAM_OREB,
            TEAM_TOV=TEAM_TOV,
            OPP_DREB=OPP_DREB,
        )
        return np.multiply(
            oreb_part,
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

    def points_produced(
        self,
        PTS: ArrayLike,
        FGA: ArrayLike,
        FGM: ArrayLike,
        FG3M: ArrayLike,
        FTM: ArrayLike,
        OREB: ArrayLike,
        AST: ArrayLike,
        MP: ArrayLike,
        TEAM_PTS: ArrayLike,
        TEAM_FGA: ArrayLike,
        TEAM_FGM: ArrayLike,
        TEAM_FG3M: ArrayLike,
        TEAM_FTA: ArrayLike,
        TEAM_FTM: ArrayLike,
        TEAM_OREB: ArrayLike,
        TEAM_AST: ArrayLike,
        TEAM_TOV: ArrayLike,
        TEAM_MP: ArrayLike,
        OPP_DREB: ArrayLike,
        **_
    ) -> ArrayLike:
        """
        Individual Points Produced (PProd) used to calculate a player's offensive rating.
            PProd = (PProd_FG_Part + PProd_AST_Part + FTM) * (1 - TEAM_OREB / TEAM_SCORING_POSS) * TEAM_OREB_WEIGHT * TEAM_SCORE_RATE) + PProd_OREB_Part

        Estimates points produced by a player.
        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            PTS (ArrayLike): points\n
            FGA (ArrayLike): field goal attempts\n
            FGM (ArrayLike): field goal makes\n
            FG3M (ArrayLike): three point field goal makes\n
            FTM (ArrayLike): free throw makes\n
            OREB (ArrayLike): offensive rebounds\n
            AST (ArrayLike): assists\n
            MP (ArrayLike): player minutes played\n
            TEAM_PTS (ArrayLike): team points\n
            TEAM_FGA (ArrayLike): team field goal attempts\n
            TEAM_FGM (ArrayLike): team field goal makes\n
            TEAM_FG3M (ArrayLike): team three point field goal makes\n
            TEAM_FTA (ArrayLike): team free throw attempts\n
            TEAM_FTM (ArrayLike): team free throw makes\n
            TEAM_OREB (ArrayLike): team offensive rebounds\n
            TEAM_AST (ArrayLike): team assists\n
            TEAM_TOV (ArrayLike): team turnovers\n
            TEAM_MP (ArrayLike): team minutes played\n
            OPP_DREB (ArrayLike): opponent defensive rebounds\n

        Returns:
            ArrayLike: Individual Points Produced.
        """
        fg_part = self._pprod_fg_part(
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
        )
        ast_part = self._pprod_ast_part(
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
        )
        team_scoring_possessions = self._team_scoring_possessions(
            TEAM_FGM=TEAM_FGM,
            TEAM_FTA=TEAM_FTA,
            TEAM_FTM=TEAM_FTM,
        )
        team_oreb_weight = self._team_oreb_weight(
            TEAM_FGA=TEAM_FGA,
            TEAM_FGM=TEAM_FGM,
            TEAM_FTA=TEAM_FTA,
            TEAM_FTM=TEAM_FTM,
            TEAM_OREB=TEAM_OREB,
            TEAM_TOV=TEAM_TOV,
            OPP_DREB=OPP_DREB,
        )
        team_score_rate = self._team_score_rate(
            TEAM_FGA=TEAM_FGA,
            TEAM_FGM=TEAM_FGM,
            TEAM_FTA=TEAM_FTA,
            TEAM_FTM=TEAM_FTM,
            TEAM_TOV=TEAM_TOV,
        )
        pprod_oreb_part = self._pprod_oreb_part(
            OREB=OREB,
            TEAM_PTS=TEAM_PTS,
            TEAM_FGA=TEAM_FGA,
            TEAM_FGM=TEAM_FGM,
            TEAM_FTA=TEAM_FTA,
            TEAM_FTM=TEAM_FTM,
            TEAM_OREB=TEAM_OREB,
            TEAM_TOV=TEAM_TOV,
            OPP_DREB=OPP_DREB,
        )
        return np.add(
            np.multiply(
                np.sum(
                    np.array(
                        [
                            fg_part,
                            ast_part,
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
                                team_scoring_possessions,
                            ),
                        ),
                        team_oreb_weight,
                    ),
                    team_score_rate,
                ),
            ),
            pprod_oreb_part,
        )

    def offensive_rating(
        self,
        PTS: ArrayLike,
        FGA: ArrayLike,
        FGM: ArrayLike,
        FG3M: ArrayLike,
        FTA: ArrayLike,
        FTM: ArrayLike,
        OREB: ArrayLike,
        AST: ArrayLike,
        TOV: ArrayLike,
        MP: ArrayLike,
        TEAM_PTS: ArrayLike,
        TEAM_FGA: ArrayLike,
        TEAM_FGM: ArrayLike,
        TEAM_FG3M: ArrayLike,
        TEAM_FTA: ArrayLike,
        TEAM_FTM: ArrayLike,
        TEAM_OREB: ArrayLike,
        TEAM_AST: ArrayLike,
        TEAM_TOV: ArrayLike,
        TEAM_MP: ArrayLike,
        OPP_DREB: ArrayLike,
        **_
    ) -> ArrayLike:
        """
        Player Offensive Rating (OffRtg)
            OffRtg = 100 * (Individual Points Produced / Total Possessions)

        Estimates points produced by a player per 100 possessions played.
        Source(s):
            - https://www.basketball-reference.com/about/ratings.html
            - https://hackastat.eu/en/learn-a-stat-individual-offensive-rating/\n
        Created by Dean Oliver.

        Args:
            PTS (ArrayLike): points\n
            FGA (ArrayLike): field goal attempts\n
            FGM (ArrayLike): field goal makes\n
            FG3M (ArrayLike): three point field goal makes\n
            FTA (ArrayLike): free throw attempts\n
            FTM (ArrayLike): free throw makes\n
            OREB (ArrayLike): offensive rebounds\n
            AST (ArrayLike): assists\n
            TOV (ArrayLike): turnovers\n
            MP (ArrayLike): player minutes played\n
            TEAM_PTS (ArrayLike): team points\n
            TEAM_FGA (ArrayLike): team field goal attempts\n
            TEAM_FGM (ArrayLike): team field goal makes\n
            TEAM_FG3M (ArrayLike): team three point field goal makes\n
            TEAM_FTA (ArrayLike): team free throw attempts\n
            TEAM_FTM (ArrayLike): team free throw makes\n
            TEAM_OREB (ArrayLike): team offensive rebounds\n
            TEAM_AST (ArrayLike): team assists\n
            TEAM_TOV (ArrayLike): team turnovers\n
            TEAM_MP (ArrayLike): team minutes played\n
            OPP_DREB (ArrayLike): opponent defensive rebounds\n

        Returns:
            ArrayLike: player offensive rating.
        """
        individual_points_produced = self.points_produced(
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
        )
        total_possessions = self.total_possessions(
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
        )
        return np.multiply(
            100,
            np.divide(
                individual_points_produced,
                total_possessions,
            ),
        )

    def floor_pct(
        self,
        PTS: ArrayLike,
        FGA: ArrayLike,
        FGM: ArrayLike,
        FTA: ArrayLike,
        FTM: ArrayLike,
        OREB: ArrayLike,
        AST: ArrayLike,
        TOV: ArrayLike,
        MP: ArrayLike,
        TEAM_PTS: ArrayLike,
        TEAM_FGA: ArrayLike,
        TEAM_FGM: ArrayLike,
        TEAM_FTA: ArrayLike,
        TEAM_FTM: ArrayLike,
        TEAM_OREB: ArrayLike,
        TEAM_AST: ArrayLike,
        TEAM_TOV: ArrayLike,
        TEAM_MP: ArrayLike,
        OPP_DREB: ArrayLike,
        **_
    ) -> ArrayLike:
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
            PTS (ArrayLike): points\n
            FGA (ArrayLike): field goal attempts\n
            FGM (ArrayLike): field goal makes\n
            FTA (ArrayLike): free throw attempts\n
            FTM (ArrayLike): free throw makes\n
            OREB (ArrayLike): offensive rebounds\n
            AST (ArrayLike): assists\n
            TOV (ArrayLike): turnovers\n
            MP (ArrayLike): player minutes played\n
            TEAM_PTS (ArrayLike): team points\n
            TEAM_FGA (ArrayLike): team field goal attempts\n
            TEAM_FGM (ArrayLike): team field goal makes\n
            TEAM_FTA (ArrayLike): team free throw attempts\n
            TEAM_FTM (ArrayLike): team free throw makes\n
            TEAM_OREB (ArrayLike): team offensive rebounds\n
            TEAM_AST (ArrayLike): team assists\n
            TEAM_TOV (ArrayLike): team turnovers\n
            TEAM_MP (ArrayLike): team minutes played\n
            OPP_DREB (ArrayLike): opponent defensive rebounds\n

        Returns:
            ArrayLike: Floor Percentage.
        """
        player_scoring_possessions = self._scoring_possessions(
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
        )
        total_possessions = self.total_possessions(
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
        )
        return np.divide(
            player_scoring_possessions,
            total_possessions,
        )

    def _stops1(self) -> ArrayLike:
        return []

    def _stops2(self) -> ArrayLike:
        return []

    def stops(self) -> ArrayLike:
        return []

    def defensive_rating(self) -> ArrayLike:
        return []

    def win_shares(self) -> ArrayLike:
        return []

    def off_win_shares(self) -> ArrayLike:
        return []

    def def_win_shares(self) -> ArrayLike:
        return []

    def box_plus_minus(self) -> ArrayLike:
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

    def value_over_replacement(self) -> ArrayLike:
        return []

    def wins_above_replacement(self) -> ArrayLike:
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
        PTS: ArrayLike,
        FGM: ArrayLike,
        FGA: ArrayLike,
        FTM: ArrayLike,
        FTA: ArrayLike,
        OREB: ArrayLike,
        DREB: ArrayLike,
        STL: ArrayLike,
        AST: ArrayLike,
        BLK: ArrayLike,
        PF: ArrayLike,
        TOV: ArrayLike,
    ) -> ArrayLike:
        """
        Game Score (GAME_SCORE)
            = PTS + 0.4*FGM - 0.7*FGA - 0.4*(FTA - FTM) + 0.7*OREB + 0.3*DREB + STL + 0.7*AST + 0.7*BLK - 0.4*PF - TOV

        Estimates a player's performance in a single game. Created by John Hollinger.
        Scale is similar to points scored (40 outstanding, 10 average).
        Source: https://www.basketball-reference.com/about/glossary.html

        Args:
            PTS (ArrayLike): points
            FGM (ArrayLike): field goal makes
            FGA (ArrayLike): field goal attempts
            FTM (ArrayLike): free throw makes
            FTA (ArrayLike): free throw attempts
            OREB (ArrayLike): offensive rebounds
            DREB (ArrayLike): defensive rebounds
            STL (ArrayLike): steals
            AST (ArrayLike): assists
            BLK (ArrayLike): blocks
            PF (ArrayLike): personal fouls
            TOV (ArrayLike): turnovers

        Returns:
            ArrayLike: game score
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
