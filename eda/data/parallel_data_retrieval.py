import multiprocessing as mp
from data.teams.team_game_data import TeamGameData

if __name__ == "__main__":
    team_game_data = TeamGameData()
    team_ids = team_game_data.all_team_ids
    num_cpu = mp.cpu_count() - 1
    pool = mp.Pool(processes=num_cpu)
    pool.starmap(func=team_game_data.get_team_games, iterable=team_ids)
    pool.close()
