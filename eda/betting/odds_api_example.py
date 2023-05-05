# %%
import numpy as np
import pandas as pd
import requests
import json
from utils.env import *


# %%
API_KEY = get_api_key()
SPORT = "americanfootball_nfl" # "upcoming"
REGIONS = "us"
MARKETS = "totals" #"h2h, spreads, totals" # spreads
ODDS_FORMAT = "decimal"
DATE_FORMAT = "iso"

sports_response = requests.get('https://api.the-odds-api.com/v4/sports', params={
    'api_key': API_KEY
})

if sports_response.status_code != 200:
    print(f'Failed to get sports: status_code {sports_response.status_code}, response body {sports_response.text}')
else:
    print('List of in season sports:', sports_response.json())
    
# %%
odds_response = requests.get(f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds/', params={
    'api_key': API_KEY,
    'regions': REGIONS,
    'markets': MARKETS,
    'oddsFormat': ODDS_FORMAT,
    'dateFormat': DATE_FORMAT,
})

if odds_response.status_code != 200:
    print(f'Failed to get odds: status_code {odds_response.status_code}, response body {odds_response.text}')
else:
    odds_json = odds_response.json()
    print('Number of events:', len(odds_json))
    print(odds_json)

    # Check the usage quota
    print('Remaining requests', odds_response.headers['x-requests-remaining'])
    print('Used requests', odds_response.headers['x-requests-used'])
    
# %%

odds_response_json = odds_response.json()
pd.DataFrame(odds_response_json)


# %%
sports_response = requests.get('https://api.the-odds-api.com/v3/sports', params={
    'api_key': API_KEY
})
sports_json = json.loads(sports_response.text)
sports_json

# %%
pd.DataFrame(sports_json["data"])
# %%
