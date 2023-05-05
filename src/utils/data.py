# %%
# Imports
import requests
import time
import pickle
from typing import Union, List, Dict, Tuple


# %%
def save_data(data: Union[Dict, List, Tuple], pth: str) -> bool:
    """Save data using pickle.

    Args:
        pth (str): path to store dictionary. Must end in ".pickle".
        d (dict): dictionary to be stored.

    Returns:
        bool: True/False depending on success.
    """
    try:
        with open(pth, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return True
    except Exception as e:
        print(e)
        print("File not saved successfully. Check path.")
        return False


def load_data(pth: str) -> Union[Dict, List, Tuple, None]:
    """Load data using pickle.

    Args:
        pth (str): path to store dictionary. Must end in ".pickle".

    Returns:
        dict: desired dictionary
    """
    try:
        with open(pth, "rb") as handle:
            data = pickle.load(handle)
        return data
    except FileNotFoundError as e:
        print(e)
        return None


# Retry Wrapper
def retry(func, retries=3):
    def retry_wrapper(*args, **kwargs):
        attempts = 0
        while attempts < retries:
            try:
                return func(*args, **kwargs)
            except requests.exceptions.RequestException as e:
                print(e)
                time.sleep(30)
                attempts += 1

    return retry_wrapper
