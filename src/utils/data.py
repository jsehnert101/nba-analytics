# %%
# Imports
import os
import time
import pickle
from typing import Union, List, Dict, Tuple, Callable
from cachetools import TTLCache
import requests
import pandas as pd


# %%
# Helper functions
def manage_path(func: Callable):
    """Save wrapper to create directories if they don't exist.

    Args:
        func (Callable): _description_
    """

    def save_wrapper(*args, **kwargs) -> bool:
        # print("Saving data...")
        try:
            return func(*args, **kwargs)
        except Exception:
            try:
                os.makedirs(kwargs["folder"], exist_ok=True)
                return func(*args, **kwargs)
            except Exception as e:
                print(e)
                return False

    return save_wrapper


def load_data(pth: str) -> Union[Dict, List, Tuple, None]:
    """Load data using pickle.

    Args:
        pth (str): path to store data. Must end in ".pickle".

    Returns:
        Union[Dict, List, Tuple, None]: requested data
    """
    try:
        with open(pth, "rb") as handle:
            data = pickle.load(handle)
        return data
    except FileNotFoundError as e:
        print(e)
        return None


def retry(func: Callable, retries=3):
    """Retry wrapper for requests to bypass throttling.

    Args:
        func (Callable): method to be retried.
        retries (int, optional): max. number retries. Defaults to 3.
    """

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


def get_pth_to_desktop() -> str:
    """Get path to desktop for any OS."""
    return os.path.expanduser("~/Desktop")


def join_str(s: str) -> str:
    """Elminate spaces in string."""
    return "".join(s.split())


cache = TTLCache(maxsize=128, ttl=300)  # TODO: improve cache
