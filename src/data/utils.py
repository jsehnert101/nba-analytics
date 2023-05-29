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
def save(func: Callable):
    """Save wrapper to create directories if they don't exist.

    Args:
        func (Callable): _description_
    """

    def save_wrapper(*args, **kwargs) -> bool:
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


@save
def save_data(data, folder: str, name: str) -> bool:
    """
    Save data to desired location.
    pandas DataFrames/Series are saves as parquet files.

    Args:
        data (_type_): data to be saved.
            If pandas DataFrame/Series, columns must be strings.
            If other, must be pickleable.
        folder (str): folder to store data. Cannot start with '/'.
        name (str): name of file to store data.

    Returns:
        bool: True/False depending on success.
    """
    if isinstance(data, pd.DataFrame):
        if name.split(".")[-1] != "parquet":
            name += ".parquet"
        pth = os.path.join(folder, name)
        data.to_parquet(pth)
        return True
    elif name.split(".")[-1] != "pickle":
        name += ".pickle"
    pth = os.path.join(folder, name)
    with open(pth, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return True


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
        # print(e)
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


cache = TTLCache(maxsize=128, ttl=300)  # TODO: improve cache
