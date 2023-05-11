# %%
# Imports
import os
import time
import pickle
import threading
from typing import Union, List, Dict, Tuple, Callable
import requests
import pandas as pd


# %%
def save(func: Callable):
    """Save wrapper to create directories if they don't exist.

    Args:
        func (Callable): _description_
    """

    def save_wrapper(*args, **kwargs) -> bool:
        print("Saving data...")
        try:
            return func(*args, **kwargs)
        except Exception as e:
            try:
                print(e)
                print("Creating folder...")
                os.makedirs(kwargs["folder"], exist_ok=True)
                print("Folder created.")
                return func(*args, **kwargs)
            except Exception as e2:
                print(e2)
                print("Folder still not found.")
                print("Removing created folder...")
                os.rmdir(kwargs["folder"])
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
        print(f"{name} saved to {os.path.abspath(folder)}")
        return True
    if name.split(".")[-1] != "pickle":
        name += ".pickle"
    pth = os.path.join(folder, name)
    with open(pth, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"{name} saved to {os.path.abspath(folder)}")
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
        print(e)
        return None


# Retry Wrapper
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