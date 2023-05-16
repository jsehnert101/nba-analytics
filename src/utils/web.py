"""
Methods in this file are used to open URLs in a web browser to assist with data inputation.
"""
from typing import List
from urllib.request import urlopen
from urllib.error import HTTPError
import webbrowser as web
import validators


def open_url(urls: List[str]) -> None:
    """
    Open URL in web browser. If first URL doesn't work, try second, so on and so forth.
    """
    errors = []
    for url in urls:
        try:
            if validators.url(url):  # type: ignore
                html = urlopen(url)
                web.open(url)
                html.close()
                return
            else:
                errors.append("Validation Error")
        except HTTPError as e:
            errors.append(e)
    print(f"No valid URLs: {errors}")
