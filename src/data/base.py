# Imports
from typing import Dict
import os
import sys
from config.directory import ROOT_DIR


class Data(object):
    def __init__(self):
        self.external_raw_folder = "data/external/raw/"
        self.external_interim_folder = "data/external/interim/"
        self.external_processed_folder = "data/external/processed/"
        self.internal_raw_folder = "data/internal/raw/"
        self.internal_interim_folder = "data/internal/interim/"
        self.internal_processed_folder = "data/internal/processed/"
        self.inputation_folder = "data/internal/inputation/"
        self.folder_map = {
            "external": {
                "raw": self.external_raw_folder,
                "interim": self.external_interim_folder,
                "processed": self.external_processed_folder,
            },
            "internal": {
                "raw": self.internal_raw_folder,
                "interim": self.internal_interim_folder,
                "processed": self.internal_processed_folder,
                "inputation": self.inputation_folder,
            },
        }

    @property
    def external_raw_folder(self) -> str:
        """Get external raw folder path."""
        return self._external_raw_folder

    @external_raw_folder.setter
    def external_raw_folder(self, folder: str) -> None:
        """Set external raw folder path."""
        self._external_raw_folder = os.path.realpath(os.path.join(ROOT_DIR, folder))

    @property
    def external_interim_folder(self) -> str:
        """Get external interim folder path."""
        return self._external_interim_folder

    @external_interim_folder.setter
    def external_interim_folder(self, folder: str) -> None:
        """Set external interim folder path."""
        self._external_interim_folder = os.path.realpath(os.path.join(ROOT_DIR, folder))

    @property
    def external_processed_folder(self) -> str:
        """Get external processed folder path."""
        return self._external_processed_folder

    @external_processed_folder.setter
    def external_processed_folder(self, folder: str) -> None:
        """Set external processed folder path."""
        self._external_processed_folder = os.path.realpath(
            os.path.join(ROOT_DIR, folder)
        )

    @property
    def internal_raw_folder(self) -> str:
        """Get internal raw folder path."""
        return self._internal_raw_folder

    @internal_raw_folder.setter
    def internal_raw_folder(self, folder: str) -> None:
        """Set internal raw folder path."""
        self._internal_raw_folder = os.path.realpath(os.path.join(ROOT_DIR, folder))

    @property
    def internal_interim_folder(self) -> str:
        """Get internal interim folder path."""
        return self._internal_interim_folder

    @internal_interim_folder.setter
    def internal_interim_folder(self, folder: str) -> None:
        """Set internal interim folder path."""
        self._internal_interim_folder = os.path.realpath(os.path.join(ROOT_DIR, folder))

    @property
    def internal_processed_folder(self) -> str:
        """Get internal processed folder path."""
        return self._internal_processed_folder

    @internal_processed_folder.setter
    def internal_processed_folder(self, folder: str) -> None:
        """Set internal processed folder path."""
        self._internal_processed_folder = os.path.realpath(
            os.path.join(ROOT_DIR, folder)
        )

    @property
    def inputation_folder(self) -> str:
        """Get inputation folder path."""
        return self._inputation_folder

    @inputation_folder.setter
    def inputation_folder(self, folder: str) -> None:
        """Set inputation folder path."""
        self._inputation_folder = os.path.realpath(os.path.join(ROOT_DIR, folder))

    @property
    def folder_map(self) -> Dict[str, Dict[str, str]]:
        """Get folder map."""
        return self._folder_map

    @folder_map.setter
    def folder_map(self, folder_map: Dict[str, Dict[str, str]]) -> None:
        """Set folder map."""
        self._folder_map = folder_map


if __name__ == "__main__":
    pass
