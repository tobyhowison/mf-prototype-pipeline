import pandas as pd


class DataLoader:
    """
    General-purpose class for loading heart disease data, designed to allow support multiple file formats.
    """
    @staticmethod
    def from_file(file_path: str) -> pd.DataFrame:
        """
          Load heard data and infer data type, only .csv supported now
            Parameters:
                       file_path (str): Path to data
               Returns:
                       df (pd.DataFrame): data loaded into a DataFrame
        """

        extension = file_path.split('.')[-1].lower()
        try:
            if extension == 'csv':
                return pd.read_csv(file_path)
            else:
                raise ValueError(f"Unsupported file format: .{extension}")
        except ValueError as e:
            raise ValueError(f"Error loading file {file_path}: {e}")
