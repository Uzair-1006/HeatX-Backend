# backend/services/dataset_service.py
import pandas as pd

def parse_dataset(filepath: str) -> pd.DataFrame:
    """Parse CSV or Excel file"""
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith(('.xlsx', '.xls')):
        return pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format")