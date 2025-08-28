import os
import pandas as pd
import numpy as np
from typing import List, Optional


def load_multiple_csv_to_df(folder_path: str, dropna_type: str = 'any', min_non_na: Optional[int] = None) -> pd.DataFrame:
    """
    Load and merge all single-ticker CSV files in a folder into one DataFrame.
    dropna_type: 'any' (default, intersection), 'all' (union), or set min_non_na for at least n non-NA values per row.
    """
    dfs = []
    for file in os.listdir(folder_path):
        if not file.endswith('.csv'):
            continue
        ticker = os.path.splitext(file)[0].upper()
        df = pd.read_csv(os.path.join(folder_path, file))
        date_col = next((c for c in df.columns if c.strip().lower() in ['date', 'ngay', 'time', 'datetime']), df.columns[0])
        price_col = next((c for c in df.columns if c.lower() in ['close', ticker, ticker+'.VN', ticker+'.HO', ticker+'.HM']), df.columns[1])
        df = df[[date_col, price_col]].rename(columns={date_col: 'Date', price_col: ticker})
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        # Remove commas and quotes, then convert to float
        df[ticker] = (
            df[ticker]
            .astype(str)
            .str.replace(',', '', regex=False)
            .str.replace('"', '', regex=False)
        )
        df[ticker] = pd.to_numeric(df[ticker], errors='coerce')
        df = df.set_index('Date').sort_index()
        dfs.append(df)
    if not dfs:
        raise ValueError("No data loaded from CSV files.")
    merged = pd.concat(dfs, axis=1).sort_index()
    merged = merged.apply(pd.to_numeric, errors='coerce')
    if min_non_na is not None:
        merged = merged.dropna(thresh=min_non_na)
    else:
        # Only allow 'any' or 'all' for dropna(how=...)
        how = 'any' if dropna_type != 'all' else 'all'
        merged = merged.dropna(how=how)
    return merged
"""
Module: data_prep.py
Description: Utility functions for downloading, cleaning, and saving data for the GARCH-EVT-Copula project.
"""

def check_negative_or_zero(price_df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with any closing price <= 0."""
    mask = (price_df <= 0).any(axis=1)
    if mask.sum() > 0:
        print(f"{mask.sum()} days have closing price <= 0 and will be removed.")
    return price_df[~mask]
def check_outliers(log_returns: pd.DataFrame, threshold: float = 0.2):
    """Check for abnormal (outlier) log return values."""
    outlier_mask = (log_returns.abs() > threshold)
    outlier_days = outlier_mask.sum().sum()
    if outlier_days > 0:
        print(f"There are {outlier_days} log return values with absolute value > {threshold} (possible data error or special event).")
    return log_returns[outlier_mask.any(axis=1)]

def report_missing_values(df: pd.DataFrame):
    """Print the number of missing values for each ticker."""
    missing = df.isnull().sum()
    print("Missing values per ticker:")
    print(missing)
    print(f"Total number of rows with missing values: {df.isnull().any(axis=1).sum()}")

def compute_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns for a closing price DataFrame."""
    log_returns = np.log(price_df / price_df.shift(1))
    log_returns = pd.DataFrame(log_returns, index=price_df.index, columns=price_df.columns)
    return log_returns.dropna()

def save_to_csv(df: pd.DataFrame, path: str):
    """Save DataFrame to CSV file."""
    df.to_csv(path)

def main_data_prep(raw_folder: str, processed_folder: str, dropna_type: str = 'any', min_non_na: Optional[int] = None):
    """
    Run the full data preparation pipeline: load, clean, check, and save processed data.
    """
    print("Loading and merging raw data...")
    data = load_multiple_csv_to_df(raw_folder, dropna_type=dropna_type, min_non_na=min_non_na)
    print(f"Data shape after merge: {data.shape}")
    # Remove rows with closing price <= 0
    data = check_negative_or_zero(data)
    print(f"Data shape after removing <=0: {data.shape}")
    # Check missing values
    report_missing_values(data)
    # Save cleaned price data
    price_path = os.path.join(processed_folder, 'price_cleaned.csv')
    save_to_csv(data, price_path)
    print(f"Saved cleaned price data to {price_path}")
    # Compute log returns
    log_returns = compute_log_returns(data)
    # Round to 6 decimals for readability
    log_returns = log_returns.round(6)
    logret_path = os.path.join(processed_folder, 'log_returns.csv')
    save_to_csv(log_returns, logret_path)
    print(f"Saved log returns to {logret_path}")
    print("Data preparation pipeline completed.")
    
    # Add CLI entry point
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run data preparation pipeline.")
    parser.add_argument('--raw_folder', type=str, required=True, help='Path to raw data folder')
    parser.add_argument('--processed_folder', type=str, required=True, help='Path to save processed data')
    parser.add_argument('--dropna_type', type=str, default='any', choices=['any', 'all'], help="Row filter: 'any' (intersection) or 'all' (union)")
    parser.add_argument('--min_non_na', type=int, default=None, help='Minimum non-NA values per row (overrides dropna_type if set)')
    args = parser.parse_args()
    main_data_prep(
        raw_folder=args.raw_folder,
        processed_folder=args.processed_folder,
        dropna_type=args.dropna_type,
        min_non_na=args.min_non_na
    )