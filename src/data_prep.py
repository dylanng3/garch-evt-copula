"python src/data_prep.py --raw_folder data/raw --processed_folder data/processed"

import os
import pandas as pd
import numpy as np
from typing import List, Optional


def load_split_adjusted_data(folder_path: str, dropna_type: str = 'any', min_non_na: Optional[int] = None) -> pd.DataFrame:
    """
    Load and merge ALL split-adjusted CSV files from Download Data pattern.
    Merge all years for each ticker to get complete time series.
    """
    print(f"Loading split-adjusted data from {folder_path}...")
    
    # Get all Download Data files
    csv_files = [f for f in os.listdir(folder_path) 
                 if f.endswith('.csv') and f.startswith('Download Data - STOCK_VN_XSTC_')]
    
    if not csv_files:
        raise ValueError(f"No 'Download Data - STOCK_VN_XSTC_*' files found in {folder_path}")
    
    print(f"Found {len(csv_files)} files to process")
    
    # Group files by ticker
    ticker_files = {}
    for file in csv_files:
        # Extract ticker: Download Data - STOCK_VN_XSTC_FPT (1).csv → FPT
        parts = file.replace('Download Data - STOCK_VN_XSTC_', '').replace('.csv', '')
        ticker = parts.split(' ')[0].split('(')[0].split('_')[0].split('-')[0].strip()
        
        if ticker not in ticker_files:
            ticker_files[ticker] = []
        ticker_files[ticker].append(file)
    
    print(f"Grouped into {len(ticker_files)} tickers: {list(ticker_files.keys())}")
    
    # Process each ticker separately
    ticker_dfs = {}
    for ticker, files in ticker_files.items():
        print(f"\nProcessing {ticker} ({len(files)} files)...")
        yearly_data = []
        for file in sorted(files):  # Sort files for consistent processing
            file_path = os.path.join(folder_path, file)
            try:
                df = pd.read_csv(file_path)
                # Find date and price columns
                date_col = next((c for c in df.columns if c.strip().lower() in ['date', 'ngay']), df.columns[0])
                price_col = next((c for c in df.columns if c.lower() in ['close', 'price']), 
                                next((c for c in df.columns if 'price' in c.lower()), df.columns[1]))
                # Clean and convert
                df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
                df[price_col] = (
                    df[price_col]
                    .astype(str)
                    .str.replace(',', '', regex=False)
                    .str.replace('"', '', regex=False)
                )
                df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
                # Create clean dataframe
                df_clean = df[[date_col, price_col]].dropna().set_index(date_col).sort_index()
                df_clean.columns = [f"{ticker}_DATA"]
                if len(df_clean) > 0:
                    yearly_data.append(df_clean)
                    print(f"  {file}: {len(df_clean)} records, {df_clean.index.min()} to {df_clean.index.max()}")
            except Exception as e:
                print(f"  Error processing {file}: {e}")
                continue
        if yearly_data:
            # Merge all years for this ticker
            ticker_merged = pd.concat(yearly_data, axis=0).sort_index()
            # Remove duplicates (keep last = most recent data)
            if ticker_merged.index.duplicated().any():
                dup_count = ticker_merged.index.duplicated().sum()
                ticker_merged = ticker_merged[~ticker_merged.index.duplicated(keep='last')]
                print(f"  {ticker}: Removed {dup_count} duplicate dates")
            ticker_dfs[ticker] = ticker_merged
            print(f"  {ticker}: Final shape {ticker_merged.shape}")
        else:
            print(f"  {ticker}: No valid data found")
    
    if not ticker_dfs:
        raise ValueError("No valid ticker data loaded")
    
    # Combine all tickers
    print(f"\nCombining {len(ticker_dfs)} tickers...")
    final_df = pd.concat(list(ticker_dfs.values()), axis=1).sort_index()
    
    # More flexible dropna - keep rows with at least half the tickers
    min_tickers = max(1, len(ticker_dfs) // 2)
    original_len = len(final_df)
    final_df = final_df.dropna(thresh=min_tickers)
    
    print(f"Applied flexible dropna (thresh={min_tickers}): {original_len} -> {len(final_df)} rows")
    print(f"Final shape: {final_df.shape}")
    
    if len(final_df) > 0:
        print(f"Date range: {final_df.index.min()} to {final_df.index.max()}")
        print(f"Columns: {list(final_df.columns)}")
    
    return final_df
"""
Module: data_prep.py
Description: Utility functions for downloading, cleaning, and saving data for the GARCH-EVT-Copula project.
"""

def check_negative_or_zero(price_df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with any closing price less than or equal to 0."""
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

def main_split_adjusted_prep(raw_folder: str, processed_folder: str, dropna_type: str = 'any', min_non_na: Optional[int] = None):
    """
    Run data preparation pipeline for split-adjusted data (yearly CSV files).
    """
    print("=== SPLIT-ADJUSTED DATA PREPARATION ===")
    
    # Create processed folder if it does not exist
    os.makedirs(processed_folder, exist_ok=True)
    
    # Load split-adjusted data
    data = load_split_adjusted_data(raw_folder, dropna_type=dropna_type, min_non_na=min_non_na)
    
    # Remove rows with closing price less than or equal to 0
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
    
    # Save the original date index 
    logret_index_path = os.path.join(processed_folder, 'log_returns_index.csv')
    log_returns.index.to_series().to_csv(logret_index_path, header=True)
    print(f"Saved log returns index to {logret_index_path}")
    
    # Quick check for remaining extreme outliers
    outliers = check_outliers(log_returns, threshold=0.15)
    if not outliers.empty:
        print(f"\nStill found {len(outliers)} potential outliers > 15%:")
        print(outliers.abs().max().sort_values(ascending=False))
    
    print("Split-adjusted data preparation completed.")
    
    # Add CLI entry point
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run data preparation pipeline.")
    parser.add_argument('--raw_folder', type=str, required=True, help='Path to raw data folder')
    parser.add_argument('--processed_folder', type=str, required=True, help='Path to save processed data')
    parser.add_argument('--dropna_type', type=str, default='any', choices=['any', 'all'], help="Row filter: 'any' (intersection) or 'all' (union)")
    parser.add_argument('--min_non_na', type=int, default=None, help='Minimum non-NA values per row (overrides dropna_type if set)')
    args = parser.parse_args()
    
    main_split_adjusted_prep(
        raw_folder=args.raw_folder,
        processed_folder=args.processed_folder,
        dropna_type=args.dropna_type,
        min_non_na=args.min_non_na
    )