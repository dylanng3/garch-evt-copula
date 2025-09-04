"""
Module: data_prep.py
Description: Utility functions for loading, cleaning, and saving data for the GARCH-EVT-Copula project.
"""

import os
import pandas as pd
import numpy as np
from typing import Optional


def smart_date_convert(date_series):
    """
    Smart date conversion with multiple format attempts.
    """
    formats_to_try = [
        '%m/%d/%Y',    # MM/dd/yyyy (US format)
        '%d/%m/%Y',    # dd/MM/yyyy (EU format)  
        '%Y-%m-%d',    # yyyy-MM-dd (ISO format)
        '%d-%m-%Y',    # dd-MM-yyyy
        '%m-%d-%Y'     # MM-dd-yyyy
    ]
    
    result = pd.Series(pd.NaT, index=date_series.index)
    remaining_mask = pd.Series(True, index=date_series.index)
    
    for fmt in formats_to_try:
        if remaining_mask.sum() == 0:
            break
            
        try:
            temp_result = pd.to_datetime(date_series[remaining_mask], format=fmt, errors='coerce')
            valid_mask = temp_result.notna()
            
            if valid_mask.sum() > 0:
                result.loc[remaining_mask] = temp_result
                remaining_mask = remaining_mask & result.isna()
        except:
            continue
    
    return result

def load_data(folder_path: str, dropna_type: str = 'any', min_non_na: Optional[int] = None) -> pd.DataFrame:
    """
    Load and merge ALL split-adjusted CSV files from Download Data pattern.
    Uses smart date conversion and proper Close price column detection.
    """
    print(f"Loading split-adjusted data from {folder_path}...")
    
    # Get all Download Data files
    csv_files = [f for f in os.listdir(folder_path) 
                 if f.endswith('.csv') and f.startswith('Download Data - STOCK_VN_XSTC_')]
    
    if not csv_files:
        raise ValueError(f"No 'Download Data - STOCK_VN_XSTC_*' files found in {folder_path}")
    
    print(f"Found {len(csv_files)} files to process")
    
    # Improved ticker grouping (get first 3 chars after XSTC_)
    ticker_files = {}
    for file in csv_files:
        if "XSTC_" in file:
            ticker = file.split("XSTC_")[1][:3]  # Get first 3 chars (FPT, HPG, etc.)
        else:
            continue  # Skip files that don't match pattern
        
        if ticker not in ticker_files:
            ticker_files[ticker] = []
        ticker_files[ticker].append(file)
    
    print(f"Grouped into {len(ticker_files)} tickers: {list(ticker_files.keys())}")
    
    # Process each ticker separately
    ticker_dfs = {}
    for ticker, files in ticker_files.items():
        print(f"\nProcessing {ticker} ({len(files)} files)...")
        yearly_data = []
        for file in sorted(files):
            file_path = os.path.join(folder_path, file)
            try:
                df = pd.read_csv(file_path)
                
                # Improved column selection: Use 'Close' column if available, otherwise second-to-last
                if 'Close' in df.columns:
                    df_clean = df[['Date', 'Close']].copy()
                else:
                    # Assume Date is first, Close is second-to-last (not last which is Volume)
                    df_clean = df.iloc[:, [0, -2]].copy()
                    df_clean.columns = ['Date', 'Close']
                
                # Smart date conversion
                df_clean['Date'] = smart_date_convert(df_clean['Date'])
                
                # Clean price data
                df_clean['Close'] = pd.to_numeric(
                    df_clean['Close'].astype(str).str.replace(',', ''), 
                    errors='coerce'
                )
                
                # Remove invalid rows
                df_clean = df_clean.dropna()
                
                if len(df_clean) > 0:
                    df_clean = df_clean.set_index('Date').sort_index()
                    df_clean.columns = [f"{ticker}_DATA"]
                    yearly_data.append(df_clean)
                    print(f"  {file}: {len(df_clean)} records, {df_clean.index.min().date()} to {df_clean.index.max().date()}")
                    
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
    
    # Debug: Show data availability per ticker
    print(f"Raw combined shape: {final_df.shape}")
    for col in final_df.columns:
        non_na = final_df[col].notna().sum()
        print(f"  {col}: {non_na} non-NA values")
    
    # More conservative dropna - only drop completely empty rows
    if dropna_type == 'any':
        # Keep rows with at least 1 ticker (was too strict before)
        min_tickers = 1  
    elif dropna_type == 'all':
        # Keep rows with all tickers
        min_tickers = len(ticker_dfs)
    else:
        # Default: keep rows with at least 1 ticker
        min_tickers = 1
    
    # Override with min_non_na if provided
    if min_non_na is not None:
        min_tickers = min_non_na
    
    original_len = len(final_df)
    final_df = final_df.dropna(thresh=min_tickers)
    
    print(f"Applied dropna (thresh={min_tickers}): {original_len} -> {len(final_df)} rows")
    print(f"Final shape: {final_df.shape}")
    
    if len(final_df) > 0:
        print(f"Date range: {final_df.index.min()} to {final_df.index.max()}")
        print(f"Columns: {list(final_df.columns)}")
    
    return final_df


def check_negative_or_zero(price_df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with any closing price less than or equal to 0."""
    mask = (price_df <= 0).any(axis=1)
    if mask.sum() > 0:
        print(f"{mask.sum()} days have closing price <= 0 and will be removed.")
    return price_df[~mask]

def compute_log_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns for a closing price DataFrame."""
    log_returns = np.log(price_df / price_df.shift(1))
    log_returns = pd.DataFrame(log_returns, index=price_df.index, columns=price_df.columns)
    return log_returns.dropna()

def save_to_csv(df: pd.DataFrame, path: str):
    """Save DataFrame to CSV file."""
    df.to_csv(path)

def debug_data_summary(df: pd.DataFrame):
    """Simple data summary for perfect datasets."""
    print("\n=== DATA SUMMARY ===")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Columns: {list(df.columns)}")
    
    # Quick availability check
    missing_count = df.isnull().sum().sum()
    if missing_count == 0:
        print("✓ No missing values - perfect data alignment")
    else:
        print(f"Found {missing_count} missing values")
    print("="*40)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run data preparation pipeline.")
    parser.add_argument('--raw_folder', type=str, required=True, help='Path to raw data folder')
    parser.add_argument('--processed_folder', type=str, required=True, help='Path to save processed data')
    parser.add_argument('--dropna_type', type=str, default='any', choices=['any', 'all'], 
                       help="Row filter: 'any' (intersection) or 'all' (union)")
    parser.add_argument('--min_non_na', type=int, default=None, 
                       help='Minimum non-NA values per row (overrides dropna_type if set)')
    args = parser.parse_args()
    
    print("=== SPLIT-ADJUSTED DATA PREPARATION ===")
    
    # Create processed folder if it does not exist
    os.makedirs(args.processed_folder, exist_ok=True)
    
    # Load split-adjusted data
    data = load_data(args.raw_folder, dropna_type=args.dropna_type, min_non_na=args.min_non_na)
    
    # Show data summary
    debug_data_summary(data)
    
    # Remove rows with closing price <= 0
    data = check_negative_or_zero(data)
    
    # Save cleaned price data
    price_path = os.path.join(args.processed_folder, 'price_cleaned.csv')
    save_to_csv(data, price_path)
    print(f"Saved cleaned price data to {price_path}")
    
    # Compute log returns
    log_returns = compute_log_returns(data).round(6)
    logret_path = os.path.join(args.processed_folder, 'log_returns.csv')
    save_to_csv(log_returns, logret_path)
    print(f"Saved log returns to {logret_path}")
    
    # Save log returns index 
    logret_index_path = os.path.join(args.processed_folder, 'log_returns_index.csv')
    index_df = pd.DataFrame(log_returns.index, columns=['Date'])
    index_df.to_csv(logret_index_path, index=False)
    print(f"Saved log returns index to {logret_index_path}")
    
    print("Split-adjusted data preparation completed.")