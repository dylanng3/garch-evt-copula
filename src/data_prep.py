"""
Module: data_prep.py
Description: Utility functions for loading, cleaning, and saving data for the GARCH-EVT-Copula project.
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import genpareto, kstest


def convert_date(date_series):
    """
    Thử nhiều format date khác nhau
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
                print(f"  Format {fmt}: converted {valid_mask.sum()} dates")
        except:
            continue
    
    return result


def load_data(raw_folder):
    """
    Merge với smart date conversion
    """
    import glob
    
    csv_files = glob.glob(os.path.join(raw_folder, "Download Data - STOCK_VN_XSTC_*.csv"))
    print(f"Found {len(csv_files)} files")
    
    tickers = {}
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        ticker = filename.split("XSTC_")[1][:3]
        
        if ticker not in tickers:
            tickers[ticker] = []
        tickers[ticker].append(file_path)
    
    result = {}
    for ticker, files in tickers.items():
        print(f"\n=== {ticker}: {len(files)} files ===")
        
        all_dfs = []
        for file_path in sorted(files):
            print(f"Processing {os.path.basename(file_path)}")
            df = pd.read_csv(file_path)
            
            # Get Date and Close columns
            if 'Close' in df.columns:
                df_clean = df[['Date', 'Close']].copy()
            else:
                # Assume Date is first col, Close is second-to-last col
                df_clean = df.iloc[:, [0, -2]].copy()
                df_clean.columns = ['Date', 'Close']
            
            print(f"  Raw: {len(df_clean)} rows")
            
            # Smart date conversion
            df_clean['Date'] = convert_date(df_clean['Date'])
            date_na = df_clean['Date'].isna().sum()
            print(f"  Failed dates: {date_na}")
            
            # Clean price
            df_clean['Close'] = pd.to_numeric(df_clean['Close'].astype(str).str.replace(',', ''), errors='coerce')
            price_na = df_clean['Close'].isna().sum()
            print(f"  Failed prices: {price_na}")
            
            # Remove invalid
            df_clean = df_clean.dropna()
            print(f"  Final: {len(df_clean)} rows")
            
            if len(df_clean) > 0:
                all_dfs.append(df_clean)
        
        # Combine
        if all_dfs:
            combined = pd.concat(all_dfs).drop_duplicates(subset=['Date'], keep='last')
            combined = combined.set_index('Date').sort_index()
            combined.columns = [f'{ticker}_DATA']
            result[ticker] = combined
            
            print(f"FINAL {ticker}: {len(combined)} rows ({combined.index.min().date()} to {combined.index.max().date()})")
    
    return result


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
    

def engle_ng_tests(x):
    # x: returns đã demean
    e = x - x.mean()
    I_neg = (e.shift(1) < 0).astype(int)
    y = e**2
    # Sign-bias
    X1 = sm.add_constant(I_neg)
    s_res = sm.OLS(y, X1, missing='drop').fit()
    # Size-bias
    X2 = sm.add_constant(pd.concat([e.shift(1).abs()], axis=1))
    z_res = sm.OLS(y, X2, missing='drop').fit()
    # Joint (sign + size)
    X3 = sm.add_constant(pd.concat([I_neg, e.shift(1).abs()], axis=1))
    j_res = sm.OLS(y, X3, missing='drop').fit()
    return s_res.f_pvalue, z_res.f_pvalue, j_res.f_pvalue


def mean_residual_life(x, qs=np.linspace(0.85, 0.99, 15)):
    x = np.asarray(x)
    us, me = [], []
    for q in qs:
        u = np.quantile(x, q)
        exceed = x[x>u] - u
        if len(exceed)>5:
            us.append(u)
            me.append(exceed.mean())
    return np.array(us), np.array(me)


def hill_plot(x, qs=np.linspace(0.90, 0.995, 25)):
    x = np.sort(x)
    hills, ks = [], []
    for q in qs:
        u = np.quantile(x, q)
        y = x[x>u] - u
        if len(y)>20:
            xi, beta, loc = genpareto.fit(y, floc=0.0)[:3]
            # KS test (thận trọng vì ước lượng tham số)
            D, p = kstest(y, 'genpareto', args=(xi, 0.0, beta))
            hills.append((q, u, len(y), xi, beta, p))
    return pd.DataFrame(hills, columns=['q','u','Nu','xi','beta','ks_p'])

