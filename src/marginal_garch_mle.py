"""
Simple ARIMA+GARCH using traditional MLE
Uses arch package for GARCH estimation
"""
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
from scipy import stats

def fit_arima(returns, order=(0,1,1)):
    """Fit ARIMA model using MLE"""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(returns, order=order, trend='c').fit(
                disp=False, 
                method='lbfgs',
                maxfun=1000
            )
        return model
    except:
        return None

def fit_garch(residuals):
    """Fit EGARCH model using MLE with arch package"""
    try:
        # EGARCH model - best performer overall
        egarch_model = arch_model(
            residuals, 
            vol='EGARCH', 
            p=1, q=1,
            dist='studentst'
        )
        
        # Fit using MLE
        egarch_fit = egarch_model.fit(disp='off', show_warning=False)
        return egarch_fit
    except Exception as e:
        print(f"    EGARCH failed: {str(e)[:50]}...")
        return None

def process_ticker(data, ticker, arima_order=(0,1,1)):
    """Process single ticker: ARIMA + EGARCH using MLE"""
    print(f"\\nProcessing {ticker}...")
    
    # Step 1: Fit ARIMA
    arima_model = fit_arima(data, arima_order)
    if arima_model is None:
        print(f"  ❌ ARIMA failed for {ticker}")
        return None
    
    print(f"  ✅ ARIMA({arima_order[0]},{arima_order[1]},{arima_order[2]}): AIC={arima_model.aic:.2f}")
    
    # Step 2: Get residuals and fit EGARCH
    residuals = arima_model.resid.dropna()
    if len(residuals) < 50:
        print(f"  ❌ Too few residuals ({len(residuals)}) for GARCH")
        return None
    
    egarch_fit = fit_garch(residuals)
    if egarch_fit is None:
        print(f"  ❌ EGARCH failed for {ticker}")
        return None
    
    print(f"  ✅ EGARCH(1,1): AIC={egarch_fit.aic:.2f}")
    
    return {
        'ticker': ticker,
        'arima_model': arima_model,
        'egarch_model': egarch_fit,
        'egarch_params': egarch_fit.params.to_dict(),
        'arima_aic': arima_model.aic,
        'egarch_aic': egarch_fit.aic
    }

def get_standardized_residuals(result):
    """Extract standardized residuals from fitted models"""
    try:
        arima_model = result['arima_model']
        egarch_fit = result['egarch_model']
        
        # Get fitted residuals and volatility
        residuals = arima_model.resid.dropna()
        volatility = egarch_fit.conditional_volatility
        
        # Align lengths (use shorter length)
        min_len = min(len(residuals), len(volatility))
        residuals = residuals[-min_len:]
        volatility = volatility[-min_len:]
        
        # Standardized residuals
        std_residuals = residuals / volatility
        return std_residuals
    except Exception as e:
        print(f"  ⚠️  Could not compute standardized residuals: {str(e)[:50]}...")
        return None

if __name__ == "__main__":
    # Step 1: Load data
    print("🔄 Step 1: Loading data...")
    data_path = "data/processed/log_returns.csv"
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        exit(1)
    
    df = pd.read_csv(data_path).drop(columns=['Date'], errors='ignore')
    print(f"✅ Loaded data: {df.shape[0]} observations, {df.shape[1]} tickers")
    
    # Step 2: Process each ticker
    print("\\n🔄 Step 2: Fitting ARIMA+EGARCH models...")
    results = {}
    
    for ticker in df.columns:
        result = process_ticker(df[ticker], ticker)
        if result:
            results[ticker] = result
            
            # Get standardized residuals for next steps
            std_resid = get_standardized_residuals(result)
            if std_resid is not None:
                result['standardized_residuals'] = std_resid
    
    print(f"\\n✅ Step 2 completed: {len(results)}/{len(df.columns)} tickers successful")
    
    # Step 3: Save results (before diagnostics)
    print("\\n🔄 Step 3: Saving results...")
    if results:
        # Create models/garch directory if not exists
        os.makedirs("models/garch", exist_ok=True)
        
        # Save complete models with fitted objects
        models_dict = {}
        summary_data = []
        std_residuals_dict = {}
        
        for ticker, result in results.items():
            # Store models for later use
            models_dict[ticker] = {
                'arima_model': result['arima_model'],
                'egarch_model': result['egarch_model'],
                'egarch_params': result['egarch_params'],
                'arima_aic': result['arima_aic'],
                'egarch_aic': result['egarch_aic']
            }
            
            # Prepare summary data
            summary_row = {'ticker': ticker, 'arima_aic': result['arima_aic'], 'egarch_aic': result['egarch_aic']}
            summary_row.update(result['egarch_params'])
            summary_data.append(summary_row)
            
            # Store standardized residuals
            if 'standardized_residuals' in result:
                std_residuals_dict[ticker] = result['standardized_residuals']
        
        # Save models (pickle format for full model objects)
        with open("models/garch/marginal_model.pkl", "wb") as f:
            pickle.dump(models_dict, f)
        print("✅ Saved full models to models/garch/marginal_model.pkl")

        # Save summary (CSV format for easy reading)
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv("models/garch/marginal_model_summary.csv", index=False)
        print("✅ Saved model summary to models/garch/marginal_model_summary.csv")

        # Save standardized residuals (CSV format for EVT stage)
        if std_residuals_dict:
            std_resids_df = pd.DataFrame(std_residuals_dict)
            std_resids_df.to_csv("models/garch/std_resids.csv", index=True)
            print("✅ Saved standardized residuals to models/garch/std_resids.csv")

        print(f"📁 All results saved for {len(results)} tickers")
    else:
        print("❌ No results to save")
    
    # Step 4: Model diagnostics and summary
    print("\\n🔄 Step 4: Model diagnostics and summary...")
    if results:
        print("\\n=== EGARCH Results Summary ===")
        for ticker, result in results.items():
            params = result['egarch_params']
            aic = result['egarch_aic']
            
            print(f"\\n{ticker}:")
            print(f"  AIC: {aic:.2f}")
            print(f"  EGARCH Parameters:")
            for param_name, param_value in params.items():
                print(f"    {param_name}: {param_value:.6f}")
            
            # Check degrees of freedom
            if 'nu' in params:
                nu = params['nu']
                if nu < 4:
                    print(f"  ⚠️  Low degrees of freedom (ν={nu:.2f}) - heavy tails")
                else:
                    print(f"  ✅ Normal degrees of freedom (ν={nu:.2f})")
                    
        print(f"\\n✅ All {len(results)} tickers fitted with EGARCH successfully!")
        print("📈 Ready for EVT and Copula modeling")
    
    print("✅ Process completed!")
