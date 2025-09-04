"""
ARIMA+EGARCH using traditional MLE with Skewed Student's t distribution
Semi-parametric marginal distribution: KDE for center + EVT (GPD) for tails
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
from scipy.stats import genpareto
from sklearn.neighbors import KernelDensity

def fit_arima(returns, order=(1,1,1)):
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
        # EGARCH model with Skewed Student's t distribution
        egarch_model = arch_model(
            residuals, 
            vol='EGARCH', 
            p=1, q=1,
            dist='skewstudent'
        )
        
        # Fit using MLE
        egarch_fit = egarch_model.fit(disp='off', show_warning=False)
        return egarch_fit
    except Exception as e:
        print(f"EGARCH failed: {str(e)[:50]}...")
        return None

def process_ticker(data, ticker, arima_order=(1,1,1)):
    """Process single ticker: ARIMA + EGARCH using MLE"""
    print(f"Processing {ticker}...")
    
    # Step 1: Fit ARIMA
    arima_model = fit_arima(data, arima_order)
    if arima_model is None:
        print(f"ARIMA failed for {ticker}")
        return None
    
    print(f"ARIMA({arima_order[0]},{arima_order[1]},{arima_order[2]}): AIC={arima_model.aic:.2f}")
    
    # Step 2: Get residuals and fit EGARCH
    residuals = arima_model.resid.dropna()
    if len(residuals) < 50:
        print(f"Too few residuals ({len(residuals)}) for GARCH")
        return None
    
    egarch_fit = fit_garch(residuals)
    if egarch_fit is None:
        print(f"EGARCH failed for {ticker}")
        return None

    print(f"EGARCH(1,1): AIC={egarch_fit.aic:.2f}")

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
        print(f"Could not compute standardized residuals: {str(e)[:50]}...")
        return None

# ============================================================================
# EVT (Extreme Value Theory) Functions for Semi-Parametric Marginal Distribution
# ============================================================================

def select_threshold(data, tail_prob=0.1):
    """
    Select threshold for POT (Peaks Over Threshold) using percentile method
    
    Parameters:
    -----------
    data : array-like
        Standardized residuals (should be i.i.d.)
    tail_prob : float, default 0.1
        Tail probability (e.g., 0.1 means top 10% for upper tail)
        
    Returns:
    --------
    dict : threshold information for both tails
    """
    data = np.array(data)
    
    # Upper tail threshold (right tail)
    upper_threshold = np.percentile(data, (1 - tail_prob) * 100)
    
    # Lower tail threshold (left tail)  
    lower_threshold = np.percentile(data, tail_prob * 100)
    
    # Count exceedances
    upper_exceedances = np.sum(data > upper_threshold)
    lower_exceedances = np.sum(data < lower_threshold)
    
    return {
        'upper_threshold': upper_threshold,
        'lower_threshold': lower_threshold,
        'upper_exceedances': upper_exceedances,
        'lower_exceedances': lower_exceedances,
        'tail_prob': tail_prob,
        'n_total': len(data)
    }

def fit_gpd_tail(data, threshold, tail='upper'):
    """
    Fit Generalized Pareto Distribution to tail exceedances using MLE
    
    Parameters:
    -----------
    data : array-like
        Standardized residuals
    threshold : float
        Threshold value
    tail : str
        'upper' for right tail, 'lower' for left tail
        
    Returns:
    --------
    dict : GPD parameters and diagnostics
    """
    data = np.array(data)
    
    if tail == 'upper':
        # Right tail: exceedances above threshold
        exceedances = data[data > threshold] - threshold
    else:
        # Left tail: exceedances below threshold (convert to positive)
        exceedances = threshold - data[data < threshold]
    
    if len(exceedances) < 10:
        return None
    
    try:
        # Fit GPD using MLE (scipy uses c=shape, scale=scale)
        shape, loc, scale = genpareto.fit(exceedances, floc=0)  # loc=0 for GPD
        
        # Calculate AIC for model selection
        log_likelihood = np.sum(genpareto.logpdf(exceedances, shape, loc=0, scale=scale))
        aic = -2 * log_likelihood + 2 * 2  # 2 parameters: shape, scale
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = stats.kstest(exceedances, lambda x: genpareto.cdf(x, shape, loc=0, scale=scale))
        
        return {
            'shape': shape,        # γ (gamma) parameter
            'scale': scale,        # β (beta) parameter  
            'n_exceedances': len(exceedances),
            'threshold': threshold,
            'tail': tail,
            'aic': aic,
            'log_likelihood': log_likelihood,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pvalue,
            'exceedances': exceedances
        }
    except Exception as e:
        print(f"GPD fitting failed for {tail} tail: {str(e)[:50]}...")
        return None

def estimate_tail_probability(gpd_params, threshold, x):
    """
    Estimate tail probability using fitted GPD
    
    Formula: F̂(u + x) ≈ (N_u/n) * (1 + γ̂*x/β̂)^(-1/γ̂)
    
    Parameters:
    -----------
    gpd_params : dict
        Fitted GPD parameters from fit_gpd_tail()
    threshold : float
        Threshold value u
    x : float or array-like
        Values above threshold (x ≥ 0)
        
    Returns:
    --------
    float or array : tail probabilities
    """
    if gpd_params is None:
        return None
    
    gamma = gpd_params['shape']
    beta = gpd_params['scale'] 
    n_exceed = gpd_params['n_exceedances']
    n_total = len(gpd_params['exceedances']) / (gpd_params['n_exceedances'] / gpd_params['n_exceedances'])  # This needs fixing
    
    # Probability of exceeding threshold
    tail_prob = n_exceed / n_total if 'n_total' in gpd_params else 0.1
    
    x = np.array(x)
    
    if gamma != 0:
        # Standard GPD case
        tail_cdf = tail_prob * (1 + gamma * x / beta) ** (-1 / gamma)
    else:
        # Exponential case (γ = 0)
        tail_cdf = tail_prob * np.exp(-x / beta)
    
    return tail_cdf

def estimate_tail_quantile(gpd_params, threshold, p):
    """
    Estimate tail quantile (VaR) using fitted GPD
    
    Parameters:
    -----------
    gpd_params : dict
        Fitted GPD parameters
    threshold : float
        Threshold value u
    p : float
        Probability level (e.g., 0.01 for 1% VaR)
        
    Returns:
    --------
    float : quantile value
    """
    if gpd_params is None:
        return None
    
    gamma = gpd_params['shape']
    beta = gpd_params['scale']
    n_exceed = gpd_params['n_exceedances']
    
    # Assuming we know the tail probability from threshold selection
    tail_prob = 0.1  # This should be passed or calculated properly
    
    if gamma != 0:
        # Quantile function for GPD
        quantile = threshold + (beta / gamma) * ((tail_prob / p) ** gamma - 1)
    else:
        # Exponential case
        quantile = threshold + beta * np.log(tail_prob / p)
    
    return quantile

def fit_kde_center(data, lower_threshold, upper_threshold, bandwidth='scott'):
    """
    Fit Kernel Density Estimation to the center part of distribution
    
    Parameters:
    -----------
    data : array-like
        Standardized residuals
    lower_threshold : float
        Lower threshold (left tail cutoff)
    upper_threshold : float
        Upper threshold (right tail cutoff)
    bandwidth : str or float
        KDE bandwidth selection method
        
    Returns:
    --------
    dict : KDE model and center data
    """
    data = np.array(data)
    
    # Extract center data (between thresholds)
    center_data = data[(data >= lower_threshold) & (data <= upper_threshold)]
    
    if len(center_data) < 20:
        return None
    
    try:
        # Fit KDE
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde.fit(center_data.reshape(-1, 1))
        
        return {
            'kde_model': kde,
            'center_data': center_data,
            'n_center': len(center_data),
            'center_range': (lower_threshold, upper_threshold)
        }
    except Exception as e:
        print(f"KDE fitting failed: {str(e)[:50]}...")
        return None

def create_semiparametric_marginal(std_residuals, tail_prob=0.1):
    """
    Create semi-parametric marginal distribution: KDE center + GPD tails
    
    Parameters:
    -----------
    std_residuals : array-like
        Standardized residuals from GARCH model
    tail_prob : float
        Tail probability for threshold selection
        
    Returns:
    --------
    dict : Complete semi-parametric marginal distribution
    """
    # Step 1: Select thresholds
    threshold_info = select_threshold(std_residuals, tail_prob)
    
    # Step 2: Fit GPD to both tails
    upper_gpd = fit_gpd_tail(std_residuals, threshold_info['upper_threshold'], 'upper')
    lower_gpd = fit_gpd_tail(std_residuals, threshold_info['lower_threshold'], 'lower')
    
    # Step 3: Fit KDE to center
    kde_center = fit_kde_center(
        std_residuals, 
        threshold_info['lower_threshold'],
        threshold_info['upper_threshold']
    )
    
    return {
        'threshold_info': threshold_info,
        'upper_tail_gpd': upper_gpd,
        'lower_tail_gpd': lower_gpd, 
        'center_kde': kde_center,
        'tail_prob': tail_prob,
        'marginal_type': 'semiparametric'
    }

def semiparametric_cdf(x, marginal_dist):
    """
    Evaluate CDF of semi-parametric marginal distribution
    
    Parameters:
    -----------
    x : float or array-like
        Values to evaluate CDF at
    marginal_dist : dict
        Semi-parametric marginal distribution from create_semiparametric_marginal()
        
    Returns:
    --------
    float or array : CDF values
    """
    x = np.array(x)
    cdf_values = np.zeros_like(x)
    
    threshold_info = marginal_dist['threshold_info']
    lower_thresh = threshold_info['lower_threshold']
    upper_thresh = threshold_info['upper_threshold']
    tail_prob = marginal_dist['tail_prob']
    
    # Lower tail (x < lower_threshold)
    lower_mask = x < lower_thresh
    if np.any(lower_mask) and marginal_dist['lower_tail_gpd'] is not None:
        lower_gpd = marginal_dist['lower_tail_gpd']
        gamma = lower_gpd['shape']
        beta = lower_gpd['scale']
        
        # For lower tail, we need to reverse the calculation
        exceedances = lower_thresh - x[lower_mask]
        if gamma != 0:
            tail_cdf = tail_prob * (1 + gamma * exceedances / beta) ** (-1 / gamma)
        else:
            tail_cdf = tail_prob * np.exp(-exceedances / beta)
        
        cdf_values[lower_mask] = tail_prob - tail_cdf
    
    # Center part (lower_threshold ≤ x ≤ upper_threshold)
    center_mask = (x >= lower_thresh) & (x <= upper_thresh)
    if np.any(center_mask) and marginal_dist['center_kde'] is not None:
        kde = marginal_dist['center_kde']['kde_model']
        center_data = marginal_dist['center_kde']['center_data']
        
        # Use empirical CDF for center part
        for i, xi in enumerate(x[center_mask]):
            cdf_values[center_mask][i] = tail_prob + (1 - 2*tail_prob) * np.mean(center_data <= xi)
    
    # Upper tail (x > upper_threshold)
    upper_mask = x > upper_thresh
    if np.any(upper_mask) and marginal_dist['upper_tail_gpd'] is not None:
        upper_gpd = marginal_dist['upper_tail_gpd']
        gamma = upper_gpd['shape']
        beta = upper_gpd['scale']
        
        exceedances = x[upper_mask] - upper_thresh
        if gamma != 0:
            tail_cdf = tail_prob * (1 + gamma * exceedances / beta) ** (-1 / gamma)
        else:
            tail_cdf = tail_prob * np.exp(-exceedances / beta)
        
        cdf_values[upper_mask] = 1 - tail_cdf
    
    return cdf_values if len(cdf_values) > 1 else cdf_values[0]

def semiparametric_quantile(p, marginal_dist):
    """
    Calculate quantile (inverse CDF) of semi-parametric marginal distribution
    
    Parameters:
    -----------
    p : float or array-like
        Probability values (0 < p < 1)
    marginal_dist : dict
        Semi-parametric marginal distribution
        
    Returns:
    --------
    float or array : Quantile values
    """
    p = np.array(p)
    quantiles = np.zeros_like(p)
    
    threshold_info = marginal_dist['threshold_info']
    lower_thresh = threshold_info['lower_threshold']
    upper_thresh = threshold_info['upper_threshold']
    tail_prob = marginal_dist['tail_prob']
    
    # Lower tail (p < tail_prob)
    lower_mask = p < tail_prob
    if np.any(lower_mask) and marginal_dist['lower_tail_gpd'] is not None:
        lower_gpd = marginal_dist['lower_tail_gpd']
        gamma = lower_gpd['shape']
        beta = lower_gpd['scale']
        
        p_tail = p[lower_mask]
        if gamma != 0:
            exceedances = (beta / gamma) * ((tail_prob / (tail_prob - p_tail)) ** gamma - 1)
        else:
            exceedances = beta * np.log(tail_prob / (tail_prob - p_tail))
        
        quantiles[lower_mask] = lower_thresh - exceedances
    
    # Center part (tail_prob ≤ p ≤ 1-tail_prob)
    center_mask = (p >= tail_prob) & (p <= 1 - tail_prob)
    if np.any(center_mask) and marginal_dist['center_kde'] is not None:
        center_data = marginal_dist['center_kde']['center_data']
        
        # Use empirical quantiles for center
        for i, pi in enumerate(p[center_mask]):
            # Convert to empirical quantile
            empirical_p = (pi - tail_prob) / (1 - 2*tail_prob)
            quantiles[center_mask][i] = np.percentile(center_data, empirical_p * 100)
    
    # Upper tail (p > 1-tail_prob)  
    upper_mask = p > 1 - tail_prob
    if np.any(upper_mask) and marginal_dist['upper_tail_gpd'] is not None:
        upper_gpd = marginal_dist['upper_tail_gpd']
        gamma = upper_gpd['shape']
        beta = upper_gpd['scale']
        
        p_tail = 1 - p[upper_mask]  # Tail probability
        if gamma != 0:
            exceedances = (beta / gamma) * ((tail_prob / p_tail) ** gamma - 1)
        else:
            exceedances = beta * np.log(tail_prob / p_tail)
        
        quantiles[upper_mask] = upper_thresh + exceedances
    
    return quantiles if len(quantiles) > 1 else quantiles[0]

def process_all_marginals(results, tail_prob=0.1):
    """
    Process all tickers to create semi-parametric marginal distributions
    
    Parameters:
    -----------
    results : dict
        Results from ARIMA+EGARCH fitting
    tail_prob : float
        Tail probability for EVT
        
    Returns:
    --------
    dict : Semi-parametric marginals for all tickers
    """
    marginal_distributions = {}
    evt_summary = []
    
    print(f"\\nCreating semi-parametric marginals (tail_prob={tail_prob})...")
    
    for ticker, result in results.items():
        if 'standardized_residuals' in result:
            std_resids = result['standardized_residuals']
            
            # Create semi-parametric marginal
            marginal = create_semiparametric_marginal(std_resids, tail_prob)
            
            if marginal['upper_tail_gpd'] is not None and marginal['lower_tail_gpd'] is not None:
                marginal_distributions[ticker] = marginal
                
                # Summary statistics
                upper_gpd = marginal['upper_tail_gpd']
                lower_gpd = marginal['lower_tail_gpd']
                
                evt_summary.append({
                    'ticker': ticker,
                    'upper_shape': upper_gpd['shape'],
                    'upper_scale': upper_gpd['scale'], 
                    'upper_threshold': upper_gpd['threshold'],
                    'upper_n_exceed': upper_gpd['n_exceedances'],
                    'upper_ks_pvalue': upper_gpd['ks_pvalue'],
                    'lower_shape': lower_gpd['shape'],
                    'lower_scale': lower_gpd['scale'],
                    'lower_threshold': lower_gpd['threshold'], 
                    'lower_n_exceed': lower_gpd['n_exceedances'],
                    'lower_ks_pvalue': lower_gpd['ks_pvalue'],
                    'center_n': marginal['center_kde']['n_center'] if marginal['center_kde'] else 0
                })
                
                print(f"  ✅ {ticker}: EVT fitted (upper γ={upper_gpd['shape']:.3f}, lower γ={lower_gpd['shape']:.3f})")
            else:
                print(f"  ❌ {ticker}: EVT fitting failed")
    
    print(f"Semi-parametric marginals completed: {len(marginal_distributions)}/{len(results)} successful")
    
    return {
        'marginal_distributions': marginal_distributions,
        'evt_summary': evt_summary,
        'tail_prob': tail_prob
    }

if __name__ == "__main__":
    # Step 1: Load data
    print("Step 1: Loading data...")
    data_path = "data/processed/log_returns.csv"
    if not os.path.exists(data_path):
        print(f" Data file not found: {data_path}")
        exit(1)
    
    df = pd.read_csv(data_path).drop(columns=['Date'], errors='ignore')
    print(f"Loaded data: {df.shape[0]} observations, {df.shape[1]} tickers")
    
    # Step 2: Process each ticker
    print("Step 2: Fitting ARIMA+EGARCH models...")
    results = {}
    
    for ticker in df.columns:
        result = process_ticker(df[ticker], ticker)
        if result:
            results[ticker] = result
            
            # Get standardized residuals for next steps
            std_resid = get_standardized_residuals(result)
            if std_resid is not None:
                result['standardized_residuals'] = std_resid
    
    print(f"Step 2 completed: {len(results)}/{len(df.columns)} tickers successful")
    
    # Step 3: Save results (before diagnostics)
    print("Step 3: Saving results...")
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
        print("Saved full models to models/garch/marginal_model.pkl")

        # Save summary (CSV format for easy reading)
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv("models/garch/marginal_model_summary.csv", index=False)
        print("Saved model summary to models/garch/marginal_model_summary.csv")

        # Save standardized residuals (CSV format for EVT stage)
        if std_residuals_dict:
            std_resids_df = pd.DataFrame(std_residuals_dict)
            std_resids_df.to_csv("models/garch/std_resids.csv", index=True)
            print("Saved standardized residuals to models/garch/std_resids.csv")

        print(f"All results saved for {len(results)} tickers")
    else:
        print("No results to save")
    
    # Step 4: EVT - Semi-parametric Marginal Distributions
    print("Step 4: Creating semi-parametric marginal distributions...")
    if results:
        marginal_results = process_all_marginals(results, tail_prob=0.1)
        
        if marginal_results['marginal_distributions']:
            # Save EVT results
            os.makedirs("models/evt", exist_ok=True)
            
            # Save marginal distributions (pickle for complex objects)
            with open("models/evt/marginal_distributions.pkl", "wb") as f:
                pickle.dump(marginal_results['marginal_distributions'], f)
            print("Saved semi-parametric marginals to models/evt/marginal_distributions.pkl")
            
            # Save EVT summary (CSV for analysis)
            evt_df = pd.DataFrame(marginal_results['evt_summary'])
            evt_df.to_csv("models/evt/evt_summary.csv", index=False)
            print("Saved EVT summary to models/evt/evt_summary.csv")
            
            # Quick EVT diagnostics
            print("\\n=== EVT Summary ===")
            for _, row in evt_df.iterrows():
                ticker = row['ticker']
                upper_shape = row['upper_shape']
                lower_shape = row['lower_shape']
                upper_ks = row['upper_ks_pvalue']
                lower_ks = row['lower_ks_pvalue']
                
                print(f"{ticker}:")
                print(f"  Upper tail: γ={upper_shape:.3f}, KS p={upper_ks:.3f} {'✅' if upper_ks > 0.05 else '⚠️'}")
                print(f"  Lower tail: γ={lower_shape:.3f}, KS p={lower_ks:.3f} {'✅' if lower_ks > 0.05 else '⚠️'}")
                
                # Interpret tail behavior
                if upper_shape > 0:
                    print(f"  Upper tail: Heavy-tailed (Pareto-type)")
                elif upper_shape < 0:
                    print(f"  Upper tail: Light-tailed (bounded)")
                else:
                    print(f"  Upper tail: Exponential-type")
    
    # Step 5: Model diagnostics and summary
    print("Step 5: Model diagnostics and summary...")
    if results:
        print("=== EGARCH Results Summary ===")
        for ticker, result in results.items():
            params = result['egarch_params']
            aic = result['egarch_aic']
            
            print(f"{ticker}:")
            print(f"  AIC: {aic:.2f}")
            print(f"  EGARCH Parameters:")
            for param_name, param_value in params.items():
                print(f"    {param_name}: {param_value:.6f}")
            
            # Check degrees of freedom
            if 'nu' in params:
                nu = params['nu']
                if nu < 4:
                    print(f"    Low degrees of freedom (ν={nu:.2f}) - heavy tails")
                else:
                    print(f"   Normal degrees of freedom (ν={nu:.2f})")
                    
        print(f"All {len(results)} tickers fitted with EGARCH successfully!")
        print("Ready for EVT and Copula modeling")

    print("Process completed!")
