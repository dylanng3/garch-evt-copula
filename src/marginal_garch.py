"""
Refactored module: ARIMA grid-search + Bayesian GARCH(1,1) via Hamiltonian Monte Carlo (NUTS)
- Removes all EVT-related code and artifacts
- Keeps residual diagnostics and PIT computation
- **No Aesara required**. Uses **PyMC v5 + PyTensor** instead of Aesara.

Dependencies (install if missing):
    pip install "pymc>=5" "pytensor>=2" numpy pandas statsmodels scipy

Outputs (compatible with previous workflow where possible):
- summary_df: overview table per ticker
- models: dict per ticker including ARIMA fit, posterior, std_resid, PIT, diagnostics
- Optional: save artifacts under models/garch/

Author: refactor for user's project
"""
from __future__ import annotations
import warnings
import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

# Suppress specific warnings more thoroughly
warnings.filterwarnings("ignore", message="A date index has been provided, but it has no associated frequency information")
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

# Configure PyTensor to use MinGW g++ compiler with proper escaping
import os
import shutil

# Try to find g++ in MinGW
mingw_gpp = shutil.which('g++') or r"C:\msys64\mingw64\bin\g++.exe"

if os.path.exists(mingw_gpp):
    # Add MinGW to PATH first
    mingw_bin = os.path.dirname(mingw_gpp)
    os.environ["PATH"] = mingw_bin + ";" + os.environ.get("PATH", "")
    
    # Set PyTensor to use g++ (just the name, not full path since it's in PATH)
    os.environ["PYTENSOR_FLAGS"] = "cxx=g++"
    print(f"Using MinGW g++ compiler from: {mingw_gpp}")
else:
    os.environ["PYTENSOR_FLAGS"] = "cxx="
    print("No g++ compiler found, using Python fallback")

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

# --- Bayesian bits (PyMC v5 + PyTensor) ---
try:
    import pymc as pm
    import pytensor.tensor as pt
    from pytensor.scan import scan
except Exception as e:  # pragma: no cover
    raise ImportError(
        "This module requires PyMC>=5 and PyTensor. Install: pip install 'pymc>=5' 'pytensor>=2'"
    ) from e


# -------------------------
# Helpers
# -------------------------

def _ljungbox_p(x: np.ndarray, lag: int = 15) -> float:
    x = pd.Series(x).dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = acorr_ljungbox(x, lags=[lag], return_df=True)['lb_pvalue'].iloc[0]
    return float(p)


def _archlm_p(x: np.ndarray, lags: int = 10) -> float:
    x = pd.Series(x).dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = het_arch(x, nlags=lags)[1]
    return float(p)


def _garch11_volatility_numpy(eps: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    """Compute conditional variances h_t for GARCH(1,1) in NumPy, given eps_t (residuals)."""
    T = len(eps)
    h = np.empty(T, dtype=float)
    # Unconditional variance as starting value
    uncond = omega / max(1e-8, (1.0 - alpha - beta))
    h[0] = max(uncond, 1e-8)
    for t in range(1, T):
        h[t] = omega + alpha * eps[t-1] ** 2 + beta * h[t-1]
        if not np.isfinite(h[t]):
            h[t] = h[t-1]
    h = np.clip(h, 1e-12, None)
    return h


def _garch11_volatility_pt(eps: pt.TensorVariable, omega, alpha, beta):
    """PyTensor scan to compute h_t for GARCH(1,1)."""
    eps2 = eps ** 2
    uncond = omega / (1e-8 + (1.0 - alpha - beta))
    h0 = pt.maximum(uncond, 1e-8)

    def step(eps2_tm1, h_tm1, omega, alpha, beta):
        h_t = omega + alpha * eps2_tm1 + beta * h_tm1
        return pt.maximum(h_t, 1e-12)

    h_seq, _ = scan(fn=step,
                    sequences=[eps2[1:]],
                    outputs_info=h0,
                    non_sequences=[omega, alpha, beta])
    # prepend h0 to align length
    h = pt.concatenate([h0[None], h_seq])
    return h


@dataclass
class BayesGARCHResult:
    omega: float
    alpha: float
    beta: float
    nu: Optional[float]
    trace: Any
    posterior_means: Dict[str, float]
    sigma2_mean: np.ndarray
    std_resid: np.ndarray
    pit: np.ndarray


# -------------------------
# Bayesian GARCH(1,1) with HMC/NUTS
# -------------------------

def fit_bayes_garch11(
    resid: np.ndarray,
    dist: str = 'studentt',
    draws: int = 1000,
    tune: int = 1000,
    target_accept: float = 0.9,
    chains: int = 2,
    random_seed: Optional[int] = None,
) -> BayesGARCHResult:
    """
    Fits a Bayesian GARCH(1,1) on ARIMA residuals using HMC/NUTS.

    Parameters
    ----------
    resid : np.ndarray
        Residuals from mean model (should be roughly mean-zero).
    dist : {'studentt','normal'}
        Innovation distribution.
    draws, tune, target_accept, chains : PyMC sampling arguments.

    Returns
    -------
    BayesGARCHResult with posterior means, sigma^2 path, standardized residuals, and PIT.
    """
    y = np.asarray(pd.Series(resid).dropna().astype(float).values)
    if y.size < 100:
        raise ValueError("Need at least 100 observations for stable GARCH-HMC fit.")

    with pm.Model() as m:
        # Priors: constrain alpha,beta so alpha+beta<1 using a 'phi' stick-break
        phi = pm.Beta('phi', alpha=20, beta=1.5)  # typically close to 1
        phi_scaled = pm.Deterministic('phi_scaled', 0.999 * phi)
        a_raw = pm.Beta('a_raw', alpha=2, beta=8)
        alpha = pm.Deterministic('alpha', a_raw * phi_scaled)
        beta  = pm.Deterministic('beta', (1.0 - a_raw) * phi_scaled)
        omega = pm.HalfNormal('omega', sigma=np.std(y) * 0.1 + 1e-6)

        if dist.lower() == 'studentt':
            # nu > 2 for finite variance
            nu = pm.Exponential('nu_minus2', lam=0.25)
            nu_eff = pm.Deterministic('nu', 2.0 + nu)
        elif dist.lower() == 'normal':
            nu_eff = None
        else:
            raise ValueError("dist must be 'studentt' or 'normal'")

        y_shared = pm.Data('y_obs', y)
        h = _garch11_volatility_pt(y_shared, omega, alpha, beta)
        sigma = pt.sqrt(h)

        if dist.lower() == 'studentt':
            pm.StudentT('lik', nu=nu_eff, mu=0.0, sigma=sigma, observed=y_shared)
        else:
            pm.Normal('lik', mu=0.0, sigma=sigma, observed=y_shared)

        trace = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=target_accept,
            chains=chains,
            cores=1,
            random_seed=random_seed,
            progressbar=True,
        )

    # Posterior means
    post = pm.summary(trace, var_names=['omega','alpha','beta'], kind='stats')
    omega_mean = float(post.loc['omega','mean'])
    alpha_mean = float(post.loc['alpha','mean'])
    beta_mean  = float(post.loc['beta','mean'])

    nu_mean = None
    if 'nu' in trace.posterior:
        nu_mean = float(pm.summary(trace, var_names=['nu'], kind='stats').loc['nu','mean'])

    # Compute mean sigma^2 path using posterior mean params
    h_mean = _garch11_volatility_numpy(y, omega_mean, alpha_mean, beta_mean)
    z = y / np.sqrt(h_mean)

    # PIT
    if dist.lower() == 'studentt':
        pit = student_t.cdf(y, df=nu_mean, loc=0.0, scale=np.sqrt(h_mean))
    else:
        pit = 0.5 * (1.0 + np.erf(y / (np.sqrt(2.0) * np.sqrt(h_mean))))

    return BayesGARCHResult(
        omega=omega_mean,
        alpha=alpha_mean,
        beta=beta_mean,
        nu=nu_mean,
        trace=trace,
        posterior_means={'omega': omega_mean, 'alpha': alpha_mean, 'beta': beta_mean, 'nu': nu_mean},
        sigma2_mean=h_mean,
        std_resid=z,
        pit=pit,
    )


# -------------------------
# ARIMA + Bayesian GARCH pipeline (grid over ARIMA only)
# -------------------------

def fit_best_marginal(
    y: pd.Series | np.ndarray,
    search_params: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Grid-search ARIMA(p,d,q) for the mean, then fit Bayesian GARCH(1,1) on residuals via HMC.

    Scoring logic:
      - Prefer higher p-value for Ljung-Box on std. residuals (lag=15)
      - and on squared std. residuals
      - Tie-breaker: ARIMA AIC (smaller is better)

    Returns a dict containing ARIMA fit, BayesGARCHResult, standardized residuals, PIT, and diagnostics.
    """
    # defaults (back to ARIMA(0,1,1) - worked before)
    defaults = dict(
        p_list=(1,),       # No AR term
        d_list=(1,),       # First difference  
        q_list=(1,),       # MA(1) term - this worked
        lb_lag=15,
        min_obs=100,
        dist='studentt',
        draws=100,         
        tune=100,          
        target_accept=0.85, 
        chains=1,          
        random_seed=42,
    )
    P = {**defaults, **(search_params or {})}

    y = pd.Series(y).dropna()
    print(f"  Data length after dropna: {len(y)}")
    
    if len(y) < P['min_obs']:
        print(f"  ERROR: Not enough data ({len(y)} < {P['min_obs']})")
        return None
    
    best = None
    models_tried = 0
    models_fitted = 0

    for d in P['d_list']:
        for p in P['p_list']:
            for q in P['q_list']:
                models_tried += 1
                # 1) ARIMA fit  
                try:
                    arima = SARIMAX(
                        y,
                        order=(p, d, q),
                        trend='c',
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                        simple_differencing=True  # Simpler differencing approach
                    ).fit(
                        disp=False, 
                        method='lbfgs', 
                        maxfun=2000,   # Max function evaluations (correct param)
                        pgtol=1e-6,    # Gradient tolerance (correct param)  
                        factr=1e7      # Factor for convergence (correct param)
                    )
                except Exception as e:
                    print(f"    ARIMA({p},{d},{q}) failed: {str(e)[:50]}...")
                    continue

                resid_mean = pd.Series(arima.resid).dropna()
                if resid_mean.size < P['min_obs']:
                    print(f"    ARIMA({p},{d},{q}) residuals too few: {resid_mean.size}")
                    continue

                # 2) Bayesian GARCH(1,1) with HMC
                try:
                    bg = fit_bayes_garch11(
                        resid=resid_mean.values,
                        dist=P['dist'],
                        draws=P['draws'],
                        tune=P['tune'],
                        target_accept=P['target_accept'],
                        chains=P['chains'],
                        random_seed=P['random_seed'],
                    )
                    models_fitted += 1
                    print(f"    ARIMA({p},{d},{q})+GARCH fitted successfully")
                except Exception as e:
                    print(f"    GARCH for ARIMA({p},{d},{q}) failed: {str(e)[:50]}...")
                    continue

                z = bg.std_resid
                p_lbz = _ljungbox_p(z, lag=P['lb_lag'])
                p_lbz2 = _ljungbox_p(z**2, lag=P['lb_lag'])
                p_arch = _archlm_p(z, lags=10)

                cand = {
                    'p': p, 'd': d, 'q': q,
                    'arima_aic': float(arima.aic),
                    'dist': P['dist'],
                    'p_ljungbox': p_lbz,
                    'p_ljungbox_sq': p_lbz2,
                    'p_arch': p_arch,
                    'diagnostic_score': float(np.mean([p_lbz, p_lbz2, p_arch])),
                    'arima': arima,
                    'bayes_garch': bg,
                    'std_resid': z,
                    'pit': bg.pit,
                }

                if best is None:
                    best = cand
                else:
                    better = (cand['p_ljungbox'] > best['p_ljungbox']) and (cand['p_ljungbox_sq'] > best['p_ljungbox_sq'])
                    if better or (
                        np.isclose(cand['p_ljungbox'], best['p_ljungbox']) and
                        np.isclose(cand['p_ljungbox_sq'], best['p_ljungbox_sq']) and
                        cand['arima_aic'] < best['arima_aic']
                    ):
                        best = cand

    print(f"  Models tried: {models_tried}, successfully fitted: {models_fitted}")
    if best is None:
        print("  No models were successfully fitted!")
    else:
        print(f"  Best model: ARIMA({best['p']},{best['d']},{best['q']}) with diagnostic_score={best['diagnostic_score']:.4f}")
    
    return best


def fit_best_marginal_with_diagnostics(y, search_params: Optional[Dict[str, Any]] = None):
    """Wrapper kept for backward compatibility with your previous API."""
    return fit_best_marginal(y, search_params)


# -------------------------
# Batch over a DataFrame
# -------------------------

def grid_search_marginal_model(
    log_returns_scaled: pd.DataFrame,
    search_params: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    rows = []
    models: Dict[str, Dict[str, Any]] = {}

    for ticker in log_returns_scaled.columns:
        print(f"Fitting Bayesian ARIMA+GARCH for {ticker}...")
        
        # Get series (should already have clean RangeIndex since we removed Date column)
        series = log_returns_scaled[ticker]
            
        best = fit_best_marginal_with_diagnostics(series, search_params)
        if best is None:
            print(f"  WARNING: Failed to fit model for {ticker}")
            continue
        row = {
            'Ticker': ticker,
            'p': best['p'], 'd': best['d'], 'q': best['q'],
            'dist': best['dist'],
            'arima_aic': best['arima_aic'],
            'p_ljungbox': best['p_ljungbox'],
            'p_ljungbox_sq': best['p_ljungbox_sq'],
            'p_arch': best['p_arch'],
            'diagnostic_score': best['diagnostic_score'],
        }
        rows.append(row)
        models[ticker] = best

    print(f"Successfully processed {len(rows)} out of {len(log_returns_scaled.columns)} tickers")
    
    if len(rows) == 0:
        print("ERROR: No tickers were successfully processed!")
        return pd.DataFrame(), {}
    
    summary_df = pd.DataFrame(rows).sort_values(['diagnostic_score','arima_aic'], ascending=[False, True]).reset_index(drop=True)
    return summary_df, models


# -------------------------
# Residual diagnostics utility (compatible with previous signature)
# -------------------------

def diagnose_residuals(std_resids: Dict[str, Any], lags: int = 10, plot: bool = False):
    diag_results = {}
    for ticker, val in std_resids.items():
        if isinstance(val, dict) and 'std_resid' in val:
            resid = pd.Series(val['std_resid']).dropna()
        else:
            resid = pd.Series(val).dropna()
        try:
            arch_p = het_arch(resid, nlags=lags)[1]
        except Exception as e:
            arch_p = np.nan
            print(f"{ticker}: ARCH test error: {e}")
        try:
            lb_p = acorr_ljungbox(resid, lags=[lags], return_df=True)['lb_pvalue'].iloc[0]
        except Exception as e:
            lb_p = np.nan
            print(f"{ticker}: Ljung-Box test error: {e}")
        try:
            lb2_p = acorr_ljungbox(resid**2, lags=[lags], return_df=True)['lb_pvalue'].iloc[0]
        except Exception as e:
            lb2_p = np.nan
            print(f"{ticker}: Ljung-Box^2 test error: {e}")
        diag_results[ticker] = {
            'arch_pvalue': float(arch_p) if np.isfinite(arch_p) else np.nan,
            'lb_pvalue': float(lb_p) if np.isfinite(lb_p) else np.nan,
            'lb_pvalue_sq': float(lb2_p) if np.isfinite(lb2_p) else np.nan,
        }
        print(f"{ticker}: ARCH p={diag_results[ticker]['arch_pvalue']:.3f}, "
              f"Ljung-Box p={diag_results[ticker]['lb_pvalue']:.3f}, "
              f"Ljung-Box^2 p={diag_results[ticker]['lb_pvalue_sq']:.3f}")
    return diag_results


# -------------------------
# Main (example)
# -------------------------
if __name__ == "__main__":  # pragma: no cover
    import os, pickle

    garch_dir = "models/garch"
    os.makedirs(garch_dir, exist_ok=True)

    # 1) Load log returns (scaled) - read without date index to avoid warnings
    log_returns = pd.read_csv("data/processed/log_returns.csv")  # Don't use index_col=0
    
    # Drop the Date column and only keep the numeric columns
    if 'Date' in log_returns.columns:
        log_returns = log_returns.drop('Date', axis=1)
        print("Removed Date column to avoid statsmodels DatetimeIndex warnings")
    
    log_returns_scaled = log_returns * 100

    # 2) Grid search
    summary_df, models = grid_search_marginal_model(log_returns_scaled)
    print(f"Completed: {len(summary_df)}/{log_returns.shape[1]} tickers.")
    summary_df.to_csv(os.path.join(garch_dir, "marginal_bayes_summary.csv"), index=False)
    with open(os.path.join(garch_dir, "marginal_bayes_models.pkl"), "wb") as f:
        pickle.dump(models, f)

    # 3) Save standardized residuals & PIT (for downstream steps if needed)
    std_resids_df = pd.DataFrame({k: v['std_resid'] for k, v in models.items()})
    std_resids_df.to_csv(os.path.join(garch_dir, "std_resids.csv"))
    pit_dict = {k: v['pit'] for k, v in models.items()}
    with open(os.path.join(garch_dir, "conditional_pit.pkl"), "wb") as f:
        pickle.dump(pit_dict, f)

    print("Saved std_resids.csv & conditional_pit.pkl to models/garch/")
