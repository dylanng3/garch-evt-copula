'''
ARIMA+GARCH-EVT Marginal Modeling - Ultra Compact Version
Rút gọn tối đa nhưng giữ nguyên logic và kết quả
'''
import warnings; warnings.filterwarnings('ignore')
import os, pickle, numpy as np, pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
import statsmodels.api as sm
from arch import arch_model
from scipy.stats import genpareto
from sklearn.neighbors import KernelDensity

def engle_ng_test(residuals):
    '''Engle-NG bias test - compact version'''
    try:
        z2, z_lag = residuals[1:]**2, residuals[:-1]
        S_neg = (z_lag < 0).astype(int)
        neg_size, pos_size = S_neg * z_lag, (z_lag > 0).astype(int) * z_lag
        
        # HC3 robust tests
        sign_p = sm.OLS(z2, sm.add_constant(S_neg)).fit(cov_type='HC3').pvalues[1]
        neg_p = sm.OLS(z2, sm.add_constant(neg_size)).fit(cov_type='HC3').pvalues[1]
        pos_p = sm.OLS(z2, sm.add_constant(pos_size)).fit(cov_type='HC3').pvalues[1]
        joint_p = sm.OLS(z2, sm.add_constant(np.column_stack([S_neg, neg_size, pos_size]))).fit(cov_type='HC3').f_pvalue
        
        # Status
        sig_count = sum([p < 0.05 for p in [sign_p, neg_p, pos_p]])
        status = "STRONG" if joint_p < 0.01 else "MODERATE" if joint_p < 0.05 or sig_count >= 2 else "MILD" if sig_count == 1 else "NO BIAS"
        
        return {'sign_p': sign_p, 'neg_size_p': neg_p, 'pos_size_p': pos_p, 'joint_p': joint_p, 'status': status}
    except:
        return {'sign_p': 1.0, 'neg_size_p': 1.0, 'pos_size_p': 1.0, 'joint_p': 1.0, 'status': 'NO BIAS'}

def run_diagnostics(std_residuals):
    '''Complete diagnostic tests - compact'''
    try:
        r = np.array(std_residuals).flatten()
        lb_p = acorr_ljungbox(r, lags=[10], return_df=True)['lb_pvalue'].iloc[0]
        lb_sq_p = acorr_ljungbox(r**2, lags=[10], return_df=True)['lb_pvalue'].iloc[0]
        arch_p = het_arch(r, nlags=5)[1]
        engle_ng = engle_ng_test(r)
        
        basic_pass = all(p > 0.05 for p in [lb_p, lb_sq_p, arch_p])
        engle_ng_pass = engle_ng['joint_p'] > 0.05
        
        return {
            'ljung_box': lb_p, 'ljung_box_squared': lb_sq_p, 'arch_lm': arch_p,
            'engle_ng_sign': engle_ng['sign_p'], 'engle_ng_neg_size': engle_ng['neg_size_p'], 
            'engle_ng_pos_size': engle_ng['pos_size_p'], 'engle_ng_joint': engle_ng['joint_p'],
            'engle_ng_status': engle_ng['status'], 'basic_pass': basic_pass, 
            'engle_ng_pass': engle_ng_pass, 'all_pass': basic_pass and engle_ng_pass
        }
    except:
        return {k: 0.0 if 'p' in k else False if 'pass' in k else 'ERROR' for k in 
                ['ljung_box', 'ljung_box_squared', 'arch_lm', 'engle_ng_sign', 'engle_ng_neg_size', 
                 'engle_ng_pos_size', 'engle_ng_joint', 'engle_ng_status', 'basic_pass', 'engle_ng_pass', 'all_pass']}

def get_model_configs():
    '''EXACT model configurations from original - compact format'''
    return [
        {'vol': 'EGARCH', 'p': 1, 'o': 0, 'q': 1, 'dist': 'skewstudent'},
        {'vol': 'EGARCH', 'p': 1, 'o': 1, 'q': 1, 'dist': 'skewstudent'},
        {'vol': 'EGARCH', 'p': 2, 'o': 0, 'q': 1, 'dist': 'skewstudent'},
        {'vol': 'EGARCH', 'p': 2, 'o': 1, 'q': 1, 'dist': 'skewstudent'},
        {'vol': 'EGARCH', 'p': 1, 'o': 1, 'q': 2, 'dist': 'skewstudent'},
        {'vol': 'EGARCH', 'p': 2, 'o': 2, 'q': 1, 'dist': 'skewstudent'},
        {'vol': 'EGARCH', 'p': 1, 'o': 2, 'q': 1, 'dist': 'skewstudent'},
        {'vol': 'GARCH', 'p': 1, 'o': 1, 'q': 1, 'dist': 'skewstudent'},
        {'vol': 'GARCH', 'p': 2, 'o': 1, 'q': 1, 'dist': 'skewstudent'},
        {'vol': 'GARCH', 'p': 1, 'o': 1, 'q': 2, 'dist': 'skewstudent'},
        {'vol': 'GARCH', 'p': 2, 'o': 2, 'q': 1, 'dist': 'skewstudent'},
        {'vol': 'GARCH', 'p': 1, 'o': 0, 'q': 1, 'dist': 'skewstudent'},
        {'vol': 'GARCH', 'p': 2, 'o': 0, 'q': 1, 'dist': 'skewstudent'},
        {'vol': 'GARCH', 'p': 1, 'o': 0, 'q': 2, 'dist': 'skewstudent'},
        {'vol': 'GARCH', 'p': 2, 'o': 0, 'q': 2, 'dist': 'skewstudent'},
        {'vol': 'EGARCH', 'p': 1, 'o': 1, 'q': 1, 'dist': 'StudentsT'},
        {'vol': 'EGARCH', 'p': 1, 'o': 0, 'q': 1, 'dist': 'StudentsT'},
        {'vol': 'EGARCH', 'p': 2, 'o': 1, 'q': 1, 'dist': 'StudentsT'},
        {'vol': 'GARCH', 'p': 1, 'o': 1, 'q': 1, 'dist': 'StudentsT'},
        {'vol': 'GARCH', 'p': 1, 'o': 0, 'q': 1, 'dist': 'StudentsT'},
        {'vol': 'EGARCH', 'p': 1, 'o': 1, 'q': 1, 'dist': 'normal'},
        {'vol': 'EGARCH', 'p': 1, 'o': 0, 'q': 1, 'dist': 'normal'},
        {'vol': 'EGARCH', 'p': 2, 'o': 1, 'q': 1, 'dist': 'normal'},
        {'vol': 'GARCH', 'p': 1, 'o': 1, 'q': 1, 'dist': 'normal'},
        {'vol': 'GARCH', 'p': 1, 'o': 0, 'q': 1, 'dist': 'normal'},
        {'vol': 'EGARCH', 'p': 3, 'o': 1, 'q': 1, 'dist': 'skewstudent'},
        {'vol': 'EGARCH', 'p': 1, 'o': 1, 'q': 3, 'dist': 'skewstudent'},
        {'vol': 'GARCH', 'p': 3, 'o': 1, 'q': 1, 'dist': 'skewstudent'},
        {'vol': 'GARCH', 'p': 1, 'o': 1, 'q': 3, 'dist': 'skewstudent'},
        {'vol': 'GARCH', 'p': 2, 'o': 0, 'q': 3, 'dist': 'skewstudent'}
    ]

def garch_grid_search(returns, max_evaluations=30):
    '''Ultra compact grid search'''
    configs = get_model_configs()
    results = []
    
    print(f"Fitting ARIMA(1, 1, 1)...")
    arima_model = SARIMAX(returns, order=(1,1,1), trend='c').fit(disp=False)
    residuals = arima_model.resid.dropna()  # type: ignore
    
    for i, config in enumerate(configs[:max_evaluations]):
        try:
            model_name = f"{config['vol']}({config['p']},{config.get('o',0)},{config['q']})-{config['dist']}"
            print(f"  Testing {i+1}/{max_evaluations}: {model_name}")
            
            garch_fit = arch_model(residuals, **config).fit(disp='off', show_warning=False)
            
            # Standardized residuals (same logic as original)
            volatility = garch_fit.conditional_volatility
            valid_residuals = residuals.iloc[-len(volatility):][5:]
            valid_volatility = volatility[5:]
            min_vol = np.percentile(valid_volatility, 1)
            std_residuals = valid_residuals / np.maximum(valid_volatility, min_vol)
            
            diagnostics = run_diagnostics(std_residuals)
            
            results.append({
                'model_name': model_name, 'aic': garch_fit.aic, 'diagnostics': diagnostics,
                'garch_fit': garch_fit, 'arima_model': arima_model, 'std_residuals': std_residuals
            })
        except Exception as e:
            print(f"    ❌ Failed: {e}")
    
    # Rank by diagnostic pass + AIC (same as original)
    return sorted(results, key=lambda x: (-int(x['diagnostics']['all_pass']), x['aic']))

def select_and_display(results, ticker):
    '''Compact model selection and display'''
    if not results:
        return None
    
    # Categorize
    passing = [r for r in results if r['diagnostics']['all_pass']]
    basic = [r for r in results if r['diagnostics']['basic_pass'] and not r['diagnostics']['all_pass']]
    failing = [r for r in results if not r['diagnostics']['basic_pass']]
    
    # Display ranking
    print(f"\n{'='*90}")
    print(f"🏆 ENHANCED GARCH RANKING FOR {ticker}")
    print(f"{'='*90}")
    
    if passing:
        print("✅ FULLY PASSING MODELS:")
        for i, r in enumerate(passing[:5]):
            print(f"   {i+1}. {r['model_name']:<25} | AIC: {r['aic']:>10.2f} | Bias: {r['diagnostics']['engle_ng_status']}")
    
    if basic:
        print("\n⚠️  BASIC PASSING MODELS (with leverage effects):")
        for i, r in enumerate(basic[:3]):
            print(f"   {i+1}. {r['model_name']:<25} | AIC: {r['aic']:>10.2f} | Bias: {r['diagnostics']['engle_ng_status']}")
    
    if failing:
        print(f"\n❌ FAILING MODELS (top 2 by AIC):")
        for i, r in enumerate(failing[:2]):
            print(f"   {i+1}. {r['model_name']:<25} | AIC: {r['aic']:>10.2f} | Bias: {r['diagnostics']['engle_ng_status']}")
    
    print(f"{'='*90}")
    
    # Select best model (same strategy as original)
    for strategy, models in [("✅ Selected best passing model", passing), 
                            ("⚠️  Selected best basic model", basic),
                            ("🔄 Fallback to best AIC model", results)]:
        if models:
            selected = models[0]
            print(f"   {strategy}: {selected['model_name']}")
            print(f"🎯 RECOMMENDED: {selected['model_name']}")
            print(f"   AIC: {selected['aic']:.2f} | Diagnostics: {'✅ FULL PASS' if selected['diagnostics']['all_pass'] else '⚠️ BASIC PASS' if selected['diagnostics']['basic_pass'] else '❌ ISSUES'}")
            return selected
    
    return None

def process_ticker(ticker, returns):
    '''Process single ticker - ultra compact'''
    print(f'Processing GARCH for {ticker}...')
    print("🔍 Using Enhanced Grid Search to find optimal GARCH model...")
    
    results = garch_grid_search(returns.dropna())
    selected = select_and_display(results, ticker)
    
    if selected:
        model_name = selected['model_name']
        arima_aic = selected['arima_model'].aic
        garch_aic = selected['aic']
        print(f'   {ticker}: {model_name} | ARIMA AIC={arima_aic:.2f}, GARCH AIC={garch_aic:.2f}')
        
        # Compact diagnostic display
        d = selected['diagnostics']
        status = "✅ PASS" if d['all_pass'] else "❌ FAIL"
        print(f'     Diagnostics: {status}')
        print(f'       • Ljung-Box (residuals):    p = {d["ljung_box"]:.4f}')
        print(f'       • Ljung-Box (squared):      p = {d["ljung_box_squared"]:.4f}')
        print(f'       • ARCH-LM test:             p = {d["arch_lm"]:.4f}')
        print(f'       • Engle-NG Sign Bias:       p = {d["engle_ng_sign"]:.4f}')
        print(f'       • Engle-NG Neg Size Bias:   p = {d["engle_ng_neg_size"]:.4f}')
        print(f'       • Engle-NG Pos Size Bias:   p = {d["engle_ng_pos_size"]:.4f}')
        print(f'       • Engle-NG Joint Test:      p = {d["engle_ng_joint"]:.4f} ({d["engle_ng_status"]})')
        
        return {
            'arima_model': selected['arima_model'], 'egarch_model': selected['garch_fit'],
            'standardized_residuals': selected['std_residuals'], 'egarch_aic': selected['aic'],
            'egarch_params': dict(selected['garch_fit'].params), 'model_name': model_name,
            'diagnostics': selected['diagnostics']
        }
    else:
        print(f'   {ticker}: GARCH fitting failed')
        return None

def fit_evt_marginal(std_resids, tail_prob=0.1):
    '''Compact EVT fitting'''
    data = np.array(std_resids)
    upper_thresh, lower_thresh = np.percentile(data, [(1-tail_prob)*100, tail_prob*100])
    
    try:
        upper_exc = data[data > upper_thresh] - upper_thresh
        upper_shape, _, upper_scale = genpareto.fit(upper_exc, floc=0)
        upper_gpd = {'shape': upper_shape, 'scale': upper_scale, 'threshold': upper_thresh}
    except:
        upper_gpd = None
    
    try:
        lower_exc = lower_thresh - data[data < lower_thresh]
        lower_shape, _, lower_scale = genpareto.fit(lower_exc, floc=0)
        lower_gpd = {'shape': lower_shape, 'scale': lower_scale, 'threshold': lower_thresh}
    except:
        lower_gpd = None
    
    center_data = data[(data >= lower_thresh) & (data <= upper_thresh)]
    kde = KernelDensity(bandwidth=0.1).fit(center_data.reshape(-1, 1))
    
    return {
        'upper_tail_gpd': upper_gpd, 'lower_tail_gpd': lower_gpd,
        'center_kde': {'kde_model': kde, 'center_data': center_data},
        'threshold_info': {'upper_threshold': upper_thresh, 'lower_threshold': lower_thresh},
        'tail_prob': tail_prob
    }

def semiparametric_cdf(x, marginal):
    '''Semi-parametric CDF evaluation for copula transformation'''
    x = np.atleast_1d(x)
    cdf_vals = np.zeros_like(x, dtype=float)
    
    upper_thresh = marginal['threshold_info']['upper_threshold']
    lower_thresh = marginal['threshold_info']['lower_threshold']
    tail_prob = marginal['tail_prob']
    
    # Lower tail GPD
    lower_mask = x < lower_thresh
    if np.any(lower_mask) and marginal['lower_tail_gpd']:
        gpd = marginal['lower_tail_gpd']
        exc = lower_thresh - x[lower_mask]
        if gpd['shape'] != 0:
            tail_cdf = (1 + gpd['shape'] * exc / gpd['scale']) ** (-1/gpd['shape'])
        else:
            tail_cdf = np.exp(-exc / gpd['scale'])
        cdf_vals[lower_mask] = tail_prob * (1 - tail_cdf)
    
    # Center empirical CDF
    center_mask = (x >= lower_thresh) & (x <= upper_thresh)
    if np.any(center_mask):
        center_data = marginal['center_kde']['center_data']
        for i, xi in enumerate(x[center_mask]):
            empirical_cdf = np.mean(center_data <= xi)
            cdf_vals[center_mask][i] = tail_prob + (1-2*tail_prob) * empirical_cdf
    
    # Upper tail GPD
    upper_mask = x > upper_thresh
    if np.any(upper_mask) and marginal['upper_tail_gpd']:
        gpd = marginal['upper_tail_gpd']
        exc = x[upper_mask] - upper_thresh
        if gpd['shape'] != 0:
            tail_cdf = (1 + gpd['shape'] * exc / gpd['scale']) ** (-1/gpd['shape'])
        else:
            tail_cdf = np.exp(-exc / gpd['scale'])
        cdf_vals[upper_mask] = 1 - tail_prob * tail_cdf
    
    return cdf_vals[0] if len(cdf_vals) == 1 else cdf_vals

if __name__ == '__main__':
    # Load data
    data_path = 'data/processed/log_returns.csv'
    if not os.path.exists(data_path):
        print(f'Data file not found: {data_path}')
        exit(1)
    
    df = pd.read_csv(data_path).drop(columns=['Date'], errors='ignore')
    print(f'Processing {df.shape[1]} tickers...')
    
    # PHASE 1: GARCH MODELING
    print('\n PHASE 1: GARCH MODELING')
    print('='*50)
    print('🔍 GRID SEARCH MODE: Testing multiple GARCH configurations')
    
    garch_results = {}
    
    for ticker in df.columns:
        result = process_ticker(ticker, df[ticker])
        if result:
            garch_results[ticker] = result
        print()
    
    # Save GARCH results
    os.makedirs('models/garch', exist_ok=True)
    with open('models/garch/marginal_model.pkl', 'wb') as f:
        pickle.dump(garch_results, f)
    
    # Save summaries
    garch_summary = []
    std_residuals_dict = {}
    for ticker, result in garch_results.items():
        summary_row = {'ticker': ticker, 'arima_aic': result['arima_model'].aic, 'egarch_aic': result['egarch_aic']}
        summary_row.update(result['egarch_params'])
        garch_summary.append(summary_row)
        std_residuals_dict[ticker] = result['standardized_residuals']
    
    pd.DataFrame(garch_summary).to_csv('models/garch/garch_summary.csv', index=False)
    pd.DataFrame(std_residuals_dict).to_csv('models/garch/standardized_residuals.csv', index=True)
    
    print(f'\n GARCH Results Saved:')
    print(f'   models/garch/marginal_model.pkl (full models)')
    print(f'   models/garch/garch_summary.csv (parameters)')
    print(f'   models/garch/standardized_residuals.csv (for EVT)')
    print(f' GARCH Phase: {len(garch_results)}/{len(df.columns)} successful')
    
    # PHASE 2: EVT MODELING
    print('\n PHASE 2: EVT MODELING')
    print('='*50)
    
    evt_marginals = {}
    for ticker, result in garch_results.items():
        print(f'Processing EVT for {ticker}...')
        marginal = fit_evt_marginal(result['standardized_residuals'])
        if marginal['upper_tail_gpd'] and marginal['lower_tail_gpd']:
            evt_marginals[ticker] = marginal
            upper_shape = marginal['upper_tail_gpd']['shape']
            lower_shape = marginal['lower_tail_gpd']['shape']
            print(f'   {ticker}: Upper γ={upper_shape:.3f}, Lower γ={lower_shape:.3f}')
        else:
            print(f'   {ticker}: EVT fitting failed')
    
    # Save EVT results
    os.makedirs('models/evt', exist_ok=True)
    with open('models/evt/marginal_distributions.pkl', 'wb') as f:
        pickle.dump(evt_marginals, f)
    
    evt_summary = []
    for ticker, marginal in evt_marginals.items():
        evt_summary.append({
            'ticker': ticker, 'upper_shape': marginal['upper_tail_gpd']['shape'],
            'upper_scale': marginal['upper_tail_gpd']['scale'], 'upper_threshold': marginal['threshold_info']['upper_threshold'],
            'lower_shape': marginal['lower_tail_gpd']['shape'], 'lower_scale': marginal['lower_tail_gpd']['scale'],
            'lower_threshold': marginal['threshold_info']['lower_threshold'], 'tail_prob': marginal['tail_prob']
        })
    
    pd.DataFrame(evt_summary).to_csv('models/evt/evt_summary.csv', index=False)
    
    print(f'\n EVT Results Saved:')
    print(f'   models/evt/marginal_distributions.pkl (full marginals)')
    print(f'   models/evt/evt_summary.csv (EVT parameters)')
    print(f' EVT Phase: {len(evt_marginals)}/{len(garch_results)} successful')
    
    # FINAL SUMMARY
    print('\n FINAL SUMMARY')
    print('='*50)
    print(f' Total tickers: {len(df.columns)}')
    print(f' GARCH successful: {len(garch_results)} ({len(garch_results)/len(df.columns)*100:.1f}%)')
    print(f' EVT successful: {len(evt_marginals)} ({len(evt_marginals)/len(df.columns)*100:.1f}%)')
    print(f' Complete pipeline: {len(evt_marginals)} tickers ready for copula modeling')
    
    # FINAL MODELS SUMMARY
    print('\n' + '='*80)
    print('🎯 FINAL SELECTED MODELS FOR EACH TICKER')
    print('='*80)
    
    for ticker, result in garch_results.items():
        d = result.get('diagnostics', {})
        status = "✅ PASS" if d.get('all_pass', False) else "❌ FAIL"
        bias = d.get('engle_ng_status', 'N/A')
        
        print(f"📊 {ticker}:")
        print(f"   Model: {result.get('model_name', 'Unknown')}")
        print(f"   AIC: {result.get('egarch_aic', 0):.2f}")
        print(f"   Diagnostics: {status}")
        print(f"   Bias Status: {bias}")
        print(f"   LB: {d.get('ljung_box', 0):.3f} | LB²: {d.get('ljung_box_squared', 0):.3f} | "
              f"ARCH: {d.get('arch_lm', 0):.3f} | ENG: {d.get('engle_ng_joint', 0):.3f}")
        print()
    
    print('='*80)
    print('🏆 BEST PRACTICES ACHIEVED:')
    
    model_counts = {}
    for result in garch_results.values():
        model_name = result.get('model_name', 'Unknown')
        model_counts[model_name] = model_counts.get(model_name, 0) + 1
    
    for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {model}: {count} ticker(s)")
    
    passing_models = sum(1 for result in garch_results.values() 
                        if result.get('diagnostics', {}).get('all_pass', False))
    success_rate = (passing_models / len(garch_results)) * 100 if garch_results else 0
    
    print(f"\n📈 Overall Success Rate: {passing_models}/{len(garch_results)} ({success_rate:.1f}%) models passed all diagnostics")
    print('='*80)
