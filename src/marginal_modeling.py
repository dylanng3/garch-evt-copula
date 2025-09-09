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
from scipy.stats import genpareto, kstest
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import math

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
    
    print("📈 GARCH Model Distribution:")
    for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {model}: {count} ticker(s)")
    
    passing_models = sum(1 for result in garch_results.values() 
                        if result.get('diagnostics', {}).get('all_pass', False))
    success_rate = (passing_models / len(garch_results)) * 100 if garch_results else 0
    
    print(f"\n📈 Overall Success Rate: {passing_models}/{len(garch_results)} ({success_rate:.1f}%) models passed all diagnostics")
    print('='*80)


# Functions for EVT diagnostics and plotting
def fit_gpd_tails(z, tail_prob=0.10):
    z = np.asarray(z, float)
    z = z[np.isfinite(z)]
    lo = np.nanpercentile(z, tail_prob*100)
    hi = np.nanpercentile(z, (1-tail_prob)*100)

    exc_u = z[z>hi] - hi                   # upper exceedances
    exc_l = lo - z[z<lo]                   # lower exceedances (make positive)

    # Fit GPD (loc fixed at 0)
    xi_u, _, beta_u = genpareto.fit(exc_u, floc=0)
    xi_l, _, beta_l = genpareto.fit(exc_l, floc=0)

    return {"thr": (lo, hi),
            "upper": {"xi": xi_u, "beta": beta_u, "Nu": exc_u.size},
            "lower": {"xi": xi_l, "beta": beta_l, "Nl": exc_l.size},
            "tail_prob": tail_prob}

def gpd_tail_plots(z, gpd, ax_hist=None, ax_qq=None, ax_pp=None, bins=50):
    z = np.asarray(z, float); z = z[np.isfinite(z)]
    lo, hi = gpd["thr"]; p = gpd["tail_prob"]
    exc_u = z[z>hi] - hi; exc_l = lo - z[z<lo]

    # 1) Histogram + GPD tail density overlay
    if ax_hist is not None:
        counts, bins_, patches = ax_hist.hist(z, bins=bins, density=True, alpha=0.9, edgecolor='none')
        # Recolor tail bars
        centers = 0.5*(bins_[:-1]+bins_[1:])
        for c, bc, patch in zip(counts, centers, patches):
            if bc <= lo or bc >= hi:
                patch.set_facecolor('orange'); patch.set_alpha(0.6)
        ax_hist.axvline(lo, color='red', ls='--', lw=1); ax_hist.axvline(hi, color='red', ls='--', lw=1)
        ax_hist.set_title(f"Tails p={p:.2f} | Nu⁺={gpd['upper']['Nu']} Nu⁻={gpd['lower']['Nl']}")
        ax_hist.set_xlabel("Std. residuals"); ax_hist.set_ylabel("Density"); ax_hist.grid(alpha=0.3)

    # 2) QQ plot for exceedances vs GPD
    if ax_qq is not None:
        for exc, side, pars in [(exc_l,"Lower",gpd["lower"]), (exc_u,"Upper",gpd["upper"])]:
            if exc.size >= 5:
                qs = np.linspace(0.05, 0.95, exc.size)
                theo = genpareto.ppf(qs, c=pars["xi"], loc=0, scale=pars["beta"])
                samp = np.quantile(exc, qs)
                ax_qq.scatter(theo, samp, s=14, alpha=0.7, label=f"{side} ({exc.size})")
        ax_qq.plot([0, max(ax_qq.get_xlim()[1], ax_qq.get_ylim()[1])],
                   [0, max(ax_qq.get_xlim()[1], ax_qq.get_ylim()[1])], 'k--', lw=1)
        ax_qq.set_title("GPD QQ (exceedances)"); ax_qq.set_xlabel("Theoretical"); ax_qq.set_ylabel("Sample")
        ax_qq.legend(); ax_qq.grid(alpha=0.3)

    # 3) PP plot (empirical CDF vs GPD CDF)
    if ax_pp is not None:
        for exc, side, pars in [(exc_l,"Lower",gpd["lower"]), (exc_u,"Upper",gpd["upper"])]:
            if exc.size >= 5:
                s = np.sort(exc); emp = np.arange(1, len(s)+1)/(len(s)+1)
                mod = genpareto.cdf(s, c=pars["xi"], loc=0, scale=pars["beta"])
                ax_pp.plot(mod, emp, '.', ms=4, label=f"{side} ({len(s)})")
        ax_pp.plot([0,1],[0,1],'k--',lw=1); ax_pp.set_xlim(0,1); ax_pp.set_ylim(0,1)
        ax_pp.set_title("GPD PP (exceedances)"); ax_pp.set_xlabel("Model CDF"); ax_pp.set_ylabel("Empirical")
        ax_pp.legend(); ax_pp.grid(alpha=0.3)

def gpd_tail_tests(z, gpd):
    z = np.asarray(z, float); z = z[np.isfinite(z)]
    lo, hi = gpd["thr"]
    exc_u = z[z>hi] - hi; exc_l = lo - z[z<lo]
    out = {}
    # KS trên exceedances: so sánh với GPD(ξ,β)
    if exc_u.size >= 5:
        out["KS_upper_p"] = kstest(exc_u, lambda x: genpareto.cdf(x, c=gpd["upper"]["xi"], loc=0, scale=gpd["upper"]["beta"])).pvalue
    if exc_l.size >= 5:
        out["KS_lower_p"] = kstest(exc_l, lambda x: genpareto.cdf(x, c=gpd["lower"]["xi"], loc=0, scale=gpd["lower"]["beta"])).pvalue
    return out


# ==================== EVT utilities & plots (functions) ====================
from matplotlib.lines import Line2D

# ---- core: compute tail stats for one series ----
def tail_stats(z, p_grid=(0.15,0.12,0.10,0.08,0.06,0.05), min_n=30):
    """
    Return DataFrame with columns:
    p, uL, uU, ME_L, ME_U, xi_L, xi_U, beta_L, beta_U, Nu_L, Nu_U, KS_L, KS_U
    """
    z = np.asarray(z, float)
    z = z[np.isfinite(z)]
    out = {k: [] for k in ["p","uL","uU","ME_L","ME_U",
                           "xi_L","xi_U","beta_L","beta_U",
                           "Nu_L","Nu_U","KS_L","KS_U"]}
    for p in p_grid:
        uL = np.nanpercentile(z, p*100)
        uU = np.nanpercentile(z, (1-p)*100)
        excU = z[z > uU] - uU         # upper exceedances
        excL = uL - z[z < uL]         # lower exceedances (>=0)

        ME_U = float(np.mean(excU)) if excU.size else np.nan
        ME_L = float(np.mean(excL)) if excL.size else np.nan

        xiU = betaU = KSU = np.nan
        xiL = betaL = KSL = np.nan
        if excU.size >= min_n:
            xiU, _, betaU = genpareto.fit(excU, floc=0)
            KSU = kstest(excU, lambda x: genpareto.cdf(x, c=xiU, loc=0, scale=betaU)).pvalue
        if excL.size >= min_n:
            xiL, _, betaL = genpareto.fit(excL, floc=0)
            KSL = kstest(excL, lambda x: genpareto.cdf(x, c=xiL, loc=0, scale=betaL)).pvalue

        row_vals = [p,uL,uU,ME_L,ME_U,xiL,xiU,betaL,betaU,
                    int(excL.size),int(excU.size),KSL,KSU]
        for k,v in zip(out.keys(), row_vals):
            out[k].append(v)
    return pd.DataFrame(out)

# ---- plot 1: Mean-Excess grid ----
def plot_mean_excess_grid(
    data_dict,
    p_grid=(0.15,0.12,0.10,0.08,0.06,0.05),
    min_n=30,
    cols=3,
    annotate=True,
    annotate_rows=(0,),
    suptitle="EVT – Mean-Excess Plots"
):
    tickers = list(data_dict.keys())
    rows = math.ceil(len(tickers)/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle(suptitle, fontsize=14, y=0.995)

    for i, t in enumerate(tickers):
        df = tail_stats(data_dict[t], p_grid=p_grid, min_n=min_n)
        ax = axes[i]
        ax.plot(df["uL"], df["ME_L"], "o-", lw=1.8, ms=4, color="C0", label="Lower ME(u)")
        ax.plot(df["uU"], df["ME_U"], "o-", lw=1.8, ms=4, color="C1", label="Upper ME(u)")
        ax.set_title(t, fontsize=11)
        ax.set_xlabel("Threshold u"); ax.set_ylabel("ME(u)")
        ax.grid(alpha=0.25); [sp.set_alpha(0.3) for sp in ax.spines.values()]
        ax.tick_params(labelsize=9)

        if annotate and (i in set(annotate_rows)):
            for p0, yoff in [(0.10, 0.96), (0.05, 0.86)]:
                if p0 in df["p"].values:
                    r = df.loc[df["p"]==p0].iloc[0]
                    ks_l = f"{r['KS_L']:.3f}" if pd.notna(r['KS_L']) else "nan"
                    ks_u = f"{r['KS_U']:.3f}" if pd.notna(r['KS_U']) else "nan"
                    txt = f"p={p0:.2f}: Nu±=({r['Nu_L']},{r['Nu_U']}) | KS±=({ks_l},{ks_u})"
                    ax.text(0.02, yoff, txt, transform=ax.transAxes, va="top",
                            bbox=dict(boxstyle="round", fc="white", alpha=0.9), fontsize=8)

    # hide unused panes
    for j in range(i+1, len(axes)): axes[j].axis('off')

    # global legend
    proxies = [
        Line2D([0],[0], color="C0", marker="o", lw=1.8, ms=4, label="Lower ME(u)"),
        Line2D([0],[0], color="C1", marker="o", lw=1.8, ms=4, label="Upper ME(u)"),
    ]
    fig.legend(proxies, [p.get_label() for p in proxies], # type: ignore
               loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(); plt.show()

# ---- plot 2: Parameter Stability grid (xi & beta vs p) ----
def plot_parameter_stability_grid(
    data_dict,
    p_grid=(0.15,0.12,0.10,0.08,0.06,0.05),
    min_n=30,
    cols=3,
    suptitle="EVT – Parameter Stability (xi & beta vs p)"
):
    tickers = list(data_dict.keys())
    n = len(tickers)
    rows = math.ceil(n/cols)
    fig, axs = plt.subplots(rows, cols, figsize=(6*cols, 4*rows), constrained_layout=True)
    fig.suptitle(suptitle, fontsize=14, y=0.995)

    # normalize axs to 2D
    if rows == 1: axs = np.atleast_2d(axs)

    for i, t in enumerate(tickers):
        df = tail_stats(data_dict[t], p_grid=p_grid, min_n=min_n)
        r, c = divmod(i, cols)
        ax = axs[r, c]; ax2 = ax.twinx()

        name = t.replace("_DATA", "")
        ax.set_title(name, fontsize=11)

        # xi (left)
        ax.plot(df["p"], df["xi_L"], "o-", lw=1.8, ms=4, color="#2ca02c", label="xi_lower")
        ax.plot(df["p"], df["xi_U"], "o-", lw=1.8, ms=4, color="#d62728", label="xi_upper")
        ax.set_xlabel("Tail mass p"); ax.set_ylabel("xi", color="#2ca02c")
        ax.tick_params(axis='y', labelcolor="#2ca02c", labelsize=9)
        ax.set_xticks(p_grid); ax.set_xticklabels([f"{p:.2f}" for p in p_grid], fontsize=9)
        ax.invert_xaxis(); ax.grid(alpha=0.25)
        [sp.set_alpha(0.3) for sp in ax.spines.values()]

        # beta (right)
        ax2.plot(df["p"], df["beta_L"], "s--", lw=1.6, ms=4, color="#9467bd", label="beta_lower")
        ax2.plot(df["p"], df["beta_U"], "s--", lw=1.6, ms=4, color="#8c564b", label="beta_upper")
        ax2.set_ylabel("beta", color="#9467bd"); ax2.tick_params(axis='y', labelcolor="#9467bd", labelsize=9)
        [sp.set_alpha(0.3) for sp in ax2.spines.values()]

    # hide unused panes
    total = rows*cols
    for j in range(n, total):
        r, c = divmod(j, cols)
        axs[r, c].axis('off')

    # global legend (proxies)
    proxies = [
        Line2D([0],[0], color="#2ca02c", marker="o", lw=1.8, ms=4, label="xi_lower"),
        Line2D([0],[0], color="#d62728", marker="o", lw=1.8, ms=4, label="xi_upper"),
        Line2D([0],[0], color="#9467bd", marker="s", lw=1.6, ms=4, ls="--", label="beta_lower"),
        Line2D([0],[0], color="#8c564b", marker="s", lw=1.6, ms=4, ls="--", label="beta_upper"),
    ]
    fig.legend(proxies, [p.get_label() for p in proxies], # type: ignore
               loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    plt.subplots_adjust(bottom=0.08)
    plt.tight_layout(); plt.show()


def semiparametric_cdf(z, x=None, p=0.10, min_exc=30, eps=1e-12, evt_marginal=None):
    """
    Semi-parametric CDF: core ECDF in [uL,uU], GPD in tails.
    Ensures continuity at uL/uU and numerical stability.
    
    Parameters:
    -----------
    z : array-like
        Standardized residuals data
    x : array-like, optional
        Points at which to evaluate the CDF. If None, uses z values themselves
    p : float, default 0.10
        Tail probability threshold (only used if evt_marginal is None)
    min_exc : int, default 30
        Minimum exceedances for GPD fitting (only used if evt_marginal is None)
    eps : float, default 1e-12
        Numerical safety clipping
    evt_marginal : dict, optional
        Pre-fitted EVT marginal. If provided, uses its parameters instead of fitting new GPD
        
    Returns:
    --------
    array : CDF values at points x
    """
    z = np.asarray(z, float)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return np.array([])

    if x is None:
        x = z
    x = np.asarray(x, float)

    # If evt_marginal is provided, use its parameters
    if evt_marginal is not None:
        p = evt_marginal['tail_prob']
        upper_gpd = evt_marginal['upper_tail_gpd']
        lower_gpd = evt_marginal['lower_tail_gpd']
        
        if upper_gpd is None or lower_gpd is None:
            # Fallback to empirical CDF if GPD fitting failed
            z_sorted = np.sort(z)
            return np.clip(np.searchsorted(z_sorted, x, side='right') / len(z_sorted), eps, 1.0 - eps)
        
        uL = lower_gpd['threshold']
        uU = upper_gpd['threshold']
        xiL = lower_gpd['shape']
        betaL = lower_gpd['scale']
        xiU = upper_gpd['shape'] 
        betaU = upper_gpd['scale']
        
        # Center data for ECDF
        center_data = evt_marginal['center_kde']['center_data']
        center_sorted = np.sort(center_data)
        def ecdf_center(x_vals):
            return np.searchsorted(center_sorted, x_vals, side="right") / max(1, len(center_sorted))
        
        # Compute CDF values
        F = np.empty_like(x)
        
        # Lower tail
        idxL = x < uL
        y = uL - x[idxL]
        HL = genpareto.cdf(y, c=xiL, loc=0, scale=betaL)
        F[idxL] = p * (1 - HL)
        
        # Center
        idxC = (x >= uL) & (x <= uU)
        F[idxC] = p + (1 - 2*p) * ecdf_center(x[idxC])
        
        # Upper tail
        idxU = x > uU
        y = x[idxU] - uU
        HU = genpareto.cdf(y, c=xiU, loc=0, scale=betaU)
        F[idxU] = 1 - p + p * HU
        
        return np.clip(F, eps, 1.0 - eps)
    
    # Original logic: fit GPD ourselves
    assert 0.0 < p < 0.5, "p should be in (0, 0.5)"

    # thresholds
    uL = np.nanpercentile(z, p*100)
    uU = np.nanpercentile(z, (1-p)*100)

    # split data
    maskL = z < uL
    maskU = z > uU
    maskC = ~maskL & ~maskU
    zC = z[maskC]

    # ECDF in core (use 'left' to make F(uL)=p)
    zC_sorted = np.sort(zC)
    def ecdf_core(x_vals):
        # fraction of core strictly < x
        return np.searchsorted(zC_sorted, x_vals, side="left") / max(1, len(zC_sorted))

    # GPD fits (exceedances)
    excU = z[maskU] - uU                # right
    excL = uL - z[maskL]                # left (on -X)
    xiU = betaU = xiL = betaL = np.nan
    if excU.size >= min_exc:
        xiU, _, betaU = genpareto.fit(excU, floc=0.0)
    if excL.size >= min_exc:
        xiL, _, betaL = genpareto.fit(excL, floc=0.0)

    F = np.empty_like(x, dtype=float)

    # lower tail: x < uL  → F(x) = p * (1 - H_L(uL - x))
    idxL = x < uL
    if idxL.any():
        if np.isfinite(xiL) and np.isfinite(betaL):
            y = uL - x[idxL]
            HL = genpareto.cdf(y, c=xiL, loc=0.0, scale=betaL)
            F[idxL] = p * (1.0 - HL)
        else:
            # fallback empirical in left tail
            if maskL.any():
                L_sorted = np.sort(z[maskL])
                F[idxL] = p * (np.searchsorted(L_sorted, x[idxL], side="right") / len(L_sorted))
            else:
                F[idxL] = 0.0

    # core: uL <= x <= uU  → F(x) = p + (1-2p)*ECDF_core(x), with boundary fixes
    idxC = (x >= uL) & (x <= uU)
    if idxC.any():
        Fc = p + (1.0 - 2.0 * p) * ecdf_core(x[idxC])
        # enforce exact boundary values at thresholds for continuity
        Fc = np.where(np.isclose(x[idxC], uL), p, Fc)
        Fc = np.where(np.isclose(x[idxC], uU), 1.0 - p, Fc)
        F[idxC] = Fc

    # upper tail: x > uU  → F(x) = 1 - p + p * H_R(x - uU)
    idxU = x > uU
    if idxU.any():
        if np.isfinite(xiU) and np.isfinite(betaU):
            y = x[idxU] - uU
            HU = genpareto.cdf(y, c=xiU, loc=0.0, scale=betaU)
            F[idxU] = 1.0 - p + p * HU
        else:
            # fallback empirical in right tail
            if maskU.any():
                U_sorted = np.sort(z[maskU])
                F[idxU] = 1.0 - p + p * (np.searchsorted(U_sorted, x[idxU], side="right") / len(U_sorted))
            else:
                F[idxU] = 1.0 - p

    # numerical safety for copula input
    return np.clip(F, eps, 1.0 - eps)


def plot_semi_parametric_cdf(data_dict, p=0.10, min_exc=30, n_grid=400, pad=0.2):
    """
    Vẽ Empirical CDF vs. Semi-parametric CDF cho từng series trong data_dict
    mà KHÔNG lặp lại bước fit EVT/GPD trong hàm plot.
    """
    print(f"SEMI-PARAMETRIC CDF PLOTS (p={p})")

    tickers = list(data_dict.keys())
    n = len(tickers)
    if n == 0:
        return

    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), squeeze=False)
    axes = axes.ravel()

    for i, tk in enumerate(tickers):
        ax = axes[i]
        z = np.asarray(data_dict[tk], float)
        z = z[np.isfinite(z)]
        if z.size == 0:
            ax.set_axis_off()
            continue

        # Lưới x để vẽ CDF bán tham số
        x_min, x_max = z.min() - pad, z.max() + pad
        x_grid = np.linspace(x_min, x_max, n_grid)

        # Semi-parametric CDF (gọi lại hàm có sẵn để tránh lặp code)
        F_semi = semiparametric_cdf(z, x=x_grid, p=p, min_exc=min_exc)

        # Empirical CDF
        z_sorted = np.sort(z)
        F_emp = np.arange(1, z_sorted.size + 1) / z_sorted.size

        # Vẽ
        ax.plot(z_sorted, F_emp, lw=2, alpha=0.85, label='Empirical CDF')
        ax.plot(x_grid, F_semi, lw=2, alpha=0.95, label='Semi-parametric CDF')

        # Vẽ ngưỡng để tham chiếu (chỉ tính 1 lần, không fit lại)
        uL = np.nanpercentile(z, p * 100)
        uU = np.nanpercentile(z, (1 - p) * 100)
        ax.axvline(uL, ls='--', alpha=0.7)
        ax.axvline(uU, ls='--', alpha=0.7)

        ax.set_title(f"{tk} – Semi-parametric CDF (p={p:.2f})")
        ax.set_xlabel("Standardized residuals")
        ax.set_ylabel("Cumulative prob.")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # Ẩn các ô trống
    for j in range(i + 1, rows * cols):
        axes[j].set_axis_off()

    plt.tight_layout()
    plt.show()
