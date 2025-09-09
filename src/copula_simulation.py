import numpy as np
import pandas as pd
import pickle
import pyvinecopulib as pv
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kstest
from typing import Tuple, Dict
from marginal_modeling import semiparametric_cdf


def compute_pit(
    std_resids_path: str = '../models/garch/standardized_residuals.csv',
    evt_marginals_path: str = '../models/evt/marginal_distributions.pkl',
    clip_eps: float = 1e-12
) -> Tuple[pd.DataFrame, Dict[str, dict]]:

    # 1) Load standardized residuals (giữ các hàng full-case để dùng chung cho copula)
    std_resids_df = pd.read_csv(std_resids_path, index_col=0)
    std_resids_clean = (
        std_resids_df
        .replace([np.inf, -np.inf], np.nan)
        .dropna(how="any")
    )

    # 2) Load EVT marginals
    with open(evt_marginals_path, 'rb') as f:
        evt_marginals: Dict[str, dict] = pickle.load(f)

    all_cols = list(std_resids_clean.columns)
    have_marg = [c for c in all_cols if c in evt_marginals]
    missing = [c for c in all_cols if c not in evt_marginals]

    print(f"Standardized residuals shape: {std_resids_clean.shape}")
    print(f"Loaded EVT marginals: {len(evt_marginals)} keys")
    if missing:
        print(f"Missing EVT marginal for: {missing}")

    # 3) Compute PIT cho từng cột có marginal
    pit = {}
    for ticker in have_marg:
        x = std_resids_clean[ticker].values.astype(float)
        u = semiparametric_cdf(x, evt_marginal=evt_marginals[ticker])  # sử dụng tham số evt_marginal
        u = np.clip(u, clip_eps, 1.0 - clip_eps)
        pit[ticker] = u

    pit_df = pd.DataFrame(pit, index=std_resids_clean.index)
    # Giữ đúng thứ tự cột như standardized_residuals (cột thiếu sẽ bị lược bỏ)
    pit_df = pit_df.reindex(columns=have_marg)

    # 4) Kiểm tra nhanh
    if not pit_df.empty:
        u_min, u_max = float(pit_df.min().min()), float(pit_df.max().max())
        print(f"PIT shape: {pit_df.shape} | range after clip: [{u_min:.3g}, {u_max:.3g}]")

    return pit_df, evt_marginals


def validate_pit(pit: pd.DataFrame, eps=1e-6):
    T, d = pit.shape
    print(f"PIT shape: {T} x {d}")
    
    # Check for invalid values
    if pit.isna().sum().sum() or np.isinf(pit.values).sum():
        print("❌ PIT có NaN/Inf"); return {'ok': False}
    
    min_val, max_val = float(pit.min().min()), float(pit.max().max())
    print(f"Range: [{min_val:.6f}, {max_val:.6f}]")
    if not (0 < min_val and max_val < 1):
        print("❌ Có giá trị = 0 hoặc 1"); return {'ok': False}
    
    # Edge rate check
    edge_rate = float(((pit <= eps*10) | (pit >= 1-eps*10)).mean().mean())
    print(f"Edge rate: {edge_rate:.3%}")
    
    # KS tests
    ks_passed = sum(kstest(pit[c].values, 'uniform')[1] >= 0.01 for c in pit.columns)
    print(f"KS test: {ks_passed}/{d} passed (p≥0.01)")
    
    return {'ok': True, 'edge_rate': edge_rate}


def fit_copula(std_resids_path, out_path='models/copula/best_copula.json'):
    """
    Fit R-vine copula từ standardized residuals và lưu model ra file json.
    """
    df = pd.read_csv(std_resids_path, index_col=0)
    # Load conditional PIT
    pit_df = pd.read_csv('models/copula/pit_data.csv', index_col=0)
    pit_df = pit_df[df.columns]  # Ensure column order matches
    u = pit_df.values
    import pyvinecopulib as pv
    family_set = [
        getattr(pv.BicopFamily, 'gaussian', 1),
        getattr(pv.BicopFamily, 'student', 2),
        getattr(pv.BicopFamily, 'clayton', 3),
        getattr(pv.BicopFamily, 'gumbel', 4),
        getattr(pv.BicopFamily, 'frank', 5)
    ]
    controls = pv.FitControlsVinecop(family_set=family_set)
    vine = pv.Vinecop(u.shape[1])
    vine.select(u, controls=controls)
    out_json = out_path if out_path.endswith('.json') else out_path.rsplit('.', 1)[0] + '.json'
    json_str = vine.to_json()
    with open(out_json, 'w', encoding='utf-8') as f:
        f.write(json_str)
    print(f"Fitted R-vine copula (pyvinecopulib) and saved to {out_json}")


def hybrid_ppf(u, marginal, eps=1e-12):
    """
    Semi-parametric PPF (quantile) khớp với cách fit của bạn:
      - Core: nội suy tuyến tính trên center_data
      - Left tail (x < uL):  x = uL - GPD_ppf_survival(alpha_L),  alpha_L = q / p
      - Right tail (x > uU): x = uU + GPD_ppf_survival(alpha_R),  alpha_R = (1-q) / p
    Trong đó p = tail_prob, uL/uU từ threshold_info.
    """
    q = np.clip(np.asarray(u, float), eps, 1.0 - eps)

    # lấy tham số biên
    p   = float(marginal['tail_prob'])
    uL  = float(marginal['threshold_info']['lower_threshold'])
    uU  = float(marginal['threshold_info']['upper_threshold'])

    xiL   = float(marginal['lower_tail_gpd']['shape'])
    betaL = float(marginal['lower_tail_gpd']['scale'])
    xiR   = float(marginal['upper_tail_gpd']['shape'])
    betaR = float(marginal['upper_tail_gpd']['scale'])

    core  = np.sort(np.asarray(marginal['center_kde']['center_data'], float))
    nC    = core.size
    if nC < 2:
        raise ValueError("center_data quá ít điểm để nội suy PPF.")

    def gpd_ppf_survival(alpha, xi, beta):
        a = np.clip(alpha, eps, 1.0 - eps)
        if abs(xi) < 1e-10:
            return -beta * np.log(a)
        return beta * (a**(-xi) - 1.0) / xi

    out = np.empty_like(q)

    # ---- Left tail: q < p  → alpha_L = q / p
    mL = q < p
    if mL.any():
        alphaL = q[mL] / p
        yL = gpd_ppf_survival(alphaL, xiL, betaL)
        out[mL] = uL - yL

    # ---- Core: p ≤ q ≤ 1-p  → nội suy trên center_data (đảm bảo liên tục ở p, 1-p)
    mC = (q >= p) & (q <= 1.0 - p)
    if mC.any():
        r = (q[mC] - p) / (1.0 - 2.0*p)
        idx = r * (nC - 1)
        i0 = np.floor(idx).astype(int)
        i1 = np.minimum(i0 + 1, nC - 1)
        w  = idx - i0
        x_core = (1 - w) * core[i0] + w * core[i1]
        x_core = np.where(np.isclose(q[mC], p),       uL, x_core)
        x_core = np.where(np.isclose(q[mC], 1.0 - p), uU, x_core)
        out[mC] = x_core

    # ---- Right tail: q > 1-p  → alpha_R = (1 - q) / p
    mR = q > 1.0 - p
    if mR.any():
        alphaR = (1.0 - q[mR]) / p
        yR = gpd_ppf_survival(alphaR, xiR, betaR)
        out[mR] = uU + yR

    return out[0] if np.isscalar(u) else out


# Hàm load dữ liệu và mô hình đã fit
def load_copula_and_marginals(copula_path, std_resids_path, evt_param_path):
    """
    Load copula model và tạo dict các marginal quantile function từ standardized residuals.
    Luôn dùng hybrid_ppf với tham số EVT, chỉ hỗ trợ file .pkl cho evt_param_path.
    """
    if copula_path.endswith('.json'):
        with open(copula_path, 'r', encoding='utf-8') as f:
            json_str = f.read()
        best_copula_model = pv.Vinecop.from_json(json_str)
    else:
        with open(copula_path, 'rb') as f:
            best_copula_model = pickle.load(f)
    std_resids_df = pd.read_csv(std_resids_path, index_col=0)
    with open(evt_param_path, 'rb') as f:
        evt_params = pickle.load(f)
    marginals = {}
    for col in std_resids_df.columns:
        marginals[col] = evt_params[col]
    return best_copula_model, marginals


def simulate_copula_portfolio(best_copula_model, marginals, n_sim=10000, random_state=None, output_path=None):
    """
    Sinh mẫu đồng thời từ copula đã fit và các marginal distribution.
    Args:
        best_copula_model: copula đã fit (ví dụ: StudentTMultivariate, GaussianMultivariate, ...)
        marginals: dict {ticker: inverse_cdf function} cho từng tài sản
        n_sim: số lượng mô phỏng
        random_state: seed
        output_path: đường dẫn file để lưu kết quả (nếu có)
    Returns:
        DataFrame các giá trị mô phỏng (shape: n_sim x n_assets)
    """
    np.random.seed(random_state)
    # Sinh mẫu uniform từ copula
    if hasattr(best_copula_model, 'sample'):
        # copulas or pyvinecopulib Vinecop
        try:
            u = best_copula_model.sample(n_sim)
        except Exception:
            # pyvinecopulib Vinecop: sample(n) returns numpy array
            u = best_copula_model.simulate(n_sim)
    elif hasattr(best_copula_model, 'simulate'):
        u = best_copula_model.simulate(n_sim)
    else:
        raise ValueError('Unknown copula model type for simulation')
    # Nếu trả về DataFrame, chuyển sang numpy
    if isinstance(u, pd.DataFrame):
        u = u.values
    # Biến đổi ngược về không gian biên
    sim_data = np.zeros_like(u)
    tickers = list(marginals.keys())
    for i, ticker in enumerate(tickers):
        sim_data[:, i] = hybrid_ppf(u[:, i], marginals[ticker])
    df_sim = pd.DataFrame(sim_data, columns=tickers)
    # Nếu có output_path thì lưu luôn ra file CSV
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_sim.to_csv(output_path, index=False)
        print(f"Simulated samples saved to: {output_path}")
    return df_sim


def analyze_simulated_data(sim_csv_path, real_csv_path=None, weights=None, alpha=0.05):
    """
    Phân tích toàn diện kết quả mô phỏng: thống kê mô tả, VaR/ES cho từng tài sản và danh mục.
    Trả về dict chứa tất cả kết quả phân tích.
    """
    import pandas as pd
    import numpy as np
    
    sim = pd.read_csv(sim_csv_path)
    results = {
        'sim_data': sim,
        'sim_stats': sim.describe(),
        'individual_risk': {},
        'portfolio_risk': {}
    }
    
    # VaR/ES từng tài sản
    for col in sim.columns:
        var = np.quantile(sim[col], alpha)
        es = sim[col][sim[col]<=var].mean()
        results['individual_risk'][col] = {'VaR': var, 'ES': es}
    
    # VaR/ES danh mục
    if weights is None:
        weights = np.ones(sim.shape[1]) / sim.shape[1]
    port = sim.values @ weights
    var_p = np.quantile(port, alpha)
    es_p = port[port<=var_p].mean()
    results['portfolio_risk'] = {'VaR': var_p, 'ES': es_p, 'returns': port}
    
    # So sánh với dữ liệu thực tế nếu có
    if real_csv_path is not None:
        real = pd.read_csv(real_csv_path)
        results['real_data'] = real
        results['real_stats'] = real.describe()
        
        # VaR/ES cho real data
        results['real_individual_risk'] = {}
        for col in real.columns:
            if col in sim.columns:
                var_real = np.quantile(real[col], alpha)
                es_real = real[col][real[col]<=var_real].mean()
                results['real_individual_risk'][col] = {'VaR': var_real, 'ES': es_real}
        
        # Portfolio real
        port_real = real[sim.columns].values @ weights
        var_p_real = np.quantile(port_real, alpha)
        es_p_real = port_real[port_real<=var_p_real].mean()
        results['real_portfolio_risk'] = {'VaR': var_p_real, 'ES': es_p_real, 'returns': port_real}
    
    return results


def plot_histogram_comparison(results, bins=50, max_cols=3):
    """
    Plot histogram comparison between simulated and real data.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import math
    
    sim = results['sim_data']
    cols = list(sim.columns)
    nrows, ncols = math.ceil(len(cols)/max_cols), min(max_cols, len(cols))
    
    # Individual asset histograms
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*6, nrows*4), squeeze=False)
    axes = axes.ravel()
    
    for i, col in enumerate(cols):
        ax = axes[i]
        
        # Plot simulated data
        sns.histplot(sim[col], bins=bins, kde=True, color='blue', 
                    label='Simulated', stat='density', alpha=0.6, ax=ax)
        
        # Plot real data if available
        if 'real_data' in results:
            real = results['real_data']
            if col in real.columns:
                sns.histplot(real[col], bins=bins, kde=True, color='red', 
                           label='Real', stat='density', alpha=0.6, ax=ax)
        
        # Add VaR lines
        var_sim = results['individual_risk'][col]['VaR']
        ax.axvline(var_sim, color='blue', linestyle='--', alpha=0.8, 
                  label=f'VaR Sim: {var_sim:.3f}')
        
        if 'real_individual_risk' in results and col in results['real_individual_risk']:
            var_real = results['real_individual_risk'][col]['VaR']
            ax.axvline(var_real, color='red', linestyle='--', alpha=0.8,
                      label=f'VaR Real: {var_real:.3f}')
        
        ax.set_title(f'{col} - Distribution Comparison')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for j in range(len(cols), len(axes)):
        axes[j].axis('off')
    
    plt.suptitle('Individual Asset Distribution Comparison', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Portfolio histogram
    plt.figure(figsize=(8, 5))
    portfolio_sim = results['portfolio_risk']['returns']
    sns.histplot(portfolio_sim, bins=bins, kde=True, color='blue', 
                label='Simulated Portfolio', stat='density', alpha=0.6)
    
    if 'real_portfolio_risk' in results:
        portfolio_real = results['real_portfolio_risk']['returns']
        sns.histplot(portfolio_real, bins=bins, kde=True, color='red', 
                    label='Real Portfolio', stat='density', alpha=0.6)
    
    # Add VaR lines
    var_sim_port = results['portfolio_risk']['VaR']
    plt.axvline(var_sim_port, color='blue', linestyle='--', alpha=0.8,
               label=f'VaR Sim: {var_sim_port:.3f}')
    
    if 'real_portfolio_risk' in results:
        var_real_port = results['real_portfolio_risk']['VaR']
        plt.axvline(var_real_port, color='red', linestyle='--', alpha=0.8,
                   label=f'VaR Real: {var_real_port:.3f}')
    
    plt.title('Portfolio Distribution Comparison')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pairplot_comparison(results, sample_size=5000):
    """
    Plot scatter plot matrix (pairplot) comparison between simulated and real data.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    sim = results['sim_data']
    
    # Sample data if too large for plotting
    if len(sim) > sample_size:
        sim_sample = sim.sample(sample_size, random_state=42)
    else:
        sim_sample = sim.copy()
    
    if 'real_data' in results:
        real = results['real_data']
        if len(real) > sample_size:
            real_sample = real[sim.columns].sample(sample_size, random_state=42)
        else:
            real_sample = real[sim.columns].copy()
        
        # Combine data for comparison
        sim_sample['type'] = 'Simulated'
        real_sample['type'] = 'Real'
        df_combined = pd.concat([sim_sample, real_sample], ignore_index=True)
        
        # Create pairplot with hue
        g = sns.pairplot(df_combined, hue='type', 
                        plot_kws={'alpha': 0.4, 's': 15},
                        diag_kws={'alpha': 0.6},
                        palette={'Simulated': 'blue', 'Real': 'red'})
        g.fig.suptitle('Pairwise Relationship Comparison (Real vs Simulated)', 
                      y=1.02, fontsize=14)
        
    else:
        # Only simulated data
        g = sns.pairplot(sim_sample, 
                        plot_kws={'alpha': 0.4, 's': 15, 'color': 'blue'},
                        diag_kws={'alpha': 0.6, 'color': 'blue'})
        g.fig.suptitle('Pairwise Relationships - Simulated Data', 
                      y=1.02, fontsize=14)
    
    plt.tight_layout()
    plt.show()
    
    # Correlation matrix heatmap
    fig, axes = plt.subplots(1, 2 if 'real_data' in results else 1, 
                            figsize=(12 if 'real_data' in results else 6, 5))
    
    if 'real_data' in results:
        # Simulated correlation
        sns.heatmap(sim[sim.columns].corr(), annot=True, cmap='coolwarm', 
                   center=0, ax=axes[0], fmt='.3f')
        axes[0].set_title('Simulated Data Correlation')
        
        # Real correlation  
        real = results['real_data']
        sns.heatmap(real[sim.columns].corr(), annot=True, cmap='coolwarm', 
                   center=0, ax=axes[1], fmt='.3f')
        axes[1].set_title('Real Data Correlation')
    else:
        sns.heatmap(sim.corr(), annot=True, cmap='coolwarm', 
                   center=0, ax=axes, fmt='.3f')
        axes.set_title('Simulated Data Correlation')
    
    plt.tight_layout()
    plt.show()