"python src/copula_simulation.py fit --std_resids data/processed/std_resids.csv --output models/best_copula.json"
# NOTE: The copula fitting now uses conditional PIT (from data/processed/conditional_pit.pkl) for U, not rank transform.
"python src/copula_simulation.py simulate --copula models/best_copula.json --std_resids data/processed/std_resids.csv --evt_params data/processed/evt_results.pkl --n_sim 10000 --output data/processed/simulated_copula.csv"
"python src/copula_simulation.py analyze --sim_csv data/processed/simulated_copula.csv --real_csv data/processed/std_resids.csv"

import numpy as np
import pandas as pd
import pickle
import pyvinecopulib as pv
from scipy.stats import rankdata
import os
import json

# Fit copula and save model
def fit_copula_and_save(std_resids_path, out_path='models/best_copula.json'):
    """
    Fit R-vine copula từ standardized residuals và lưu model ra file json.
    """
    df = pd.read_csv(std_resids_path, index_col=0)
    # Load conditional PIT
    with open('data/processed/conditional_pit.pkl', 'rb') as f:
        pit_dict = pickle.load(f)
    # Align PIT to df index and columns
    pit_df = pd.DataFrame({k: pd.Series(v, index=df.index) for k, v in pit_dict.items()})
    pit_df = pit_df[df.columns]  # Ensure column order matches
    u = pit_df.values
    import pyvinecopulib as pv
    if not out_path.endswith('.json'):
        raise ValueError('Output file must have .json extension')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        family_set = [
            getattr(pv.BicopFamily, 'gaussian', 1),
            getattr(pv.BicopFamily, 'student', 2),
            getattr(pv.BicopFamily, 'clayton', 3),
            getattr(pv.BicopFamily, 'gumbel', 4),
            getattr(pv.BicopFamily, 'frank', 5)
        ]
    except Exception:
        family_set = [1, 3, 4, 5]
    controls = pv.FitControlsVinecop(family_set=family_set)
    vine = pv.Vinecop(u.shape[1])
    vine.select(u, controls=controls)
    out_json = out_path if out_path.endswith('.json') else out_path.rsplit('.', 1)[0] + '.json'
    json_str = vine.to_json()
    with open(out_json, 'w') as f:
        f.write(json_str)
    print(f"Fitted R-vine copula (pyvinecopulib) and saved to {out_json}")


# Hàm tạo hàm nghịch đảo CDF (quantile function) từ standardized residuals
def hybrid_ppf(sample, left_params, right_params, qL=0.95, qU=0.95):
    sorted_vals = np.sort(sample)
    n = len(sorted_vals)
    xL = np.quantile(sorted_vals, 1 - qL)
    xU = np.quantile(sorted_vals, qU)
    thL, shapeL, scaleL = left_params
    thR, shapeR, scaleR = right_params

    def gpd_ppf(q, shape, scale):
        # q in (0,1) for GPD
        return scale * (q ** (-shape) - 1) / shape if abs(shape) > 1e-8 else scale * np.log(1.0 / q)

    def ppf(q):
        q = np.asarray(q)
        out = np.empty_like(q, dtype=float)
        # Center
        mask_mid = (q >= (1 - qL)) & (q <= qU)
        idx = (q[mask_mid] * (n - 1)).astype(int)
        out[mask_mid] = sorted_vals[idx]
        # Left tail
        mask_left = q < (1 - qL)
        p_exceed = (1 - qL)
        u = q[mask_left] / p_exceed
        out[mask_left] = - (thL + gpd_ppf(1 - u, shapeL, scaleL))
        # Right tail
        mask_right = q > qU
        p_exceed_r = (1 - qU)
        u = (1 - q[mask_right]) / p_exceed_r
        out[mask_right] = thR + gpd_ppf(1 - u, shapeR, scaleR)
        return out
    return ppf


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
        sample = std_resids_df[col].dropna().values
        left = evt_params[col]['left']   # [thL, shapeL, scaleL]
        right = evt_params[col]['right'] # [thR, shapeR, scaleR]
        marginals[col] = hybrid_ppf(sample, left, right)
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
        sim_data[:, i] = marginals[ticker](u[:, i])
    df_sim = pd.DataFrame(sim_data, columns=tickers)
    # Nếu có output_path thì lưu luôn ra file CSV
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_sim.to_csv(output_path, index=False)
        print(f"Simulated samples saved to: {output_path}")
    return df_sim


def comprehensive_analysis(sim_csv_path, real_csv_path=None, weights=None, alpha=0.05, out_dir='reports/figures'):
    """
    Phân tích toàn diện kết quả mô phỏng: thống kê mô tả, histogram, pairplot, VaR/ES cho từng tài sản và danh mục.
    Nếu cung cấp real_csv_path thì so sánh với dữ liệu thực tế.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    os.makedirs(out_dir, exist_ok=True)
    sim = pd.read_csv(sim_csv_path)
    print('--- Thống kê mô tả dữ liệu mô phỏng ---')
    print(sim.describe())
    # Histogram từng tài sản
    for col in sim.columns:
        plt.figure(figsize=(6,3))
        sns.histplot(sim[[col]], bins=50, kde=True, color='skyblue')
        plt.title(f'Histogram (Simulated) - {col}')
        plt.tight_layout()
        plt.savefig(f'{out_dir}/hist_sim_{col}.png')
        plt.close()
    # Pairplot
    sns.pairplot(sim, kind='scatter', plot_kws={'alpha':0.3, 's':10})
    plt.suptitle('Pairplot Simulated', y=1.02)
    plt.savefig(f'{out_dir}/pairplot_sim.png')
    plt.close()
    # VaR/ES từng tài sản
    print(f'--- VaR({int(alpha*100)}%) và ES({int(alpha*100)}%) từng tài sản ---')
    for col in sim.columns:
        var = np.quantile(sim[col], alpha)
        es = sim[col][sim[col]<=var].mean()
        print(f'{col}: VaR={var:.4f}, ES={es:.4f}')
    # VaR/ES danh mục (equal weight hoặc weights)
    if weights is None:
        weights = np.ones(sim.shape[1]) / sim.shape[1]
    port = sim.values @ weights
    var_p = np.quantile(port, alpha)
    es_p = port[port<=var_p].mean()
    print(f'Portfolio VaR({int(alpha*100)}%): {var_p:.4f}, ES: {es_p:.4f}')
    plt.figure(figsize=(6,3))
    sns.histplot(port, bins=50, kde=True, color='orange')
    plt.title('Histogram (Simulated Portfolio)')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/hist_sim_portfolio.png')
    plt.close()
    # So sánh với dữ liệu thực tế nếu có
    if real_csv_path is not None:
        real = pd.read_csv(real_csv_path)
        print('\n--- Thống kê mô tả dữ liệu thực tế ---')
        print(real.describe())
        for col in sim.columns:
            plt.figure(figsize=(6,3))
            sns.histplot(real[[col]], bins=50, kde=True, color='salmon', label='Real', stat='density', alpha=0.5)
            sns.histplot(sim[[col]], bins=50, kde=True, color='skyblue', label='Sim', stat='density', alpha=0.5)
            plt.legend()
            plt.title(f'Histogram So sánh - {col}')
            plt.tight_layout()
            plt.savefig(f'{out_dir}/hist_compare_{col}.png')
            plt.close()
        # Pairplot so sánh
        real['type'] = 'Real'
        sim['type'] = 'Sim'
        df_all = pd.concat([real, sim], ignore_index=True)
        sns.pairplot(df_all, hue='type', plot_kws={'alpha':0.3, 's':10})
        plt.suptitle('Pairplot So sánh', y=1.02)
        plt.savefig(f'{out_dir}/pairplot_compare.png')
        plt.close()


import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Copula-based portfolio risk simulation and fitting.")
    subparsers = parser.add_subparsers(dest='command')

    # Subparser for simulation
    sim_parser = subparsers.add_parser('simulate', help='Simulate portfolio risk using fitted copula')
    sim_parser.add_argument('--copula', type=str, default='models/best_copula.json', help='Path to copula model pickle file')
    sim_parser.add_argument('--std_resids', type=str, default='data/processed/std_resids.csv', help='Path to standardized residuals CSV')
    sim_parser.add_argument('--evt_params', type=str, required=True, help='Path to EVT params JSON file')
    sim_parser.add_argument('--n_sim', type=int, default=10000, help='Number of simulations')
    sim_parser.add_argument('--output', type=str, default='data/processed/simulated_copula_samples.csv', help='Output CSV file for simulated samples')

    # Subparser for fitting copula
    fit_parser = subparsers.add_parser('fit', help='Fit copula model and save to pickle')
    fit_parser.add_argument('--std_resids', type=str, default='data/processed/std_resids.csv', help='Path to standardized residuals CSV')
    fit_parser.add_argument('--output', type=str, default='models/best_copula.json', help='Output path for copula JSON')

    # Subparser for analysis
    analyze_parser = subparsers.add_parser('analyze', help='Comprehensive analysis of simulated results')
    analyze_parser.add_argument('--sim_csv', type=str, default='data/processed/simulated_copula_samples.csv', help='Path to simulated samples CSV')
    analyze_parser.add_argument('--real_csv', type=str, default=None, help='Path to real data CSV (optional)')
    analyze_parser.add_argument('--alpha', type=float, default=0.05, help='VaR/ES quantile (default 0.05)')
    analyze_parser.add_argument('--out_dir', type=str, default='reports/figures', help='Output directory for figures')

    args = parser.parse_args()

    if args.command == 'fit':
        fit_copula_and_save(args.std_resids, args.output)
    elif args.command == 'simulate':
        print(f"Loading copula model from: {args.copula}")
        print(f"Loading standardized residuals from: {args.std_resids}")
        print(f"Loading EVT params from: {args.evt_params}")
        best_copula_model, marginals = load_copula_and_marginals(args.copula, args.std_resids, args.evt_params)
        print(f"Simulating {args.n_sim} samples...")
        simulate_copula_portfolio(best_copula_model, marginals, n_sim=args.n_sim, output_path=args.output)
    elif args.command == 'analyze':
        comprehensive_analysis(args.sim_csv, real_csv_path=args.real_csv, alpha=args.alpha, out_dir=args.out_dir)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
