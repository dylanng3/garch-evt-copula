"python src/copula_simulation.py fit --std_resids data/processed/std_resids.csv --copula_type rvine --output models/best_copula.json"
"python src/copula_simulation.py simulate --copula models/best_copula.json --std_resids data/processed/std_resids.csv --n_sim 10000 --output data/processed/simulated_copula.csv"
"python src/copula_simulation.py analyze --sim_csv data/processed/simulated_copula.csv --real_csv data/processed/std_resids.csv"

import numpy as np
import pandas as pd
import pickle
import pyvinecopulib as pv
from scipy.stats import rankdata
import os

# Fit copula and save model
def fit_copula_and_save(std_resids_path, copula_type='student', out_path='models/best_copula.json'):
    """
    Fit copula (student, gaussian, clayton, gumbel) từ standardized residuals và lưu model ra file pickle.
    """
    df = pd.read_csv(std_resids_path, index_col=0)
    u = df.apply(lambda x: rankdata(x, method='average') / (len(x) + 1), axis=0)
    if copula_type == 'rvine':
        import pyvinecopulib as pv
        if not out_path.endswith('.json'):
            raise ValueError('For copula_type rvine, output file must have .json extension')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        u_np = u.values
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
        vine = pv.Vinecop(u_np.shape[1])
        vine.select(u_np, controls=controls)
        out_json = out_path if out_path.endswith('.json') else out_path.rsplit('.', 1)[0] + '.json'
        json_str = vine.to_json()
        with open(out_json, 'w') as f:
            f.write(json_str)
        print(f"Fitted R-vine copula (pyvinecopulib) and saved to {out_json}")
    else:
        raise NotImplementedError('Chỉ hỗ trợ xuất file json cho copula_type=rvine với pyvinecopulib.')


# Hàm tạo hàm nghịch đảo CDF (quantile function) từ standardized residuals
def empirical_ppf(series):
    """
    Trả về hàm nghịch đảo CDF thực nghiệm (empirical quantile function) cho 1 chuỗi dữ liệu.
    """
    sorted_vals = np.sort(series)
    def ppf(q):
        # q: array-like, các giá trị trong [0,1]
        idx = (q * (len(sorted_vals)-1)).astype(int)
        return sorted_vals[idx]
    return ppf


# Hàm load dữ liệu và mô hình đã fit

def load_copula_and_marginals(copula_path, std_resids_path):
    """
    Load copula model và tạo dict các marginal quantile function từ standardized residuals.
    """
    # If loading a pyvinecopulib R-vine model, use from_json
    if copula_path.endswith('.json'):
        # Đọc nội dung JSON từ file rồi mới parse
        with open(copula_path, 'r', encoding='utf-8') as f:
            json_str = f.read()
        best_copula_model = pv.Vinecop.from_json(json_str)
    else:
        with open(copula_path, 'rb') as f:
            best_copula_model = pickle.load(f)
    std_resids_df = pd.read_csv(std_resids_path, index_col=0)
    marginals = {col: empirical_ppf(std_resids_df[col].dropna().values) for col in std_resids_df.columns}
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
    sim_parser.add_argument('--n_sim', type=int, default=10000, help='Number of simulations')
    sim_parser.add_argument('--output', type=str, default='data/processed/simulated_copula_samples.csv', help='Output CSV file for simulated samples')

    # Subparser for fitting copula
    fit_parser = subparsers.add_parser('fit', help='Fit copula model and save to pickle')
    fit_parser.add_argument('--std_resids', type=str, default='data/processed/std_resids.csv', help='Path to standardized residuals CSV')
    fit_parser.add_argument('--copula_type', type=str, default='vine_student', choices=['vine_student','vine_gaussian','gaussian','clayton','gumbel','rvine'], help='Type of copula to fit (vine_student, vine_gaussian, gaussian, clayton, gumbel, rvine)')
    fit_parser.add_argument('--output', type=str, default='models/best_copula.json', help='Output path for copula JSON')

    # Subparser for analysis
    analyze_parser = subparsers.add_parser('analyze', help='Comprehensive analysis of simulated results')
    analyze_parser.add_argument('--sim_csv', type=str, default='data/processed/simulated_copula_samples.csv', help='Path to simulated samples CSV')
    analyze_parser.add_argument('--real_csv', type=str, default=None, help='Path to real data CSV (optional)')
    analyze_parser.add_argument('--alpha', type=float, default=0.05, help='VaR/ES quantile (default 0.05)')
    analyze_parser.add_argument('--out_dir', type=str, default='reports/figures', help='Output directory for figures')

    args = parser.parse_args()

    if args.command == 'fit':
        fit_copula_and_save(args.std_resids, args.copula_type, args.output)
    elif args.command == 'simulate':
        print(f"Loading copula model from: {args.copula}")
        print(f"Loading standardized residuals from: {args.std_resids}")
        best_copula_model, marginals = load_copula_and_marginals(args.copula, args.std_resids)
        print(f"Simulating {args.n_sim} samples...")
        simulate_copula_portfolio(best_copula_model, marginals, n_sim=args.n_sim, output_path=args.output)
    elif args.command == 'analyze':
        comprehensive_analysis(args.sim_csv, real_csv_path=args.real_csv, alpha=args.alpha, out_dir=args.out_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
