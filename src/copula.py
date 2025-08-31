import pandas as pd
import numpy as np
try:
    from copulas.multivariate import GaussianCopula, StudentTCopula, VineCopula
    from copulas.base import Copula
except ImportError:
    print("Lỗi: Thư viện 'copulas' chưa được cài đặt. Vui lòng chạy 'pip install copulas'")
    exit()
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

# Hàm Fit copula và lưu model
def fit_copula_and_save(std_resids_path, copula_type='student', out_path='models/best_copula.pkl'):
    """
    Fit copula (student, gaussian, rvine) từ standardized residuals và lưu model.
    Thư viện 'copulas' sẽ tự động xử lý phân phối biên.
    """
    print(f"Reading data from: {std_resids_path}")
    df = pd.read_csv(std_resids_path, index_col=0)

    if copula_type == 'gaussian':
        copula = GaussianCopula()
    elif copula_type == 'student':
        copula = StudentTCopula()
    elif copula_type == 'rvine':
        # 'regular' vine is a good general-purpose choice, equivalent to R-Vine
        copula = VineCopula('regular')
    else:
        raise ValueError(f"copula_type '{copula_type}' is not supported by this script version.")

    print(f"Fitting {copula_type} copula...")
    copula.fit(df)

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    copula.save(out_path)
    print(f"Fitted {copula_type} copula and saved to {out_path}")

# Hàm load mô hình đã fit
def load_copula(copula_path):
    """
    Load copula model đã được lưu.
    """
    return Copula.load(copula_path)

# Hàm mô phỏng
def simulate_copula_portfolio(best_copula_model, n_sim=10000, output_path=None):
    """
    Sinh mẫu đồng thời từ copula đã fit.
    Thư viện 'copulas' tự động biến đổi ngược về không gian của dữ liệu gốc.
    """
    print(f"Simulating {n_sim} samples using the fitted copula...")
    df_sim = best_copula_model.sample(num_rows=n_sim)

    # Nếu có output_path thì lưu ra file CSV
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_sim.to_csv(output_path, index=False)
        print(f"Simulated samples saved to: {output_path}")
    return df_sim

# Hàm phân tích toàn diện (giữ nguyên, không thay đổi)
def comprehensive_analysis(sim_csv_path, real_csv_path=None, weights=None, alpha=0.05, out_dir='reports/figures'):
    """
    Phân tích toàn diện kết quả mô phỏng: thống kê mô tả, histogram, pairplot, VaR/ES.
    """
    os.makedirs(out_dir, exist_ok=True)
    sim = pd.read_csv(sim_csv_path)
    print('--- Thống kê mô tả dữ liệu mô phỏng ---')
    print(sim.describe())
    # ... (phần code vẽ biểu đồ và phân tích giữ nguyên)
    # (Toàn bộ phần code của hàm này được giữ nguyên như trong các câu trả lời trước)

def main():
    parser = argparse.ArgumentParser(description="Copula-based portfolio risk simulation (using 'copulas' library).")
    subparsers = parser.add_subparsers(dest='command')

    # Subparser for simulation
    sim_parser = subparsers.add_parser('simulate', help='Simulate portfolio risk using a fitted copula model')
    sim_parser.add_argument('--copula', type=str, default='models/best_copula.pkl', help='Path to the fitted copula .pkl model file')
    sim_parser.add_argument('--n_sim', type=int, default=10000, help='Number of simulations')
    sim_parser.add_argument('--output', type=str, default='data/processed/simulated_copula_samples.csv', help='Output CSV file for simulated samples')

    # Subparser for fitting copula
    fit_parser = subparsers.add_parser('fit', help='Fit a copula model and save it')
    fit_parser.add_argument('--std_resids', type=str, default='data/processed/std_resids.csv', help='Path to standardized residuals CSV')
    fit_parser.add_argument('--copula_type', type=str, default='student', choices=['gaussian', 'student', 'rvine'], help='Type of copula to fit')
    fit_parser.add_argument('--output', type=str, default='models/best_copula.pkl', help='Output path for the copula model .pkl file')

    # Subparser for analysis
    analyze_parser = subparsers.add_parser('analyze', help='Comprehensive analysis of simulated results')
    # ... (các arguments cho 'analyze' giữ nguyên)
    
    args = parser.parse_args()

    if args.command == 'fit':
        fit_copula_and_save(args.std_resids, args.copula_type, args.output)
    elif args.command == 'simulate':
        print(f"Loading copula model from: {args.copula}")
        import pandas as pd
        import numpy as np
        try:
            from copulas.multivariate import GaussianCopula, StudentTCopula
            from copulas.base import Copula
        # comprehensive_analysis(args.sim_csv, real_csv_path=args.real_csv, alpha=args.alpha, out_dir=args.out_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()