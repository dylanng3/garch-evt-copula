import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from scipy.stats import genpareto
import matplotlib.pyplot as plt

def fit_garch_t(log_returns):
    """
    Fit GARCH(1,1) với Student's t cho từng chuỗi lợi suất.
    Trả về:
        - garch_results: dict kết quả fit arch_model
        - std_resids: dict phần dư chuẩn hóa
    """
    garch_results = {}
    std_resids = {}
    for ticker in log_returns.columns:
        print(f"Fitting GARCH for {ticker}...")
        am = arch_model(log_returns[ticker].dropna(), vol='Garch', p=1, q=1, dist='t')
        res = am.fit(update_freq=10, disp='off')
        print(res.summary())
        garch_results[ticker] = res
        std_resids[ticker] = res.std_resid
    return garch_results, std_resids

def diagnostic_residuals(std_resids, lags=10, plot=False):
    """
    Kiểm định ARCH và Ljung-Box cho phần dư chuẩn hóa và phần dư bình phương.
    """
    diag_results = {}
    for ticker, resid in std_resids.items():
        result = {}
        # Bỏ phần plot
        # ARCH test
        result['arch_pvalue'] = het_arch(resid, nlags=lags)[1]
        result['arch_pvalue_sq'] = het_arch(resid**2, nlags=lags)[1]
        # Ljung-Box test
        result['lb_pvalue'] = acorr_ljungbox(resid, lags=[lags], return_df=True)['lb_pvalue'].iloc[0]
        result['lb_pvalue_sq'] = acorr_ljungbox(resid**2, lags=[lags], return_df=True)['lb_pvalue'].iloc[0]
        diag_results[ticker] = result
        print(f"{ticker}: ARCH p={result['arch_pvalue']:.3f}, Ljung-Box p={result['lb_pvalue']:.3f}, "
              f"ARCH^2 p={result['arch_pvalue_sq']:.3f}, Ljung-Box^2 p={result['lb_pvalue_sq']:.3f}")
    return diag_results

def fit_evt(std_resids, quantile=0.95):
    """
    Fit Generalized Pareto Distribution (GPD) cho đuôi âm của phần dư chuẩn hóa.
    Trả về dictionary với threshold, shape, scale cho từng ticker.
    """
    evt_results = {}
    for ticker, resid in std_resids.items():
        losses = -resid.dropna()
        threshold = losses.quantile(quantile)
        exceedances = losses[losses > threshold] - threshold
        if len(exceedances) < 10:
            print(f"Warning: Too few exceedances for {ticker}. Skipping EVT fit.")
            continue
        shape, loc, scale = genpareto.fit(exceedances)
        evt_results[ticker] = {'threshold': threshold, 'shape': shape, 'scale': scale}
        print(f"EVT for {ticker}: threshold={threshold:.4f}, shape={shape:.4f}, scale={scale:.4f}")
    return evt_results


if __name__ == "__main__":
    # Demo: Đọc log_returns từ file CSV đã chuẩn hóa trước đó
    log_returns = pd.read_csv("data/processed/log_returns.csv", index_col=0)
    print("Loaded log_returns shape:", log_returns.shape)

    # Scale log returns (nếu muốn)
    log_returns_scaled = log_returns * 100

    # Fit GARCH(1,1) với Student's t
    garch_results, std_resids = fit_garch_t(log_returns_scaled)

    # Kiểm tra chẩn đoán phần dư
    diag_results = diagnostic_residuals(std_resids, lags=10, plot=True)

    # Fit EVT cho phần dư chuẩn hóa
    evt_results = fit_evt(std_resids, quantile=0.95)