import numpy as np
import pandas as pd
import pickle
import pyvinecopulib as pv
from scipy.stats import rankdata, kendalltau
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import chi2, binomtest
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import json

from src.copula_simulation import hybrid_ppf


def kendalltau_matrix(df):
    n = df.shape[1]
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = 1
            elif i < j:
                tau, _ = kendalltau(df.iloc[:, i], df.iloc[:, j])
                mat[i, j] = tau # type: ignore
                mat[j, i] = tau # type: ignore
    return mat

def empirical_tail_dependence(u, v, q=0.95):
    eps = 1.0 - q
    lam_L = np.mean((u < eps) & (v < eps)) / eps
    lam_U = np.mean((u > q) & (v > q)) / eps
    return lam_L, lam_U

def rolling_var_backtest(std_resids, evt_params, window=500, alpha=0.05, step=1):
    """
    Rolling backtest VaR for each asset and portfolio with R-vine.
    """
    tickers = std_resids.columns
    n = len(std_resids)
    var_violations = {col: [] for col in tickers}
    var_values = {col: [] for col in tickers}
    idx_list = []
    pit_df = pd.read_csv('../models/copula/pit_data.csv', index_col=0)
    pit_df = pit_df[std_resids.columns]  # Ensure column order matches
    for start in tqdm(range(0, n-window, step), desc="Rolling VaR"):
        end = start + window
        idx_list.append(end)
        window_data = std_resids.iloc[start:end]
        window_pit = pit_df.iloc[start:end]
        u = window_pit
        # Fit vine
        vine = pv.Vinecop(u.shape[1])
        family_set = [
            getattr(pv.BicopFamily, 'gaussian', 1),
            getattr(pv.BicopFamily, 'student', 2),
            getattr(pv.BicopFamily, 'clayton', 3),
            getattr(pv.BicopFamily, 'gumbel', 4)
        ]
        controls = pv.FitControlsVinecop(family_set=family_set)
        u_clip = np.clip(u.values.astype(float), 1e-6, 1-1e-6)
        vine.select(u_clip, controls=controls)
        u_sim = np.asarray(vine.simulate(1000))
        u_sim = np.clip(u_sim, 1e-6, 1-1e-6)
        marginals = {col: evt_params[col] for col in tickers}
        sim_data = np.zeros_like(u_sim)
        for i, col in enumerate(tickers):
            sim_data[:, i] = hybrid_ppf(u_sim[:, i], marginals[col])

        sim_df = pd.DataFrame(sim_data, columns=tickers)
        # VaR for each asset
        for col in tickers:
            var = np.quantile(sim_df[col], alpha)
            var_values[col].append(var)
            # Check VaR violation on the next day
            if end < n:
                real_val = std_resids.iloc[end][col]
                var_violations[col].append(real_val < var)
    # Aggregate results
    results = {}
    for col in tickers:
        violations = np.array(var_violations[col])
        n_test = len(violations)
        n_violate = violations.sum()
        p_hat = n_violate / n_test if n_test > 0 else np.nan
        results[col] = {
            "n_test": n_test,
            "n_violate": n_violate,
            "violation_rate": p_hat,
            "violations": violations.astype(bool).tolist(),
        }
    return results, var_values, idx_list


def plot_rolling_var(std_resids, var_values, idx_list, var_violations, out_dir):
    """
    Plot rolling VaR, realized values, and violations for each asset.
    """
    os.makedirs(out_dir, exist_ok=True)
    tickers = std_resids.columns
    for col in tickers:
        plt.figure(figsize=(12,4))
        # Realized values
        realized = std_resids[col].iloc[idx_list].values
        # VaR values
        var_series = np.array(var_values[col])
        # Violation mask
        violations = np.array(var_violations[col], dtype=bool)
        plt.plot(idx_list, realized, label='Realized', color='blue')
        plt.plot(idx_list, var_series, label='VaR', color='red')
        # Mark violations
        if violations.any():
            plt.scatter(np.array(idx_list)[violations], realized[violations], color='orange', label='Violation', zorder=5)
        plt.title(f'Rolling VaR Backtest - {col}')
        plt.xlabel('Time Index')
        plt.ylabel('Value')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'rolling_var_{col}.png'))
        plt.close()


def plot_pit_diagnostics(pit, out_dir, prefix="pit"):
    os.makedirs(out_dir, exist_ok=True)
    n_dim = pit.shape[1]
    pit_dir = os.path.join(out_dir, "pit")
    os.makedirs(pit_dir, exist_ok=True)
    for i in range(n_dim):
        plt.figure(figsize=(5,3))
        sns.histplot(pit[:, i], bins=30, kde=True, stat='density')
        plt.title(f'PIT Histogram - Dim {i+1}')
        plt.tight_layout()
        plt.savefig(os.path.join(pit_dir, f"{prefix}_hist_{i+1}.png"))
        plt.close()
        # QQ plot
        plt.figure(figsize=(5,3))
        sorted_pit = np.sort(pit[:, i])
        plt.plot(np.linspace(0,1,len(sorted_pit)), sorted_pit, marker='.', linestyle='')
        plt.plot([0,1],[0,1],'r--')
        plt.title(f'PIT QQ Uniform - Dim {i+1}')
        plt.tight_layout()
        plt.savefig(os.path.join(pit_dir, f"{prefix}_qq_{i+1}.png"))
        plt.close()
        # Autocorrelation
        lb_p = acorr_ljungbox(pit[:, i], lags=[10], return_df=True)['lb_pvalue'].iloc[0]
        print(f"PIT dim {i+1}: Ljung-Box p-value = {lb_p:.4f}")


def compare_dependence(real_u, sim_u, out_dir):
    tau_real = kendalltau_matrix(real_u)
    tau_sim = kendalltau_matrix(sim_u)
    delta = np.abs(tau_real - tau_sim)
    frob_norm = np.linalg.norm(delta, ord='fro')
    print(f"Frobenius norm of Kendall's tau difference: {frob_norm:.4f}")
    plt.figure(figsize=(8,4))
    sns.heatmap(delta, annot=True, fmt=".2f", cmap="Reds")
    plt.title("Δ Kendall's tau (real - sim)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/delta_kendalltau.png")
    plt.close()
    return frob_norm


def compare_tail_dependence(real_u, sim_u, out_dir, q=0.95):
    n = real_u.shape[1]
    lambdaL_real = np.zeros((n,n))
    lambdaU_real = np.zeros((n,n))
    lambdaL_sim = np.zeros((n,n))
    lambdaU_sim = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i < j:
                lLr, lUr = empirical_tail_dependence(real_u.iloc[:, i], real_u.iloc[:, j], q)
                lLs, lUs = empirical_tail_dependence(sim_u.iloc[:, i], sim_u.iloc[:, j], q)
                lambdaL_real[i,j] = lLr
                lambdaU_real[i,j] = lUr
                lambdaL_sim[i,j] = lLs
                lambdaU_sim[i,j] = lUs
    plt.figure(figsize=(8,4))
    sns.heatmap(np.abs(lambdaL_real - lambdaL_sim), annot=True, fmt=".2f", cmap="Blues")
    plt.title("Δ Lower Tail Dependence (real - sim)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/delta_taildep_lower.png")
    plt.close()
    plt.figure(figsize=(8,4))
    sns.heatmap(np.abs(lambdaU_real - lambdaU_sim), annot=True, fmt=".2f", cmap="Oranges")
    plt.title("Δ Upper Tail Dependence (real - sim)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/delta_taildep_upper.png")
    plt.close()


def kupiec_test(real, var, alpha=0.05, method='lr'):
   
    r = np.asarray(real, float)
    q = np.asarray(var,  float)
    m = np.isfinite(r) & np.isfinite(q)
    r, q = r[m], q[m]

    n = r.size
    I = (r < q).astype(int)  # vi phạm: r_t < VaR_t (đuôi trái)
    x = int(I.sum())
    pi_hat = x / n
    expected = n * alpha

    if method == 'lr':  # Kupiec POF (likelihood-ratio)
        eps = 1e-12
        ll_null = (n - x) * np.log(1 - alpha) + x * np.log(alpha + eps)
        ll_mle  = (n - x) * np.log(1 - pi_hat + eps) + x * np.log(pi_hat + eps)
        LR_pof  = -2.0 * (ll_null - ll_mle)
        p_value = 1 - chi2.cdf(LR_pof, df=1)
        return {
            'n': n, 'violations': x, 'expected': expected,
            'hit_rate': pi_hat, 'LR_pof': float(LR_pof),
            'p_value': float(p_value), 'reject_H0_5%': bool(p_value < 0.05)
        }
    else:  # 'binom' – exact test (hai phía), hơi bảo thủ hơn
        p_value = binomtest(x, n, alpha, alternative='two-sided').pvalue
        return {
            'n': n, 'violations': x, 'expected': expected,
            'hit_rate': pi_hat, 'p_value': float(p_value),
            'reject_H0_5%': bool(p_value < 0.05)
        }


def christoffersen_test(violations, alpha):
    """
    Christoffersen (1998) Conditional Coverage test for VaR backtest.
    violations: array-like of bool/int (1 nếu vi phạm VaR, 0 nếu không)
    alpha: mức VaR (e.g., 0.05)
    Returns: dict with LRuc, LRind, LRcc and p-values + stats phụ
    """
    v = np.asarray(violations, dtype=int)
    n = v.size
    n1 = int(v.sum())
    n0 = int(n - n1)

    # --- Unconditional coverage (Kupiec LRuc) ---
    pi_hat = n1 / n if n > 0 else 0.0
    eps = 1e-12
    LRuc = -2.0 * (
        n1 * np.log(alpha + eps) + n0 * np.log(1.0 - alpha + eps)
        - (n1 * np.log(pi_hat + eps) + n0 * np.log(1.0 - pi_hat + eps))
    )
    p_uc = 1.0 - chi2.cdf(LRuc, df=1)

    # --- Independence (Markov 1) ---
    if n < 2:
        # Không đủ quan sát để lập ma trận chuyển trạng thái
        LRind = 0.0
        p_ind = 1.0
        n00 = n01 = n10 = n11 = 0
    else:
        n00 = int(np.sum((v[:-1] == 0) & (v[ 1:] == 0)))
        n01 = int(np.sum((v[:-1] == 0) & (v[ 1:] == 1)))
        n10 = int(np.sum((v[:-1] == 1) & (v[ 1:] == 0)))
        n11 = int(np.sum((v[:-1] == 1) & (v[ 1:] == 1)))

        denom0 = n00 + n01
        denom1 = n10 + n11
        denom  = denom0 + denom1

        pi01 = n01 / (denom0 + eps)      # P(vi phạm | trước đó không)
        pi11 = n11 / (denom1 + eps)      # P(vi phạm | trước đó có)
        pi1  = (n01 + n11) / (denom + eps)  # Xác suất vi phạm chung theo chuyển trạng thái

        LRind = -2.0 * (
            (denom0) * np.log(1.0 - pi1 + eps) + (denom1) * np.log(pi1 + eps)
            - n00 * np.log(1.0 - pi01 + eps) - n01 * np.log(pi01 + eps)
            - n10 * np.log(1.0 - pi11 + eps) - n11 * np.log(pi11 + eps)
        )
        p_ind = 1.0 - chi2.cdf(LRind, df=1)

    # --- Conditional coverage ---
    LRcc = LRuc + LRind
    p_cc = 1.0 - chi2.cdf(LRcc, df=2)

    return {
        'LRuc': float(LRuc), 'p_uc': float(p_uc),
        'LRind': float(LRind), 'p_ind': float(p_ind),
        'LRcc': float(LRcc), 'p_cc': float(p_cc),
        'n_violate': n1, 'n_test': n,
        'hit_rate': (n1 / n) if n > 0 else np.nan,
        'expected': n * alpha,
        'n00': n00, 'n01': n01, 'n10': n10, 'n11': n11
    }

    
def vine_aic_bic_loglik(vine, u):
    """
    Compute log-likelihood, AIC, and BIC for a Vinecop model fitted on data u (numpy array)
    """
    n, d = u.shape
    loglik = vine.loglik(u)
    n_params = sum(arr.size for tree in vine.parameters for arr in tree)
    aic = -2 * loglik + 2 * n_params
    bic = -2 * loglik + np.log(n) * n_params
    return {'loglik': float(loglik), 'aic': float(aic), 'bic': float(bic), 'n_params': int(n_params)}


def energy_distance(u1, u2):
    """
    Compute the energy distance between two samples (numpy arrays)
    """
    d1 = cdist(u1, u1, metric='euclidean')
    d2 = cdist(u2, u2, metric='euclidean')
    d12 = cdist(u1, u2, metric='euclidean')
    n, m = len(u1), len(u2)
    e = 2 * d12.mean() - d1.mean() - d2.mean()
    return float(e)


def convert_for_json(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):  # type: ignore
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)): # type: ignore
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_for_json(v) for v in obj]
    return obj


if __name__ == "__main__":
    # === Setup ===
    std_resids_path = "models/garch/standardized_residuals.csv"
    evt_param_path = "models/evt/marginal_distributions.pkl"
    copula_path = "models/copula/best_copula.json"
    out_dir = "validation"
    os.makedirs(out_dir, exist_ok=True)

    std_resids = pd.read_csv(std_resids_path, index_col=0)
    with open(evt_param_path, "rb") as f:
        evt_params = pickle.load(f)

    validation_summary = {}

    # === 1. Rolling VaR backtest ===
    rolling_var_path = os.path.join(out_dir, "rolling_var_results.pkl")
    if os.path.exists(rolling_var_path):
        with open(rolling_var_path, "rb") as f:
            results, var_values, idx_list = pickle.load(f)
        print("Loaded rolling VaR backtest results from file.")
    else:
        print("Running rolling VaR backtest...")
        results, var_values, idx_list = rolling_var_backtest(std_resids, evt_params, window=500, alpha=0.05, step=1)
        with open(rolling_var_path, "wb") as f:
            pickle.dump((results, var_values, idx_list), f)
        print(f"Saved rolling VaR backtest results to {rolling_var_path}")

    validation_summary["rolling_var"] = results
    validation_summary["var_values"] = var_values
    validation_summary["idx_list"] = idx_list
    
    # 1a. Kupiec test for VaR
    kupiec_summary = {}
    print("Kupiec test for VaR:")
    for col, res in results.items():
        # Lấy lại chuỗi giá trị thực và VaR dự báo
        # VaR dự báo: var_values[col], giá trị thực: std_resids.iloc[idx_list][col].values
        var_series = np.array(var_values[col])
        realized = std_resids[col].iloc[idx_list].values
        kupiec_summary[col] = kupiec_test(realized, var_series, alpha=0.05)
        print(f"{col}: {kupiec_summary[col]}")
    validation_summary["kupiec"] = kupiec_summary

    # 1b. Christoffersen test for VaR
    christoffersen_summary = {}
    for col, res in results.items():
        violations = res.get("violations", None)
        if violations is not None:
            christoffersen_summary[col] = christoffersen_test(
                violations, alpha=0.05
            )
    validation_summary["christoffersen"] = christoffersen_summary
    print("Christoffersen test and Kupiec test for VaR calculated.")

    # 1c. Plot rolling VaR backtest results
    var_violations = {}
    for col in var_values:
        if "violations" in results.get(col, {}):
            var_violations[col] = results[col]["violations"]
        else:
            var_violations[col] = [False] * len(var_values[col])
    plot_rolling_var(std_resids, var_values, idx_list, var_violations, out_dir=os.path.join(out_dir, "rolling_var_plots"))
    print(f"Rolling VaR plots saved to {os.path.join(out_dir, 'rolling_var_plots')}")

    # === 2. Fit Vinecop on the entire dataset ===
    pit_df = pd.read_csv('../models/copula/pit_data.csv', index_col=0)
    pit_df = pit_df[std_resids.columns]  # Ensure column order matches
    u = pit_df
    vine = pv.Vinecop(u.shape[1])
    family_set = [
        getattr(pv.BicopFamily, "gaussian", 1),
        getattr(pv.BicopFamily, "student", 2),
        getattr(pv.BicopFamily, "clayton", 3),
        getattr(pv.BicopFamily, "gumbel", 4),
    ]
    controls = pv.FitControlsVinecop(family_set=family_set)
    u_clip = np.clip(u.values.astype(float), 1e-6, 1-1e-6)
    vine.select(u_clip, controls=controls)

    # === 3. PIT diagnostics ===
    print("=== Rosenblatt PIT diagnostics ===")
    pit = vine.rosenblatt(u.values)
    plot_pit_diagnostics(pit, out_dir)
    pit_ljungbox_pvalues = []
    for i in range(pit.shape[1]): # type: ignore
        lb_p = acorr_ljungbox(pit[:, i], lags=[10], return_df=True)["lb_pvalue"].iloc[0] # type: ignore
        pit_ljungbox_pvalues.append(lb_p)
    validation_summary["pit_ljungbox_pvalues"] = pit_ljungbox_pvalues

    # === 4. Dependence metrics ===
    u_sim = vine.simulate(len(u))
    frob_norm = compare_dependence(pd.DataFrame(u.values), pd.DataFrame(u_sim), out_dir) # type: ignore
    validation_summary["frob_norm_kendalltau"] = frob_norm
    compare_tail_dependence(pd.DataFrame(u.values), pd.DataFrame(u_sim), out_dir, q=0.95) # type: ignore
    validation_summary["tail_dependence_heatmap"] = [
        os.path.join(out_dir, "delta_taildep_lower.png"),
        os.path.join(out_dir, "delta_taildep_upper.png"),
    ]

    # === 5. Model selection metrics (AIC/BIC/loglik) ===
    vine_metrics = vine_aic_bic_loglik(vine, u.values)
    validation_summary["vine_aic_bic_loglik"] = vine_metrics
    print("AIC/BIC/loglik of Vinecop:", vine_metrics)

    # === 6. Distance metrics ===
    e_dist = energy_distance(u.values, u_sim)
    validation_summary["energy_distance"] = e_dist
    print(f"Energy distance (real U vs simulated U): {e_dist:.4f}")

    # === 7. Save summary ===
    validation_summary_json = {
        k: convert_for_json(v) for k, v in validation_summary.items()
    }
    with open(
        os.path.join(out_dir, "validation_summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(validation_summary_json, f, indent=2, ensure_ascii=False)

    print(f"Saved summary results to {os.path.join(out_dir, 'validation_summary.json')}")
    print(f"Validation plots saved to {out_dir}")
