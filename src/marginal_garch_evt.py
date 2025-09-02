import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import genpareto, kstest
from typing import Dict, Any, Tuple, Optional
import pickle
import matplotlib.pyplot as plt
import os

# Grid-search cho MỘT series
def fit_best_marginal(y, search_params=None):
    """
    Grid search ARIMA + {GARCH, GJR, EGARCH} + {t, ged, skewt} cho 1 series.
    Trả về: dict {'p','d','q','vol','o','dist','pval_pit','aic','model','arima','pit','std_resid'}
    """
    # ---- helpers nội bộ (để chỉ cần 2 hàm public) ----
    def ljungbox_p(x, lag=10):
        x = pd.Series(x).dropna()
        # Không print cảnh báo statsmodels
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(acorr_ljungbox(x, lags=[lag], return_df=True)['lb_pvalue'].iloc[0])

    # PIT dùng đúng phân phối của arch (tránh sai tham số hóa)
    from arch.univariate.distribution import StudentsT, GeneralizedError, SkewStudent
    def pit_from_arch_dist(stdresid, dist_name, params):
        x = np.asarray(pd.Series(stdresid).dropna(), float)
        if dist_name == 't':
            dist = StudentsT();  par = [float(params['nu'])]
        elif dist_name == 'ged':
            dist = GeneralizedError();  par = [float(params['nu'])]
        elif dist_name == 'skewt':
            dist = SkewStudent()
            nu = float(params.get('eta', params.get('nu')))
            lam = float(params.get('lambda', params.get('skew')))
            par = [nu, lam]
        else:
            raise ValueError('Unsupported dist')
        return dist.cdf(x, par)

    # ---- tham số mặc định - EXPANDED GRID ----
    defaults = dict(
        p_list=(0,1,2,3),       # thêm AR(3) để capture hết tự tương quan
        q_list=(0,1,2),         # thêm MA(2)
        d_list=(0,1),           # thử cả stationary (d=0) và difference (d=1)
        vol_list=('GARCH','EGARCH','GARCH'),  # GJR=GARCH với o=1
        asym_o=(1,0,0),         # o=1 cho GJR; 0 cho EGARCH & GARCH
        vol_p_list=(1,2),       # thêm GARCH(2,1), GARCH(2,2)
        vol_q_list=(1,2),
        dist_list=('t','ged','skewt'),
        lag_lb=15,              # kiểm tra lag dài hơn
        min_obs=150
    )
    P = {**defaults, **(search_params or {})}

    y = pd.Series(y).dropna()
    best = None

    for d in P['d_list']:
        for p in P['p_list']:
            for q in P['q_list']:
                # 1) Ước lượng mean bằng ARIMA
                try:
                    arima = SARIMAX(
                        y, order=(p,d,q), trend='c',
                        enforce_stationarity=False, enforce_invertibility=False
                    ).fit(disp=False)
                except Exception:
                    continue

                resid_mean = pd.Series(arima.resid).dropna()
                if resid_mean.size < P['min_obs']:
                    continue

                # 2) Volatility: GARCH/GJR/EGARCH với nhiều cấu hình (p,q)
                for vol, o in zip(P['vol_list'], P['asym_o']):
                    for vol_p in P['vol_p_list']:
                        for vol_q in P['vol_q_list']:
                            for dist in P['dist_list']:
                                try:
                                    am = arch_model(resid_mean, mean='Zero', vol=vol, 
                                                   p=vol_p, o=o, q=vol_q, dist=dist)
                                    res = am.fit(disp='off', update_freq=0, show_warning=False)
                                    stdr = pd.Series(res.std_resid).dropna()

                                    # 3) PIT đúng phân phối
                                    pit = pit_from_arch_dist(stdr, dist, res.params)

                                    # 4) chấm điểm theo Ljung-Box trên PIT (+ tie-break AIC)
                                    pval_pit = ljungbox_p(pit, lag=P['lag_lb'])
                                    aic = float(res.aic)

                                    cand = {
                                        'p':p, 'd':d, 'q':q, 'vol':vol, 'o':o, 'dist':dist,
                                        'vol_p': vol_p, 'vol_q': vol_q,  # lưu thêm GARCH order
                                        'pval_pit': float(pval_pit), 'aic': aic,
                                        'model': res, 'arima': arima,
                                        'pit': pit, 'std_resid': stdr
                                    }

                                    if best is None:
                                        best = cand
                                    else:
                                        # Ưu tiên p-value PIT lớn hơn; nếu gần bằng thì AIC nhỏ hơn
                                        if (cand['pval_pit'] > best['pval_pit']) or \
                                           (np.isclose(cand['pval_pit'], best['pval_pit']) and cand['aic'] < best['aic']):
                                            best = cand
                                except Exception:
                                    continue
    return best

def fit_best_marginal_with_diagnostics(y, search_params=None):
    """
    Enhanced version with residual diagnostics scoring.
    Includes ARCH test, Ljung-Box test on residuals and squared residuals.
    """
    from statsmodels.stats.diagnostic import het_arch
    
    # Get base result from standard grid search
    base_result = fit_best_marginal(y, search_params)
    if base_result is None:
        return None
        
    # Add residual diagnostics
    try:
        std_resid = base_result['std_resid'].dropna()
        
        # ARCH test on standardized residuals  
        arch_result = het_arch(std_resid, nlags=10)
        arch_pval = arch_result[1]  # p-value is second element
        
        # Ljung-Box test on standardized residuals  
        lb_resid = acorr_ljungbox(std_resid, lags=[10], return_df=True)
        ljungbox_pval = float(lb_resid['lb_pvalue'].iloc[0])
        
        # ARCH test on squared standardized residuals
        arch2_result = het_arch(std_resid**2, nlags=10) 
        arch2_pval = arch2_result[1]  # p-value is second element
        
        # Ljung-Box test on squared standardized residuals
        lb_resid2 = acorr_ljungbox(std_resid**2, lags=[10], return_df=True) 
        ljungbox2_pval = float(lb_resid2['lb_pvalue'].iloc[0])
        
        # Add diagnostic results
        base_result['arch_pval'] = float(arch_pval)
        base_result['ljungbox_pval'] = ljungbox_pval
        base_result['arch2_pval'] = float(arch2_pval)
        base_result['ljungbox2_pval'] = ljungbox2_pval
        
        # Compute combined diagnostic score (higher is better)
        diag_score = (arch_pval + ljungbox_pval + arch2_pval + ljungbox2_pval) / 4
        base_result['diagnostic_score'] = diag_score
        
    except Exception as e:
        print(f"Warning: Could not compute diagnostics: {e}")
        base_result['arch_pval'] = 0.0
        base_result['ljungbox_pval'] = 0.0 
        base_result['arch2_pval'] = 0.0
        base_result['ljungbox2_pval'] = 0.0
        base_result['diagnostic_score'] = 0.0
    
    return base_result


# Chạy cho TOÀN BỘ DataFrame
def grid_search_marginal_model(log_returns_scaled, search_params=None):
    """
    Chạy fit_best_marginal cho từng cột trong DataFrame.
    Trả về:
      - summary_df: DataFrame (Ticker, p,d,q,vol,o,dist,pval_pit,aic)
      - models: dict {Ticker: dict_best} chứa cả model, arima, pit, std_resid
    """
    rows = []
    models = {}

    for ticker in log_returns_scaled.columns:
        print(f"Fitting enhanced model for {ticker}...")
        best = fit_best_marginal_with_diagnostics(log_returns_scaled[ticker], search_params)
        if best is None:
            continue
        row = {
            'Ticker': ticker,
            'p': best['p'], 'd': best['d'], 'q': best['q'],
            'vol': best['vol'], 'o': best['o'], 'dist': best['dist'],
            'vol_p': best.get('vol_p', 1), 'vol_q': best.get('vol_q', 1), 
            'pval_pit': best['pval_pit'], 'aic': best['aic'],
            'arch_pval': best.get('arch_pval', np.nan),
            'ljungbox_pval': best.get('ljungbox_pval', np.nan),
            'arch2_pval': best.get('arch2_pval', np.nan),
            'ljungbox2_pval': best.get('ljungbox2_pval', np.nan),
            'diagnostic_score': best.get('diagnostic_score', np.nan)
        }
        rows.append(row)
        models[ticker] = best

    summary_df = pd.DataFrame(rows).sort_values(['pval_pit', 'aic'], ascending=[False, True]).reset_index(drop=True)
    return summary_df, models


def diagnose_residuals(std_resids, lags=10, plot=False):
    """
    Perform ARCH and Ljung-Box tests for standardized residuals and squared residuals.
    Đầu vào có thể là:
      - dict {ticker: std_resid} (chuẩn cũ)
      - dict {ticker: model_dict} (output từ grid_search_marginal_model)
    """
    diag_results = {}
    # Tự động nhận diện đầu vào
    for ticker, val in std_resids.items():
        # Nếu là dict model (có key 'std_resid'), lấy standardized residuals
        if isinstance(val, dict) and 'std_resid' in val:
            resid = pd.Series(val['std_resid']).dropna()
        else:
            resid = pd.Series(val).dropna()
        result = {}
        # ARCH test
        try:
            result['arch_pvalue'] = het_arch(resid, nlags=lags)[1]
        except Exception as e:
            result['arch_pvalue'] = np.nan
            print(f"{ticker}: ARCH test error: {e}")
        try:
            result['arch_pvalue_sq'] = het_arch(resid**2, nlags=lags)[1]
        except Exception as e:
            result['arch_pvalue_sq'] = np.nan
            print(f"{ticker}: ARCH^2 test error: {e}")
        # Ljung-Box test
        try:
            result['lb_pvalue'] = acorr_ljungbox(resid, lags=[lags], return_df=True)['lb_pvalue'].iloc[0]
        except Exception as e:
            result['lb_pvalue'] = np.nan
            print(f"{ticker}: Ljung-Box test error: {e}")
        try:
            result['lb_pvalue_sq'] = acorr_ljungbox(resid**2, lags=[lags], return_df=True)['lb_pvalue'].iloc[0]
        except Exception as e:
            result['lb_pvalue_sq'] = np.nan
            print(f"{ticker}: Ljung-Box^2 test error: {e}")
        diag_results[ticker] = result
        print(f"{ticker}: ARCH p={result['arch_pvalue']:.3f}, Ljung-Box p={result['lb_pvalue']:.3f}, "
              f"ARCH^2 p={result['arch_pvalue_sq']:.3f}, Ljung-Box^2 p={result['lb_pvalue_sq']:.3f}")
    return diag_results


def _prep_series(std_resids):
    if isinstance(std_resids, pd.DataFrame):
        return {c: std_resids[c].dropna().astype(float).values for c in std_resids.columns}
    return {k: pd.Series(v).dropna().astype(float).values for k, v in std_resids.items()}

def _runs_declustering(x: np.ndarray, u: float, r: int) -> np.ndarray:
    ex = x > u
    clusters, cur, below = [], [], 0
    for xi, flag in zip(x, ex):
        if flag:
            cur.append(xi); below = 0
        else:
            if cur:
                below += 1
                if below >= r:
                    clusters.append(max(cur))
                    cur, below = [], 0
    if cur: clusters.append(max(cur))
    return (np.asarray(clusters, float) - u) if clusters else np.empty(0, float)

def _exceedances(x: np.ndarray, u: float, decluster_runs: Optional[int]) -> np.ndarray:
    return (x[x > u] - u) if decluster_runs is None else _runs_declustering(x, u, decluster_runs)

def _fit_gpd_exc(exc: np.ndarray):
    exc = np.asarray(exc, float)
    exc = exc[np.isfinite(exc)]
    if exc.size == 0:
        return None
    # KHÓA loc=0 (chuẩn POT vì đã trừ ngưỡng)
    c, loc, s = genpareto.fit(exc, floc=0)
    ll = float(np.sum(genpareto.logpdf(exc, c, 0, s)))
    k = 2  # (shape, scale)
    aic = -2.0 * ll + 2 * k
    bic = -2.0 * ll + k * np.log(exc.size)
    ks_p = float(kstest(exc, 'genpareto', args=(c, 0, s)).pvalue)
    return {'shape': float(c), 'scale': float(s), 'loglik': ll,
            'aic': aic, 'bic': bic, 'ks_p': ks_p, 'n_exc': int(exc.size),
            'mrl': float(exc.mean())}

# ---------------- Scoring 1 q cho 1 đuôi ----------------
def _score_gpd_for_q(x, q, min_exc=30, decluster_runs: Optional[int] = None):
    u = float(np.quantile(x, q))
    exc = _exceedances(x, u, decluster_runs)
    n = exc.size
    if n < min_exc:
        return {'q': float(q), 'thr': u, 'shape': np.nan, 'scale': np.nan,
                'n_exc': int(n), 'ks_p': np.nan, 'aic': np.nan}
    fit = _fit_gpd_exc(exc)
    return {'q': float(q), 'thr': u, 'shape': fit['shape'], 'scale': fit['scale'],
            'n_exc': fit['n_exc'], 'ks_p': fit['ks_p'], 'aic': fit['aic']}

def _pick_best_q(x, q_grid, min_exc=30, decluster_runs: Optional[int] = None):
    best = None
    for q in np.asarray(q_grid, float):
        res = _score_gpd_for_q(x, q, min_exc=min_exc, decluster_runs=decluster_runs)
        # bỏ các hàng không đủ dữ liệu
        if np.isnan(res['ks_p']): 
            continue
        if (best is None) or (res['ks_p'] > best['ks_p']) or \
           (np.isclose(res['ks_p'], best['ks_p']) and res['aic'] < best['aic']):
            best = res
    return best  # có thể None nếu mọi q đều thiếu exceedances

# ---------------- 1) Fit + chọn ngưỡng (per ticker, per tail) ----------------
def fit_evt_auto_all(std_resids,
                     q_grid_left=np.linspace(0.95, 0.995, 10),
                     q_grid_right=np.linspace(0.95, 0.995, 10),
                     min_exc=30,
                     decluster_runs: Optional[int] = None):
    """
    Trả về:
      - evt_dict: {ticker: {'left': {...}, 'right': {...}}} với keys: q, thr, shape, scale, n_exc, ks_p, aic
      - summary_df: bảng tổng hợp (mỗi hàng = 1 đuôi / 1 ticker)
    """
    series = _prep_series(std_resids)
    evt_dict, rows = {}, []

    for ticker, r in series.items():
        # left: losses = -r
        left_best  = _pick_best_q(-r, q_grid_left,  min_exc=min_exc, decluster_runs=decluster_runs)
        # right: gains = +r
        right_best = _pick_best_q(+r, q_grid_right, min_exc=min_exc, decluster_runs=decluster_runs)

        evt_dict[ticker] = {'left': left_best, 'right': right_best}
        if left_best  is not None: rows.append({'Ticker': ticker, 'Side': 'left',  **left_best})
        if right_best is not None: rows.append({'Ticker': ticker, 'Side': 'right', **right_best})

    summary_df = pd.DataFrame(rows).sort_values(['Ticker','Side']).reset_index(drop=True)
    return evt_dict, summary_df

# ---------------- 2) Diagnostics lưới ngưỡng (không chọn lại ngưỡng) ----------------
def diagnose_evt(std_resids,
                 evt_params: Dict[str, Dict[str, Any]],
                 q_grid_left  = np.linspace(0.90, 0.995, 25),
                 q_grid_right = np.linspace(0.90, 0.995, 25),
                 min_exc: int = 30,
                 decluster_runs: Optional[int] = None
                 ) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], pd.DataFrame]:
    """
    Tạo DIAGNOSTIC grids cho mỗi đuôi (q,u,shape,scale,ks_p,aic,bic,loglik,n_exc,mrl),
    và tính lại metrics TẠI NGƯỠNG ĐÃ CHỌN (evt_params). Không chọn lại ngưỡng, không vẽ.
    """
    series = _prep_series(std_resids)
    grids: Dict[str, Dict[str, pd.DataFrame]] = {}
    rows_best = []

    for ticker, r in series.items():
        grids[ticker] = {}

        # LEFT grid
        X_left = -r
        left_rows = []
        for q in np.asarray(q_grid_left, float):
            u = float(np.quantile(X_left, q))
            exc = _exceedances(X_left, u, decluster_runs)
            if exc.size >= min_exc:
                fit = _fit_gpd_exc(exc)
                left_rows.append({'q': q, 'u': u, **fit})
            else:
                left_rows.append({'q': q, 'u': u, 'shape': np.nan, 'scale': np.nan,
                                  'ks_p': np.nan, 'aic': np.nan, 'bic': np.nan,
                                  'loglik': np.nan, 'n_exc': int(exc.size),
                                  'mrl': float(exc.mean()) if exc.size>0 else np.nan})
        grids[ticker]['left'] = pd.DataFrame(left_rows)

        # RIGHT grid
        X_right = +r
        right_rows = []
        for q in np.asarray(q_grid_right, float):
            u = float(np.quantile(X_right, q))
            exc = _exceedances(X_right, u, decluster_runs)
            if exc.size >= min_exc:
                fit = _fit_gpd_exc(exc)
                right_rows.append({'q': q, 'u': u, **fit})
            else:
                right_rows.append({'q': q, 'u': u, 'shape': np.nan, 'scale': np.nan,
                                   'ks_p': np.nan, 'aic': np.nan, 'bic': np.nan,
                                   'loglik': np.nan, 'n_exc': int(exc.size),
                                   'mrl': float(exc.mean()) if exc.size>0 else np.nan})
        grids[ticker]['right'] = pd.DataFrame(right_rows)

        # Metrics tại NGƯỠNG ĐÃ CHỌN
        for side, X in [('left', X_left), ('right', X_right)]:
            p = (evt_params.get(ticker) or {}).get(side) or {}
            q_star, u_star = p.get('q'), p.get('thr')
            # an toàn kiểu/NaN
            if q_star is None or u_star is None:
                continue
            if not (np.isfinite(q_star) and np.isfinite(u_star)):
                continue
            exc_star = _exceedances(X, float(u_star), decluster_runs)
            if exc_star.size >= min_exc:
                fit_star = _fit_gpd_exc(exc_star)
                rows_best.append({'Ticker': ticker, 'Side': side,
                                  'q': float(q_star), 'u': float(u_star),
                                  'shape': fit_star['shape'], 'scale': fit_star['scale'],
                                  'ks_p': fit_star['ks_p'], 'aic': fit_star['aic'],
                                  'bic': fit_star['bic'], 'loglik': fit_star['loglik'],
                                  'n_exc': fit_star['n_exc']})

    at_best_df = pd.DataFrame(rows_best).sort_values(['Ticker','Side']).reset_index(drop=True)
    return grids, at_best_df


def plot_mrl_multi(tickers,
                   side="left",
                   grids_dir="../models/evt/grids",
                   at_best_path="../models/evt/evt_at_best.csv",
                   cols=3,
                   figsize_factor=(6, 4)
                ):
    
    at_best = None
    if os.path.exists(at_best_path):
        at_best = pd.read_csv(at_best_path)

    n = len(tickers)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_factor[0], rows * figsize_factor[1]), sharex=False)
    axes = np.atleast_1d(axes).ravel()

    for i, ticker in enumerate(tickers):
        ax = axes[i]
        grid_path = os.path.join(grids_dir, f"{ticker}_{side}_grid.csv")

        if not os.path.exists(grid_path):
            ax.set_title(f"{ticker} ({side}) - thiếu file grid")
            ax.axis("off")
            continue

        df = pd.read_csv(grid_path)
        if not set(["u","mrl"]).issubset(df.columns):
            ax.set_title(f"{ticker} ({side}) - thiếu cột u/mrl")
            ax.axis("off")
            continue

        d = df[["u","mrl"]].dropna().sort_values("u")
        if d.empty:
            ax.set_title(f"{ticker} ({side}) - không có MRL hợp lệ")
            ax.axis("off")
            continue

        # MRL vs u
        ax.plot(d["u"].values, d["mrl"].values, marker="o")
        ax.set_title(f"MRL vs u — {ticker} ({side})")
        ax.set_xlabel("u (threshold)")
        ax.set_ylabel("Mean Residual Life")
        ax.grid(True, linestyle="--", alpha=0.4)

        # Vẽ đường dọc tại u* nếu có
        if at_best is not None:
            m = (at_best["Ticker"] == ticker) & (at_best["Side"] == side)
            if m.any():
                u_star = float(at_best.loc[m, "u"].iloc[0])
                ax.axvline(u_star, linestyle="--", alpha=0.85)
                ymax = np.nanmax(d["mrl"].values)
                ax.text(u_star, ymax*0.95, "u*", rotation=90, va="top", ha="right", fontsize=9)

    # Ẩn subplot thừa
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    import os, pickle

    # ==== folders ====
    garch_dir = "models/garch"
    evt_dir   = "models/evt"
    evt_grids = os.path.join(evt_dir, "grids")
    os.makedirs(garch_dir, exist_ok=True)
    os.makedirs(evt_dir,   exist_ok=True)
    os.makedirs(evt_grids, exist_ok=True)

    # 1) Load log returns
    log_returns = pd.read_csv("data/processed/log_returns.csv", index_col=0)
    print("Loaded log_returns shape:", log_returns.shape)

    # 2) Fix index to avoid SARIMAX frequency warning
    index_path = "data/processed/log_returns_index.csv"
    if os.path.exists(index_path):
        date_index = pd.read_csv(index_path, index_col=0).iloc[:, 0]
        if len(date_index) == len(log_returns):
            log_returns.index = pd.to_datetime(date_index.values)
            log_returns.index = pd.RangeIndex(len(log_returns))
        else:
            print("Warning: log_returns_index.csv không khớp; dùng RangeIndex.")
            log_returns.index = pd.RangeIndex(len(log_returns))
    else:
        if isinstance(log_returns.index, pd.DatetimeIndex) and log_returns.index.freq is None:
            log_returns.index = pd.RangeIndex(len(log_returns))

    # 3) Scaling
    log_returns_scaled = log_returns * 100

    # 4) Marginal grid search (ARIMA + GARCH/GJR/EGARCH + dist)
    summary_df, models = grid_search_marginal_model(log_returns_scaled)
    print(f"Grid search hoàn tất: {len(summary_df)}/{log_returns.shape[1]} mã thành công.")
    summary_df.to_csv(os.path.join(garch_dir, "marginal_model_summary.csv"), index=False)
    with open(os.path.join(garch_dir, "marginal_model_full.pkl"), "wb") as f:
        pickle.dump(models, f)

    # 5) Lấy standardized residuals & PIT, lưu vào models/garch/
    std_resids_df = pd.DataFrame({k: v['std_resid'] for k, v in models.items()})
    std_resids_df.to_csv(os.path.join(garch_dir, "std_resids.csv"))
    pit_dict = {k: v['pit'] for k, v in models.items()}
    with open(os.path.join(garch_dir, "conditional_pit.pkl"), "wb") as f:
        pickle.dump(pit_dict, f)
    print("Saved std_resids.csv & conditional_pit.pkl vào models/garch/")

    # 6) EVT auto-fit: chọn q/u tốt nhất và fit GPD cho HAI ĐUÔI
    evt_dict, evt_summary = fit_evt_auto_all(
        std_resids_df,
        q_grid_left=np.linspace(0.95, 0.995, 10),
        q_grid_right=np.linspace(0.95, 0.995, 10),
        min_exc=30,
        # decluster_runs=None,
    )
    evt_summary.to_csv(os.path.join(evt_dir, "evt_summary_auto.csv"), index=False)
    print("Saved models/evt/evt_summary_auto.csv")

    # 7) Lưu EVT params tương thích hybrid_ppf/copula_simulation.py: [thr, shape, scale]
    evt_params_compat = {}
    for ticker, tails in evt_dict.items():
        L = tails.get('left')
        R = tails.get('right')
        evt_params_compat[ticker] = {
            'left' : [L['thr'], L['shape'], L['scale']] if L else [np.nan, np.nan, np.nan],
            'right': [R['thr'], R['shape'], R['scale']] if R else [np.nan, np.nan, np.nan],
        }
    with open(os.path.join(evt_dir, "evt_results.pkl"), "wb") as f:
        pickle.dump(evt_params_compat, f)
    print("Saved models/evt/evt_results.pkl")

    # 8) Diagnostics grids
    grids, at_best_df = diagnose_evt(
        std_resids_df,
        evt_params=evt_dict,
        q_grid_left=np.linspace(0.90, 0.995, 25),
        q_grid_right=np.linspace(0.90, 0.995, 25),
        min_exc=30,
        # decluster_runs=None,
    )
    at_best_df.to_csv(os.path.join(evt_dir, "evt_at_best.csv"), index=False)
    for ticker, d in grids.items():
        d['left'].to_csv(os.path.join(evt_grids, f"{ticker}_left_grid.csv"), index=False)
        d['right'].to_csv(os.path.join(evt_grids, f"{ticker}_right_grid.csv"), index=False)
    print("Saved EVT diagnostic grids vào models/evt/grids/")

    print("=== DONE: artifacts saved to models/garch/ và models/evt/ ===")


