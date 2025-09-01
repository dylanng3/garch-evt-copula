import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import genpareto
from scipy.stats import t
import pickle

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

    # ---- tham số mặc định ----
    defaults = dict(
        p_list=(0,1,2),
        q_list=(0,1),
        d_list=(1,0),           # ưu tiên d=1, sau đó thử d=0
        vol_list=('GARCH','EGARCH','GARCH'),  # GJR=GARCH với o=1
        asym_o=(1,0,0),         # o=1 cho GJR; 0 cho EGARCH & GARCH
        dist_list=('t','ged','skewt'),
        lag_lb=10,
        min_obs=150             # bỏ qua nếu residual quá ngắn
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

                # 2) Volatility: GARCH / GJR(o=1) / EGARCH
                for vol, o in zip(P['vol_list'], P['asym_o']):
                    for dist in P['dist_list']:
                        try:
                            am = arch_model(resid_mean, mean='Zero', vol=vol, p=1, o=o, q=1, dist=dist)
                            res = am.fit(disp='off', update_freq=0, show_warning=False)
                            stdr = pd.Series(res.std_resid).dropna()

                            # 3) PIT đúng phân phối
                            pit = pit_from_arch_dist(stdr, dist, res.params)

                            # 4) chấm điểm theo Ljung-Box trên PIT (+ tie-break AIC)
                            pval_pit = ljungbox_p(pit, lag=P['lag_lb'])
                            aic = float(res.aic)

                            cand = {
                                'p':p, 'd':d, 'q':q, 'vol':vol, 'o':o, 'dist':dist,
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


# Chạy cho TOÀN BỘ DataFrame
def grid_search_marginal_model(log_returns_scaled, search_params=None, verbose=True):
    """
    Chạy fit_best_marginal cho từng cột trong DataFrame.
    Trả về:
      - summary_df: DataFrame (Ticker, p,d,q,vol,o,dist,pval_pit,aic)
      - models: dict {Ticker: dict_best} chứa cả model, arima, pit, std_resid
    """
    rows = []
    models = {}

    for ticker in log_returns_scaled.columns:
        best = fit_best_marginal(log_returns_scaled[ticker], search_params, verbose)
        if best is None:
            continue
        row = {
            'Ticker': ticker,
            'p': best['p'], 'd': best['d'], 'q': best['q'],
            'vol': best['vol'], 'o': best['o'], 'dist': best['dist'],
            'pval_pit': best['pval_pit'], 'aic': best['aic']
        }
        rows.append(row)
        models[ticker] = best

    summary_df = pd.DataFrame(rows).sort_values(['pval_pit', 'aic'], ascending=[False, True]).reset_index(drop=True)
    return summary_df, models


def diagnostic_residuals(std_resids, lags=10, plot=False):
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

def fit_evt_two_tails(std_resids, quantile=0.95):
    """
    Fit GPD (Generalized Pareto Distribution) for both tails of standardized residuals.
    Returns dict: {ticker: {'left': [thL, shapeL, scaleL], 'right': [thR, shapeR, scaleR]}}
    """
    evt_results = {}
    for ticker, resid in std_resids.items():
        # Đuôi âm (left)
        losses = -resid.dropna()
        thL = losses.quantile(quantile)
        excL = losses[losses > thL] - thL
        if len(excL) < 10:
            print(f"Warning: Too few left-tail exceedances for {ticker}. Skipping left EVT.")
            left = [float('nan'), float('nan'), float('nan')]
        else:
            shapeL, locL, scaleL = genpareto.fit(excL)
            left = [thL, shapeL, scaleL]
        # Đuôi dương (right)
        gains = resid.dropna()
        thR = gains.quantile(quantile)
        excR = gains[gains > thR] - thR
        if len(excR) < 10:
            print(f"Warning: Too few right-tail exceedances for {ticker}. Skipping right EVT.")
            right = [float('nan'), float('nan'), float('nan')]
        else:
            shapeR, locR, scaleR = genpareto.fit(excR)
            right = [thR, shapeR, scaleR]
        evt_results[ticker] = {'left': left, 'right': right}
    return evt_results


if __name__ == "__main__":
    # Demo: Read log_returns from preprocessed CSV file
        log_returns = pd.read_csv("data/processed/log_returns.csv", index_col=0)
        print("Loaded log_returns shape:", log_returns.shape)

        # Đọc lại index ngày tháng gốc nếu có file
        import os
        index_path = "data/processed/log_returns_index.csv"
        if os.path.exists(index_path):
            date_index = pd.read_csv(index_path, index_col=0)
            date_index = date_index.iloc[:, 0]
            # Đảm bảo thứ tự index khớp với log_returns
            if len(date_index) == len(log_returns):
                log_returns.index = pd.to_datetime(date_index.values)
                # Đã có file index, không cần lưu lại vào models/
                # Chuyển về số nguyên để modeling không cảnh báo
                log_returns.index = pd.RangeIndex(len(log_returns))
            else:
                print("Warning: log_returns_index.csv không khớp số dòng với log_returns, giữ index số nguyên.")
        else:
            # Nếu không có file index, chuyển về số nguyên để tránh cảnh báo
            if isinstance(log_returns.index, pd.DatetimeIndex) and log_returns.index.freq is None:
                log_returns.index = pd.RangeIndex(len(log_returns))

        # Scale log returns (if desired)
        log_returns_scaled = log_returns * 100

        # Grid search ARIMA+GARCH/EGARCH/GJR+dist cho tất cả mã
        summary_df, models = grid_search_marginal_model(log_returns_scaled, verbose=True)
        print(f"Grid search hoàn tất: {len(summary_df)}/{log_returns.shape[1]} mã thành công. Không có cảnh báo hoặc lỗi nghiêm trọng.")

        # Lưu summary và models
        summary_df.to_csv("models/marginal_model_summary.csv", index=False)
        with open("models/marginal_model_full.pkl", "wb") as f:
            pickle.dump(models, f)
        print("Saved marginal_model_summary.csv và marginal_model_full.pkl vào models/")

        # Lấy standardized residuals từ models
        std_resids = {k: v['std_resid'] for k, v in models.items()}

        # Tính và lưu conditional PIT
        pit_dict = {k: v['pit'] for k, v in models.items()}
        with open("models/conditional_pit.pkl", "wb") as f:
            pickle.dump(pit_dict, f)
        print("Saved conditional_pit.pkl vào models/")

        # Check residual diagnostics
        diag_results = diagnostic_residuals(std_resids, lags=10, plot=True)

        # Fit EVT for standardized residuals
        evt_results = fit_evt_two_tails(std_resids, quantile=0.95)
        with open("models/evt_results.pkl", "wb") as f:
            pickle.dump(evt_results, f)
        print("Saved evt_results.pkl vào models/ (correct format for copula_simulation.py)")