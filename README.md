# GARCH-EVT-Copula Portfolio Risk Modeling

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## 📊 Mô tả Dự án

Dự án này thực hiện mô hình hóa rủi ro danh mục đầu tư cho các cổ phiếu Việt Nam (FPT, HPG, MBB, MWG, VIC) sử dụng phương pháp kết hợp **GARCH-EVT-Copula**. Pipeline bao gồm ba bước chính:

1. **GARCH Modeling**: Mô hình hóa biến động và heteroskedasticity của log returns
2. **EVT (Extreme Value Theory)**: Mô hình hóa phân phối tail (đuôi) của standardized residuals
3. **Vine Copula**: Mô hình hóa cấu trúc phụ thuộc đa biến giữa các tài sản

## 🎯 Mục tiêu

- Mô hình hóa chính xác rủi ro tail của từng tài sản và danh mục
- Tái tạo cấu trúc phụ thuộc thực tế giữa các cổ phiếu
- Mô phỏng Monte Carlo cho phân tích rủi ro và tính toán VaR
- Kiểm định và validation toàn diện pipeline

## 🚀 Tính năng chính

- **Data Processing**: Tự động merge và clean dữ liệu từ multiple CSV files
- **Marginal Modeling**: ARIMA+GARCH với EVT cho tail modeling
- **Dependence Modeling**: R-vine copula với multiple families (Gaussian, Student, Clayton, Gumbel)
- **Risk Simulation**: Monte Carlo simulation cho portfolio VaR
- **Comprehensive Validation**: Rolling VaR backtest, Kupiec test, Christoffersen test, PIT diagnostics

## 📁 Cấu trúc Dự án

```
├── LICENSE                     <- Giấy phép mã nguồn mở
├── Makefile                    <- Commands tiện ích
├── README.md                   <- README này
├── pyproject.toml             <- Cấu hình dự án Python
├── requirements.txt           <- Dependencies Python
│
├── data/
│   ├── raw/                   <- Dữ liệu gốc (CSV files từ XSTC)
│   │   ├── Download Data - STOCK_VN_XSTC_FPT*.csv
│   │   ├── Download Data - STOCK_VN_XSTC_HPG*.csv
│   │   ├── Download Data - STOCK_VN_XSTC_MBB*.csv
│   │   ├── Download Data - STOCK_VN_XSTC_MWG*.csv
│   │   └── Download Data - STOCK_VN_XSTC_VIC*.csv
│   └── processed/             <- Dữ liệu đã xử lý
│       ├── price_cleaned.csv        <- Giá đã clean
│       ├── log_returns.csv          <- Log returns
│       ├── log_returns_index.csv    <- Log returns với index
│       └── simulated_copula.csv     <- Dữ liệu mô phỏng
│
├── src/                       <- Source code chính
│   ├── __init__.py
│   ├── data_prep.py          <- Xử lý và clean dữ liệu
│   ├── marginal_modeling.py  <- ARIMA+GARCH+EVT modeling
│   ├── copula_simulation.py  <- PIT computation và copula fitting
│   └── validation.py         <- Validation và backtesting functions
│
├── notebooks/                 <- Jupyter notebooks workflow
│   ├── eda.ipynb             <- Exploratory Data Analysis
│   ├── marginal_garch_evt.ipynb  <- Marginal modeling workflow
│   ├── copula.ipynb          <- Copula modeling và simulation
│   └── pipeline_validation.ipynb <- Comprehensive validation
│
├── models/                    <- Trained models và outputs
│   ├── garch/                <- GARCH model outputs
│   ├── evt/                  <- EVT marginal distributions
│   ├── copula/               <- Copula models và PIT data
│
├── validation/                <- Validation results
│   ├── rolling_var_plots/    <- Rolling VaR plots
│   ├── pit/                  <- PIT diagnostics plots
│   ├── validation_summary.json    <- Tổng hợp validation results
│   └── *.png                 <- Dependence comparison plots
│
├── reports/                   <- Generated analysis reports
├── tests/                     <- Unit tests
└── docs/                      <- Project documentation
```

## 🔧 Cài đặt và Sử dụng

### 1. Cài đặt Dependencies

```bash
# Clone repository
git clone <repository-url>
cd garch-evt-copula

# Tạo virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# hoặc
.venv\Scripts\activate     # Windows

# Cài đặt packages
pip install -r requirements.txt
pip install -e .
```

### 2. Chạy Pipeline

#### Option 1: Sử dụng Jupyter Notebooks (Recommended)

```bash
jupyter lab
```

Chạy notebooks theo thứ tự:
1. `eda.ipynb` - Exploratory data analysis
2. `marginal_garch_evt.ipynb` - Marginal modeling  
3. `copula.ipynb` - Copula modeling và simulation
4. `pipeline_validation.ipynb` - Validation và backtesting

#### Option 2: Sử dụng Python Scripts

```python
# 1. Data preparation
from src.data_prep import load_data, clean_data, calculate_returns
price_data = load_data("data/raw")
returns = calculate_returns(price_data)

# 2. Marginal modeling
from src.marginal_modeling import fit_garch_evt_models
models = fit_garch_evt_models(returns)

# 3. Copula simulation
from src.copula_simulation import fit_copula_and_simulate
simulated_data = fit_copula_and_simulate()

# 4. Validation
from src.validation import run_validation_pipeline
validation_results = run_validation_pipeline()
```

## 📊 Workflow Chi tiết

### 1. Data Processing (`data_prep.py`)
- Merge multiple CSV files với smart date conversion
- Clean outliers và missing values
- Calculate log returns và handle holidays

### 2. Marginal Modeling (`marginal_modeling.py`)
- **ARIMA fitting** cho mean equation
- **GARCH modeling** với multiple specifications (GARCH, GJR-GARCH, EGARCH)
- **EVT fitting** với Generalized Pareto Distribution cho tails
- **Semiparametric CDF** combining empirical và parametric parts

### 3. Copula Modeling (`copula_simulation.py`)
- **PIT transformation** từ standardized residuals
- **R-vine copula fitting** với multiple families
- **Monte Carlo simulation** cho risk analysis

### 4. Validation (`validation.py`)
- **Rolling VaR backtest** với 500-day window
- **Kupiec test** cho unconditional coverage
- **Christoffersen test** cho independence
- **PIT diagnostics** với Ljung-Box tests
- **Dependence comparison** (Kendall tau, tail dependence)

## 📈 Kết quả Chính

### Model Performance
- **VaR Accuracy**: Tất cả assets đạt unconditional coverage test (p-value > 0.05)
- **Independence**: Không có violation clustering (Christoffersen test passed)
- **PIT Quality**: Hầu hết dimensions đạt independence test
- **Dependence Reproduction**: Frob norm khác biệt Kendall tau < 0.1

### Risk Metrics
- **Individual VaR** cho từng asset
- **Portfolio VaR** với copula aggregation
- **Tail dependence** upper/lower bounds
- **Energy distance** cho distribution comparison

## 🔬 Technical Details

### Key Libraries
- `arch`: GARCH modeling
- `pyvinecopulib`: Vine copula fitting và simulation
- `scipy.stats`: Statistical tests và distributions
- `statsmodels`: Time series analysis
- `pandas/numpy`: Data manipulation

### Model Specifications
- **GARCH**: GJR-GARCH(1,1) với Student-t innovations
- **EVT**: GPD cho tails (threshold = 95th percentile)
- **Copula**: R-vine với Gaussian, Student, Clayton, Gumbel families

## 📝 Citation

Nếu sử dụng code này trong nghiên cứu, vui lòng cite:

```
@misc{garch-evt-copula-2024,
  author = {Duong N.C.K},
  title = {GARCH-EVT-Copula Portfolio Risk Modeling for Vietnamese Stocks},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/dylanng3/garch-evt-copula}
}
```

## 📄 License

Dự án được phát hành dưới [MIT License](LICENSE).

## 🤝 Contributing

Contributions được hoan nghênh! Vui lòng:
1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## 📧 Contact

**Author**: Duong N.C.K  
**Project**: GARCH-EVT-Copula Portfolio Risk Modeling

---

