# Medical Insurance Charge Prediction (End-to-End Data Science)

Predict insurance charges from demographic and behavioral features using a **reproducible, leakage-safe pipeline**.

**Tech**: Python, pandas, NumPy, scikit-learn, Pipelines/ColumnTransformer, Cross-Validation

---

## ✨ What’s inside
- **Leakage-safe pipeline**: in-pipeline feature engineering (`bmi^2`, `age×bmi`, `smoker×bmi`) → encoding/standardization via `ColumnTransformer` → model training & small-grid CV — all in one `Pipeline`.
- **Model zoo**: Linear, Ridge, ElasticNet, Gradient Boosting (K-fold CV).
- **Subgroup audit**: RMSE/MAE by `smoker` and `region` for fairness/robustness.
- **Artifacts**: best model (`models/best_<model>.pkl`), metrics JSON, subgroup JSON, figures.

---

## 📂 Project Structure
```
insurance-pricing/
├─ data/
│  └─ insurance.csv
├─ models/
│  └─ best_<model>.pkl
├─ notebooks/
│  ├─ 01_end_to_end.ipynb          # full code pipeline (train/eval/export/figures)
│  └─ 00_README_tutorial.ipynb     # README-as-notebook (optional)
├─ reports/
│  ├─ metrics_enhanced.json        # per-model metrics + best params
│  ├─ subgroup_metrics.json        # smoker / region audit
│  └─ figures/
│     ├─ residuals_vs_pred.png
│     └─ learning_curve.png
├─ requirements.txt
└─ README.md
```

---

## 🚀 Quickstart

### 1) Environment
```bash
conda create -n insurance-py python=3.11 -y
conda activate insurance-py
pip install -r requirements.txt
```

### 2) Data
Put `insurance.csv` under `data/`. Expected columns:  
`age, sex, bmi, children, smoker, region, charges`.

### 3) Train & Evaluate
Open `notebooks/01_end_to_end.ipynb` and run all cells.  
It will:
- build a single `Pipeline` with in-pipeline feature engineering
- compare Linear / Ridge / ElasticNet / GBRT via K-fold CV
- evaluate on a hold-out set
- export artifacts/metrics/figures

Artifacts go to:
- `models/best_<model>.pkl`
- `reports/metrics_enhanced.json`
- `reports/subgroup_metrics.json`
- `reports/figures/*.png`

---

## 🧪 Reference Results (Hold-out)
- **Linear** baseline: R² **0.8676**, RMSE **4534.26**, MAE **2740.27**  
- **Best model: GBRT** → R² **0.8792**, RMSE **4330.05**, MAE **2450.58**  
- vs. baseline: **RMSE ↓ 4.50%**, **MAE ↓ 10.57%**



---

## 🧰 Implementation Notes
- No lambdas or `feature_names_out` in `FunctionTransformer` → **pickle-safe** models.
- RMSE helper is **compatible** with older scikit-learn (no `squared=` required).
- To scriptify, copy the big training cell into `src/train.py` and run:
  ```bash
  python src/train.py
  ```

---

## 📜 License
MIT 
