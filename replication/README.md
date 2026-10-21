# TVP-LR-NVAR-SV replication notes

Manuscript: Dynamic Low-Rank Network Vector Autoregressions with Stochastic Trends and Volatility.

The stored inputs live in the project `data` folder. The scripts below read those files. They do not re-estimate the Gibbs chains.

## Data

- `data/djia_adj_close_wide_10y.csv`: adjusted closes for the 30 DJIA names.
- `data/djia_components_daily_10y.csv`: long daily panel.

## Gibbs samplers

- `tvp_lr_nvar_sv_6m_forecast.py`: TVP-LR-NVAR-SV.
- `ls_lr_nvar_6m_forecast.py`: LS-LR-NVAR (v2).
- `ls_lr_nvar_spec_gibbs.py`: LS-LR-NVAR (Spec).

Saved operator arrays:

- `data/tvplrnvarsv_A_mean_train.npz`
- `data/lslrnvar_A_mean_train_v2.npz`
- `data/lslrnvar_A_mean_train_spec.npz`

## Tables from stored output

- Table 3, equal-weight portfolio path: `data/forecast_comparison_6m.csv` and `data/forecast_comparison_6m.json`. The TVP column mean of `tvp_mape` is the portfolio path MAPE, 9.37.
- Table C.3, firm rows: reported in the manuscript. The Panel Mean row is the unweighted average of those 30 rows (Path MAPE 8.18, Price MAE 18.44, Return MAE 10.94 pp, Directional Hit 90.0, Random Walk MAPE 9.01, Chen Direct MAPE 8.81). That average is not the Table 3 portfolio path.
- Table 1, SBM community recovery: `data/sim_sbm_three_densities.json`, produced by `run_sim_sbm_three_densities.py`.
- Table 5, connectedness and log scores: `data/aux_steps_12_25.json`, produced by `compare_aux_steps_12_25.py` from the saved operators.

`python replication/reproduce_stored_tables.py` checks the price panel and reprints the stored TVP path-MAPE mean from `forecast_comparison_6m.csv`.

Chain seeds for a full re-estimation are the seeds recorded in `data/tvplrnvarsv_fit_meta.json`, `data/lslrnvar_fit_meta.json`, and the simulation settings JSON files. A public GitHub or Zenodo deposit can be cut from this folder plus `data/` once the archive account is available.
