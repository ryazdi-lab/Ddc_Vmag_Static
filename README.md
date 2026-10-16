# AC Voltage Magnitude Prediction from DC Dependency Matrices

An empirical benchmark of deep learning and gradient boosting for predicting complete bus voltage magnitude profiles from DC power flow dependency matrices.

The study covers **six IEEE test systems (14, 30, 39, 57, 118, and 300 buses)** and **six model families**. The supplied experiment archive contains **34 final test evaluations out of 36 model–system combinations**.

## Task

Given a DC dependency matrix, estimate one AC voltage magnitude per bus:

$$
\hat{\mathbf v}=f_\theta(\mathbf D_{\mathrm{dc}}),
\qquad \mathbf D_{\mathrm{dc}}\in\mathbb R^{N\times N},
\quad \hat{\mathbf v}\in\mathbb R^N.
$$

Voltage magnitudes are expressed in per unit (p.u.). Each test system is trained separately. This work predicts a **voltage vector**, rather than the reactive dependency matrix targeted in the earlier DDC-to-DQ study.

## Models

| Model | Main mechanism |
|---|---|
| Plain U-Net | Convolutional encoder–decoder with skip connections and a voltage-vector head |
| Attention U-Net | Channel/spatial attention on skips and bottleneck self-attention |
| Centrality-Aware U-Net | Attention backbone with an auxiliary head supervised using input DDC row/column sums |
| XGBoost | One independent tree ensemble per output bus, using the flattened dependency matrix |
| Global Transformer | Global attention over matrix-entry tokens; implemented in `DQTransformer` with the Vmag head |
| Axial Transformer | Alternating row and column attention with a voltage-vector head |

The voltage heads have linear outputs, allowing voltages above 1 p.u. Centrality supervision is derived from **input DDC**, not from an AC reactive dependency target or from predicted voltage.

## Data

Each system has the following recorded split sizes:

| Split | Scenarios |
|---|---:|
| Training | 7,734 |
| Validation | 1,657 |
| Test | 1,659 |
| Total | 11,050 |

The loaders expect NPZ files containing:

- `X`: dependency matrices with stored shape `(N, N, S)`.
- `Vmag`: voltage targets with stored shape `(N, S)`.

Neural loaders convert these arrays to `(S, 1, N, N)` and `(S, N)`. XGBoost flattens each input matrix to `N²` features and trains **N output regressors**.

**Data availability:** the supplied archive includes executed notebooks, model code, training histories, and XGBoost result JSON files. Raw NPZ datasets, trained checkpoints, and the original scenario-generation code are not included. Training therefore requires obtaining those datasets and updating the notebook paths. Exact perturbation distributions and generator-voltage sampling assumptions cannot be verified from this archive.

## Results

XGBoost achieves the lowest reported RMSE and MAE in every test system. Its results are:

| IEEE system | Test R² | RMSE (p.u.) | MAE (p.u.) |
|---|---:|---:|---:|
| 14 buses | 0.9890 | 0.002208 | 0.001189 |
| 30 buses | 0.9832 | 0.002809 | 0.001763 |
| 39 buses | 0.9657 | 0.004635 | 0.002726 |
| 57 buses | 0.9961 | 0.005140 | 0.002981 |
| 118 buses | 0.9976 | 0.001139 | 0.000481 |
| 300 buses | 0.9792 | 0.007310 | 0.003085 |

Other observations from the recorded runs:

- Attention U-Net reduces RMSE relative to plain U-Net in all six systems; MAE improves in five.
- Axial attention reduces the reported errors relative to global attention in all five completed Transformer comparisons.
- The implemented DC-centrality auxiliary task provides no consistent voltage-accuracy improvement.
- Both 300-bus Transformer notebooks lack a final test evaluation. Validation scores are not substituted for missing test results.

## Training configuration

The standard neural loss combines equally weighted voltage MSE and MAE:

$$
\mathcal L_V=0.5\operatorname{MSE}(\hat{\mathbf v},\mathbf v)
+0.5\operatorname{MAE}(\hat{\mathbf v},\mathbf v).
$$

The centrality-aware loss adds `0.3 × (betweenness MSE + closeness MSE)` using DDC-derived auxiliary targets.

| Setting | U-Net variants | Transformers |
|---|---|---|
| Optimizer | Adam | AdamW |
| Initial learning rate | `1e-3` | `5e-4` |
| Weight decay | `1e-4` | `1e-4` |
| Maximum epochs | 100 | 100 |
| Checkpoint selection | Lowest validation voltage RMSE | Highest reported validation R² |
| Schedule | ReduceLROnPlateau | Five-epoch warm-up and cosine decay |

XGBoost uses learning rate `0.05`, row/feature subsampling `0.8`, and early-stopping patience 20. Depths vary from 4 to 6. The implementation raises the estimator limit to at least 600 when early stopping is enabled; nominal estimator settings are not fitted tree counts.

## Authors

- Mohammad Shahraeini — Department of Electrical Engineering, Golestan University, Gorgan, Iran.
- Reza Yazdi — Department of Computer Engineering, Golestan University, Gorgan, Iran.

The manuscript includes the relevant operational-centrality and model references. Publication metadata or a DOI for this benchmark should be added once available.
