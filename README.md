# Turbo_Net

MATLAB code for the paper **"Turbo_Net: A lightweight convolutional network for diagnosing lubricating-oil starvation of diesel engine turbochargers from on-engine knock-sensor vibration"**.

Turbo_Net classifies four oil-supply levels of a journal-bearing turbocharger from the vibration measured by four knock sensors. Three sensors are mounted on the turbocharger housing (A, B, C) and one reference sensor on the cylinder-head cover (D).

| Level | Oil-supply pressure |
|---|---|
| P0 (normal) | 4.0 kg/cm² (392 kPa) |
| P1 | 3.0 kg/cm² (294 kPa) |
| P2 | 2.0 kg/cm² (196 kPa) |
| P3 | 1.5 kg/cm² (147 kPa) |

| Network | Input | Parameters | Test accuracy (10 runs) |
|---|---|---|---|
| Turbo_Net-1D | raw 4-channel signal, 10 000 × 4 | 651 892 | 98.54 ± 0.28 % |
| Turbo_Net-2D | 4-channel STFT image, 128 × 128 × 4 | 803 316 | 99.14 ± 0.22 % |

## Content

The whole pipeline is in a single file, `TurboNet.m`, organised in seven steps:

| Step | Description |
|---|---|
| 1 | Read the 84 PicoScope CSV records, decimate by 2, quality control |
| 2 | Non-overlapping 0.1 s windows (16 800 samples), 123 handcrafted features, STFT images |
| 3 | 1D group: Turbo_Net-1D, SVM, random forest (10 stratified random 70/20/10 splits) |
| 4 | 2D group: Turbo_Net-2D, GoogLeNet, AlexNet, SqueezeNet (same splits) |
| 5 | Robustness to additive white Gaussian noise |
| 6 | Ablation study |
| 7 | Excel tables and PNG figures |

## Requirements

- MATLAB R2024a or later (the results in the paper were obtained with R2025a)
- Deep Learning Toolbox, Signal Processing Toolbox, Statistics and Machine Learning Toolbox
- Support packages: Deep Learning Toolbox Model for GoogLeNet Network, for AlexNet Network and for SqueezeNet Network
- An NVIDIA GPU is recommended (an RTX A2000 with 6 GB was used), with about 32 GB of RAM and 15 GB of free disk space

## Data

The data were recorded on a FAWDE SD490ZL turbocharged diesel engine fitted with a Garrett TB28 turbocharger. The campaign has 7 sessions × 4 oil-supply levels × 3 engine speeds (1500, 2000 and 2500 rpm), with 50 s per record at 200 kS/s. The data are available from the corresponding author upon reasonable request.

Place the data in a folder named `DATA CSV`, either next to `TurboNet.m` or one level above it:

```
DATA CSV/
  S1/
    P0/  1500.csv  2000.csv  2500.csv
    P1/  ...
    P2/  ...
    P3/  ...
  S2/ ... S7/
```

Each CSV file is a PicoScope export with the columns `Time, Channel A, Channel B, Channel C, Channel D`. To use another location, change `cfg.csvRoot` in the function `tn_config`.

## Usage

```matlab
TurboNet            % full pipeline, about 11-12 h on an RTX A2000
TurboNet(5:7)       % selected steps only
TurboNet('params')  % number of parameters of every network variant
```

Each step skips the work whose result file already exists, so an interrupted run can simply be started again. Intermediate data are written to `Work/` and the results to `Result/`:

- `Tables_TurboNet.xlsx`
- `Figures/*.png`
- one `.mat` file per run

To test the code quickly on a small subset (about 10 min; the numbers are not meant for reporting):

```matlab
setenv('TURBONET_QUICK','1');  TurboNet;  setenv('TURBONET_QUICK','0');
```

The quick mode writes to `Work_quick/` and `Result_quick/`.

## Network variants

| Name in code | Description |
|---|---|
| `lean` | Turbo_Net-1D: multi-scale stem (kernels 15, 63, 255) + 4 residual blocks, inputs A, B, C, D |
| `lean_noMS` | Turbo_Net-2D: single 5 × 5 stem + 4 residual blocks, inputs A, B, C, D |
| `lean_noD`, `lean_noMS_noD` | ablation: without the reference sensor D |
| `full` | ablation: multi-scale stem + squeeze-and-excitation attention + reference-gated fusion |

## Citation

If you use this code, please cite the paper. The full reference will be added after publication.
