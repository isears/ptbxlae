# PTB-XL Autoencoder

## Requirements
- Python 3.11.11
- lightning: deep learning framework
- neurokit2: for basic EKG data processing
- wfdb: for reading EKG data files
- jsonargparse[signatures]>=4.27.7: for pytorch lightning CLI configs
- neptune: for logging to neptune.ai
- tslearn: for SoftDTWLoss
- ecg_plot: for clinically recognizable ECGs
- torchinfo: for detailed model summaries

## Downloading Data

Download PTB-XL data to `./data/ptbxl` (see https://physionet.org/content/ptb-xl/1.0.3/)
Download MIMIC IV ECG to `./data/mimiciv-ecg` (see https://physionet.org/content/mimic-iv-ecg/1.0/)

## Building Cache

### Single Cardiac Cycle Cache

For use with scc-* configs

```bash
python ptbxlae/dataprocessing/buildSingleCycleCache.py
```

### Synthetic Data Cache

May be useful for faster training of synthetic data models. Use of synthetic data cache not yet implemented.

```bash
python ptbxlae/dataprocessing/buildSyntheticCache.py
```

## Setup Neptune Logger (Optional)

- Register free account at https://neptune.ai
- Create project / join existing and download API token
- Add token to .bashrc e.g. `export NEPTUNE_API_TOKEN="..."`
- Reload shell and check token is available: `env | grep "NEPTUNE_API_TOKEN"`
- To use neptune-connected jupyter notebooks in VSCode, add API token to `notebooks/.env`

If running existing configs without Neptune, override logger on CLI, e.g.:

```bash
# CLI logging only
python main.py fit --config configs/single_cycle.yaml --trainer.logger False
```


## Training Models

```bash
# Setup cache directory (first-time only)
mkdir -p cache/savedmodels

# Run with single-cycle configuration
python main.py fit --config configs/single_cycle.yaml

# Run synthetic data training
python main.py fit --config configs/synthetic_base.yaml
```

## Running Test Loop

```bash
python main.py test --config configs/model_10s.yaml --ckpt_path /path/to/checkpoint
```

- Best model by validation loss will be saved to cache/savedmodels

## Distributed Hyperparameter Tuning on Slurm

First create the optuna study

```bash
optuna create-study --study-name "synthetic-ekgs" --storage "sqlite:///cache/synthetic-ekgs.db"
```

Run an example tuning script
```bash
python slurm/distributed_tune.py --debug
```

Check the optuna results (should optimize to 2)
```bash
python main.py test --config path/to/original/config.yaml --ckpt_path cache/savedmodels/checkpoint_name_here.ckpt
```

## Distributed Hyperparameter Tuning on Slurm

First create the optuna study

```bash
optuna create-study --study-name "synthetic-ekgs" --storage "sqlite:///cache/synthetic-ekgs.db"
```

Run an example tuning script
```bash
python slurm/distributed_tune.py --debug
```

Check the optuna results (should optimize to 2)
```bash