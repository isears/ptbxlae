# PTB-XL Autoencoder

## Requirements
- Python 3.11.11
- pytorch-lightning
- neurokit
- wfdb
- jsonargparse[signatures]>=4.27.7

## Downloading Data

Ensure PTB-XL data is downloaded to ./data (see https://physionet.org/content/ptb-xl/1.0.3/)

```bash
mkdir data
cd data
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
```

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
python main.py --config configs/scc_defaults.yaml --trainer.logger=False
```


## Running Models

```bash
# Setup cache directory (first-time only)
mkdir -p cache/savedmodels

# Run training with single-cycle configuration
python main.py fit --config configs/scc_defaults.yaml

# Run synthetic data training
python main.py fit --config configs/synthetic_multichannel.yaml
```

- Best model by validation loss will be saved to cache/savedmodels