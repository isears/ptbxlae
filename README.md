# PTB-XL Autoencoder

## Requirements
- Python 3.11.11
- pytorch-lightning
- neurokit
- wfdb
- jsonargparse[signatures]>=4.27.7

## Running

```bash
# Setup cache directory (first-time only)
mkdir -p cache/savedmodels

# Run main training script
python main.py --config configs/scc_defaults.yaml
```

- Best model by validation loss will be saved to cache/savedmodels
- Training logs in lightning_logs/