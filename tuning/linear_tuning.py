import optuna
from ptbxlae.modeling.feedForwardVAE import SingleChannelFFNNVAE
from ptbxlae.dataprocessing.dataModules import PtbxlSmallSigDM
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from optuna.integration import PyTorchLightningPruningCallback


def objective(trial: optuna.trial.Trial) -> float:
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 8, 5012, log=True)
    n_layers = trial.suggest_int("linear_depth", 1, 15)

    model = SingleChannelFFNNVAE(lr=lr, n_layers=n_layers, latent_dim=40, seq_len=500)

    dm = PtbxlSmallSigDM(batch_size=batch_size, seq_len=500, single_channel=True)
    es = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    trainer = pl.Trainer(
        logger=TensorBoardLogger(save_dir="cache/optuna"),
        max_epochs=100,
        callbacks=[
            # PyTorchLightningPruningCallback(trial, monitor="val_loss"),
            ModelCheckpoint(
                dirpath="cache/optuna/checkpoints",
                save_top_k=1,
                monitor="val_loss",
                mode="min",
                filename=f"{model.__class__.__name__}_{trial.number}",
            ),
            es,
        ],
        gradient_clip_algorithm="norm",
        gradient_clip_val=4.0,
    )

    trainer.logger.log_hyperparams(trial.params)
    trainer.fit(model, datamodule=dm)

    return es.best_score.item()


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.NopPruner())

    study.optimize(objective, timeout=(24 * 60 * 60))

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
