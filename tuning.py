import optuna
from ptbxlae.modeling.singleCycleConv import SingleCycleConvVAE
from ptbxlae.dataprocessing.dataModules import SingleCycleCachedDM
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from optuna.integration import PyTorchLightningPruningCallback
import datetime


def objective(trial: optuna.trial.Trial) -> float:
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 8, 5012, log=True)
    kernel_size = trial.suggest_int("kernel_size", 3, 15, step=2)
    conv_depth = trial.suggest_int("conv_depth", 1, 5)
    fc_depth = trial.suggest_int("linear_depth", 1, 5)
    batchnorm = trial.suggest_categorical("batchnorm", [True, False])
    dropout_on = trial.suggest_categorical("dropout_on", [True, False])

    if dropout_on:
        dropout = trial.suggest_float("dropout", low=0.1, high=0.9)
    else:
        dropout = None

    model = SingleCycleConvVAE(
        lr=lr,
        kernel_size=kernel_size,
        conv_depth=conv_depth,
        fc_depth=fc_depth,
        batchnorm=batchnorm,
        dropout=dropout,
    )

    dm = SingleCycleCachedDM(batch_size=batch_size)

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
                filename=f"{model.__class__.__name__}_{int(datetime.datetime.now().timestamp())}",
            ),
            EarlyStopping(monitor="val_loss", patience=5),
        ],
        gradient_clip_algorithm="norm",
        gradient_clip_val=4.0,
    )

    trainer.logger.log_hyperparams(trial.params)
    trainer.fit(model, datamodule=dm)

    return trainer.callback_metrics["val_loss"].item()


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
