import optuna
from ptbxlae.modeling.convolutionalVAE import ConvolutionalEcgVAE
from ptbxlae.dataprocessing.dataModules import SyntheticCachedDM
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import NeptuneLogger

from optuna.integration import PyTorchLightningPruningCallback
import argparse


def objective(trial: optuna.trial.Trial) -> float:
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 8, 1024, log=True)
    kernel_size = trial.suggest_int("kernel_size", 3, 15, step=2)
    conv_depth = trial.suggest_int("conv_depth", 1, 10)
    fc_depth = trial.suggest_int("linear_depth", 1, 10)
    batchnorm = trial.suggest_categorical("batchnorm", [True, False])
    dropout_on = trial.suggest_categorical("dropout_on", [True, False])

    if dropout_on:
        dropout = trial.suggest_float("dropout", low=0.1, high=0.9)
    else:
        dropout = None

    model = ConvolutionalEcgVAE(
        lr=lr,
        kernel_size=kernel_size,
        conv_depth=conv_depth,
        fc_depth=fc_depth,
        batchnorm=batchnorm,
        dropout=dropout,
    )

    dm = SyntheticCachedDM(
        batch_size=batch_size, train_split=0.9, valid_split=0.1, test_split=0.0
    )
    es = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    trainer = pl.Trainer(
        logger=NeptuneLogger(
            project="isears/ptbxlae",
            name=f"optuna-{trial.number}",
            log_model_checkpoints=False,
            tags=["tuning"],
        ),
        max_epochs=1000,
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lightweight optuna example to troubleshoot distributed training"
    )

    parser.add_argument(
        "--timelimit",
        type=float,
        default=10.0,
        help="Time limit for slurm jobs in minutes",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # study = optuna.create_study(direction="minimize", pruner=optuna.pruners.NopPruner())
    study = optuna.create_study(
        study_name="synthetic-cvae",
        storage="sqlite:///cache/synthetic-cvae.db",
        direction="minimize",
        load_if_exists=True,
    )

    study.optimize(objective, timeout=(args.timelimit * 60))

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
