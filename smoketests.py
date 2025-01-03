from ptbxlae.modeling.convolutionalVAE import ConvolutionalEcgVAE
from ptbxlae.dataprocessing.dataModules import DefaultDM
from ptbxlae.dataprocessing.cachedDS import SingleCycleCachedDS
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import NeptuneLogger
import tempfile
from neptune import management


class SmokeTester:
    def __init__(self):
        self.trainer = pl.Trainer()
        self.tempdir = tempfile.TemporaryDirectory()
        self.logger = NeptuneLogger()

    def setup_smoketest(self):
        self.logger = NeptuneLogger(
            project="isears/ptbxlae",
            log_model_checkpoints=False,
            tags=["smoketests"],
        )

        self.trainer = pl.Trainer(
            logger=self.logger,
            max_epochs=2,
            gradient_clip_algorithm="norm",
            gradient_clip_val=4.0,
            callbacks=[
                ModelCheckpoint(
                    dirpath=self.tempdir.name,
                    save_top_k=1,
                    monitor="val_loss",
                    mode="min",
                    filename="smoketest",
                ),
                EarlyStopping(monitor="val_loss", patience=5, mode="min"),
            ],
        )

    def teardown_smoketest(self):
        management.trash_objects(
            project="isears/ptbxlae", ids=self.logger.run["sys/id"].fetch()
        )
        self.tempdir.cleanup()

    def smoketest_ptbxlae_singlecycle(self):
        m = ConvolutionalEcgVAE(
            latent_dim=2,
            seq_len=500,
            lr=0.001,
            kernel_size=5,
            conv_depth=1,
            fc_depth=1,
            batchnorm=False,
            dropout=False,
        )

        ds = SingleCycleCachedDS()
        ds.patient_ids = ds.patient_ids[0:100]

        dm = DefaultDM(
            ds,
            batch_size=8,
        )

        self.trainer.fit(m, datamodule=dm)
        self.trainer.test(m, datamodule=dm)

    def smoketest_ptbxlae10s(self):
        pass

    def smoketest_mimic10s(self):
        pass

    def smoketest_synthetic(self):
        pass

    def smoketest_syntheticfinetuning(self):
        pass

    def runall(self):

        test_methods = [m for m in dir(self) if m.startswith("smoketest_")]

        for m in test_methods:
            print(f"[*] SMOKE TEST: {m}")
            self.setup_smoketest()
            getattr(self, m)()
            self.teardown_smoketest()


if __name__ == "__main__":
    st = SmokeTester()

    st.runall()
