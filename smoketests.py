from ptbxlae.modeling.convolutionalVAE import ConvolutionalEcgVAE
from ptbxlae.dataprocessing.dataModules import DefaultDM
from ptbxlae.dataprocessing.cachedDS import SingleCycleCachedDS
from ptbxlae.dataprocessing.ptbxlDS import PtbxlCleanDS
from ptbxlae.dataprocessing.mimicDS import MimicDS
from ptbxlae.dataprocessing.nkSyntheticDS import NkSyntheticDS
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import NeptuneLogger
import tempfile
from neptune import management
import sys


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
        m = ConvolutionalEcgVAE(
            latent_dim=2,
            seq_len=1000,
            lr=0.001,
            kernel_size=5,
            conv_depth=1,
            fc_depth=1,
            batchnorm=False,
            dropout=False,
        )

        ds = PtbxlCleanDS()
        ds.patient_ids = ds.patient_ids[0:100]

        dm = DefaultDM(
            ds,
            batch_size=8,
        )

        self.trainer.fit(m, datamodule=dm)
        self.trainer.test(m, datamodule=dm)

    def smoketest_mimic10s(self):
        m = ConvolutionalEcgVAE(
            latent_dim=2,
            seq_len=1000,
            lr=0.001,
            kernel_size=5,
            conv_depth=1,
            fc_depth=1,
            batchnorm=False,
            dropout=False,
        )

        ds = MimicDS()
        ds.record_list = ds.record_list.head(100)

        dm = DefaultDM(
            ds,
            batch_size=8,
        )

        self.trainer.fit(m, datamodule=dm)
        self.trainer.test(m, datamodule=dm)

    def smoketest_synthetic(self):
        m = ConvolutionalEcgVAE(
            latent_dim=2,
            seq_len=1000,
            lr=0.001,
            kernel_size=5,
            conv_depth=1,
            fc_depth=1,
            batchnorm=False,
            dropout=False,
        )

        ds = NkSyntheticDS(examples_per_epoch=50)

        dm = DefaultDM(
            ds,
            batch_size=8,
        )

        self.trainer.fit(m, datamodule=dm)
        self.trainer.test(m, datamodule=dm)

    def smoketest_syntheticfinetuning(self):
        pass

    def runall(self):

        test_methods = [m for m in dir(self) if m.startswith("smoketest_")]

        results = list()

        for m in test_methods:
            print(f"[*] SMOKE TEST: {m}")
            self.setup_smoketest()
            try:
                getattr(self, m)()
                results.append(f"{m}: SUCCEEDED")

            except Exception as e:
                print(f"[-] SMOKE TEST FAILED: {m}")
                print(e)

                results.append(f"{m}: FAILED")

            self.teardown_smoketest()

        print("\n\nSMOKE TEST RESULTS:\n\n")
        for msg in results:
            print(msg)

    def runone(self, name: str):
        self.setup_smoketest()

        try:
            getattr(self, name)()
            print(f"{name}: SUCCEEDED")

        except Exception as e:
            print(f"[-] SMOKE TEST FAILED: {name}")
            print(e)

        self.teardown_smoketest()


if __name__ == "__main__":
    st = SmokeTester()

    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        st.runone(sys.argv[1])
    else:

        st.runall()
