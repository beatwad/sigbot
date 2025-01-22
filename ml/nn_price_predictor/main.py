import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.tuner import Tuner
from lit_models.lit_tabular_model import LitTabularModel
from models.fc_net import FCNet

from data.tabular_data_module import TabularDataModule

max_epochs = 30
patience = 4


class MyCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # instantiate DataModule and pass its parameters to LightningModule
        parser.link_arguments("data.data_config", "model.data_config", apply_on="instantiate")
        # set early stopping callback default settings
        parser.add_lightning_class_args(EarlyStopping, "early_stopping")
        parser.set_defaults(
            {"early_stopping.monitor": "val_loss", "early_stopping.patience": patience}
        )
        # trainer default settings
        parser.set_defaults({"trainer.precision": "32", "trainer.max_epochs": max_epochs})
        # # LR finder
        # parser.set_defaults({"trainer.tuner.lr_find": "32"})


if __name__ == "__main__":
    # CLI interface module
    cli = MyCLI(LitTabularModel, TabularDataModule, seed_everything_default=123)

    # find the best LR
    # data = TabularDataModule()
    # model = LitTabularModel()
    # trainer = Trainer()
    # tuner = Tuner(trainer)
    # tuner.lr_find(model, datamodule=data)
