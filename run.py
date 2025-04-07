from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.trainer import Trainer
from EventEgoPoseEstimation.modules import EventEgoPoseEstimation

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


if __name__ == "__main__":
    LightningCLI(save_config_kwargs={"overwrite": True})
