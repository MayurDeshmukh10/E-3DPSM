from pytorch_lightning.cli import LightningCLI


if __name__ == "__main__":
    LightningCLI(save_config_kwargs={"overwrite": True})
