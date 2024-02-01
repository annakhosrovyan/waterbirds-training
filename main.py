import torch
import hydra
import logging
import warnings
import pytorch_lightning as pl

from dotenv import load_dotenv

load_dotenv('.env')
torch.set_float32_matmul_precision('medium')

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg):
    from src import utils, train
    pl.seed_everything(cfg.seed, workers=True)

    if cfg.get("print_config"):
        utils.print_configs(cfg, fields=tuple(cfg.keys()), resolve=True)

    if cfg.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    if cfg.name == "train":
        return train(cfg)


if __name__ == "__main__":
    main()
