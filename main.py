import torch
import hydra
import logging
import pytorch_lightning as pl

from dotenv import load_dotenv
from omegaconf import OmegaConf
from hydra.utils import instantiate
from pytorch_lightning.loggers import WandbLogger


load_dotenv('.env')
torch.set_float32_matmul_precision('medium')

log = logging.getLogger(__name__)

wandb_logger = WandbLogger(project = "waterbirds_training_pl", 
                           log_model = 'all')

      
@hydra.main(version_base = None, config_path = "conf", config_name = "config")
def main(cfg):

      log.info(OmegaConf.to_yaml(cfg))
      device = (torch.device("cuda" if torch.cuda.is_available() else "cpu") 
                if cfg.device == "flexible" else cfg.device)

      pl.seed_everything(42, workers = True)
      
      model = instantiate(cfg.model.network_config, 
                          loss_fn = cfg.loss, 
                          optimizer_config = cfg.optimizer,
                          scheduler_config = cfg.scheduler
                          ).to(device)
      
      data_module = instantiate(cfg.datamodule.data_config)

      algorithm = instantiate(cfg.algorithm,
                              data_module = data_module,
                              model = model
                              )
      algorithm.device = device
      algorithm.cfg = cfg

      algorithm.train()


if __name__ == "__main__":
      main()